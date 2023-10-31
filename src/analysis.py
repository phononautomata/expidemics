import os
import pickle as pk
import pandas as pd
import numpy as np
from scipy.spatial import distance

import utils as ut

cwd_path = os.getcwd()

def build_spatial_data_frame(fullname):
    # Load agent results data
    input_data = open(fullname, 'rb')
    space_dict = pk.load(input_data)

    # Create an empty list to store the location data
    location_data = []

    # Iterate over each location in the space_dict
    for index in range(len(space_dict['inner'])):
        # Append the location information to the location_data list
        location_info = space_dict['inner'][index]
        location_data.append(location_info)

    # Create a DataFrame from the location_data list
    df = pd.DataFrame(location_data)
    print("Spatial data frame built")
    
    return df

def build_gravity_law_od_matrix(space_df):
    # Extract the necessary columns from the space_df DataFrame
    locations = space_df[['id', 'attractiveness', 'x', 'y']]
    
    # Calculate the number of locations
    nlocs = len(locations)
    
    # Initialize an empty OD matrix
    od_matrix = np.zeros((nlocs, nlocs))

    # Calculate the Euclidean distances between locations
    distances = distance.cdist(locations[['x', 'y']], locations[['x', 'y']], metric='euclidean')
    
     # Calculate the gravity law probabilities
    for l in range(nlocs):
        for k in range(nlocs):
             if l != k:
                attractiveness_k = locations.loc[k, 'attractiveness']
                distance_lk = distance.euclidean(locations.loc[l, ['x', 'y']], locations.loc[k, ['x', 'y']])
                od_matrix[l, k] = attractiveness_k / distance_lk
    print("OD flow matrix built")

    return od_matrix

def build_gravity_law_od_rates(space_df):
    # Extract the necessary columns from the space_df DataFrame
    locations = space_df[['id', 'attractiveness', 'x', 'y']]
    
    # Calculate the number of locations
    nlocs = len(locations)
    
    # Initialize an empty OD matrix
    od_matrix = np.zeros((nlocs, nlocs))

    # Calculate the Euclidean distances between locations
    distances = distance.cdist(locations[['x', 'y']], locations[['x', 'y']], metric='euclidean')
    
     # Calculate the gravity law probabilities
    for l in range(nlocs):
        for k in range(nlocs):
             if l != k:
                attractiveness_k = locations.loc[k, 'attractiveness']
                x_l, y_l = locations.loc[l, 'x'], locations.loc[l, 'y']
                x_k, y_k = locations.loc[k, 'x'], locations.loc[k, 'y']
                distance_lk = np.sqrt((x_k - x_l)**2 + (y_k - y_l)**2)
                od_matrix[l, k] = attractiveness_k / distance_lk
    
    # Normalize the rates for each origin location l
    od_rates = od_matrix / np.sum(od_matrix, axis=1, keepdims=True)
    print("OD rates built")
    
    return od_rates

def build_trajectory_data_frame(fullname, nagents_load=1):
    # Load mobility trajectories data
    input_data = open(fullname, 'rb')
    mob_dict = pk.load(input_data)
    
    # Select a subset of agents based on nagents_load
    mob_dict_subset = mob_dict[:nagents_load]
    
    # Create an empty list to store the trajectory data
    trajectory_data = []

    # Iterate over each agent in mob_dict_subset
    for agent_info in mob_dict_subset:
        rho = agent_info['rho']
        trajectory = agent_info['trajectory']
        trajectory_data.append({'rho': rho, 'trajectory': trajectory})

    # Create a DataFrame from the trajectory_data list
    df = pd.DataFrame(trajectory_data)
    
    return df

def build_lonlat_gravity_law_od_rates(space_df, value_id='count'):
    # Extract the necessary columns from the space_df DataFrame
    locations = space_df[[value_id, 'lon_medoid', 'lat_medoid']]
    
    # Convert longitude and latitude to Cartesian coordinates (Mercator projection)
    lon_rad = np.radians(locations['lon_medoid'])
    lat_rad = np.radians(locations['lat_medoid'])
    R = 6371.0  # Earth's radius in kilometers
    x = R * lon_rad
    y = R * np.log(np.tan(lat_rad / 2.0 + np.pi / 4.0))
    
    # Calculate the number of locations
    nlocs = len(locations)
    
    # Initialize an empty OD matrix
    od_matrix = np.zeros((nlocs, nlocs))
 
    # Calculate the gravity law probabilities
    for l in range(nlocs):
        for k in range(nlocs):
            if (l % 100 == 0) and (k % 100) == 0:
                print("Location {0} with {1}".format(l, k))
            if l != k:
                attractiveness_k = locations.loc[k, value_id]
                x_l, y_l = locations.loc[l, 'lon_medoid'], locations.loc[l, 'lat_medoid']
                x_k, y_k = locations.loc[k, 'lon_medoid'], locations.loc[k, 'lat_medoid']
                distance_lk = np.sqrt((x_k - x_l)**2 + (y_k - y_l)**2)
                od_matrix[l, k] = attractiveness_k / distance_lk
    
    # Normalize the rates for each origin location l
    od_rates = od_matrix / np.sum(od_matrix, axis=1, keepdims=True)
    print("OD rates built")
    
    return od_rates

def get_agent_trajectory(df, agent_id):
    agent_row = df[df.index == agent_id]
    trajectory = agent_row['trajectory'].values[0]
    return np.array(trajectory)

def get_agent_rho(df, agent_id):
    agent_row = df[df.index == agent_id]
    rho = agent_row['rho'].values[0]
    return rho

def compute_most_visited_location(trajectory):
    unique_locations, location_counts = np.unique(trajectory, return_counts=True)
    max_visit_index = np.argmax(location_counts)
    most_visited_location = unique_locations[max_visit_index]
    visit_count = location_counts[max_visit_index]
    return most_visited_location, visit_count

def compute_location_visits(trajectory, location_label):
    visit_count = sum(1 for loc in trajectory if loc == location_label)
    return visit_count

def compute_location_rank(trajectory):
    unique_locations, location_counts = np.unique(trajectory, return_counts=True)
    sorted_indices = np.argsort(location_counts)[::-1]
    sorted_locations = unique_locations[sorted_indices]
    sorted_counts = location_counts[sorted_indices]
    location_rank = list(zip(sorted_locations, sorted_counts))
    return location_rank

def compute_cumulative_visits(trajectory):
    unique_locations, location_counts = np.unique(trajectory, return_counts=True)
    sorted_indices = np.argsort(location_counts)[::-1]
    sorted_locations = unique_locations[sorted_indices]
    sorted_counts = location_counts[sorted_indices]
    cumulative_visits = np.cumsum(sorted_counts)
    location_cumulative_visits = list(zip(sorted_locations, cumulative_visits))
    return location_cumulative_visits

def compute_num_unique_visits(trajectory):
    num_unique_visits = len(set(trajectory))
    return num_unique_visits

def compute_cumulative_unique_visits(trajectory):
    cumulative_visits = [len(set(trajectory[:i])) for i in range(1, len(trajectory) + 1)]
    return np.array(cumulative_visits)

def compute_exploration_steps(trajectory):
    visited_locations = set()
    exploration_steps = 0

    for location in trajectory:
        if location not in visited_locations:
            visited_locations.add(location)
            exploration_steps += 1

    return exploration_steps

def compute_preferential_return_steps(trajectory):
    visited_locations = set()
    preferential_return_steps = 0

    for location in trajectory:
        if location in visited_locations:
            preferential_return_steps += 1

        visited_locations.add(location)

    return preferential_return_steps

def compute_histogram(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

def compute_average(data):
    average = np.mean(data)
    return average

def compute_median(data):
    median = np.median(data)
    return median

def compute_std_deviation(data):
    std_deviation = np.std(data)
    return std_deviation

def compute_quantiles(data, quantiles=[0.25, 0.5, 0.75]):
    quantiles_values = np.quantile(data, quantiles)
    return quantiles_values

def load_grid(fullname):
    # Load agent results data
    with open(fullname, 'rb') as input_data:
        grid_object = pk.load(input_data)

    grid_lt = grid_object['inner']
    
    return grid_lt

def build_grid_from_trajectories(fullname, nagents_load=1):
    
    trajectory_df = build_trajectory_data_frame(fullname, nagents_load=nagents_load)

    # Determine t_max and nlocs
    t_max = len(trajectory_df.iloc[0]['trajectory'])
    nlocs = trajectory_df['trajectory'].apply(lambda x: max(x)).max() + 1

    # Initialize the space-time grid
    agent_grid = {}
    for t in range(t_max):
        agent_grid[t] = {}
        for l in range(nlocs):
            agent_grid[t][l] = []

    # Populate the space-time grid with agent IDs
    for agent_id, agent_info in trajectory_df.iterrows():
        trajectory = agent_info['trajectory']
        for t, location_id in enumerate(trajectory):
            agent_grid[t][location_id].append(agent_id)

    return agent_grid

def build_trajectories_from_grid(agent_grid):
    nlocs = len(agent_grid)
    t_max = len(agent_grid[0])
    nagents = max(max(agent_ids) for location in agent_grid for agent_ids in location if agent_ids) + 1

    trajectory_data = {'agent_id': [], 'trajectory': []}

    for agent_id in range(nagents):
        trajectory = []
        for t in range(t_max):
            location_id = -1
            for location in range(nlocs):
                agent_ids = agent_grid[location][t]
                if agent_ids and agent_id in agent_ids:
                    location_id = location
                    break
            trajectory.append(location_id)

        trajectory_data['agent_id'].append(agent_id)
        trajectory_data['trajectory'].append(trajectory)

    trajectory_df = pd.DataFrame(trajectory_data)
    return trajectory_df

def save_trajectory_dataframe(trajectory_df, fullname):
    pass

def load_trajectory_dataframe(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        trajectory_object = pk.load(input_data)
    trajectory_df = pd.DataFrame(trajectory_object)
    trajectory_df['mob_id'] = trajectory_df.index
    return trajectory_df

def build_grid_data_frame(full_name):
    # Load grid results data
    input_data = open(full_name, 'rb')
    grid_dict = pk.load(input_data)
    pass

def count_individuals(agent_grid):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    individuals_count = np.zeros((t_max, nlocs))

    for t in range(t_max):
        for l in range(nlocs):
            individuals_count[t, l] = len(agent_grid[t][l])

    return individuals_count

def compute_average_rho(agent_grid, rho_values):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    average_rho = np.zeros((t_max, nlocs))

    for t in range(t_max):
        for l in range(nlocs):
            agent_ids = agent_grid[t][l]
            if agent_ids:
                agent_rho_values = rho_values[agent_ids]
                average_rho[t, l] = np.mean(agent_rho_values)

    return average_rho

def compute_incidence_time_series(agent_grid, health_statuses, infection_times, removal_times):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    incidence_time_series = np.zeros((t_max, nlocs))

    for t in range(t_max):
        for l in range(nlocs):
            agent_ids = agent_grid[t][l]
            if agent_ids:
                for agent_id in agent_ids:
                    if (
                        health_statuses[agent_id] == "Infected"
                        and infection_times[agent_id] <= t < removal_times[agent_id]
                    ):
                        incidence_time_series[t, l] += 1

    return incidence_time_series

def compute_prevalence_time_series(agent_grid, health_statuses, infection_times):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    prevalence_time_series = np.zeros((t_max, nlocs))

    for t in range(t_max):
        for l in range(nlocs):
            agent_ids = agent_grid[t][l]
            if agent_ids:
                for agent_id in agent_ids:
                    if (
                        health_statuses[agent_id] == "Infected"
                        and infection_times[agent_id] <= t
                    ):
                        prevalence_time_series[t, l] += 1

    return prevalence_time_series

def compute_incidence_time_series_with_rho(agent_grid, health_statuses, infection_times, removal_times, rho_values, nbins):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    incidence_time_series = np.zeros((t_max, nlocs, nbins))

    rho_min = np.min(rho_values)
    rho_max = np.max(rho_values)
    bin_boundaries = np.linspace(rho_min, rho_max, nbins + 1)

    for t in range(t_max):
        for l in range(nlocs):
            agent_ids = agent_grid[t][l]
            if agent_ids:
                for agent_id in agent_ids:
                    if (
                        health_statuses[agent_id] == "Infected"
                        and infection_times[agent_id] <= t < removal_times[agent_id]
                    ):
                        rho_bin = np.searchsorted(bin_boundaries, rho_values[agent_id]) - 1
                        incidence_time_series[t, l, rho_bin] += 1

    return incidence_time_series

def compute_prevalence_time_series_with_rho(agent_grid, health_statuses, infection_times, rho_values, nbins):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    prevalence_time_series = np.zeros((t_max, nlocs, nbins))

    rho_min = np.min(rho_values)
    rho_max = np.max(rho_values)
    bin_boundaries = np.linspace(rho_min, rho_max, nbins + 1)

    for t in range(t_max):
        for l in range(nlocs):
            agent_ids = agent_grid[t][l]
            if agent_ids:
                for agent_id in agent_ids:
                    if (
                        health_statuses[agent_id] == "Infected"
                        and infection_times[agent_id] <= t
                    ):
                        rho_bin = np.searchsorted(bin_boundaries, rho_values[agent_id]) - 1
                        prevalence_time_series[t, l, rho_bin] += 1

    return prevalence_time_series

def count_absorbing_health_status(agent_grid, health_statuses):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    health_status_count = {}

    for l in range(nlocs):
        agent_ids = agent_grid[t_max-1][l]
        if agent_ids:
            agent_health_statuses = health_statuses[agent_ids]
            for status in agent_health_statuses:
                if status in health_status_count:
                    health_status_count[status] += 1
                else:
                    health_status_count[status] = 1

    return health_status_count

def count_absorbing_recovered_by_rho_category(agent_grid, health_status, rho_values, nbins):
    t_max = len(agent_grid)
    nlocs = len(agent_grid[0])
    recovered_count = np.zeros((nlocs, nbins))

    # Compute the bin boundaries
    min_rho = 0.0
    max_rho = 1.0
    delta_rho = (max_rho - min_rho) / nbins
    bin_boundaries = np.linspace(min_rho, max_rho, nbins + 1)

    for l in range(nlocs):
        agent_ids = agent_grid[t_max-1][l]
        if agent_ids:
            agent_rho_values = rho_values[agent_ids]
            agent_health_statuses = health_status[agent_ids]
            # Compute the recovered count for each rho category
            for i in range(nbins):
                bin_min = bin_boundaries[i]
                bin_max = bin_boundaries[i + 1]
                mask = np.logical_and(bin_min <= agent_rho_values, agent_rho_values < bin_max)
                recovered_mask = np.logical_and(mask, agent_health_statuses == 'Recovered')
                recovered_count[l, i] = np.sum(recovered_mask)

    return recovered_count

def build_agent_data_frame(fullname, nsims_load=1):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edyna_dict = pk.load(input_data)

    # Determine the actual number of simulations available
    nsims_load = min(nsims_load, len(edyna_dict))

    # Create an empty list to store the agent data
    agent_data = []

    # Iterate over the specified number of simulations to load
    for s in range(nsims_load):
        # Iterate over each agent in the current simulation
        for agent_id, agent_info in enumerate(edyna_dict[s]['agent']['inner']):
            # Append the agent information to the agent_data list
            agent_info['sim'] = s  # Include the simulation index as a column
            agent_info['id'] = agent_id  # Include the agent ID as a column
            agent_data.append(agent_info)

    # Create a DataFrame from the agent_data list
    agent_df = pd.DataFrame(agent_data)

    return agent_df

def number_of_agents_in_simulation(agent_df, sim_index):
    return len(agent_df[agent_df['sim'] == sim_index])

def number_of_simulations(agent_df):
    simulation_count = len(set(agent_df['sim']))
    return simulation_count

def number_of_agents_in_rho_interval(agent_df, rho_interval):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}

    nagents_s = np.zeros(nsims)

    for sim_index in unique_sims:
        sim_data = agent_df[agent_df['sim'] == sim_index]
        s = sim_mapping[sim_index]

        for _, agent in sim_data.iterrows():
            rho = agent['mobility']
            if rho_interval[0] <= rho <= rho_interval[1]:
                nagents_s[s] += 1
    
    return nagents_s

def compute_prevalence(df, sim_index):
    infected_agent_ids = get_infected_agent_ids(df[df['sim'] == sim_index])
    num_agents = number_of_agents_in_simulation(df, sim_index)
    prevalence = len(infected_agent_ids) / num_agents
    return prevalence

def outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01):
    # Create a copy of the agent_df DataFrame for filtering
    filtered_df = agent_df.copy()

    # Iterate over unique simulation indices
    for sim_index in agent_df['sim'].unique():
        # Compute the prevalence for the current simulation
        prevalence = compute_prevalence(filtered_df, sim_index)
        # Check if the prevalence is below the threshold
        if prevalence < prevalence_threshold:
            # Remove rows with the current simulation index
            filtered_df = filtered_df[filtered_df['sim'] != sim_index]

    return filtered_df

def health_status_filter_agent_data_frame(agent_df, health_status='Removed'):
    filtered_df = agent_df[agent_df['status'] == health_status]
    return filtered_df

def simulations_filter_agent_data_frame(agent_df, nsims_load=1):
    unique_sims = agent_df['sim'].unique()
    filtered_df = agent_df[agent_df['sim'].isin(unique_sims[:nsims_load])]
    return filtered_df

def get_rho_values(agent_df):
    return agent_df['mobility'].values

def get_infected_agent_ids(agent_df):
    infected_agents = agent_df[agent_df['status'] == 'Removed']
    return infected_agents['id'].values

def get_vaccinated_agent_ids(agent_df):
    infected_agents = agent_df[agent_df['status'] == 'Vaccinated']
    return infected_agents['id'].values

def get_infectors(agent_df):
    infected_agents = agent_df[agent_df['status'] == 'Removed']
    return infected_agents['infected_by'].values

def get_infection_locations(agent_df):
    infected_agents = agent_df[agent_df['status'] == 'Removed']
    return infected_agents['infected_where'].values

def get_infection_times(agent_df):
    infected_agents = agent_df[agent_df['status'] == 'Removed']
    return infected_agents['infected_when'].values

def get_removal_times(agent_df):
    removed_agents = agent_df[agent_df['status'] == 'Removed']
    return removed_agents['removed_when'].values

def rebuild_incidence_time_series_for_rho_group(agent_df, rho_interval, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}

    incidence_st = np.zeros((nsims, t_max))

    for sim_index in unique_sims:
        sim_data = agent_df[agent_df['sim'] == sim_index]
        s = sim_mapping[sim_index]

        for _, agent in sim_data.iterrows():
            rho = agent['mobility']
            if rho_interval[0] <= rho <= rho_interval[1]:
                infected_when = int(agent['infected_when'])
                removed_when = int(agent['removed_when'])
                incidence_st[s, infected_when:(removed_when + 1)] += 1

    return incidence_st

def rebuild_prevalence_time_series_for_rho_group(agent_df, rho_interval, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}

    prevalence_st = np.zeros((nsims, t_max))

    for sim_index in unique_sims:
        sim_data = agent_df[agent_df['sim'] == sim_index]
        s = sim_mapping[sim_index]

        for _, agent in sim_data.iterrows():
            rho = agent['mobility']
            if rho_interval[0] <= rho <= rho_interval[1]:
                infected_when = int(agent['infected_when'])
                prevalence_st[s, infected_when:] += 1

    return prevalence_st

def rebuild_local_incidence_time_series_for_rho_group(agent_df, rho_interval, trajectory_df, nlocs, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}

    incidence_slt = np.zeros((nsims, nlocs, t_max))

    for sim_index in unique_sims:
        sim_data = agent_df[agent_df['sim'] == sim_index]
        s = sim_mapping[sim_index]

        for _, agent in sim_data.iterrows():
            rho = agent['mobility']
            if rho_interval[0] <= rho <= rho_interval[1]:
                mob_id = int(agent['mob_id'])
                infected_when = int(agent['infected_when'])
                removed_when = int(agent['removed_when'])
                # Get agent's trajectory
                trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
                traj_inf = trajectory[infected_when + 1:removed_when + 1]
                for t in range(len(traj_inf) - 1):
                    loc = traj_inf[t]
                    incidence_slt[s, loc, infected_when + t + 1] += 1

    return incidence_slt

def rebuild_local_prevalence_time_series_for_rho_group(agent_df, rho_interval, trajectory_df, nlocs, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}

    prevalence_slt = np.zeros((nsims, nlocs, t_max))

    for sim_index in unique_sims:
        sim_data = agent_df[agent_df['sim'] == sim_index]
        s = sim_mapping[sim_index]

        for _, agent in sim_data.iterrows():
            rho = agent['mobility']
            if rho_interval[0] <= rho <= rho_interval[1]:
                mob_id = int(agent['mob_id'])
                infected_when = int(agent['infected_when'])
                # Get agent's trajectory
                trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
                traj_inf = trajectory[infected_when + 1:]
                for t in range(len(traj_inf) - 1):
                    loc = traj_inf[t]
                    prevalence_slt[s, loc, infected_when + t + 1] += 1

    return prevalence_slt

def count_visits_until_infection_and_where(agent_df, trajectory_df):
    visits_counts = []
    infected_locations = []
    homes = []
    rho_values = []
    
    for _, agent in agent_df.iterrows():
        s = agent['sim']
        mob_id = agent['mob_id']
        infected_where = int(agent['infected_where'])
        infected_when = int(agent['infected_when'])
        rho = agent['mobility']

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t0 = infected_when + 1
        traj_bef_inf = trajectory[:t0]
        home = trajectory[0]

        visit_count = compute_location_visits(traj_bef_inf, infected_where)

        visits_counts.append(visit_count)
        infected_locations.append(infected_where)
        homes.append(home)
        rho_values.append(rho)

    return visits_counts, infected_locations, homes, rho_values

def build_event_data_frame(fullname, nsims_load=1):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edyna_dict = pk.load(input_data)

    # Create an empty list to store the event data
    event_data = []

    # Iterate over the specified number of simulations to load
    for s in range(nsims_load):
        # Iterate over each event in the current simulation
        for event_id, event_info in enumerate(edyna_dict[s]['event']['inner']):
            # Append the event information to the event_data list
            event_info['sim'] = s  # Include the simulation index as a column
            event_data.append(event_info)
    # Create a DataFrame from the event_data list
    event_df = pd.DataFrame(event_data)
    
    return event_df

def outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01):
    # Create a copy of the agent_df DataFrame for filtering
    filtered_df = event_df.copy()

    # Iterate over unique simulation indices
    for sim_index in agent_df['sim'].unique():
        # Compute the prevalence for the current simulation
        prevalence = compute_prevalence(agent_df, sim_index)
        # Check if the prevalence is below the threshold
        if prevalence < prevalence_threshold:
            # Remove rows with the current simulation index
            filtered_df = filtered_df[filtered_df['sim'] != sim_index]

    return filtered_df

def simulations_filter_event_data_frame(event_df, nsims_load=1):
    unique_sims = event_df['sim'].unique()
    filtered_df = event_df[event_df['sim'].isin(unique_sims[:nsims_load])]
    return filtered_df

def build_event_id_array(event_df, nlocs, t_max):

    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)

    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    
    event_id_slt = np.full((nsims, nlocs, t_max), -1, dtype=int)

    for _, event in event_df.iterrows():
        event_id = int(event['id'])
        s = int(event['sim'])
        if s in sim_mapping:
            s_new = sim_mapping[s]
            t = int(event['t'])
            l = int(event['location'])
            event_id_slt[s_new, l, t] = event_id

    return event_id_slt

def get_event_ids(event_df):
    return event_df['id'].values

def get_event_locations(event_df):
    return event_df['location'].values

def get_event_times(event_df):
    return event_df['t'].values

def get_event_sizes(event_df):
    return event_df['size'].values

def get_event_infector(event_df):
    return event_df['infector'].values

def get_infector_rho(event_df):
    return event_df['infector_rho'].values

def collect_location_invaders_rho(event_df, space_df, nlocs, nlocs_eff, t_max):
    # Call build_event_id_matrix to get event_id_matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001

    # Initialize invader_rho_array
    invader_rho_sl = np.zeros((nsims, nlocs_eff))

    for s in range(nsims):
        loc_count = 0
        for l in range(nlocs):
            attr = attr_l[l]
            if attr > attr_cutoff:
                event_id_t = event_id_slt[s, l]
                nonneg_event_id_t = [ind for ind in range(len(event_id_t)) if event_id_t[ind] > 0]

                # Was there an invasion at all?
                if len(nonneg_event_id_t) == 0:
                    invader_rho_sl[s, loc_count] = np.nan
                else:
                    # Location was invaded: Obtain event index & rest of info
                    index = nonneg_event_id_t[0]
                    event_id = event_id_slt[s, l, index]

                    #print("event_id={0}, s={1}, l={2}, index={3}".format(event_id, s, l, index))
                    sim = inverse_sim_mapping[s]
                    infector_rho = event_df.loc[(event_df['id'] == event_id) & (event_df['sim'] == sim), 'infector_rho'].values[0]
                    if infector_rho <= 1.0:
                        invader_rho_sl[s, loc_count] = infector_rho
                    else:
                        invader_rho_sl[s, loc_count] = np.nan
                loc_count += 1

    return invader_rho_sl

def collect_invasion_times(event_df, space_df, nlocs, nlocs_eff, t_max):
    # Call build_event_id_matrix to get event_id_matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001

    # Initialize invader_rho_array
    invasion_time_sl = np.zeros((nsims, nlocs_eff))

    for s in range(nsims):
        loc_count = 0
        for l in range(nlocs):
            attr = attr_l[l]
            if attr > attr_cutoff:
                event_id_t = event_id_slt[s, l]
                nonneg_event_id_t = [ind for ind in range(len(event_id_t)) if event_id_t[ind] > 0]

                # Was there an invasion at all?
                if len(nonneg_event_id_t) == 0:
                    invasion_time_sl[s, loc_count] = np.nan
                else:
                    # Location was invaded: Obtain event index & rest of info
                    index = nonneg_event_id_t[0]
                    event_id = event_id_slt[s, l, index]

                    #print("event_id={0}, s={1}, l={2}, index={3}".format(event_id, s, l, index))
                    sim = inverse_sim_mapping[s]
                    invasion_time_sl[s, loc_count] = event_df.loc[(event_df['id'] == event_id) & (event_df['sim'] == sim), 't'].values[0]
                loc_count += 1
    return invasion_time_sl

def build_invaders_dataframe(event_df, nlocs, t_max):
    pass

def collect_trajectories(fullname_trajectory, fullname_chosen):
    # Load all trajectories
    trajectory_df = load_trajectory_dataframe(fullname_trajectory)
    # Load chosen agent labels & filter
    chosen_ids_rho = ut.read_pickle_file(fullname_chosen)
    chosen_mob_id = [triple[1] for triple in chosen_ids_rho]
    trajectory_df = trajectory_df[trajectory_df['mob_id'].isin(chosen_mob_id)] 
    return trajectory_df

def count_event_sizes_experienced_in_rho_t_inf_groups(agent_df, event_df, trajectory_df, nlocs, t_max):
    # Build event id matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    agent_triad = []

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        infected_when = int(agent['infected_when'])
        removed_when = int(agent['removed_when'])
        t_inf = removed_when - infected_when
        rho = agent['mobility']

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t0 = infected_when + 1
        traj_inf = trajectory[t0:removed_when+1]

        # Initialize cumulative event size for the agent
        cumulative_size = 0

        for t in range(len(traj_inf)):
            l = traj_inf[t]
            event_id = event_id_slt[s, l, t0 + t]
            if event_id != -1 and event_id != 0:
                event_size = event_df[(event_df['id'] == event_id) & (event_df['sim'] == sim)]['size'].values[0]
                cumulative_size += event_size
        agent_triad.append((rho, t_inf, cumulative_size))

    return agent_triad

def collect_offsprings_by_event(event_df):
    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    infectors_sa = []

    for s in range(nsims):
        infector_a = []
        # Filter event_df by the corresponding simulation
        sim_mask = event_df['sim'] == inverse_sim_mapping[s]
        sim_df = event_df[sim_mask]

        infector_data = {}
        infector_rho = {}

        sim_df_iterator = sim_df.iterrows()
        next(sim_df_iterator) 

        for _, event in sim_df_iterator:
            infector = int(event['infector_epi_id'])
            size = int(event['size'])
            time = int(event['t'])
            rho = event['infector_rho']

            if infector in infector_data:
                infector_data[infector][0].append(size)
                infector_data[infector][1].append(time)
            else:
                infector_data[infector] = ([size], [time])
                infector_rho[infector] = rho

        infector_list = []
        for infector in infector_data:
            sizes, times = infector_data[infector]
            infector_list.append((infector_rho[infector], (sizes, times)))

        infectors_sa.extend(infector_list)

    return infectors_sa

def collect_cases_by_location(event_df):
     # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

     # Dictionary to store the total size and count for each location across simulations
    location_data = {}
    # Dictionary to store the total infector_rho for each location across simulations
    location_rho = {}

    for s in range(nsims):
        # Filter event_df by the corresponding simulation
        sim_mask = event_df['sim'] == inverse_sim_mapping[s]
        sim_df = event_df[sim_mask]

        for _, event in sim_df.iterrows():
            location = int(event['location'])
            size = int(event['size'])
            rho = event['infector_rho']

            if location in location_data:
                location_data[location][0] += size
                location_data[location][1] += 1
                location_rho[location] += rho
            else:
                location_data[location] = [size, 1]
                location_rho[location] = rho

    # Calculate the mean infector_rho for each location
    for location in location_data:
        location_data[location][0] /= nsims
        location_rho[location] /= location_data[location][1]

    # Create a list to store the final results (location, total size, mean infector_rho)
    location_list = []
    for location, (total_size, count) in location_data.items():
        mean_rho = location_rho[location]
        location_list.append((location, total_size, mean_rho))

    return location_list


def compute_invader_average_rho_map(event_df, nlocs, t_max):
     # Call build_event_id_matrix to get event_id_matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    # Initialize invader_rho_array
    invader_rho_sl = np.zeros((nsims, nlocs))

    # Prepare output structure
    x_cells = int(np.sqrt(nlocs))
    y_cells = int(np.sqrt(nlocs))    
    RHO_AVG = np.zeros((x_cells, y_cells))

    # Loop over locations
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            for s in range(nsims):
                # Get event ids for a given simulation and location
                event_id_t = event_id_slt[s, l]
                nonneg_event_id_t = [ind for ind in range(len(event_id_t)) if event_id_t[ind] > 0]

                # Was there an invasion at all?
                if len(nonneg_event_id_t) == 0:
                    invader_rho_sl[s, l] = np.nan
                else:
                    # Location was invaded: Obtain event index & rest of info
                    index = nonneg_event_id_t[0]
                    event_id = event_id_slt[s, l, index]

                    # Obtain original simulation identifier
                    sim = inverse_sim_mapping[s]
                    # Extract invader's rho
                    infector_rho = event_df.loc[(event_df['id'] == event_id) & (event_df['sim'] == sim), 'infector_rho'].values[0]
                    if infector_rho <= 1.0:
                        invader_rho_sl[s, l] = infector_rho
                    else:
                        invader_rho_sl[s, l] = np.nan
            RHO_AVG[y_cells - 1 - j, i] = np.nanmean(invader_rho_sl[:, l], axis=0)
            l += 1

    return RHO_AVG


def compute_infected_average_rho_map(agent_df, nlocs):
    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    # Initialize invader_rho_array
    avg_infected_rho_sl = np.zeros((nsims, nlocs))

    # Prepare output structure
    x_cells = int(np.sqrt(nlocs))
    y_cells = int(np.sqrt(nlocs))    
    RHO_AVG = np.zeros((x_cells, y_cells))

    # Loop over locations
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            for s in range(nsims):
                # Obtain original simulation identifier
                sim = inverse_sim_mapping[s]
                infected_where = l

                # Filter agents in data frame by 'sim' column and 'infected_where' column values
                filtered_agents = agent_df[(agent_df['sim'] == sim) & (agent_df['infected_where'] == infected_where)]
                
                # Now, compute the average value of the 'mobility' column of the filtered df
                avg_infected_rho_sl[s, l] = filtered_agents['mobility'].mean()

            RHO_AVG[y_cells - 1 - j, i] = np.nanmean(avg_infected_rho_sl[:, l])
            l += 1

    return RHO_AVG

def collect_infection_epr_steps(agent_df, trajectory_df, nsims, nlocs):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    inf_epr_a = []
    inf_rho_a = []
    infected_where_a = []
    infected_when_a = []
    infected_where_freq_a = []

    inv_time_sl = np.full((nsims, nlocs), 99999)
    inv_rho_sl = np.zeros((nsims, nlocs))
    inv_epr_sl = np.zeros((nsims, nlocs), dtype=np.int_)
    inv_where_sl = np.zeros((nsims, nlocs), dtype=np.int_)
    inv_where_freq_sl = np.zeros((nsims, nlocs))
    
    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        infected_where = int(agent['infected_where'])
        infected_when = int(agent['infected_when'])
        rho = agent['mobility']
        inf_rho_a.append(rho)
        infected_where_a.append(infected_where)
        infected_when_a.append(infected_when)

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t_max = len(trajectory)
        t0 = infected_when + 1
        traj_before = trajectory[:t0]

        # Compute visitation frequency for the infection's location
        infected_where_counts = trajectory.count(infected_where)
        infected_where_freq = infected_where_counts / t_max
        infected_where_freq_a.append(infected_where_freq)

        visit_count = compute_location_visits(traj_before, infected_where)
        if visit_count == 1:
            inf_epr_a.append(1)
        else:
            inf_epr_a.append(0)
        
        # Check invasion
        if infected_when < inv_time_sl[s, infected_where]:
            inv_where_sl[s, infected_where] = infected_where
            inv_time_sl[s, infected_where] = infected_when
            inv_rho_sl[s, infected_where] = rho
            inv_epr_sl[s, infected_where] = 1 if visit_count == 1 else 0
            inv_where_freq_sl[s, infected_where] = infected_where_freq

    inv_when_a = inv_time_sl.flatten()
    inv_rho_a = inv_rho_sl.flatten()
    inv_epr_a = inv_epr_sl.flatten()
    inv_where_a = inv_where_sl.flatten()
    inv_where_freq_a = inv_where_freq_sl.flatten()
    inv_results = inv_epr_a, inv_rho_a, inv_where_a, inv_when_a, inv_where_freq_a 

    inf_results = inf_epr_a, inf_rho_a, infected_where_a, infected_when_a, infected_where_freq_a

    return inv_results, inf_results

def collect_stay_times(agent_df, trajectory_df, space_df, nlocs, t_max, top_k):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    rho_a = []
    rho_inf_a = []
    S_a = []
    S_inf_a = []
    t_avg_k_a = []
    t_avg_k_inf_a = []
    t_inf_a = []
    attr_a = []
    attr_inf_a = []
    
    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        status = agent['status']
        rho = agent['mobility']
        
        # Get agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t_max = len(trajectory)
        S = len(set(trajectory))
        rank_list = compute_location_rank(trajectory)
        #new_top_k = min(top_k, len(rank_list))
        if len(rank_list) >= top_k:
            rho_a.append(rho)
            S_a.append(S)
            top_k_elements = rank_list[:top_k]
            sum_counts = sum(count for _, count in top_k_elements)
            sum_f_k = sum_counts / t_max
            t_avg_k = (1.0 / top_k) * sum_f_k
            t_avg_k_a.append(t_avg_k)
            #attract_tr = [space_df.loc[space_df['id'] == l, 'attractiveness'].values[0] for l in trajectory]
            avg_attr = 0.0 #np.mean(np.array(attract_tr))
            attr_a.append(avg_attr)
    
            if status == 'Removed':
                infected_when = int(agent['infected_when'])
                removed_when = int(agent['removed_when'])
                t0 = infected_when + 1
                traj_while = trajectory[t0:removed_when+1]
                T_I = removed_when - infected_when
                if T_I >= top_k:
                    S_while = len(set(traj_while))
                    rank_list = compute_location_rank(traj_while)
                    #new_top_k = min(top_k, len(rank_list))
                    if len(rank_list) >= top_k:
                        rho_inf_a.append(rho)
                        t_inf_a.append(T_I)
                        S_inf_a.append(S_while)
                        top_k_elements = rank_list[:top_k]
                        sum_counts = sum(count for _, count in top_k_elements)
                        sum_f_k = sum_counts / T_I
                        t_avg_k = (1.0 / top_k) * sum_f_k
                        t_avg_k_inf_a.append(t_avg_k)
                        attract_tr = [space_df.loc[space_df['id'] == l, 'attractiveness'].values[0] for l in traj_while]
                        avg_a = np.mean(np.array(attract_tr))
                        attr_inf_a.append(avg_a)
        
    all_results = rho_a, S_a, t_avg_k_a, attr_a
    inf_results = rho_inf_a, S_inf_a, t_avg_k_inf_a, t_inf_a, attr_inf_a
        
    return all_results, inf_results

def count_visits_until_infection_where_and_freq(agent_df, trajectory_df):
    visits_counts = []
    infected_locations = []
    homes = []
    rho_values = []
    infected_where_freq_tbinf = []
    infected_where_freq_tmax = []
    
    for _, agent in agent_df.iterrows():
        s = agent['sim']
        mob_id = agent['mob_id']
        infected_where = int(agent['infected_where'])
        infected_when = int(agent['infected_when'])
        rho = agent['mobility']

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t_max = len(trajectory)
        t0 = infected_when + 1
        traj_bef_inf = trajectory[:t0]
        t_binf = len(traj_bef_inf)
        home = trajectory[0]

        visit_count = compute_location_visits(traj_bef_inf, infected_where)

        visits_counts.append(visit_count)
        infected_locations.append(infected_where)
        homes.append(home)
        rho_values.append(rho)

        infected_where_counts = traj_bef_inf.count(infected_where)
        infected_where_freq_tbinf.append(infected_where_counts / t_binf)

        infected_where_counts = trajectory.count(infected_where)
        infected_where_freq_tmax.append(infected_where_counts / t_max)

    return visits_counts, infected_locations, infected_where_freq_tbinf, infected_where_freq_tmax, homes, rho_values

def count_home_recurrence_and_attractiveness_for_infectors(agent_df, event_df, trajectory_df, space_df, nlocs, t_max):
    # Get attractiveness
    attr_l = space_df['attractiveness'].to_numpy()

    # Build event id matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    agent_multi = []

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        infected_when = int(agent['infected_when'])
        removed_when = int(agent['removed_when'])
        t_inf = removed_when - infected_when
        rho = agent['mobility']

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t_max = len(trajectory)
        t0 = infected_when + 1
        traj_inf = trajectory[t0:removed_when+1]
        home = trajectory[0]
        home_counts = compute_location_visits(trajectory, home)
        home_freq = home_counts / t_max
        home_attr = attr_l[home]

        # Initialize cumulative event size for the agent
        home_cum_ce = 0
        home_cum_size = 0
        out_cum_ce = 0
        out_cum_size = 0
        out_cum_attr = 0.0
        out_cum_freq = 0.0
        out_avg_freq = 0.0
        out_avg_attr = 0.0

        for t in range(len(traj_inf)):
            l = traj_inf[t]
            event_id = event_id_slt[s, l, t0 + t]
            if event_id != -1 and event_id != 0:
                event_size = event_df[(event_df['id'] == event_id) & (event_df['sim'] == sim)]['size'].values[0]
                if l == home:
                    home_cum_ce += 1
                    home_cum_size += event_size
                else:
                    out_cum_ce += 1
                    out_cum_size += event_size
                    out_cum_attr += attr_l[l]
                    out_cum_freq += compute_location_visits(trajectory, l) / t_max
        
        if out_cum_ce != 0:
            out_avg_freq = out_cum_freq / out_cum_ce
            out_avg_attr = out_cum_attr / out_cum_ce
                
        agent_multi.append((rho, t_inf, home_cum_ce, out_cum_ce, home_cum_size, out_cum_size, home_freq, out_avg_freq, home_attr, out_avg_attr))

    return agent_multi

def count_home_and_out(agent_df, event_df, trajectory_df, space_df, nlocs, t_max):
    # Get attractiveness
    attr_l = space_df['attractiveness'].to_numpy()

    # Build event id matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    agent_multi = []

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        infected_when = int(agent['infected_when'])
        removed_when = int(agent['removed_when'])
        t_inf = removed_when - infected_when
        rho = agent['mobility']
        infected_where = int(agent['infected_where'])

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t_max = len(trajectory)
        t0 = infected_when + 1
        traj_inf = trajectory[t0:removed_when+1]
        home = trajectory[0]
        if home == infected_where:
            o = 0
        else:
            o = 1

        home_home = 0
        home_out = 0
        out_home = 0
        out_out = 0

        for t in range(len(traj_inf)):
            l = traj_inf[t]
            event_id = event_id_slt[s, l, t0 + t]
            if event_id != -1 and event_id != 0:
                if ((l == home) and (o == 0)):
                    home_home += 1
                elif ((l != home) and (o == 0)):
                    out_home += 1
                elif ((l == home) and (o == 1)):
                    home_out += 1
                elif ((l != home) and (o == 1)):
                    out_out += 1
                    
        agent_multi.append((rho, home_home, home_out, out_home, out_out))

    return agent_multi

def get_top_locations_attractiveness(agent_df, trajectory_df, space_df, t_max):
      # Get attractiveness
    attr_l = space_df['attractiveness'].to_numpy()

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    agent_multi = []

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        infected_when = int(agent['infected_when'])
        removed_when = int(agent['removed_when'])
        t_inf = removed_when - infected_when
        rho = agent['mobility']

        # Get agent's infected trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        t_max = len(trajectory)
        t0 = infected_when + 1
        traj_inf = trajectory[t0:removed_when+1]
        home = trajectory[0]
        home_counts = compute_location_visits(trajectory, home)
        home_freq = home_counts / t_max
        home_attr = attr_l[home]
        top_freq_attr = home_attr

        loc_count_array = compute_location_rank(trajectory)
        loc_top2 = loc_count_array[1][0]
        top2_freq_attr = attr_l[loc_top2]

        agent_multi.append((rho, top_freq_attr, top2_freq_attr))

    return agent_multi

def collect_recurrency(agent_df, trajectory_df, space_df, nlocs, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    visits_loc_dict = {}
    rho_loc_dict = {}
    freq_loc_dict = {}

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        status = agent['status']
    
        rho = agent['mobility']

        # Get agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        S = len(set(trajectory))
        t_max = len(trajectory)
        rank_list = compute_location_rank(trajectory)
        for l, count in rank_list:
            f_l = count / t_max
            if l in visits_loc_dict:
                visits_loc_dict[l].append(S - 1)
                rho_loc_dict[l].append(rho)
                freq_loc_dict[l].append(f_l)
            else:
                visits_loc_dict[l] = [S - 1]
                rho_loc_dict[l] = [rho]
                freq_loc_dict[l] = [f_l]
 
    return visits_loc_dict, rho_loc_dict, freq_loc_dict

def collect_infected_recurrency(agent_df, trajectory_df, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    visits_loc_dict = {}
    rho_loc_dict = {}
    freq_loc_dict = {}

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        status = agent['status']
        infected_when = int(agent['infected_when'])
        removed_when = int(agent['removed_when'])

        rho = agent['mobility']

        # Get agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        trajectory = trajectory[infected_when + 1:removed_when + 1]

        T_I = removed_when - infected_when
        if T_I > 0:
            S = len(set(trajectory))
            t_max = len(trajectory)
            rank_list = compute_location_rank(trajectory)
            for l, count in rank_list:
                f_l = count / t_max
                if l in visits_loc_dict:
                    visits_loc_dict[l].append(S - 1)
                    rho_loc_dict[l].append(rho)
                    freq_loc_dict[l].append(f_l)
                else:
                    visits_loc_dict[l] = [S - 1]
                    rho_loc_dict[l] = [rho]
                    freq_loc_dict[l] = [f_l]
 
    return visits_loc_dict, rho_loc_dict, freq_loc_dict

def collect_connectivity(agent_df, trajectory_df):
    weight_dict = {}
    rho_dict = {}
    inf_weight_dict = {}
    inf_rho_dict = {}

    for _, agent in agent_df.iterrows():
        status = agent['status']
        mob_id = agent['mob_id']
        rho = agent['mobility']

        # Get agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]

        for i in range(len(trajectory) - 1):
            l = trajectory[i]
            k = trajectory[i + 1]
            if (l, k) in weight_dict:
                weight_dict[(l, k)] += 1
                rho_dict[(l, k)].append(rho)
            else:
                weight_dict[(l, k)] = 1
                rho_dict[(l, k)] = [rho]
        
        if status == 'Removed':
            infected_when = int(agent['infected_when'])
            removed_when = int(agent['removed_when'])
            t0 = infected_when + 1
            traj_while = trajectory[t0:removed_when+1]
            T_I = removed_when - infected_when
            if T_I > 3.0:
                for i in range(len(traj_while) - 1):
                    l = traj_while[i]
                    k = traj_while[i + 1]
                    if (l, k) in inf_weight_dict:
                        inf_weight_dict[(l, k)] += 1
                        inf_rho_dict[(l, k)].append(rho)
                    else:
                        inf_weight_dict[(l, k)] = 1
                        inf_rho_dict[(l, k)] = [rho]

    return weight_dict, rho_dict, inf_weight_dict, inf_rho_dict

def collect_case_flow(agent_df, trajectory_df):
    loc_source = {}
    rho_source = {}
    loc_sink = {}
    rho_sink = {}
    loc_export = {}
    loc_import = {}
    rho_export = {}
    rho_import = {}

    for _, agent in agent_df.iterrows():
        status = agent['status']
        mob_id = agent['mob_id']
        rho = agent['mobility']
        infected_where = int(agent['infected_where'])
        infected_when = int(agent['infected_when'])
        removed_when = int(agent['removed_when'])

        loc_source.setdefault(infected_where, 0)
        loc_source[infected_where] += 1
        if infected_where in rho_source:
            rho_source[infected_where].append(rho)
        else:
            rho_source[infected_where] = [rho]
    
        # Get agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        traj_inf = trajectory[infected_when:removed_when + 1]

        for i in range(len(traj_inf) - 1):
            l = traj_inf[i]
            k = traj_inf[i + 1]

            loc_export.setdefault(l, 0)
            loc_export[l] += 1

            loc_import.setdefault(k, 0)
            loc_import[k] += 1

            if l in rho_export:
                rho_export[l].append(rho)
            else:
                rho_export[l] = [rho]
            if k in rho_import:
                rho_import[k].append(rho)
            else:
                rho_import[k] = [rho]
        
        removed_where = traj_inf[-1]
        loc_sink.setdefault(removed_where, 0)
        loc_sink[removed_where] += 1
        if removed_where in rho_sink:
            rho_sink[removed_where].append(rho)
        else:
            rho_sink[removed_where] = [rho]

    loc_import = dict(sorted(loc_import.items(), key=lambda x: x[0]))
    loc_export = dict(sorted(loc_export.items(), key=lambda x: x[0]))
    loc_source = dict(sorted(loc_source.items(), key=lambda x: x[0]))
    loc_sink = dict(sorted(loc_sink.items(), key=lambda x: x[0]))

    imports = loc_import, rho_import
    exports = loc_export, rho_export
    sources = loc_source, rho_source
    sinks = loc_sink, rho_sink

    return imports, exports, sources, sinks

def collect_location_invaders_rho_and_attractiveness(event_df, nlocs, t_max):
    # Call build_event_id_matrix to get event_id_matrix
    event_id_slt = build_event_id_array(event_df, nlocs, t_max)

    # Get the number of simulations (nsims) from the event_df DataFrame
    unique_sims = event_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    # Initialize invader_rho_array
    invader_sl = np.zeros((nsims, nlocs))
    invader_rho_sl = np.zeros((nsims, nlocs))
    invaded_loc_sl = np.zeros((nsims, nlocs))
    attractiveness_sl = np.zeros((nsims, nlocs))

    for s in range(nsims):
        for l in range(nlocs):
            event_id_t = event_id_slt[s, l]
            nonneg_event_id_t = [ind for ind in range(len(event_id_t)) if event_id_t[ind] > 0]

            # Was there an invasion at all?
            if len(nonneg_event_id_t) == 0:
                invader_rho_sl[s, l] = np.nan
            else:
                # Location was invaded: Obtain event index & rest of info
                index = nonneg_event_id_t[0]
                event_id = event_id_slt[s, l, index]
                
                #print("event_id={0}, s={1}, l={2}, index={3}".format(event_id, s, l, index))
                sim = inverse_sim_mapping[s]
                infector_rho = event_df.loc[(event_df['id'] == event_id) & (event_df['sim'] == sim), 'infector_rho'].values[0]
                infector_mob_id = event_df.loc[(event_df['id'] == event_id) & (event_df['sim'] == sim), 'infector_mob_id'].values[0]
                invader_sl[s, l] = infector_mob_id
                attractiveness_sl[s, l] = 0.0
                invaded_loc_sl[s, l] = l
                if infector_rho <= 1.0:
                    invader_rho_sl[s, l] = infector_rho
                else:
                    invader_rho_sl[s, l] = np.nan

    return invader_sl, invader_rho_sl, attractiveness_sl, invaded_loc_sl

def compute_average_degree_in_time(agent_df, trajectory_df, grid, t_max):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    avg_k_st = np.zeros((nsims, t_max))

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        epi_id = agent['epi_id']
        rho = agent['mobility']

        # Obtain agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        trajectory = trajectory[:t_max]

        # Set to keep track of agent labels seen so far along the trajectory
        seen_labels = set()

        # Loop over the trajectory
        for t, l in enumerate(trajectory):
            agent_labels = grid[l][t]
            #seen_labels.update(agent_labels)
            #degree = len(seen_labels)
            degree = len(agent_labels)
            avg_k_st[s, t] += degree - 1

    avg_k_st /= len(agent_df)

    return avg_k_st

def compute_average_degree(agent_df, trajectory_df, grid, t_ini, t_fin):
    unique_sims = agent_df['sim'].unique()
    nsims = len(unique_sims)
    sim_mapping = {sim: idx for idx, sim in enumerate(unique_sims)}
    inverse_sim_mapping = {v: k for k, v in sim_mapping.items()}

    avg_k_st = np.zeros((nsims, t_fin - t_ini))

    for _, agent in agent_df.iterrows():
        sim = agent['sim']
        s = sim_mapping[sim]
        mob_id = agent['mob_id']
        epi_id = agent['epi_id']
        rho = agent['mobility']

        # Obtain agent's trajectory
        trajectory = trajectory_df.loc[trajectory_df['mob_id'] == mob_id, 'trajectory'].values[0]
        trajectory = trajectory[t_ini:t_fin]

        # Set to keep track of agent labels seen so far along the trajectory
        seen_labels = set()

        # Loop over the trajectory
        for t, l in enumerate(trajectory):
            agent_labels = grid[l][t_ini + t]
            #seen_labels.update(agent_labels)
            #degree = len(seen_labels)
            degree = len(agent_labels)
            avg_k_st[s, t] += degree - 1

    avg_k_st /= len(agent_df)

    return np.mean(avg_k_st, axis=1)

def filter_databased_space_by_geo_limits(space_df, coord_limits):

    # Extract the coordinate limits
    n_lat = float(coord_limits['LTN'])
    s_lat = float(coord_limits['LTS'])
    w_lon = float(coord_limits['LNW'])
    e_lon = float(coord_limits['LNE'])

    # Convert DMS coordinates to decimal degrees if not None
    if coord_limits['DMS'] == True:
        if n_lat is not None:
            n_lat = ut.dms_to_decimal(n_lat[0], n_lat[1], n_lat[2])
        if s_lat is not None:
            s_lat = ut.dms_to_decimal(s_lat[0], s_lat[1], s_lat[2])
        if w_lon is not None:
            w_lon = ut.dms_to_decimal(w_lon[0], w_lon[1], w_lon[2])
        if e_lon is not None:
            e_lon = ut.dms_to_decimal(e_lon[0], e_lon[1], e_lon[2])

    # Filter the DataFrame based on coordinate limits
    if (n_lat is None) and (s_lat is None) and (w_lon is None) and (e_lon is None):
        # No filtering if all coordinates are None
        filtered_df = space_df
    else:
        filtered_df = space_df[
            ((s_lat is None) | (space_df['lat_medoid'] >= s_lat)) &
            ((n_lat is None) | (space_df['lat_medoid'] <= n_lat)) &
            ((w_lon is None) | (space_df['lon_medoid'] <= w_lon)) &
            ((e_lon is None) | (space_df['lon_medoid'] >= e_lon))
        ]

    return filtered_df

def round_coordinates(df, decimals=4):
    # Copy the dataframe to avoid modifying the original
    rounded_df = df.copy()
    
    # Trim the latitude and longitude values
    rounded_df['lat_medoid'] = rounded_df['lat_medoid'].apply(lambda x: round(x, decimals))
    rounded_df['lon_medoid'] = rounded_df['lon_medoid'].apply(lambda x: round(x, decimals))
    
    # Convert loc_id column to integer type
    rounded_df['loc_id'] = rounded_df['loc_id'].astype(int)
    
    return rounded_df

def coarse_grain_locations(df):

    desired_order = ['row', 'loc_id', 'lat_medoid', 'lon_medoid', 'counts', 'cum_duration', 'i_index', 'j_index']
    df = df.reindex(columns=desired_order)

    # Group the dataframe by latitude and longitude coordinates
    grouped = df.groupby(['lat_medoid', 'lon_medoid'], as_index=False)
    
    # Aggregate the counts and cumulative duration
    aggregated = grouped.agg({
        'loc_id': 'min',
        'counts': 'count',
        'cum_duration': 'sum'
    })
    
    # Rename the columns
    aggregated.columns = ['lat_medoid', 'lon_medoid', 'loc_id', 'counts', 'cum_duration']
    # Convert loc_id column to integer type
    aggregated['loc_id'] = aggregated['loc_id'].astype(int)
    # Create the mapping for lon_medoid
    lon_mapping = {value: index for index, value in enumerate(sorted(aggregated['lon_medoid'].unique()))}
    # Create the mapping for lat_medoid
    lat_mapping = {value: index for index, value in enumerate(sorted(aggregated['lat_medoid'].unique()))}
    # Add the i_index column to the dataframe
    aggregated['i_index'] = aggregated['lon_medoid'].map(lon_mapping)
    # Add the j_index column to the dataframe
    aggregated['j_index'] = aggregated['lat_medoid'].map(lat_mapping)

    # Should I relabel loc_id here?
    
    return aggregated

def curate_space_df(space_df, curation_dict, limit_flag, round_flag):
    # Unpack dict
    rounding = int(curation_dict['rd'])

    # Remove longitude negative signs 
    space_df['lon_medoid'] = space_df['lon_medoid'].abs()

    # Filter by geographical limits
    if limit_flag == True:
        space_df = filter_databased_space_by_geo_limits(space_df, curation_dict)
    # Coarse-grain locations
    if round_flag == True:
        space_df = round_coordinates(space_df, rounding)
        space_df = coarse_grain_locations(space_df)
    # Save to pickle
    curation_dict['nl'] = len(space_df)
    par_str = ut.dict_to_string(curation_dict)
    ext = '.pickle'
    curated_filename = 'boston_df_' + par_str
    lower_path = 'data'
    full_path = os.path.join(cwd_path, lower_path, curated_filename + ext)
    ut.save_to_pickle(space_df, full_path)

    return space_df

def build_databased_regular_lattice_space_df(space_df, regular_dict, curation_dict, norm=False):

    # Unpack parameters
    DX = float(regular_dict['DX'])
    DY = float(regular_dict['DY'])
    Lx = int(regular_dict['x'])
    Ly = int(regular_dict['y'])
    LON0 = float(regular_dict['LN0'])
    LAT0 = float(regular_dict['LT0'])
    
    # Convert grid spatial boundaries in km to meters and to decimal degrees
    DLON = ut.meters_to_decimal_longitude(DX * 1000.0, LAT0)
    DLAT = ut.meters_to_decimal_latitude(DY * 1000.0)

    # Get number of cells along every dimension
    dx = DX * 1000.0 / float(Lx)
    dy = DY * 1000.0 / float(Ly)

    # Get cell dimensions in decimal degrees
    dlon = ut.meters_to_decimal_longitude(dx, LAT0)
    dlat = ut.meters_to_decimal_latitude(dy)

    # Create new columns in space_df
    space_df['x'] = None
    space_df['y'] = None
    space_df['lon'] = None
    space_df['lat'] = None
    space_df['i_index'] = None
    space_df['j_index'] = None

    # Loop over dataframe rows
    for index, row in space_df.iterrows():
        lon = row['lon_medoid'] - LON0
        lat = row['lat_medoid'] - LAT0
        att = row['counts']
        
        # Convert longitude and latitude to x and y values
        x = ut.decimal_longitude_to_meters(lon, LAT0)
        y = ut.decimal_latitude_to_meters(lat)

        # Calculate the grid cell indices
        j_index = int(x / dx)
        i_index = int(y / dy)

        # Update the corresponding columns in space_df
        space_df.at[index, 'x'] = x
        space_df.at[index, 'y'] = y
        space_df.at[index, 'lon'] = -ut.meters_to_decimal_longitude(x, LAT0)
        space_df.at[index, 'lat'] = ut.meters_to_decimal_latitude(y)
        space_df.at[index, 'i_index'] = i_index
        space_df.at[index, 'j_index'] = j_index

    #Lx = space_df['i_index'].max() + 1
    #Ly = space_df['j_index'].max() + 1

    space_df = space_df[(space_df['i_index'] < Ly) & (space_df['j_index'] < Lx)].reset_index(drop=True)

    regular_lattice_df = pd.DataFrame(columns=['i_index', 'j_index', 'loc_id', 'attractiveness', 'x', 'y', 'lon', 'lat'])

    mapping = {}
    current_loc_id = 0

    for i_index, j_index in space_df[['i_index', 'j_index']].drop_duplicates().itertuples(index=False):
        subset = space_df[(space_df['i_index'] == i_index) & (space_df['j_index'] == j_index)]
        loc_id = mapping.get((i_index, j_index), current_loc_id)
        if (i_index, j_index) not in mapping:
            mapping[(i_index, j_index)] = current_loc_id
            current_loc_id += 1
        attractiveness = subset['counts'].sum()
        x = subset['x'].mean()
        y = subset['y'].mean()
        lon = subset['lon'].mean()
        lat = subset['lat'].mean()

        new_row = {
            'i_index': int(i_index), 
            'j_index': int(j_index), 
            'loc_id': int(loc_id), 
            'attractiveness': attractiveness, 
            'x': x, 
            'y': y,
            'lon': lon,
            'lat': lat,
            }
        #regular_lattice_df = regular_lattice_df.append(new_row, ignore_index=True)
        regular_lattice_df = pd.concat([regular_lattice_df, pd.DataFrame([new_row])], ignore_index=True)

    # Set current_loc_id to 0 before the nested loops
    current_loc_id = 0

    # Loop over all possible unique pairs (i, j) in increasing order
    for i in range(Ly):
        for j in range(Lx):
            # Check if the pair (i, j) exists in the regular_lattice_df
            if not ((regular_lattice_df['i_index'] == i) & (regular_lattice_df['j_index'] == j)).any():
                # Pair (i, j) does not exist, add a new row
                x = j * dx + dx / 2.0 # Calculate the x value based on intercell distance dx
                y = i * dy + dy / 2.0 # Calculate the y value based on intercell distance dy

                # Create the new row dictionary
                new_row = {
                    'i_index': int(i),
                    'j_index': int(j),
                    'loc_id': int(current_loc_id),
                    'attractiveness': 0.00000001,
                    'x': x,
                    'y': y,
                    'lon': -ut.meters_to_decimal_longitude(x, LAT0),
                    'lat': ut.meters_to_decimal_latitude(y),  # Replace None with your calculated latitude value
                }

                # Append the new row to the regular_lattice_df DataFrame
                regular_lattice_df = pd.concat([regular_lattice_df, pd.DataFrame([new_row])], ignore_index=True)

            else:
                # Pair (i, j) already exists, update its loc_id to the current_loc_id
                regular_lattice_df.loc[(regular_lattice_df['i_index'] == i) & (regular_lattice_df['j_index'] == j), 'loc_id'] = current_loc_id

            # Increment the current_loc_id for the next unique loc_id
            current_loc_id += 1
    
    # Cast the columns to integers
    regular_lattice_df['i_index'] = regular_lattice_df['i_index'].astype(int)
    regular_lattice_df['j_index'] = regular_lattice_df['j_index'].astype(int)
    regular_lattice_df['loc_id'] = regular_lattice_df['loc_id'].astype(int)

    regular_lattice_df = regular_lattice_df.sort_values(['i_index', 'j_index']).reset_index(drop=True)

    # Normalize attractiveness
    if norm == True:
        total_attractiveness = regular_lattice_df['attractiveness'].sum()
        regular_lattice_df['attractiveness'] = regular_lattice_df['attractiveness'] / total_attractiveness

    # Save to pickle
    reg_str = ut.dict_to_string(regular_dict)
    cur_str = ut.dict_to_string(curation_dict)
    ext = '.pickle'
    curated_filename = 'bl_' + reg_str + '_' + cur_str
    lower_path = 'data'
    full_path = os.path.join(cwd_path, lower_path, curated_filename + ext)
    ut.save_to_pickle(regular_lattice_df, full_path)

    return regular_lattice_df
    