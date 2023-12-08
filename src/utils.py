import os
import re
import json
import math
import subprocess
import numpy as np
import pandas as pd
import pickle as pk
from collections import Counter

def compute_distribution_statistics(dist):
    dist_ = dist.copy()
    dist_array = np.array(dist_)
    dist = dist_array[~np.isnan(dist_array)]
    #dist = dist_[~np.isnan(dist)]
    
    if dist.size == False:
        dist = dist_.copy()

    # Compute average value of the distribution
    dist_avg = np.mean(dist)
    # Compute standard deviation
    dist_std = np.std(dist)
    # Compute 95% confidence interval
    z = 1.96
    nsims = len(dist)
    dist_l95 = dist_avg - (z * dist_std / np.sqrt(nsims))
    dist_u95 = dist_avg + (z * dist_std / np.sqrt(nsims))
    # Compute median
    dist_med = np.median(dist)
    # Compute 5th-percentile
    #dist_p05 = np.percentile(dist, 5)
    # Compute 95th-percentiñe
    #dist_p95 = np.percentile(dist, 95)
    # Prepare output dictionary & store results
    dist_dict = {}
    dist_dict['avg'] = dist_avg
    dist_dict['std'] = dist_std
    dist_dict['l95'] = dist_l95
    dist_dict['u95'] = dist_u95
    dist_dict['med'] = dist_med
    #dist_dict['p05'] = dist_p05
    #dist_dict['p95'] = dist_p95
    dist_dict['nsims'] = nsims
    
    return dist_dict

def find_most_repeated_numbers(array, n):
    counter = Counter(array)
    most_common = counter.most_common(n)
    
    return most_common

def count_element_appearances(array, nlocs):
    counter = [0] * nlocs
    for element in array:
        counter[element] += 1
    return counter

def sir_prevalence(R0, r_0=0.0):
    # Initialize r_inf
    r_inf = 0.0
    # Self-consistent solver for r_inf
    guess = 0.8
    escape = 0
    condition = True
    while condition:
        r_inf = 1.0 - (1.0 - r_0) * np.exp(-(R0 * (guess - r_0)))
        if r_inf == guess:
            condition = False
        guess = r_inf
        escape += 1
        if escape > 10000:
            r_inf = 0.0
            condition = False
    return r_inf

def read_data_from_files(folders, file_name, extension):
    for folder in folders:
        file_path = os.path.join(folder, file_name + extension)
        if os.path.exists(file_path):
            if extension == ".json":
                with open(file_path) as json_file:
                    data = json.load(json_file)
                    return data
            elif extension == ".pickle":
                with open(file_path, "rb") as pickle_file:
                    data = pk.load(pickle_file)
                    return data
    return None

def build_path(folders):
    return os.path.join(*folders)

def build_full_path(folders, file_name, extension):
    full_path = os.path.join(*folders, file_name + extension)
    return full_path

def read_json_file(fullname):
    if not fullname.endswith('.json'):
        fullname += '.json'
    with open(fullname) as json_file:
        data = json.load(json_file)
        return data

def read_pickle_file(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as pickle_file:
        data = pk.load(pickle_file)
    return data

def build_dict_from_config_file(full_path):
    dictionary = read_json_file(full_path)
    return dictionary

def build_space_dict_from_config_file(full_path):
    space_dict = read_json_file(full_path)
    return space_dict
    
def build_mobility_dict_from_config_file(full_path):
    mobility_dict = read_json_file(full_path)
    return mobility_dict

def build_epidemic_dict_from_config_file(full_path):
    epidemic_dict = read_json_file(full_path)
    return epidemic_dict

def get_space_timestamps(space_dict):
    pass

def get_mobility_timestamps(mobility_dict):
    pass

def dict_to_string(dictionary):
    return '_'.join([f"{key}{value}" for key, value in dictionary.items()])

def find_files_with_string(string, folder_path):
    file_paths = []
    for file_name in os.listdir(folder_path):
        if string in file_name:
            full_path = os.path.join(folder_path, file_name)
            file_paths.append(full_path)
    return file_paths

def trim_file_extension(file_string):
    extensions = ['.pickle', '.pdf', '.png', '.txt', '.csv']
    
    for ext in extensions:
        if file_string.endswith(ext):
            file_string = file_string[:-(len(ext))]
            break
    
    return file_string

def trim_file_path(file_path):
    sequences = ['mdyna', 'mmeta', 'mrhod', 'mgrid', 'emeta', 'edyna', 'space']
    
    for sequence in sequences:
        if sequence in file_path:
            start_index = file_path.index(sequence)
            file_path = file_path[start_index:]
            break
    
    return file_path

def build_header_string(header_dictionary):
    return '_'.join([f"{value}" for key, value in header_dictionary.items()])

def build_space_dictionary(
        TessellationModel,
        PoleModel,
        BoundaryModel,
        AttractivenessModel,
        npoles=0,
        x_cells=None,
        y_cells=None,
        ap1=None,
    ):
    spars = {}
    if TessellationModel == 'DataBased':
        pass
    elif TessellationModel == 'RegularLaticce':
        spars['x'] = x_cells
        spars['y'] = y_cells
        spars['np'] = npoles
        spars['pm'] = PoleModel
        spars['bm'] = BoundaryModel
        spars['am'] = AttractivenessModel
        spars['ap1'] = ap1
    return spars

def build_mobility_dictionary(
        gamma,
        home_weight,
        t_max,
        RhoModel,
        p1,
        p2,
    ):
    mpars = {}
    mpars['gm'] = gamma
    mpars['hw'] = home_weight
    mpars['t'] = t_max
    mpars['rm'] = RhoModel
    mpars['p1'] = p1
    mpars['p2'] = p2
    return mpars

def build_epidemic_dictionary(
        AgentSeedModel,
        LocationSeedModel,
        nagents,
        nepicenters,
        nseeds,
        removal_rate,
        seed_fraction,
        t_epidemic,
        transmission_rate,
):
    epars = {}
    epars['as'] = AgentSeedModel
    epars['ls'] = LocationSeedModel
    epars['na'] = nagents
    epars['ne'] = nepicenters
    epars['ns'] = nseeds
    epars['rr'] = removal_rate
    epars['sf'] = seed_fraction
    epars['te'] = t_epidemic
    epars['tr'] = transmission_rate
    return epars
    
def build_string_from_json(json_file_name):
    pass

def build_file_name_from_json(head):
    pass
    
def build_space_string(path):
    lower_path = 'config/'
    path_list = [path, lower_path]
    extension = '.json'
    spa_cfn = 'config_space'
    tessellation_str = 'rl_retriever'
    spa_cfn = spa_cfn + '_' + tessellation_str
    spa_full_path = build_full_path(path_list, spa_cfn, extension)
    spa_dict = read_json_file(spa_full_path)
    spa_str = dict_to_string(spa_dict)
    return spa_str

def build_mobility_string(path):
    lower_path = 'config/'
    path_list = [path, lower_path]
    extension = '.json'
    mob_cfn = 'config_mobility_retriever'
    mob_full_path = build_full_path(path_list, mob_cfn, extension)
    mob_dict = read_json_file(mob_full_path)
    mob_str = dict_to_string(mob_dict)
    return mob_str

def build_epidemic_string(path):
    lower_path = 'config/'
    path_list = [path, lower_path]
    extension = '.json'
    epi_cfn = 'config_epidemic'
    epi_full_path = build_full_path(path_list, epi_cfn, extension)
    epi_dict = read_json_file(epi_full_path)
    epi_str = dict_to_string(epi_dict)
    return epi_str

def purge_dict(dictionary, *keys_to_remove):
    keys_to_remove = set(keys_to_remove)
    return {key: value for key, value in dictionary.items() if key not in keys_to_remove}

def purge_string(string, *str_to_remove):
    result = ""
    remove_next = False

    for i in range(len(string)):
        if string[i] == '_':
            remove_next = False
            result += string[i]
        elif remove_next:
            continue
        elif string[i:i + len(str_to_remove[0])] == str_to_remove[0]:
            remove_next = True
            continue
        else:
            result += string[i]

    return result

def collect_mobility_set_timestamps(fullpath, scenario_flag='depr', distribution_flag='Beta'):
    timestamps = []
    files = os.listdir(fullpath)
    
    for file_name in files:
        if file_name.startswith('mset_ms' + scenario_flag):
            if distribution_flag in file_name:
                match = re.search(r'_tm(\d+)_', file_name)
                if match:
                    timestamp = match.group(1)
                    timestamps.append(timestamp)
    
    return timestamps

def collect_mobility_grid_timestamps(fullpath, scenario_flag='depr', distribution_flag='Beta'):
    timestamps = []
    files = os.listdir(fullpath)
    
    for file_name in files:
        if file_name.startswith('mgrid_ms' + scenario_flag):
            if distribution_flag in file_name:
                match = re.search(r'_tm(\d+)_', file_name)
                if match:
                    timestamp = match.group(1)
                    timestamps.append(timestamp)
    
    return timestamps

def collect_agent_grid_file_names(fullpath):
    file_names = []
    for file_name in os.listdir(fullpath):
        if file_name.startswith('mgrid'):
            file_names.append(file_name)
    return file_names

def collect_digested_epidemic_file_names(fullpath):
    file_names = []
    for file_name in os.listdir(fullpath):
        if file_name.startswith('edig'):
            file_names.append(file_name)
    return file_names

def build_digested_baseline_epidemic_results_filename(bl_flag, epi_pars, grid_pars, timestamp):

    epi_str = dict_to_string(epi_pars)
    epi_head = 'edigbl'
    grid_str = dict_to_string(grid_pars)
    grid_head = 'mgrid'

    gm_index = grid_str.index('_gm')
    modified_grid_str = grid_str[:gm_index] + '_tm' + timestamp + grid_str[gm_index:]

    full_str = epi_head + '_' + epi_str + '_' + grid_head + '_' + modified_grid_str

    if not full_str.endswith('.pickle'):
        full_str += '.pickle'

    return full_str

def build_digested_epidemic_results_filename(epi_pars, grid_pars, timestamp):

    epi_str = dict_to_string(epi_pars)
    epi_head = 'edig'
    grid_str = dict_to_string(grid_pars)
    grid_head = 'mgrid'

    ms_index = grid_str.index('_ms')
    modified_grid_str = grid_str[:ms_index] + '_tm' + timestamp + grid_str[ms_index:]

    full_str = epi_head + '_' + epi_str + '_' + grid_head + '_' + modified_grid_str

    if not full_str.endswith('.pickle'):
        full_str += '.pickle'

    return full_str

def build_epidemic_results_filename(epi_pars, grid_pars, timestamp):

    epi_str = dict_to_string(epi_pars)
    epi_head = 'edyna'
    grid_str = dict_to_string(grid_pars)
    grid_head = 'mgrid'

    gm_index = grid_str.index('_gm')
    modified_grid_str = grid_str[:gm_index] + '_tm' + timestamp + grid_str[gm_index:]

    full_str = epi_head + '_' + epi_str + '_' + grid_head + '_' + modified_grid_str

    return full_str

def build_grid_results_filename(grid_pars):
    grid_str = dict_to_string(grid_pars)
    return 'mgrid_' + grid_str

def build_mobility_results_filename(grid_pars):

    # Delete irrelevant keys
    del grid_pars['na']
    del grid_pars['qm']
    del grid_pars['qf']

    mob_str = dict_to_string(grid_pars)
    mob_head = 'mdyna'
    full_str = mob_head + '_' + mob_str

    return full_str

def build_chosen_agents_filename(grid_pars, timestamp):
    mcage_str = dict_to_string(grid_pars)
    mcage_head = 'mcage'

    gm_index = mcage_str.index('_gm')
    modified_grid_str = mcage_str[:gm_index] + '_tm' + timestamp + mcage_str[gm_index:]

    full_str = mcage_head + '_' + modified_grid_str

    return full_str

def collect_pickle_filenames(fullpath, header, string_segments):
    # Get the list of files in the directory
    file_list = os.listdir(fullpath)

    # Filter files based on header and string segments
    result = []
    for file_name in file_list:
        if file_name.startswith(header) and (string_segments is None or all(segment in file_name for segment in string_segments)):
            result.append(file_name)

    return result

def collect_pickle_filenames_by_exclusion(fullpath, header, string_segment):
    # Get the list of files in the directory
    file_list = os.listdir(fullpath)

    # Filter files based on header and string segment
    result = []
    for file_name in file_list:
        if file_name.startswith(header) and (string_segment is None or not any(segment in file_name for segment in string_segment)):
            result.append(file_name)

    return result


def modify_json_file(file_path, key, value):
    with open(file_path, 'r') as file:
        data = json.load(file)
        data[key] = value

    with open(file_path, 'w') as file:
        json.dump(data, file)

def call_rust_file(
        file_path, 
        agent_seed_model=None,
        attractiveness_model=None,
        boundary_model=None,
        config_flag=None,
        escape_condition=None,
        exp_flag=None,
        expedited_escape_flag=None,
        gamma=None,
        home_model=None,
        home_weight=None,
        location_threshold=None,
        lockdown_model=None,
        locked_fraction=None,
        location_seed_model=None,
        pole_model=None,
        quarantine_model=None,
        quarantined_fraction=None,
        raw_output_flag=None,
        rho=None,
        rho_distribution_model=None,
        mobility_selection_flag=None,
        mobility_scenario_model=None,
        nagents=None,
        nepicenters=None,
        nsims=None,
        pseudomass_exponent=None,
        removal_rate=None,
        seed_fraction=None,
        t_epidemic=None,
        t_max=None,
        tessellation_model=None,
        transmission_rate=None,
        vaccination_model=None,
        vaccination_fraction=None,
        ):
    
    command = ['cargo', 'run', '-r', '--']
    if agent_seed_model is not None:
        command.extend(['--agent-seed-model', str(agent_seed_model)])
    if attractiveness_model is not None:
        command.extend(['--attractiveness-model', str(attractiveness_model)])
    if boundary_model is not None:
        command.extend(['--boundary-model', str(boundary_model)])
    if config_flag is not None:
        command.extend(['--config-flag', str(config_flag).lower()])
    if escape_condition is not None:
        command.extend(['--escape-condition', str(escape_condition)])
    if exp_flag is not None:
        command.extend(['--exp-flag', str(exp_flag)])
    if expedited_escape_flag is not None:
        command.extend(['--expedited-escape-flag', str(expedited_escape_flag).lower()])
    if gamma is not None:
        command.extend(['--gamma', str(gamma)])
    if home_model is not None:
        command.extend(['--home-model', str(home_model)])
    if home_weight is not None:
        command.extend(['--home-weight', str(home_weight)])
    if location_threshold is not None:
        command.extend(['--location-threshold', str(location_threshold)])
    if lockdown_model is not None:
        command.extend(['--lockdown-model', str(lockdown_model).lower()])
    if locked_fraction is not None:
        command.extend(['--locked-fraction', str(locked_fraction)])
    if location_seed_model is not None:
        command.extend(['--location-seed-model', str(location_seed_model).lower()])
    if mobility_selection_flag is not None:
        command.extend(['--mobility-selection-flag', str(mobility_selection_flag).lower()])
    if mobility_scenario_model is not None:
        command.extend(['--mobility-scenario-model', str(mobility_scenario_model).lower()])
    if nagents is not None:
        command.extend(['--nagents', str(nagents)])
    if nepicenters is not None:
        command.extend(['--nepicenters', str(nepicenters)])
    if nsims is not None:
        command.extend(['--nsims', str(nsims)])
    if pole_model is not None:
        command.extend(['--pole-flag', str(pole_model).lower()])
    if pseudomass_exponent is not None:
        command.extend(['--pseudomass-exponent', str(pseudomass_exponent)])
    if quarantine_model is not None:
        command.extend(['--quarantine-model', str(quarantine_model)])
    if quarantined_fraction is not None:
        command.extend(['--quarantined-fraction', str(quarantined_fraction)])
    if raw_output_flag is not None:
        command.extend(['--raw-output-flag', str(raw_output_flag).lower()])
    if rho is not None:
        command.extend(['--rho', str(rho)])
    if rho_distribution_model is not None:
        command.extend(['--rho-distribution-model', str(rho_distribution_model)])
    if removal_rate is not None:
        command.extend(['--removal-rate', str(removal_rate)])
    if seed_fraction is not None:
        command.extend(['--seed-fraction', str(seed_fraction)])
    if t_epidemic is not None:
        command.extend(['--t-epidemic', str(t_epidemic)])
    if t_max is not None:
        command.extend(['--t-max', str(t_max)])
    if tessellation_model is not None:
        command.extend(['--tessellation-model', str(tessellation_model)])
    if transmission_rate is not None:
        command.extend(['--transmission-rate', str(transmission_rate)])
    if vaccination_model is not None:
        command.extend(['--vaccination-model', str(vaccination_model)])
    if vaccination_fraction is not None:
        command.extend(['--vaccinated-fraction', str(vaccination_fraction)])

    subprocess.run(command, cwd=file_path)

def dms_to_decimal(degrees, minutes, seconds):
    decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
    return decimal_degrees

def decimal_to_dms(decimal_degrees):
    degrees = int(decimal_degrees)
    decimal_minutes = (decimal_degrees - degrees) * 60
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return degrees, minutes, seconds

def decimal_latitude_to_meters(latitude):
    return latitude * 111320.0

def decimal_longitude_to_meters(longitude, ref_latitude):
    return longitude * 40075000.0 * np.cos(np.radians(ref_latitude)) / 360.0

def meters_to_decimal_latitude(meters):
    return meters / 111320.0

def meters_to_decimal_longitude(meters, ref_latitude):
    return meters * 360.0 / (40075000.0 * np.cos(np.radians(ref_latitude))) 

def save_to_pickle(object, fullpath):
    with open(fullpath, 'wb') as f:
        pk.dump(object, f)

def open_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.pickle'):
        with open(file_path, 'rb') as f:
            df = pk.load(f)
    else:
        raise ValueError("Unsupported file extension. Only .csv and .pickle files are supported.")

    return df

def space_object_to_rust_as_json(space_df, fullname):
    # Add 'x_pbc' and 'y_pbc' columns filled with None values
    space_df['x_pbc'] = None
    space_df['y_pbc'] = None
    # Convert DataFrame to a list of dictionaries
    space_list = space_df.to_dict(orient='records')
    # Save list as JSON
    with open(fullname, 'w') as f:
        json.dump(space_list, f)

def load_chapter_figure1_data(
        fullname, 
        stats_flag=False, 
        t_inv_flag=False, 
        t_inf_flag=False, 
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['invaders'] = edig_dict['invaders_per_rho']
    output_dict['nlocs_invaded'] = edig_dict['nlocs_invaded']
    output_dict['total_cases'] = edig_dict['total_cases_per_loc']
    
    if stats_flag:
        t_inv_stats_list = []
        t_inf_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            t_inv_stats_list.append(nested_dict['t_inv_stats'])
            t_inf_stats_list.append(nested_dict['t_inf_stats'])  
        output_dict['t_inv_stats'] = t_inv_stats_list
        output_dict['t_inf_stats'] = t_inf_stats_list
    
    if t_inv_flag:
        output_dict['t_inv_dist'] = edig_dict['t_inv_dist_per_rho']
    if t_inf_flag:
        output_dict['t_inf_dist'] = edig_dict['t_inf_dist_per_rho']

    return output_dict

def load_chapter_figure2_data(
        fullname, 
        stats_flag=False, 
        r_inv_flag=False, 
        r_inf_flag=False,
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['invaders'] = edig_dict['invaders_per_rho']
    output_dict['nlocs_invaded'] = edig_dict['nlocs_invaded']
    output_dict['total_cases'] = edig_dict['total_cases_per_loc']

    if stats_flag:
        r_inv_stats_list = []
        r_inf_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            r_inv_stats_list.append(nested_dict['inv_rho_stats'])
            r_inf_stats_list.append(nested_dict['inf_rho_stats'])    
        output_dict['r_inf_stats'] = r_inv_stats_list
        output_dict['r_inv_stats'] = r_inf_stats_list

    if r_inv_flag:
        output_dict['r_inv_dist'] = edig_dict['invader_rho_per_l']
    if r_inf_flag:
        output_dict['r_inf_dist'] = edig_dict['infected_rho_per_l']

    return output_dict

def load_chapter_figure3_data(
        fullname, 
        stats_flag=False, 
        f_inf_flag=False, 
        a_inf_flag=False,
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['infected_h'] = edig_dict['infected_h_per_rho']
    output_dict['infected_o'] = edig_dict['infected_o_per_rho']

    if stats_flag:
        f_inf_h_stats_list = []
        f_inf_o_stats_list = []
        a_inf_h_stats_list = []
        a_inf_o_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            f_inf_h_stats_list.append(nested_dict['f_inf_h_stats'])
            f_inf_o_stats_list.append(nested_dict['f_inf_o_stats'])
            a_inf_h_stats_list.append(nested_dict['a_inf_h_stats'])
            a_inf_o_stats_list.append(nested_dict['a_inf_o_stats'])
        
        output_dict['f_inf_h_stats'] = f_inf_h_stats_list
        output_dict['f_inf_o_stats'] = f_inf_o_stats_list
        output_dict['a_inf_h_stats'] = a_inf_h_stats_list
        output_dict['a_inf_o_stats'] = a_inf_o_stats_list

    if f_inf_flag:
        output_dict['f_inf_h_dist'] = edig_dict['f_inf_h_dist_per_rho']
        output_dict['f_inf_o_dist'] = edig_dict['f_inf_o_dist_per_rho']
    if a_inf_flag:
        output_dict['a_inf_h_dist'] = edig_dict['a_inf_h_dist_per_rho']
        output_dict['a_inf_o_dist'] = edig_dict['a_inf_o_dist_per_rho']

    return output_dict

def load_chapter_figure4_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['events_hh'] = edig_dict['events_hh_per_rho']
    output_dict['events_ho'] = edig_dict['events_ho_per_rho']
    output_dict['events_oh'] = edig_dict['events_oh_per_rho']
    output_dict['events_oo'] = edig_dict['events_oo_per_rho']

    return output_dict

def load_chapter_figure5_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['new_cases'] = edig_dict['cases_per_rho']
    output_dict['shared_cases'] = edig_dict['shared_per_rho']
    output_dict['new_cases'] = edig_dict['cases_per_rho']
    output_dict['cum_i_pop'] = edig_dict['cum_i_pop_per_rho']
    output_dict['cum_t_pop'] = edig_dict['cum_t_pop_per_rho']
    output_dict['avg_size'] = edig_dict['avg_size_per_rho']
    output_dict['avg_foi'] = edig_dict['avg_foi_per_rho']

    return output_dict

def load_depr_chapter_panel2_data(
        fullname, 
        stats_flag=False, 
        t_inv_flag=False, 
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['invaders'] = edig_dict['invaders_per_rho']
    output_dict['nlocs_invaded'] = edig_dict['nlocs_invaded']
    
    if stats_flag:
        t_inv_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            t_inv_stats_list.append(nested_dict['t_inv_stats'])
        output_dict['t_inv_stats'] = t_inv_stats_list
    
    if t_inv_flag:
        output_dict['t_inv_dist'] = edig_dict['t_inv_dist_per_rho']

    return output_dict

def load_depr_chapter_panel3_data(
        fullname, 
        stats_flag=False, 
        r_inv_flag=False, 
        r_inf_flag=False,
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['invaders'] = edig_dict['invaders_per_rho']
    output_dict['nlocs_invaded'] = edig_dict['nlocs_invaded']
    output_dict['total_cases'] = edig_dict['total_cases_per_loc']

    if stats_flag:
        r_inv_stats_list = []
        r_inf_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            r_inv_stats_list.append(nested_dict['inv_rho_stats'])
            r_inf_stats_list.append(nested_dict['inf_rho_stats'])    
        output_dict['r_inf_stats'] = r_inv_stats_list
        output_dict['r_inv_stats'] = r_inf_stats_list

    if r_inv_flag:
        output_dict['r_inv_dist'] = edig_dict['invader_rho_per_l']
    if r_inf_flag:
        output_dict['r_inf_dist'] = edig_dict['infected_rho_per_l']

    return output_dict

def load_depr_chapter_panel4_data(
        fullname, 
        stats_flag=False,  
        t_inf_flag=False, 
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['total_cases'] = edig_dict['total_cases_per_loc']
    
    if stats_flag:
        t_inf_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            t_inf_stats_list.append(nested_dict['t_inf_stats'])  
        output_dict['t_inf_stats'] = t_inf_stats_list

    if t_inf_flag:
        output_dict['t_inf_dist'] = edig_dict['t_inf_dist_per_rho']

    return output_dict

def load_depr_chapter_panel5_data(
        fullname, 
        stats_flag=False, 
        f_inf_flag=False, 
        a_inf_flag=False,
        ):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['infected_h'] = edig_dict['infected_h_per_rho']
    output_dict['infected_o'] = edig_dict['infected_o_per_rho']

    if stats_flag:
        f_inf_h_stats_list = []
        f_inf_o_stats_list = []
        a_inf_h_stats_list = []
        a_inf_o_stats_list = []

        for nested_dict in edig_dict['binned_stats_per_sim']:
            f_inf_h_stats_list.append(nested_dict['f_inf_h_stats'])
            f_inf_o_stats_list.append(nested_dict['f_inf_o_stats'])
            a_inf_h_stats_list.append(nested_dict['a_inf_h_stats'])
            a_inf_o_stats_list.append(nested_dict['a_inf_o_stats'])
        
        output_dict['f_inf_h_stats'] = f_inf_h_stats_list
        output_dict['f_inf_o_stats'] = f_inf_o_stats_list
        output_dict['a_inf_h_stats'] = a_inf_h_stats_list
        output_dict['a_inf_o_stats'] = a_inf_o_stats_list

    if f_inf_flag:
        output_dict['f_inf_h_dist'] = edig_dict['f_inf_h_dist_per_rho']
        output_dict['f_inf_o_dist'] = edig_dict['f_inf_o_dist_per_rho']
    if a_inf_flag:
        output_dict['a_inf_h_dist'] = edig_dict['a_inf_h_dist_per_rho']
        output_dict['a_inf_o_dist'] = edig_dict['a_inf_o_dist_per_rho']

    return output_dict

def load_depr_chapter_panel6_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['avg_foi'] = edig_dict['avg_foi_per_rho']
    output_dict['avg_pc_foi'] = edig_dict['avg_pc_foi_per_rho']
    output_dict['avg_shared'] = edig_dict['avg_shared_per_rho']
    output_dict['avg_size'] = edig_dict['avg_size_per_rho']
    output_dict['avg_t_pop'] = edig_dict['avg_t_pop_per_rho']
    output_dict['cum_i_pop'] = edig_dict['cum_i_pop_per_rho']
    output_dict['cum_shared'] = edig_dict['cum_shared_per_rho']
    output_dict['cum_size'] = edig_dict['cum_size_per_rho']
    output_dict['cum_t_pop'] = edig_dict['cum_t_pop_per_rho']
    output_dict['nevents_eff'] = edig_dict['nevents_eff_per_rho']

    num_bins = 30

    cases_inf_pop_avg_rho_per_rho = []
    cases_infector_rho_per_rho = []
    size_from_ipar_per_rho = []
    size_from_ir_per_rho = []
    tot_pop_from_ipar_per_rho = []
    tot_pop_from_ir_per_rho = []

    for sim in range(len(edig_dict['event_output'])):

        inf_pop_avg_rho = edig_dict['event_output'][sim]['inf_pop_avg_rho']
        infector_rho = edig_dict['event_output'][sim]['infector_rho']
        size = edig_dict['event_output'][sim]['size']
        tot_pop = edig_dict['event_output'][sim]['tot_pop']

        counts, _ = np.histogram(inf_pop_avg_rho, bins=num_bins + 1, range=(0.0, 1.0))
        cases_inf_pop_avg_rho_per_rho.append(counts)

        counts, _ = np.histogram(infector_rho, bins=num_bins + 1, range=(0.0, 1.0))
        cases_infector_rho_per_rho.append(counts)

        size_from_ipar_cum = np.zeros(num_bins + 1)
        size_from_ir_cum = np.zeros(num_bins + 1)
        tot_pop_from_ipar_cum = np.zeros(num_bins + 1)
        tot_pop_from_ir_cum = np.zeros(num_bins + 1)

        for e, rho in enumerate(infector_rho):
            if rho <= 1.0:
                rho_idx = np.int(rho * (num_bins + 1))
                size_from_ir_cum[rho_idx] += size[e]
                tot_pop_from_ir_cum[rho_idx] += tot_pop[e]

        for e, rho in enumerate(inf_pop_avg_rho):
            rho_idx = np.int(rho * (num_bins + 1))
            size_from_ipar_cum[rho_idx] += size[e]
            tot_pop_from_ipar_cum[rho_idx] += tot_pop[e]

        size_from_ipar_per_rho.append(size_from_ipar_cum)
        size_from_ir_per_rho.append(size_from_ir_cum)
        tot_pop_from_ipar_per_rho.append(tot_pop_from_ipar_cum)
        tot_pop_from_ir_per_rho.append(tot_pop_from_ir_cum)

    output_dict['event_rho'] = cases_inf_pop_avg_rho_per_rho
    output_dict['event_infector_rho'] = cases_infector_rho_per_rho
    output_dict['event_size_from_ipar'] = size_from_ipar_per_rho
    output_dict['event_size_from_ir'] = size_from_ir_per_rho
    output_dict['event_tot_pop_from_ipar'] = tot_pop_from_ipar_per_rho
    output_dict['event_tot_pop_from_ir'] = tot_pop_from_ir_per_rho
    
    return output_dict

def load_depr_chapter_panel6_old_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['events_hh'] = edig_dict['events_hh_per_rho']
    output_dict['events_ho'] = edig_dict['events_ho_per_rho']
    output_dict['events_oh'] = edig_dict['events_oh_per_rho']
    output_dict['events_oo'] = edig_dict['events_oo_per_rho']

    return output_dict

def load_depr_chapter_panel7_data_old(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['avg_a_h'] = edig_dict['avg_a_h_per_rho']
    output_dict['avg_a_o'] = edig_dict['avg_a_h_per_rho']
    output_dict['avg_foi'] = edig_dict['avg_foi_per_rho']
    output_dict['avg_pc_foi'] = edig_dict['avg_pc_foi_per_rho']
    output_dict['avg_shared'] = edig_dict['avg_shared_per_rho']
    output_dict['avg_size'] = edig_dict['avg_size_per_rho']
    output_dict['avg_t_pop'] = edig_dict['avg_t_pop_per_rho']
    output_dict['cum_i_pop'] = edig_dict['cum_i_pop_per_rho']
    output_dict['cum_shared'] = edig_dict['cum_shared_per_rho']
    output_dict['cum_size'] = edig_dict['cum_size_per_rho']
    output_dict['cum_t_pop'] = edig_dict['cum_t_pop_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['infected_h'] = edig_dict['infected_h_per_rho']
    output_dict['infected_o'] = edig_dict['infected_o_per_rho']

    attractiveness = []
    inf_pop_avg_rho = []
    infector_rho = []
    size = []
    tot_pop = []
    for sim in range(len(edig_dict['event_output'])):
        attractiveness.append(edig_dict['event_output'][sim]['attractiveness'])
        inf_pop_avg_rho.append(edig_dict['event_output'][sim]['inf_pop_avg_rho'])
        infector_rho.append(edig_dict['event_output'][sim]['infector_rho'])
        size.append(edig_dict['event_output'][sim]['size'])
        tot_pop.append(edig_dict['event_output'][sim]['tot_pop'])

    output_dict['event_attr'] = attractiveness
    output_dict['event_rho'] = inf_pop_avg_rho
    output_dict['event_infector_rho'] = infector_rho
    output_dict['event_size'] = size
    output_dict['event_tot_pop'] = tot_pop

    return output_dict

def load_depr_chapter_panel3A_data(fullname):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['invaders'] = edig_dict['invaders_per_rho']
    output_dict['r_inv_dist'] = edig_dict['invader_rho_per_loc']
    output_dict['t_inv_dist'] = edig_dict['t_inv_dist_per_loc']

    return output_dict

def load_depr_chapter_panel3B_data(fullname):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['nlocs_invaded'] = edig_dict['nlocs_invaded']
    output_dict['total_cases'] = edig_dict['total_cases_per_loc']
    output_dict['r_inf_dist'] = edig_dict['infected_rho_per_loc']
    output_dict['pt_dist'] = edig_dict['t_peak_dist_per_loc']

    return output_dict

def load_depr_chapter_panel6extra_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['events_hh'] = edig_dict['events_hh_per_rho']
    output_dict['events_ho'] = edig_dict['events_ho_per_rho']
    output_dict['events_oh'] = edig_dict['events_oh_per_rho']
    output_dict['events_oo'] = edig_dict['events_oo_per_rho']

    output_dict['f_trip_hh_dist'] = edig_dict['f_trip_hh_dist_per_rho']
    output_dict['f_trip_ho_dist'] = edig_dict['f_trip_ho_dist_per_rho']
    output_dict['f_trip_oh_dist'] = edig_dict['f_trip_oh_dist_per_rho']
    output_dict['f_trip_oo_dist'] = edig_dict['f_trip_oo_dist_per_rho']
    output_dict['da_trip_hh_dist'] = edig_dict['da_trip_hh_dist_per_rho']
    output_dict['da_trip_ho_dist'] = edig_dict['da_trip_ho_dist_per_rho']
    output_dict['da_trip_oh_dist'] = edig_dict['da_trip_oh_dist_per_rho']
    output_dict['da_trip_oo_dist'] = edig_dict['da_trip_oo_dist_per_rho']

    output_dict['a_exp_dist'] = edig_dict['a_exp_dist_per_rho']
    output_dict['cum_p_exp'] = edig_dict['cum_p_exp_per_rho']

    output_dict['f_inf_tr_h_dist'] = edig_dict['f_inf_tr_h_dist_per_rho']
    output_dict['f_inf_tr_o_dist'] = edig_dict['f_inf_tr_o_dist_per_rho']
    
    return output_dict

def load_depr_chapter_panel7_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['avg_a_h'] = edig_dict['avg_a_h_dist_per_rho']
    output_dict['avg_a_o'] = edig_dict['avg_a_o_dist_per_rho']
    output_dict['events_hh'] = edig_dict['events_hh_per_rho']
    output_dict['events_ho'] = edig_dict['events_ho_per_rho']
    output_dict['events_oh'] = edig_dict['events_oh_per_rho']
    output_dict['events_oo'] = edig_dict['events_oo_per_rho']
    output_dict['f_inf_tr_h_dist'] = edig_dict['f_inf_tr_h_dist_per_rho']
    output_dict['f_inf_tr_o_dist'] = edig_dict['f_inf_tr_o_dist_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['infected_h'] = edig_dict['infected_h_per_rho']
    output_dict['infected_o'] = edig_dict['infected_o_per_rho']

    return output_dict

def load_depr_chapter_panelfinal_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }
    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['events_hh'] = edig_dict['events_hh_per_rho']
    output_dict['events_ho'] = edig_dict['events_ho_per_rho']
    output_dict['events_oh'] = edig_dict['events_oh_per_rho']
    output_dict['events_oo'] = edig_dict['events_oo_per_rho']

    output_dict['f_trip_hh_dist'] = edig_dict['f_trip_hh_dist_per_rho']
    output_dict['f_trip_ho_dist'] = edig_dict['f_trip_ho_dist_per_rho']
    output_dict['f_trip_oh_dist'] = edig_dict['f_trip_oh_dist_per_rho']
    output_dict['f_trip_oo_dist'] = edig_dict['f_trip_oo_dist_per_rho']
    output_dict['da_trip_hh_dist'] = edig_dict['da_trip_hh_dist_per_rho']
    output_dict['da_trip_ho_dist'] = edig_dict['da_trip_ho_dist_per_rho']
    output_dict['da_trip_oh_dist'] = edig_dict['da_trip_oh_dist_per_rho']
    output_dict['da_trip_oo_dist'] = edig_dict['da_trip_oo_dist_per_rho']

    output_dict['a_exp_dist'] = edig_dict['a_exp_dist_per_rho']
    output_dict['cum_p_exp'] = edig_dict['cum_p_exp_per_rho']

    output_dict['f_inf_tr_h_dist'] = edig_dict['f_inf_tr_h_dist_per_rho']
    output_dict['f_inf_tr_o_dist'] = edig_dict['f_inf_tr_o_dist_per_rho']

    output_dict['avg_foi'] = edig_dict['avg_foi_per_rho']
    output_dict['avg_pc_foi'] = edig_dict['avg_pc_foi_per_rho']
    output_dict['avg_shared'] = edig_dict['avg_shared_per_rho']
    output_dict['avg_size'] = edig_dict['avg_size_per_rho']
    output_dict['avg_t_pop'] = edig_dict['avg_t_pop_per_rho']
    output_dict['cum_i_pop'] = edig_dict['cum_i_pop_per_rho']
    output_dict['cum_shared'] = edig_dict['cum_shared_per_rho']
    output_dict['cum_size'] = edig_dict['cum_size_per_rho']
    output_dict['cum_t_pop'] = edig_dict['cum_t_pop_per_rho']
    output_dict['infected_h'] = edig_dict['infected_h_per_rho']
    output_dict['infected_o'] = edig_dict['infected_o_per_rho']
    output_dict['nevents_eff'] = edig_dict['nevents_eff_per_rho']

    attractiveness = []
    inf_pop_avg_rho = []
    size = []
    tot_pop = []
    for sim in range(len(edig_dict['event_output'])):
        attractiveness.append(edig_dict['event_output'][sim]['attractiveness'])
        inf_pop_avg_rho.append(edig_dict['event_output'][sim]['inf_pop_avg_rho'])
        size.append(edig_dict['event_output'][sim]['size'])
        tot_pop.append(edig_dict['event_output'][sim]['tot_pop'])

    output_dict['event_attr'] = attractiveness
    output_dict['event_rho'] = inf_pop_avg_rho
    output_dict['event_size'] = size
    output_dict['event_tot_pop'] = tot_pop
    
    return output_dict

def extend_depr_chapter_panel2_results(
        out_sim_data,
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        invaders_per_rho_sim=None,
        nlocs_invaded_sim=None,
        stats_flag=False,
        t_inv_stats_per_rho_sim=None,
        t_inv_flag=False,
        t_inv_dist_per_rho_sim=None,
    ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    invaders_per_rho_sim.extend(out_sim_data['invaders'])
    nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
    
    if stats_flag:
        t_inv_stats_per_rho_sim.extend(out_sim_data['t_inv_stats'])
    if t_inv_flag:
        t_inv_dist_per_rho_sim.extend(out_sim_data['t_inv_dist'])

def extend_depr_chapter_panel3_results(
        out_sim_data,
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        total_cases_loc_sim=None,
        stats_flag=False,
        r_inv_stats_per_loc_sim=None,
        r_inf_stats_per_loc_sim=None,
        r_inv_flag=False,
        r_inv_dist_per_loc_sim=None,
        r_inf_flag=False,
        r_inf_dist_per_loc_sim=None,
    ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    total_cases_loc_sim.extend(out_sim_data['total_cases'])
    
    if stats_flag:
        r_inv_stats_per_loc_sim.extend(out_sim_data['r_inv_stats'])
        r_inf_stats_per_loc_sim.extend(out_sim_data['r_inf_stats'])
    if r_inv_flag:
        r_inv_dist_per_loc_sim.extend(out_sim_data['r_inv_dist'])
    if r_inf_flag:
        r_inf_dist_per_loc_sim.extend(out_sim_data['r_inf_dist'])

def extend_depr_chapter_panel4_results(
    out_sim_data,
    agents_per_rho_sim=None,
    infected_per_rho_sim=None,
    stats_flag=False,
    t_inf_stats_per_rho_sim=None,
    t_inf_flag=False,
    t_inf_dist_per_rho_sim=None,
    ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
            
    if stats_flag:
        t_inf_stats_per_rho_sim.extend(out_sim_data['t_inf_stats'])
    if t_inf_flag:
        t_inf_dist_per_rho_sim.extend(out_sim_data['t_inf_dist'])

def extend_depr_chapter_panel5_results(
        out_sim_data,
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        infected_h_per_rho_sim=None, 
        infected_o_per_rho_sim=None, 
        stats_flag=False,
        f_inf_h_stats_per_rho_sim=None,
        f_inf_o_stats_per_rho_sim=None,
        a_inf_h_stats_per_rho_sim=None,
        a_inf_o_stats_per_rho_sim=None,
        f_inf_flag=False,
        f_inf_h_dist_per_rho_sim=None, 
        f_inf_o_dist_per_rho_sim=None, 
        a_inf_flag=False,
        a_inf_h_dist_per_rho_sim=None, 
        a_inf_o_dist_per_rho_sim=None,
        ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
    infected_o_per_rho_sim.extend(out_sim_data['infected_o'])

    if stats_flag:
        f_inf_h_stats_per_rho_sim.extend(out_sim_data['f_inf_h_stats'])
        f_inf_o_stats_per_rho_sim.extend(out_sim_data['f_inf_o_stats'])
        a_inf_h_stats_per_rho_sim.extend(out_sim_data['a_inf_h_stats'])
        a_inf_o_stats_per_rho_sim.extend(out_sim_data['a_inf_o_stats'])
    if f_inf_flag:
        f_inf_h_dist_per_rho_sim.extend(out_sim_data['f_inf_h_dist'])
        f_inf_o_dist_per_rho_sim.extend(out_sim_data['f_inf_o_dist'])
    if a_inf_flag:
        a_inf_h_dist_per_rho_sim.extend(out_sim_data['a_inf_h_dist'])
        a_inf_o_dist_per_rho_sim.extend(out_sim_data['a_inf_o_dist'])

def extend_depr_chapter_panel6_results_old(
        out_sim_data, 
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        events_hh_per_rho_sim=None,
        events_ho_per_rho_sim=None,
        events_oh_per_rho_sim=None,
        events_oo_per_rho_sim=None,
        ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
    events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
    events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
    events_oo_per_rho_sim.extend(out_sim_data['events_oo'])

def extend_depr_chapter_panel6_results(
        out_sim_data,
        agents_per_rho_sim,
        infected_per_rho_sim,
        nevents_eff_per_rho_sim,
        sum_avg_foi_per_rho_sim,
        sum_avg_pc_foi_per_rho_sim,
        sum_avg_shared_per_rho_sim,
        sum_avg_size_per_rho_sim,
        sum_avg_t_pop_per_rho_sim,
        sum_cum_i_pop_per_rho_sim,
        sum_cum_shared_per_rho_sim,
        sum_cum_size_per_rho_sim,
        sum_cum_t_pop_per_rho_sim,
        event_inf_pop_avg_rho_per_rho_sim,
        event_infector_rho_per_rho_sim,
        event_size_from_ipar_per_rho_sim,
        event_size_from_ir_per_rho_sim,
        event_tot_pop_from_ipar_per_rho_sim,
        event_tot_pop_from_ir_per_rho_sim,
    ):

    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    nevents_eff_per_rho_sim.extend(out_sim_data['nevents_eff'])
    sum_avg_foi_per_rho_sim.extend(out_sim_data['avg_foi'])
    sum_avg_pc_foi_per_rho_sim.extend(out_sim_data['avg_pc_foi'])
    sum_avg_shared_per_rho_sim.extend(out_sim_data['avg_shared'])
    sum_avg_size_per_rho_sim.extend(out_sim_data['avg_size'])
    sum_avg_t_pop_per_rho_sim.extend(out_sim_data['avg_t_pop'])
    sum_cum_i_pop_per_rho_sim.extend(out_sim_data['cum_i_pop'])
    sum_cum_shared_per_rho_sim.extend(out_sim_data['cum_shared'])
    sum_cum_size_per_rho_sim.extend(out_sim_data['cum_size'])
    sum_cum_t_pop_per_rho_sim.extend(out_sim_data['cum_t_pop'])

    event_inf_pop_avg_rho_per_rho_sim.extend(out_sim_data['event_rho'])
    event_infector_rho_per_rho_sim.extend(out_sim_data['event_infector_rho'])
    event_size_from_ipar_per_rho_sim.extend(out_sim_data['event_size_from_ipar'])
    event_size_from_ir_per_rho_sim.extend(out_sim_data['event_size_from_ir'])
    event_tot_pop_from_ipar_per_rho_sim.extend(out_sim_data['event_tot_pop_from_ipar'])
    event_tot_pop_from_ir_per_rho_sim.extend(out_sim_data['event_tot_pop_from_ir'])

def extend_depr_chapter_panel7_results(
        out_sim_data,
        agents_per_rho_sim,
        avg_a_h_dist_per_rho_sim,
        avg_a_o_dist_per_rho_sim,
        events_hh_per_rho_sim,
        events_ho_per_rho_sim,
        events_oh_per_rho_sim,
        events_oo_per_rho_sim,
        f_inf_tr_h_dist_per_rho_sim,
        f_inf_tr_o_dist_per_rho_sim,
        infected_per_rho_sim,
        infected_h_per_rho_sim,
        infected_o_per_rho_sim,
    ):

    agents_per_rho_sim.extend(out_sim_data['agents'])
    events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
    events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
    events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
    events_oo_per_rho_sim.extend(out_sim_data['events_oo'])
    f_inf_tr_h_dist_per_rho_sim.extend(out_sim_data['f_inf_tr_h_dist'])
    f_inf_tr_o_dist_per_rho_sim.extend(out_sim_data['f_inf_tr_o_dist'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
    infected_o_per_rho_sim.extend(out_sim_data['infected_o'])
    avg_a_h_dist_per_rho_sim.extend(out_sim_data['avg_a_h'])
    avg_a_o_dist_per_rho_sim.extend(out_sim_data['avg_a_o'])

def extend_depr_chapter_panel7_results_old(
        out_sim_data, 
        agents_per_rho_sim,
        infected_per_rho_sim,
        infected_h_per_rho_sim,
        infected_o_per_rho_sim,
        sum_avg_a_h_per_rho_sim,
        sum_avg_a_o_per_rho_sim,
        sum_avg_foi_per_rho_sim,
        sum_avg_pc_foi_per_rho_sim,
        sum_avg_shared_per_rho_sim,
        sum_avg_size_per_rho_sim,
        sum_avg_t_pop_per_rho_sim,
        sum_cum_i_pop_per_rho_sim,
        sum_cum_shared_per_rho_sim,
        sum_cum_size_per_rho_sim,
        sum_cum_t_pop_per_rho_sim,
        event_attractiveness_sim,
        event_inf_pop_avg_rho_sim,
        event_infector_rho_sim,
        event_size_sim,
        event_tot_pop_sim,
        ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
    infected_o_per_rho_sim.extend(out_sim_data['infected_o'])
    sum_avg_a_h_per_rho_sim.extend(out_sim_data['avg_a_h'])
    sum_avg_a_o_per_rho_sim.extend(out_sim_data['avg_a_o'])
    sum_avg_foi_per_rho_sim.extend(out_sim_data['avg_foi'])
    sum_avg_pc_foi_per_rho_sim.extend(out_sim_data['avg_pc_foi'])
    sum_avg_shared_per_rho_sim.extend(out_sim_data['avg_shared'])
    sum_avg_size_per_rho_sim.extend(out_sim_data['avg_size'])
    sum_avg_t_pop_per_rho_sim.extend(out_sim_data['avg_t_pop'])
    sum_cum_i_pop_per_rho_sim.extend(out_sim_data['cum_i_pop'])
    sum_cum_shared_per_rho_sim.extend(out_sim_data['cum_shared'])
    sum_cum_size_per_rho_sim.extend(out_sim_data['cum_size'])
    sum_cum_t_pop_per_rho_sim.extend(out_sim_data['cum_t_pop'])

    event_attractiveness_sim.extend(out_sim_data['event_attr'])
    event_inf_pop_avg_rho_sim.extend(out_sim_data['event_rho'])
    event_infector_rho_sim.extend(out_sim_data['event_infector_rho'])
    event_size_sim.extend(out_sim_data['event_size'])
    event_tot_pop_sim.extend(out_sim_data['event_tot_pop'])

def extend_depr_chapter_panel3A_results(
        out_sim_data,
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        r_inv_dist_per_loc_sim=None,
        t_inv_dist_per_loc_sim=None,
    ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    r_inv_dist_per_loc_sim.extend(out_sim_data['r_inv_dist'])
    t_inv_dist_per_loc_sim.extend(out_sim_data['t_inv_dist'])

def extend_depr_chapter_panel3B_results(
        out_sim_data,
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        total_cases_loc_sim=None,
        r_inf_dist_per_loc_sim=None,
        pt_dist_per_loc_sim=None,
    ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    total_cases_loc_sim.extend(out_sim_data['total_cases'])
    r_inf_dist_per_loc_sim.extend(out_sim_data['r_inf_dist'])
    pt_dist_per_loc_sim.extend(out_sim_data['pt_dist'])

def extend_depr_chapter_panel6extra_results(
        out_sim_data, 
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        events_hh_per_rho_sim=None,
        events_ho_per_rho_sim=None,
        events_oh_per_rho_sim=None,
        events_oo_per_rho_sim=None,
        f_trip_hh_dist_per_rho_sim=None,
        f_trip_ho_dist_per_rho_sim=None,
        f_trip_oh_dist_per_rho_sim=None,
        f_trip_oo_dist_per_rho_sim=None,
        da_trip_hh_dist_per_rho_sim=None,
        da_trip_ho_dist_per_rho_sim=None,
        da_trip_oh_dist_per_rho_sim=None,
        da_trip_oo_dist_per_rho_sim=None,
        a_exp_dist_per_rho_sim=None,
        sum_p_exp_per_rho_sim=None,
        ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
    events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
    events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
    events_oo_per_rho_sim.extend(out_sim_data['events_oo'])
    f_trip_hh_dist_per_rho_sim.extend(out_sim_data['f_trip_hh_dist'])
    f_trip_ho_dist_per_rho_sim.extend(out_sim_data['f_trip_ho_dist'])
    f_trip_oh_dist_per_rho_sim.extend(out_sim_data['f_trip_oh_dist'])
    f_trip_oo_dist_per_rho_sim.extend(out_sim_data['f_trip_oo_dist'])
    da_trip_hh_dist_per_rho_sim.extend(out_sim_data['da_trip_hh_dist'])
    da_trip_ho_dist_per_rho_sim.extend(out_sim_data['da_trip_ho_dist'])
    da_trip_oh_dist_per_rho_sim.extend(out_sim_data['da_trip_oh_dist'])
    da_trip_oo_dist_per_rho_sim.extend(out_sim_data['da_trip_oo_dist'])
    a_exp_dist_per_rho_sim.extend(out_sim_data['a_exp_dist'])
    sum_p_exp_per_rho_sim.extend(out_sim_data['cum_p_exp'])

def extend_depr_chapter_panelfinal_results(
        out_sim_data, 
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        events_hh_per_rho_sim=None,
        events_ho_per_rho_sim=None,
        events_oh_per_rho_sim=None,
        events_oo_per_rho_sim=None,
        f_trip_hh_dist_per_rho_sim=None,
        f_trip_ho_dist_per_rho_sim=None,
        f_trip_oh_dist_per_rho_sim=None,
        f_trip_oo_dist_per_rho_sim=None,
        da_trip_hh_dist_per_rho_sim=None,
        da_trip_ho_dist_per_rho_sim=None,
        da_trip_oh_dist_per_rho_sim=None,
        da_trip_oo_dist_per_rho_sim=None,
        a_exp_dist_per_rho_sim=None,
        sum_p_exp_per_rho_sim=None,
        infected_h_per_rho_sim=None,
        infected_o_per_rho_sim=None,
        sum_avg_foi_per_rho_sim=None,
        sum_avg_pc_foi_per_rho_sim=None,
        sum_avg_shared_per_rho_sim=None,
        sum_avg_size_per_rho_sim=None,
        sum_avg_t_pop_per_rho_sim=None,
        sum_cum_i_pop_per_rho_sim=None,
        sum_cum_shared_per_rho_sim=None,
        sum_cum_size_per_rho_sim=None,
        sum_cum_t_pop_per_rho_sim=None,
        event_attractiveness_sim=None,
        event_inf_pop_avg_rho_sim=None,
        event_size_sim=None,
        event_tot_pop_sim=None,
        f_inf_tr_h_dist_per_rho_sim=None,
        f_inf_tr_o_dist_per_rho_sim=None,
        nevents_eff_per_rho_sim=None,
        ):
    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
    events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
    events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
    events_oo_per_rho_sim.extend(out_sim_data['events_oo'])
    f_trip_hh_dist_per_rho_sim.extend(out_sim_data['f_trip_hh_dist'])
    f_trip_ho_dist_per_rho_sim.extend(out_sim_data['f_trip_ho_dist'])
    f_trip_oh_dist_per_rho_sim.extend(out_sim_data['f_trip_oh_dist'])
    f_trip_oo_dist_per_rho_sim.extend(out_sim_data['f_trip_oo_dist'])
    da_trip_hh_dist_per_rho_sim.extend(out_sim_data['da_trip_hh_dist'])
    da_trip_ho_dist_per_rho_sim.extend(out_sim_data['da_trip_ho_dist'])
    da_trip_oh_dist_per_rho_sim.extend(out_sim_data['da_trip_oh_dist'])
    da_trip_oo_dist_per_rho_sim.extend(out_sim_data['da_trip_oo_dist'])
    a_exp_dist_per_rho_sim.extend(out_sim_data['a_exp_dist'])
    sum_p_exp_per_rho_sim.extend(out_sim_data['cum_p_exp'])

    infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
    infected_o_per_rho_sim.extend(out_sim_data['infected_o'])
    sum_avg_foi_per_rho_sim.extend(out_sim_data['avg_foi'])
    sum_avg_pc_foi_per_rho_sim.extend(out_sim_data['avg_pc_foi'])
    sum_avg_shared_per_rho_sim.extend(out_sim_data['avg_shared'])
    sum_avg_size_per_rho_sim.extend(out_sim_data['avg_size'])
    sum_avg_t_pop_per_rho_sim.extend(out_sim_data['avg_t_pop'])
    sum_cum_i_pop_per_rho_sim.extend(out_sim_data['cum_i_pop'])
    sum_cum_shared_per_rho_sim.extend(out_sim_data['cum_shared'])
    sum_cum_size_per_rho_sim.extend(out_sim_data['cum_size'])
    sum_cum_t_pop_per_rho_sim.extend(out_sim_data['cum_t_pop'])

    event_attractiveness_sim.extend(out_sim_data['event_attr'])
    event_inf_pop_avg_rho_sim.extend(out_sim_data['event_rho'])
    event_size_sim.extend(out_sim_data['event_size'])
    event_tot_pop_sim.extend(out_sim_data['event_tot_pop'])

    f_inf_tr_h_dist_per_rho_sim.extend(out_sim_data['f_inf_tr_h_dist'])
    f_inf_tr_o_dist_per_rho_sim.extend(out_sim_data['f_inf_tr_o_dist'])

    nevents_eff_per_rho_sim.extend(out_sim_data['nevents_eff'])

def compute_depr_chapter_panel2_stats(
        agents_per_rho_sim,
        infected_per_rho_sim,
        invaders_per_rho_sim,
        nlocs_invaded_sim,
        prevalence_cutoff=0.025,
        stats_flag=False, 
        t_inv_stats_per_rho_sim=False,
        t_inv_flag=False,
        t_inv_dist_per_rho_sim=False,
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    invaders_per_rho_sim = np.array(invaders_per_rho_sim)
    nlocs_invaded_sim = np.array(nlocs_invaded_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    invaders_per_rho_sim = np.delete(invaders_per_rho_sim, failed_outbreaks, axis=0)
    nlocs_invaded_sim = np.delete(nlocs_invaded_sim, failed_outbreaks, axis=0)

    invaded_fraction_per_rho_sim = invaders_per_rho_sim / nlocs_invaded_sim[:, np.newaxis]
    invaded_fraction_sim = np.sum(invaders_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)

    if stats_flag:
        t_inv_stats_per_loc_sim = [sim for i, sim in enumerate(t_inv_stats_per_loc_sim) if i not in failed_outbreaks]
        t_inv_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in t_inv_stats_per_rho_sim]

        avg_t_inv_avg_per_rho = np.nanmean(np.array(t_inv_avg_dist_per_rho), axis=0)
        
        std_t_inv_avg_per_rho = np.nanstd(np.array(t_inv_avg_dist_per_rho), axis=0)

        z = 1.96
        moe = z * (std_t_inv_avg_per_rho / np.sqrt(len(avg_t_inv_avg_per_rho)))
        u95_t_inv_avg_per_rho = avg_t_inv_avg_per_rho + moe
        l95_t_inv_avg_per_rho = avg_t_inv_avg_per_rho - moe

    if t_inv_flag:
        t_inv_dist_per_rho_sim = [sim for i, sim in enumerate(t_inv_dist_per_rho_sim) if i not in failed_outbreaks]
        nbins = len(t_inv_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
        t_inv_dist_per_rho = [[] for _ in range(nbins)]

        for sim_idx in range(len(t_inv_dist_per_rho_sim)):
            for rho_idx in range(nbins):
                t_inv_values = t_inv_dist_per_rho_sim[sim_idx][rho_idx]
                t_inv_dist_per_rho[rho_idx].extend(t_inv_values)

        t_inv_avg_per_rho = np.array([np.nanmean(sublist) for sublist in t_inv_dist_per_rho])
        
        t_inv_std_per_rho = np.array([np.nanstd(sublist) for sublist in t_inv_dist_per_rho])

        z = 1.96
        moe = z * (t_inv_std_per_rho / np.sqrt([len(sublist) for sublist in t_inv_dist_per_rho]))
        t_inv_u95_per_rho = t_inv_avg_per_rho + moe
        t_inv_l95_per_rho = t_inv_avg_per_rho - moe

        flattened_list = [num for sublist1 in t_inv_dist_per_rho_sim for sublist2 in sublist1 for num in sublist2]
        t_inv_avg_global = np.nanmean(flattened_list)
        t_inv_std_global = np.nanstd(flattened_list)
        moe = z * t_inv_std_global / np.sqrt(len(flattened_list))
        t_inv_u95_global = t_inv_avg_global + moe
        t_inv_l95_global = t_inv_avg_global - moe

    invaded_fraction_avg_per_rho = np.nanmean(invaded_fraction_per_rho_sim, axis=0)
    invaded_fraction_avg = np.nanmean(invaded_fraction_sim)
    
    std = np.std(invaded_fraction_per_rho_sim, axis=0)
    nsims = len(invaded_fraction_per_rho_sim)
    
    z = 1.96
    moe = z * (std / np.sqrt(nsims))
    invaded_fraction_u95_per_rho = invaded_fraction_avg_per_rho + moe
    invaded_fraction_l95_per_rho = invaded_fraction_avg_per_rho - moe

    output = {}
    output['inv_avg_per_rho'] = invaded_fraction_avg_per_rho
    output['inv_l95_per_rho'] = invaded_fraction_l95_per_rho
    output['inv_u95_per_rho'] = invaded_fraction_u95_per_rho
    output['inv_avg'] = invaded_fraction_avg
    
    if t_inv_flag:
        output['t_inv_avg_per_rho'] = t_inv_avg_per_rho
        output['t_inv_l95_per_rho'] = t_inv_l95_per_rho
        output['t_inv_u95_per_rho'] = t_inv_u95_per_rho
        output['t_inv_avg_global'] = t_inv_avg_global
        output['t_inv_l95_global'] = t_inv_l95_global
        output['t_inv_u95_global'] = t_inv_u95_global

    return output

def compute_depr_chapter_panel3_stats(
        agents_per_rho_sim, 
        infected_per_rho_sim,
        total_cases_loc_sim,
        space_df,
        stats_flag=False,
        r_inv_stats_per_loc_sim=False,
        prevalence_cutoff=0.025,
        r_inv_flag=False,
        r_inv_dist_per_loc_sim=False,
        r_inf_flag=False,
        r_inf_dist_per_loc_sim=False,
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    total_cases_loc_sim = np.array(total_cases_loc_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    total_cases_loc_sim = np.delete(total_cases_loc_sim, failed_outbreaks, axis=0)

    total_cases_avg_loc = np.nanmean(total_cases_loc_sim, axis=0)
    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001
    nlocs_eff = len(attr_l[attr_l > attr_cutoff])

    if stats_flag:
        r_inv_stats_per_loc_sim = [sim for i, sim in enumerate(r_inv_stats_per_loc_sim) if i not in failed_outbreaks]
        r_inf_stats_per_loc_sim = [sim for i, sim in enumerate(r_inf_stats_per_loc_sim) if i not in failed_outbreaks]

        r_inv_avg_dist_per_loc = [[loc_data['mean'] for loc_data in sim_data] for sim_data in r_inv_stats_per_loc_sim]
        avg_r_inv_avg_per_loc = np.nanmean(np.array(r_inv_avg_dist_per_loc), axis=0)

        r_inf_avg_dist_per_loc = [[loc_data['mean'] for loc_data in sim_data] for sim_data in r_inf_stats_per_loc_sim]
        avg_r_inf_avg_per_loc = np.nanmean(np.array(r_inf_avg_dist_per_loc), axis=0)

        nlocs = 2500
        x_cells = int(np.sqrt(nlocs))
        y_cells = x_cells
        rho_avg_lattice = np.zeros((x_cells, y_cells))
        l = 0
        for i in range(x_cells):
            for j in range(y_cells):
                rho_avg_lattice[y_cells - 1 -j, i] = avg_r_inv_avg_per_loc[l]
                l += 1

    if r_inv_flag:
        r_inv_dist_per_loc_sim = [sim for i, sim in enumerate(r_inv_dist_per_loc_sim) if i not in failed_outbreaks]
        
        nlocs = len(r_inv_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
        r_inv_dist_per_loc = [[] for _ in range(nlocs)]

        invaded_loc_sim = np.zeros((len(r_inv_dist_per_loc_sim), nlocs))

        for sim_idx in range(len(r_inv_dist_per_loc_sim)):
            for loc_idx in range(nlocs):
                r_inv_values = r_inv_dist_per_loc_sim[sim_idx][loc_idx]
                r_inv_dist_per_loc[loc_idx].extend(r_inv_values)

                if len(r_inv_values) > 0:
                    if not math.isnan(r_inv_values[0]):
                        invaded_loc_sim[sim_idx][loc_idx] = 1

        invasion_fraction_avg_loc = np.mean(invaded_loc_sim, axis=0)
        invasion_fraction_avg = np.mean(np.sum(invaded_loc_sim, axis=1)) / nlocs_eff

        r_inv_avg_per_loc = np.array([np.nanmean(sublist) for sublist in r_inv_dist_per_loc])
        
        r_inv_std_per_loc = np.array([np.nanstd(sublist) for sublist in r_inv_dist_per_loc])
        
        z = 1.96
        moe = z * (r_inv_std_per_loc / np.sqrt(len(r_inv_avg_per_loc)))
        r_inv_u95_per_loc = r_inv_avg_per_loc + moe
        r_inv_l95_per_loc = r_inv_avg_per_loc - moe

        nlocs = 2500
        x_cells = int(np.sqrt(nlocs))
        y_cells = x_cells
        inv_rho_avg_lattice = np.zeros((x_cells, y_cells))
        inv_rate_avg_lattice = np.zeros((x_cells, y_cells))
        l = 0
        for i in range(x_cells):
            for j in range(y_cells):
                inv_rho_avg_lattice[y_cells - 1 - j, i] = r_inv_avg_per_loc[l]
                inv_rate_avg_lattice[y_cells - 1 - j, i] = invasion_fraction_avg_loc[l]
                l += 1

    if r_inf_flag:
        r_inf_dist_per_loc_sim = [sim for i, sim in enumerate(r_inf_dist_per_loc_sim) if i not in failed_outbreaks]
        
        nlocs = len(r_inf_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
        r_inf_dist_per_loc = [[] for _ in range(nlocs)]

        for sim_idx in range(len(r_inf_dist_per_loc_sim)):
            for loc_idx in range(nlocs):
                r_inf_values = r_inf_dist_per_loc_sim[sim_idx][loc_idx]
                r_inf_dist_per_loc[loc_idx].extend(r_inf_values)

        r_inf_avg_per_loc = np.array([np.nanmean(sublist) for sublist in r_inf_dist_per_loc])
        
        r_inf_std_per_loc = np.array([np.nanstd(sublist) for sublist in r_inf_dist_per_loc])
        
        z = 1.96
        moe = z * (r_inf_std_per_loc / np.sqrt(len(r_inf_avg_per_loc)))
        r_inf_u95_per_loc = r_inf_avg_per_loc + moe
        r_inf_l95_per_loc = r_inf_avg_per_loc - moe

        nlocs = 2500
        x_cells = int(np.sqrt(nlocs))
        y_cells = x_cells
        inf_rho_avg_lattice = np.zeros((x_cells, y_cells))
        l = 0
        for i in range(x_cells):
            for j in range(y_cells):
                inf_rho_avg_lattice[y_cells - 1 -j, i] = r_inf_avg_per_loc[l]
                l += 1

    output = {}
    output['total_cases_avg_loc'] = total_cases_avg_loc
    output['attractiveness_l'] = attr_l

    if r_inv_flag:
        output['inv_rho_avg_lattice'] = inv_rho_avg_lattice
        output['inv_rho_avg_loc'] = r_inv_avg_per_loc
        output['invasion_fraction_avg'] = invasion_fraction_avg
        output['inv_rate_avg_lattice'] = inv_rate_avg_lattice
        output['invasions_fraction_avg_loc'] = invasion_fraction_avg_loc

    if r_inf_flag:
        output['inf_rho_avg_lattice'] = inf_rho_avg_lattice
        output['inf_rho_avg_loc'] = r_inf_avg_per_loc

    return output

def compute_depr_chapter_panel4_stats(
        agents_per_rho_sim,
        infected_per_rho_sim,
        prevalence_cutoff=0.025,
        stats_flag=False, 
        t_inf_stats_per_rho_sim=False,
        t_inf_flag=False,
        t_inf_dist_per_rho_sim=False,
        ):

    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)

    infected_fraction_per_rho_sim = infected_per_rho_sim / agents_per_rho_sim
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)

    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_fraction_sim = np.delete(infected_fraction_sim, failed_outbreaks)
    infected_fraction_per_rho_sim = np.delete(infected_fraction_per_rho_sim, failed_outbreaks, axis=0)

    if stats_flag:
        t_inf_stats_per_loc_sim = [sim for i, sim in enumerate(t_inf_stats_per_loc_sim) if i not in failed_outbreaks]
        t_inf_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in t_inf_stats_per_rho_sim]
        
        avg_t_inf_avg_per_rho = np.nanmean(np.array(t_inf_avg_dist_per_rho), axis=0)
        
        std_t_inf_avg_per_rho = np.nanstd(np.array(t_inf_avg_dist_per_rho), axis=0)
        
        z = 1.96
        moe = z * (std_t_inf_avg_per_rho / np.sqrt(len(avg_t_inf_avg_per_rho)))
        u95_t_inf_avg_per_rho = avg_t_inf_avg_per_rho + moe
        l95_t_inf_avg_per_rho = avg_t_inf_avg_per_rho - moe

    if t_inf_flag:
        t_inf_dist_per_rho_sim = [sim for i, sim in enumerate(t_inf_dist_per_rho_sim) if i not in failed_outbreaks]
        nbins = len(t_inf_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
        t_inf_dist_per_rho = [[] for _ in range(nbins)]

        for sim_idx in range(len(t_inf_dist_per_rho_sim)):
            for rho_idx in range(nbins):
                t_inf_values = t_inf_dist_per_rho_sim[sim_idx][rho_idx]
                t_inf_dist_per_rho[rho_idx].extend(t_inf_values)
        
        t_inf_avg_per_rho = np.array([np.nanmean(sublist) for sublist in t_inf_dist_per_rho])
        
        t_inf_std_per_rho = np.array([np.nanstd(sublist) for sublist in t_inf_dist_per_rho])
        
        z = 1.96
        moe = z * (t_inf_std_per_rho / np.sqrt([len(sublist) for sublist in t_inf_dist_per_rho]))
        t_inf_u95_per_rho = t_inf_avg_per_rho + moe
        t_inf_l95_per_rho = t_inf_avg_per_rho - moe

        flattened_list = [num for sublist1 in t_inf_dist_per_rho_sim for sublist2 in sublist1 for num in sublist2]
        t_inf_avg_global = np.nanmean(flattened_list)
        t_inf_std_global = np.nanstd(flattened_list)
        moe = z * t_inf_std_global / np.sqrt(len(flattened_list))
        t_inf_u95_global = t_inf_avg_global + moe
        t_inf_l95_global = t_inf_avg_global - moe

    infected_fraction_avg_per_rho = np.nanmean(infected_fraction_per_rho_sim, axis=0)
    std = np.std(infected_fraction_per_rho_sim, axis=0)
    nsims = len(infected_fraction_per_rho_sim)
    z = 1.96
    moe = z * (std / np.sqrt(nsims))
    infected_fraction_u95_per_rho = infected_fraction_avg_per_rho + moe
    infected_fraction_l95_per_rho = infected_fraction_avg_per_rho - moe

    infected_fraction_avg = np.nanmean(infected_fraction_sim)
    infected_fraction_std = np.nanstd(infected_fraction_sim)
    moe = z * (infected_fraction_std / np.sqrt(len(infected_fraction_sim)))
    infected_fraction_u95 = infected_fraction_avg + moe
    infected_fraction_l95 = infected_fraction_avg - moe

    output = {}
    output['inf_avg_per_rho'] = infected_fraction_avg_per_rho
    output['inf_l95_per_rho'] = infected_fraction_l95_per_rho
    output['inf_u95_per_rho'] = infected_fraction_u95_per_rho
    output['inf_avg'] = infected_fraction_avg
    output['inf_l95'] = infected_fraction_l95
    output['inf_u95'] = infected_fraction_u95

    if t_inf_flag:
        output['t_inf_avg_per_rho'] = t_inf_avg_per_rho
        output['t_inf_l95_per_rho'] = t_inf_l95_per_rho
        output['t_inf_u95_per_rho'] = t_inf_u95_per_rho
        output['t_inf_avg_global'] = t_inf_avg_global
        output['t_inf_l95_global'] = t_inf_l95_global
        output['t_inf_u95_global'] = t_inf_u95_global

    return output

def compute_depr_chapter_panel5_stats(
        agents_per_rho_sim,
        infected_per_rho_sim,
        infected_h_per_rho_sim=False,
        infected_o_per_rho_sim=False,
        prevalence_cutoff=0.025, 
        stats_flag=False, 
        f_inf_h_stats_per_rho_sim=False,
        f_inf_o_stats_per_rho_sim=False,
        a_inf_h_stats_per_rho_sim=False,
        a_inf_o_stats_per_rho_sim=False,
        f_inf_flag=False, 
        f_inf_h_dist_per_rho_sim=False,
        f_inf_o_dist_per_rho_sim=False,
        a_inf_flag=False,
        a_inf_h_dist_per_rho_sim=False,
        a_inf_o_dist_per_rho_sim=False,
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    infected_h_per_rho_sim = np.array(infected_h_per_rho_sim)
    infected_o_per_rho_sim = np.array(infected_o_per_rho_sim)
    
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    infected_h_per_rho_sim = np.delete(infected_h_per_rho_sim, failed_outbreaks, axis=0)
    infected_o_per_rho_sim = np.delete(infected_o_per_rho_sim, failed_outbreaks, axis=0)

    if stats_flag:
        f_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(f_inf_h_stats_per_rho_sim) if i not in failed_outbreaks]
        f_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(f_inf_o_stats_per_rho_sim) if i not in failed_outbreaks]
        
        a_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(a_inf_h_stats_per_rho_sim) if i not in failed_outbreaks]
        a_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(a_inf_o_stats_per_rho_sim) if i not in failed_outbreaks]
        
        # Compute stats
        f_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in f_inf_h_stats_per_rho_sim]
        avg_f_inf_h_avg_per_rho = np.mean(np.array(f_inf_h_avg_dist_per_rho), axis=0)
        f_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in f_inf_h_stats_per_rho_sim]
        u95_f_inf_h_avg_per_rho = np.mean(np.array(f_inf_h_u95_dist_per_rho), axis=0)
        f_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in f_inf_h_stats_per_rho_sim]
        l95_f_inf_h_avg_per_rho = np.mean(np.array(f_inf_h_l95_dist_per_rho), axis=0)
        
        f_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in f_inf_o_stats_per_rho_sim]
        avg_f_inf_o_avg_per_rho = np.mean(np.array(f_inf_o_avg_dist_per_rho), axis=0)
        f_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in f_inf_o_stats_per_rho_sim]
        u95_f_inf_o_avg_per_rho = np.mean(np.array(f_inf_o_u95_dist_per_rho), axis=0)
        f_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in f_inf_o_stats_per_rho_sim]
        l95_f_inf_o_avg_per_rho = np.mean(np.array(f_inf_o_l95_dist_per_rho), axis=0)

        a_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in a_inf_h_stats_per_rho_sim]
        avg_a_inf_h_avg_per_rho = np.mean(np.array(a_inf_h_avg_dist_per_rho), axis=0)
        a_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in a_inf_h_stats_per_rho_sim]
        u95_a_inf_h_avg_per_rho = np.mean(np.array(a_inf_h_u95_dist_per_rho), axis=0)
        a_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in a_inf_h_stats_per_rho_sim]
        l95_a_inf_h_avg_per_rho = np.mean(np.array(a_inf_h_l95_dist_per_rho), axis=0)

        a_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in a_inf_o_stats_per_rho_sim]
        avg_a_inf_o_avg_per_rho = np.mean(np.array(a_inf_o_avg_dist_per_rho), axis=0)
        a_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in a_inf_o_stats_per_rho_sim]
        u95_a_inf_o_avg_per_rho = np.mean(np.array(a_inf_o_u95_dist_per_rho), axis=0)
        a_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in a_inf_o_stats_per_rho_sim]
        l95_a_inf_o_avg_per_rho = np.mean(np.array(a_inf_o_l95_dist_per_rho), axis=0)

    if f_inf_flag:
        f_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_h_dist_per_rho_sim) if i not in failed_outbreaks]
        nbins = len(f_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
        f_inf_h_dist_per_rho = [[] for _ in range(nbins)]

        for sim_idx in range(len(f_inf_h_dist_per_rho_sim)):
            for rho_idx in range(nbins):
                f_inf_h_values = f_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                f_inf_h_dist_per_rho[rho_idx].extend(f_inf_h_values)
        
        f_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_h_dist_per_rho])
        
        f_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_h_dist_per_rho])
        
        z = 1.96
        moe = z * (f_inf_h_std_per_rho / np.sqrt(len(f_inf_h_avg_per_rho)))
        f_inf_h_u95_per_rho = f_inf_h_avg_per_rho + moe
        f_inf_h_l95_per_rho = f_inf_h_avg_per_rho - moe
        
        f_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_o_dist_per_rho_sim) if i not in failed_outbreaks]
        nbins = len(f_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
        f_inf_o_dist_per_rho = [[] for _ in range(nbins)]

        for sim_idx in range(len(f_inf_o_dist_per_rho_sim)):
            for rho_idx in range(nbins):
                f_inf_o_values = f_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                f_inf_o_dist_per_rho[rho_idx].extend(f_inf_o_values)
        
        f_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_o_dist_per_rho])
        
        f_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_o_dist_per_rho])

        moe = z * (f_inf_o_std_per_rho / np.sqrt(len(f_inf_o_avg_per_rho)))
        f_inf_o_u95_per_rho = f_inf_o_avg_per_rho + moe
        f_inf_o_l95_per_rho = f_inf_o_avg_per_rho - moe

        flattened_list = [num for sublist1 in f_inf_h_dist_per_rho_sim for sublist2 in sublist1 for num in sublist2]
        f_inf_h_avg_global = np.nanmean(flattened_list)
        f_inf_h_std_global = np.nanstd(flattened_list)
        moe = z * f_inf_h_std_global / np.sqrt(len(flattened_list))
        f_inf_h_u95_global = f_inf_h_avg_global + moe
        f_inf_h_l95_global = f_inf_h_avg_global - moe

        flattened_list = [num for sublist1 in f_inf_o_dist_per_rho_sim for sublist2 in sublist1 for num in sublist2]
        f_inf_o_avg_global = np.nanmean(flattened_list)
        f_inf_o_std_global = np.nanstd(flattened_list)
        moe = z * f_inf_o_std_global / np.sqrt(len(flattened_list))
        f_inf_o_u95_global = f_inf_o_avg_global + moe
        f_inf_o_l95_global = f_inf_o_avg_global - moe

    if a_inf_flag:
        a_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(a_inf_h_dist_per_rho_sim) if i not in failed_outbreaks]
        nbins = len(f_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
        a_inf_h_dist_per_rho = [[] for _ in range(nbins)]
    
        for sim_idx in range(len(a_inf_h_dist_per_rho_sim)):
            for rho_idx in range(nbins):
                a_inf_h_values = a_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                a_inf_h_dist_per_rho[rho_idx].extend(a_inf_h_values)

        a_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in a_inf_h_dist_per_rho])
        a_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in a_inf_h_dist_per_rho])
        
        z = 1.96
        moe = z * (a_inf_h_std_per_rho / np.sqrt(len(a_inf_h_avg_per_rho)))
        a_inf_h_u95_per_rho = a_inf_h_avg_per_rho + moe
        a_inf_h_l95_per_rho = a_inf_h_avg_per_rho - moe
        
        a_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(a_inf_o_dist_per_rho_sim) if i not in failed_outbreaks]
        nbins = len(a_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
        a_inf_o_dist_per_rho = [[] for _ in range(nbins)]

        for sim_idx in range(len(a_inf_o_dist_per_rho_sim)):
            for rho_idx in range(nbins):
                a_inf_o_values = a_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                a_inf_o_dist_per_rho[rho_idx].extend(a_inf_o_values)
        
        a_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in a_inf_o_dist_per_rho])
        a_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in a_inf_o_dist_per_rho])
        
        z = 1.96
        moe = z * (a_inf_o_std_per_rho / np.sqrt(len(a_inf_o_avg_per_rho)))
        a_inf_o_u95_per_rho = a_inf_o_avg_per_rho + moe
        a_inf_o_l95_per_rho = a_inf_o_avg_per_rho - moe

        flattened_list = [num for sublist1 in a_inf_h_dist_per_rho_sim for sublist2 in sublist1 for num in sublist2]
        a_inf_h_avg_global = np.nanmean(flattened_list)
        a_inf_h_std_global = np.nanstd(flattened_list)
        moe = z * a_inf_h_std_global / np.sqrt(len(flattened_list))
        a_inf_h_u95_global = a_inf_h_avg_global + moe
        a_inf_h_l95_global = a_inf_h_avg_global - moe

        flattened_list = [num for sublist1 in a_inf_o_dist_per_rho_sim for sublist2 in sublist1 for num in sublist2]
        a_inf_o_avg_global = np.nanmean(flattened_list)
        a_inf_o_std_global = np.nanstd(flattened_list)
        moe = z * a_inf_o_std_global / np.sqrt(len(flattened_list))
        a_inf_o_u95_global = a_inf_o_avg_global + moe
        a_inf_o_l95_global = a_inf_o_avg_global - moe

    infected_fraction_per_rho_sim = infected_per_rho_sim / agents_per_rho_sim
    infected_h_fraction_per_rho_sim = infected_h_per_rho_sim / infected_per_rho_sim
    infected_o_fraction_per_rho_sim = infected_o_per_rho_sim / infected_per_rho_sim

    total = np.sum(infected_h_per_rho_sim) + np.sum(infected_o_per_rho_sim)
    home_global_fraction = np.sum(infected_h_per_rho_sim) / total
    out_global_fraction = np.sum(infected_o_per_rho_sim) / total
    
    infected_h_fraction_avg_per_rho = np.mean(infected_h_fraction_per_rho_sim, axis=0)
    inf_h_frac_std_per_rho = np.std(infected_h_fraction_per_rho_sim, axis=0)
    z = 1.96
    moe = z * (inf_h_frac_std_per_rho / np.sqrt(len(infected_h_fraction_per_rho_sim)))
    infected_h_fraction_u95_per_rho = infected_h_fraction_avg_per_rho + moe
    infected_h_fraction_l95_per_rho = infected_h_fraction_avg_per_rho - moe
    
    infected_o_fraction_avg_per_rho = np.mean(infected_o_fraction_per_rho_sim, axis=0)
    inf_o_frac_std_per_rho = np.std(infected_o_fraction_per_rho_sim, axis=0)
    moe = z * (inf_o_frac_std_per_rho / np.sqrt(len(infected_o_fraction_per_rho_sim)))
    infected_o_fraction_u95_per_rho = infected_o_fraction_avg_per_rho + moe
    infected_o_fraction_l95_per_rho = infected_o_fraction_avg_per_rho - moe

    output = {}

    output['inf_h_frac_avg_per_rho'] = infected_h_fraction_avg_per_rho
    output['inf_h_frac_l95_per_rho'] = infected_h_fraction_l95_per_rho
    output['inf_h_frac_u95_per_rho'] = infected_h_fraction_u95_per_rho
    output['inf_o_frac_avg_per_rho'] = infected_o_fraction_avg_per_rho
    output['inf_o_frac_l95_per_rho'] = infected_o_fraction_l95_per_rho
    output['inf_o_frac_u95_per_rho'] = infected_o_fraction_u95_per_rho
    output['inf_h_frac_global'] = home_global_fraction
    output['inf_o_frac_global'] = out_global_fraction

    if f_inf_flag:
        output['f_inf_h_avg_per_rho'] = f_inf_h_avg_per_rho
        output['f_inf_h_l95_per_rho'] = f_inf_h_l95_per_rho
        output['f_inf_h_u95_per_rho'] = f_inf_h_u95_per_rho
        output['f_inf_h_avg_global'] = f_inf_h_avg_global
        output['f_inf_h_l95_global'] = f_inf_h_l95_global
        output['f_inf_h_u95_global'] = f_inf_h_u95_global
        output['f_inf_o_avg_per_rho'] = f_inf_o_avg_per_rho
        output['f_inf_o_l95_per_rho'] = f_inf_o_l95_per_rho
        output['f_inf_o_u95_per_rho'] = f_inf_o_u95_per_rho
        output['f_inf_o_avg_global'] = f_inf_o_avg_global
        output['f_inf_o_l95_global'] = f_inf_o_l95_global
        output['f_inf_o_u95_global'] = f_inf_o_u95_global

    if a_inf_flag:
        output['a_inf_h_avg_per_rho'] = a_inf_h_avg_per_rho
        output['a_inf_h_l95_per_rho'] = a_inf_h_l95_per_rho
        output['a_inf_h_u95_per_rho'] = a_inf_h_u95_per_rho
        output['a_inf_h_avg_global'] = a_inf_h_avg_global
        output['a_inf_h_l95_global'] = a_inf_h_l95_global
        output['a_inf_h_u95_global'] = a_inf_h_u95_global
        output['a_inf_o_avg_per_rho'] = a_inf_o_avg_per_rho
        output['a_inf_o_l95_per_rho'] = a_inf_o_l95_per_rho
        output['a_inf_o_u95_per_rho'] = a_inf_o_u95_per_rho
        output['a_inf_o_avg_global'] = a_inf_o_avg_global
        output['a_inf_o_l95_global'] = a_inf_o_l95_global
        output['a_inf_o_u95_global'] = a_inf_o_u95_global

    return output

def compute_depr_chapter_panel6old_stats(
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        events_hh_per_rho_sim=None,
        events_ho_per_rho_sim=None,
        events_oh_per_rho_sim=None,
        events_oo_per_rho_sim=None,
        prevalence_cutoff=0.025, 
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    events_hh_per_rho_sim = np.array(events_hh_per_rho_sim)
    events_ho_per_rho_sim = np.array(events_ho_per_rho_sim)
    events_oh_per_rho_sim = np.array(events_oh_per_rho_sim)
    events_oo_per_rho_sim = np.array(events_oo_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    events_hh_per_rho_sim = np.delete(events_hh_per_rho_sim, failed_outbreaks, axis=0)
    events_ho_per_rho_sim = np.delete(events_ho_per_rho_sim, failed_outbreaks, axis=0)
    events_oh_per_rho_sim = np.delete(events_oh_per_rho_sim, failed_outbreaks, axis=0) 
    events_oo_per_rho_sim = np.delete(events_oo_per_rho_sim, failed_outbreaks, axis=0)

    hh_events_per_sim = np.sum(events_hh_per_rho_sim, axis=1)
    ho_events_per_sim = np.sum(events_ho_per_rho_sim, axis=1)
    oh_events_per_sim = np.sum(events_oh_per_rho_sim, axis=1)
    oo_events_per_sim = np.sum(events_oo_per_rho_sim, axis=1)
    
    total_events_sim = hh_events_per_sim + ho_events_per_sim + oh_events_per_sim + oo_events_per_sim
    hh_events_per_sim = hh_events_per_sim / total_events_sim
    ho_events_per_sim = ho_events_per_sim / total_events_sim
    oh_events_per_sim = oh_events_per_sim / total_events_sim
    oo_events_per_sim = oo_events_per_sim / total_events_sim

    hh_avg_global = np.nanmean(hh_events_per_sim)
    hh_std_global = np.nanstd(hh_events_per_sim)
    z = 1.96
    moe = z * hh_std_global / np.sqrt(len(hh_events_per_sim))
    hh_u95_global = hh_avg_global + moe
    hh_l95_global = hh_avg_global - moe

    ho_avg_global = np.nanmean(ho_events_per_sim)
    ho_std_global = np.nanstd(ho_events_per_sim)
    z = 1.96
    moe = z * ho_std_global / np.sqrt(len(ho_events_per_sim))
    ho_u95_global = ho_avg_global + moe
    ho_l95_global = ho_avg_global - moe

    oh_avg_global = np.nanmean(oh_events_per_sim)
    oh_std_global = np.nanstd(oh_events_per_sim)
    z = 1.96
    moe = z * oh_std_global / np.sqrt(len(oh_events_per_sim))
    oh_u95_global = oh_avg_global + moe
    oh_l95_global = oh_avg_global - moe

    oo_avg_global = np.nanmean(oo_events_per_sim)
    oo_std_global = np.nanstd(oo_events_per_sim)
    z = 1.96
    moe = z * oo_std_global / np.sqrt(len(oo_events_per_sim))
    oo_u95_global = oo_avg_global + moe
    oo_l95_global = oo_avg_global - moe

    normalized_data = []

    for sim in range(len(events_hh_per_rho_sim)):
        normalized_sim_data = []
        for group in range(len(events_hh_per_rho_sim[sim])):
            total_events = (
                events_hh_per_rho_sim[sim][group] +
                events_ho_per_rho_sim[sim][group] +
                events_oh_per_rho_sim[sim][group] +
                events_oo_per_rho_sim[sim][group]
            )

            #if total_events != 0:
            normalized_hh = events_hh_per_rho_sim[sim][group] / total_events
            normalized_ho = events_ho_per_rho_sim[sim][group] / total_events
            normalized_oh = events_oh_per_rho_sim[sim][group] / total_events
            normalized_oo = events_oo_per_rho_sim[sim][group] / total_events
            #else:
            #    normalized_hh = 0
            #    normalized_ho = 0
            #    normalized_oh = 0
            #    normalized_oo = 0
            
            normalized_sim_data.append([normalized_hh, normalized_ho, normalized_oh, normalized_oo])

        normalized_data.append(normalized_sim_data)
    
    # Convert new_normalized_data to a NumPy array
    normalized_data = np.array(normalized_data)

    events_hh_avg_per_rho = np.nanmean(normalized_data[:, :, 0], axis=0)
    events_ho_avg_per_rho = np.nanmean(normalized_data[:, :, 1], axis=0)
    events_oh_avg_per_rho = np.nanmean(normalized_data[:, :, 2], axis=0)
    events_oo_avg_per_rho = np.nanmean(normalized_data[:, :, 3], axis=0)
    
    events_hh_std_per_rho = np.nanstd(normalized_data[:, :, 0], axis=0)
    events_ho_std_per_rho = np.nanstd(normalized_data[:, :, 1], axis=0)
    events_oh_std_per_rho = np.nanstd(normalized_data[:, :, 2], axis=0)
    events_oo_std_per_rho = np.nanstd(normalized_data[:, :, 3], axis=0)

    z = 1.96
    nsims = len(normalized_data)
    events_hh_u95_per_rho = events_hh_avg_per_rho + z * events_hh_std_per_rho / np.sqrt(nsims)
    events_hh_l95_per_rho = events_hh_avg_per_rho - z * events_hh_std_per_rho / np.sqrt(nsims)
    events_ho_u95_per_rho = events_ho_avg_per_rho + z * events_ho_std_per_rho / np.sqrt(nsims)
    events_ho_l95_per_rho = events_ho_avg_per_rho - z * events_ho_std_per_rho / np.sqrt(nsims)
    events_oh_u95_per_rho = events_oh_avg_per_rho + z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oh_l95_per_rho = events_oh_avg_per_rho - z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oo_u95_per_rho = events_oo_avg_per_rho + z * events_oo_std_per_rho / np.sqrt(nsims)
    events_oo_l95_per_rho = events_oo_avg_per_rho - z * events_oo_std_per_rho / np.sqrt(nsims)

    output = {}
    output['hh_avg_per_rho'] = events_hh_avg_per_rho
    output['hh_l95_per_rho'] = events_hh_l95_per_rho
    output['hh_u95_per_rho'] = events_hh_u95_per_rho
    output['ho_avg_per_rho'] = events_ho_avg_per_rho
    output['ho_l95_per_rho'] = events_ho_l95_per_rho
    output['ho_u95_per_rho'] = events_ho_u95_per_rho
    output['oh_avg_per_rho'] = events_oh_avg_per_rho
    output['oh_l95_per_rho'] = events_oh_l95_per_rho
    output['oh_u95_per_rho'] = events_oh_u95_per_rho
    output['oo_avg_per_rho'] = events_oo_avg_per_rho
    output['oo_l95_per_rho'] = events_oo_l95_per_rho
    output['oo_u95_per_rho'] = events_oo_u95_per_rho

    output['hh_avg_global'] = hh_avg_global
    output['hh_l95_global'] = hh_l95_global
    output['hh_u95_global'] = hh_u95_global
    output['ho_avg_global'] = ho_avg_global
    output['ho_l95_global'] = ho_l95_global
    output['ho_u95_global'] = ho_u95_global
    output['oh_avg_global'] = oh_avg_global
    output['oh_l95_global'] = oh_l95_global
    output['oh_u95_global'] = oh_u95_global
    output['oo_avg_global'] = oo_avg_global
    output['oo_l95_global'] = oo_l95_global
    output['oo_u95_global'] = oo_u95_global

    return output

def compute_depr_chapter_panel7_stats_old(
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        infected_h_per_rho_sim=None,
        infected_o_per_rho_sim=None,
        sum_avg_a_h_per_rho_sim=None,
        sum_avg_a_o_per_rho_sim=None,
        sum_avg_foi_per_rho_sim=None,
        sum_avg_pc_foi_per_rho_sim=None,
        sum_avg_shared_per_rho_sim=None,
        sum_avg_size_per_rho_sim=None,
        sum_avg_t_pop_per_rho_sim=None,
        sum_cum_i_pop_per_rho_sim=None,
        sum_cum_shared_per_rho_sim=None,
        sum_cum_size_per_rho_sim=None,
        sum_cum_t_pop_per_rho_sim=None,
        prevalence_cutoff=0.025, 
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    infected_h_per_rho_sim = np.array(infected_h_per_rho_sim)
    infected_o_per_rho_sim = np.array(infected_o_per_rho_sim)
    sum_avg_a_h_per_rho_sim = np.array(sum_avg_a_h_per_rho_sim)
    sum_avg_a_o_per_rho_sim = np.array(sum_avg_a_o_per_rho_sim)
    sum_avg_foi_per_rho_sim = np.array(sum_avg_foi_per_rho_sim)
    sum_avg_pc_foi_per_rho_sim = np.array(sum_avg_pc_foi_per_rho_sim)
    sum_avg_shared_per_rho_sim = np.array(sum_avg_shared_per_rho_sim)
    sum_avg_size_per_rho_sim = np.array(sum_avg_size_per_rho_sim)
    sum_avg_t_pop_per_rho_sim = np.array(sum_avg_t_pop_per_rho_sim)
    sum_cum_i_pop_per_rho_sim = np.array(sum_cum_i_pop_per_rho_sim)
    sum_cum_shared_per_rho_sim = np.array(sum_cum_shared_per_rho_sim)
    sum_cum_size_per_rho_sim = np.array(sum_cum_size_per_rho_sim)
    sum_cum_t_pop_per_rho_sim = np.array(sum_cum_t_pop_per_rho_sim)
    
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    infected_h_per_rho_sim = np.delete(infected_h_per_rho_sim, failed_outbreaks, axis=0)
    infected_o_per_rho_sim = np.delete(infected_o_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_a_h_per_rho_sim = np.delete(sum_avg_a_h_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_a_o_per_rho_sim = np.delete(sum_avg_a_o_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_foi_per_rho_sim = np.delete(sum_avg_foi_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_pc_foi_per_rho_sim = np.delete(sum_avg_pc_foi_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_shared_per_rho_sim = np.delete(sum_avg_shared_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_size_per_rho_sim = np.delete(sum_avg_size_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_t_pop_per_rho_sim = np.delete(sum_avg_t_pop_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_i_pop_per_rho_sim = np.delete(sum_cum_i_pop_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_shared_per_rho_sim = np.delete(sum_cum_shared_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_size_per_rho_sim = np.delete(sum_cum_size_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_t_pop_per_rho_sim = np.delete(sum_cum_t_pop_per_rho_sim, failed_outbreaks, axis=0)

    agents_sim = np.sum(agents_per_rho_sim, axis=1)
    infected_sim = np.sum(infected_per_rho_sim, axis=1)
    infected_h_sim = np.sum(infected_h_per_rho_sim, axis=1)
    infected_o_sim = np.sum(infected_o_per_rho_sim, axis=1)
    fra_avg_a_h_sim = np.sum(sum_avg_a_h_per_rho_sim, axis=1) / infected_sim
    fra_avg_a_o_sim = np.sum(sum_avg_a_o_per_rho_sim, axis=1) / infected_sim
    fra_avg_foi_sim = np.sum(sum_avg_foi_per_rho_sim, axis=1) / infected_sim
    fra_avg_pc_foi_sim = np.sum(sum_avg_pc_foi_per_rho_sim, axis=1) / infected_sim
    fra_avg_shared_sim = np.sum(sum_avg_shared_per_rho_sim, axis=1) / infected_sim
    fra_avg_size_sim = np.sum(sum_avg_size_per_rho_sim, axis=1) / infected_sim
    fra_avg_t_pop_sim = np.sum(sum_avg_t_pop_per_rho_sim, axis=1) / infected_sim
    fra_cum_i_pop_sim = np.sum(sum_cum_i_pop_per_rho_sim, axis=1) / infected_sim
    fra_cum_shared_sim = np.sum(sum_cum_shared_per_rho_sim, axis=1) / infected_sim
    fra_cum_size_sim = np.sum(sum_cum_size_per_rho_sim, axis=1) / infected_sim
    fra_cum_t_pop_sim = np.sum(sum_cum_t_pop_per_rho_sim, axis=1) / infected_sim

    fra_avg_a_h_per_rho_sim = sum_avg_a_h_per_rho_sim / infected_h_per_rho_sim
    fra_avg_a_o_per_rho_sim = sum_avg_a_o_per_rho_sim / infected_o_per_rho_sim
    fra_avg_a_per_rho_sim = (sum_avg_a_h_per_rho_sim + sum_avg_a_o_per_rho_sim) / infected_per_rho_sim
    fra_avg_foi_per_rho_sim = sum_avg_foi_per_rho_sim / infected_per_rho_sim
    fra_avg_pc_foi_per_rho_sim = sum_avg_pc_foi_per_rho_sim / infected_per_rho_sim
    fra_avg_shared_per_rho_sim = sum_avg_shared_per_rho_sim / infected_per_rho_sim
    fra_avg_size_per_rho_sim = sum_avg_size_per_rho_sim / infected_per_rho_sim
    fra_avg_t_pop_per_rho_sim = sum_avg_t_pop_per_rho_sim / infected_per_rho_sim
    fra_cum_shared_per_rho_sim = sum_cum_shared_per_rho_sim / infected_per_rho_sim
    fra_cum_size_per_rho_sim = sum_cum_size_per_rho_sim / infected_per_rho_sim
    fra_cum_t_pop_per_rho_sim = sum_cum_t_pop_per_rho_sim / infected_per_rho_sim

    output = {}

    z = 1.96
    nsims = len(fra_avg_a_h_per_rho_sim)

    fra_avg_a_h_avg_per_rho = np.mean(fra_avg_a_h_per_rho_sim, axis=0)
    fra_avg_a_h_std_per_rho = np.std(fra_avg_a_h_avg_per_rho, axis=0)
    fra_avg_a_h_l95_per_rho = fra_avg_a_h_avg_per_rho - z * fra_avg_a_h_std_per_rho / np.sqrt(nsims)
    fra_avg_a_h_u95_per_rho = fra_avg_a_h_avg_per_rho + z * fra_avg_a_h_std_per_rho / np.sqrt(nsims)

    output['fra_avg_a_h_avg_per_rho'] = fra_avg_a_h_avg_per_rho
    output['fra_avg_a_h_l95_per_rho'] = fra_avg_a_h_l95_per_rho
    output['fra_avg_a_h_u95_per_rho'] = fra_avg_a_h_u95_per_rho

    fra_avg_a_o_avg_per_rho = np.mean(fra_avg_a_o_per_rho_sim, axis=0)
    fra_avg_a_o_std_per_rho = np.std(fra_avg_a_o_avg_per_rho, axis=0)
    fra_avg_a_o_l95_per_rho = fra_avg_a_o_avg_per_rho - z * fra_avg_a_o_std_per_rho / np.sqrt(nsims)
    fra_avg_a_o_u95_per_rho = fra_avg_a_o_avg_per_rho + z * fra_avg_a_o_std_per_rho / np.sqrt(nsims)

    output['fra_avg_a_o_avg_per_rho'] = fra_avg_a_o_avg_per_rho
    output['fra_avg_a_o_l95_per_rho'] = fra_avg_a_o_l95_per_rho
    output['fra_avg_a_o_u95_per_rho'] = fra_avg_a_o_u95_per_rho
    
    fra_avg_a_avg_per_rho = np.mean(fra_avg_a_per_rho_sim, axis=0)
    fra_avg_a_std_per_rho = np.std(fra_avg_a_avg_per_rho, axis=0)
    fra_avg_a_l95_per_rho = fra_avg_a_avg_per_rho - z * fra_avg_a_std_per_rho / np.sqrt(nsims)
    fra_avg_a_u95_per_rho = fra_avg_a_avg_per_rho + z * fra_avg_a_std_per_rho / np.sqrt(nsims)

    output['fra_avg_a_avg_per_rho'] = fra_avg_a_avg_per_rho
    output['fra_avg_a_l95_per_rho'] = fra_avg_a_l95_per_rho
    output['fra_avg_a_u95_per_rho'] = fra_avg_a_u95_per_rho

    fra_avg_foi_avg_per_rho = np.mean(fra_avg_foi_per_rho_sim, axis=0)
    fra_avg_foi_std_per_rho = np.std(fra_avg_foi_avg_per_rho, axis=0)
    fra_avg_foi_l95_per_rho = fra_avg_foi_avg_per_rho - z * fra_avg_foi_std_per_rho / np.sqrt(nsims)
    fra_avg_foi_u95_per_rho = fra_avg_foi_avg_per_rho + z * fra_avg_foi_std_per_rho / np.sqrt(nsims)

    output['fra_avg_foi_avg_per_rho'] = fra_avg_foi_avg_per_rho
    output['fra_avg_foi_l95_per_rho'] = fra_avg_foi_l95_per_rho
    output['fra_avg_foi_u95_per_rho'] = fra_avg_foi_u95_per_rho

    fra_avg_pc_foi_avg_per_rho = np.mean(fra_avg_pc_foi_per_rho_sim, axis=0)
    fra_avg_pc_foi_std_per_rho = np.std(fra_avg_pc_foi_avg_per_rho, axis=0)
    fra_avg_pc_foi_l95_per_rho = fra_avg_pc_foi_avg_per_rho - z * fra_avg_pc_foi_std_per_rho / np.sqrt(nsims)
    fra_avg_pc_foi_u95_per_rho = fra_avg_pc_foi_avg_per_rho + z * fra_avg_pc_foi_std_per_rho / np.sqrt(nsims)

    output['fra_avg_pc_foi_avg_per_rho'] = fra_avg_pc_foi_avg_per_rho
    output['fra_avg_pc_foi_l95_per_rho'] = fra_avg_pc_foi_l95_per_rho
    output['fra_avg_pc_foi_u95_per_rho'] = fra_avg_pc_foi_u95_per_rho

    fra_avg_shared_avg_per_rho = np.mean(fra_avg_shared_per_rho_sim, axis=0)
    fra_avg_shared_std_per_rho = np.std(fra_avg_shared_avg_per_rho, axis=0)
    fra_avg_shared_l95_per_rho = fra_avg_shared_avg_per_rho - z * fra_avg_shared_std_per_rho / np.sqrt(nsims)
    fra_avg_shared_u95_per_rho = fra_avg_shared_avg_per_rho + z * fra_avg_shared_std_per_rho / np.sqrt(nsims)

    output['fra_avg_shared_avg_per_rho'] = fra_avg_shared_avg_per_rho
    output['fra_avg_shared_l95_per_rho'] = fra_avg_shared_l95_per_rho
    output['fra_avg_shared_u95_per_rho'] = fra_avg_shared_u95_per_rho

    fra_avg_size_avg_per_rho = np.mean(fra_avg_size_per_rho_sim, axis=0)
    fra_avg_size_std_per_rho = np.std(fra_avg_size_avg_per_rho, axis=0)
    fra_avg_size_l95_per_rho = fra_avg_size_avg_per_rho - z * fra_avg_size_std_per_rho / np.sqrt(nsims)
    fra_avg_size_u95_per_rho = fra_avg_size_avg_per_rho + z * fra_avg_size_std_per_rho / np.sqrt(nsims)

    output['fra_avg_size_avg_per_rho'] = fra_avg_size_avg_per_rho
    output['fra_avg_size_l95_per_rho'] = fra_avg_size_l95_per_rho
    output['fra_avg_size_u95_per_rho'] = fra_avg_size_u95_per_rho

    fra_avg_t_pop_avg_per_rho = np.mean(fra_avg_t_pop_per_rho_sim, axis=0)
    fra_avg_t_pop_std_per_rho = np.std(fra_avg_t_pop_avg_per_rho, axis=0)
    fra_avg_t_pop_l95_per_rho = fra_avg_t_pop_avg_per_rho - z * fra_avg_t_pop_std_per_rho / np.sqrt(nsims)
    fra_avg_t_pop_u95_per_rho = fra_avg_t_pop_avg_per_rho + z * fra_avg_t_pop_std_per_rho / np.sqrt(nsims)

    output['fra_avg_t_pop_avg_per_rho'] = fra_avg_t_pop_avg_per_rho
    output['fra_avg_t_pop_l95_per_rho'] = fra_avg_t_pop_l95_per_rho
    output['fra_avg_t_pop_u95_per_rho'] = fra_avg_t_pop_u95_per_rho

    fra_cum_shared_avg_per_rho = np.mean(fra_cum_shared_per_rho_sim, axis=0)
    fra_cum_shared_std_per_rho = np.std(fra_cum_shared_avg_per_rho, axis=0)
    fra_cum_shared_l95_per_rho = fra_cum_shared_avg_per_rho - z * fra_cum_shared_std_per_rho / np.sqrt(nsims)
    fra_cum_shared_u95_per_rho = fra_cum_shared_avg_per_rho + z * fra_cum_shared_std_per_rho / np.sqrt(nsims)

    output['fra_cum_shared_avg_per_rho'] = fra_cum_shared_avg_per_rho
    output['fra_cum_shared_l95_per_rho'] = fra_cum_shared_l95_per_rho
    output['fra_cum_shared_u95_per_rho'] = fra_cum_shared_u95_per_rho

    fra_cum_size_avg_per_rho = np.mean(fra_cum_size_per_rho_sim, axis=0)
    fra_cum_size_std_per_rho = np.std(fra_cum_size_avg_per_rho, axis=0)
    fra_cum_size_l95_per_rho = fra_cum_size_avg_per_rho - z * fra_cum_size_std_per_rho / np.sqrt(nsims)
    fra_cum_size_u95_per_rho = fra_cum_size_avg_per_rho + z * fra_cum_size_std_per_rho / np.sqrt(nsims)

    output['fra_cum_size_avg_per_rho'] = fra_cum_size_avg_per_rho
    output['fra_cum_size_l95_per_rho'] = fra_cum_size_l95_per_rho
    output['fra_cum_size_u95_per_rho'] = fra_cum_size_u95_per_rho

    fra_cum_t_pop_avg_per_rho = np.mean(fra_cum_t_pop_per_rho_sim, axis=0)
    fra_cum_t_pop_std_per_rho = np.std(fra_cum_t_pop_avg_per_rho, axis=0)
    fra_cum_t_pop_l95_per_rho = fra_cum_t_pop_avg_per_rho - z * fra_cum_t_pop_std_per_rho / np.sqrt(nsims)
    fra_cum_t_pop_u95_per_rho = fra_cum_t_pop_avg_per_rho + z * fra_cum_t_pop_std_per_rho / np.sqrt(nsims)

    output['fra_cum_t_pop_avg_per_rho'] = fra_cum_t_pop_avg_per_rho
    output['fra_cum_t_pop_l95_per_rho'] = fra_cum_t_pop_l95_per_rho
    output['fra_cum_t_pop_u95_per_rho'] = fra_cum_t_pop_u95_per_rho

    fra_avg_a_h_avg = np.mean(fra_avg_a_h_sim)
    fra_avg_a_h_std = np.std(fra_avg_a_h_sim)
    fra_avg_a_h_l95 = fra_avg_a_h_avg - z * fra_avg_a_h_std / np.sqrt(nsims)
    fra_avg_a_h_u95 = fra_avg_a_h_avg - z * fra_avg_a_h_std / np.sqrt(nsims)

    output['fra_avg_a_h_pop_avg'] = fra_avg_a_h_avg
    output['fra_avg_a_h_pop_l95'] = fra_avg_a_h_l95
    output['fra_avg_a_h_pop_u95'] = fra_avg_a_h_u95
    
    fra_avg_a_o_avg = np.mean(fra_avg_a_o_sim)
    fra_avg_a_o_std = np.std(fra_avg_a_o_sim)
    fra_avg_a_o_l95 = fra_avg_a_o_avg - z * fra_avg_a_o_std / np.sqrt(nsims)
    fra_avg_a_o_u95 = fra_avg_a_o_avg - z * fra_avg_a_o_std / np.sqrt(nsims)

    output['fra_avg_a_o_avg'] = fra_avg_a_o_avg
    output['fra_avg_a_o_l95'] = fra_avg_a_o_l95
    output['fra_avg_a_o_u95'] = fra_avg_a_o_u95

    fra_avg_foi_avg = np.mean(fra_avg_foi_sim)
    fra_avg_foi_std = np.std(fra_avg_foi_sim)
    fra_avg_foi_l95 = fra_avg_foi_avg - z * fra_avg_foi_std / np.sqrt(nsims)
    fra_avg_foi_u95 = fra_avg_foi_avg - z * fra_avg_foi_std / np.sqrt(nsims)

    output['fra_avg_foi_avg'] = fra_avg_foi_avg
    output['fra_avg_foi_l95'] = fra_avg_foi_l95
    output['fra_avg_foi_u95'] = fra_avg_foi_u95

    fra_avg_pc_foi_avg = np.mean(fra_avg_pc_foi_sim)
    fra_avg_pc_foi_std = np.std(fra_avg_pc_foi_sim)
    fra_avg_pc_foi_l95 = fra_avg_pc_foi_avg - z * fra_avg_pc_foi_std / np.sqrt(nsims)
    fra_avg_pc_foi_u95 = fra_avg_pc_foi_avg - z * fra_avg_pc_foi_std / np.sqrt(nsims)

    output['fra_avg_pc_foi_avg'] = fra_avg_pc_foi_avg
    output['fra_avg_pc_foi_l95'] = fra_avg_pc_foi_l95
    output['fra_avg_pc_foi_u95'] = fra_avg_pc_foi_u95

    fra_avg_shared_avg = np.mean(fra_avg_shared_sim)
    fra_avg_shared_std = np.std(fra_avg_shared_sim)
    fra_avg_shared_l95 = fra_avg_shared_avg - z * fra_avg_shared_std / np.sqrt(nsims)
    fra_avg_shared_u95 = fra_avg_shared_avg - z * fra_avg_shared_std / np.sqrt(nsims)

    output['fra_avg_shared_avg'] = fra_avg_shared_avg
    output['fra_avg_shared_l95'] = fra_avg_shared_l95
    output['fra_avg_shared_u95'] = fra_avg_shared_u95

    fra_avg_size_avg = np.mean(fra_avg_size_sim)
    fra_avg_size_std = np.std(fra_avg_size_sim)
    fra_avg_size_l95 = fra_avg_size_avg - z * fra_avg_size_std / np.sqrt(nsims)
    fra_avg_size_u95 = fra_avg_size_avg - z * fra_avg_size_std / np.sqrt(nsims)

    output['fra_avg_size_avg'] = fra_avg_size_avg
    output['fra_avg_size_l95'] = fra_avg_size_l95
    output['fra_avg_size_u95'] = fra_avg_size_u95

    fra_cum_shared_avg = np.mean(fra_cum_shared_sim)
    fra_cum_shared_std = np.std(fra_cum_shared_sim)
    fra_cum_shared_l95 = fra_cum_shared_avg - z * fra_cum_shared_std / np.sqrt(nsims)
    fra_cum_shared_u95 = fra_cum_shared_avg - z * fra_cum_shared_std / np.sqrt(nsims)

    output['fra_cum_shared_avg'] = fra_cum_shared_avg
    output['fra_cum_shared_l95'] = fra_cum_shared_l95
    output['fra_cum_shared_u95'] = fra_cum_shared_u95

    fra_cum_size_avg = np.mean(fra_cum_size_sim)
    fra_cum_size_std = np.std(fra_cum_size_sim)
    fra_cum_size_l95 = fra_cum_size_avg - z * fra_cum_size_std / np.sqrt(nsims)
    fra_cum_size_u95 = fra_cum_size_avg - z * fra_cum_size_std / np.sqrt(nsims)

    output['fra_cum_size_avg'] = fra_cum_size_avg
    output['fra_cum_size_l95'] = fra_cum_size_l95
    output['fra_cum_size_u95'] = fra_cum_size_u95

    fra_cum_t_pop_avg = np.mean(fra_cum_t_pop_sim)
    fra_cum_t_pop_std = np.std(fra_cum_t_pop_sim)
    fra_cum_t_pop_l95 = fra_cum_t_pop_avg - z * fra_cum_t_pop_std / np.sqrt(nsims)
    fra_cum_t_pop_u95 = fra_cum_t_pop_avg - z * fra_cum_t_pop_std / np.sqrt(nsims)

    output['fra_cum_t_pop_avg'] = fra_cum_t_pop_avg
    output['fra_cum_t_pop_l95'] = fra_cum_t_pop_l95
    output['fra_cum_t_pop_u95'] = fra_cum_t_pop_u95

    return output

def compute_depr_chapter_panel7_stats(
        agents_per_rho_sim,
        avg_a_h_dist_per_rho_sim,
        avg_a_o_dist_per_rho_sim,
        events_hh_per_rho_sim,
        events_ho_per_rho_sim,
        events_oh_per_rho_sim,
        events_oo_per_rho_sim,
        f_inf_tr_h_dist_per_rho_sim,
        f_inf_tr_o_dist_per_rho_sim,
        infected_per_rho_sim,
        infected_h_per_rho_sim,
        infected_o_per_rho_sim,
        prevalence_cutoff=0.025, 
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    events_hh_per_rho_sim = np.array(events_hh_per_rho_sim)
    events_ho_per_rho_sim = np.array(events_ho_per_rho_sim)
    events_oh_per_rho_sim = np.array(events_oh_per_rho_sim)
    events_oo_per_rho_sim = np.array(events_oo_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    infected_h_per_rho_sim = np.array(infected_h_per_rho_sim)
    infected_o_per_rho_sim = np.array(infected_o_per_rho_sim)
 
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    events_hh_per_rho_sim = np.delete(events_hh_per_rho_sim, failed_outbreaks, axis=0)
    events_ho_per_rho_sim = np.delete(events_ho_per_rho_sim, failed_outbreaks, axis=0)
    events_oh_per_rho_sim = np.delete(events_oh_per_rho_sim, failed_outbreaks, axis=0) 
    events_oo_per_rho_sim = np.delete(events_oo_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    infected_h_per_rho_sim = np.delete(infected_h_per_rho_sim, failed_outbreaks, axis=0)
    infected_o_per_rho_sim = np.delete(infected_o_per_rho_sim, failed_outbreaks, axis=0)

    hh_events_per_sim = np.sum(events_hh_per_rho_sim, axis=1)
    ho_events_per_sim = np.sum(events_ho_per_rho_sim, axis=1)
    oh_events_per_sim = np.sum(events_oh_per_rho_sim, axis=1)
    oo_events_per_sim = np.sum(events_oo_per_rho_sim, axis=1)
    
    total_events_sim = hh_events_per_sim + ho_events_per_sim + oh_events_per_sim + oo_events_per_sim
    hh_events_per_sim = hh_events_per_sim / total_events_sim
    ho_events_per_sim = ho_events_per_sim / total_events_sim
    oh_events_per_sim = oh_events_per_sim / total_events_sim
    oo_events_per_sim = oo_events_per_sim / total_events_sim

    output = {}

    z = 1.96

    normalized_data = []

    for sim in range(len(events_hh_per_rho_sim)):
        normalized_sim_data = []
        for group in range(len(events_hh_per_rho_sim[sim])):
            total_events = (
                events_hh_per_rho_sim[sim][group] +
                events_ho_per_rho_sim[sim][group] +
                events_oh_per_rho_sim[sim][group] +
                events_oo_per_rho_sim[sim][group]
            )

            #if total_events != 0:
            normalized_hh = events_hh_per_rho_sim[sim][group] / total_events
            normalized_ho = events_ho_per_rho_sim[sim][group] / total_events
            normalized_oh = events_oh_per_rho_sim[sim][group] / total_events
            normalized_oo = events_oo_per_rho_sim[sim][group] / total_events
            #else:
            #    normalized_hh = 0
            #    normalized_ho = 0
            #    normalized_oh = 0
            #    normalized_oo = 0
            
            normalized_sim_data.append([normalized_hh, normalized_ho, normalized_oh, normalized_oo])

        normalized_data.append(normalized_sim_data)
    
    # Convert new_normalized_data to a NumPy array
    normalized_data = np.array(normalized_data)

    nsims = len(normalized_data)

    events_hh_avg_per_rho = np.nanmean(normalized_data[:, :, 0], axis=0)
    events_hh_std_per_rho = np.nanstd(normalized_data[:, :, 0], axis=0)
    events_hh_u95_per_rho = events_hh_avg_per_rho + z * events_hh_std_per_rho / np.sqrt(nsims)
    events_hh_l95_per_rho = events_hh_avg_per_rho - z * events_hh_std_per_rho / np.sqrt(nsims)
    
    output['hh_avg_per_rho'] = events_hh_avg_per_rho
    output['hh_l95_per_rho'] = events_hh_l95_per_rho
    output['hh_u95_per_rho'] = events_hh_u95_per_rho
    
    events_ho_avg_per_rho = np.nanmean(normalized_data[:, :, 1], axis=0)
    events_ho_std_per_rho = np.nanstd(normalized_data[:, :, 1], axis=0)
    events_ho_u95_per_rho = events_ho_avg_per_rho + z * events_ho_std_per_rho / np.sqrt(nsims)
    events_ho_l95_per_rho = events_ho_avg_per_rho - z * events_ho_std_per_rho / np.sqrt(nsims)

    output['ho_avg_per_rho'] = events_ho_avg_per_rho
    output['ho_l95_per_rho'] = events_ho_l95_per_rho
    output['ho_u95_per_rho'] = events_ho_u95_per_rho
    
    events_oh_avg_per_rho = np.nanmean(normalized_data[:, :, 2], axis=0)
    events_oh_std_per_rho = np.nanstd(normalized_data[:, :, 2], axis=0)
    events_oh_u95_per_rho = events_oh_avg_per_rho + z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oh_l95_per_rho = events_oh_avg_per_rho - z * events_oh_std_per_rho / np.sqrt(nsims)

    output['oh_avg_per_rho'] = events_oh_avg_per_rho
    output['oh_l95_per_rho'] = events_oh_l95_per_rho
    output['oh_u95_per_rho'] = events_oh_u95_per_rho
    
    events_oo_avg_per_rho = np.nanmean(normalized_data[:, :, 3], axis=0)
    events_oo_std_per_rho = np.nanstd(normalized_data[:, :, 3], axis=0)
    events_oo_u95_per_rho = events_oo_avg_per_rho + z * events_oo_std_per_rho / np.sqrt(nsims)
    events_oo_l95_per_rho = events_oo_avg_per_rho - z * events_oo_std_per_rho / np.sqrt(nsims)

    output['oo_avg_per_rho'] = events_oo_avg_per_rho
    output['oo_l95_per_rho'] = events_oo_l95_per_rho
    output['oo_u95_per_rho'] = events_oo_u95_per_rho

    agents_sim = np.sum(agents_per_rho_sim, axis=1)
    infected_sim = np.sum(infected_per_rho_sim, axis=1)
    infected_h_sim = np.sum(infected_h_per_rho_sim, axis=1)
    infected_o_sim = np.sum(infected_o_per_rho_sim, axis=1)
   
    avg_a_h_dist_per_rho_sim = [sim for i, sim in enumerate(avg_a_h_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(avg_a_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    avg_a_h_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(avg_a_h_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            avg_a_h_values = avg_a_h_dist_per_rho_sim[sim_idx][rho_idx]
            avg_a_h_dist_per_rho[rho_idx].extend(avg_a_h_values)

    avg_a_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in avg_a_h_dist_per_rho])
    avg_a_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in avg_a_h_dist_per_rho])
    moe = z * (avg_a_h_std_per_rho / np.sqrt(len(avg_a_h_avg_per_rho)))
    avg_a_h_u95_per_rho = avg_a_h_avg_per_rho + moe
    avg_a_h_l95_per_rho = avg_a_h_avg_per_rho - moe

    output['avg_a_h_dist_per_rho'] = avg_a_h_dist_per_rho
    output['avg_a_h_avg_per_rho'] = avg_a_h_avg_per_rho
    output['avg_a_h_l95_per_rho'] = avg_a_h_l95_per_rho
    output['avg_a_h_u95_per_rho'] = avg_a_h_u95_per_rho

    avg_a_o_dist_per_rho_sim = [sim for i, sim in enumerate(avg_a_o_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(avg_a_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    avg_a_o_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(avg_a_o_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            avg_a_o_values = avg_a_o_dist_per_rho_sim[sim_idx][rho_idx]
            avg_a_o_dist_per_rho[rho_idx].extend(avg_a_o_values)

    avg_a_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in avg_a_o_dist_per_rho])
    avg_a_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in avg_a_o_dist_per_rho])
    moe = z * (avg_a_o_std_per_rho / np.sqrt(len(avg_a_o_avg_per_rho)))
    avg_a_o_u95_per_rho = avg_a_o_avg_per_rho + moe
    avg_a_o_l95_per_rho = avg_a_o_avg_per_rho - moe

    output['avg_a_o_dist_per_rho'] = avg_a_o_dist_per_rho
    output['avg_a_o_avg_per_rho'] = avg_a_o_avg_per_rho
    output['avg_a_o_l95_per_rho'] = avg_a_o_l95_per_rho
    output['avg_a_o_u95_per_rho'] = avg_a_o_u95_per_rho

    f_inf_tr_h_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_tr_h_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_inf_tr_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_inf_tr_h_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_inf_tr_h_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_inf_tr_h_values = f_inf_tr_h_dist_per_rho_sim[sim_idx][rho_idx]
            f_inf_tr_h_dist_per_rho[rho_idx].extend(f_inf_tr_h_values)

    f_inf_tr_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_tr_h_dist_per_rho])
    f_inf_tr_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_tr_h_dist_per_rho])
    moe = z * (f_inf_tr_h_std_per_rho / np.sqrt(len(f_inf_tr_h_avg_per_rho)))
    f_inf_tr_h_u95_per_rho = f_inf_tr_h_avg_per_rho + moe
    f_inf_tr_h_l95_per_rho = f_inf_tr_h_avg_per_rho - moe

    output['f_inf_tr_h_dist_per_rho'] = f_inf_tr_h_dist_per_rho
    output['f_inf_tr_h_avg_per_rho'] = f_inf_tr_h_avg_per_rho
    output['f_inf_tr_h_l95_per_rho'] = f_inf_tr_h_l95_per_rho
    output['f_inf_tr_h_u95_per_rho'] = f_inf_tr_h_u95_per_rho

    f_inf_tr_o_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_tr_o_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_inf_tr_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_inf_tr_o_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_inf_tr_o_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_inf_tr_o_values = f_inf_tr_o_dist_per_rho_sim[sim_idx][rho_idx]
            f_inf_tr_o_dist_per_rho[rho_idx].extend(f_inf_tr_o_values)

    f_inf_tr_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_tr_o_dist_per_rho])
    f_inf_tr_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_tr_o_dist_per_rho])
    moe = z * (f_inf_tr_o_std_per_rho / np.sqrt(len(f_inf_tr_o_avg_per_rho)))
    f_inf_tr_o_u95_per_rho = f_inf_tr_o_avg_per_rho + moe
    f_inf_tr_o_l95_per_rho = f_inf_tr_o_avg_per_rho - moe

    output['f_inf_tr_o_dist_per_rho'] = f_inf_tr_o_dist_per_rho
    output['f_inf_tr_o_avg_per_rho'] = f_inf_tr_o_avg_per_rho
    output['f_inf_tr_o_l95_per_rho'] = f_inf_tr_o_l95_per_rho
    output['f_inf_tr_o_u95_per_rho'] = f_inf_tr_o_u95_per_rho

    nsims = len(agents_per_rho_sim)

    agents_avg_per_rho = np.mean(agents_per_rho_sim, axis=0)
    agents_std_per_rho = np.std(agents_avg_per_rho, axis=0)
    agents_l95_per_rho = agents_avg_per_rho - z * agents_std_per_rho / np.sqrt(nsims)
    agents_u95_per_rho = agents_avg_per_rho + z * agents_std_per_rho / np.sqrt(nsims)

    output['agents_avg_per_rho'] = agents_avg_per_rho
    output['agents_l95_per_rho'] = agents_l95_per_rho
    output['agents_u95_per_rho'] = agents_u95_per_rho

    infected_avg_per_rho = np.mean(infected_per_rho_sim, axis=0)
    infected_std_per_rho = np.std(infected_per_rho_sim, axis=0)
    infected_l95_per_rho = infected_avg_per_rho - z * infected_std_per_rho / np.sqrt(nsims)
    infected_u95_per_rho = infected_avg_per_rho + z * infected_std_per_rho / np.sqrt(nsims)

    output['infected_avg_per_rho'] = infected_avg_per_rho
    output['infected_l95_per_rho'] = infected_l95_per_rho
    output['infected_u95_per_rho'] = infected_u95_per_rho

    infected_h_avg_per_rho = np.mean(infected_h_per_rho_sim, axis=0)
    infected_h_std_per_rho = np.std(infected_h_per_rho_sim, axis=0)
    infected_h_l95_per_rho = infected_h_avg_per_rho - z * infected_h_std_per_rho / np.sqrt(nsims)
    infected_h_u95_per_rho = infected_h_avg_per_rho + z * infected_h_std_per_rho / np.sqrt(nsims)

    output['infected_h_avg_per_rho'] = infected_h_avg_per_rho
    output['infected_h_l95_per_rho'] = infected_h_l95_per_rho
    output['infected_h_u95_per_rho'] = infected_h_u95_per_rho

    infected_o_avg_per_rho = np.mean(infected_o_per_rho_sim, axis=0)
    infected_o_std_per_rho = np.std(infected_o_per_rho_sim, axis=0)
    infected_o_l95_per_rho = infected_o_avg_per_rho - z * infected_o_std_per_rho / np.sqrt(nsims)
    infected_o_u95_per_rho = infected_o_avg_per_rho + z * infected_o_std_per_rho / np.sqrt(nsims)

    output['infected_o_avg_per_rho'] = infected_o_avg_per_rho
    output['infected_o_l95_per_rho'] = infected_o_l95_per_rho
    output['infected_o_u95_per_rho'] = infected_o_u95_per_rho

    hh_avg_global = np.nanmean(hh_events_per_sim)
    hh_std_global = np.nanstd(hh_events_per_sim)
    moe = z * hh_std_global / np.sqrt(len(hh_events_per_sim))
    hh_u95_global = hh_avg_global + moe
    hh_l95_global = hh_avg_global - moe

    output['hh_avg_global'] = hh_avg_global
    output['hh_l95_global'] = hh_l95_global
    output['hh_u95_global'] = hh_u95_global

    ho_avg_global = np.nanmean(ho_events_per_sim)
    ho_std_global = np.nanstd(ho_events_per_sim)
    moe = z * ho_std_global / np.sqrt(len(ho_events_per_sim))
    ho_u95_global = ho_avg_global + moe
    ho_l95_global = ho_avg_global - moe

    output['ho_avg_global'] = ho_avg_global
    output['ho_l95_global'] = ho_l95_global
    output['ho_u95_global'] = ho_u95_global

    oh_avg_global = np.nanmean(oh_events_per_sim)
    oh_std_global = np.nanstd(oh_events_per_sim)
    moe = z * oh_std_global / np.sqrt(len(oh_events_per_sim))
    oh_u95_global = oh_avg_global + moe
    oh_l95_global = oh_avg_global - moe

    output['oh_avg_global'] = oh_avg_global
    output['oh_l95_global'] = oh_l95_global
    output['oh_u95_global'] = oh_u95_global

    oo_avg_global = np.nanmean(oo_events_per_sim)
    oo_std_global = np.nanstd(oo_events_per_sim)
    moe = z * oo_std_global / np.sqrt(len(oo_events_per_sim))
    oo_u95_global = oo_avg_global + moe
    oo_l95_global = oo_avg_global - moe

    output['oo_avg_global'] = oo_avg_global
    output['oo_l95_global'] = oo_l95_global
    output['oo_u95_global'] = oo_u95_global

    return output

def compute_depr_chapter_panel8_stats(
        agents_per_rho_sim,
        events_hh_per_rho_sim,
        events_ho_per_rho_sim,
        events_oh_per_rho_sim,
        events_oo_per_rho_sim,
        f_inf_tr_h_dist_per_rho_sim,
        f_inf_tr_o_dist_per_rho_sim,
        infected_per_rho_sim,
        infected_h_per_rho_sim,
        infected_o_per_rho_sim,
        sum_avg_a_h_per_rho_sim,
        sum_avg_a_o_per_rho_sim,
        prevalence_cutoff=0.025, 
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    events_hh_per_rho_sim = np.array(events_hh_per_rho_sim)
    events_ho_per_rho_sim = np.array(events_ho_per_rho_sim)
    events_oh_per_rho_sim = np.array(events_oh_per_rho_sim)
    events_oo_per_rho_sim = np.array(events_oo_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    infected_h_per_rho_sim = np.array(infected_h_per_rho_sim)
    infected_o_per_rho_sim = np.array(infected_o_per_rho_sim)
    sum_avg_a_h_per_rho_sim = np.array(sum_avg_a_h_per_rho_sim)
    sum_avg_a_o_per_rho_sim = np.array(sum_avg_a_o_per_rho_sim)
    
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    events_hh_per_rho_sim = np.delete(events_hh_per_rho_sim, failed_outbreaks, axis=0)
    events_ho_per_rho_sim = np.delete(events_ho_per_rho_sim, failed_outbreaks, axis=0)
    events_oh_per_rho_sim = np.delete(events_oh_per_rho_sim, failed_outbreaks, axis=0) 
    events_oo_per_rho_sim = np.delete(events_oo_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    infected_h_per_rho_sim = np.delete(infected_h_per_rho_sim, failed_outbreaks, axis=0)
    infected_o_per_rho_sim = np.delete(infected_o_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_a_h_per_rho_sim = np.delete(sum_avg_a_h_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_a_o_per_rho_sim = np.delete(sum_avg_a_o_per_rho_sim, failed_outbreaks, axis=0)

    hh_events_per_sim = np.sum(events_hh_per_rho_sim, axis=1)
    ho_events_per_sim = np.sum(events_ho_per_rho_sim, axis=1)
    oh_events_per_sim = np.sum(events_oh_per_rho_sim, axis=1)
    oo_events_per_sim = np.sum(events_oo_per_rho_sim, axis=1)
    
    total_events_sim = hh_events_per_sim + ho_events_per_sim + oh_events_per_sim + oo_events_per_sim
    hh_events_per_sim = hh_events_per_sim / total_events_sim
    ho_events_per_sim = ho_events_per_sim / total_events_sim
    oh_events_per_sim = oh_events_per_sim / total_events_sim
    oo_events_per_sim = oo_events_per_sim / total_events_sim

    output = {}

    z = 1.96

    normalized_data = []

    for sim in range(len(events_hh_per_rho_sim)):
        normalized_sim_data = []
        for group in range(len(events_hh_per_rho_sim[sim])):
            total_events = (
                events_hh_per_rho_sim[sim][group] +
                events_ho_per_rho_sim[sim][group] +
                events_oh_per_rho_sim[sim][group] +
                events_oo_per_rho_sim[sim][group]
            )

            #if total_events != 0:
            normalized_hh = events_hh_per_rho_sim[sim][group] / 1.0 #total_events
            normalized_ho = events_ho_per_rho_sim[sim][group] / 1.0 #total_events
            normalized_oh = events_oh_per_rho_sim[sim][group] / 1.0 #total_events
            normalized_oo = events_oo_per_rho_sim[sim][group] / 1.0 #total_events
            #else:
            #    normalized_hh = 0
            #    normalized_ho = 0
            #    normalized_oh = 0
            #    normalized_oo = 0
            
            normalized_sim_data.append([normalized_hh, normalized_ho, normalized_oh, normalized_oo])

        normalized_data.append(normalized_sim_data)
    
    # Convert new_normalized_data to a NumPy array
    normalized_data = np.array(normalized_data)

    nsims = len(normalized_data)

    events_hh_avg_per_rho = np.nanmean(normalized_data[:, :, 0], axis=0)
    events_hh_std_per_rho = np.nanstd(normalized_data[:, :, 0], axis=0)
    events_hh_u95_per_rho = events_hh_avg_per_rho + z * events_hh_std_per_rho / np.sqrt(nsims)
    events_hh_l95_per_rho = events_hh_avg_per_rho - z * events_hh_std_per_rho / np.sqrt(nsims)
    
    output['hh_avg_per_rho'] = events_hh_avg_per_rho
    output['hh_l95_per_rho'] = events_hh_l95_per_rho
    output['hh_u95_per_rho'] = events_hh_u95_per_rho
    
    events_ho_avg_per_rho = np.nanmean(normalized_data[:, :, 1], axis=0)
    events_ho_std_per_rho = np.nanstd(normalized_data[:, :, 1], axis=0)
    events_ho_u95_per_rho = events_ho_avg_per_rho + z * events_ho_std_per_rho / np.sqrt(nsims)
    events_ho_l95_per_rho = events_ho_avg_per_rho - z * events_ho_std_per_rho / np.sqrt(nsims)

    output['ho_avg_per_rho'] = events_ho_avg_per_rho
    output['ho_l95_per_rho'] = events_ho_l95_per_rho
    output['ho_u95_per_rho'] = events_ho_u95_per_rho
    
    events_oh_avg_per_rho = np.nanmean(normalized_data[:, :, 2], axis=0)
    events_oh_std_per_rho = np.nanstd(normalized_data[:, :, 2], axis=0)
    events_oh_u95_per_rho = events_oh_avg_per_rho + z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oh_l95_per_rho = events_oh_avg_per_rho - z * events_oh_std_per_rho / np.sqrt(nsims)

    output['oh_avg_per_rho'] = events_oh_avg_per_rho
    output['oh_l95_per_rho'] = events_oh_l95_per_rho
    output['oh_u95_per_rho'] = events_oh_u95_per_rho
    
    events_oo_avg_per_rho = np.nanmean(normalized_data[:, :, 3], axis=0)
    events_oo_std_per_rho = np.nanstd(normalized_data[:, :, 3], axis=0)
    events_oo_u95_per_rho = events_oo_avg_per_rho + z * events_oo_std_per_rho / np.sqrt(nsims)
    events_oo_l95_per_rho = events_oo_avg_per_rho - z * events_oo_std_per_rho / np.sqrt(nsims)

    output['oo_avg_per_rho'] = events_oo_avg_per_rho
    output['oo_l95_per_rho'] = events_oo_l95_per_rho
    output['oo_u95_per_rho'] = events_oo_u95_per_rho

    agents_sim = np.sum(agents_per_rho_sim, axis=1)
    infected_sim = np.sum(infected_per_rho_sim, axis=1)
    infected_h_sim = np.sum(infected_h_per_rho_sim, axis=1)
    infected_o_sim = np.sum(infected_o_per_rho_sim, axis=1)
    fra_avg_a_h_sim = np.sum(sum_avg_a_h_per_rho_sim, axis=1) / infected_sim
    fra_avg_a_o_sim = np.sum(sum_avg_a_o_per_rho_sim, axis=1) / infected_sim
    
    fra_avg_a_h_per_rho_sim = sum_avg_a_h_per_rho_sim / infected_per_rho_sim
    fra_avg_a_o_per_rho_sim = sum_avg_a_o_per_rho_sim / infected_per_rho_sim
    fra_avg_a_per_rho_sim = (sum_avg_a_h_per_rho_sim + sum_avg_a_o_per_rho_sim) / infected_per_rho_sim

    f_inf_tr_h_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_tr_h_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_inf_tr_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_inf_tr_h_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_inf_tr_h_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_inf_tr_h_values = f_inf_tr_h_dist_per_rho_sim[sim_idx][rho_idx]
            f_inf_tr_h_dist_per_rho[rho_idx].extend(f_inf_tr_h_values)

    f_inf_tr_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_tr_h_dist_per_rho])
    f_inf_tr_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_tr_h_dist_per_rho])
    moe = z * (f_inf_tr_h_std_per_rho / np.sqrt(len(f_inf_tr_h_avg_per_rho)))
    f_inf_tr_h_u95_per_rho = f_inf_tr_h_avg_per_rho + moe
    f_inf_tr_h_l95_per_rho = f_inf_tr_h_avg_per_rho - moe

    output['f_inf_tr_h_dist_per_rho'] = f_inf_tr_h_dist_per_rho
    output['f_inf_tr_h_avg_per_rho'] = f_inf_tr_h_avg_per_rho
    output['f_inf_tr_h_l95_per_rho'] = f_inf_tr_h_l95_per_rho
    output['f_inf_tr_h_u95_per_rho'] = f_inf_tr_h_u95_per_rho

    f_inf_tr_o_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_tr_o_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_inf_tr_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_inf_tr_o_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_inf_tr_o_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_inf_tr_o_values = f_inf_tr_o_dist_per_rho_sim[sim_idx][rho_idx]
            f_inf_tr_o_dist_per_rho[rho_idx].extend(f_inf_tr_o_values)

    f_inf_tr_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_tr_o_dist_per_rho])
    f_inf_tr_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_tr_o_dist_per_rho])
    moe = z * (f_inf_tr_o_std_per_rho / np.sqrt(len(f_inf_tr_o_avg_per_rho)))
    f_inf_tr_o_u95_per_rho = f_inf_tr_o_avg_per_rho + moe
    f_inf_tr_o_l95_per_rho = f_inf_tr_o_avg_per_rho - moe

    output['f_inf_tr_o_dist_per_rho'] = f_inf_tr_o_dist_per_rho
    output['f_inf_tr_o_avg_per_rho'] = f_inf_tr_o_avg_per_rho
    output['f_inf_tr_o_l95_per_rho'] = f_inf_tr_o_l95_per_rho
    output['f_inf_tr_o_u95_per_rho'] = f_inf_tr_o_u95_per_rho

    nsims = len(fra_avg_a_h_per_rho_sim)

    agents_avg_per_rho = np.mean(agents_per_rho_sim, axis=0)
    agents_std_per_rho = np.std(agents_avg_per_rho, axis=0)
    agents_l95_per_rho = agents_avg_per_rho - z * agents_std_per_rho / np.sqrt(nsims)
    agents_u95_per_rho = agents_avg_per_rho + z * agents_std_per_rho / np.sqrt(nsims)

    output['agents_avg_per_rho'] = agents_avg_per_rho
    output['agents_l95_per_rho'] = agents_l95_per_rho
    output['agents_u95_per_rho'] = agents_u95_per_rho

    infected_avg_per_rho = np.mean(infected_per_rho_sim, axis=0)
    infected_std_per_rho = np.std(infected_per_rho_sim, axis=0)
    infected_l95_per_rho = infected_avg_per_rho - z * infected_std_per_rho / np.sqrt(nsims)
    infected_u95_per_rho = infected_avg_per_rho + z * infected_std_per_rho / np.sqrt(nsims)

    output['infected_avg_per_rho'] = infected_avg_per_rho
    output['infected_l95_per_rho'] = infected_l95_per_rho
    output['infected_u95_per_rho'] = infected_u95_per_rho

    infected_h_avg_per_rho = np.mean(infected_h_per_rho_sim, axis=0)
    infected_h_std_per_rho = np.std(infected_h_per_rho_sim, axis=0)
    infected_h_l95_per_rho = infected_h_avg_per_rho - z * infected_h_std_per_rho / np.sqrt(nsims)
    infected_h_u95_per_rho = infected_h_avg_per_rho + z * infected_h_std_per_rho / np.sqrt(nsims)

    output['infected_h_avg_per_rho'] = infected_h_avg_per_rho
    output['infected_h_l95_per_rho'] = infected_h_l95_per_rho
    output['infected_h_u95_per_rho'] = infected_h_u95_per_rho

    infected_o_avg_per_rho = np.mean(infected_o_per_rho_sim, axis=0)
    infected_o_std_per_rho = np.std(infected_o_per_rho_sim, axis=0)
    infected_o_l95_per_rho = infected_o_avg_per_rho - z * infected_o_std_per_rho / np.sqrt(nsims)
    infected_o_u95_per_rho = infected_o_avg_per_rho + z * infected_o_std_per_rho / np.sqrt(nsims)

    output['infected_o_avg_per_rho'] = infected_o_avg_per_rho
    output['infected_o_l95_per_rho'] = infected_o_l95_per_rho
    output['infected_o_u95_per_rho'] = infected_o_u95_per_rho

    fra_avg_a_h_avg_per_rho = np.mean(fra_avg_a_h_per_rho_sim, axis=0)
    fra_avg_a_h_std_per_rho = np.std(fra_avg_a_h_per_rho_sim, axis=0)
    fra_avg_a_h_l95_per_rho = fra_avg_a_h_avg_per_rho - z * fra_avg_a_h_std_per_rho / np.sqrt(nsims)
    fra_avg_a_h_u95_per_rho = fra_avg_a_h_avg_per_rho + z * fra_avg_a_h_std_per_rho / np.sqrt(nsims)

    output['fra_avg_a_h_avg_per_rho'] = fra_avg_a_h_avg_per_rho
    output['fra_avg_a_h_l95_per_rho'] = fra_avg_a_h_l95_per_rho
    output['fra_avg_a_h_u95_per_rho'] = fra_avg_a_h_u95_per_rho

    fra_avg_a_o_avg_per_rho = np.mean(fra_avg_a_o_per_rho_sim, axis=0)
    fra_avg_a_o_std_per_rho = np.std(fra_avg_a_o_per_rho_sim, axis=0)
    fra_avg_a_o_l95_per_rho = fra_avg_a_o_avg_per_rho - z * fra_avg_a_o_std_per_rho / np.sqrt(nsims)
    fra_avg_a_o_u95_per_rho = fra_avg_a_o_avg_per_rho + z * fra_avg_a_o_std_per_rho / np.sqrt(nsims)

    output['fra_avg_a_o_avg_per_rho'] = fra_avg_a_o_avg_per_rho
    output['fra_avg_a_o_l95_per_rho'] = fra_avg_a_o_l95_per_rho
    output['fra_avg_a_o_u95_per_rho'] = fra_avg_a_o_u95_per_rho
    
    fra_avg_a_avg_per_rho = np.mean(fra_avg_a_per_rho_sim, axis=0)
    fra_avg_a_std_per_rho = np.std(fra_avg_a_per_rho_sim, axis=0)
    fra_avg_a_l95_per_rho = fra_avg_a_avg_per_rho - z * fra_avg_a_std_per_rho / np.sqrt(nsims)
    fra_avg_a_u95_per_rho = fra_avg_a_avg_per_rho + z * fra_avg_a_std_per_rho / np.sqrt(nsims)

    output['fra_avg_a_avg_per_rho'] = fra_avg_a_avg_per_rho
    output['fra_avg_a_l95_per_rho'] = fra_avg_a_l95_per_rho
    output['fra_avg_a_u95_per_rho'] = fra_avg_a_u95_per_rho

    fra_avg_a_h_avg = np.mean(fra_avg_a_h_sim)
    fra_avg_a_h_std = np.std(fra_avg_a_h_sim)
    fra_avg_a_h_l95 = fra_avg_a_h_avg - z * fra_avg_a_h_std / np.sqrt(nsims)
    fra_avg_a_h_u95 = fra_avg_a_h_avg - z * fra_avg_a_h_std / np.sqrt(nsims)

    output['fra_avg_a_h_pop_avg'] = fra_avg_a_h_avg
    output['fra_avg_a_h_pop_l95'] = fra_avg_a_h_l95
    output['fra_avg_a_h_pop_u95'] = fra_avg_a_h_u95
    
    fra_avg_a_o_avg = np.mean(fra_avg_a_o_sim)
    fra_avg_a_o_std = np.std(fra_avg_a_o_sim)
    fra_avg_a_o_l95 = fra_avg_a_o_avg - z * fra_avg_a_o_std / np.sqrt(nsims)
    fra_avg_a_o_u95 = fra_avg_a_o_avg - z * fra_avg_a_o_std / np.sqrt(nsims)

    output['fra_avg_a_o_avg'] = fra_avg_a_o_avg
    output['fra_avg_a_o_l95'] = fra_avg_a_o_l95
    output['fra_avg_a_o_u95'] = fra_avg_a_o_u95

    hh_avg_global = np.nanmean(hh_events_per_sim)
    hh_std_global = np.nanstd(hh_events_per_sim)
    moe = z * hh_std_global / np.sqrt(len(hh_events_per_sim))
    hh_u95_global = hh_avg_global + moe
    hh_l95_global = hh_avg_global - moe

    output['hh_avg_global'] = hh_avg_global
    output['hh_l95_global'] = hh_l95_global
    output['hh_u95_global'] = hh_u95_global

    ho_avg_global = np.nanmean(ho_events_per_sim)
    ho_std_global = np.nanstd(ho_events_per_sim)
    moe = z * ho_std_global / np.sqrt(len(ho_events_per_sim))
    ho_u95_global = ho_avg_global + moe
    ho_l95_global = ho_avg_global - moe

    output['ho_avg_global'] = ho_avg_global
    output['ho_l95_global'] = ho_l95_global
    output['ho_u95_global'] = ho_u95_global

    oh_avg_global = np.nanmean(oh_events_per_sim)
    oh_std_global = np.nanstd(oh_events_per_sim)
    moe = z * oh_std_global / np.sqrt(len(oh_events_per_sim))
    oh_u95_global = oh_avg_global + moe
    oh_l95_global = oh_avg_global - moe

    output['oh_avg_global'] = oh_avg_global
    output['oh_l95_global'] = oh_l95_global
    output['oh_u95_global'] = oh_u95_global

    oo_avg_global = np.nanmean(oo_events_per_sim)
    oo_std_global = np.nanstd(oo_events_per_sim)
    moe = z * oo_std_global / np.sqrt(len(oo_events_per_sim))
    oo_u95_global = oo_avg_global + moe
    oo_l95_global = oo_avg_global - moe

    output['oo_avg_global'] = oo_avg_global
    output['oo_l95_global'] = oo_l95_global
    output['oo_u95_global'] = oo_u95_global

    return output

def compute_depr_chapter_panel3A_stats(
        agents_per_rho_sim, 
        infected_per_rho_sim,
        space_df,
        prevalence_cutoff=0.025,
        r_inv_dist_per_loc_sim=False,
        t_inv_dist_per_loc_sim=False,
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)

    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001
    nlocs_eff = len(attr_l[attr_l > attr_cutoff])

    r_inv_dist_per_loc_sim = [sim for i, sim in enumerate(r_inv_dist_per_loc_sim) if i not in failed_outbreaks]
    t_inv_dist_per_loc_sim = [sim for i, sim in enumerate(t_inv_dist_per_loc_sim) if i not in failed_outbreaks]
    
    nlocs = len(r_inv_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
    r_inv_dist_per_loc = [[] for _ in range(nlocs)]
    t_inv_dist_per_loc = [[] for _ in range(nlocs)]
    invaded_loc_sim = np.zeros((len(r_inv_dist_per_loc_sim), nlocs))
    t_inv_loc_sim = np.zeros((len(t_inv_dist_per_loc_sim), nlocs))
    
    for sim_idx in range(len(r_inv_dist_per_loc_sim)):
        for loc_idx in range(nlocs):
            r_inv_values = r_inv_dist_per_loc_sim[sim_idx][loc_idx]
            r_inv_dist_per_loc[loc_idx].extend(r_inv_values)
            t_inv_values = t_inv_dist_per_loc_sim[sim_idx][loc_idx]
            if len(r_inv_values) > 0:
                if not math.isnan(r_inv_values[0]):
                    invaded_loc_sim[sim_idx][loc_idx] = 1
                    t_inv_dist_per_loc[loc_idx].append(t_inv_values)
                else:
                    invaded_loc_sim[sim_idx][loc_idx] = np.nan
                    t_inv_dist_per_loc[loc_idx].append(np.nan)

    invasion_fraction_avg_loc = np.nanmean(invaded_loc_sim, axis=0)
    invasion_fraction_avg = np.nanmean(np.nansum(invaded_loc_sim, axis=1)) / nlocs_eff
    t_inv_avg_loc = np.nanmean(t_inv_loc_sim, axis=0)
    t_inv_avg = np.nanmean(t_inv_avg_loc)

    r_inv_avg_per_loc = np.array([np.nanmean(sublist) for sublist in r_inv_dist_per_loc])
    r_inv_std_per_loc = np.array([np.nanstd(sublist) for sublist in r_inv_dist_per_loc])
    z = 1.96
    moe = z * (r_inv_std_per_loc / np.sqrt(len(r_inv_avg_per_loc)))
    r_inv_u95_per_loc = r_inv_avg_per_loc + moe
    r_inv_l95_per_loc = r_inv_avg_per_loc - moe

    t_inv_avg_per_loc = np.array([np.nanmean(sublist) for sublist in t_inv_dist_per_loc])
    t_inv_std_per_loc = np.array([np.nanstd(sublist) for sublist in t_inv_dist_per_loc])
    z = 1.96
    moe = z * (t_inv_std_per_loc / np.sqrt(len(t_inv_avg_per_loc)))
    t_inv_u95_per_loc = t_inv_avg_per_loc + moe
    t_inv_l95_per_loc = t_inv_avg_per_loc - moe
    
    nlocs = 2500
    x_cells = int(np.sqrt(nlocs))
    y_cells = x_cells
    inv_rho_avg_lattice = np.zeros((x_cells, y_cells))
    inv_rate_avg_lattice = np.zeros((x_cells, y_cells))
    t_inv_avg_lattice = np.zeros((x_cells, y_cells))
    
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            inv_rho_avg_lattice[y_cells - 1 - j, i] = r_inv_avg_per_loc[l]
            if invasion_fraction_avg_loc[l] == 0:
                inv_rate_avg_lattice[y_cells - 1 - j, i] = np.nan
            else:
                inv_rate_avg_lattice[y_cells - 1 - j, i] = invasion_fraction_avg_loc[l]
            t_inv_avg_lattice[y_cells - 1 - j, i] = t_inv_avg_per_loc[l]
            l += 1

    output = {}
   
    output['inv_rho_avg_lattice'] = inv_rho_avg_lattice
    output['inv_rho_avg_loc'] = r_inv_avg_per_loc
    output['invasion_fraction_avg'] = invasion_fraction_avg
    output['inv_rate_avg_lattice'] = inv_rate_avg_lattice
    output['invasion_fraction_avg_loc'] = invasion_fraction_avg_loc
    output['t_inv_avg_lattice'] = t_inv_avg_lattice
    output['t_inv_avg_loc'] = t_inv_avg_per_loc

    return output

def compute_depr_chapter_panel3B_stats(
        agents_per_rho_sim, 
        infected_per_rho_sim,
        total_cases_loc_sim,
        space_df,
        prevalence_cutoff=0.025,
        r_inf_dist_per_loc_sim=False,
        pt_dist_per_loc_sim=False,
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    total_cases_loc_sim = np.array(total_cases_loc_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    total_cases_loc_sim = np.delete(total_cases_loc_sim, failed_outbreaks, axis=0)

    total_cases_avg_loc = np.nanmean(total_cases_loc_sim, axis=0)
    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001
    nlocs_eff = len(attr_l[attr_l > attr_cutoff])

    r_inf_dist_per_loc_sim = [sim for i, sim in enumerate(r_inf_dist_per_loc_sim) if i not in failed_outbreaks]
    pt_dist_per_loc_sim = [sim for i, sim in enumerate(pt_dist_per_loc_sim) if i not in failed_outbreaks]
    
    nlocs = len(r_inf_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
    r_inf_dist_per_loc = [[] for _ in range(nlocs)]
    pt_dist_per_loc = [[] for _ in range(nlocs)]
    
    for sim_idx in range(len(r_inf_dist_per_loc_sim)):
        for loc_idx in range(nlocs):
            r_inf_values = r_inf_dist_per_loc_sim[sim_idx][loc_idx]
            r_inf_dist_per_loc[loc_idx].extend(r_inf_values)

            if len(r_inf_values) > 0:
                if math.isnan(r_inf_values[0]):
                    pt_dist_per_loc[loc_idx].append(np.nan)
                else:
                    pt_values = pt_dist_per_loc_sim[sim_idx][loc_idx]
                    pt_dist_per_loc[loc_idx].append(pt_values)
    
    r_inf_avg_per_loc = np.array([np.nanmean(sublist) for sublist in r_inf_dist_per_loc])
    r_inf_std_per_loc = np.array([np.nanstd(sublist) for sublist in r_inf_dist_per_loc])
    z = 1.96
    moe = z * (r_inf_std_per_loc / np.sqrt(len(r_inf_avg_per_loc)))
    r_inf_u95_per_loc = r_inf_avg_per_loc + moe
    r_inf_l95_per_loc = r_inf_avg_per_loc - moe

    pt_avg_per_loc = np.array([np.nanmean(sublist) for sublist in pt_dist_per_loc])
    pt_std_per_loc = np.array([np.nanstd(sublist) for sublist in pt_dist_per_loc])
    z = 1.96
    moe = z * (pt_std_per_loc / np.sqrt(len(pt_avg_per_loc)))
    pt_u95_per_loc = pt_avg_per_loc + moe
    pt_l95_per_loc = pt_avg_per_loc - moe
    
    nlocs = 2500
    x_cells = int(np.sqrt(nlocs))
    y_cells = x_cells
    inf_rho_avg_lattice = np.zeros((x_cells, y_cells))
    pt_avg_lattice = np.zeros((x_cells, y_cells))
    
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            inf_rho_avg_lattice[y_cells - 1 - j, i] = r_inf_avg_per_loc[l]
            if math.isnan(r_inf_avg_per_loc[l]):
                pt_avg_lattice[y_cells - 1 - j, i] = np.nan
            else:
                pt_avg_lattice[y_cells - 1 - j, i] = pt_avg_per_loc[l]
            l += 1

    output = {}
    output['total_cases_avg_loc'] = total_cases_avg_loc
    output['attractiveness_l'] = attr_l
    output['inf_rho_avg_lattice'] = inf_rho_avg_lattice
    output['inf_rho_avg_loc'] = r_inf_avg_per_loc
    output['pt_avg_lattice'] = pt_avg_lattice
    output['pt_avg_loc'] = pt_avg_per_loc

    return output
    
def compute_depr_chapter_panel6extra_stats(
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        events_hh_per_rho_sim=None,
        events_ho_per_rho_sim=None,
        events_oh_per_rho_sim=None,
        events_oo_per_rho_sim=None,
        f_trip_hh_dist_per_rho_sim=None,
        f_trip_ho_dist_per_rho_sim=None,
        f_trip_oh_dist_per_rho_sim=None,
        f_trip_oo_dist_per_rho_sim=None,
        da_trip_hh_dist_per_rho_sim=None,
        da_trip_ho_dist_per_rho_sim=None,
        da_trip_oh_dist_per_rho_sim=None,
        da_trip_oo_dist_per_rho_sim=None,
        a_exp_dist_per_rho_sim=None,
        sum_p_exp_per_rho_sim=None,
        prevalence_cutoff=0.025, 
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    events_hh_per_rho_sim = np.array(events_hh_per_rho_sim)
    events_ho_per_rho_sim = np.array(events_ho_per_rho_sim)
    events_oh_per_rho_sim = np.array(events_oh_per_rho_sim)
    events_oo_per_rho_sim = np.array(events_oo_per_rho_sim)
    sum_p_exp_per_rho_sim = np.array(sum_p_exp_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    events_hh_per_rho_sim = np.delete(events_hh_per_rho_sim, failed_outbreaks, axis=0)
    events_ho_per_rho_sim = np.delete(events_ho_per_rho_sim, failed_outbreaks, axis=0)
    events_oh_per_rho_sim = np.delete(events_oh_per_rho_sim, failed_outbreaks, axis=0) 
    events_oo_per_rho_sim = np.delete(events_oo_per_rho_sim, failed_outbreaks, axis=0)
    sum_p_exp_per_rho_sim = np.delete(sum_p_exp_per_rho_sim, failed_outbreaks, axis=0)

    z = 1.96

    a_exp_dist_per_rho_sim = [sim for i, sim in enumerate(a_exp_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(a_exp_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    a_exp_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(a_exp_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            a_exp_values = a_exp_dist_per_rho_sim[sim_idx][rho_idx]
            a_exp_dist_per_rho[rho_idx].extend(a_exp_values)
    
    a_exp_avg_per_rho = np.array([np.nanmean(sublist) for sublist in a_exp_dist_per_rho])
    a_exp_std_per_rho = np.array([np.nanstd(sublist) for sublist in a_exp_dist_per_rho])
    moe = z * (a_exp_std_per_rho / np.sqrt(len(a_exp_avg_per_rho)))
    a_exp_u95_per_rho = a_exp_avg_per_rho + moe
    a_exp_l95_per_rho = a_exp_avg_per_rho - moe

    f_trip_hh_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_hh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_hh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_hh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_hh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_hh_values = f_trip_hh_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_hh_dist_per_rho[rho_idx].extend(f_trip_hh_values)
    
    f_trip_hh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_hh_dist_per_rho])
    f_trip_hh_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_hh_dist_per_rho])
    moe = z * (f_trip_hh_std_per_rho / np.sqrt(len(f_trip_hh_avg_per_rho)))
    f_trip_hh_u95_per_rho = f_trip_hh_avg_per_rho + moe
    f_trip_hh_l95_per_rho = f_trip_hh_avg_per_rho - moe

    f_trip_ho_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_ho_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_ho_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_ho_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_ho_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_ho_values = f_trip_ho_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_ho_dist_per_rho[rho_idx].extend(f_trip_ho_values)
    
    f_trip_ho_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_ho_dist_per_rho])
    f_trip_ho_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_ho_dist_per_rho])
    moe = z * (f_trip_ho_std_per_rho / np.sqrt(len(f_trip_ho_avg_per_rho)))
    f_trip_ho_u95_per_rho = f_trip_ho_avg_per_rho + moe
    f_trip_ho_l95_per_rho = f_trip_ho_avg_per_rho - moe
    
    f_trip_oh_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_oh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_oh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_oh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_oh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_oh_values = f_trip_oh_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_oh_dist_per_rho[rho_idx].extend(f_trip_oh_values)
    
    f_trip_oh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_oh_dist_per_rho])
    f_trip_oh_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_oh_dist_per_rho])
    moe = z * (f_trip_oh_std_per_rho / np.sqrt(len(f_trip_oh_avg_per_rho)))
    f_trip_oh_u95_per_rho = f_trip_oh_avg_per_rho + moe
    f_trip_oh_l95_per_rho = f_trip_oh_avg_per_rho - moe
    
    f_trip_oo_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_oo_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_oo_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_oo_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_oo_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_oo_values = f_trip_oo_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_oo_dist_per_rho[rho_idx].extend(f_trip_oo_values)
    
    f_trip_oo_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_oo_dist_per_rho])
    f_trip_oo_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_oo_dist_per_rho])
    moe = z * (f_trip_oo_std_per_rho / np.sqrt(len(f_trip_oo_avg_per_rho)))
    f_trip_oo_u95_per_rho = f_trip_oo_avg_per_rho + moe
    f_trip_oo_l95_per_rho = f_trip_oo_avg_per_rho - moe
    
    da_trip_hh_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_hh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_hh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_hh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_hh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_hh_values = da_trip_hh_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_hh_dist_per_rho[rho_idx].extend(da_trip_hh_values)
    
    da_trip_hh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_hh_dist_per_rho])
    da_trip_hh_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_hh_dist_per_rho])
    moe = z * (da_trip_hh_std_per_rho / np.sqrt(len(da_trip_hh_avg_per_rho)))
    da_trip_hh_u95_per_rho = da_trip_hh_avg_per_rho + moe
    da_trip_hh_l95_per_rho = da_trip_hh_avg_per_rho - moe

    da_trip_ho_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_ho_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_ho_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_ho_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_ho_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_ho_values = da_trip_ho_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_ho_dist_per_rho[rho_idx].extend(da_trip_ho_values)
    
    da_trip_ho_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_ho_dist_per_rho])
    da_trip_ho_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_ho_dist_per_rho])
    moe = z * (da_trip_ho_std_per_rho / np.sqrt(len(da_trip_ho_avg_per_rho)))
    da_trip_ho_u95_per_rho = da_trip_ho_avg_per_rho + moe
    da_trip_ho_l95_per_rho = da_trip_ho_avg_per_rho - moe
    
    da_trip_oh_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_oh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_oh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_oh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_oh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_oh_values = da_trip_oh_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_oh_dist_per_rho[rho_idx].extend(da_trip_oh_values)
    
    da_trip_oh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_oh_dist_per_rho])
    da_trip_oh_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_oh_dist_per_rho])
    moe = z * (da_trip_oh_std_per_rho / np.sqrt(len(da_trip_oh_avg_per_rho)))
    da_trip_oh_u95_per_rho = da_trip_oh_avg_per_rho + moe
    da_trip_oh_l95_per_rho = da_trip_oh_avg_per_rho - moe
    
    da_trip_oo_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_oo_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_oo_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_oo_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_oo_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_oo_values = da_trip_oo_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_oo_dist_per_rho[rho_idx].extend(da_trip_oo_values)
    
    da_trip_oo_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_oo_dist_per_rho])
    da_trip_oo_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_oo_dist_per_rho])
    moe = z * (da_trip_oo_std_per_rho / np.sqrt(len(da_trip_oo_avg_per_rho)))
    da_trip_oo_u95_per_rho = da_trip_oo_avg_per_rho + moe
    da_trip_oo_l95_per_rho = da_trip_oo_avg_per_rho - moe

    hh_events_per_sim = np.sum(events_hh_per_rho_sim, axis=1)
    ho_events_per_sim = np.sum(events_ho_per_rho_sim, axis=1)
    oh_events_per_sim = np.sum(events_oh_per_rho_sim, axis=1)
    oo_events_per_sim = np.sum(events_oo_per_rho_sim, axis=1)
    
    total_events_sim = hh_events_per_sim + ho_events_per_sim + oh_events_per_sim + oo_events_per_sim
    hh_events_per_sim = hh_events_per_sim / total_events_sim
    ho_events_per_sim = ho_events_per_sim / total_events_sim
    oh_events_per_sim = oh_events_per_sim / total_events_sim
    oo_events_per_sim = oo_events_per_sim / total_events_sim

    z = 1.96

    hh_avg_global = np.nanmean(hh_events_per_sim)
    hh_std_global = np.nanstd(hh_events_per_sim)
    moe = z * hh_std_global / np.sqrt(len(hh_events_per_sim))
    hh_u95_global = hh_avg_global + moe
    hh_l95_global = hh_avg_global - moe

    ho_avg_global = np.nanmean(ho_events_per_sim)
    ho_std_global = np.nanstd(ho_events_per_sim)
    moe = z * ho_std_global / np.sqrt(len(ho_events_per_sim))
    ho_u95_global = ho_avg_global + moe
    ho_l95_global = ho_avg_global - moe

    oh_avg_global = np.nanmean(oh_events_per_sim)
    oh_std_global = np.nanstd(oh_events_per_sim)
    moe = z * oh_std_global / np.sqrt(len(oh_events_per_sim))
    oh_u95_global = oh_avg_global + moe
    oh_l95_global = oh_avg_global - moe

    oo_avg_global = np.nanmean(oo_events_per_sim)
    oo_std_global = np.nanstd(oo_events_per_sim)
    moe = z * oo_std_global / np.sqrt(len(oo_events_per_sim))
    oo_u95_global = oo_avg_global + moe
    oo_l95_global = oo_avg_global - moe

    normalized_data = []

    for sim in range(len(events_hh_per_rho_sim)):
        normalized_sim_data = []
        for group in range(len(events_hh_per_rho_sim[sim])):
            total_events = (
                events_hh_per_rho_sim[sim][group] +
                events_ho_per_rho_sim[sim][group] +
                events_oh_per_rho_sim[sim][group] +
                events_oo_per_rho_sim[sim][group]
            )

            #if total_events != 0:
            normalized_hh = events_hh_per_rho_sim[sim][group] / total_events
            normalized_ho = events_ho_per_rho_sim[sim][group] / total_events
            normalized_oh = events_oh_per_rho_sim[sim][group] / total_events
            normalized_oo = events_oo_per_rho_sim[sim][group] / total_events
            #else:
            #    normalized_hh = 0
            #    normalized_ho = 0
            #    normalized_oh = 0
            #    normalized_oo = 0
            
            normalized_sim_data.append([normalized_hh, normalized_ho, normalized_oh, normalized_oo])

        normalized_data.append(normalized_sim_data)
    
    # Convert new_normalized_data to a NumPy array
    normalized_data = np.array(normalized_data)

    events_hh_avg_per_rho = np.nanmean(normalized_data[:, :, 0], axis=0)
    events_ho_avg_per_rho = np.nanmean(normalized_data[:, :, 1], axis=0)
    events_oh_avg_per_rho = np.nanmean(normalized_data[:, :, 2], axis=0)
    events_oo_avg_per_rho = np.nanmean(normalized_data[:, :, 3], axis=0)
    
    events_hh_std_per_rho = np.nanstd(normalized_data[:, :, 0], axis=0)
    events_ho_std_per_rho = np.nanstd(normalized_data[:, :, 1], axis=0)
    events_oh_std_per_rho = np.nanstd(normalized_data[:, :, 2], axis=0)
    events_oo_std_per_rho = np.nanstd(normalized_data[:, :, 3], axis=0)

    nsims = len(normalized_data)
    events_hh_u95_per_rho = events_hh_avg_per_rho + z * events_hh_std_per_rho / np.sqrt(nsims)
    events_hh_l95_per_rho = events_hh_avg_per_rho - z * events_hh_std_per_rho / np.sqrt(nsims)
    events_ho_u95_per_rho = events_ho_avg_per_rho + z * events_ho_std_per_rho / np.sqrt(nsims)
    events_ho_l95_per_rho = events_ho_avg_per_rho - z * events_ho_std_per_rho / np.sqrt(nsims)
    events_oh_u95_per_rho = events_oh_avg_per_rho + z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oh_l95_per_rho = events_oh_avg_per_rho - z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oo_u95_per_rho = events_oo_avg_per_rho + z * events_oo_std_per_rho / np.sqrt(nsims)
    events_oo_l95_per_rho = events_oo_avg_per_rho - z * events_oo_std_per_rho / np.sqrt(nsims)

    fra_p_exp_per_rho_sim = sum_p_exp_per_rho_sim / infected_per_rho_sim
    nsims = len(fra_p_exp_per_rho_sim)
    fra_p_exp_avg_per_rho = np.mean(fra_p_exp_per_rho_sim, axis=0)
    fra_p_exp_std_per_rho = np.std(fra_p_exp_per_rho_sim, axis=0)
    fra_p_exp_l95_per_rho = fra_p_exp_avg_per_rho - z * fra_p_exp_std_per_rho / np.sqrt(nsims)
    fra_p_exp_u95_per_rho = fra_p_exp_avg_per_rho + z * fra_p_exp_std_per_rho / np.sqrt(nsims)

    output = {}
    output['hh_avg_per_rho'] = events_hh_avg_per_rho
    output['hh_l95_per_rho'] = events_hh_l95_per_rho
    output['hh_u95_per_rho'] = events_hh_u95_per_rho
    output['ho_avg_per_rho'] = events_ho_avg_per_rho
    output['ho_l95_per_rho'] = events_ho_l95_per_rho
    output['ho_u95_per_rho'] = events_ho_u95_per_rho
    output['oh_avg_per_rho'] = events_oh_avg_per_rho
    output['oh_l95_per_rho'] = events_oh_l95_per_rho
    output['oh_u95_per_rho'] = events_oh_u95_per_rho
    output['oo_avg_per_rho'] = events_oo_avg_per_rho
    output['oo_l95_per_rho'] = events_oo_l95_per_rho
    output['oo_u95_per_rho'] = events_oo_u95_per_rho

    output['hh_avg_global'] = hh_avg_global
    output['hh_l95_global'] = hh_l95_global
    output['hh_u95_global'] = hh_u95_global
    output['ho_avg_global'] = ho_avg_global
    output['ho_l95_global'] = ho_l95_global
    output['ho_u95_global'] = ho_u95_global
    output['oh_avg_global'] = oh_avg_global
    output['oh_l95_global'] = oh_l95_global
    output['oh_u95_global'] = oh_u95_global
    output['oo_avg_global'] = oo_avg_global
    output['oo_l95_global'] = oo_l95_global
    output['oo_u95_global'] = oo_u95_global

    output['f_trip_hh_avg_per_rho'] = f_trip_hh_avg_per_rho
    output['f_trip_hh_l95_per_rho'] = f_trip_hh_l95_per_rho
    output['f_trip_hh_u95_per_rho'] = f_trip_hh_u95_per_rho
    output['f_trip_ho_avg_per_rho'] = f_trip_ho_avg_per_rho
    output['f_trip_ho_l95_per_rho'] = f_trip_ho_l95_per_rho
    output['f_trip_ho_u95_per_rho'] = f_trip_ho_u95_per_rho
    output['f_trip_oh_avg_per_rho'] = f_trip_oh_avg_per_rho
    output['f_trip_oh_l95_per_rho'] = f_trip_oh_l95_per_rho
    output['f_trip_oh_u95_per_rho'] = f_trip_oh_u95_per_rho
    output['f_trip_oo_avg_per_rho'] = f_trip_oo_avg_per_rho
    output['f_trip_oo_l95_per_rho'] = f_trip_oo_l95_per_rho
    output['f_trip_oo_u95_per_rho'] = f_trip_oo_u95_per_rho
    output['da_trip_hh_avg_per_rho'] = da_trip_hh_avg_per_rho
    output['da_trip_hh_l95_per_rho'] = da_trip_hh_l95_per_rho
    output['da_trip_hh_u95_per_rho'] = da_trip_hh_u95_per_rho
    output['da_trip_ho_avg_per_rho'] = da_trip_ho_avg_per_rho
    output['da_trip_ho_l95_per_rho'] = da_trip_ho_l95_per_rho
    output['da_trip_ho_u95_per_rho'] = da_trip_ho_u95_per_rho
    output['da_trip_oh_avg_per_rho'] = da_trip_oh_avg_per_rho
    output['da_trip_oh_l95_per_rho'] = da_trip_oh_l95_per_rho
    output['da_trip_oh_u95_per_rho'] = da_trip_oh_u95_per_rho
    output['da_trip_oo_avg_per_rho'] = da_trip_oo_avg_per_rho
    output['da_trip_oo_l95_per_rho'] = da_trip_oo_l95_per_rho
    output['da_trip_oo_u95_per_rho'] = da_trip_oo_u95_per_rho

    output['a_exp_dist_per_rho'] = a_exp_dist_per_rho
    output['a_exp_avg_per_rho'] = a_exp_avg_per_rho
    output['a_exp_l95_per_rho'] = a_exp_l95_per_rho
    output['a_exp_u95_per_rho'] = a_exp_u95_per_rho
    output['p_exp_avg_per_rho'] = fra_p_exp_avg_per_rho
    output['p_exp_l95_per_rho'] = fra_p_exp_l95_per_rho
    output['p_exp_u95_per_rho'] = fra_p_exp_u95_per_rho

    return output

def compute_depr_chapter_panelfinal_stats(
        agents_per_rho_sim=None,
        infected_per_rho_sim=None,
        events_hh_per_rho_sim=None,
        events_ho_per_rho_sim=None,
        events_oh_per_rho_sim=None,
        events_oo_per_rho_sim=None,
        f_trip_hh_dist_per_rho_sim=None,
        f_trip_ho_dist_per_rho_sim=None,
        f_trip_oh_dist_per_rho_sim=None,
        f_trip_oo_dist_per_rho_sim=None,
        da_trip_hh_dist_per_rho_sim=None,
        da_trip_ho_dist_per_rho_sim=None,
        da_trip_oh_dist_per_rho_sim=None,
        da_trip_oo_dist_per_rho_sim=None,
        a_exp_dist_per_rho_sim=None,
        sum_p_exp_per_rho_sim=None,
        infected_h_per_rho_sim=None,
        infected_o_per_rho_sim=None,
        sum_avg_foi_per_rho_sim=None,
        sum_avg_pc_foi_per_rho_sim=None,
        sum_avg_shared_per_rho_sim=None,
        sum_avg_size_per_rho_sim=None,
        sum_avg_t_pop_per_rho_sim=None,
        sum_cum_i_pop_per_rho_sim=None,
        sum_cum_shared_per_rho_sim=None,
        sum_cum_size_per_rho_sim=None,
        sum_cum_t_pop_per_rho_sim=None,
        f_inf_tr_h_dist_per_rho_sim=None,
        f_inf_tr_o_dist_per_rho_sim=None,
        nevents_eff_per_rho_sim=None,
        prevalence_cutoff=0.025, 
        ):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    events_hh_per_rho_sim = np.array(events_hh_per_rho_sim)
    events_ho_per_rho_sim = np.array(events_ho_per_rho_sim)
    events_oh_per_rho_sim = np.array(events_oh_per_rho_sim)
    events_oo_per_rho_sim = np.array(events_oo_per_rho_sim)
    sum_p_exp_per_rho_sim = np.array(sum_p_exp_per_rho_sim)
    sum_avg_foi_per_rho_sim = np.array(sum_avg_foi_per_rho_sim)
    sum_avg_pc_foi_per_rho_sim = np.array(sum_avg_pc_foi_per_rho_sim)
    sum_avg_shared_per_rho_sim = np.array(sum_avg_shared_per_rho_sim)
    sum_avg_size_per_rho_sim = np.array(sum_avg_size_per_rho_sim)
    sum_avg_t_pop_per_rho_sim = np.array(sum_avg_t_pop_per_rho_sim)
    sum_cum_i_pop_per_rho_sim = np.array(sum_cum_i_pop_per_rho_sim)
    sum_cum_shared_per_rho_sim = np.array(sum_cum_shared_per_rho_sim)
    sum_cum_size_per_rho_sim = np.array(sum_cum_size_per_rho_sim)
    sum_cum_t_pop_per_rho_sim = np.array(sum_cum_t_pop_per_rho_sim)
    f_inf_tr_h_dist_per_rho_sim= np.array(f_inf_tr_h_dist_per_rho_sim)
    f_inf_tr_o_dist_per_rho_sim= np.array(f_inf_tr_o_dist_per_rho_sim)
    nevents_eff_per_rho_sim = np.array(nevents_eff_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    events_hh_per_rho_sim = np.delete(events_hh_per_rho_sim, failed_outbreaks, axis=0)
    events_ho_per_rho_sim = np.delete(events_ho_per_rho_sim, failed_outbreaks, axis=0)
    events_oh_per_rho_sim = np.delete(events_oh_per_rho_sim, failed_outbreaks, axis=0) 
    events_oo_per_rho_sim = np.delete(events_oo_per_rho_sim, failed_outbreaks, axis=0)
    sum_p_exp_per_rho_sim = np.delete(sum_p_exp_per_rho_sim, failed_outbreaks, axis=0)
    infected_h_per_rho_sim = np.delete(infected_h_per_rho_sim, failed_outbreaks, axis=0)
    infected_o_per_rho_sim = np.delete(infected_o_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_foi_per_rho_sim = np.delete(sum_avg_foi_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_pc_foi_per_rho_sim = np.delete(sum_avg_pc_foi_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_shared_per_rho_sim = np.delete(sum_avg_shared_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_size_per_rho_sim = np.delete(sum_avg_size_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_t_pop_per_rho_sim = np.delete(sum_avg_t_pop_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_i_pop_per_rho_sim = np.delete(sum_cum_i_pop_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_shared_per_rho_sim = np.delete(sum_cum_shared_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_size_per_rho_sim = np.delete(sum_cum_size_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_t_pop_per_rho_sim = np.delete(sum_cum_t_pop_per_rho_sim, failed_outbreaks, axis=0)
    nevents_eff_per_rho_sim = np.delete(nevents_eff_per_rho_sim, failed_outbreaks, axis=0)

    agents_sim = np.sum(agents_per_rho_sim, axis=1)
    infected_sim = np.sum(infected_per_rho_sim, axis=1)
    infected_h_sim = np.sum(infected_h_per_rho_sim, axis=1)
    infected_o_sim = np.sum(infected_o_per_rho_sim, axis=1)
    fra_avg_foi_sim = np.sum(sum_avg_foi_per_rho_sim, axis=1) / infected_sim
    fra_avg_pc_foi_sim = np.sum(sum_avg_pc_foi_per_rho_sim, axis=1) / infected_sim
    fra_avg_shared_sim = np.sum(sum_avg_shared_per_rho_sim, axis=1) / infected_sim
    fra_avg_size_sim = np.sum(sum_avg_size_per_rho_sim, axis=1) / infected_sim
    fra_avg_t_pop_sim = np.sum(sum_avg_t_pop_per_rho_sim, axis=1) / infected_sim
    fra_cum_i_pop_sim = np.sum(sum_cum_i_pop_per_rho_sim, axis=1) / infected_sim
    fra_cum_shared_sim = np.sum(sum_cum_shared_per_rho_sim, axis=1) / infected_sim
    fra_cum_size_sim = np.sum(sum_cum_size_per_rho_sim, axis=1) / infected_sim
    fra_cum_t_pop_sim = np.sum(sum_cum_t_pop_per_rho_sim, axis=1) / infected_sim

    fra_avg_foi_per_rho_sim = sum_avg_foi_per_rho_sim / infected_per_rho_sim
    fra_avg_pc_foi_per_rho_sim = sum_avg_pc_foi_per_rho_sim / infected_per_rho_sim
    fra_avg_shared_per_rho_sim = sum_avg_shared_per_rho_sim / infected_per_rho_sim
    fra_avg_size_per_rho_sim = sum_avg_size_per_rho_sim / 1.0 #infected_per_rho_sim
    fra_avg_t_pop_per_rho_sim = sum_avg_t_pop_per_rho_sim / infected_per_rho_sim
    fra_cum_shared_per_rho_sim = sum_cum_shared_per_rho_sim / infected_per_rho_sim
    fra_cum_size_per_rho_sim = sum_cum_size_per_rho_sim / 1.0 #infected_per_rho_sim
    fra_cum_t_pop_per_rho_sim = sum_cum_t_pop_per_rho_sim / infected_per_rho_sim
    fra_nevents_eff_per_rho_sim = nevents_eff_per_rho_sim / infected_per_rho_sim

    output = {}

    z = 1.96

    a_exp_dist_per_rho_sim = [sim for i, sim in enumerate(a_exp_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(a_exp_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    a_exp_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(a_exp_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            a_exp_values = a_exp_dist_per_rho_sim[sim_idx][rho_idx]
            a_exp_dist_per_rho[rho_idx].extend(a_exp_values)
    
    a_exp_avg_per_rho = np.array([np.nanmean(sublist) for sublist in a_exp_dist_per_rho])
    a_exp_std_per_rho = np.array([np.nanstd(sublist) for sublist in a_exp_dist_per_rho])
    moe = z * (a_exp_std_per_rho / np.sqrt(len(a_exp_avg_per_rho)))
    a_exp_u95_per_rho = a_exp_avg_per_rho + moe
    a_exp_l95_per_rho = a_exp_avg_per_rho - moe

    #output['a_exp_dist_per_rho'] = a_exp_dist_per_rho
    output['a_exp_avg_per_rho'] = a_exp_avg_per_rho
    output['a_exp_l95_per_rho'] = a_exp_l95_per_rho
    output['a_exp_u95_per_rho'] = a_exp_u95_per_rho

    f_inf_tr_h_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_tr_h_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_inf_tr_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_inf_tr_h_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_inf_tr_h_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_inf_tr_h_values = f_inf_tr_h_dist_per_rho_sim[sim_idx][rho_idx]
            f_inf_tr_h_dist_per_rho[rho_idx].extend(f_inf_tr_h_values)

    f_inf_tr_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_tr_h_dist_per_rho])
    f_inf_tr_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_tr_h_dist_per_rho])
    moe = z * (f_inf_tr_h_std_per_rho / np.sqrt(len(f_inf_tr_h_avg_per_rho)))
    f_inf_tr_h_u95_per_rho = f_inf_tr_h_avg_per_rho + moe
    f_inf_tr_h_l95_per_rho = f_inf_tr_h_avg_per_rho - moe

    output['f_inf_tr_h_dist_per_rho'] = f_inf_tr_h_dist_per_rho
    output['f_inf_tr_h_avg_per_rho'] = f_inf_tr_h_avg_per_rho
    output['f_inf_tr_h_l95_per_rho'] = f_inf_tr_h_l95_per_rho
    output['f_inf_tr_h_u95_per_rho'] = f_inf_tr_h_u95_per_rho

    f_inf_tr_o_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_tr_o_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_inf_tr_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_inf_tr_o_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_inf_tr_o_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_inf_tr_o_values = f_inf_tr_o_dist_per_rho_sim[sim_idx][rho_idx]
            f_inf_tr_o_dist_per_rho[rho_idx].extend(f_inf_tr_o_values)

    f_inf_tr_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_tr_o_dist_per_rho])
    f_inf_tr_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_tr_o_dist_per_rho])
    moe = z * (f_inf_tr_o_std_per_rho / np.sqrt(len(f_inf_tr_o_avg_per_rho)))
    f_inf_tr_o_u95_per_rho = f_inf_tr_o_avg_per_rho + moe
    f_inf_tr_o_l95_per_rho = f_inf_tr_o_avg_per_rho - moe

    output['f_inf_tr_o_dist_per_rho'] = f_inf_tr_o_dist_per_rho
    output['f_inf_tr_o_avg_per_rho'] = f_inf_tr_o_avg_per_rho
    output['f_inf_tr_o_l95_per_rho'] = f_inf_tr_o_l95_per_rho
    output['f_inf_tr_o_u95_per_rho'] = f_inf_tr_o_u95_per_rho

    f_trip_hh_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_hh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_hh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_hh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_hh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_hh_values = f_trip_hh_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_hh_dist_per_rho[rho_idx].extend(f_trip_hh_values)
    
    f_trip_hh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_hh_dist_per_rho])
    f_trip_hh_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_hh_dist_per_rho])
    moe = z * (f_trip_hh_std_per_rho / np.sqrt(len(f_trip_hh_avg_per_rho)))
    f_trip_hh_u95_per_rho = f_trip_hh_avg_per_rho + moe
    f_trip_hh_l95_per_rho = f_trip_hh_avg_per_rho - moe

    f_trip_ho_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_ho_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_ho_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_ho_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_ho_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_ho_values = f_trip_ho_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_ho_dist_per_rho[rho_idx].extend(f_trip_ho_values)
    
    f_trip_ho_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_ho_dist_per_rho])
    f_trip_ho_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_ho_dist_per_rho])
    moe = z * (f_trip_ho_std_per_rho / np.sqrt(len(f_trip_ho_avg_per_rho)))
    f_trip_ho_u95_per_rho = f_trip_ho_avg_per_rho + moe
    f_trip_ho_l95_per_rho = f_trip_ho_avg_per_rho - moe

    f_trip_oh_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_oh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_oh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_oh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_oh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_oh_values = f_trip_oh_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_oh_dist_per_rho[rho_idx].extend(f_trip_oh_values)
    
    f_trip_oh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_oh_dist_per_rho])
    f_trip_oh_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_oh_dist_per_rho])
    moe = z * (f_trip_oh_std_per_rho / np.sqrt(len(f_trip_oh_avg_per_rho)))
    f_trip_oh_u95_per_rho = f_trip_oh_avg_per_rho + moe
    f_trip_oh_l95_per_rho = f_trip_oh_avg_per_rho - moe
    
    f_trip_oo_dist_per_rho_sim = [sim for i, sim in enumerate(f_trip_oo_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(f_trip_oo_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    f_trip_oo_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(f_trip_oo_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            f_trip_oo_values = f_trip_oo_dist_per_rho_sim[sim_idx][rho_idx]
            f_trip_oo_dist_per_rho[rho_idx].extend(f_trip_oo_values)
    
    f_trip_oo_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_trip_oo_dist_per_rho])
    f_trip_oo_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_trip_oo_dist_per_rho])
    moe = z * (f_trip_oo_std_per_rho / np.sqrt(len(f_trip_oo_avg_per_rho)))
    f_trip_oo_u95_per_rho = f_trip_oo_avg_per_rho + moe
    f_trip_oo_l95_per_rho = f_trip_oo_avg_per_rho - moe

    output['f_trip_hh_avg_per_rho'] = f_trip_hh_avg_per_rho
    output['f_trip_hh_l95_per_rho'] = f_trip_hh_l95_per_rho
    output['f_trip_hh_u95_per_rho'] = f_trip_hh_u95_per_rho
    output['f_trip_ho_avg_per_rho'] = f_trip_ho_avg_per_rho
    output['f_trip_ho_l95_per_rho'] = f_trip_ho_l95_per_rho
    output['f_trip_ho_u95_per_rho'] = f_trip_ho_u95_per_rho
    output['f_trip_oh_avg_per_rho'] = f_trip_oh_avg_per_rho
    output['f_trip_oh_l95_per_rho'] = f_trip_oh_l95_per_rho
    output['f_trip_oh_u95_per_rho'] = f_trip_oh_u95_per_rho
    output['f_trip_oo_avg_per_rho'] = f_trip_oo_avg_per_rho
    output['f_trip_oo_l95_per_rho'] = f_trip_oo_l95_per_rho
    output['f_trip_oo_u95_per_rho'] = f_trip_oo_u95_per_rho
    
    da_trip_hh_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_hh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_hh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_hh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_hh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_hh_values = da_trip_hh_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_hh_dist_per_rho[rho_idx].extend(da_trip_hh_values)
    
    da_trip_hh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_hh_dist_per_rho])
    da_trip_hh_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_hh_dist_per_rho])
    moe = z * (da_trip_hh_std_per_rho / np.sqrt(len(da_trip_hh_avg_per_rho)))
    da_trip_hh_u95_per_rho = da_trip_hh_avg_per_rho + moe
    da_trip_hh_l95_per_rho = da_trip_hh_avg_per_rho - moe

    da_trip_ho_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_ho_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_ho_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_ho_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_ho_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_ho_values = da_trip_ho_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_ho_dist_per_rho[rho_idx].extend(da_trip_ho_values)
    
    da_trip_ho_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_ho_dist_per_rho])
    da_trip_ho_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_ho_dist_per_rho])
    moe = z * (da_trip_ho_std_per_rho / np.sqrt(len(da_trip_ho_avg_per_rho)))
    da_trip_ho_u95_per_rho = da_trip_ho_avg_per_rho + moe
    da_trip_ho_l95_per_rho = da_trip_ho_avg_per_rho - moe
    
    da_trip_oh_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_oh_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_oh_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_oh_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_oh_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_oh_values = da_trip_oh_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_oh_dist_per_rho[rho_idx].extend(da_trip_oh_values)
    
    da_trip_oh_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_oh_dist_per_rho])
    da_trip_oh_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_oh_dist_per_rho])
    moe = z * (da_trip_oh_std_per_rho / np.sqrt(len(da_trip_oh_avg_per_rho)))
    da_trip_oh_u95_per_rho = da_trip_oh_avg_per_rho + moe
    da_trip_oh_l95_per_rho = da_trip_oh_avg_per_rho - moe
    
    da_trip_oo_dist_per_rho_sim = [sim for i, sim in enumerate(da_trip_oo_dist_per_rho_sim) if i not in failed_outbreaks]
    nbins = len(da_trip_oo_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
    da_trip_oo_dist_per_rho = [[] for _ in range(nbins)]
    for sim_idx in range(len(da_trip_oo_dist_per_rho_sim)):
        for rho_idx in range(nbins):
            da_trip_oo_values = da_trip_oo_dist_per_rho_sim[sim_idx][rho_idx]
            da_trip_oo_dist_per_rho[rho_idx].extend(da_trip_oo_values)
    
    da_trip_oo_avg_per_rho = np.array([np.nanmean(sublist) for sublist in da_trip_oo_dist_per_rho])
    da_trip_oo_std_per_rho = np.array([np.nanstd(sublist) for sublist in da_trip_oo_dist_per_rho])
    moe = z * (da_trip_oo_std_per_rho / np.sqrt(len(da_trip_oo_avg_per_rho)))
    da_trip_oo_u95_per_rho = da_trip_oo_avg_per_rho + moe
    da_trip_oo_l95_per_rho = da_trip_oo_avg_per_rho - moe

    #output['da_trip_hh_avg_per_rho'] = da_trip_hh_avg_per_rho
    #output['da_trip_hh_l95_per_rho'] = da_trip_hh_l95_per_rho
    #output['da_trip_hh_u95_per_rho'] = da_trip_hh_u95_per_rho
    #output['da_trip_ho_avg_per_rho'] = da_trip_ho_avg_per_rho
    #output['da_trip_ho_l95_per_rho'] = da_trip_ho_l95_per_rho
    #output['da_trip_ho_u95_per_rho'] = da_trip_ho_u95_per_rho
    #output['da_trip_oh_avg_per_rho'] = da_trip_oh_avg_per_rho
    #output['da_trip_oh_l95_per_rho'] = da_trip_oh_l95_per_rho
    #output['da_trip_oh_u95_per_rho'] = da_trip_oh_u95_per_rho
    #output['da_trip_oo_avg_per_rho'] = da_trip_oo_avg_per_rho
    #output['da_trip_oo_l95_per_rho'] = da_trip_oo_l95_per_rho
    #output['da_trip_oo_u95_per_rho'] = da_trip_oo_u95_per_rho

    hh_events_per_sim = np.sum(events_hh_per_rho_sim, axis=1)
    ho_events_per_sim = np.sum(events_ho_per_rho_sim, axis=1)
    oh_events_per_sim = np.sum(events_oh_per_rho_sim, axis=1)
    oo_events_per_sim = np.sum(events_oo_per_rho_sim, axis=1)
    
    total_events_sim = hh_events_per_sim + ho_events_per_sim + oh_events_per_sim + oo_events_per_sim
    hh_events_per_sim = hh_events_per_sim / total_events_sim
    ho_events_per_sim = ho_events_per_sim / total_events_sim
    oh_events_per_sim = oh_events_per_sim / total_events_sim
    oo_events_per_sim = oo_events_per_sim / total_events_sim

    hh_avg_global = np.nanmean(hh_events_per_sim)
    hh_std_global = np.nanstd(hh_events_per_sim)
    moe = z * hh_std_global / np.sqrt(len(hh_events_per_sim))
    hh_u95_global = hh_avg_global + moe
    hh_l95_global = hh_avg_global - moe

    ho_avg_global = np.nanmean(ho_events_per_sim)
    ho_std_global = np.nanstd(ho_events_per_sim)
    moe = z * ho_std_global / np.sqrt(len(ho_events_per_sim))
    ho_u95_global = ho_avg_global + moe
    ho_l95_global = ho_avg_global - moe

    oh_avg_global = np.nanmean(oh_events_per_sim)
    oh_std_global = np.nanstd(oh_events_per_sim)
    moe = z * oh_std_global / np.sqrt(len(oh_events_per_sim))
    oh_u95_global = oh_avg_global + moe
    oh_l95_global = oh_avg_global - moe

    oo_avg_global = np.nanmean(oo_events_per_sim)
    oo_std_global = np.nanstd(oo_events_per_sim)
    moe = z * oo_std_global / np.sqrt(len(oo_events_per_sim))
    oo_u95_global = oo_avg_global + moe
    oo_l95_global = oo_avg_global - moe

    normalized_data = []

    for sim in range(len(events_hh_per_rho_sim)):
        normalized_sim_data = []
        for group in range(len(events_hh_per_rho_sim[sim])):
            total_events = (
                events_hh_per_rho_sim[sim][group] +
                events_ho_per_rho_sim[sim][group] +
                events_oh_per_rho_sim[sim][group] +
                events_oo_per_rho_sim[sim][group]
            )

            #if total_events != 0:
            normalized_hh = events_hh_per_rho_sim[sim][group] / 1.0 #total_events
            normalized_ho = events_ho_per_rho_sim[sim][group] / 1.0 #total_events
            normalized_oh = events_oh_per_rho_sim[sim][group] / 1.0 #total_events
            normalized_oo = events_oo_per_rho_sim[sim][group] / 1.0 #total_events
            #else:
            #    normalized_hh = 0
            #    normalized_ho = 0
            #    normalized_oh = 0
            #    normalized_oo = 0
            
            normalized_sim_data.append([normalized_hh, normalized_ho, normalized_oh, normalized_oo])

        normalized_data.append(normalized_sim_data)
    
    # Convert new_normalized_data to a NumPy array
    normalized_data = np.array(normalized_data)

    events_hh_avg_per_rho = np.nanmean(normalized_data[:, :, 0], axis=0)
    events_ho_avg_per_rho = np.nanmean(normalized_data[:, :, 1], axis=0)
    events_oh_avg_per_rho = np.nanmean(normalized_data[:, :, 2], axis=0)
    events_oo_avg_per_rho = np.nanmean(normalized_data[:, :, 3], axis=0)
    
    events_hh_std_per_rho = np.nanstd(normalized_data[:, :, 0], axis=0)
    events_ho_std_per_rho = np.nanstd(normalized_data[:, :, 1], axis=0)
    events_oh_std_per_rho = np.nanstd(normalized_data[:, :, 2], axis=0)
    events_oo_std_per_rho = np.nanstd(normalized_data[:, :, 3], axis=0)

    nsims = len(normalized_data)
    events_hh_u95_per_rho = events_hh_avg_per_rho + z * events_hh_std_per_rho / np.sqrt(nsims)
    events_hh_l95_per_rho = events_hh_avg_per_rho - z * events_hh_std_per_rho / np.sqrt(nsims)
    events_ho_u95_per_rho = events_ho_avg_per_rho + z * events_ho_std_per_rho / np.sqrt(nsims)
    events_ho_l95_per_rho = events_ho_avg_per_rho - z * events_ho_std_per_rho / np.sqrt(nsims)
    events_oh_u95_per_rho = events_oh_avg_per_rho + z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oh_l95_per_rho = events_oh_avg_per_rho - z * events_oh_std_per_rho / np.sqrt(nsims)
    events_oo_u95_per_rho = events_oo_avg_per_rho + z * events_oo_std_per_rho / np.sqrt(nsims)
    events_oo_l95_per_rho = events_oo_avg_per_rho - z * events_oo_std_per_rho / np.sqrt(nsims)

    output['hh_avg_per_rho'] = events_hh_avg_per_rho
    output['hh_l95_per_rho'] = events_hh_l95_per_rho
    output['hh_u95_per_rho'] = events_hh_u95_per_rho
    output['ho_avg_per_rho'] = events_ho_avg_per_rho
    output['ho_l95_per_rho'] = events_ho_l95_per_rho
    output['ho_u95_per_rho'] = events_ho_u95_per_rho
    output['oh_avg_per_rho'] = events_oh_avg_per_rho
    output['oh_l95_per_rho'] = events_oh_l95_per_rho
    output['oh_u95_per_rho'] = events_oh_u95_per_rho
    output['oo_avg_per_rho'] = events_oo_avg_per_rho
    output['oo_l95_per_rho'] = events_oo_l95_per_rho
    output['oo_u95_per_rho'] = events_oo_u95_per_rho

    output['hh_avg_global'] = hh_avg_global
    output['hh_l95_global'] = hh_l95_global
    output['hh_u95_global'] = hh_u95_global
    output['ho_avg_global'] = ho_avg_global
    output['ho_l95_global'] = ho_l95_global
    output['ho_u95_global'] = ho_u95_global
    output['oh_avg_global'] = oh_avg_global
    output['oh_l95_global'] = oh_l95_global
    output['oh_u95_global'] = oh_u95_global
    output['oo_avg_global'] = oo_avg_global
    output['oo_l95_global'] = oo_l95_global
    output['oo_u95_global'] = oo_u95_global

    fra_p_exp_per_rho_sim = sum_p_exp_per_rho_sim / infected_per_rho_sim
    nsims = len(fra_p_exp_per_rho_sim)
    fra_p_exp_avg_per_rho = np.mean(fra_p_exp_per_rho_sim, axis=0)
    fra_p_exp_std_per_rho = np.std(fra_p_exp_per_rho_sim, axis=0)
    fra_p_exp_l95_per_rho = fra_p_exp_avg_per_rho - z * fra_p_exp_std_per_rho / np.sqrt(nsims)
    fra_p_exp_u95_per_rho = fra_p_exp_avg_per_rho + z * fra_p_exp_std_per_rho / np.sqrt(nsims)

    output['p_exp_avg_per_rho'] = fra_p_exp_avg_per_rho
    output['p_exp_l95_per_rho'] = fra_p_exp_l95_per_rho
    output['p_exp_u95_per_rho'] = fra_p_exp_u95_per_rho

    agents_avg_per_rho = np.mean(agents_per_rho_sim, axis=0)
    infected_avg_per_rho = np.mean(infected_per_rho_sim, axis=0)

    output['agents_avg_per_rho'] = agents_avg_per_rho
    output['infected_avg_per_rho'] = infected_avg_per_rho

    nevents_eff_avg_per_rho = np.mean(nevents_eff_per_rho_sim, axis=0)
    fra_nevents_eff_avg_per_rho = np.mean(fra_nevents_eff_per_rho_sim, axis=0)

    output['nevents_eff_avg_per_rho'] = nevents_eff_avg_per_rho
    output['fra_nevents_eff_avg_per_rho'] = fra_nevents_eff_avg_per_rho

    nsims = len(fra_avg_foi_per_rho_sim)
    
    fra_avg_foi_avg_per_rho = np.mean(fra_avg_foi_per_rho_sim, axis=0)
    fra_avg_foi_std_per_rho = np.std(fra_avg_foi_avg_per_rho, axis=0)
    fra_avg_foi_l95_per_rho = fra_avg_foi_avg_per_rho - z * fra_avg_foi_std_per_rho / np.sqrt(nsims)
    fra_avg_foi_u95_per_rho = fra_avg_foi_avg_per_rho + z * fra_avg_foi_std_per_rho / np.sqrt(nsims)

    output['fra_avg_foi_avg_per_rho'] = fra_avg_foi_avg_per_rho
    output['fra_avg_foi_l95_per_rho'] = fra_avg_foi_l95_per_rho
    output['fra_avg_foi_u95_per_rho'] = fra_avg_foi_u95_per_rho

    fra_avg_pc_foi_avg_per_rho = np.mean(fra_avg_pc_foi_per_rho_sim, axis=0)
    fra_avg_pc_foi_std_per_rho = np.std(fra_avg_pc_foi_avg_per_rho, axis=0)
    fra_avg_pc_foi_l95_per_rho = fra_avg_pc_foi_avg_per_rho - z * fra_avg_pc_foi_std_per_rho / np.sqrt(nsims)
    fra_avg_pc_foi_u95_per_rho = fra_avg_pc_foi_avg_per_rho + z * fra_avg_pc_foi_std_per_rho / np.sqrt(nsims)

    output['fra_avg_pc_foi_avg_per_rho'] = fra_avg_pc_foi_avg_per_rho
    output['fra_avg_pc_foi_l95_per_rho'] = fra_avg_pc_foi_l95_per_rho
    output['fra_avg_pc_foi_u95_per_rho'] = fra_avg_pc_foi_u95_per_rho

    fra_avg_shared_avg_per_rho = np.mean(fra_avg_shared_per_rho_sim, axis=0)
    fra_avg_shared_std_per_rho = np.std(fra_avg_shared_avg_per_rho, axis=0)
    fra_avg_shared_l95_per_rho = fra_avg_shared_avg_per_rho - z * fra_avg_shared_std_per_rho / np.sqrt(nsims)
    fra_avg_shared_u95_per_rho = fra_avg_shared_avg_per_rho + z * fra_avg_shared_std_per_rho / np.sqrt(nsims)

    output['fra_avg_shared_avg_per_rho'] = fra_avg_shared_avg_per_rho
    output['fra_avg_shared_l95_per_rho'] = fra_avg_shared_l95_per_rho
    output['fra_avg_shared_u95_per_rho'] = fra_avg_shared_u95_per_rho

    fra_avg_size_avg_per_rho = np.mean(fra_avg_size_per_rho_sim, axis=0)
    fra_avg_size_std_per_rho = np.std(fra_avg_size_avg_per_rho, axis=0)
    fra_avg_size_l95_per_rho = fra_avg_size_avg_per_rho - z * fra_avg_size_std_per_rho / np.sqrt(nsims)
    fra_avg_size_u95_per_rho = fra_avg_size_avg_per_rho + z * fra_avg_size_std_per_rho / np.sqrt(nsims)

    output['fra_avg_size_avg_per_rho'] = fra_avg_size_avg_per_rho
    output['fra_avg_size_l95_per_rho'] = fra_avg_size_l95_per_rho
    output['fra_avg_size_u95_per_rho'] = fra_avg_size_u95_per_rho

    fra_avg_t_pop_avg_per_rho = np.mean(fra_avg_t_pop_per_rho_sim, axis=0)
    fra_avg_t_pop_std_per_rho = np.std(fra_avg_t_pop_avg_per_rho, axis=0)
    fra_avg_t_pop_l95_per_rho = fra_avg_t_pop_avg_per_rho - z * fra_avg_t_pop_std_per_rho / np.sqrt(nsims)
    fra_avg_t_pop_u95_per_rho = fra_avg_t_pop_avg_per_rho + z * fra_avg_t_pop_std_per_rho / np.sqrt(nsims)

    output['fra_avg_t_pop_avg_per_rho'] = fra_avg_t_pop_avg_per_rho
    output['fra_avg_t_pop_l95_per_rho'] = fra_avg_t_pop_l95_per_rho
    output['fra_avg_t_pop_u95_per_rho'] = fra_avg_t_pop_u95_per_rho

    fra_cum_shared_avg_per_rho = np.mean(fra_cum_shared_per_rho_sim, axis=0)
    fra_cum_shared_std_per_rho = np.std(fra_cum_shared_avg_per_rho, axis=0)
    fra_cum_shared_l95_per_rho = fra_cum_shared_avg_per_rho - z * fra_cum_shared_std_per_rho / np.sqrt(nsims)
    fra_cum_shared_u95_per_rho = fra_cum_shared_avg_per_rho + z * fra_cum_shared_std_per_rho / np.sqrt(nsims)

    output['fra_cum_shared_avg_per_rho'] = fra_cum_shared_avg_per_rho
    output['fra_cum_shared_l95_per_rho'] = fra_cum_shared_l95_per_rho
    output['fra_cum_shared_u95_per_rho'] = fra_cum_shared_u95_per_rho

    fra_cum_size_avg_per_rho = np.mean(fra_cum_size_per_rho_sim, axis=0)
    fra_cum_size_std_per_rho = np.std(fra_cum_size_avg_per_rho, axis=0)
    fra_cum_size_l95_per_rho = fra_cum_size_avg_per_rho - z * fra_cum_size_std_per_rho / np.sqrt(nsims)
    fra_cum_size_u95_per_rho = fra_cum_size_avg_per_rho + z * fra_cum_size_std_per_rho / np.sqrt(nsims)

    output['fra_cum_size_avg_per_rho'] = fra_cum_size_avg_per_rho
    output['fra_cum_size_l95_per_rho'] = fra_cum_size_l95_per_rho
    output['fra_cum_size_u95_per_rho'] = fra_cum_size_u95_per_rho

    fra_cum_t_pop_avg_per_rho = np.mean(fra_cum_t_pop_per_rho_sim, axis=0)
    fra_cum_t_pop_std_per_rho = np.std(fra_cum_t_pop_avg_per_rho, axis=0)
    fra_cum_t_pop_l95_per_rho = fra_cum_t_pop_avg_per_rho - z * fra_cum_t_pop_std_per_rho / np.sqrt(nsims)
    fra_cum_t_pop_u95_per_rho = fra_cum_t_pop_avg_per_rho + z * fra_cum_t_pop_std_per_rho / np.sqrt(nsims)

    output['fra_cum_t_pop_avg_per_rho'] = fra_cum_t_pop_avg_per_rho
    output['fra_cum_t_pop_l95_per_rho'] = fra_cum_t_pop_l95_per_rho
    output['fra_cum_t_pop_u95_per_rho'] = fra_cum_t_pop_u95_per_rho

    fra_avg_foi_avg = np.mean(fra_avg_foi_sim)
    fra_avg_foi_std = np.std(fra_avg_foi_sim)
    fra_avg_foi_l95 = fra_avg_foi_avg - z * fra_avg_foi_std / np.sqrt(nsims)
    fra_avg_foi_u95 = fra_avg_foi_avg - z * fra_avg_foi_std / np.sqrt(nsims)

    output['fra_avg_foi_avg'] = fra_avg_foi_avg
    output['fra_avg_foi_l95'] = fra_avg_foi_l95
    output['fra_avg_foi_u95'] = fra_avg_foi_u95

    fra_avg_pc_foi_avg = np.mean(fra_avg_pc_foi_sim)
    fra_avg_pc_foi_std = np.std(fra_avg_pc_foi_sim)
    fra_avg_pc_foi_l95 = fra_avg_pc_foi_avg - z * fra_avg_pc_foi_std / np.sqrt(nsims)
    fra_avg_pc_foi_u95 = fra_avg_pc_foi_avg - z * fra_avg_pc_foi_std / np.sqrt(nsims)

    output['fra_avg_pc_foi_avg'] = fra_avg_pc_foi_avg
    output['fra_avg_pc_foi_l95'] = fra_avg_pc_foi_l95
    output['fra_avg_pc_foi_u95'] = fra_avg_pc_foi_u95

    fra_avg_shared_avg = np.mean(fra_avg_shared_sim)
    fra_avg_shared_std = np.std(fra_avg_shared_sim)
    fra_avg_shared_l95 = fra_avg_shared_avg - z * fra_avg_shared_std / np.sqrt(nsims)
    fra_avg_shared_u95 = fra_avg_shared_avg - z * fra_avg_shared_std / np.sqrt(nsims)

    output['fra_avg_shared_avg'] = fra_avg_shared_avg
    output['fra_avg_shared_l95'] = fra_avg_shared_l95
    output['fra_avg_shared_u95'] = fra_avg_shared_u95

    fra_avg_size_avg = np.mean(fra_avg_size_sim)
    fra_avg_size_std = np.std(fra_avg_size_sim)
    fra_avg_size_l95 = fra_avg_size_avg - z * fra_avg_size_std / np.sqrt(nsims)
    fra_avg_size_u95 = fra_avg_size_avg - z * fra_avg_size_std / np.sqrt(nsims)

    output['fra_avg_size_avg'] = fra_avg_size_avg
    output['fra_avg_size_l95'] = fra_avg_size_l95
    output['fra_avg_size_u95'] = fra_avg_size_u95

    fra_cum_shared_avg = np.mean(fra_cum_shared_sim)
    fra_cum_shared_std = np.std(fra_cum_shared_sim)
    fra_cum_shared_l95 = fra_cum_shared_avg - z * fra_cum_shared_std / np.sqrt(nsims)
    fra_cum_shared_u95 = fra_cum_shared_avg - z * fra_cum_shared_std / np.sqrt(nsims)

    output['fra_cum_shared_avg'] = fra_cum_shared_avg
    output['fra_cum_shared_l95'] = fra_cum_shared_l95
    output['fra_cum_shared_u95'] = fra_cum_shared_u95

    fra_cum_size_avg = np.mean(fra_cum_size_sim)
    fra_cum_size_std = np.std(fra_cum_size_sim)
    fra_cum_size_l95 = fra_cum_size_avg - z * fra_cum_size_std / np.sqrt(nsims)
    fra_cum_size_u95 = fra_cum_size_avg - z * fra_cum_size_std / np.sqrt(nsims)

    output['fra_cum_size_avg'] = fra_cum_size_avg
    output['fra_cum_size_l95'] = fra_cum_size_l95
    output['fra_cum_size_u95'] = fra_cum_size_u95

    fra_cum_t_pop_avg = np.mean(fra_cum_t_pop_sim)
    fra_cum_t_pop_std = np.std(fra_cum_t_pop_sim)
    fra_cum_t_pop_l95 = fra_cum_t_pop_avg - z * fra_cum_t_pop_std / np.sqrt(nsims)
    fra_cum_t_pop_u95 = fra_cum_t_pop_avg - z * fra_cum_t_pop_std / np.sqrt(nsims)

    output['fra_cum_t_pop_avg'] = fra_cum_t_pop_avg
    output['fra_cum_t_pop_l95'] = fra_cum_t_pop_l95
    output['fra_cum_t_pop_u95'] = fra_cum_t_pop_u95

    infected_h_avg_per_rho = np.mean(infected_h_per_rho_sim, axis=0)
    infected_h_std_per_rho = np.std(infected_h_avg_per_rho, axis=0)
    infected_h_l95_per_rho = infected_h_avg_per_rho - z * infected_h_std_per_rho / np.sqrt(nsims)
    infected_h_u95_per_rho = infected_h_avg_per_rho + z * infected_h_std_per_rho / np.sqrt(nsims)

    output['infected_h_avg_per_rho'] = infected_h_avg_per_rho
    output['infected_h_l95_per_rho'] = infected_h_l95_per_rho
    output['infected_h_u95_per_rho'] = infected_h_u95_per_rho

    infected_o_avg_per_rho = np.mean(infected_o_per_rho_sim, axis=0)
    infected_o_std_per_rho = np.std(infected_o_avg_per_rho, axis=0)
    infected_o_l95_per_rho = infected_o_avg_per_rho - z * infected_o_std_per_rho / np.sqrt(nsims)
    infected_o_u95_per_rho = infected_o_avg_per_rho + z * infected_o_std_per_rho / np.sqrt(nsims)

    output['infected_o_avg_per_rho'] = infected_o_avg_per_rho
    output['infected_o_l95_per_rho'] = infected_o_l95_per_rho
    output['infected_o_u95_per_rho'] = infected_o_u95_per_rho

    return output

def compute_depr_chapter_panel6_stats(
        agents_per_rho_sim,
        infected_per_rho_sim,
        nevents_eff_per_rho_sim,
        sum_avg_foi_per_rho_sim,
        sum_avg_pc_foi_per_rho_sim,
        sum_avg_shared_per_rho_sim,
        sum_avg_size_per_rho_sim,
        sum_avg_t_pop_per_rho_sim,
        sum_cum_i_pop_per_rho_sim,
        sum_cum_shared_per_rho_sim,
        sum_cum_size_per_rho_sim,
        sum_cum_t_pop_per_rho_sim,
        event_inf_pop_avg_rho_per_rho_sim,
        event_infector_rho_per_rho_sim,
        event_size_from_ipar_per_rho_sim,
        event_size_from_ir_per_rho_sim,
        event_tot_pop_from_ipar_per_rho_sim,
        event_tot_pop_from_ir_per_rho_sim,
        prevalence_cutoff=0.025, 
        ):
    
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    nevents_eff_per_rho_sim = np.array(nevents_eff_per_rho_sim)
    sum_avg_foi_per_rho_sim = np.array(sum_avg_foi_per_rho_sim)
    sum_avg_pc_foi_per_rho_sim = np.array(sum_avg_pc_foi_per_rho_sim)
    sum_avg_shared_per_rho_sim = np.array(sum_avg_shared_per_rho_sim)
    sum_avg_size_per_rho_sim = np.array(sum_avg_size_per_rho_sim)
    sum_avg_t_pop_per_rho_sim = np.array(sum_avg_t_pop_per_rho_sim)
    sum_cum_i_pop_per_rho_sim = np.array(sum_cum_i_pop_per_rho_sim)
    sum_cum_shared_per_rho_sim = np.array(sum_cum_shared_per_rho_sim)
    sum_cum_size_per_rho_sim = np.array(sum_cum_size_per_rho_sim)
    sum_cum_t_pop_per_rho_sim = np.array(sum_cum_t_pop_per_rho_sim)

    event_inf_pop_avg_rho_per_rho_sim = np.array(event_inf_pop_avg_rho_per_rho_sim)
    event_infector_rho_per_rho_sim = np.array(event_infector_rho_per_rho_sim)
    event_size_from_ipar_per_rho_sim = np.array(event_size_from_ipar_per_rho_sim)
    event_size_from_ir_per_rho_sim = np.array(event_size_from_ir_per_rho_sim)
    event_tot_pop_from_ipar_per_rho_sim = np.array(event_tot_pop_from_ipar_per_rho_sim)
    event_tot_pop_from_ir_per_rho_sim = np.array(event_tot_pop_from_ir_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]
    
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    nevents_eff_per_rho_sim = np.delete(nevents_eff_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_foi_per_rho_sim = np.delete(sum_avg_foi_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_pc_foi_per_rho_sim = np.delete(sum_avg_pc_foi_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_shared_per_rho_sim = np.delete(sum_avg_shared_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_size_per_rho_sim = np.delete(sum_avg_size_per_rho_sim, failed_outbreaks, axis=0)
    sum_avg_t_pop_per_rho_sim = np.delete(sum_avg_t_pop_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_i_pop_per_rho_sim = np.delete(sum_cum_i_pop_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_shared_per_rho_sim = np.delete(sum_cum_shared_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_size_per_rho_sim = np.delete(sum_cum_size_per_rho_sim, failed_outbreaks, axis=0)
    sum_cum_t_pop_per_rho_sim = np.delete(sum_cum_t_pop_per_rho_sim, failed_outbreaks, axis=0)

    event_inf_pop_avg_rho_per_rho_sim = np.delete(event_inf_pop_avg_rho_per_rho_sim, failed_outbreaks, axis=0)
    event_infector_rho_per_rho_sim = np.delete(event_infector_rho_per_rho_sim, failed_outbreaks, axis=0)
    event_size_from_ipar_per_rho_sim = np.delete(event_size_from_ipar_per_rho_sim, failed_outbreaks, axis=0)
    event_size_from_ir_per_rho_sim = np.delete(event_size_from_ir_per_rho_sim, failed_outbreaks, axis=0)
    event_tot_pop_from_ipar_per_rho_sim = np.delete(event_tot_pop_from_ipar_per_rho_sim, failed_outbreaks, axis=0)
    event_tot_pop_from_ir_per_rho_sim = np.delete(event_tot_pop_from_ir_per_rho_sim, failed_outbreaks, axis=0)

    agents_sim = np.sum(agents_per_rho_sim, axis=1)
    infected_sim = np.sum(infected_per_rho_sim, axis=1)
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    nevents_eff_sim = np.sum(nevents_eff_per_rho_sim, axis=1)
    fra_avg_foi_sim = np.sum(sum_avg_foi_per_rho_sim, axis=1) / infected_sim
    fra_avg_pc_foi_sim = np.sum(sum_avg_pc_foi_per_rho_sim, axis=1) / infected_sim
    fra_avg_shared_sim = np.sum(sum_avg_shared_per_rho_sim, axis=1) / infected_sim
    fra_avg_size_sim = np.sum(sum_avg_size_per_rho_sim, axis=1) / infected_sim
    fra_avg_t_pop_sim = np.sum(sum_avg_t_pop_per_rho_sim, axis=1) / infected_sim
    fra_cum_i_pop_sim = np.sum(sum_cum_i_pop_per_rho_sim, axis=1) / infected_sim
    fra_cum_shared_sim = np.sum(sum_cum_shared_per_rho_sim, axis=1) / infected_sim
    fra_cum_size_sim = np.sum(sum_cum_size_per_rho_sim, axis=1) / infected_sim

    fra_cum_t_pop_sim = np.sum(sum_cum_t_pop_per_rho_sim, axis=1) / infected_sim
    fra_nevents_eff_sim = np.sum(nevents_eff_per_rho_sim, axis=1) / infected_sim

    fra_ev_inf_pop_avg_rho_sim = np.sum(event_inf_pop_avg_rho_per_rho_sim, axis=1) / infected_sim
    fra_ev_infector_rho_sim = np.sum(event_infector_rho_per_rho_sim, axis=1) / infected_sim
    ipar_nevents_sim = np.sum(event_inf_pop_avg_rho_per_rho_sim, axis=1)
    ir_nevents_sim = np.sum(event_infector_rho_per_rho_sim, axis=1)
    fra_ev_size_from_ipar_sim = np.sum(event_size_from_ipar_per_rho_sim, axis=1) / infected_sim
    fra_ev_size_from_ir_sim = np.sum(event_size_from_ir_per_rho_sim, axis=1) / infected_sim
    ir_size_sim = np.sum(event_size_from_ir_per_rho_sim, axis=1)
    fra_ev_tot_pop_from_ipar_sim = np.sum(event_tot_pop_from_ipar_per_rho_sim, axis=1) / infected_sim
    fra_ev_tot_pop_from_ir_sim = np.sum(event_tot_pop_from_ir_per_rho_sim, axis=1) / infected_sim

    fra_avg_foi_per_rho_sim = sum_avg_foi_per_rho_sim / infected_per_rho_sim
    fra_avg_pc_foi_per_rho_sim = sum_avg_pc_foi_per_rho_sim / infected_per_rho_sim
    fra_avg_shared_per_rho_sim = sum_avg_shared_per_rho_sim / infected_per_rho_sim
    fra_avg_size_per_rho_sim = sum_avg_size_per_rho_sim / infected_per_rho_sim
    fra_avg_t_pop_per_rho_sim = sum_avg_t_pop_per_rho_sim / infected_per_rho_sim
    fra_cum_shared_per_rho_sim = sum_cum_shared_per_rho_sim / infected_per_rho_sim
    fra_cum_size_per_rho_sim = sum_cum_size_per_rho_sim / infected_per_rho_sim
    fra_cum_t_pop_per_rho_sim = sum_cum_t_pop_per_rho_sim / infected_per_rho_sim
    fra_infected_per_rho_sim = infected_per_rho_sim / agents_per_rho_sim
    fra_nevents_eff_per_rho_sim = nevents_eff_per_rho_sim / infected_per_rho_sim

    #fra_ev_inf_pop_avg_rho_per_rho_sim = event_inf_pop_avg_rho_per_rho_sim / 1.0
    #fra_ev_infector_rho_per_rho_sim = event_infector_rho_per_rho_sim / 1.0

    fra_nevents_eff_per_rho_sim = (nevents_eff_per_rho_sim / nevents_eff_sim[:, np.newaxis]) / (infected_per_rho_sim / infected_sim[:, np.newaxis])
    fra_ev_inf_pop_avg_rho_per_rho_sim = (event_inf_pop_avg_rho_per_rho_sim / ipar_nevents_sim[:, np.newaxis]) / (infected_per_rho_sim / infected_sim[:, np.newaxis])
    fra_ev_infector_rho_per_rho_sim = (event_infector_rho_per_rho_sim / ir_nevents_sim[:, np.newaxis]) / (infected_per_rho_sim / infected_sim[:, np.newaxis])

    #fra_ev_size_from_ipar_per_rho_sim = (event_size_from_ipar_per_rho_sim) / (infected_per_rho_sim)
    fra_ev_size_from_ipar_per_rho_sim = (event_size_from_ipar_per_rho_sim) / (event_inf_pop_avg_rho_per_rho_sim)
    fra_ev_size_from_ir_per_rho_sim = (event_size_from_ir_per_rho_sim / ir_size_sim[:, np.newaxis]) / (infected_per_rho_sim / infected_sim[:, np.newaxis])
    #fra_ev_size_from_ir_per_rho_sim = (event_size_from_ir_per_rho_sim) / (event_infector_rho_per_rho_sim)
    fra_cum_size_per_rho_sim = (sum_cum_size_per_rho_sim / np.sum(sum_cum_size_per_rho_sim, axis=1)[:, np.newaxis]) / (infected_per_rho_sim / infected_sim[:, np.newaxis])

    fra_ev_tot_pop_from_ipar_per_rho_sim = (event_tot_pop_from_ir_per_rho_sim) / (event_inf_pop_avg_rho_per_rho_sim)
    fra_ev_tot_pop_from_ir_per_rho_sim = (event_tot_pop_from_ipar_per_rho_sim) / (event_infector_rho_per_rho_sim)

    output = {}

    z = 1.96

    nsims = len(fra_avg_foi_per_rho_sim)

    agents_avg_per_rho = np.mean(agents_per_rho_sim, axis=0)
    agents_std_per_rho = np.std(agents_avg_per_rho, axis=0)
    agents_l95_per_rho = agents_avg_per_rho - z * agents_std_per_rho / np.sqrt(nsims)
    agents_u95_per_rho = agents_avg_per_rho + z * agents_std_per_rho / np.sqrt(nsims)

    output['agents_avg_per_rho'] = agents_avg_per_rho
    output['agents_l95_per_rho'] = agents_l95_per_rho
    output['agents_u95_per_rho'] = agents_u95_per_rho

    infected_avg_per_rho = np.mean(infected_per_rho_sim, axis=0)
    infected_std_per_rho = np.std(infected_per_rho_sim, axis=0)
    infected_l95_per_rho = infected_avg_per_rho - z * infected_std_per_rho / np.sqrt(nsims)
    infected_u95_per_rho = infected_avg_per_rho + z * infected_std_per_rho / np.sqrt(nsims)

    output['infected_avg_per_rho'] = infected_avg_per_rho
    output['infected_l95_per_rho'] = infected_l95_per_rho
    output['infected_u95_per_rho'] = infected_u95_per_rho

    nevents_eff_avg_per_rho = np.mean(nevents_eff_per_rho_sim, axis=0)
    nevents_eff_std_per_rho = np.std(nevents_eff_per_rho_sim, axis=0)
    nevents_eff_l95_per_rho = nevents_eff_avg_per_rho - z * nevents_eff_std_per_rho / np.sqrt(nsims)
    nevents_eff_u95_per_rho = nevents_eff_avg_per_rho + z * nevents_eff_std_per_rho / np.sqrt(nsims)

    output['nevents_eff_avg_per_rho'] = nevents_eff_avg_per_rho
    output['nevents_eff_l95_per_rho'] = nevents_eff_l95_per_rho
    output['nevents_eff_u95_per_rho'] = nevents_eff_u95_per_rho

    sum_avg_shared_avg_per_rho = np.mean(sum_avg_shared_per_rho_sim, axis=0)
    sum_avg_shared_std_per_rho = np.std(sum_avg_shared_per_rho_sim, axis=0)
    sum_avg_shared_l95_per_rho = sum_avg_shared_avg_per_rho - z * sum_avg_shared_std_per_rho / np.sqrt(nsims)
    sum_avg_shared_u95_per_rho = sum_avg_shared_avg_per_rho + z * sum_avg_shared_std_per_rho / np.sqrt(nsims)

    output['sum_avg_shared_avg_per_rho'] = sum_avg_shared_avg_per_rho
    output['sum_avg_shared_l95_per_rho'] = sum_avg_shared_l95_per_rho
    output['sum_avg_shared_u95_per_rho'] = sum_avg_shared_u95_per_rho

    sum_avg_size_avg_per_rho = np.mean(sum_avg_size_per_rho_sim, axis=0)
    sum_avg_size_std_per_rho = np.std(sum_avg_size_per_rho_sim, axis=0)
    sum_avg_size_l95_per_rho = sum_avg_size_avg_per_rho - z * sum_avg_size_std_per_rho / np.sqrt(nsims)
    sum_avg_size_u95_per_rho = sum_avg_size_avg_per_rho + z * sum_avg_size_std_per_rho / np.sqrt(nsims)

    output['sum_avg_size_avg_per_rho'] = sum_avg_size_avg_per_rho
    output['sum_avg_size_l95_per_rho'] = sum_avg_size_l95_per_rho
    output['sum_avg_size_u95_per_rho'] = sum_avg_size_u95_per_rho

    sum_avg_t_pop_avg_per_rho = np.mean(sum_avg_t_pop_per_rho_sim, axis=0)
    sum_avg_t_pop_std_per_rho = np.std(sum_avg_t_pop_per_rho_sim, axis=0)
    sum_avg_t_pop_l95_per_rho = sum_avg_t_pop_avg_per_rho - z * sum_avg_t_pop_std_per_rho / np.sqrt(nsims)
    sum_avg_t_pop_u95_per_rho = sum_avg_t_pop_avg_per_rho + z * sum_avg_t_pop_std_per_rho / np.sqrt(nsims)

    output['sum_avg_t_pop_avg_per_rho'] = sum_avg_t_pop_avg_per_rho
    output['sum_avg_t_pop_l95_per_rho'] = sum_avg_t_pop_l95_per_rho
    output['sum_avg_t_pop_u95_per_rho'] = sum_avg_t_pop_u95_per_rho

    sum_cum_shared_avg_per_rho = np.mean(sum_cum_shared_per_rho_sim, axis=0)
    sum_cum_shared_std_per_rho = np.std(sum_cum_shared_per_rho_sim, axis=0)
    sum_cum_shared_l95_per_rho = sum_cum_shared_avg_per_rho - z * sum_cum_shared_std_per_rho / np.sqrt(nsims)
    sum_cum_shared_u95_per_rho = sum_cum_shared_avg_per_rho + z * sum_cum_shared_std_per_rho / np.sqrt(nsims)

    output['sum_cum_shared_avg_per_rho'] = sum_cum_shared_avg_per_rho
    output['sum_cum_shared_l95_per_rho'] = sum_cum_shared_l95_per_rho
    output['sum_cum_shared_u95_per_rho'] = sum_cum_shared_u95_per_rho

    sum_cum_size_avg_per_rho = np.mean(sum_cum_size_per_rho_sim, axis=0)
    sum_cum_size_std_per_rho = np.std(sum_cum_size_per_rho_sim, axis=0)
    sum_cum_size_l95_per_rho = sum_cum_size_avg_per_rho - z * sum_cum_size_std_per_rho / np.sqrt(nsims)
    sum_cum_size_u95_per_rho = sum_cum_size_avg_per_rho + z * sum_cum_size_std_per_rho / np.sqrt(nsims)

    output['sum_cum_size_avg_per_rho'] = sum_cum_size_avg_per_rho
    output['sum_cum_size_l95_per_rho'] = sum_cum_size_l95_per_rho
    output['sum_cum_size_u95_per_rho'] = sum_cum_size_u95_per_rho

    sum_cum_t_pop_avg_per_rho = np.mean(sum_cum_t_pop_per_rho_sim, axis=0)
    sum_cum_t_pop_std_per_rho = np.std(sum_cum_t_pop_per_rho_sim, axis=0)
    sum_cum_t_pop_l95_per_rho = sum_cum_t_pop_avg_per_rho - z * sum_cum_t_pop_std_per_rho / np.sqrt(nsims)
    sum_cum_t_pop_u95_per_rho = sum_cum_t_pop_avg_per_rho + z * sum_cum_t_pop_std_per_rho / np.sqrt(nsims)

    output['sum_cum_t_pop_avg_per_rho'] = sum_cum_t_pop_avg_per_rho
    output['sum_cum_t_pop_l95_per_rho'] = sum_cum_t_pop_l95_per_rho
    output['sum_cum_t_pop_u95_per_rho'] = sum_cum_t_pop_u95_per_rho

    fra_avg_foi_avg_per_rho = np.mean(fra_avg_foi_per_rho_sim, axis=0)
    fra_avg_foi_std_per_rho = np.std(fra_avg_foi_per_rho_sim, axis=0)
    fra_avg_foi_l95_per_rho = fra_avg_foi_avg_per_rho - z * fra_avg_foi_std_per_rho / np.sqrt(nsims)
    fra_avg_foi_u95_per_rho = fra_avg_foi_avg_per_rho + z * fra_avg_foi_std_per_rho / np.sqrt(nsims)

    output['fra_avg_foi_avg_per_rho'] = fra_avg_foi_avg_per_rho
    output['fra_avg_foi_l95_per_rho'] = fra_avg_foi_l95_per_rho
    output['fra_avg_foi_u95_per_rho'] = fra_avg_foi_u95_per_rho

    fra_avg_pc_foi_avg_per_rho = np.mean(fra_avg_pc_foi_per_rho_sim, axis=0)
    fra_avg_pc_foi_std_per_rho = np.std(fra_avg_pc_foi_per_rho_sim, axis=0)
    fra_avg_pc_foi_l95_per_rho = fra_avg_pc_foi_avg_per_rho - z * fra_avg_pc_foi_std_per_rho / np.sqrt(nsims)
    fra_avg_pc_foi_u95_per_rho = fra_avg_pc_foi_avg_per_rho + z * fra_avg_pc_foi_std_per_rho / np.sqrt(nsims)

    output['fra_avg_pc_foi_avg_per_rho'] = fra_avg_pc_foi_avg_per_rho
    output['fra_avg_pc_foi_l95_per_rho'] = fra_avg_pc_foi_l95_per_rho
    output['fra_avg_pc_foi_u95_per_rho'] = fra_avg_pc_foi_u95_per_rho

    fra_avg_shared_avg_per_rho = np.mean(fra_avg_shared_per_rho_sim, axis=0)
    fra_avg_shared_std_per_rho = np.std(fra_avg_shared_per_rho_sim, axis=0)
    fra_avg_shared_l95_per_rho = fra_avg_shared_avg_per_rho - z * fra_avg_shared_std_per_rho / np.sqrt(nsims)
    fra_avg_shared_u95_per_rho = fra_avg_shared_avg_per_rho + z * fra_avg_shared_std_per_rho / np.sqrt(nsims)

    output['fra_avg_shared_avg_per_rho'] = fra_avg_shared_avg_per_rho
    output['fra_avg_shared_l95_per_rho'] = fra_avg_shared_l95_per_rho
    output['fra_avg_shared_u95_per_rho'] = fra_avg_shared_u95_per_rho

    fra_avg_size_avg_per_rho = np.mean(fra_avg_size_per_rho_sim, axis=0)
    fra_avg_size_std_per_rho = np.std(fra_avg_size_per_rho_sim, axis=0)
    fra_avg_size_l95_per_rho = fra_avg_size_avg_per_rho - z * fra_avg_size_std_per_rho / np.sqrt(nsims)
    fra_avg_size_u95_per_rho = fra_avg_size_avg_per_rho + z * fra_avg_size_std_per_rho / np.sqrt(nsims)

    output['fra_avg_size_avg_per_rho'] = fra_avg_size_avg_per_rho
    output['fra_avg_size_l95_per_rho'] = fra_avg_size_l95_per_rho
    output['fra_avg_size_u95_per_rho'] = fra_avg_size_u95_per_rho

    fra_avg_t_pop_avg_per_rho = np.mean(fra_avg_t_pop_per_rho_sim, axis=0)
    fra_avg_t_pop_std_per_rho = np.std(fra_avg_t_pop_per_rho_sim, axis=0)
    fra_avg_t_pop_l95_per_rho = fra_avg_t_pop_avg_per_rho - z * fra_avg_t_pop_std_per_rho / np.sqrt(nsims)
    fra_avg_t_pop_u95_per_rho = fra_avg_t_pop_avg_per_rho + z * fra_avg_t_pop_std_per_rho / np.sqrt(nsims)

    output['fra_avg_t_pop_avg_per_rho'] = fra_avg_t_pop_avg_per_rho
    output['fra_avg_t_pop_l95_per_rho'] = fra_avg_t_pop_l95_per_rho
    output['fra_avg_t_pop_u95_per_rho'] = fra_avg_t_pop_u95_per_rho

    fra_cum_shared_avg_per_rho = np.mean(fra_cum_shared_per_rho_sim, axis=0)
    fra_cum_shared_std_per_rho = np.std(fra_cum_shared_per_rho_sim, axis=0)
    fra_cum_shared_l95_per_rho = fra_cum_shared_avg_per_rho - z * fra_cum_shared_std_per_rho / np.sqrt(nsims)
    fra_cum_shared_u95_per_rho = fra_cum_shared_avg_per_rho + z * fra_cum_shared_std_per_rho / np.sqrt(nsims)

    output['fra_cum_shared_avg_per_rho'] = fra_cum_shared_avg_per_rho
    output['fra_cum_shared_l95_per_rho'] = fra_cum_shared_l95_per_rho
    output['fra_cum_shared_u95_per_rho'] = fra_cum_shared_u95_per_rho

    fra_cum_size_avg_per_rho = np.mean(fra_cum_size_per_rho_sim, axis=0)
    fra_cum_size_std_per_rho = np.std(fra_cum_size_per_rho_sim, axis=0)
    fra_cum_size_l95_per_rho = fra_cum_size_avg_per_rho - z * fra_cum_size_std_per_rho / np.sqrt(nsims)
    fra_cum_size_u95_per_rho = fra_cum_size_avg_per_rho + z * fra_cum_size_std_per_rho / np.sqrt(nsims)

    output['fra_cum_size_avg_per_rho'] = fra_cum_size_avg_per_rho
    output['fra_cum_size_l95_per_rho'] = fra_cum_size_l95_per_rho
    output['fra_cum_size_u95_per_rho'] = fra_cum_size_u95_per_rho

    fra_cum_t_pop_avg_per_rho = np.mean(fra_cum_t_pop_per_rho_sim, axis=0)
    fra_cum_t_pop_std_per_rho = np.std(fra_cum_t_pop_per_rho_sim, axis=0)
    fra_cum_t_pop_l95_per_rho = fra_cum_t_pop_avg_per_rho - z * fra_cum_t_pop_std_per_rho / np.sqrt(nsims)
    fra_cum_t_pop_u95_per_rho = fra_cum_t_pop_avg_per_rho + z * fra_cum_t_pop_std_per_rho / np.sqrt(nsims)

    output['fra_cum_t_pop_avg_per_rho'] = fra_cum_t_pop_avg_per_rho
    output['fra_cum_t_pop_l95_per_rho'] = fra_cum_t_pop_l95_per_rho
    output['fra_cum_t_pop_u95_per_rho'] = fra_cum_t_pop_u95_per_rho

    fra_infected_avg_per_rho = np.mean(fra_infected_per_rho_sim, axis=0)
    fra_infected_std_per_rho = np.std(fra_infected_per_rho_sim, axis=0)
    fra_infected_l95_per_rho = fra_infected_avg_per_rho - z * fra_infected_std_per_rho / np.sqrt(nsims)
    fra_infected_u95_per_rho = fra_infected_avg_per_rho + z * fra_infected_std_per_rho / np.sqrt(nsims)

    output['fra_infected_avg_per_rho'] = fra_infected_avg_per_rho
    output['fra_infected_l95_per_rho'] = fra_infected_l95_per_rho
    output['fra_infected_u95_per_rho'] = fra_infected_u95_per_rho

    fra_nevents_eff_avg_per_rho = np.mean(fra_nevents_eff_per_rho_sim, axis=0)
    fra_nevents_eff_std_per_rho = np.std(fra_nevents_eff_per_rho_sim, axis=0)
    fra_nevents_eff_l95_per_rho = fra_nevents_eff_avg_per_rho - z * fra_nevents_eff_std_per_rho / np.sqrt(nsims)
    fra_nevents_eff_u95_per_rho = fra_nevents_eff_avg_per_rho + z * fra_nevents_eff_std_per_rho / np.sqrt(nsims)

    output['fra_nevents_eff_avg_per_rho'] = fra_nevents_eff_avg_per_rho
    output['fra_nevents_eff_l95_per_rho'] = fra_nevents_eff_l95_per_rho
    output['fra_nevents_eff_u95_per_rho'] = fra_nevents_eff_u95_per_rho

    fra_ev_inf_pop_avg_rho_avg_per_rho = np.mean(fra_ev_inf_pop_avg_rho_per_rho_sim, axis=0)
    fra_ev_inf_pop_avg_rho_std_per_rho = np.std(fra_ev_inf_pop_avg_rho_per_rho_sim, axis=0)
    fra_ev_inf_pop_avg_rho_l95_per_rho = fra_ev_inf_pop_avg_rho_avg_per_rho - z * fra_ev_inf_pop_avg_rho_std_per_rho / np.sqrt(nsims)
    fra_ev_inf_pop_avg_rho_u95_per_rho = fra_ev_inf_pop_avg_rho_avg_per_rho + z * fra_ev_inf_pop_avg_rho_std_per_rho / np.sqrt(nsims)

    output['fra_ev_inf_pop_avg_rho_avg_per_rho'] = fra_ev_inf_pop_avg_rho_avg_per_rho
    output['fra_ev_inf_pop_avg_rho_l95_per_rho'] = fra_ev_inf_pop_avg_rho_l95_per_rho
    output['fra_ev_inf_pop_avg_rho_u95_per_rho'] = fra_ev_inf_pop_avg_rho_u95_per_rho

    fra_ev_infector_rho_avg_per_rho = np.mean(fra_ev_infector_rho_per_rho_sim, axis=0)
    fra_ev_infector_rho_std_per_rho = np.std(fra_ev_infector_rho_per_rho_sim, axis=0)
    fra_ev_infector_rho_l95_per_rho = fra_ev_infector_rho_avg_per_rho - z * fra_ev_infector_rho_std_per_rho / np.sqrt(nsims)
    fra_ev_infector_rho_u95_per_rho = fra_ev_infector_rho_avg_per_rho + z * fra_ev_infector_rho_std_per_rho / np.sqrt(nsims)

    output['fra_ev_infector_rho_avg_per_rho'] = fra_ev_infector_rho_avg_per_rho
    output['fra_ev_infector_rho_l95_per_rho'] = fra_ev_infector_rho_l95_per_rho
    output['fra_ev_infector_rho_u95_per_rho'] = fra_ev_infector_rho_u95_per_rho

    fra_ev_size_from_ipar_avg_per_rho = np.mean(fra_ev_size_from_ipar_per_rho_sim, axis=0)
    fra_ev_size_from_ipar_std_per_rho = np.std(fra_ev_size_from_ipar_per_rho_sim, axis=0)
    fra_ev_size_from_ipar_l95_per_rho = fra_ev_size_from_ipar_avg_per_rho - z * fra_ev_size_from_ipar_std_per_rho / np.sqrt(nsims)
    fra_ev_size_from_ipar_u95_per_rho = fra_ev_size_from_ipar_avg_per_rho + z * fra_ev_size_from_ipar_std_per_rho / np.sqrt(nsims)

    output['fra_ev_size_from_ipar_avg_per_rho'] = fra_ev_size_from_ipar_avg_per_rho
    output['fra_ev_size_from_ipar_l95_per_rho'] = fra_ev_size_from_ipar_l95_per_rho
    output['fra_ev_size_from_ipar_u95_per_rho'] = fra_ev_size_from_ipar_u95_per_rho

    fra_ev_size_from_ir_avg_per_rho = np.mean(fra_ev_size_from_ir_per_rho_sim, axis=0)
    fra_ev_size_from_ir_std_per_rho = np.std(fra_ev_size_from_ir_per_rho_sim, axis=0)
    fra_ev_size_from_ir_l95_per_rho = fra_ev_size_from_ir_avg_per_rho - z * fra_ev_size_from_ir_std_per_rho / np.sqrt(nsims)
    fra_ev_size_from_ir_u95_per_rho = fra_ev_size_from_ir_avg_per_rho + z * fra_ev_size_from_ir_std_per_rho / np.sqrt(nsims)

    output['fra_ev_size_from_ir_avg_per_rho'] = fra_ev_size_from_ir_avg_per_rho
    output['fra_ev_size_from_ir_l95_per_rho'] = fra_ev_size_from_ir_l95_per_rho
    output['fra_ev_size_from_ir_u95_per_rho'] = fra_ev_size_from_ir_u95_per_rho

    fra_avg_foi_avg = np.mean(fra_avg_foi_sim)
    fra_avg_foi_std = np.std(fra_avg_foi_sim)
    fra_avg_foi_l95 = fra_avg_foi_avg - z * fra_avg_foi_std / np.sqrt(nsims)
    fra_avg_foi_u95 = fra_avg_foi_avg - z * fra_avg_foi_std / np.sqrt(nsims)
    
    output['fra_avg_foi_avg'] = fra_avg_foi_avg
    output['fra_avg_foi_l95'] = fra_avg_foi_l95
    output['fra_avg_foi_u95'] = fra_avg_foi_u95

    fra_avg_pc_foi_avg = np.mean(fra_avg_pc_foi_sim)
    fra_avg_pc_foi_std = np.std(fra_avg_pc_foi_sim)
    fra_avg_pc_foi_l95 = fra_avg_pc_foi_avg - z * fra_avg_pc_foi_std / np.sqrt(nsims)
    fra_avg_pc_foi_u95 = fra_avg_pc_foi_avg - z * fra_avg_pc_foi_std / np.sqrt(nsims)

    output['fra_avg_pc_foi_avg'] = fra_avg_pc_foi_avg
    output['fra_avg_pc_foi_l95'] = fra_avg_pc_foi_l95
    output['fra_avg_pc_foi_u95'] = fra_avg_pc_foi_u95

    fra_avg_shared_avg = np.mean(fra_avg_shared_sim)
    fra_avg_shared_std = np.std(fra_avg_shared_sim)
    fra_avg_shared_l95 = fra_avg_shared_avg - z * fra_avg_shared_std / np.sqrt(nsims)
    fra_avg_shared_u95 = fra_avg_shared_avg - z * fra_avg_shared_std / np.sqrt(nsims)

    output['fra_avg_shared_avg'] = fra_avg_shared_avg
    output['fra_avg_shared_l95'] = fra_avg_shared_l95
    output['fra_avg_shared_u95'] = fra_avg_shared_u95

    fra_avg_size_avg = np.mean(fra_avg_size_sim)
    fra_avg_size_std = np.std(fra_avg_size_sim)
    fra_avg_size_l95 = fra_avg_size_avg - z * fra_avg_size_std / np.sqrt(nsims)
    fra_avg_size_u95 = fra_avg_size_avg - z * fra_avg_size_std / np.sqrt(nsims)

    output['fra_avg_size_avg'] = fra_avg_size_avg
    output['fra_avg_size_l95'] = fra_avg_size_l95
    output['fra_avg_size_u95'] = fra_avg_size_u95

    fra_avg_t_pop_avg = np.mean(fra_avg_t_pop_sim)
    fra_avg_t_pop_std = np.std(fra_avg_t_pop_sim)
    fra_avg_t_pop_l95 = fra_avg_t_pop_avg - z * fra_avg_t_pop_std / np.sqrt(nsims)
    fra_avg_t_pop_u95 = fra_avg_t_pop_avg - z * fra_avg_t_pop_std / np.sqrt(nsims)

    output['fra_avg_t_pop_avg'] = fra_avg_t_pop_avg
    output['fra_avg_t_pop_l95'] = fra_avg_t_pop_l95
    output['fra_avg_t_pop_u95'] = fra_avg_t_pop_u95

    fra_cum_shared_avg = np.mean(fra_cum_shared_sim)
    fra_cum_shared_std = np.std(fra_cum_shared_sim)
    fra_cum_shared_l95 = fra_cum_shared_avg - z * fra_cum_shared_std / np.sqrt(nsims)
    fra_cum_shared_u95 = fra_cum_shared_avg - z * fra_cum_shared_std / np.sqrt(nsims)
    
    output['fra_cum_shared_avg'] = fra_cum_shared_avg
    output['fra_cum_shared_l95'] = fra_cum_shared_l95
    output['fra_cum_shared_u95'] = fra_cum_shared_u95

    fra_cum_size_avg = np.mean(fra_cum_size_sim)
    fra_cum_size_std = np.std(fra_cum_size_sim)
    fra_cum_size_l95 = fra_cum_size_avg - z * fra_cum_size_std / np.sqrt(nsims)
    fra_cum_size_u95 = fra_cum_size_avg - z * fra_cum_size_std / np.sqrt(nsims)

    output['fra_cum_size_avg'] = fra_cum_size_avg
    output['fra_cum_size_l95'] = fra_cum_size_l95
    output['fra_cum_size_u95'] = fra_cum_size_u95

    fra_cum_t_pop_avg = np.mean(fra_cum_t_pop_sim)
    fra_cum_t_pop_std = np.std(fra_cum_t_pop_sim)
    fra_cum_t_pop_l95 = fra_cum_t_pop_avg - z * fra_cum_t_pop_std / np.sqrt(nsims)
    fra_cum_t_pop_u95 = fra_cum_t_pop_avg - z * fra_cum_t_pop_std / np.sqrt(nsims)

    output['fra_cum_t_pop_avg'] = fra_cum_t_pop_avg
    output['fra_cum_t_pop_l95'] = fra_cum_t_pop_l95
    output['fra_cum_t_pop_u95'] = fra_cum_t_pop_u95

    fra_ev_inf_pop_avg_rho_avg = np.mean(fra_ev_inf_pop_avg_rho_sim)
    fra_ev_inf_pop_avg_rho_std = np.std(fra_ev_inf_pop_avg_rho_sim)
    fra_ev_inf_pop_avg_rho_l95 = fra_ev_inf_pop_avg_rho_avg - z * fra_ev_inf_pop_avg_rho_std / np.sqrt(nsims) 
    fra_ev_inf_pop_avg_rho_u95 = fra_ev_inf_pop_avg_rho_avg + z * fra_ev_inf_pop_avg_rho_std / np.sqrt(nsims)

    output['fra_ev_inf_pop_avg_rho_avg'] = fra_ev_inf_pop_avg_rho_avg
    output['fra_ev_inf_pop_avg_rho_l95'] = fra_ev_inf_pop_avg_rho_l95
    output['fra_ev_inf_pop_avg_rho_u95'] = fra_ev_inf_pop_avg_rho_u95

    fra_ev_infector_rho_avg = np.mean(fra_ev_infector_rho_sim)
    fra_ev_infector_rho_std = np.std(fra_ev_infector_rho_sim)
    fra_ev_infector_rho_l95 = fra_ev_infector_rho_avg - z * fra_ev_infector_rho_std / np.sqrt(nsims) 
    fra_ev_infector_rho_u95 = fra_ev_infector_rho_avg + z * fra_ev_infector_rho_std / np.sqrt(nsims)

    output['fra_ev_infector_rho_avg'] = fra_ev_infector_rho_avg
    output['fra_ev_infector_rho_l95'] = fra_ev_infector_rho_l95
    output['fra_ev_infector_rho_u95'] = fra_ev_infector_rho_u95

    return output

def load_depr_chapter_panel_event_data(fullname):
    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict = { }

    attractiveness = []
    inf_pop_avg_rho = []
    location = []
    size = []
    time = []
    tot_pop = []
    for sim in range(len(edig_dict['event_output'])):
        attractiveness.append(edig_dict['event_output'][sim]['attractiveness'])
        inf_pop_avg_rho.append(edig_dict['event_output'][sim]['inf_pop_avg_rho'])
        location.append(edig_dict['event_output'][sim]['location'])
        size.append(edig_dict['event_output'][sim]['size'])
        time.append(edig_dict['event_output'][sim]['time'])
        tot_pop.append(edig_dict['event_output'][sim]['tot_pop'])

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']

    output_dict['attractiveness'] = attractiveness
    output_dict['inf_pop_avg_rho'] = inf_pop_avg_rho
    output_dict['location'] = location
    output_dict['size'] = size
    output_dict['time'] = time
    output_dict['tot_pop'] = tot_pop

    return output_dict

def extend_depr_chapter_panel_events_results(
        out_sim_data, 
        agents_per_rho_sim,
        infected_per_rho_sim,
        event_attractiveness_sim,
        event_inf_pop_avg_rho_sim,
        event_location_sim,
        event_size_sim,
        event_time_sim,
        event_tot_pop_sim,
    ):

    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])

    event_attractiveness_sim.extend(out_sim_data['attractiveness'])
    event_inf_pop_avg_rho_sim.extend(out_sim_data['inf_pop_avg_rho'])
    event_location_sim.extend(out_sim_data['location'])
    event_size_sim.extend(out_sim_data['size'])
    event_time_sim.extend(out_sim_data['time'])
    event_tot_pop_sim.extend(out_sim_data['tot_pop'])

def classify_events_by_location(attractiveness_se, inf_pop_avg_rho_se, location_se, size_se, time_se, tot_pop_se):
    # Create a dictionary to store events by simulation index and location
    events_by_simulation_and_location = {}

    # Iterate through simulations
    for sim_idx in range(len(location_se)):
        # Create a sub-dictionary for the current simulation index
        events_by_location = {}

        # Iterate through events in the current simulation
        for event_idx, location in enumerate(location_se[sim_idx]):
            attractiveness = attractiveness_se[sim_idx][event_idx]
            inf_rho = inf_pop_avg_rho_se[sim_idx][event_idx]
            size = size_se[sim_idx][event_idx]
            time = time_se[sim_idx][event_idx]
            top_pop = tot_pop_se[sim_idx][event_idx]

            # Create a unique identifier for each event (e.g., event_index)
            event_id = f"{event_idx}"

            # Check if the location is already in the sub-dictionary
            if location in events_by_location:
                # Append values to existing lists
                events_by_location[location]['attractiveness'] = attractiveness
                events_by_location[location]['inf_rho'].append(inf_rho)
                events_by_location[location]['size'].append(size)
                events_by_location[location]['time'].append(time)
                events_by_location[location]['top_pop'].append(top_pop)
                # Append more attributes as needed
            else:
                # Create lists for each attribute
                events_by_location[location] = {
                    'attractiveness': attractiveness,
                    'inf_rho': [inf_rho],
                    'time': [time],
                    'size': [size],
                    'top_pop': [top_pop],
                    # Add more attributes as needed
                }

        # Store the sub-dictionary for the current simulation index
        events_by_simulation_and_location[sim_idx] = events_by_location

    return events_by_simulation_and_location

def compute_additional_statistics(events_by_simulation_and_location):
    # Iterate through simulations and locations
    for sim_idx, location_data in events_by_simulation_and_location.items():
        for location, attributes in location_data.items():
            # Calculate average size
            size_list = attributes['size']
            avg_size = sum(size_list) / len(size_list)
            attributes['size_avg'] = avg_size

            # Calculate inter-event time distribution
            time_list = attributes['time']
            time_list.sort()  # Sort event times
            inter_time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
            attributes['inter_time'] = inter_time_list

            # Calculate average inter-event time
            if len(inter_time_list) != 0:
                avg_inter_time = sum(inter_time_list) / len(inter_time_list)
            else:
                avg_inter_time = np.nan
            attributes['inter_time_avg'] = avg_inter_time

            # Calculate average event time
            if len(time_list) != 0:
                avg_time = sum(time_list) / len(time_list)
            else:
                avg_time = np.nan
            attributes['time_avg'] = avg_time

            # Calculate time window
            time_end = np.nanmax(time_list)
            time_inv = np.nanmin(time_list)
            attributes['time_inv'] = time_inv
            rho_inv = attributes['inf_rho'][0]
            attributes['rho_inv'] = rho_inv
            time_window = time_end - time_inv + 1
            if math.isnan(time_window):
                time_window = np.nan
            attributes['time_window'] = time_window

            # Get the size of the 'size' list and add it as 'number_of_events'
            number_of_events = len(attributes['size'])
            attributes['number_of_events'] = number_of_events

            # Calculate event rate
            if time_window == 0:
                event_rate = number_of_events / 1.0
            else:
                event_rate = number_of_events / time_window
            
            attributes['event_rate'] = event_rate

            # Calculate average infected population rho
            inf_rho_list = attributes['inf_rho']
            avg_inf_rho = sum(inf_rho_list) / len(inf_rho_list)
            attributes['inf_rho_avg'] = avg_inf_rho

def load_depr_chapter_panel_locs_data(fullname):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['inf_rho_dist_per_loc'] = edig_dict['infected_rho_per_loc']
    output_dict['inf_rho_h_dist_per_loc'] = edig_dict['infected_rho_h_per_loc']
    output_dict['inf_rho_o_dist_per_loc'] = edig_dict['infected_rho_o_per_loc']

    return output_dict

def extend_depr_chapter_panel_locs_results(
        out_sim_data, 
        agents_per_rho_sim,
        infected_per_rho_sim,
        inf_rho_dist_per_loc_sim,
        inf_rho_h_dist_per_loc_sim,
        inf_rho_o_dist_per_loc_sim,
    ):

    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    inf_rho_dist_per_loc_sim.extend(out_sim_data['inf_rho_dist_per_loc'])
    inf_rho_h_dist_per_loc_sim.extend(out_sim_data['inf_rho_h_dist_per_loc'])
    inf_rho_o_dist_per_loc_sim.extend(out_sim_data['inf_rho_o_dist_per_loc'])

def compute_depr_chapter_panel_locs(
        agents_per_rho_sim,
        infected_per_rho_sim,
        inf_rho_dist_per_loc_sim,
        inf_rho_h_dist_per_loc_sim,
        inf_rho_o_dist_per_loc_sim,
        prevalence_cutoff=0.025,
):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)

    infected_h_per_loc_sim = []
    infected_o_per_loc_sim = []

    inf_rho_dist_per_loc_sim = [sim for i, sim in enumerate(inf_rho_dist_per_loc_sim) if i not in failed_outbreaks]
    inf_rho_h_dist_per_loc_sim = [sim for i, sim in enumerate(inf_rho_h_dist_per_loc_sim) if i not in failed_outbreaks]
    inf_rho_o_dist_per_loc_sim = [sim for i, sim in enumerate(inf_rho_o_dist_per_loc_sim) if i not in failed_outbreaks]
    
    nsims = len(inf_rho_dist_per_loc_sim)
    nlocs = len(inf_rho_dist_per_loc_sim[0])  # Assuming all inner lists have the same size

    inf_rho_dist_per_loc = [[] for _ in range(nlocs)]
    infected_per_loc_sim = np.zeros((nsims, nlocs))

    for sim_idx in range(len(inf_rho_dist_per_loc_sim)):
        for loc_idx in range(nlocs):
            inf_rho_values = inf_rho_dist_per_loc_sim[sim_idx][loc_idx]
            inf_rho_dist_per_loc[loc_idx].extend(inf_rho_values)
            infected_per_loc_sim[sim_idx][loc_idx] = len(inf_rho_values)

    inf_rho_avg_per_loc = np.array([np.nanmean(sublist) for sublist in inf_rho_dist_per_loc])
    inf_rho_std_per_loc = np.array([np.nanstd(sublist) for sublist in inf_rho_dist_per_loc])

    z = 1.96
    moe = z * (inf_rho_std_per_loc / np.sqrt(len(inf_rho_avg_per_loc)))
    inf_rho_u95_per_loc = inf_rho_avg_per_loc + moe
    inf_rho_l95_per_loc = inf_rho_avg_per_loc - moe

    avg_infected_per_loc = np.nanmean(infected_per_loc_sim, axis=0)
    std_infected_per_loc = np.nanmean(infected_per_loc_sim, axis=0)
    moe = z * (std_infected_per_loc / np.sqrt(len(infected_per_loc_sim)))
    l95_infected_per_loc = avg_infected_per_loc - moe
    u95_infected_per_loc = avg_infected_per_loc + moe

    nsims = len(inf_rho_h_dist_per_loc_sim)
    nlocs = len(inf_rho_h_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
    
    inf_rho_h_dist_per_loc = [[] for _ in range(nlocs)]
    infected_h_per_loc_sim = np.zeros((nsims, nlocs))
    fra_inf_h_per_loc_sim = np.zeros((nsims, nlocs))
    
    for sim_idx in range(len(inf_rho_h_dist_per_loc_sim)):
        for loc_idx in range(nlocs):
            inf_rho_h_values = inf_rho_h_dist_per_loc_sim[sim_idx][loc_idx]
            inf_rho_h_dist_per_loc[loc_idx].extend(inf_rho_h_values)
            infected_h_per_loc_sim[sim_idx][loc_idx] = len(inf_rho_h_values)
            fra_inf_h_per_loc_sim[sim_idx][loc_idx] = len(inf_rho_h_values) / infected_per_loc_sim[sim_idx][loc_idx]

    inf_rho_h_avg_per_loc = np.array([np.nanmean(sublist) for sublist in inf_rho_h_dist_per_loc])
    inf_rho_h_std_per_loc = np.array([np.nanstd(sublist) for sublist in inf_rho_h_dist_per_loc])

    moe = z * (inf_rho_h_std_per_loc / np.sqrt(len(inf_rho_h_avg_per_loc)))
    inf_rho_h_u95_per_loc = inf_rho_h_avg_per_loc + moe
    inf_rho_h_l95_per_loc = inf_rho_h_avg_per_loc - moe

    avg_fra_inf_h_per_loc = np.nanmean(fra_inf_h_per_loc_sim, axis=0)
    std_fra_inf_h_per_loc = np.nanmean(fra_inf_h_per_loc_sim, axis=0)
    moe = z * (std_fra_inf_h_per_loc / np.sqrt(len(fra_inf_h_per_loc_sim)))
    l95_fra_inf_h_per_loc = avg_fra_inf_h_per_loc - moe
    u95_fra_inf_h_per_loc = avg_fra_inf_h_per_loc + moe

    nlocs = len(inf_rho_o_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
    inf_rho_o_dist_per_loc = [[] for _ in range(nlocs)]
    infected_o_per_loc_sim = np.zeros((nsims, nlocs))
    fra_inf_o_per_loc_sim = np.zeros((nsims, nlocs))

    for sim_idx in range(len(inf_rho_o_dist_per_loc_sim)):
        for loc_idx in range(nlocs):
            inf_rho_o_values = inf_rho_o_dist_per_loc_sim[sim_idx][loc_idx]
            inf_rho_o_dist_per_loc[loc_idx].extend(inf_rho_o_values)
            infected_o_per_loc_sim[sim_idx][loc_idx] = len(inf_rho_o_values)
            fra_inf_o_per_loc_sim[sim_idx][loc_idx] = len(inf_rho_o_values) / infected_per_loc_sim[sim_idx][loc_idx]

    inf_rho_o_avg_per_loc = np.array([np.nanmean(sublist) for sublist in inf_rho_o_dist_per_loc])
    inf_rho_o_std_per_loc = np.array([np.nanstd(sublist) for sublist in inf_rho_o_dist_per_loc])

    moe = z * (inf_rho_o_std_per_loc / np.sqrt(len(inf_rho_o_avg_per_loc)))
    inf_rho_o_u95_per_loc = inf_rho_o_avg_per_loc + moe
    inf_rho_o_l95_per_loc = inf_rho_o_avg_per_loc - moe

    avg_fra_inf_o_per_loc = np.nanmean(fra_inf_o_per_loc_sim, axis=0)
    std_fra_inf_o_per_loc = np.nanmean(fra_inf_o_per_loc_sim, axis=0)
    moe = z * (std_fra_inf_o_per_loc / np.sqrt(len(fra_inf_o_per_loc_sim)))
    l95_infected_o_per_loc = avg_fra_inf_o_per_loc - moe
    u95_infected_o_per_loc = avg_fra_inf_o_per_loc + moe

    nlocs = 2500
    x_cells = int(np.sqrt(nlocs))
    y_cells = x_cells

    avg_inf_rho_lattice = np.zeros((x_cells, y_cells))
    avg_inf_rho_h_lattice = np.zeros((x_cells, y_cells))
    avg_inf_rho_o_lattice = np.zeros((x_cells, y_cells))
    avg_fra_inf_h_lattice = np.zeros((x_cells, y_cells))
    avg_fra_inf_o_lattice = np.zeros((x_cells, y_cells))

    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            avg_inf_rho_lattice[y_cells - 1 - j, i] = inf_rho_avg_per_loc[l]
            avg_inf_rho_h_lattice[y_cells - 1 - j, i] = inf_rho_h_avg_per_loc[l]
            avg_inf_rho_o_lattice[y_cells - 1 - j, i] = inf_rho_o_avg_per_loc[l]
            avg_fra_inf_h_lattice[y_cells - 1 - j, i] = avg_fra_inf_h_per_loc[l]
            avg_fra_inf_o_lattice[y_cells - 1 - j, i] = avg_fra_inf_o_per_loc[l]
            l += 1

    output = {}
    output['avg_inf_rho_lattice'] = avg_inf_rho_lattice
    output['avg_inf_rho_h_lattice'] = avg_inf_rho_h_lattice
    output['avg_inf_rho_o_lattice'] = avg_inf_rho_o_lattice
    output['avg_fra_inf_h_lattice'] = avg_fra_inf_h_lattice
    output['avg_fra_inf_o_lattice'] = avg_fra_inf_o_lattice

    return output

def load_depr_chapter_panel_netmob_data(fullname):
    output_dict = { }

    if not fullname.endswith('.pickle'):
        fullname += '.pickle'
    with open(fullname, 'rb') as input_data:
        edig_dict = pk.load(input_data)

    output_dict['agents'] = edig_dict['agents_per_rho']
    output_dict['infected'] = edig_dict['infected_per_rho']
    output_dict['returner_netmob'] = edig_dict['returner_netmob']
    output_dict['commuter_netmob'] = edig_dict['commuter_netmob']
    output_dict['explorer_netmob'] = edig_dict['explorer_netmob']

    return output_dict

def extend_depr_chapter_panel_netmob_results(
        out_sim_data, 
        agents_per_rho_sim,
        infected_per_rho_sim,
        returner_netmob_sim,
        commuter_netmob_sim,
        explorer_netmob_sim,
    ):

    agents_per_rho_sim.extend(out_sim_data['agents'])
    infected_per_rho_sim.extend(out_sim_data['infected'])
    returner_netmob_sim.extend(out_sim_data['returner_netmob'])
    commuter_netmob_sim.extend(out_sim_data['commuter_netmob'])
    explorer_netmob_sim.extend(out_sim_data['explorer_netmob'])

import numpy as np

def compute_depr_chapter_panel_netmob(
        agents_per_rho_sim,
        infected_per_rho_sim,
        returner_netmob_sim,
        commuter_netmob_sim,
        explorer_netmob_sim,
        space_df,
        prevalence_cutoff=0.025,
        attr_cutoff=0.000000001,
):
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)

    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)

    attr_l = space_df['attractiveness'].to_numpy()

    # Sort locations based on attractiveness in decreasing order
    sorted_indices = np.argsort(attr_l)[::-1]
    attr_l = attr_l[sorted_indices]

    nlocs = len(attr_l)
    nsims = len(agents_per_rho_sim)

    avg_returner_netmob = np.zeros((nlocs, nlocs))
    avg_commuter_netmob = np.zeros((nlocs, nlocs))
    avg_explorer_netmob = np.zeros((nlocs, nlocs))

    for sim_idx in range(nsims):
        for key_tuple, value in returner_netmob_sim[sim_idx].items():
            origin = key_tuple[0]
            destination = key_tuple[1]

            avg_returner_netmob[sorted_indices[origin]][sorted_indices[destination]] += value
            #avg_returner_netmob[sorted_indices[destination]][sorted_indices[origin]] += value  # Assuming symmetry

        for key_tuple, value in commuter_netmob_sim[sim_idx].items():
            origin = key_tuple[0]
            destination = key_tuple[1]

            avg_commuter_netmob[sorted_indices[origin]][sorted_indices[destination]] += value
            #avg_commuter_netmob[sorted_indices[destination]][sorted_indices[origin]] += value  # Assuming symmetry

        for key_tuple, value in explorer_netmob_sim[sim_idx].items():
            origin = key_tuple[0]
            destination = key_tuple[1]

            avg_explorer_netmob[sorted_indices[origin]][sorted_indices[destination]] += value
            #avg_explorer_netmob[sorted_indices[destination]][sorted_indices[origin]] += value  # Assuming symmetry

    # Divide the sums by nsims to compute the averages
    avg_returner_netmob /= nsims
    avg_commuter_netmob /= nsims
    avg_explorer_netmob /= nsims

    # Identify irrelevant locations based on the attractiveness criterion
    irrelevant_locations = np.where(attr_l < attr_cutoff)[0]

    # Determine the indices of relevant locations
    relevant_locations = np.setdiff1d(np.arange(len(attr_l)), irrelevant_locations)

    # Filter out irrelevant locations from the averages
    avg_returner_netmob = avg_returner_netmob[relevant_locations][:, relevant_locations]
    avg_commuter_netmob = avg_commuter_netmob[relevant_locations][:, relevant_locations]
    avg_explorer_netmob = avg_explorer_netmob[relevant_locations][:, relevant_locations]

    # Renormalize by setting non-zero elements to 1
    avg_returner_netmob = np.where(avg_returner_netmob != 0, 1, 0)
    avg_commuter_netmob = np.where(avg_commuter_netmob != 0, 1, 0)
    avg_explorer_netmob = np.where(avg_explorer_netmob != 0, 1, 0)

    output = {}
    output['avg_returner_netmob'] = avg_returner_netmob
    output['avg_commuter_netmob'] = avg_commuter_netmob
    output['avg_explorer_netmob'] = avg_explorer_netmob

    return output

def build_color_dictionary():
    color_dict = {
        'b1hom': 'slateblue', 
        'b1het': 'slateblue',
        'depr': 'teal', 
        'plain': 'steelblue',
        'uniform': 'purple', 
        }
    return color_dict

def build_marker_dictionary():
    marker_dict = {
        'b1hom': 's',
        'b1het': 's',
        'depr': 'o',
        'plain': 'x',
        'uniform': '+',
        }
    return marker_dict

def build_linestyle_dictionary():
    linestyle_dict = {
        'b1hom': 'dashed',
        'b1het': 'dashed',
        'depr': 'dashed',
        'plain': 'dotted',
        'uniform': 'dashdot',
        }
    return linestyle_dict


def build_label_dictionary():
    label_dict = {
        'b1hom': 'mem-less',
        'b1het': 'mem-less',
        'depr': 'd-EPR',  
        'plain': 'plain',
        'uniform': 'uniform', 
        }
    return label_dict

def build_boston_lattice_file_name(cwd_path, lower_path, config_file_name):
    file_path = os.path.join(cwd_path, lower_path, config_file_name)

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    DX = data['DX']
    DY = data['DY']
    LN0 = data['LN0']
    LT0 = data['LT0']
    rd = data['rd']
    x = data['x']
    y = data['y']
    ts = data['ts']
    
    file_name = f"space_DX{DX}_DY{DY}_LN0{LN0}_LT0{LT0}_rd{rd}_x{x}_y{y}_ts{ts}.pickle"

    return file_name

def build_boston_scatter_file_name(cwd_path, lower_path, space_config_file_name):
    space_file_path = os.path.join(cwd_path, lower_path, space_config_file_name)

    with open(space_file_path, 'r') as file:
        space_metadata = json.load(file)
    
    LNE = space_metadata['LNE']
    LNW = space_metadata['LNW']
    LTN = space_metadata['LTN']
    LTS = space_metadata['LTS']
    nl = space_metadata['nl']
    rd = space_metadata['rd']
    ts = space_metadata['ts']

    space_file_name = f"space_LNE{LNE}_LNW{LNW}_LTN{LTN}_LTS{LTS}_nl{nl}_rd{rd}_ts{ts}.pickle"

    return space_file_name

def build_regular_lattice_file_name(cwd_path, lower_path, space_config_file_name):
    space_file_path = os.path.join(cwd_path, lower_path, space_config_file_name)

    with open(space_file_path, 'r') as file:
        space_metadata = json.load(file)
    
    am = space_metadata['am']
    aa = space_metadata['aa']
    ab = space_metadata['ab']
    bm = space_metadata['bm']
    np = space_metadata['np']
    x = space_metadata['x']
    y = space_metadata['y']
    ts = space_metadata['ts']

    space_file_name = f"space_am{am}_aa{aa}_ab{ab}_bm{bm}_np{np}_x{x}_y{y}_ts{ts}.pickle"

    return space_file_name

def build_mobility_file_name(cwd_path, lower_path, mobility_config_file_name, mobility_selection_flag, mobility_scenario_model, space_config_file_name, tessellation_model):
    mobility_file_path = os.path.join(cwd_path, lower_path, mobility_config_file_name)

    with open(mobility_file_path, 'r') as file:
        mobility_metadata = json.load(file)

    gm = mobility_metadata['gm']
    hw = mobility_metadata['hw']
    t = mobility_metadata['t']
    rm = mobility_metadata['rm']
    ra = mobility_metadata['ra']
    na = mobility_metadata['na']
    lm = mobility_metadata['lm']
    lf = mobility_metadata['lf']
    qm = mobility_metadata['qm']
    qf = mobility_metadata['qf']
    tm = mobility_metadata['tm']

    head = "m" + mobility_selection_flag + "_"
    subhead = "ms" + mobility_scenario_model + "_"
    chain = f"gm{gm}_hw{hw}_t{t}_rm{rm}_"
    if rm == 'Beta' or rm == 'Gamma' or rm == 'Gaussian' or rm == 'LogNormal' or rm == 'NegativeBinomial':
        rb = mobility_metadata['rb']
        rho_chain = f"ra{ra}_rb{rb}_"
    elif rm == 'DeltaBimodal':
        rc = mobility_metadata['rc']
        rho_chain = f"ra{ra}_rb{rb}_rc{rc}_"
    elif rm == 'Exponential' or rm == 'Homogeneous' or rm == 'Uniform':
        rho_chain = f"ra{ra}_"

    lock_chain = f"lm{lm}_lf{lf}_"
    time_chain = f"tm{tm}_"   

    if tessellation_model == 'boston-lattice':
        space_chain = build_boston_lattice_file_name(cwd_path, lower_path, space_config_file_name)
    elif tessellation_model == 'boston-scatter':
        space_chain = build_boston_scatter_file_name(cwd_path, lower_path, space_config_file_name)
    elif tessellation_model == 'synthetic-lattice':
        space_chain = build_regular_lattice_file_name(cwd_path, lower_path, space_config_file_name)

    mobility_file_name = head + subhead + chain + rho_chain + lock_chain + time_chain + space_chain

    return mobility_file_name

def find_latest_file_with_timestamp(path, lower_path, mobility_scenario_model):
    fullpath = os.path.join(path, lower_path)
    files = [f for f in os.listdir(fullpath) 
             if f.startswith(f'mset_ms{mobility_scenario_model}') and f.endswith('.pickle')]

    if not files:
        raise FileNotFoundError("No matching files found")

    # Sort files by creation time (most recent first)
    full_paths = [os.path.join(fullpath, f) for f in files]
    full_paths.sort(key=os.path.getctime, reverse=True)

    # Extract mobility_time_stamp from the most recent file
    latest_file = full_paths[0]
    start_index = latest_file.find('tm') + 2
    end_index = latest_file.find('_', start_index)
    mobility_time_stamp = latest_file[start_index:end_index]

    return mobility_time_stamp

def rho_distribution_model_parameter_dictionary(rho_distribution_model):
    rdmp_dict = {}

    if rho_distribution_model == 'beta':
        rdmp_dict['ra'] = 2.0
        rdmp_dict['rb'] = 2.0
    elif rho_distribution_model == 'gaussian':
        rdmp_dict['ra'] = 0.5
        rdmp_dict['rb'] = 0.1
    
    rdmp_dict
