use rand::Rng;
use rand::thread_rng;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::WeightedIndex;
use serde_pickle::ser::SerOptions;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use strum_macros::Display;

use crate::epidemic::AgentGrid;
use crate::{
    space::Space, 
    utils::{
        build_bostonlattice_census_filename,
        load_databased_census_data,
        sample_uniform, 
        sample_beta, 
        sample_gamma, 
        sample_delta_bimodal, 
        sample_exponential, 
        sample_truncated_gaussian, 
        sample_log_normal, 
        sample_negative_binomial
    }
};

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq, 
    Deserialize
)]
pub enum MobilitySelection {
    Pool,
    Set,
}

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq, 
    Deserialize
)]
pub enum MobilityScenario {
    B1het,
    B1hom,
    B2,
    Depr,
    Plain,
    Uniform,
}

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq, 
    Deserialize
)]
pub enum QuarantineStrategy {
    Explorers,
    Random,
    Returners,
    TopExplorers,
    TopReturners,
    Unmitigated,
}

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq, 
    Deserialize
)]
pub enum LockdownStrategy {
    LeastAttractive,
    MostAttractive,
    Random,
    Unmitigated,
}

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq, 
    Deserialize
)]
pub enum HomeModel {
    Attractiveness,
    Census,
    Random,
}

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq, 
    Deserialize
)]
pub enum RhoDistributionModel {
    Beta,
    DeltaBimodal,
    Exponential,
    Gamma,
    Gaussian,
    Homogeneous,
    LogNormal,
    NegativeBinomial,
    Uniform,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MobilityPars {
    pub gamma: f64,
    pub home_weight: u32,
    pub location_threshold: Option<f64>,
    pub lockdown_strategy: LockdownStrategy,
    pub locked_fraction: f64,
    pub nagents: u32,
    pub nlocs: u32,
    pub quarantine_strategy: QuarantineStrategy,
    pub quarantined_fraction: f64,
    pub rho: f64,
    pub rho_model: RhoDistributionModel,
    pub selection: MobilitySelection,
    pub scenario: MobilityScenario,
    pub t_max: u32,
}

impl MobilityPars {
    pub fn new(
        lockdown_strategy: LockdownStrategy,
        mob_hm: &HashMap<String, f64>,
        quarantine_strategy: QuarantineStrategy,
        rho_model: RhoDistributionModel, 
        selection: MobilitySelection,
        scenario: MobilityScenario,
    ) -> Self {
        Self { 
            gamma: *mob_hm.get("gamma").unwrap(), 
            home_weight: *mob_hm.get("home_weight").unwrap() as u32,
            location_threshold: 
            Some(*mob_hm.get("location_threshold").unwrap()),
            lockdown_strategy,
            locked_fraction: *mob_hm.get("locked_fraction").unwrap(),
            nagents: *mob_hm.get("nagents").unwrap() as u32, 
            nlocs: 0,
            quarantine_strategy,
            quarantined_fraction: 
            *mob_hm.get("quarantined_fraction").unwrap(),
            rho: *mob_hm.get("rho").unwrap(), 
            rho_model,
            selection,
            scenario,
            t_max: *mob_hm.get("t_max").unwrap() as u32, 
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct MobileAgent {
    pub current: u32,
    pub id: u32,
    pub gamma: f64,
    pub home: u32,
    pub rho: f64,
    pub trajectory: Vec<u32>,
    pub visits_hm: HashMap<u32, u32>,
}

impl MobileAgent {
    pub fn new() -> Self {
        Self {
            current: 0,
            id: 0,
            gamma: 0.21,
            home: 0,
            rho: 0.6,
            trajectory: Vec::new(),
            visits_hm: HashMap::new(),
        }
    }

    pub fn set_id(&mut self, id: u32) {
        self.id = id;
    }

    /// Return agent's exploration probability
    pub fn exploration_probability(&self) -> f64 {
        let locations_visited = self.distinct_visited_locations();
        let rho = self.rho;
        let gamma = self.gamma;
        let p_new = rho * f64::powf(locations_visited as f64, -gamma);
        p_new
    }

    /// Obtain new location to move through preferential exploration mechanism
    pub fn preferential_exploration(&self, od_rates: &Vec<Vec<f64>>) -> u32 {
        // Prepare data
        let origin = self.current;
        let number_of_locations = od_rates[origin as usize].len() as u32;
        let mut cumulative_probability = 0.0;
        let mut rng = rand::thread_rng();
        let trial: f64 = rng.gen();
        
        // Loop over every location to perform a weighted sampling
        for destination in 0..number_of_locations {
            if destination != origin {
                cumulative_probability += 
                od_rates[origin as usize][destination as usize];
                if trial < cumulative_probability {
                    return destination as u32;
                }
            }
        }
        panic!();
    }

    pub fn preferential_exploration_with_exclusion(
        &self, 
        od_rates: &Vec<Vec<f64>>,
    ) -> u32 {
        let origin = self.current;
        
        let visited_locations: Vec<usize> = 
        self
        .visited_locations()
        .iter()
        .map(|x| *x as usize)
        .collect();
        
        let weights: Vec<f64> = 
        (0..od_rates[origin as usize].len())
        .map(|i| if visited_locations.contains(&i) { 
            0.0 
        } else { 
            od_rates[origin as usize][i] 
        }
        ).collect();
        
        let total_weight: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = 
        weights
        .iter()
        .map(|w| w / total_weight)
        .collect();
        let weight_summa: f64 = normalized_weights.iter().sum();
        if weight_summa == 0.0 {
            println!("WTF!");
            return origin;
        }
    
        let dist = WeightedIndex::new(&normalized_weights).unwrap();
        let mut rng = rand::thread_rng();
        let destination = dist.sample(&mut rng) as u32;
        if visited_locations.iter().any(|x| x == &(destination as usize)) {
            println!("You have already explored {destination}");
        }
        destination
    }
    
    /// Obtain location to move through preferential return mechanism
    pub fn preferential_return(&self) -> u32 {
        // Prepare data
        let total_visits = self.total_visits();
        let number_of_locations = self.visited_locations().len() as u32;
        let normalized_v_f: Vec<f64> = self
            .visits_hm
            .iter()
            .map(|(_, v)| *v as f64 / total_visits as f64)
            .collect();
        let destination: u32;
        let mut cumulative_probability = 0.0;
        let mut rng = rand::thread_rng();
        let trial: f64 = rng.gen();
        
        // Loop over every location to perform a weighted sampling
        for loc_index in 0..number_of_locations {
            cumulative_probability += normalized_v_f[loc_index as usize];
            if trial < cumulative_probability {
                destination = self.visited_locations()[loc_index as usize];
                return destination;
            }
        }
        panic!();
    }

    pub fn preferential_return_with_exclusion(&self) -> u32 {
        let current_location = self.current;
        let mut locations: Vec<u32> = vec![];
        let mut values: Vec<u32> = vec![];
    
        for (k, v) in self.visits_hm.iter() {
            if k != &current_location {
                locations.push(*k);
                values.push(*v);
            }
        }
    
        let total_visits: u32 = values.iter().sum();
        let weights: Vec<f64> = values
            .iter()
            .map(|v| *v as f64 / total_visits as f64)
            .collect();
        let total_weight: f64 = weights.iter().sum();
        if total_weight == 0.0 {
            println!("WTF??");
            return current_location;
        }
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = rand::thread_rng();
        let index = dist.sample(&mut rng) as usize;
        let destination = locations[index];
        if current_location == destination {
            println!("You have already returned to {destination}");
        }
        destination
    }

    pub fn forget_location(&mut self, threshold: f64) {
        let current_location = self.current;
        let visit_hm_share = self.visit_share();
        let key_to_remove = visit_hm_share
            .iter()
            .filter(|(key, value)| *value < &threshold 
            && *key != &current_location)
            .map(|(key, _)| *key)
            .next();
    
        if let Some(key) = key_to_remove {
            self.visits_hm.remove(&key);
            //println!("Forgotten location {key}");
        }
    }

    pub fn forget_all_locations(&mut self, threshold: f64) {
        let visit_hm_share = self.visit_share();
        let keys_to_remove: Vec<u32> = visit_hm_share
            .iter()
            .filter(|(_, value)| *value < &threshold)
            .map(|(key, _)| *key)
            .collect();
    
        self.visits_hm
            .drain()
            .filter(|(k, _)| keys_to_remove.contains(k))
            .for_each(drop);
    }

    /// Return a $\rho$ value for the agent
    pub fn sample_rho(
        &mut self, 
        flag: RhoDistributionModel, 
        rho_hm: &HashMap<String, f64>
    ) {
        let rho = match flag {
            RhoDistributionModel::Beta => {
                let alpha = *rho_hm.get("alpha").unwrap();
                let beta = *rho_hm.get("beta").unwrap();
                sample_beta(alpha, beta)
            },
            RhoDistributionModel::DeltaBimodal => {
                let share = *rho_hm.get("share").unwrap();
                let mode1 = *rho_hm.get("mode1").unwrap();
                let mode2 = *rho_hm.get("mode2").unwrap();
                sample_delta_bimodal(share, mode1, mode2)
            }
            RhoDistributionModel::Exponential => {
                let rate = *rho_hm.get("rate").unwrap();
                sample_exponential(rate)
            }
            RhoDistributionModel::Gamma => {
                let shape = *rho_hm.get("shape").unwrap();
                let scale = *rho_hm.get("scale").unwrap();
                sample_gamma(shape, scale)
            },
            RhoDistributionModel::Gaussian => {
                let mean = *rho_hm.get("mean").unwrap();
                let std_dev = *rho_hm.get("std_dev").unwrap();
                sample_truncated_gaussian(mean, std_dev)
            }
            RhoDistributionModel::Homogeneous => { 
                *rho_hm.get("rho").unwrap()
            },
            RhoDistributionModel::LogNormal => {
                let mean = *rho_hm.get("mean").unwrap();
                let variance = *rho_hm.get("variance").unwrap();
                sample_log_normal(mean, variance)
            }
            RhoDistributionModel::NegativeBinomial => {
                let mean = *rho_hm.get("mean").unwrap();
                let variance = *rho_hm.get("variance").unwrap();
                sample_negative_binomial(mean, variance)
            }
            RhoDistributionModel::Uniform => {
                sample_uniform()
            },
        };
        self.rho = rho
    }

    /// Assign randomly uniformly a home for the agent.
    pub fn set_home(
        &mut self, 
        space: &Space, 
        space_hm: &HashMap<String, f64>, 
        home_weight: u32, 
        home_flag: HomeModel,
    ) { 

        match home_flag {
            HomeModel::Attractiveness => {},
            HomeModel::Census => {
                let mut rng = rand::thread_rng();
                let attractiveness_cutoff = 0.0000001;
    
                let filename = build_bostonlattice_census_filename(space_hm);
                let home_fraction_array = load_databased_census_data(&filename);

                // Calculate weights from the values in homes_array
                let dist = WeightedIndex::new(&home_fraction_array).unwrap();

                loop {
                    let trial = dist.sample(&mut rng);
                    let attractiveness = space.inner()[trial as usize].attractiveness.unwrap();
                    if attractiveness >= attractiveness_cutoff {
                        self.home = trial as u32;
                        self.current = self.home;
                        self.visits_hm.insert(self.home, home_weight);
                        self.trajectory.push(self.home);
                        break; // Exit the loop once a suitable location is found
                    }
                }
            },
            HomeModel::Random => {
                let number_of_locations = space.number_of_cells();
                let mut rng = rand::thread_rng();
                let attractiveness_cutoff = 0.0000001;

                loop {
                    let trial = rng.gen_range(0..number_of_locations);
                    let attractiveness = space.inner()[trial as usize].attractiveness.unwrap();
                    if attractiveness >= attractiveness_cutoff {
                        self.home = trial as u32;
                        self.current = self.home;
                        self.visits_hm.insert(self.home, home_weight);
                        self.trajectory.push(self.home);
                        break; // Exit the loop once a suitable location is found
                    }
                }
            },
        }
    }

    /// Return total number of visits performed by an agent to all locations
    pub fn total_visits(&self) -> u32 {
        self.visits_hm.values().sum()
    }

    /// Return the number of distinct visited locations by the agent
    pub fn distinct_visited_locations(&self) -> u32 {
        self.visits_hm.keys().len() as u32
    }

    /// Return a vector with all the visited locations by the agents
    pub fn visited_locations(&self) -> Vec<u32> {
        self.visits_hm.keys().map(|x| *x).collect()
    }

    /// Return the most visited location by the agent.
    pub fn most_visited_location(&self) -> u32 {
        self.visits_hm
            .iter()
            .max_by_key(|(_, &value)| value)
            .map(|(key, _)| *key)
            .unwrap()
    }

    /// Return HashMap where key is the location id and value is the visit share,
    pub fn visit_share(&self) -> HashMap<u32, f64> {
        let total_visits = self.total_visits();
        let mut hm_visit_share = HashMap::new();
        
        for (location, frequency) in &self.visits_hm {
            let visit_share = *frequency as f64 / total_visits as f64;
            hm_visit_share.insert(*location, visit_share);
        }
        hm_visit_share
    }

    pub fn run_depr_dynamics(
        &mut self, 
        space: &Space,
        od_rates: &Vec<Vec<f64>>, 
        mpars: &MobilityPars,
    ) {
        let mut rng = rand::thread_rng();

        for _ in 0..mpars.t_max {
            // Compute exploration probability
            let p_exp = self.exploration_probability();

            // Explore or return
            let trial: f64 = rng.gen();
            if trial < p_exp 
            && self.distinct_visited_locations() < space.number_of_cells()
            || self.distinct_visited_locations() == 1 {
                // Sample new location through gravity-law preferential exploration
                let destination = 
                self.preferential_exploration_with_exclusion(&od_rates);
    
                // Update agent's current location, visits hashmaps & trajectory
                self.current = destination;
                match self.visits_hm.get_mut(&destination) {
                    Some(frequency) => *frequency += 1,
                    None => {
                        self.visits_hm.insert(destination, 1);
                    }
                }
                self.trajectory.push(self.current);
            } else {
                // Sample location through preferential return
                let destination = self.preferential_return();

                // Update agent's struct: 
                self.current = destination;
                match self.visits_hm.get_mut(&destination) {
                    Some(frequency) => *frequency += 1,
                    None => {
                        self.visits_hm.insert(destination, 1);
                    }
                }
                self.trajectory.push(self.current);
            }
        }
    }

    pub fn run_baseline1_dynamics(
        &mut self, 
        space: &Space,
        od_rates: &Vec<Vec<f64>>, 
        mpars: &MobilityPars,
        hom_flag: bool,
    ) {
        let mut rng = rand::thread_rng();

        for _ in 0..mpars.t_max {
            // Compute exploration probability
            let p_exp = match hom_flag {
                true => 0.5,
                false => self.rho,
            };
          
            // Explore or return
            let trial: f64 = rng.gen();
            if trial < p_exp 
            && self.distinct_visited_locations() < space.number_of_cells()
            || self.distinct_visited_locations() == 1 {
                // Sample new location through gravity-law preferential exploration
                let mut destination = self.preferential_exploration(&od_rates);
                while destination == self.current {
                    destination = self.preferential_exploration(&od_rates);
                }

                // Update agent's current location, visits hashmaps & trajectory
                self.current = destination;
                match self.visits_hm.get_mut(&destination) {
                    Some(frequency) => *frequency += 1,
                    None => {
                        self.visits_hm.insert(destination, 1);
                    }
                }
                self.trajectory.push(self.current);
            } else {
                // Sample location through preferential return
                let destination = self.current;

                // Update agent's struct: 
                self.current = destination;
                match self.visits_hm.get_mut(&destination) {
                    Some(frequency) => *frequency += 1,
                    None => {
                        self.visits_hm.insert(destination, 1);
                    }
                }
                self.trajectory.push(self.current);
            }
        }
    }

    pub fn run_baseline2_dynamics(
        &mut self, 
        space: &Space,
        od_rates: &Vec<Vec<f64>>, 
        mob_grid: &Vec<Vec<f64>>,
        mpars: &MobilityPars,
    ) {
        let mut rng = rand::thread_rng();

        for t in 0..mpars.t_max {

            let l = self.current;

            let p_exp = mob_grid[l as usize][t as usize];
            self.rho += p_exp;

            // Explore or return
            let trial: f64 = rng.gen();
            if trial < p_exp 
            && self.distinct_visited_locations() < space.number_of_cells()
            || self.distinct_visited_locations() == 1 {
                // Sample new location through gravity-law preferential exploration
                let mut destination = self.preferential_exploration(&od_rates);
                while destination == self.current {
                    destination = self.preferential_exploration(&od_rates);
                }

                // Update agent's current location, visits hashmaps & trajectory
                self.current = destination;
                match self.visits_hm.get_mut(&destination) {
                    Some(frequency) => *frequency += 1,
                    None => {
                        self.visits_hm.insert(destination, 1);
                    }
                }
                self.trajectory.push(self.current);
            } else {
                // Sample location through preferential return
                let destination = self.current;

                // Update agent's struct: 
                self.current = destination;
                match self.visits_hm.get_mut(&destination) {
                    Some(frequency) => *frequency += 1,
                    None => {
                        self.visits_hm.insert(destination, 1);
                    }
                }
                self.trajectory.push(self.current);
            }
        }

        let rho_avg = self.rho / mpars.t_max as f64;
        self.rho = rho_avg;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MobileAgentOutput {
    pub rho: f64,
    pub trajectory: Vec<u32>,
}

impl MobileAgentOutput {
    pub fn new(agent: &MobileAgent) -> Self {
        Self { 
            rho: agent.rho, 
            trajectory: agent.trajectory.clone()
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MobileEnsembleOutput {
    inner: Vec<MobileAgentOutput>,
}

impl MobileEnsembleOutput {
    pub fn new() -> Self {
        Self { 
            inner: Vec::new(), 
        }
    }

    pub fn inner(&self) -> &Vec<MobileAgentOutput> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<MobileAgentOutput> {
        &mut self.inner
    }

    pub fn number_of_agents(&self) -> u32 {
        self.inner.len() as u32
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MobilityMetadata {
    pub pars: MobilityPars,
}

impl MobilityMetadata {
    pub fn new(pars: &MobilityPars) -> Self {
        Self { 
            pars: *pars, 
        }
    }

    pub fn write_mobility_metadata_file_name(
        &self, 
        mob_file_name: &String
    ) -> String {
        //let first_underscore_index = mob_file_name.find('_').unwrap();
        //let new_file_name = &mob_file_name[first_underscore_index + 1..];
        format!("meta_{}", mob_file_name)
    }

    pub fn to_pickle(&self, mob_file_name: &String) {

        let serialized = 
        serde_pickle::to_vec(self, SerOptions::new()).unwrap();
        let file_name = 
        self.write_mobility_metadata_file_name(mob_file_name);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
    }
}

pub fn get_max_agents(mobility_data: &[MobileAgentOutput]) -> u32 {
    mobility_data.len() as u32
}

pub fn get_chosen_agents(
    mobility_data: &[MobileAgentOutput], 
    nchosen: usize
) -> Vec<u32> {
    let mut agent_ids: Vec<u32> = (0..mobility_data.len() as u32).collect();
    let mut rng = thread_rng();
    agent_ids.shuffle(&mut rng);
    agent_ids.truncate(nchosen);
    agent_ids
}

pub fn target_quarantined_agents(
    rho_vec: &mut Vec<f64>, 
    quarantine_flag: QuarantineStrategy,
    quarantined_fraction: f64,
) -> Vec<u32> {
    let nagents = rho_vec.len();
    let quarantined_agents = Vec::new();
    
        match quarantine_flag {
            QuarantineStrategy::Unmitigated => {},
            QuarantineStrategy::Random => {
                let quarantined_agents = 
                (quarantined_fraction * nagents as f64) as usize;
                let mut rng = rand::thread_rng();
                let samples: Vec<usize> = 
                (0..nagents)
                .choose_multiple(&mut rng, quarantined_agents);
                
                for agent_id in samples {
                    rho_vec[agent_id] = 0.0;
                }
            },
            QuarantineStrategy::Explorers => {
                let mut quarantined_agents = 
                (quarantined_fraction * nagents as f64) as usize;
    
                for agent_id in 0..nagents {
                    if quarantined_agents == 0 {
                        break;
                    }
                    if rho_vec[agent_id] >= 0.5 {
                        rho_vec[agent_id] = 0.0;
                        quarantined_agents -= 1;
                    }
                }
            },
            QuarantineStrategy::Returners => {
                let mut quarantined_agents = 
                (quarantined_fraction * nagents as f64) as usize;
    
                for agent_id in 0..nagents {
                    if quarantined_agents == 0 {
                        break;
                    }
                    if rho_vec[agent_id] < 0.5 {
                        rho_vec[agent_id] = 0.0;
                        quarantined_agents -= 1;
                    }
                }
            },
            QuarantineStrategy::TopExplorers => {
                let quarantine_target = 
                (quarantined_fraction * nagents as f64) as usize;
    
                let indices: Vec<usize> = (0..nagents as usize).collect();
    
                // Sort the indices based on rho in descending order
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    rho_vec[b].partial_cmp(&rho_vec[a]).unwrap().reverse()
                });
    
                // Select the indices of the most attractive locations
                let selected_indices = &sorted_indices[..quarantine_target];
            
                // Set the Vaccinated status of the selected locations to true
                for &index in selected_indices {
                    rho_vec[index] = 0.0;
                }
            },
            QuarantineStrategy::TopReturners => {
                let quarantine_target = 
                (quarantined_fraction * nagents as f64) as usize;
    
                let indices: Vec<usize> = (0..nagents as usize).collect();
    
                // Sort the indices based on rho in ascending order
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    rho_vec[a].partial_cmp(&rho_vec[b]).unwrap().reverse()
                });
    
                // Select the indices of the most attractive locations
                let selected_indices = &sorted_indices[..quarantine_target];
            
                // Set the Vaccinated status of the selected locations to true
                for &index in selected_indices {
                    rho_vec[index] = 0.0;
                }
            },
        }

        quarantined_agents
}

pub fn build_mobility_parameter_grid(agent_grid: &AgentGrid) -> Vec<Vec<f64>> {
    let nlocs = agent_grid.inner().len();
    let t_max = agent_grid.inner()[0].len();
    
    // Initialize the mobility grid with zeros
    let mut mobility_grid = vec![vec![0.0; t_max]; nlocs];

    for t in 0..t_max {
        for l in 0..nlocs {
            if let Some(agent_ids) = &agent_grid.inner()[l][t] {
                let total_population = agent_ids.len() as f64;

                // Calculate remaining population in the next time step
                let remaining_population = if t < t_max - 1 {
                    let next_time_step = &agent_grid.inner()[l][t + 1];
                    if let Some(next_agent_ids) = next_time_step {
                        let common_agents = agent_ids
                            .iter()
                            .filter(|&id| next_agent_ids.contains(id))
                            .count() as f64;
                        common_agents / total_population
                    } else {
                        0.0 // No agents in the next time step
                    }
                } else {
                    0.0 // No next time step
                };

                // Calculate the mobility metric and store it in the mobility grid
                mobility_grid[l][t] = 1.0 - remaining_population;
            }
        }
    }

    mobility_grid
}