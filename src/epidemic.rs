use chrono::{Local, Datelike, Timelike};
use rand::prelude::*;
use rgsl::types::rng::Rng as gsl_Rng;
//use rand_distr::WeightedIndex;
use serde_pickle::ser::SerOptions;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use strum_macros::Display;

use crate::{
    event::{
        EventEnsemble, 
        Event
    },
    utils::{
        write_epidemic_data_file_name, 
        sir_prevalence
    }, 
    mobility::{
        MobileAgentOutput, 
        MobilityPars, 
        QuarantineStrategy, MobilityScenario, MobilitySelection
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
pub enum AgentSeedModel {
    Explorers,
    Random,
    Returners,
    TopExplorers,
    TopReturners,
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
pub enum LocationSeedModel {
    LeastAttractive,
    MostAttractive,
    Random,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct EpidemicPars {
    pub agent_seed: AgentSeedModel,
    pub escape_condition: f64,
    pub expedited_escape: bool,
    pub location_seed: LocationSeedModel,
    pub nagents: u32,
    pub nepicenters: u32,
    pub nsims: u32,
    pub pseudomass_exponent: f64,
    pub removal_rate: f64,
    pub seed_fraction: f64,
    pub t_epidemic: u32,
    pub transmission_rate: f64,
    pub vaccinated_fraction: f64,
    pub vaccination_strategy: VaccinationStrategy,
}

impl EpidemicPars {
    pub fn new(
        agent_seed_model: AgentSeedModel,
        escape_condition: f64,
        expedited_escape: bool,
        location_seed_model: LocationSeedModel, 
        nagents: u32,
        nepicenters: u32,
        nsims: u32,
        pseudomass_exponent: f64,
        removal_rate: f64,
        seed_fraction: f64,
        t_epidemic: u32,
        transmission_rate: f64,
        vaccinated_fraction: f64,
        vaccination_model: VaccinationStrategy,
    ) -> Self {
        Self { 
            agent_seed: agent_seed_model,
            escape_condition: escape_condition,
            expedited_escape: expedited_escape,
            location_seed: location_seed_model,
            nagents: nagents,
            nepicenters: nepicenters,
            nsims: nsims,
            pseudomass_exponent: pseudomass_exponent,
            removal_rate: removal_rate,
            seed_fraction: seed_fraction,
            t_epidemic: t_epidemic,
            transmission_rate: transmission_rate,
            vaccinated_fraction: vaccinated_fraction,
            vaccination_strategy: vaccination_model,
        }
    }

    pub fn to_pickle(&self, epi_file_name: &String) {
        let serialized = 
        serde_pickle::to_vec(self, SerOptions::new()).unwrap();
        let file_name = write_epidemic_data_file_name(epi_file_name);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
    }
}

pub fn set_epicenters(
    ls_model: LocationSeedModel, 
    nepicenters: u32,
    a_vec: &Vec<f64>,
) -> Vec<u32> {
    let epicenters = match ls_model {
        LocationSeedModel::LeastAttractive => {
            let mut indices: Vec<usize> = (0..a_vec.len()).collect();
            indices
            .sort_unstable_by(|&a, &b| a_vec[a]
                .partial_cmp(&a_vec[b])
                .unwrap());
            indices
                .iter()
                .take(nepicenters as usize)
                .map(|&i| i as u32)
                .collect()
        }
        LocationSeedModel::MostAttractive => {
            let mut indices: Vec<usize> = (0..a_vec.len()).collect();
            indices
            .sort_unstable_by(|&a, &b| a_vec[b]
                .partial_cmp(&a_vec[a])
                .unwrap());
            indices
                .iter()
                .take(nepicenters as usize)
                .map(|&i| i as u32)
                .collect()
        }
        LocationSeedModel::Random => {
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..a_vec.len()).collect();
            indices.shuffle(&mut rng);
            indices
                .iter()
                .take(nepicenters as usize)
                .map(|&i| i as u32)
                .collect()
        }
    };
    epicenters
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
pub enum HealthStatus {
    Susceptible,
    Infected,
    Removed,
    Vaccinated,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct EpidemicAgent {
    pub event_id: Option<u32>,
    pub epi_id: u32,
    pub mob_id: u32,
    pub infected_by: Option<u32>,
    pub infected_when: Option<u32>,
    pub infected_where: Option<u32>,
    pub mobility: f64,
    pub removed_when: Option<u32>,
    pub status: HealthStatus,
}

impl EpidemicAgent {
    pub fn new(epi_id: u32, mob_id: u32, status: HealthStatus, mobility: f64) -> Self {
        Self { 
            event_id: None,
            epi_id,
            mob_id, 
            infected_by: None,
            infected_when: None,
            infected_where: None,
            mobility,
            removed_when: None,
            status,
        }
    }
    
    pub fn infect(&mut self) {
        self.status = HealthStatus::Infected;
    }

    pub fn remove(&mut self) {
        self.status = HealthStatus::Removed;
    }

    pub fn vaccinate(&mut self) {
        self.status = HealthStatus::Vaccinated;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EpidemicAgentEnsemble {
    inner: Vec<EpidemicAgent>,
}

impl EpidemicAgentEnsemble {
    pub fn new(
        number_of_agents: u32, 
        chosen_ids_rho: &[(u32, u32, f64)],
    ) -> Self {
        let mut agent_ensemble = EpidemicAgentEnsemble { inner: Vec::new() };
        agent_ensemble.inner.reserve_exact(number_of_agents as usize);
    
        for count in 0..number_of_agents as usize {
            let epi_id = chosen_ids_rho[count].0;
            let mob_id = chosen_ids_rho[count].1;
            let rho = chosen_ids_rho[count].2;
            let status = HealthStatus::Susceptible;
            let agent = EpidemicAgent::new(epi_id, mob_id, status, rho);
            agent_ensemble.inner.push(agent);
        }
        agent_ensemble
    }

    pub fn inner(&self) -> &Vec<EpidemicAgent> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<EpidemicAgent> {
        &mut self.inner
    }

    pub fn into_inner(self) -> Vec<EpidemicAgent> {
        self.inner
    }

    pub fn mob_ids_from_epi_ids(&self) -> Vec<u32> {
        let mut mob_id_vec = vec![0; self.number_of_agents() as usize];
        for agent in self.inner() {
            let epi_id = agent.epi_id as usize;
            mob_id_vec[epi_id] = agent.mob_id;
        }
        mob_id_vec
    }

    pub fn epi_id_from_mob_id(&self) -> HashMap<u32, u32> {
        let mut epi_id_from_mob_id_hm = HashMap::new();
        for agent in self.inner() {
            let mob_id = agent.mob_id;
            let epi_id = agent.epi_id;
            epi_id_from_mob_id_hm.insert(mob_id, epi_id);
        }
        epi_id_from_mob_id_hm
    }

    pub fn number_of_agents(&self) -> u32 {
        self.inner.len() as u32
    }

    pub fn total_explorers(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.mobility >= 0.5 {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_returners(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.mobility < 0.5 {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_susceptible(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Susceptible {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_infected(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Infected {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_removed(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Removed {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_vaccinated(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Vaccinated {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_explorer_susceptible(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Susceptible {
                if agent.mobility >= 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_explorer_infected(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Infected {
                if agent.mobility >= 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_explorer_removed(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Removed {
                if agent.mobility >= 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_explorer_vaccinated(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Vaccinated {
                if agent.mobility >= 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_returner_susceptible(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Susceptible {
                if agent.mobility < 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_returner_infected(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Infected {
                if agent.mobility < 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_returner_removed(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Removed {
                if agent.mobility < 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn total_returner_vaccinated(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.status == HealthStatus::Vaccinated {
                if agent.mobility < 0.5 {
                    summa += 1;
                }
            }
        }
        summa
    }

    pub fn update_health_status(
        &mut self,
        chosen_individuals: &[u32],
        update_status: HealthStatus,
        t: u32,
        loc_id: u32,
        infector: u32,
        event_id: u32
    ) {
        let health_status = match update_status {
            HealthStatus::Susceptible => HealthStatus::Infected,
            HealthStatus::Infected => HealthStatus::Removed,
            HealthStatus::Removed => panic!(),
            HealthStatus::Vaccinated => panic!(),
        };

        for agent in chosen_individuals {
            self.inner_mut()[*agent as usize].status = health_status;
            if health_status == HealthStatus::Infected {
                self.inner_mut()[*agent as usize].infected_when = 
                Some((t + 1) as u32);
                self.inner_mut()[*agent as usize].infected_where = 
                Some(loc_id);
                self.inner_mut()[*agent as usize].infected_by = 
                Some(infector);
                self.inner_mut()[*agent as usize].event_id = Some(event_id);
            }
            if health_status == HealthStatus::Removed {
                self.inner_mut()[*agent as usize].removed_when = 
                Some((t + 1) as u32);
            }
        }
    }

    pub fn set_quarantines(&mut self, mpars: &MobilityPars) {
        let quarantine_model = mpars.quarantine_strategy;
        let quarantined_fraction = mpars.quarantined_fraction;
        let nagents = self.number_of_agents() as usize;
    
        match quarantine_model {
            QuarantineStrategy::Unmitigated => {},
            QuarantineStrategy::Random => {
                let quarantined_agents = 
                (quarantined_fraction * nagents as f64) as usize;
                let mut rng = rand::thread_rng();
                let samples: Vec<usize> = 
                (0..nagents).choose_multiple(&mut rng, quarantined_agents);
                
                for agent_id in samples {
                    self.inner_mut()[agent_id].mobility = 0.0;
                }
            },
            QuarantineStrategy::Explorers => {
                let mut quarantined_agents = 
                (quarantined_fraction * nagents as f64) as usize;
    
                for agent_id in 0..nagents {
                    if quarantined_agents == 0 {
                        break;
                    }
                    if self.inner_mut()[agent_id].mobility > 0.5 {
                        self.inner_mut()[agent_id].mobility = 0.0;
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
                    if self.inner_mut()[agent_id].mobility < 0.5 {
                        self.inner_mut()[agent_id].mobility = 0.0;
                        quarantined_agents -= 1;
                    }
                }
            },
            QuarantineStrategy::TopExplorers => {
                let quarantine_target = 
                (quarantined_fraction * nagents as f64) as usize;
    
                let indices: 
                Vec<usize> = (0..self.number_of_agents() as usize).collect();
    
                // Sort the indices based on rho in descending order
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    self
                    .inner()[b]
                    .mobility
                    .partial_cmp(&self.inner()[a].mobility)
                    .unwrap()
                    .reverse()
                });
    
                // Select the indices of the most attractive locations
                let selected_indices = &sorted_indices[..quarantine_target];
            
                // Set the Vaccinated status of the selected locations to true
                for &index in selected_indices {
                    self.inner_mut()[index].mobility = 0.0;
                }
            },
            QuarantineStrategy::TopReturners => {
                let quarantine_target = 
                (quarantined_fraction * nagents as f64) as usize;
    
                let indices: 
                Vec<usize> = (0..self.number_of_agents() as usize).collect();
    
                // Sort the indices based on rho in ascending order
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    self
                    .inner()[a]
                    .mobility
                    .partial_cmp(&self.inner()[b].mobility)
                    .unwrap()
                    .reverse()
                });
    
                // Select the indices of the most attractive locations
                let selected_indices = &sorted_indices[..quarantine_target];
            
                // Set the Vaccinated status of the selected locations to true
                for &index in selected_indices {
                    self.inner_mut()[index].mobility = 0.0;
                }
            },
        }
    }

    pub fn rollout_vaccines(&mut self, epars: &EpidemicPars) {
        let vaccine_model = epars.vaccination_strategy;
        let vaccinated_fraction = epars.vaccinated_fraction;
        let nagents = self.number_of_agents() as usize;
    
        match vaccine_model {
            VaccinationStrategy::Unmitigated => {},
            VaccinationStrategy::Random => {
                let vaccinated_agents = 
                (vaccinated_fraction * nagents as f64) as usize;
                let mut rng = rand::thread_rng();
                let samples: Vec<usize> = 
                (0..nagents).choose_multiple(&mut rng, vaccinated_agents);
                
                for agent_id in samples {
                    self.inner_mut()[agent_id].status = 
                    HealthStatus::Vaccinated;
                }
            },
            VaccinationStrategy::Explorers => {
                let mut vaccinated_agents = 
                (vaccinated_fraction * nagents as f64) as usize;
    
                for agent_id in 0..nagents {
                    if vaccinated_agents == 0 {
                        break;
                    }
                    if self.inner_mut()[agent_id].mobility > 0.5 {
                        self.inner_mut()[agent_id].status = 
                        HealthStatus::Vaccinated;
                        vaccinated_agents -= 1;
                    }
                }
            },
            VaccinationStrategy::Returners => {
                let mut vaccinated_agents = 
                (vaccinated_fraction * nagents as f64) as usize;
    
                for agent_id in 0..nagents {
                    if vaccinated_agents == 0 {
                        break;
                    }
                    if self.inner_mut()[agent_id].mobility < 0.5 {
                        self.inner_mut()[agent_id].status = 
                        HealthStatus::Vaccinated;
                        vaccinated_agents -= 1;
                    }
                }
            },
            VaccinationStrategy::TopExplorers => {
                let vaccination_target = 
                (vaccinated_fraction * nagents as f64) as usize;

                let indices: 
                Vec<usize> = (0..self.number_of_agents() as usize).collect();

                // Sort the indices based on rho in descending order
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    self
                    .inner()[b]
                    .mobility
                    .partial_cmp(&self.inner()[a].mobility)
                    .unwrap()
                });

                // Select the indices of the most attractive locations
                let selected_indices = &sorted_indices[..vaccination_target];
            
                // Set the Vaccinated status of the selected locations to true
                for &index in selected_indices {
                    self.inner_mut()[index].status = HealthStatus::Vaccinated;
                }
            },
            VaccinationStrategy::TopReturners => {
                let vaccination_target = 
                (vaccinated_fraction * nagents as f64) as usize;

                let indices: 
                Vec<usize> = (0..self.number_of_agents() as usize).collect();
                
                // Sort the indices based on rho in descending order
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    self
                    .inner()[b]
                    .mobility
                    .partial_cmp(&self.inner()[a].mobility)
                    .unwrap()
                    .reverse()
                });
    
                // Select the indices of the most attractive locations
                let selected_indices = &sorted_indices[..vaccination_target];
            
                // Set the Vaccinated status of the selected locations to true
                for &index in selected_indices {
                    self.inner_mut()[index].status = HealthStatus::Vaccinated;
                }
            },
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct EpidemicAgentOutput {
    pub epi_id: u32,
    pub mob_id: u32,
    pub infected_by: Option<u32>,
    pub infected_when: Option<u32>,
    pub infected_where: Option<u32>,
    pub removed_when: Option<u32>,
    pub rho: f64,
    pub status: HealthStatus,
    
}

impl EpidemicAgentOutput {
    pub fn new(agent: &EpidemicAgent) -> Self {
        let epi_id = agent.epi_id;
        let mob_id = agent.mob_id;
        let status = agent.status;
        let rho = agent.mobility;
        Self { 
            epi_id,
            mob_id,
            infected_by: None,
            infected_when: None,
            infected_where: None,
            removed_when: None,
            rho,
            status,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EpidemicAgentOutputEnsemble {
    inner: Vec<EpidemicAgentOutput>,
}

impl EpidemicAgentOutputEnsemble {
    pub fn new(agent_ensemble: &EpidemicAgentEnsemble) -> Self {
        let mut inner = Vec::new();
        let nagents = agent_ensemble.number_of_agents();
        for (_count, a) in (0..nagents).enumerate() {
            let agent = agent_ensemble.inner()[a as usize].clone();
            let ea_output = EpidemicAgentOutput::new(&agent);
            inner.push(ea_output);
        }
    
        Self { 
            inner,
        }
    }

    pub fn inner(&self) -> &Vec<EpidemicAgentOutput> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<EpidemicAgentOutput> {
        &mut self.inner
    }

    pub fn number_of_agents(&self) -> u32 {
        self.inner.len() as u32
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Output {
    pub agent: EpidemicAgentEnsemble,
    pub event: EventEnsemble,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OutputEnsemble {
    inner: Vec<Output>,
}

impl OutputEnsemble {
    pub fn new() -> Self {
        Self { 
            inner: Vec::new(),
        }
    }

    pub fn inner(&self) -> &Vec<Output> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<Output> {
        &mut self.inner
    }

    pub fn number_of_sims(&self) -> u32 {
        self.inner.len() as u32
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AgentGrid {
    inner: Vec<Vec<Option<Vec<u32>>>>,
}

impl AgentGrid {
    pub fn new(
        mobility_data: &[MobileAgentOutput],
        chosen_ids_rho: &[(u32, u32, f64)],
        nlocs: u32,
        t_max: u32,
    ) -> Self {
        let inner = vec![vec![None; t_max as usize]; nlocs as usize];
        let mut agent_grid = AgentGrid { inner };
    
        for &(epi_id, mob_id, _rho) in chosen_ids_rho {
            let mobile_agent = &mobility_data[mob_id as usize];
            for (time, location) in mobile_agent.trajectory.iter().enumerate() {
                if *location < nlocs && time < t_max as usize {
                    agent_grid.insert_agent(epi_id, *location, time);
                }
            }
        }
    
        agent_grid
    }

    pub fn inner(&self) -> &Vec<Vec<Option<Vec<u32>>>> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<Vec<Option<Vec<u32>>>> {
        &mut self.inner
    }

    pub fn number_of_locations(&self) -> usize {
        self.inner.len()
    }

    pub fn maximum_time(&self) -> usize {
        if let Some(row) = self.inner.get(0) {
            row.len()
        } else {
            0
        }
    }

    pub fn total_population(&self, l: usize, t: usize) -> Option<usize> {
        if let Some(row) = self.inner.get(l) {
            if let Some(cell) = row.get(t) {
                if let Some(population) = cell {
                    return Some(population.len());
                }
            }
        }
        None
    }

    pub fn total_susceptible(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32
    ) -> u32 {
        let mut total_susceptible = 0;
    
        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Susceptible {
                    total_susceptible += 1;
                }
            }
        }
        total_susceptible
    }

    pub fn total_infected(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32
    ) -> u32 {
        let mut total_infected = 0;
    
        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Infected {
                    total_infected += 1;
                }
            }
        }
        total_infected
    }

    pub fn total_removed(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32
    ) -> u32 {
        let mut total_removed = 0;
    
        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Removed {
                    total_removed += 1;
                }
            }
        }
        total_removed
    }

    pub fn total_vaccinated(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32
    ) -> u32 {
        let mut total_vaccinated = 0;
    
        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Vaccinated {
                    total_vaccinated += 1;
                }
            }
        }
        total_vaccinated
    }

    pub fn susceptible_population(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32,
    ) -> Vec<u32> {
        let mut pop_sus = Vec::new();

        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Susceptible {
                    pop_sus.push(*agent_id);
                }
            }
        }
        pop_sus
    }

    pub fn infected_population(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32,
    ) -> Vec<u32> {
        let mut pop_inf = Vec::new();
        
        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Infected {
                    pop_inf.push(*agent_id);
                }
            }
        }
        pop_inf
    }

    pub fn removed_population(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32,
    ) -> Vec<u32> {
        let mut pop_rem = Vec::new();

        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Removed {
                    pop_rem.push(*agent_id);
                }
            }
        }
        pop_rem
    }

    pub fn vaccinated_population(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32,
    ) -> Vec<u32> {
        let mut pop_vac = Vec::new();

        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {

                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Vaccinated {
                    pop_vac.push(*agent_id);
                }
            }
        }
        pop_vac
    }

    pub fn status_population(
        &self,
        agent_ensemble: &EpidemicAgentEnsemble,
        location: u32,
        time: u32,
    ) -> [Vec<u32>; 4] {
        let mut pop_status: [Vec<u32>; 4] = Default::default();

        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                match agent_ensemble.inner()[*agent_id as usize].status {
                    HealthStatus::Susceptible => {
                        pop_status[0].push(*agent_id);
                    }
                    HealthStatus::Infected => {
                        pop_status[1].push(*agent_id);
                    }
                    HealthStatus::Removed => {
                        pop_status[2].push(*agent_id);
                    }
                    HealthStatus::Vaccinated => {
                        pop_status[3].push(*agent_id);
                    }
                }
            }
        }
        pop_status
    }

    pub fn insert_agent(
        &mut self, 
        agent_id: u32, 
        location: u32, 
        time: usize
    ) {
        if let Some(agent_mob_ids) = 
        &mut self.inner[location as usize][time as usize] {
            agent_mob_ids.push(agent_id);
        } else {
            self.inner[location as usize][time as usize] = 
            Some(vec![agent_id]);
        }
    }

    pub fn set_quarantines(&mut self, _quarantined_agents: &[u32]) {
        todo!()
    }

    pub fn write_grid_file_name(
        &self,
        mob_file_name: &String,
        mpars: &MobilityPars,
    ) -> String {

        let head = "mgrid_".to_string(); // head is the same for all scenarios

        let subhead = match mpars.scenario {
            MobilityScenario::B1het => "msb1het_".to_string(),
            MobilityScenario::B1hom => "msb1hom_".to_string(),
            MobilityScenario::B2 => "msb2_".to_string(),
            MobilityScenario::Depr => "msdepr_".to_string(),
            MobilityScenario::Plain => "msplain_".to_string(),
            MobilityScenario::Real => "msreal_".to_string(),
            MobilityScenario::Uniform => "msuniform_".to_string(),
        };

        let mut time_chain = match mpars.selection {
            MobilitySelection::Pool => {
                let timestamp = Local::now();
                format!(
                    "tm{:02}{:02}{:02}{:02}{:02}{:02}",
                    timestamp.year() % 100,
                    timestamp.month(),
                    timestamp.day(),
                    timestamp.hour(),
                    timestamp.minute(),
                    timestamp.second(),
                )
            },
            MobilitySelection::Real => {
                format!("")
            }
            MobilitySelection::Set => {
                let tm_start = mob_file_name.find("tm").unwrap();
                let next_underscore = mob_file_name[tm_start..].find('_').unwrap_or(mob_file_name.len());
                let time_chain = mob_file_name[tm_start..tm_start + next_underscore].to_string();
                time_chain
            },
        };
        time_chain += "_";

        let nagent_chain = format!("na{}_", mpars.nagents);
    
        let quarantine_abbreviation = match mpars.quarantine_strategy {
            QuarantineStrategy::Explorers => "Exp",
            QuarantineStrategy::Random => "Ran",
            QuarantineStrategy::Returners => "Ret",
            QuarantineStrategy::TopExplorers => "TEx",
            QuarantineStrategy::TopReturners => "TRe",
            QuarantineStrategy::Unmitigated => "Unm",
        };
    
        let quar_chain = format!("qm{}_qf{}_", quarantine_abbreviation, mpars.quarantined_fraction);
    
        // Extract the remaining elements from mob_file_name
        let mob_file_name_without_time = if mpars.selection == MobilitySelection::Set {
            // Find the starting index of "gm"
            let gm_start = mob_file_name.find("gm").unwrap_or(0); 
            // Find the starting index of "tm" after "gm" (starting search from gm_start)
            let tm_start = mob_file_name[gm_start..].find("tm").unwrap_or(mob_file_name.len());
            // Find the starting index of "space" after "gm" (starting search from gm_start)
            let space_start = mob_file_name[gm_start..].find("space").unwrap_or(mob_file_name.len());
            // Extract the desired substring
            let extracted_part = &mob_file_name[gm_start..gm_start + tm_start];
            // Create the final string without the "tm" part
            format!("{}{}", extracted_part, &mob_file_name[gm_start + space_start..])
        } else {
            mob_file_name.to_string()
        };
    
        let mgrid_file_name = format!("{}{}{}{}{}{}", head, subhead, time_chain, nagent_chain, quar_chain, mob_file_name_without_time);

        mgrid_file_name
    }

    pub fn to_pickle(&self, mob_file_name: &String, mpars: &MobilityPars) {
        let serialized = 
        serde_pickle::to_vec(self, SerOptions::new()).unwrap();
        let file_name = self.write_grid_file_name(mob_file_name, mpars);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
    }

    pub fn introduce_infections(
        &self, 
        as_model: AgentSeedModel,
        seed_fraction: f64,
        t_epidemic: u32,
        agent_ensemble: &mut EpidemicAgentEnsemble, 
        event_ensemble: &mut EventEnsemble,
        chosen_ids_rho: &mut Vec<(u32, u32, f64)>,
        epicenters: &Vec<u32>,
    ) {    
        for epicenter in epicenters {
            if let Some(agent_ids) = 
            &self.inner[*epicenter as usize][t_epidemic as usize] {
                let pop = agent_ids.len() as u32;
                let nseeds = 
                (seed_fraction * agent_ids.len() as f64).round() as usize;
                let nseeds = if nseeds == 0 { 1 } else { nseeds };
    
                match as_model {
                    AgentSeedModel::Explorers => {
                        let explorers = agent_ids
                        .iter()
                        .filter(|&agent_mob_id| {
                            let agent_id = *agent_mob_id;
                            let rho = chosen_ids_rho
                                .iter()
                                .find(|&(id, _, _)| *id == agent_id)
                                .map(|(_, _, rho)| *rho);
                            rho.unwrap_or(0.0) >= 0.5
                        })
                        .collect::<Vec<_>>();

                        let selected_seeds = 
                        explorers
                        .choose_multiple(&mut rand::thread_rng(), nseeds);

                        for &seed in selected_seeds {
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .status = 
                            HealthStatus::Infected;
                            
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .event_id = 
                            Some(0);
                            
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .infected_when = 
                            Some(t_epidemic);

                            let epi = *epicenter as u32;
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .infected_where = 
                            Some(epi);
                        }

                        let event = Event {
                            id: 0,
                            infector_epi_id: 9999999,
                            infector_mob_id: 9999999,
                            infector_rho: 9.9,
                            location: *epicenter,
                            size: nseeds as u32,
                            susceptible_population: agent_ids.len() as u32,
                            infected_population: nseeds as u32,
                            total_population: pop as u32,
                            inf_pop_avg_rho: self.compute_infected_population_average_rho(agent_ensemble, *epicenter, 0),
                            t: 0,
                        };
                        event_ensemble.inner_mut().push(event);
                    }
                    AgentSeedModel::Random => {
                        let selected_seeds = 
                        agent_ids
                        .choose_multiple(&mut rand::thread_rng(), nseeds);

                        for &seed in selected_seeds {
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .status = 
                            HealthStatus::Infected;
                            
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .event_id = 
                            Some(0);
                            
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .infected_when = 
                            Some(t_epidemic);

                            let epi = *epicenter as u32;
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .infected_where = 
                            Some(epi);
                        }

                        let event = Event {
                            id: 1,
                            infector_epi_id: 9999999,
                            infector_mob_id: 9999999,
                            infector_rho: 9.9,
                            location: *epicenter,
                            size: nseeds as u32,
                            infected_population: nseeds as u32,
                            susceptible_population: agent_ids.len() as u32,
                            total_population: nseeds as u32,
                            inf_pop_avg_rho: self.compute_infected_population_average_rho(agent_ensemble, *epicenter, 0),
                            t: 0,
                        };
                        event_ensemble.inner_mut().push(event);
                    }
                    AgentSeedModel::Returners => {
                        let returners = agent_ids
                        .iter()
                        .filter(|&agent_id| {
                            let epi_id = *agent_id;
                            let rho = chosen_ids_rho
                                .iter()
                                .find(|&(id, _, _)| *id == epi_id)
                                .map(|(_, _, rho)| *rho);
                            rho.unwrap_or(0.0) < 0.5
                        })
                        .collect::<Vec<_>>();
                        
                        let selected_seeds = 
                        returners
                        .choose_multiple(&mut rand::thread_rng(), nseeds);
                        
                        for &seed in selected_seeds {
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .status = 
                            HealthStatus::Infected;
                            
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .event_id = 
                            Some(0);
                            
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .infected_when = 
                            Some(t_epidemic);

                            let epi = *epicenter as u32;
                            agent_ensemble
                            .inner_mut()[*seed as usize]
                            .infected_where = 
                            Some(epi);
                        }

                        let event = Event {
                            id: 0,
                            infector_epi_id: 9999999,
                            infector_mob_id: 9999999,
                            infector_rho: 9.9,
                            location: *epicenter,
                            size: nseeds as u32,
                            infected_population: nseeds as u32,
                            susceptible_population: agent_ids.len() as u32,
                            total_population: pop as u32,
                            inf_pop_avg_rho: self.compute_infected_population_average_rho(agent_ensemble, *epicenter, 0),
                            t: 0,
                        };
                        event_ensemble.inner_mut().push(event);
                    }
                    AgentSeedModel::TopExplorers => {
                        chosen_ids_rho.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

                        let selected_seeds: Vec<u32> = chosen_ids_rho
                        .iter()
                        .filter(|&(id, _, _)| agent_ids.contains(id))
                        .map(|&(id, _, _)| id)
                        .take(nseeds)
                        .collect();
    
                        for seed in selected_seeds {
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .status = 
                            HealthStatus::Infected;
                            
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .event_id = 
                            Some(0);
                            
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .infected_when = 
                            Some(t_epidemic);

                            let epi = *epicenter as u32;
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .infected_where = 
                            Some(epi);
                        }

                        let event = Event {
                            id: 0,
                            inf_pop_avg_rho: self.compute_infected_population_average_rho(agent_ensemble, *epicenter, 0),
                            infector_epi_id: 9999999,
                            infector_mob_id: 9999999,
                            infector_rho: 9.9,
                            location: *epicenter,
                            size: nseeds as u32,
                            t: 0,
                            infected_population: nseeds as u32,
                            susceptible_population: agent_ids.len() as u32,
                            total_population: pop as u32,
                        };
                        event_ensemble.inner_mut().push(event);
                    }
                    AgentSeedModel::TopReturners => {
                        chosen_ids_rho.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

                        let selected_seeds: Vec<u32> = chosen_ids_rho
                        .iter()
                        .filter(|&(id, _, _)| agent_ids.contains(id))
                        .map(|&(id, _, _)| id)
                        .take(nseeds)
                        .collect();
    
                        for seed in selected_seeds {
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .status = 
                            HealthStatus::Infected;
                            
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .event_id = 
                            Some(0);
                            
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .infected_when = 
                            Some(t_epidemic);

                            let epi = *epicenter as u32;
                            agent_ensemble
                            .inner_mut()[seed as usize]
                            .infected_where = 
                            Some(epi);
                        }

                        let event = Event {
                            id: 0,
                            inf_pop_avg_rho: self.compute_infected_population_average_rho(agent_ensemble, *epicenter, 0),
                            infector_epi_id: 9999999,
                            infector_mob_id: 9999999,
                            infector_rho: 9.9,
                            location: *epicenter,
                            size: nseeds as u32,
                            infected_population: nseeds as u32,
                            susceptible_population: agent_ids.len() as u32,
                            total_population: pop as u32,
                            t: 0,
                        };
                        event_ensemble.inner_mut().push(event);
                    }
                }
            }
        }
    }

    pub fn epidemic_reactions(
        &self, 
        agent_ensemble: &mut EpidemicAgentEnsemble, 
        total_infected: &mut f64,
        total_removed: &mut f64,
        event_ensemble: &mut EventEnsemble,
        epars: &EpidemicPars,
        rngs: &mut (&mut ThreadRng, &mut gsl_Rng),
        t: usize,
    ) {
        let nlocs = self.number_of_locations();
   
        for l in 0..nlocs {

            // Get agent population by health status at current time t
            let status_pop = 
            self.status_population(agent_ensemble, l as u32, t as u32);

            // Get populations
            let sus = status_pop[0].len();
            let inf = status_pop[1].len();
            let rem = status_pop[2].len();
            let vac = status_pop[3].len();
            let pop = sus + inf + rem + vac;

            if inf > 0 {
                // Compute infection probability
                let norm = f64::powf(pop as f64, epars.pseudomass_exponent);
                let base = 1.0 - epars.transmission_rate * 1.0 / norm;
                let p_total = 1.0 - f64::powi(base, inf as i32);

                // Get agent population by health status
                let mut susceptible_pop = status_pop[0].clone();
                let mut infected_pop1 = status_pop[1].clone();
                let mut infected_pop2 = status_pop[1].clone();
                // Choose an infector
                let (infector, _) = infected_pop1.partial_shuffle(rngs.0, 1);
                let infector_epi_id = infector[0];
                let infector_mob_id = agent_ensemble.inner()[infector[0] as usize].mob_id;
                // Produce new cases
                let new_cases = rngs.1.binomial(p_total, sus as u32);
                // Choose infected
                let (chosen_infected, _) = 
                susceptible_pop.partial_shuffle(rngs.0, new_cases as usize);

                // Register contagion event
                let event_id = event_ensemble.number_of_events();
                if new_cases != 0 {
                    let mut event = Event::new();
                    event.id = event_id;
                    event.location = l as u32;
                    event.t = (t + 1) as u32;
                    event.infector_epi_id = infector_epi_id;
                    event.infector_mob_id = infector_mob_id;
                    event.infector_rho = agent_ensemble.inner()[infector[0] as usize].mobility;
                    event.size = new_cases;
                    event.infected_population = inf as u32;
                    event.susceptible_population = sus as u32;
                    event.total_population = pop as u32;
                    event.inf_pop_avg_rho = self.compute_infected_population_average_rho(agent_ensemble, l as u32, t as u32);
                    event_ensemble.inner_mut().push(event);
                }

                // Exponentially distributed decay
                let p_rem = epars.removal_rate * 1.0;
                let new_removed = rngs.1.binomial(p_rem, inf as u32);
                // Choose removed
                let (chosen_removed, _) = 
                infected_pop2.partial_shuffle(rngs.0, new_removed as usize);

                // Update agents' health statuses
                agent_ensemble.update_health_status(
                    chosen_infected,
                    HealthStatus::Susceptible,
                    t as u32,
                    l as u32,
                    infector[0],
                    event_id,
                );
                agent_ensemble.update_health_status(
                    chosen_removed,
                    HealthStatus::Infected,
                    t as u32,
                    l as u32,
                    infector[0],
                    event_id,
                );

                *total_infected += new_cases as f64 - new_removed as f64;
                *total_removed += new_removed as f64;
            }
        }
    }

    pub fn compute_infected_population_average_rho(
        &self, 
        agent_ensemble: &EpidemicAgentEnsemble, 
        location: u32, 
        time: u32,
    ) -> f64 {
        let mut summa = 0.0;
        let mut total_infected = 0;
        
        if let Some(agent_ids) = 
        &self.inner[location as usize][time as usize] {
            for agent_id in agent_ids {
                if agent_ensemble
                .inner()[*agent_id as usize].status 
                == HealthStatus::Infected {
                    summa += agent_ensemble.inner()[*agent_id as usize].mobility;
                    total_infected += 1;
                }
            }
        }
        summa / total_infected as f64
    }
}

pub fn initialize_susceptible_population(
    pop_l: &mut Vec<Vec<u32>>, 
    agent_grid: &AgentGrid,
    agent_ensemble: &EpidemicAgentEnsemble,
) {
    for l in 0..agent_grid.number_of_locations() {
        let sus = agent_grid.total_susceptible(agent_ensemble, l as u32, 0);
        pop_l[l][0] = sus;
    }
}

pub fn initialize_infected_population(
    pop_l: &mut Vec<Vec<u32>>, 
    agent_grid: &AgentGrid,
    agent_ensemble: &EpidemicAgentEnsemble,
) {
    for l in 0..agent_grid.number_of_locations() {
        let inf = agent_grid.total_infected(agent_ensemble, l as u32, 0);
        pop_l[l][1] = inf;
    }
}

pub fn initialize_removed_population(
    pop_l: &mut Vec<Vec<u32>>, 
    agent_grid: &AgentGrid,
    agent_ensemble: &EpidemicAgentEnsemble,
) {
    for l in 0..agent_grid.number_of_locations() {
        let rem = agent_grid.total_removed(agent_ensemble, l as u32, 0);
        pop_l[l][2] = rem;
    }
}

pub fn initialize_vaccinated_population(
    pop_l: &mut Vec<Vec<u32>>, 
    agent_grid: &AgentGrid,
    agent_ensemble: &EpidemicAgentEnsemble,
) {
    for l in 0..agent_grid.number_of_locations() {
        let vac = agent_grid.total_vaccinated(agent_ensemble, l as u32, 0);
        pop_l[l][3] = vac;
    }
}

pub fn compute_susceptible_population(pop_l: &Vec<Vec<u32>>) -> u32 {
    let mut sus = 0;
    for l in 0..pop_l.len() {
        sus += pop_l[l][0];
    }
    sus
}

pub fn compute_infected_population(pop_l: &Vec<Vec<u32>>) -> u32 {
    let mut inf = 0;
    for l in 0..pop_l.len() {
        inf += pop_l[l][1];
    }
    inf
}

pub fn compute_removed_population(pop_l: &Vec<Vec<u32>>) -> u32 {
    let mut rem = 0;
    for l in 0..pop_l.len() {
        rem += pop_l[l][2];
    }
    rem
}

pub fn compute_vaccinated_population(pop_l: &Vec<Vec<u32>>) -> u32 {
    let mut vac = 0;
    for l in 0..pop_l.len() {
        vac += pop_l[l][3];
    }
    vac
}

pub fn sir_dynamics(
    agent_grid: &AgentGrid, 
    agent_ensemble: &mut EpidemicAgentEnsemble,
    event_ensemble: &mut EventEnsemble,
    epars: &EpidemicPars,
    rngs: &mut (&mut ThreadRng, &mut gsl_Rng),
) -> Output {
    // Unpack some parameters
    let expedited_escape = epars.expedited_escape;
    let nagents = epars.nagents;
    let escape_condition 
    = epars.escape_condition * nagents as f64;

    // Initialize loop conditions
    let mut total_removed = agent_ensemble.total_removed() as f64;
    let t_epi = epars.t_epidemic;
    let t_max = agent_grid.maximum_time(); 
    let mut total_infected = agent_ensemble.total_infected() as f64;
    let mut t = 0;

    // Run dynamical loop
    while (
        t < t_max - 1) && 
        total_infected > 0.0 && 
        !(total_removed > escape_condition && expedited_escape as bool) {

        // Epidemic reactions in every location
        if t >= t_epi as usize {
            agent_grid.epidemic_reactions(
                agent_ensemble,
                &mut total_infected,
                &mut total_removed,
                event_ensemble,
                epars,
                rngs,
                t,
            );
        }

        // Move to next time step
        t += 1;
        //println!("Time={t}, total infected={total_infected}, total removed={total_removed}");
    }

    // Info print
    //let total_removed = compute_removed_population(&pop_l);
    let prev_ratio = total_removed as f64 / epars.nagents as f64;
    let r0 = epars.transmission_rate / epars.removal_rate;
    let r_inf = sir_prevalence(r0);
    println!(
        "Time={t}. Incidence={total_infected}. 
        Prevalence ratio={prev_ratio}, Analytical={r_inf}"
    );

    Output {
        agent: agent_ensemble.clone(),
        event: event_ensemble.clone(),
    }
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
pub enum VaccinationStrategy {
    Explorers,
    Random,
    Returners,
    TopExplorers,
    TopReturners,
    Unmitigated,
}
