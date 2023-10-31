use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Event {
    pub id: u32,
    pub location: u32,
    pub t: u32,
    pub size: u32,
    pub infected_population: u32,
    pub infector_epi_id: u32,
    pub infector_mob_id: u32,
    pub infector_rho: f64,
    pub susceptible_population: u32,
    pub total_population: u32,
    pub inf_pop_avg_rho: f64,
}

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

impl Event {
    pub fn new() -> Self {
        Self {
            id: 0,
            location: 0,
            t: 0,
            size: 0,
            infected_population: 0,
            infector_epi_id: 0,
            infector_mob_id: 0,
            infector_rho: 0.0,
            susceptible_population: 0,
            total_population: 0,
            inf_pop_avg_rho: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EventEnsemble {
    inner: Vec<Event>,
}

impl Default for EventEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEnsemble {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn inner(&self) -> &Vec<Event> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<Event> {
        &mut self.inner
    }

    pub fn number_of_events(&self) -> u32 {
        self.inner.len() as u32
    }
}
