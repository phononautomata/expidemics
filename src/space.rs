use chrono::{Local, Datelike, Timelike};
use rand::prelude::*;
use serde::{Serialize, Deserialize};
use serde_pickle::ser::SerOptions;
use std::collections::HashMap;
use strum_macros::Display;

use crate::{mobility::LockdownStrategy, utils::{load_databased_space_data, build_bostonlattice_filename, build_bostonscatter_filename}};

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
pub enum TessellationModel {
    BostonLattice,
    BostonScatter,
    SyntheticLattice,
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
pub enum PoleModel {
    RandomCartesian,
    RandomPolar,
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
pub enum BoundaryModel {
    Finite,
    Periodic,
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
pub enum AttractivenessModel {
    DataBased,
    Exponential,
    Gaussian,
    InverseSquare,
    InverseSquareRoot,
    Linear,
    PowerLaw,
    RandomUniform,
    Uniform,
}

#[derive(Serialize, Clone, Copy, Deserialize)]
pub struct SpaceFlags {
    pub attractiveness: Option<AttractivenessModel>,
    pub boundary: Option<BoundaryModel>,
    pub pole: Option<PoleModel>,
    pub tessellation: TessellationModel,
}

#[derive(Serialize, Clone, Copy, Deserialize)]
pub struct CellPars {
    pub i: u32,
    pub id: u32,
    pub j: u32,
}

#[derive(Serialize, Clone, Copy, Deserialize)]
pub struct Cell {
    pub attractiveness: Option<f64>,
    pub i_index: Option<u32>,
    pub id: u32,
    pub j_index: Option<u32>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub x: f64,
    pub x_pbc: Option<f64>,
    pub y: f64,
    pub y_pbc: Option<f64>,
}

impl Cell {
    pub fn new(
        spars: SpaceFlags, 
        space_hm: HashMap<String, f64>, cpars: CellPars
    ) -> Self {
        match spars.tessellation {
            TessellationModel::BostonLattice => {
                let intercell_distance = 
                *space_hm.get("intercell_distance").unwrap();

                // Read data

                Cell {
                    attractiveness: None,
                    i_index: Some(cpars.i),
                    id: cpars.i,
                    j_index: Some(cpars.j),
                    lat: None,
                    lon: None,
                    x: 
                    cpars.i as f64 
                    * intercell_distance 
                    + intercell_distance / 2.0,
                    x_pbc: 
                    Some(
                        cpars.i as f64 
                        * intercell_distance 
                        + intercell_distance / 2.0
                    ),
                    y: 
                    cpars.j as f64 
                    * intercell_distance 
                    + intercell_distance / 2.0,
                    y_pbc: 
                    Some(
                        cpars.j as f64 
                        * intercell_distance 
                        + intercell_distance / 2.0
                    ),
                }
            },
            TessellationModel::BostonScatter => {
                // Read data

                Cell {
                    attractiveness: None,
                    i_index: None,
                    id: cpars.id,
                    j_index: None,
                    lat: None,
                    lon: None,
                    x: 0.0,
                    x_pbc: None,
                    y: 0.0,
                    y_pbc: None,
                }
            },
            TessellationModel::SyntheticLattice => {
                let intercell_distance = 
                *space_hm.get("intercell_distance").unwrap();
                
                Cell {
                    attractiveness: None,
                    i_index: Some(cpars.i),
                    id: cpars.i,
                    j_index: Some(cpars.j),
                    lat: None,
                    lon: None,
                    x: 
                    cpars.i as f64 
                    * intercell_distance 
                    + intercell_distance / 2.0,
                    x_pbc: 
                    Some(
                        cpars.i as f64 
                        * intercell_distance 
                        + intercell_distance / 2.0
                    ),
                    y: 
                    cpars.j as f64 
                    * intercell_distance 
                    + intercell_distance / 2.0,
                    y_pbc: 
                    Some(
                        cpars.j as f64 
                        * intercell_distance 
                        + intercell_distance / 2.0
                    ),
                }
            }
        }
    }

    /// Get periodic boundary condition coordinates for the cell
    pub fn get_pbc_coordinates(
        &mut self, 
        space_hm: &HashMap<String, f64>,
        center_coords: &[f64; 2]
    ) {
        // Unpack parameters
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();
        let center_x = center_coords[0];
        let center_y = center_coords[1];
        let mut pbc_distance;
        let mut distance;

        // Get cell's FBC coordinates
        let cell_x = self.x;
        let cell_y = self.y;

        // Compute center-cell's FBC coordinates distance
        distance = f64::sqrt(f64::powi(cell_x - center_x, 2)
        + f64::powi(cell_y - center_y, 2));

        // Assume this is the minimum distance & assign PBC coordinates
        pbc_distance = distance;
        self.x_pbc = Some(cell_x);
        self.y_pbc = Some(cell_y);

        // Get cell's north replica coordinates
        let cell_x_n = self.x;
        let cell_y_n = y_cells as f64 + self.y;

        // Compute center-cell's north replica coordinates distance
        distance = f64::sqrt(f64::powi(cell_x_n - center_x, 2) 
        + f64::powi(cell_y_n - center_y, 2));

        // If this distance is shorter, assume new minimum distance & assign new PBC coordinates
        if distance < pbc_distance {
            pbc_distance = distance;
            self.x_pbc = Some(cell_x_n);
            self.y_pbc = Some(cell_y_n);
        }

        // Get cell's south replica coordinates
        let cell_x_s = self.x;
        let cell_y_s = -y_cells + self.y;

        // Compute center-cell's south replica coordiantes distance
        distance = f64::sqrt(f64::powi(cell_x_s - center_x, 2) 
        + f64::powi(cell_y_s - center_y, 2));

        // If this distance is shorter, assume new minimum distance & assign new PBC coordinates
        if distance < pbc_distance {
            pbc_distance = distance;
            self.x_pbc = Some(cell_x_s);
            self.y_pbc = Some(cell_y_s);
        }

        // Get cell's west replica coordinates
        let cell_x_w = -x_cells + self.x;
        let cell_y_w = self.y;

        // Compute center-cell's west replica coordiantes distance
        distance = f64::sqrt(f64::powi(cell_x_w - center_x, 2) 
        + f64::powi(cell_y_w - center_y, 2));

        // If this distance is shorter, assume new minimum distance & assign new PBC coordinates
        if distance < pbc_distance {
            pbc_distance = distance;
            self.x_pbc = Some(cell_x_w);
            self.y_pbc = Some(cell_y_w);
        }
        
        // Get cell's east replica coordinates
        let cell_x_e = x_cells + self.x;
        let cell_y_e = self.y;
        
        // Compute center-cell's east replica coordinates distance
        distance = f64::sqrt(f64::powi(cell_x_e - center_x, 2) 
        + f64::powi(cell_y_e - center_y, 2));
        
        // If this distance is shorter, assign definitive new PBC coordinates
        if distance < pbc_distance {
            self.x_pbc = Some(cell_x_e);
            self.y_pbc = Some(cell_y_e);
        }
    }

    /// Return the minium distance between two cells assuming PBC hold
    pub fn compute_cell_pair_minimum_distance(
        &self, 
        cell: &Cell, 
        space_hm: &HashMap<String, f64>
    ) -> f64 {
        // Unpack parameters
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();

        // Get cells' coordinates (referred to finite lattice coordinates)
        let x_i = self.x;
        let y_i = self.y;
        let x_j = cell.x;
        let y_j = cell.y;

        // Get cell's north replica coordinates
        let x_j_n = x_j;
        let y_j_n = y_cells + y_j;

        // Get cell's south replica coordinates
        let x_j_s = x_j;
        let y_j_s = -y_cells + y_j;

        // Get cell's west replica coordinates
        let x_j_w = -x_cells + x_j;
        let y_j_w = y_j;

        // Get cell's east replica coordinates
        let x_j_e = x_cells + x_j;
        let y_j_e = y_j;

        // Prepare vector of distances
        let mut distance_collection = Vec::new();

        // Compute distances & append
        let regular_distance_ij = f64::sqrt(f64::powi(x_i - x_j, 2) 
        + f64::powi(y_i - y_j, 2));
        distance_collection.push(regular_distance_ij);
        let north_distance_ij = f64::sqrt(f64::powi(x_i - x_j_n, 2) 
        + f64::powi(y_i - y_j_n, 2));
        distance_collection.push(north_distance_ij);
        let south_distance_ij = f64::sqrt(f64::powi(x_i - x_j_s, 2) 
        + f64::powi(y_i - y_j_s, 2));
        distance_collection.push(south_distance_ij);
        let west_distance_ij = f64::sqrt(f64::powi(x_i - x_j_w, 2) 
        + f64::powi(y_i - y_j_w, 2));
        distance_collection.push(west_distance_ij);
        let east_distance_ij = f64::sqrt(f64::powi(x_i - x_j_e, 2) 
        + f64::powi(y_i - y_j_e, 2));
        distance_collection.push(east_distance_ij);

        // Find minimum distance among computed distances
        let mut minimum_distance = regular_distance_ij;
        for dist in distance_collection {
            if dist < minimum_distance {
                minimum_distance = dist;
            }
        }
        minimum_distance
    }

    pub fn compute_cell_to_pole_minimum_distance(
        &self,
        pole: &Pole,
        space_hm: &HashMap<String, f64>,
    ) -> f64 {
        // Unpack parameters
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();

        // Get cell's coordinates (referred to finite lattice coordinates)
        let cell_x = self.x;
        let cell_y = self.y;
        let pole_x = pole.x;
        let pole_y = pole.y;

        // Get pole's north replica coordinates
        let pole_x_n = pole_x;
        let pole_y_n = y_cells + pole_y;

        // Get pole's south replica coordinates
        let pole_x_s = pole_x;
        let pole_y_s = -y_cells + pole_y;

        // Get pole's west replica coordinates
        let pole_x_w = -x_cells + pole_x;
        let pole_y_w = pole_y;

        // Get pole's east replica coordinates
        let pole_x_e = x_cells + pole_x;
        let pole_y_e = pole_y;
        // Prepare vector of distances
        let mut distance_collection = Vec::new();
        // Compute distances & append
        let regular_distance_ij =
            f64::sqrt(f64::powi(cell_x - pole_x, 2) 
            + f64::powi(cell_y - pole_y, 2));
        distance_collection.push(regular_distance_ij);
        let north_distance_ij =
            f64::sqrt(f64::powi(cell_x - pole_x_n, 2) 
            + f64::powi(cell_y - pole_y_n, 2));
        distance_collection.push(north_distance_ij);
        let south_distance_ij =
            f64::sqrt(f64::powi(cell_x - pole_x_s, 2) 
            + f64::powi(cell_y - pole_y_s, 2));
        distance_collection.push(south_distance_ij);
        let west_distance_ij =
            f64::sqrt(f64::powi(cell_x - pole_x_w, 2) 
            + f64::powi(cell_y - pole_y_w, 2));
        distance_collection.push(west_distance_ij);
        let east_distance_ij =
            f64::sqrt(f64::powi(cell_x - pole_x_e, 2) 
            + f64::powi(cell_y - pole_y_e, 2));
        distance_collection.push(east_distance_ij);

        // Find minimum distance among computed distances
        let mut minimum_distance = regular_distance_ij;
        for dist in distance_collection {
            if dist < minimum_distance {
                minimum_distance = dist;
            }
        }
        minimum_distance
    }
}

#[derive(Serialize, Clone, Deserialize)]
pub struct Pole {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub weight: f64,
    pub x_pbc: Option<f64>,
    pub y_pbc: Option<f64>,
}

impl Pole {
    /// Constructor method for Pole struct
    pub fn new(
        id: u32, 
        x: f64, 
        y: f64, 
        weight: f64, 
    ) -> Self {
        Pole {
            id,
            x,
            y,
            weight,
            x_pbc: Some(x),
            y_pbc: Some(y),
        }
    }

    /// Get periodic boundary condition coordinates for the pole
    pub fn get_pbc_coordinates(
        &mut self, 
        space_hm: &HashMap<String, f64>, 
        center_coords: &[f64; 2]
    ) {
        // Unpack parameters
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();
        let center_x = center_coords[0];
        let center_y = center_coords[1];
        let mut pbc_distance;
        let mut distance;

        // Get cell's FBC coordinates
        let pole_x = self.x;
        let pole_y = self.y;
        
        // Compute center-pole's FBC coordinates distance
        distance = 
        f64::sqrt(f64::powi(pole_x - center_x, 2) 
        + f64::powi(pole_y - center_y, 2));
        
        // Assume this is the minimum distance & assign PBC coordinates
        pbc_distance = distance;
        self.x_pbc = Some(pole_x);
        self.y_pbc = Some(pole_y);
        
        // Get pole's north replica coordinates
        let pole_x_n = self.x;
        let pole_y_n = y_cells + self.y;
        
        // Compute center-pole's north replica coordinates distance
        distance = 
        f64::sqrt(f64::powi(pole_x_n - center_x, 2) 
        + f64::powi(pole_y_n - center_y, 2));
        
        // If this distance is shorter, assume new minimum distance & assign new PBC coordinates
        if distance < pbc_distance {
            pbc_distance = distance;
            self.x_pbc = Some(pole_x_n);
            self.y_pbc = Some(pole_y_n);
        }
        
        // Get pole's south replica coordinates
        let pole_x_s = self.x;
        let pole_y_s = -y_cells + self.y;
        
        // Compute center-pole's south replica coordiantes distance
        distance = 
        f64::sqrt(f64::powi(pole_x_s - center_x, 2) 
        + f64::powi(pole_y_s - center_y, 2));
        
        // If this distance is shorter, assume new minimum distance & assign new PBC coordinates
        if distance < pbc_distance {
            pbc_distance = distance;
            self.x_pbc = Some(pole_x_s);
            self.y_pbc = Some(pole_y_s);
        }
        
        // Get pole's west replica coordinates
        let pole_x_w = -x_cells + self.x;
        let pole_y_w = self.y;
        
        // Compute center-pole's west replica coordiantes distance
        distance = 
        f64::sqrt(f64::powi(pole_x_w - center_x, 2) 
        + f64::powi(pole_y_w - center_y, 2));
        
        // If this distance is shorter, assume new minimum distance & assign new PBC coordinates
        if distance < pbc_distance {
            pbc_distance = distance;
            self.x_pbc = Some(pole_x_w);
            self.y_pbc = Some(pole_y_w);
        }
        
        // Get pole's east replica coordinates
        let pole_x_e = x_cells + self.x;
        let pole_y_e = self.y;
        
        // Compute center-pole's east replica coordinates distance
        distance = 
        f64::sqrt(f64::powi(pole_x_e - center_x, 2) 
        + f64::powi(pole_y_e - center_y, 2));
        
        // If this distance is shorter, assign definitive new PBC coordinates
        if distance < pbc_distance {
            self.x_pbc = Some(pole_x_e);
            self.y_pbc = Some(pole_y_e);
        }
    }

    /// Return the minimum distance under PBC conditions between given pole and cell
    pub fn compute_pole_to_cell_minimum_distance(
        &self,
        cell: &Cell,
        space_hm: &HashMap<String, f64>, 
    ) -> f64 {
        // Unpack parameters
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();
    
        // Get cells coordinates (referred to finite lattice coordinates)
        let pole_x = self.x;
        let pole_y = self.y;
        let cell_x = cell.x;
        let cell_y = cell.y;
        
        // Get north replica coordinates
        let cell_x_n = cell_x;
        let cell_y_n = y_cells + cell_y;
        
        // Get south replica coordinates
        let cell_x_s = cell_x;
        let cell_y_s = -y_cells + cell_y;
        
        // Get west replica coordinates
        let cell_x_w = -x_cells + cell_x;
        let cell_y_w = cell_y;
        
        // Get east replica coordinates
        let cell_x_e = x_cells as f64 + cell_x;
        let cell_y_e = cell_y;
        
        // Prepare vector of distances
        let mut distance_collection = Vec::new();
        
        // Compute distances & append
        let regular_distance_ij =
            f64::sqrt(f64::powi(pole_x - cell_x, 2) 
            + f64::powi(pole_y - cell_y, 2));
        distance_collection.push(regular_distance_ij);
        let north_distance_ij =
            f64::sqrt(f64::powi(pole_x - cell_x_n, 2) 
            + f64::powi(pole_y - cell_y_n, 2));
        distance_collection.push(north_distance_ij);
        let south_distance_ij =
            f64::sqrt(f64::powi(pole_x - cell_x_s, 2) 
            + f64::powi(pole_y - cell_y_s, 2));
        distance_collection.push(south_distance_ij);
        let west_distance_ij =
            f64::sqrt(f64::powi(pole_x - cell_x_w, 2) 
            + f64::powi(pole_y - cell_y_w, 2));
        distance_collection.push(west_distance_ij);
        let east_distance_ij =
            f64::sqrt(f64::powi(pole_x - cell_x_e, 2) 
            + f64::powi(pole_y - cell_y_e, 2));
        distance_collection.push(east_distance_ij);
        
        // Find minimum distance among computed distances
        let mut minimum_distance = regular_distance_ij;
        for dist in distance_collection {
            if dist < minimum_distance {
                minimum_distance = dist;
            }
        }
        minimum_distance
    }
    /// Return point field value for the pole-cell pair
    pub fn get_point_field(
        &self, 
        cell: &Cell, 
        spars: &SpaceFlags,
        space_hm: &HashMap<String, f64>, 
    ) -> f64 {
        match spars.attractiveness.unwrap() {
            AttractivenessModel::Uniform => self.uniform_point_field(),
            AttractivenessModel::Linear => {
                self.linear_point_field(cell, space_hm)
            },
            AttractivenessModel::Exponential => {
                self.exponential_point_field(cell, space_hm)
            },
            AttractivenessModel::Gaussian => {
                self.gaussian_point_field(cell, space_hm)
            },
            AttractivenessModel::InverseSquare => {
                self.inverse_square_point_field(cell, space_hm)
            }
            AttractivenessModel::InverseSquareRoot => {
                self.inverse_square_root_point_field(cell, space_hm)
            }
            AttractivenessModel::PowerLaw => {
                self.power_law_point_field(cell, space_hm)
            }
            _ => 1.0,
        }
    }

    pub fn uniform_point_field(&self) -> f64 {
        1.0
    }

    pub fn linear_point_field(
        &self, 
        cell: &Cell, 
        space_hm: &HashMap<String, f64>, 
    ) -> f64 {
        let intercept = *space_hm.get("intercept").unwrap();
        let slope = *space_hm.get("slope").unwrap();
        let r = self.compute_pole_to_cell_minimum_distance(cell, space_hm);
        let point_field = intercept - slope * r;
        point_field
    }

    pub fn gaussian_point_field(
        &self, 
        cell: &Cell, 
        space_hm: &HashMap<String, f64>, 
    ) -> f64 {
        let sigma_x = *space_hm.get("std_dev").unwrap();
        let r = self.compute_pole_to_cell_minimum_distance(cell, space_hm);
        let point_field = 1.0 * f64::exp(-r * r / sigma_x);
        point_field
    }

    pub fn exponential_point_field(
        &self, 
        cell: &Cell, 
        space_hm: &HashMap<String, f64>, 
    ) -> f64 {
        let rate = *space_hm.get("rate").unwrap();
        let r = self.compute_pole_to_cell_minimum_distance(cell, space_hm);
        let point_field = 1.0 * f64::exp(-rate * r);
        point_field
    }

    pub fn inverse_square_point_field(
        &self,
        cell: &Cell,
        space_hm: &HashMap<String, f64>,
    ) -> f64 {
        let epsilon = *space_hm.get("epsilon").unwrap();
        let r = self.compute_pole_to_cell_minimum_distance(cell, space_hm);
        let point_field = 1.0 / (r * r + epsilon);
        point_field
    }

    pub fn inverse_square_root_point_field(
        &self,
        cell: &Cell,
        space_hm: &HashMap<String, f64>,
    ) -> f64 {
        let epsilon = *space_hm.get("epsilon").unwrap();
        let r = self.compute_pole_to_cell_minimum_distance(cell, space_hm);
        let point_field = 1.0 / (r + epsilon);
        point_field
    }

    pub fn power_law_point_field(
        &self,
        cell: &Cell,
        space_hm: &HashMap<String, f64>,
    ) -> f64 {
        let exponent = *space_hm.get("exponent").unwrap();
        let epsilon = *space_hm.get("epsilon").unwrap();
        let r = self.compute_pole_to_cell_minimum_distance(cell, space_hm);
        let point_field = 1.0 / (f64::powf(r + epsilon, exponent * 0.5));
        point_field
    }

    pub fn random_uniform_field(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let point_field: f64 = rng.gen();
        point_field
    }
}

#[derive(Serialize, Deserialize)]
pub struct Space {
    pub flags: SpaceFlags,
    inner: Vec<Cell>,
    pub pars: HashMap<String, f64>,
    poles: Option<Vec<Pole>>,
}

impl Space {
    pub fn new(spars: &SpaceFlags, space_hm: &HashMap<String, f64>) -> Self {
        let mut space = Space {
            flags: *spars,
            inner: Vec::new(),
            pars: space_hm.clone(),
            poles: None,
        };

        match spars.tessellation {
            TessellationModel::BostonLattice => {
                let x_cells = *space_hm.get("x").unwrap() as u32;
                let y_cells = *space_hm.get("y").unwrap() as u32;
                let diff_y = *space_hm.get("DX").unwrap();
                let diff_x = *space_hm.get("DY").unwrap();
                let intercell_distance_x = diff_x / x_cells as f64;
                let intercell_distance_y = diff_y / y_cells as f64;
                // Load here the object
                let filename = build_bostonlattice_filename(space_hm);
                let space_df = load_databased_space_data(&filename);

                let mut l = 0; // Counter to generate cell_id
                for j in 0..y_cells {
                    for i in 0..x_cells {
                        let cell = Cell {
                            id: space_df[l].loc_id,
                            x:
                            i as f64 
                            * intercell_distance_x 
                            + intercell_distance_x / 2.0,
                            y: 
                            j as f64 
                            * intercell_distance_y 
                            + intercell_distance_y / 2.0,
                            i_index: Some(i),
                            j_index: Some(j),
                            lat: None, // Read from the object
                            lon: None, // Read from the object
                            x_pbc: 
                            Some(
                                i as f64 
                                * intercell_distance_x 
                                + intercell_distance_x / 2.0
                            ),
                            y_pbc: 
                            Some(
                                j as f64 
                                * intercell_distance_y 
                                + intercell_distance_y / 2.0
                            ),
                            attractiveness: Some(space_df[l].attractiveness),
                        };

                        space.inner.push(cell);
                        l += 1;
                    }
                }
            }
            TessellationModel::BostonScatter => { 
                let filename = build_bostonscatter_filename(space_hm);
                let space_df = load_databased_space_data(&filename);
                let nlocs = space_df.len();

                for l in 0..nlocs {
                    let cell = Cell {
                        id: space_df[l].loc_id,
                        x: space_df[l].x,
                        y: space_df[l].y,
                        i_index: Some(space_df[l].i_index),
                        j_index: Some(space_df[l].j_index),
                        lat: Some(space_df[l].lat),
                        lon: Some(space_df[l].lon),
                        x_pbc: None,
                        y_pbc: None,
                        attractiveness: Some(space_df[l].attractiveness),
                    };

                    space.inner.push(cell);
                }
             },
            TessellationModel::SyntheticLattice => {
                let x_cells = *space_hm.get("x_cells").unwrap() as u32;
                let y_cells = *space_hm.get("y_cells").unwrap() as u32;
                let intercell_distance = 
                *space_hm.get("intercell_distance").unwrap();

                let mut count = 0; // Counter to generate cell_id
                for j in 0..y_cells {
                    for i in 0..x_cells {
                        let cell = Cell {
                            id: count,
                            x: 
                            i as f64 
                            * intercell_distance 
                            + intercell_distance / 2.0,
                            y: 
                            j as f64 
                            * intercell_distance 
                            + intercell_distance / 2.0,
                            i_index: Some(i),
                            j_index: Some(j),
                            lat: None,
                            lon: None,
                            x_pbc: 
                            Some(
                                i as f64 
                                * intercell_distance 
                                + intercell_distance / 2.0
                            ),
                            y_pbc: 
                            Some(
                                j as f64 
                                * intercell_distance 
                                + intercell_distance / 2.0
                            ),
                            attractiveness: None,
                        };

                        space.inner.push(cell);
                        count += 1;
                    }
                }
            }
        }
        space
    }

    /// Return a reference to the vector of Cell objects
    pub fn inner(&self) -> &Vec<Cell> {
        &self.inner
    }

    /// Return a mutable reference to the vector of Cell objects
    pub fn inner_mut(&mut self) -> &mut Vec<Cell> {
        &mut self.inner
    }

    /// Return a reference to the vector of Pole objects
    pub fn poles(&self) -> &Vec<Pole> {
        self.poles.as_ref().expect("Poles are not available")
    }

    /// Return a mutable reference to the vector of Pole objects
    pub fn poles_mut(&mut self) -> &mut Vec<Pole> {
        self.poles.as_mut().expect("Poles are not available")
    }

    /// Return the total number of cells in the lattice
    pub fn number_of_cells(&self) -> u32 {
        self.inner.len() as u32
    }

    pub fn uniformize(&mut self) {
        let attractiveness_cutoff = 0.0000000001;
        for cell in self.inner_mut() {
            if cell.attractiveness.unwrap() > attractiveness_cutoff {
                cell.attractiveness = Some(1.0);
            }
        }
    }

    /// Return a HashMap where key is the cell's single-index corresponding with its position in the vector of Cell and
    /// value is the cell identifier
    pub fn hashmap_cell_id_from_single_index(&self) -> HashMap<u32, u32> {
        let mut hm_cell_id_from_single_index = HashMap::new();
        let mut i: u32 = 0;
        for cell in &self.inner {
            hm_cell_id_from_single_index.insert(cell.id, i);
            i += 1;
        }
        hm_cell_id_from_single_index
    }

    /// Return a HashMap where key is the cell identifier and
    /// value is the cell's single-index corresponding with its position in the vector of Cell
    pub fn hashmap_cell_single_index_from_id(&self) -> HashMap<u32, u32> {
        let mut hm_cell_single_from_index_id = HashMap::new();
        let mut i: u32 = 0;
        for cell in &self.inner {
            hm_cell_single_from_index_id.insert(i, cell.id);
            i += 1;
        }
        hm_cell_single_from_index_id
    }

    /// Return a HashMap where key is the cell ij-index pair and
    /// value is the cell identifier
    pub fn hashmap_cell_id_from_ij_index(&self) -> HashMap<(u32, u32), u32> {
        let mut hm_cell_id_from_ij_index = HashMap::new();
        for cell in &self.inner {
            let i = cell.i_index.unwrap();
            let j = cell.j_index.unwrap();
            let id = cell.id;
            hm_cell_id_from_ij_index.insert((i, j), id);
        }
        hm_cell_id_from_ij_index
    }

    /// Return a HashMap where key is the cell identifier and
    /// value is the cell ij-index pair
    pub fn hashmap_cell_ij_index_from_id(&self) -> HashMap<u32, (u32, u32)> {
        let mut hm_cell_ij_index_from_id = HashMap::new();
        for cell in &self.inner {
            let id = cell.id;
            let i = cell.i_index.unwrap();
            let j = cell.j_index.unwrap();
            hm_cell_ij_index_from_id.insert(id, (i, j));
        }
        hm_cell_ij_index_from_id
    }

    /// Return the lattice geometrical center given the square cell 
    /// characteristic dimension and total number of cells along 2d
    pub fn lattice_geometrical_center(
        &self, 
        d: f64, 
        x_cells: u32, 
        y_cells: u32
    ) -> [f64; 2] {
        let x_center = d * x_cells as f64 / 2.0;
        let y_center = d * y_cells as f64 / 2.0;
        [x_center, y_center]
    }

    /// Translate every cell's coordinates relative to a new given cell center 
    /// specified by the cell's identifier.
    /// Warning: this method should be adapted for the PBC case
    pub fn change_cell_origin_by_cell_id(&mut self, new_id: u32) {
        let new_x = self.inner[new_id as usize].x;
        let new_y = self.inner[new_id as usize].y;
        for cell in &mut self.inner {
            cell.x -= new_x;
            cell.y -= new_y;
        }
    }

    /// Translate every cell's coordinates relative to a new given cell center 
    /// specified by the cell's ij-index pair.
    /// Warning: this method should be adapted for the PBC case
    pub fn change_cell_origin_by_cell_ij_index(
        &mut self, 
        new_i: u32, 
        new_j: u32
    ) {
        let hm_cell_id_from_ij_index = self.hashmap_cell_id_from_ij_index();
        let new_id = hm_cell_id_from_ij_index.get(&(new_i, new_j)).unwrap();
        let new_x = self.inner[*new_id as usize].x;
        let new_y = self.inner[*new_id as usize].y;
        for cell in &mut self.inner {
            cell.x -= new_x;
            cell.y -= new_y;
        }
    }

    /// Translate every cell's coordinates relative to a new given cell center 
    /// specified by the cell's $(x, y)$ coordinates.
    /// Warning: this method should be adapted for the PBC case
    pub fn change_cell_origin_by_cell_coordinates(
        &mut self, 
        new_x: f64, 
        new_y: f64
    ) {
        for cell in &mut self.inner {
            cell.x -= new_x;
            cell.y -= new_y;
        }
    }

    /// Compute the periodic boundary condition (minimum-distance) coordinates
    pub fn get_pbc_coordinates(
        &mut self, 
        space_hm: &HashMap<String, f64>, 
        center_coords: &[f64; 2],
    ) {
        // Loop over cells in tessellation
        for cell in &mut self.inner {
            cell.get_pbc_coordinates(space_hm, center_coords);
        }
        // Loop over poles in tessellation
        if let Some(poles) = &mut self.poles {
            for pole in poles {
                pole.get_pbc_coordinates(space_hm, center_coords);
            }
        }
    }

    /// Compute the minimum distance coordinates under PBC conditions
    /// Warning: this method only works when the original coordinates are 
    /// referred to the bottom-left origin. It should be generalized.
    pub fn get_minimum_distance_coordinates(
        &mut self, 
        space_hm: &HashMap<String, f64>,
    ) {
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();

        for cell in &mut self.inner {
            if cell.x > x_cells / 2.0 {
                cell.x_pbc = Some(-x_cells + cell.x);
            } else {
                cell.x_pbc = Some(cell.x);
            }
            if cell.y > y_cells / 2.0 {
                cell.y_pbc = Some(-y_cells + cell.y);
            } else {
                cell.y_pbc = Some(cell.y);
            }
        }
    }

    /// Return the minimum distance between two cells in the tessellation 
    /// assuming the PBC case.
    pub fn compute_inter_cell_minimum_distance(
        &self,
        cell_i_id: u32,
        cell_j_id: u32,
        spars: &SpaceFlags,
        space_hm: &HashMap<String, f64>,
    ) -> f64 {
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();
    
        // Get cells
        let cell_i_index = *self
            .hashmap_cell_single_index_from_id()
            .get(&cell_i_id)
            .unwrap() as usize;
        let cell_j_index = *self
            .hashmap_cell_single_index_from_id()
            .get(&cell_j_id)
            .unwrap() as usize;
        let cell_i = self.inner()[cell_i_index];
        let cell_j = self.inner()[cell_j_index];
        
        // Get cells coordinates (referred to finite lattice coordinates)
        let x_i = cell_i.x;
        let y_i = cell_i.y;
        let x_j = cell_j.x;
        let y_j = cell_j.y;
        
        // Get north replica coordinates
        let x_j_n = x_j;
        let y_j_n = y_cells + y_j;
        
        // Get south replica coordinates
        let x_j_s = x_j;
        let y_j_s = -y_cells + y_j;
        
        // Get west replica coordinates
        let x_j_w = -x_cells + x_j;
        let y_j_w = y_j;
        
        // Get east replica coordinates
        let x_j_e = x_cells + x_j;
        let y_j_e = y_j;
        
        // Prepare vector of distances
        let mut distance_collection = Vec::new();
        
        // Compute regular distance, append but check boundary conditions if needed
        let regular_distance_ij = f64::sqrt(f64::powi(x_i - x_j, 2) 
        + f64::powi(y_i - y_j, 2));
        distance_collection.push(regular_distance_ij);
        if spars.boundary.unwrap() == BoundaryModel::Finite {
            return regular_distance_ij;
        }
        
        // Compute cardinal replica distances
        let north_distance_ij = f64::sqrt(f64::powi(x_i - x_j_n, 2) 
        + f64::powi(y_i - y_j_n, 2));
        distance_collection.push(north_distance_ij);
        let south_distance_ij = f64::sqrt(f64::powi(x_i - x_j_s, 2) 
        + f64::powi(y_i - y_j_s, 2));
        distance_collection.push(south_distance_ij);
        let west_distance_ij = f64::sqrt(f64::powi(x_i - x_j_w, 2) 
        + f64::powi(y_i - y_j_w, 2));
        distance_collection.push(west_distance_ij);
        let east_distance_ij = f64::sqrt(f64::powi(x_i - x_j_e, 2) 
        + f64::powi(y_i - y_j_e, 2));
        distance_collection.push(east_distance_ij);
        
        // Find minimum distance among computed distances
        let mut minimum_distance = regular_distance_ij;
        for dist in distance_collection {
            if dist < minimum_distance {
                minimum_distance = dist;
            }
        }
        minimum_distance
    }

    pub fn compute_inter_point_minimum_distance(
        &self,
        point_a: &[f64; 2],
        point_b: &[f64; 2],
        space_hm: &HashMap<String, f64>,
    ) -> f64 {
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();

        // Get north replica coordinates
        let x_j_n = point_b[0];
        let y_j_n = y_cells + point_b[1];
        
        // Get south replica coordinates
        let x_j_s = point_b[0];
        let y_j_s = -y_cells + point_b[1];
        
        // Get west replica coordinates
        let x_j_w = -x_cells + point_b[0];
        let y_j_w = point_b[1];
        
        // Get east replica coordinates
        let x_j_e = x_cells + point_b[0];
        let y_j_e = point_b[1];
        
        // Prepare vector of distances
        let mut distance_collection = Vec::new();
        
        // Compute distances & append
        let regular_distance_ij =
            f64::powi(point_a[0] - point_b[0], 2) 
            + f64::powi(point_a[1] - point_b[1], 2);
        distance_collection.push(regular_distance_ij);
        let north_distance_ij = f64::powi(point_a[0] - x_j_n, 2) 
        + f64::powi(point_a[1] - y_j_n, 2);
        distance_collection.push(north_distance_ij);
        let south_distance_ij = f64::powi(point_a[0] - x_j_s, 2) 
        + f64::powi(point_a[1] - y_j_s, 2);
        distance_collection.push(south_distance_ij);
        let west_distance_ij = f64::powi(point_a[0] - x_j_w, 2) 
        + f64::powi(point_a[1] - y_j_w, 2);
        distance_collection.push(west_distance_ij);
        let east_distance_ij = f64::powi(point_a[0] - x_j_e, 2) 
        + f64::powi(point_a[1] - y_j_e, 2);
        distance_collection.push(east_distance_ij);
        
        // Find minimum squared distance among computed distances
        let mut minimum_distance = regular_distance_ij;
        for dist in distance_collection {
            if dist < minimum_distance {
                minimum_distance = dist;
            }
        }
        minimum_distance
    }

    /// Add a pole with pre-defined characteristic paramters
    pub fn add_pole(&mut self, pole: Pole) {
        self.poles.as_mut().unwrap().push(pole);
    }

    /// Add a new pole to the tessellation given its characteristic parameters
    pub fn add_new_pole(&mut self, id: u32, x: f64, y: f64, weight: f64) {
        let pole = Pole::new(id, x, y, weight);
        if self.poles.is_none() {
            self.poles = Some(Vec::new());
        }
        self.poles.as_mut().unwrap().push(pole);
    }

    /// Set all given poles to tessellation.
    pub fn set_new_poles(
        &mut self,
        vec_x: &Vec<f64>,
        vec_y: &Vec<f64>,
        vec_weight: &Vec<f64>,
        space_hm: &HashMap<String, f64>,
    ) {
        let npoles = *space_hm.get("npoles").unwrap() as usize;
        for p in 0..npoles {
            let id = p as u32;
            let x = vec_x[p];
            let y = vec_y[p];
            let weight = vec_weight[p];
            self.add_new_pole(id, x, y, weight);
        }
    }

    /// Generate random uniform poles
    pub fn generate_random_cartesian_new_poles(
        &mut self, 
        space_hm: &HashMap<String, f64>,
    ) {
        let npoles = *space_hm.get("npoles").unwrap() as usize;
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();
    
        let mut rng = rand::thread_rng();
        for p in 0..npoles {
            let id = p as u32;
            let trial_x: f64 = rng.gen();
            let x = trial_x * x_cells;
            let trial_y: f64 = rng.gen();
            let y: f64 = trial_y * y_cells;
            let weight: f64 = 1.0;
            self.add_new_pole(id, x, y, weight);
        }
    }

    /// Generate random polar poles
    pub fn generate_random_polar_new_poles(
        &mut self, 
        space_hm: &HashMap<String, f64>,
    ) {
        let npoles = *space_hm.get("npoles").unwrap() as usize;
        let x_cells = *space_hm.get("x_cells").unwrap();
        let y_cells = *space_hm.get("y_cells").unwrap();
        let intercell_distance = *space_hm.get("intercell_distance").unwrap();

        let mut rng = rand::thread_rng();
        for p in 0..npoles {
            let id = p as u32;
            let trial_r: f64 = rng.gen();
            let r = trial_r * x_cells / 2.0;
            let trial_theta: f64 = rng.gen();
            let theta: f64 = trial_theta * (2.0 * std::f64::consts::PI);
            let weight: f64 = 1.0;
            let x = intercell_distance * x_cells / 2.0 + r * f64::cos(theta);
            let y = intercell_distance * y_cells / 2.0
            + r * f64::sin(theta);
            self.add_new_pole(id, x, y, weight);
        }
    }

    /// Create a superposition field in every cell by multipoles
    pub fn create_multipolar_field(&mut self) {
        let pars = self.pars.clone();
        let flags = self.flags;
        for cell in &mut self.inner {
            let mut total_field = 0.0;
            if let Some(poles) = &mut self.poles {
                for pole in poles {
                    total_field += pole.weight 
                    * pole.get_point_field(cell, &flags, &pars);
                }
            }
            cell.attractiveness = Some(total_field);
        }
    }

    /// Write file name for LatticeTessellation struct
    pub fn write_space_string_identifier(&self) -> String {
        let flags = self.flags;
        let pars = self.pars.clone();
        let head = format!("space_");

        let timestamp = Local::now();
        let timestamp_string = format!("{:02}{:02}{:02}{:02}{:02}{:02}",
                                       timestamp.year() % 100,
                                       timestamp.month(),
                                       timestamp.day(),
                                       timestamp.hour(),
                                       timestamp.minute(),
                                       timestamp.second());

        match self.flags.tessellation {
            TessellationModel::BostonLattice => {
                let bl_pars = RegularBostonPars::new(&pars);
                let chain = format!(
                    "DX{0}_DY{1}_LN0{2}_LT0{3}_rd{4}_x{5}_y{6}_ts{7}",
                    bl_pars.diff_x,
                    bl_pars.diff_y,
                    bl_pars.lon_ref,
                    bl_pars.lat_ref,
                    bl_pars.rounding,
                    bl_pars.x_cells,
                    bl_pars.y_cells,
                    timestamp_string,
                    );

                head + &chain
            },
            TessellationModel::BostonScatter => {
                let bs_pars = BostonScatterPars::new(&pars);
                let chain = format!(
                    "LNE{0}_LNW{1}_LTN{2}_LTS{3}_nl{4}_rd{5}_ts{6}",
                    bs_pars.lon_east,
                    bs_pars.lon_west,
                    bs_pars.lat_north,
                    bs_pars.lat_south,
                    bs_pars.nlocs,
                    bs_pars.rounding,
                    timestamp_string,
                    );
                head + &chain
            },
            TessellationModel::SyntheticLattice => {
                let rl_pars = RegularLatticePars::new(&flags, &pars);

                let boundary_abbreviation = match rl_pars.boundary {
                    BoundaryModel::Finite => "FBC",
                    BoundaryModel::Periodic => "PBC",
                };

                let pole_abbreviation = match rl_pars.pole {
                    PoleModel::RandomCartesian => "RC",
                    PoleModel::RandomPolar => "RP",
                };

                let chain = match flags.attractiveness.unwrap() {
                    AttractivenessModel::DataBased => {
                        let attract_abbreviation = "DaB";
                        format!("am{0}_x{1}_y{2}_ts{3}",
                                attract_abbreviation,
                                rl_pars.x_cells,
                                rl_pars.y_cells,
                                timestamp_string,
                        )
                    },
                    AttractivenessModel::Exponential => {
                        let attract_abbreviation = "Exp";
                        format!(
                            "am{0}_aa{1}_bm{2}_np{3}_pm{4}_x{5}_y{6}_ts{7}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::Gaussian => {
                        let attract_abbreviation = "Gau";
                        format!(
                            "am{0}_aa{1}_ab{2}_bm{3}_np{4}_pm{5}_x{6}_y{7}_ts{8}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            rl_pars.a_par2.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::InverseSquare => {
                        let attract_abbreviation = "InS";
                        format!(
                            "am{0}_aa{1}_bm{2}_np{3}_pm{4}_x{5}_y{6}_ts{7}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::InverseSquareRoot => {
                        let attract_abbreviation = "ISR";
                        format!(
                            "am{0}_aa{1}_ab{2}_bm{3}_np{4}_pm{5}_x{6}_y{7}_t{8}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            rl_pars.a_par2.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::Linear => {
                        let attract_abbreviation = "Lin";
                        format!(
                            "am{0}_aa{1}_ab{2}_bm{3}_np{4}_pm{5}_x{6}_y{7}_t{8}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            rl_pars.a_par2.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::PowerLaw => {
                        let attract_abbreviation = "PoL";
                        format!(
                            "am{0}_aa{1}_ab{2}_bm{3}_np{4}_pm{5}_x{6}_y{7}_ts{8}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            rl_pars.a_par2.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::RandomUniform => {
                        let attract_abbreviation = "RaU";
                        format!(
                            "am{0}_aa{1}_ab{2}_bm{3}_np{4}_pm{5}_x{6}_y{7}_ts{8}",
                            attract_abbreviation,
                            rl_pars.a_par1.unwrap(),
                            rl_pars.a_par2.unwrap(),
                            boundary_abbreviation,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    },
                    AttractivenessModel::Uniform => {
                        let attract_abbreviation = "Uni";
                        format!(
                            "am{0}_bm{1}_np{2}_pm{3}_x{4}_y{5}_ts{6}",
                            attract_abbreviation,
                            rl_pars.boundary,
                            rl_pars.npoles,
                            pole_abbreviation,
                            rl_pars.x_cells,
                            rl_pars.y_cells,
                            timestamp_string,
                        )
                    }, 
                };
                head + &chain
            },
        } 
    }

    pub fn to_pickle(&self) {
        let serialized = 
        serde_pickle::to_vec(self, SerOptions::new()).unwrap();
        let file_name = self.write_space_string_identifier();
        
        let path = "data/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }

    pub fn target_most_attractive_locations(&mut self, locked_fraction: f64) {
        // Create a Vec of indices for the locations
        let indices: Vec<usize> = (0..self.number_of_cells() as usize).collect();
    
        // Sort the indices based on point_field in descending order
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable_by(|&a, &b| {
            self.inner()[b].attractiveness.partial_cmp(&self.inner()[a].attractiveness).unwrap().reverse()
        });
    
        // Calculate the number of locations to target
        let target_count = (self.number_of_cells() as f64 * locked_fraction) as usize;
    
        // Select the indices of the most attractive locations
        let selected_indices = &sorted_indices[..target_count];
    
        // Set the locked status of the selected locations to true
        for &index in selected_indices {
            //self.inner_mut()[index].locked = true;
            self.inner_mut()[index].attractiveness = Some(0.000000001);
        }
    }
    
    pub fn target_least_attractive_locations(&mut self, locked_fraction: f64) {
        // Create a Vec of indices for the locations
        let indices: Vec<usize> = (0..self.number_of_cells() as usize).collect();
    
        // Sort the indices based on point_field in ascending order
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable_by(|&a, &b| {
            self.inner()[a].attractiveness.partial_cmp(&self.inner()[b].attractiveness).unwrap()
        });
    
        // Calculate the number of locations to target
        let target_count = (self.number_of_cells() as f64 * locked_fraction) as usize;
    
        // Select the indices of the least attractive locations
        let selected_indices = &sorted_indices[..target_count];
    
        // Set the locked status of the selected locations to true
        for &index in selected_indices {
            //self.inner_mut()[index].locked = true;
            self.inner_mut()[index].attractiveness = Some(0.000000001);
        }
    }
    
    pub fn target_random_attractive_locations(&mut self, locked_fraction: f64) {
        // Calculate the number of locations to target
        let target_count = (self.number_of_cells() as f64 * locked_fraction) as usize;
    
        // Create a Vec of indices for the locations
        let indices: Vec<usize> = (0..self.number_of_cells() as usize).collect();
    
        // Shuffle the indices randomly
        let mut rng = rand::thread_rng();
        let mut selected_indices = indices.clone();
        selected_indices.shuffle(&mut rng);
    
        // Select the first target_count indices as the random locations to target
        let selected_indices = &selected_indices[..target_count];
    
        // Set the locked status of the selected locations to true
        for &index in selected_indices {
            //self.inner_mut()[index].locked = true;
            self.inner_mut()[index].attractiveness = Some(0.000000001);
        }
    }
    
    pub fn set_lockdowns(
        &mut self, 
        lockdown_flag: LockdownStrategy, 
        locked_fraction: f64
    ) {
        match lockdown_flag {
            LockdownStrategy::Unmitigated => {},
            LockdownStrategy::Random => {
                self.target_random_attractive_locations(locked_fraction);
            },
            LockdownStrategy::LeastAttractive => {
                self.target_least_attractive_locations(locked_fraction);
            },
            LockdownStrategy::MostAttractive => {
                self.target_most_attractive_locations(locked_fraction);
            },
        }
    }

    pub fn gravity_model_od_rate_matrix(&self) -> Vec<Vec<f64>> {
        // Prepare data
        let boundary_flag = self.flags.boundary;
        let locs = self.number_of_cells();
        let mut od_rates = vec![vec![0.0; locs as usize]; locs as usize];
        
        // Loop over every cell in the tessellation to compute its related probabilities of exploring every other cell
        for cell_i in self.inner() {
            // Obtain origin cell attributes
            let i_id = cell_i.id;
            let attribute_i = cell_i.attractiveness;
            let x_i = cell_i.x;
            let y_i = cell_i.y;
            
            // Reset normalization constant for every origin cell
            let mut cumulative_j = 0.0;
            
            // Loop over every potential destination cell in the tessellation
            for cell_j in self.inner() {
                // Obtain destination cell information
                let j_id = cell_j.id;
                let attribute_j = cell_j.attractiveness;
                let x_j = cell_j.x;
                let y_j = cell_j.y;
                
                // Compute squared distance between origin & destination cells (subjected to boundary conditions)
                let squared_distance_ij;
                match boundary_flag.unwrap() {
                    BoundaryModel::Finite => {
                        squared_distance_ij = 
                        f64::powi(x_i - x_j, 2) + f64::powi(y_i - y_j, 2)
                    }
                    BoundaryModel::Periodic => {
                        squared_distance_ij = {
                            let distance_ij =
                                cell_i
                                .compute_cell_pair_minimum_distance(
                                    &cell_j, 
                                    &self.pars
                                );
                            distance_ij * distance_ij
                        }
                    }
                }
                
                // Compute probability of exploring j from i
                od_rates[i_id as usize][j_id as usize] =
                    attribute_i.unwrap() 
                    * attribute_j.unwrap() 
                    / squared_distance_ij;
                
                    // Compute normalization constant
                if i_id != j_id {
                    cumulative_j += od_rates[i_id as usize][j_id as usize];
                }
            }
            
            // Normalize probabilities
            for cell_j in self.inner() {
                let j_id = cell_j.id;
                od_rates[i_id as usize][j_id as usize] /= cumulative_j;
            }
        }
        od_rates
    }

    pub fn collect_attractiveness(&self) -> Vec<f64> {
        let mut a_vec = Vec::new();
        for cell in self.inner() {
            a_vec.push(cell.attractiveness.unwrap());
        }
        a_vec
    }
}

pub struct RegularBostonPars {
    pub diff_x: f64,
    pub diff_y: f64,
    pub lat_ref: f64,
    pub lon_ref: f64,
    pub rounding: u32,
    pub x_cells: u32,
    pub y_cells: u32,
}

impl RegularBostonPars {
    pub fn new(pars: &HashMap<String, f64>) -> Self {
        Self { 
            diff_x: *pars.get("DX").unwrap(),
            diff_y: *pars.get("DY").unwrap(),
            lat_ref: *pars.get("LT0").unwrap(),
            lon_ref: *pars.get("LN0").unwrap(),
            rounding: *pars.get("rd").unwrap() as u32,
            x_cells: *pars.get("x").unwrap() as u32,
            y_cells: *pars.get("y").unwrap() as u32,
        }
    }
}

pub struct BostonScatterPars {
    pub lat_north: f64,
    pub lat_south: f64,
    pub lon_east: f64,
    pub lon_west: f64,
    pub nlocs: u32,
    pub rounding: u32,
}

impl BostonScatterPars {
    pub fn new(pars: &HashMap<String, f64>) -> Self {
        Self { 
            lat_north: *pars.get("LTN").unwrap(),
            lat_south: *pars.get("LTS").unwrap(),
            lon_east: *pars.get("LNE").unwrap(),
            lon_west: *pars.get("LNW").unwrap(),
            nlocs: *pars.get("nl").unwrap() as u32,
            rounding: *pars.get("rd").unwrap() as u32,
        }
    }
}

pub struct RegularLatticePars {
    pub attractiveness: AttractivenessModel,
    pub boundary: BoundaryModel,
    pub a_par1: Option<f64>,
    pub a_par2: Option<f64>,
    pub npoles: u32,
    pub pole: PoleModel,
    pub x_cells: u32,
    pub y_cells: u32,
}

pub struct AttractivenessParameters {
    param1: f64,
    param2: f64,
}

impl RegularLatticePars {
    pub fn new(flags: &SpaceFlags, pars: &HashMap<String, f64>) -> Self {
        let a_pars = match flags.attractiveness.unwrap() {
            AttractivenessModel::DataBased => {
                AttractivenessParameters {
                    param1: 0.0,
                    param2: 0.0,
                }
            }
            AttractivenessModel::Exponential => {
                AttractivenessParameters {
                    param1: *pars.get("rate").unwrap(),
                    param2: 0.0,
                }
            },
            AttractivenessModel::Gaussian => {
                AttractivenessParameters {
                    param1: *pars.get("mean").unwrap(),
                    param2: *pars.get("std_dev").unwrap(), 
                }
            },
            AttractivenessModel::InverseSquare => {
                AttractivenessParameters {
                    param1: *pars.get("epsilon").unwrap(),
                    param2: 0.0, 
                }
            },
            AttractivenessModel::InverseSquareRoot => {
                AttractivenessParameters {
                    param1: *pars.get("exponent").unwrap(),
                    param2: *pars.get("epsilon").unwrap(),
                }
            },
            AttractivenessModel::Linear => {
                AttractivenessParameters {
                    param1: *pars.get("slope").unwrap(),
                    param2: *pars.get("intercept").unwrap(),
                }
            },
            AttractivenessModel::PowerLaw => {
                AttractivenessParameters {
                    param1: *pars.get("exponent").unwrap(),
                    param2: 0.0,
                }
            },
            AttractivenessModel::RandomUniform => {
                AttractivenessParameters {
                    param1: *pars.get("min").unwrap(),
                    param2: *pars.get("max").unwrap(),
                }
            },
            AttractivenessModel::Uniform => {
                AttractivenessParameters {
                    param1: 1.0,
                    param2: 0.0,
                }
            },
        };

        Self { 
            attractiveness: flags.attractiveness.unwrap(), 
            a_par1: Some(a_pars.param1),
            a_par2: Some(a_pars.param2),
            boundary: flags.boundary.unwrap(), 
            npoles: *pars.get("npoles").unwrap() as u32, 
            pole: flags.pole.unwrap(), 
            x_cells: *pars.get("x_cells").unwrap() as u32, 
            y_cells: *pars.get("x_cells").unwrap() as u32, 
        }
    }
}