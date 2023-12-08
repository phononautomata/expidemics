use chrono::{Local, Datelike, Timelike};
use rand_distr::{Beta, Exp, Gamma, LogNormal, Normal, Distribution};
use rand::prelude::*;
use serde_json::{Value, from_str};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
//use std::io;
use std::io::{BufReader, BufWriter, Read};
use std::path::PathBuf;
use serde::Deserialize;
use serde_pickle::de::DeOptions;
use serde_pickle::ser::SerOptions;

use crate::analysis::DigestedOutput;
use crate::epidemic::{AgentGrid, EpidemicPars, Output, AgentSeedModel, LocationSeedModel, VaccinationStrategy};
use crate::mobility::MobilityScenario;
use crate::mobility::MobilitySelection;
use crate::mobility::{MobileAgentOutput, MobilityMetadata, LockdownStrategy, QuarantineStrategy};
use crate::mobility::{RhoDistributionModel, MobilityPars};
use crate::space::{TessellationModel, AttractivenessModel, Space};

pub fn build_bostonlattice_filename(space_hm: &HashMap<String, f64>) -> String {
    let head = format!("bl_");
    let chain = format!(
                    "DX{0}_DY{1}_LN0{2}_LT0{3}_rd{4}_x{5}_y{6}.json",
                    space_hm.get("DX").unwrap(),
                    space_hm.get("DY").unwrap(),
                    space_hm.get("LN0").unwrap(),
                    space_hm.get("LT0").unwrap(),
                    space_hm.get("rd").unwrap(),
                    space_hm.get("x").unwrap(),
                    space_hm.get("y").unwrap(),
                    );
    head + &chain
}

pub fn build_bostonlattice_census_filename(space_hm: &HashMap<String, f64>) -> String {
    let head = format!("bl_census_");
    let chain = format!(
                    "DX{0}_DY{1}_LN0{2}_LT0{3}_rd{4}_x{5}_y{6}.json",
                    space_hm.get("DX").unwrap(),
                    space_hm.get("DY").unwrap(),
                    space_hm.get("LN0").unwrap(),
                    space_hm.get("LT0").unwrap(),
                    space_hm.get("rd").unwrap(),
                    space_hm.get("x").unwrap(),
                    space_hm.get("y").unwrap(),
                    );
    head + &chain
}

pub fn build_bostonscatter_filename(space_hm: &HashMap<String, f64>) -> String {
    let head = format!("bs_");
    let chain = format!(
                    "LNE{0}_LNW{1}_LTN{2}_LTS{3}_nl{4}_rd{5}.json",
                    space_hm.get("LNE").unwrap(),
                    space_hm.get("LNW").unwrap(),
                    space_hm.get("LTN").unwrap(),
                    space_hm.get("LTS").unwrap(),
                    space_hm.get("nl").unwrap(),
                    space_hm.get("rd").unwrap(),
                    );
    head + &chain
}

pub fn build_grid_retriever_file_name(
    mpars: &MobilityPars, 
    rho_hm: &HashMap<String, f64>,
    space_str: &String,
    tessellation_flag: TessellationModel, 
    ) -> String {

        let head = match mpars.scenario {
            MobilityScenario::B1het => {
                format!("mgrid_msb1het_")
            },
            MobilityScenario::B1hom => {
                format!("mgrid_msb1hom_")
            },
            MobilityScenario::B2 => {
                format!("mgrid_msb2_")
            },
            MobilityScenario::Depr => {
                format!("mgrid_msdepr_")
            },
            MobilityScenario::Plain => {
                format!("mgrid_msplain_")
            },
            MobilityScenario::Real => {
                format!("mgrid_msreal_")
            }
            MobilityScenario::Uniform => {
                format!("mgrid_msuniform_")
            },
        };

        let time_chain = {
            let mob_data = match tessellation_flag {
                TessellationModel::BostonLattice => {
                    load_json_data("config_grid_bl_retriever")
                },
                TessellationModel::BostonScatter => {
                    load_json_data("config_grid_bs_retriever")
                },
                TessellationModel::SyntheticLattice => {
                    load_json_data("config_grid_rl_retriever")
                },
            };
            
            let timestamp = mob_data.get("tm").unwrap().as_str().unwrap();

            // Split the timestamp string into separate substrings
            let year = &timestamp[0..2];
            let month = &timestamp[2..4];
            let day = &timestamp[4..6];
            let hour = &timestamp[6..8];
            let minute = &timestamp[8..10];
            let second = &timestamp[10..12];
            // Create the formatted timestamp string
            format!(
                "tm{}{}{}{}{}{}_",
                year, month, day, hour, minute, second
            )
        };

        let chain_one = format!(
            "na{0}_",
            mpars.nagents,
        );

        let quarantine_abbreviation = match mpars.quarantine_strategy {
            QuarantineStrategy::Explorers => "Exp",
            QuarantineStrategy::Random => "Ran",
            QuarantineStrategy::Returners => "Ret",
            QuarantineStrategy::TopExplorers => "TEx",
            QuarantineStrategy::TopReturners => "TRe",
            QuarantineStrategy::Unmitigated => "Unm",
        };

        let chain_two = format!(
            "qm{0}_qf{1}_gm{2}_hw{3}_t{4}_rm{5}_",
            quarantine_abbreviation,
            mpars.quarantined_fraction,
            mpars.gamma,
            mpars.home_weight,    
            mpars.t_max,
            mpars.rho_model,
        );

        let rho_chain = match mpars.rho_model {
            RhoDistributionModel::Beta => { 
                format!(
                    "ra{0}_rb{1}_", 
                    rho_hm.get("alpha").unwrap(),
                    rho_hm.get("beta").unwrap()
                )
            },
            RhoDistributionModel::DeltaBimodal => {
                format!(
                    "ra{0}_rb{1}_rc{2}_", 
                    rho_hm.get("share").unwrap(), 
                    rho_hm.get("mode1").unwrap(), 
                    rho_hm.get("mode2").unwrap()
                )
            },
            RhoDistributionModel::Exponential => {
                format!(
                    "ra{0}_", 
                    rho_hm.get("rate").unwrap()
                )
            },
            RhoDistributionModel::Gamma => {
                format!(
                    "ra{0}_rb{1}_", 
                    rho_hm.get("shape").unwrap(),
                    rho_hm.get("scale").unwrap()
                )
            },
            RhoDistributionModel::Gaussian => {
                format!(
                    "ra{0}_rb{1}_", 
                    rho_hm.get("mean").unwrap(),
                    rho_hm.get("std_dev").unwrap()
                )
            },
            RhoDistributionModel::Homogeneous => {
                format!(
                    "ra{0}_", 
                    rho_hm.get("rho").unwrap()
                )
            },
            RhoDistributionModel::LogNormal => {
                format!(
                    "ra{0}_rb{1}_", 
                    rho_hm.get("mean").unwrap(),
                    rho_hm.get("variance").unwrap()
                )
            },
            RhoDistributionModel::NegativeBinomial => {
                format!(
                    "ra{0}_rb{1}_", 
                    rho_hm.get("mean").unwrap(),
                    rho_hm.get("variance").unwrap()
                )
            },
            RhoDistributionModel::Uniform => {
                format!(
                    "ra_",
                )
            },
        };

        let lockdown_abbreviation = match mpars.lockdown_strategy {
            LockdownStrategy::LeastAttractive => "LAt",
            LockdownStrategy::MostAttractive => "MAt",
            LockdownStrategy::Random => "Ran",
            LockdownStrategy::Unmitigated => "Unm",
        };
    
        let lock_chain = format!(
            "lm{0}_lf{1}_",
            lockdown_abbreviation,
            mpars.locked_fraction,
            );

        head + &time_chain + &chain_one + &chain_two + &rho_chain + &lock_chain + &space_str
}

pub fn build_grid_retriever_file_name_old(
    mpars: &MobilityPars,
    tessellation_flag: TessellationModel, 
    file_name: &str) -> String {

    let head = match mpars.scenario {
        MobilityScenario::B1het => {
            "mgrid_msb1het_"
        },
        MobilityScenario::B1hom => {
            "mgrid_msb1hom_"
        },
        MobilityScenario::B2 => {
            "mgrid_msb2_"
        },
        MobilityScenario::Depr => {
            "mgrid_msdepr_"
        },
        MobilityScenario::Plain => {
            "mgrid_msplain_"
        },
        MobilityScenario::Real => {
            "mgrid_msreal_"
        }
        MobilityScenario::Uniform => {
            "mgrid_msuniform_"
        }
    };

    let formatted_string = match tessellation_flag {
        TessellationModel::BostonLattice => {
            let filename_value = load_sorted_json_data(file_name);

            let mut formatted_string = String::from(head);

            if let Some(obj) = filename_value.as_object() {
                let mut chain_elements: Vec<(&str, String)> = Vec::new();
                chain_elements.push(("tm", String::new()));
                chain_elements.push(("na", String::new()));
                chain_elements.push(("qm", String::new()));
                chain_elements.push(("qf", String::new()));
                chain_elements.push(("gm", String::new()));
                chain_elements.push(("hw", String::new()));
                chain_elements.push(("t", String::new()));
                chain_elements.push(("rm", String::new()));
                chain_elements.push(("ra", String::new()));
                chain_elements.push(("rb", String::new()));
                chain_elements.push(("lm", String::new()));
                chain_elements.push(("lf", String::new()));
                chain_elements.push(("space", String::new()));
                chain_elements.push(("DX", String::new()));
                chain_elements.push(("DY", String::new()));
                chain_elements.push(("LN0", String::new()));
                chain_elements.push(("LT0", String::new()));
                chain_elements.push(("rd", String::new()));
                chain_elements.push(("x", String::new()));
                chain_elements.push(("y", String::new()));
                chain_elements.push(("ts", String::new()));
            
                for (key, value) in obj.iter() {
                    if let Some((_, chain_value)) = 
                    chain_elements.iter_mut().find(|(k, _)| *k == key) {
                        *chain_value = 
                        format!("{}{}", key, value.as_str().unwrap_or(""));
                    }
                }
            
                for (i, (_, value)) in chain_elements.iter().enumerate() {
                    if i > 0 {
                        formatted_string.push('_');
                    }
                    formatted_string.push_str(value);
                }
            }

            formatted_string.push_str(".pickle");
            formatted_string
        },
        TessellationModel::BostonScatter => {
            let filename_value = load_sorted_json_data(file_name);

            let mut formatted_string = String::from(head);

            if let Some(obj) = filename_value.as_object() {
                let mut chain_elements: Vec<(&str, String)> = Vec::new();
                chain_elements.push(("tm", String::new()));
                chain_elements.push(("na", String::new()));
                chain_elements.push(("qm", String::new()));
                chain_elements.push(("qf", String::new()));
                chain_elements.push(("gm", String::new()));
                chain_elements.push(("hw", String::new()));
                chain_elements.push(("t", String::new()));
                chain_elements.push(("rm", String::new()));
                chain_elements.push(("ra", String::new()));
                chain_elements.push(("rb", String::new()));
                chain_elements.push(("lm", String::new()));
                chain_elements.push(("lf", String::new()));
                chain_elements.push(("space", String::new()));
                chain_elements.push(("LNE", String::new()));
                chain_elements.push(("LNW", String::new()));
                chain_elements.push(("LTN", String::new()));
                chain_elements.push(("LTS", String::new()));
                chain_elements.push(("nl", String::new()));
                chain_elements.push(("rd", String::new()));
                chain_elements.push(("ts", String::new()));
            
                for (key, value) in obj.iter() {
                    if let Some((_, chain_value)) = 
                    chain_elements.iter_mut().find(|(k, _)| *k == key) {
                        *chain_value = 
                        format!("{}{}", key, value.as_str().unwrap_or(""));
                    }
                }
            
                for (i, (_, value)) in chain_elements.iter().enumerate() {
                    if i > 0 {
                        formatted_string.push('_');
                    }
                    formatted_string.push_str(value);
                }
            }

            formatted_string.push_str(".pickle");
            formatted_string
        },
        TessellationModel::SyntheticLattice => {
            let filename_value = load_sorted_json_data(file_name);

            let mut formatted_string = String::from(head);

            if let Some(obj) = filename_value.as_object() {
                let mut chain_elements: Vec<(&str, String)> = Vec::new();
                chain_elements.push(("tm", String::new()));
                chain_elements.push(("na", String::new()));
                chain_elements.push(("qm", String::new()));
                chain_elements.push(("qf", String::new()));
                chain_elements.push(("gm", String::new()));
                chain_elements.push(("hw", String::new()));
                chain_elements.push(("t", String::new()));
                chain_elements.push(("rm", String::new()));
                chain_elements.push(("ra", String::new()));
                chain_elements.push(("rb", String::new()));
                chain_elements.push(("lm", String::new()));
                chain_elements.push(("lf", String::new()));
                chain_elements.push(("space", String::new()));
                chain_elements.push(("am", String::new()));
                chain_elements.push(("aa", String::new()));
                chain_elements.push(("ab", String::new()));
                chain_elements.push(("bm", String::new()));
                chain_elements.push(("np", String::new()));
                chain_elements.push(("pm", String::new()));
                chain_elements.push(("x", String::new()));
                chain_elements.push(("y", String::new()));
                chain_elements.push(("ts", String::new()));
            
                for (key, value) in obj.iter() {
                    if let Some((_, chain_value)) = 
                    chain_elements.iter_mut().find(|(k, _)| *k == key) {
                        *chain_value = 
                        format!("{}{}", key, value.as_str().unwrap_or(""));
                    }
                }
            
                for (i, (_, value)) in chain_elements.iter().enumerate() {
                    if i > 0 {
                        formatted_string.push('_');
                    }
                    formatted_string.push_str(value);
                }
            }

            formatted_string.push_str(".pickle");
            formatted_string
        },  
    };

    formatted_string
}

pub fn build_mobility_retriever_file_name(
    mpars: &MobilityPars, 
    rho_hm: &HashMap<String, f64>,
    space_str: &String,
) -> String {
    let head = match mpars.selection {
        MobilitySelection::Pool => {
            format!("mpool_")
        },
        MobilitySelection::Real => {
            format!("mreal_")
        },
        MobilitySelection::Set => {
            format!("mset_")
        }
    };

    let subhead = match mpars.scenario {
        MobilityScenario::B1het => {
            format!("msb1het_")
        },
        MobilityScenario::B1hom => {
            format!("msb1hom_")
        },
        MobilityScenario::B2 => {
            format!("msb2_")
        },
        MobilityScenario::Depr => {
            format!("msdepr_")
        },
        MobilityScenario::Plain => {
            format!("msplain_")
        },
        MobilityScenario::Real => {
            format!("msreal_")
        }
        MobilityScenario::Uniform => {
            format!("msuniform_")
        },
    };

    let chain = format!(
        "gm{0}_hw{1}_t{2}_rm{3}_",
        mpars.gamma,
        mpars.home_weight,
        mpars.t_max,
        mpars.rho_model,
    );
        
    let rho_chain = match mpars.rho_model {
        RhoDistributionModel::Beta => { 
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("alpha").unwrap(),
                rho_hm.get("beta").unwrap()
            )
        },
        RhoDistributionModel::DeltaBimodal => {
            format!(
                "ra{0}_rb{1}_rc{2}_", 
                rho_hm.get("share").unwrap(), 
                rho_hm.get("mode1").unwrap(), 
                rho_hm.get("mode2").unwrap()
            )
        },
        RhoDistributionModel::Exponential => {
            format!(
                "ra{0}_", 
                rho_hm.get("rate").unwrap()
            )
        },
        RhoDistributionModel::Gamma => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("shape").unwrap(),
                rho_hm.get("scale").unwrap()
            )
        },
        RhoDistributionModel::Gaussian => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("mean").unwrap(),
                rho_hm.get("std_dev").unwrap()
            )
        },
        RhoDistributionModel::Homogeneous => {
            format!(
                "ra{0}_", 
                rho_hm.get("rho").unwrap()
            )
        },
        RhoDistributionModel::LogNormal => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("mean").unwrap(),
                rho_hm.get("variance").unwrap()
            )
        },
        RhoDistributionModel::NegativeBinomial => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("mean").unwrap(),
                rho_hm.get("variance").unwrap()
            )
        },
        RhoDistributionModel::Uniform => {
            format!(
                "ra_",
            )
        },
    };

    let lockdown_abbreviation = match mpars.lockdown_strategy {
        LockdownStrategy::LeastAttractive => "LAt",
        LockdownStrategy::MostAttractive => "MAt",
        LockdownStrategy::Random => "Ran",
        LockdownStrategy::Unmitigated => "Unm",
    };

    let lock_chain = format!(
        "lm{0}_lf{1}_",
        lockdown_abbreviation,
        mpars.locked_fraction,
        );

    let time_chain = match mpars.selection {
        MobilitySelection::Pool => {
            let timestamp = Local::now();
            format!(
                "tm{:02}{:02}{:02}{:02}{:02}{:02}_",
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
            let mob_data = load_json_data("config_mobility_retriever");
            let timestamp = mob_data.get("tm").unwrap().as_str().unwrap();

            // Split the timestamp string into separate substrings
            let year = &timestamp[0..2];
            let month = &timestamp[2..4];
            let day = &timestamp[4..6];
            let hour = &timestamp[6..8];
            let minute = &timestamp[8..10];
            let second = &timestamp[10..12];
            // Create the formatted timestamp string
            format!(
                "tm{}{}{}{}{}{}_",
                year, month, day, hour, minute, second
            )
        },
    };

    head + &subhead + &chain + &rho_chain + &lock_chain + &time_chain + &space_str
}

pub fn build_rho_list(mobility_data: &[MobileAgentOutput], chosen_agents: &[u32]) -> Vec<f64> {
    let mut rho_vec = Vec::new();
    for &agent_id in chosen_agents.iter() {
        let mobile_agent = &mobility_data[agent_id as usize];
        rho_vec.push(mobile_agent.rho);
    }
    rho_vec
}

pub fn build_id_rho_list(mobility_data: &[MobileAgentOutput], chosen_agents: &[u32]) -> Vec<(u32, f64)> {
    let mut rho_vec = Vec::new();
    for &agent_id in chosen_agents.iter() {
        let mobile_agent = &mobility_data[agent_id as usize];
        let rho_pair = (agent_id, mobile_agent.rho);
        rho_vec.push(rho_pair);
    }
    rho_vec
}

pub fn build_chosen_ids_rho(mobility_data: &[MobileAgentOutput], chosen_agents: &[u32]) -> Vec<(u32, u32, f64)> {
    let mut ids_rho_list = Vec::new();
    for (count, &agent_id) in chosen_agents.iter().enumerate() {
        let mobile_agent = &mobility_data[agent_id as usize];
        let tuple = (count as u32, agent_id, mobile_agent.rho);
        ids_rho_list.push(tuple);
    }
    ids_rho_list
}

/* Beware: this only applies to regular lattice space object */
pub fn build_space_retriever_file_name(tessellation_flag: TessellationModel, file_name: &str) -> String {
    let formatted_string = match tessellation_flag {
        TessellationModel::BostonLattice => {
            let filename_value = load_sorted_json_data(file_name);

            let mut formatted_string = String::from("space_");

            if let Some(obj) = filename_value.as_object() {
                let mut chain_elements: Vec<(&str, String)> = Vec::new();
                chain_elements.push(("DX", String::new()));
                chain_elements.push(("DY", String::new()));
                chain_elements.push(("LN0", String::new()));
                chain_elements.push(("LT0", String::new()));
                chain_elements.push(("rd", String::new()));
                chain_elements.push(("x", String::new()));
                chain_elements.push(("y", String::new()));
                chain_elements.push(("ts", String::new()));

                for (key, value) in obj.iter() {
                    if let Some((_, chain_value)) = 
                    chain_elements.iter_mut().find(|(k, _)| *k == key) {
                        *chain_value = 
                        format!("{}{}", key, value.as_str().unwrap_or(""));
                    }
                }

                for (i, (_, value)) in chain_elements.iter().enumerate() {
                    if i > 0 {
                        formatted_string.push('_');
                    }
                        formatted_string.push_str(value);
                }
            }

            formatted_string.push_str(".pickle");
            formatted_string
        },
        TessellationModel::BostonScatter => {
            let filename_value = load_sorted_json_data(file_name);

            let mut formatted_string = String::from("space_");

            if let Some(obj) = filename_value.as_object() {
                let mut chain_elements: Vec<(&str, String)> = Vec::new();
                chain_elements.push(("LNE", String::new()));
                chain_elements.push(("LNW", String::new()));
                chain_elements.push(("LTN", String::new()));
                chain_elements.push(("LTS", String::new()));
                chain_elements.push(("nl", String::new()));
                chain_elements.push(("rd", String::new()));
                chain_elements.push(("ts", String::new()));

                for (key, value) in obj.iter() {
                    if let Some((_, chain_value)) = 
                    chain_elements.iter_mut().find(|(k, _)| *k == key) {
                        *chain_value = 
                        format!("{}{}", key, value.as_str().unwrap_or(""));
                    }
                }

                for (i, (_, value)) in chain_elements.iter().enumerate() {
                    if i > 0 {
                        formatted_string.push('_');
                    }
                        formatted_string.push_str(value);
                }
            }

            formatted_string.push_str(".pickle");
            formatted_string
        },
        TessellationModel::SyntheticLattice => {
            let filename_value = load_sorted_json_data(file_name);

            let mut formatted_string = String::from("space_");

            if let Some(obj) = filename_value.as_object() {
                let mut chain_elements: Vec<(&str, String)> = Vec::new();
                chain_elements.push(("am", String::new()));
                chain_elements.push(("aa", String::new()));
                chain_elements.push(("ab", String::new()));
                chain_elements.push(("bm", String::new()));
                chain_elements.push(("np", String::new()));
                chain_elements.push(("pm", String::new()));
                chain_elements.push(("x", String::new()));
                chain_elements.push(("y", String::new()));
                chain_elements.push(("ts", String::new()));

                for (key, value) in obj.iter() {
                    if let Some((_, chain_value)) = 
                    chain_elements.iter_mut().find(|(k, _)| *k == key) {
                        *chain_value = 
                        format!("{}{}", key, value.as_str().unwrap_or(""));
                    }
                }

                for (i, (_, value)) in chain_elements.iter().enumerate() {
                    if i > 0 {
                        formatted_string.push('_');
                    }
                        formatted_string.push_str(value);
                }
            }

            formatted_string.push_str(".pickle");
            formatted_string
        },
    };

    formatted_string
}

pub fn collect_pickle_filenames(
    header: &str, 
    segment: Option<&str>
) -> Vec<String> {
    let mut directory = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
    directory.push("data");

    let file_names: Vec<String> = fs::read_dir(&directory)
        .expect("Failed to read directory")
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                let file_name = entry.file_name().into_string().ok()?;
                if file_name.starts_with(header) && segment.map_or(true, |seg| file_name.contains(seg)) {
                    Some(file_name)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    file_names
}

pub fn chosen_agents_to_pickle(
    chosen_agents: &Vec<u32>, 
    mob_file_name: &String, 
    mpars: &MobilityPars
) {
    let serialized = 
    serde_pickle::to_vec(chosen_agents, SerOptions::new()).unwrap();
        let file_name = write_chosen_agents_file_name(mob_file_name, mpars);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
}

pub fn convert_hm_value(
    space_hm: HashMap<String, Value>
) -> HashMap<String, f64> {
    space_hm
        .into_iter()
        .map(|(key, value)| {
            let f64_value = match value.as_f64() {
                Some(v) => v,
                None => panic!("Value conversion error for key: {}", key),
            };
            (key, f64_value)
        })
        .collect()
}

pub fn convert_value_to_string(
    map: HashMap<String, Value>,
    key: &str,
) -> Option<String> {
    map.get(key).and_then(|value| {
        // Use `to_string` to convert the value to a string
        Some(value.to_string())
    })
}

pub enum CustomValue {
    Float(f64),
    Str(String),
}

pub fn get_hm_value(space_hm: HashMap<String, Value>) -> HashMap<String, CustomValue> {
    space_hm
        .into_iter()
        .map(|(key, value)| {
            let converted_value = match value {
                Value::Number(num) => {
                    if let Some(f64_value) = num.as_f64() {
                        CustomValue::Float(f64_value)
                    } else {
                        panic!("Value conversion error for key: {}", key);
                    }
                }
                Value::String(string_value) => CustomValue::Str(string_value),
                _ => panic!("Value conversion error for key: {}", key),
            };
            (key, converted_value)
        })
        .collect()
}

pub fn id_rho_list_to_pickle(
    id_rho_list: &Vec<(u32, f64)>, 
    mob_file_name: &String, 
    mpars: &MobilityPars
) {
    let serialized = 
    serde_pickle::to_vec(id_rho_list, SerOptions::new()).unwrap();
        let file_name = write_chosen_agents_file_name(mob_file_name, mpars);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
}

pub fn chosen_ids_rho_to_pickle(
    ids_rho_list: &Vec<(u32, u32, f64)>, 
    mob_file_name: &String, 
    mpars: &MobilityPars
) {
    let serialized = 
    serde_pickle::to_vec(ids_rho_list, SerOptions::new()).unwrap();
        let file_name = write_chosen_agents_file_name(mob_file_name, mpars);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
}

pub fn load_databased_census_data(file_name: &str) -> Vec<f64> {
    let wd = std::env::current_dir().expect("Failed to get current directory");
    let mut path = PathBuf::from(wd);
    path.push("data");
    path.push(file_name);

    let mut file = File::open(&path).expect("Failed to open JSON file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read JSON file");

    let data: Vec<f64> = serde_json::from_str(&contents).expect("Failed to deserialize JSON data");

    data
}

#[derive(Debug, Deserialize)]
pub struct SpaceData {
    pub attractiveness: f64,
    pub i_index: u32,
    pub loc_id: u32,
    pub j_index: u32,
    pub lat: f64,
    pub lon: f64,
    pub x: f64,
    pub x_pbc: Option<f64>,
    pub y: f64,
    pub y_pbc: Option<f64>,   
}

pub fn load_databased_space_data(file_name: &str) -> Vec<SpaceData> {
    let wd = std::env::current_dir().expect("Failed to get current directory");
    let mut path = PathBuf::from(wd);
    path.push("data");
    path.push(file_name);

    let mut file = File::open(&path).expect("Failed to open JSON file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read JSON file");

    let data: Vec<SpaceData> = serde_json::from_str(&contents).expect("Failed to deserialize JSON data");

    data
}

pub fn _load_databased_space_data() -> Vec<HashMap<String, f64>> {
    let wd = env::current_dir().expect("Failed to get current directory");
    let mut path = PathBuf::from(wd);
    path.push("data");
    let file_name = "boston_space_data";
    path.push(file_name);

    let mut file = File::open(&path).expect("Failed to open pickle file");
    let mut serialized = Vec::new();
    file.read_to_end(&mut serialized).expect("Failed to read pickle file");

    let options = serde_pickle::DeOptions::new();
    serde_pickle::from_slice(&serialized, options)
    .expect("Failed to deserialize pickle data")
}

pub fn load_json_data(file_name: &str) -> HashMap<String, Value> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("config");
    path.push(format!("{}.json", file_name));

    // Open the file
    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    // Parse the JSON data into a HashMap
    let data: HashMap<String, Value> = 
    serde_json::from_str(&contents).unwrap();

    data
}

pub fn load_sorted_json_data(file_name: &str) -> Value {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("config");
    path.push(format!("{}.json", file_name));

    // Open the file
    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    // Parse the JSON data into a Value
    let data: Value = from_str(&contents).unwrap();
    data
}

pub fn merge_hashmaps(
    first_hm: HashMap<String, Value>,
    second_hm: HashMap<String, Value>,
) -> HashMap<String, Value> {
    let mut merged_hm = HashMap::new();

    for (key, value) in first_hm {
        merged_hm.insert(key, value);
    }

    for (key, value) in second_hm {
        merged_hm.insert(key, value);
    }

    merged_hm
}

pub fn mobility_metadata_from_pickle(
    file_name: &String
) -> MobilityMetadata {
    let file_name = write_mobility_metadata_file_name(file_name);
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");
    path.push(file_name);
    println!("{{&file_name}}");
    
    let bytes = std::fs::read(&path).expect("Failed to read pickle file");
    serde_pickle::from_slice(&bytes, Default::default())
    .expect("Failed to deserialize object")
}

pub fn retrieve_epidemic(file_name: &str) -> Vec<Output> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");
    path.push(file_name);
    if !path.exists() {
        return Vec::new();
    }

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Vec<Output>, _> = 
    serde_pickle::from_reader(r, options); 
    match result {
        Ok(data) => data,
        Err(_) => Vec::new(),
    }
}

pub fn retrieve_chosen_ids_rho(
    file_name: &String, 
) -> Vec<(u32, u32, f64)> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");

    let first_underscore_index = file_name.find('_').unwrap();
    let new_file_name = &file_name[first_underscore_index + 1..];
    let file_name = format!("mcage_{}", new_file_name); 
    path.push(file_name);

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Vec<(u32, u32, f64)>, _> = 
    serde_pickle::from_reader(r, options); 

    match result {
        Ok(data) => data,
        Err(_) => panic!(
            "Failed to retrieve rho vector from file: {}", 
            &path.display()
        ),
    }
}

pub fn retrieve_id_rho_list(
    file_name: &String, 
) -> Vec<(u32, f64)> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");

    let first_underscore_index = file_name.find('_').unwrap();
    let new_file_name = &file_name[first_underscore_index + 1..];
    let file_name = format!("mcage_{}", new_file_name); 
    path.push(file_name);

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Vec<(u32, f64)>, _> = 
    serde_pickle::from_reader(r, options); 

    match result {
        Ok(data) => data,
        Err(_) => panic!(
            "Failed to retrieve rho vector from file: {}", 
            &path.display()
        ),
    }
}

pub fn retrieve_ids_rho_list(
    file_name: &String, 
) -> Vec<(u32, u32, f64)> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");

    let first_underscore_index = file_name.find('_').unwrap();
    let new_file_name = &file_name[first_underscore_index + 1..];
    let file_name = format!("mcage_{}", new_file_name); 
    path.push(file_name);

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Vec<(u32, u32, f64)>, _> = 
    serde_pickle::from_reader(r, options); 

    match result {
        Ok(data) => data,
        Err(_) => panic!(
            "Failed to retrieve rho vector from file: {}", 
            &path.display()
        ),
    }
}

pub fn retrieve_grid(file_name: &String) -> AgentGrid {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");

    //let file_name = write_agent_grid_file_name(file_name);
    path.push(file_name);
    println!("{{&file_name}}");

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<AgentGrid, _> = 
    serde_pickle::from_reader(r, options); 

    match result {
        Ok(data) => data,
        Err(_) => panic!(
            "Failed to retrieve AgentGrid from file: {}", 
            &path.display()
        ),
    }
}

pub fn retrieve_mobility(file_name: &str) -> Vec<MobileAgentOutput> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");
    path.push(file_name);
    if !path.exists() {
        return Vec::new();
    }

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Vec<MobileAgentOutput>, _> = 
    serde_pickle::from_reader(r, options);
    match result {
        Ok(data) => data,
        Err(_) => Vec::new(),
    }
}

pub fn retrieve_rho_vec(
    file_name: &String, 
) -> Vec<f64> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");

    let first_underscore_index = file_name.find('_').unwrap();
    let new_file_name = &file_name[first_underscore_index + 1..];
    let file_name = format!("mrhod_{}", new_file_name); 
    path.push(file_name);

    let file = File::open(&path).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Vec<f64>, _> = 
    serde_pickle::from_reader(r, options); 

    match result {
        Ok(data) => data,
        Err(_) => panic!(
            "Failed to retrieve rho vector from file: {}", 
            &path.display()
        ),
    }
}

pub fn retrieve_space(file_name: &str) -> Space {
    let wd = env::current_dir().expect("Failed to get current directory");
    let mut path = PathBuf::from(wd);
    path.push("data");
    path.push(file_name);

    let mut file = File::open(&path).expect("Failed to open pickle file");
    let mut serialized = Vec::new();
    file.read_to_end(&mut serialized).expect("Failed to read pickle file");

    let options = serde_pickle::DeOptions::new();
    serde_pickle::from_slice(&serialized, options)
    .expect("Failed to deserialize pickle data")
}

pub fn rho_vec_to_pickle(
    rho_vec: &Vec<f64>, 
    mob_file_name: &String, 
    mpars: &MobilityPars
) {
    let serialized = 
    serde_pickle::to_vec(rho_vec, SerOptions::new()).unwrap();
        let file_name = write_rho_vec_file_name(mob_file_name, mpars);
        let path = "data/".to_owned() + &file_name;
        std::fs::write(path, serialized).unwrap();
}

pub fn sample_beta(a: f64, b: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let beta = Beta::new(a, b).unwrap();
    beta.sample(&mut rng)
}

pub fn sample_delta_bimodal(share: f64, mode1: f64, mode2: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    let rho: f64 = if trial < share { mode1 } else { mode2 };
    rho
}

pub fn sample_exponential(rate: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let exp = Exp::new(rate).unwrap();
    exp.sample(&mut rng)
}

pub fn sample_gamma(shape: f64, scale: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let gamma = Gamma::new(shape, scale).unwrap();
    gamma.sample(&mut rng)
}

pub fn sample_integer_power_law(beta: f64, _x_min: f64, x_max: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let mut sample = x_max + 1.0;
    while sample > x_max || sample == 0.0 {
        let u: f64 = rng.gen();
        sample = (u.powf(1.0 / -beta)).floor();
    }
    sample
}

pub fn sample_log_normal(mean: f64, variance: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let log_normal = LogNormal::new(mean, variance).unwrap();
    log_normal.sample(&mut rng)
}

pub fn sample_negative_binomial(_mean: f64, _variance: f64) -> f64 {
    1.0
}

pub fn sample_power_law(beta: f64, x_min: f64, x_max: f64) -> f64 {
    let mut rng = thread_rng();
    let rand_val: f64 = rng.gen();
    let denominator = 
    rand_val * (x_max.powf(-beta + 1.0) 
    - x_min.powf(-beta + 1.0)) 
    + x_min.powf(-beta + 1.0);
    denominator.powf(-1.0 / (-beta + 1.0))
}

pub fn sample_truncated_gaussian(mean: f64, std_dev: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let lower_bound = 0.0;
    let upper_bound = 1.0;
    let normal = Normal::new(mean, std_dev).unwrap();
    loop {
        let x = normal.sample(&mut rng);
        if x > lower_bound && x <= upper_bound {
            return x;
        }
    }
}

pub fn sample_uniform() -> f64 {
    let mut rng = rand::thread_rng();
    let rho: f64 = rng.gen();
    rho
}

pub fn save_digested_epidemic_output(
    digested_output: &DigestedOutput, 
    file_name: &str
) {
    // Get the current directory
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    // Append your specific directory
    path.push("data");
    // Append the file name
    path.push(file_name);
    // open the file in write mode
    let file = File::create(&path).expect("Unable to create file");
    let mut w = BufWriter::new(file);
    // Get a mutable reference to the underlying file
    let file_writer = &mut w;

    // serialize your data structure into the file
    let options = SerOptions::new();
    let result = 
    serde_pickle::to_writer(file_writer, &digested_output, options);
    match result {
        Ok(_) => println!(
            "Saved output_ensemble into {}", 
            path.to_str().unwrap()
        ),
        Err(error) => println!("Failed to serialize data: {}", error),
    }
}

pub fn save_epidemic_output_ensemble(
    output_ensemble: &Vec<Output>, 
    file_name: &str
) {
    // load existing data
    let mut existing_data = retrieve_epidemic(file_name);
    for output in output_ensemble {
        existing_data.push(output.clone());
    }
    // Get the current directory
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    // Append your specific directory
    path.push("data");
    // Append the file name
    path.push(file_name);
    // open the file in write mode
    let file = File::create(&path).expect("Unable to create file");
    let mut w = BufWriter::new(file);
    // Get a mutable reference to the underlying file
    let file_writer = &mut w;

    // serialize your data structure into the file
    let options = SerOptions::new();
    let result = 
    serde_pickle::to_writer(file_writer, &existing_data, options);
    match result {
        Ok(_) => println!(
            "Saved output_ensemble into {}", 
            path.to_str().unwrap()
        ),
        Err(error) => println!("Failed to serialize data: {}", error),
    }
}

pub fn save_mobility_output_ensemble(
    output_ensemble: &Vec<MobileAgentOutput>, 
    file_name: &str
) {
    // load existing data
    let mut existing_data = retrieve_mobility(file_name);
    for output in output_ensemble {
        existing_data.push(output.clone());
    }
    // Get the current directory
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    // Append your specific directory
    path.push("data");
    // Append the file name
    path.push(file_name);
    // open the file in write mode
    let file = File::create(&path).expect("Unable to create file");
    let mut w = BufWriter::new(file);
    // Get a mutable reference to the underlying file
    let file_writer = &mut w;

    // serialize your data structure into the file
    let options = SerOptions::new();
    let result = 
    serde_pickle::to_writer(file_writer, &existing_data, options);
    match result {
        Ok(_) => println!(
            "Saved output_ensemble into {}", 
            path.to_str().unwrap()
        ),
        Err(error) => println!("Failed to serialize data: {}", error),
    }
}

pub fn select_attractiveness_model(
    attr_enum: AttractivenessModel
) -> HashMap<String, Value> {
    match attr_enum {
        AttractivenessModel::DataBased => {
            let file_name = "config_attractiveness_databased";
            load_json_data(file_name)
        },
        AttractivenessModel::Exponential => {
            let file_name = "config_attractiveness_exponential";
            load_json_data(file_name)
        },
        AttractivenessModel::Gaussian => {
            let file_name = "config_attractiveness_gaussian";
            load_json_data(file_name)
        },
        AttractivenessModel::InverseSquare => {
            let file_name = "config_attractiveness_inverse";
            load_json_data(file_name)
        },
        AttractivenessModel::InverseSquareRoot => {
            let file_name = "config_attractiveness_inverseroot";
            load_json_data(file_name)
        },
        AttractivenessModel::Linear => {
            let file_name = "config_attractiveness_linear";
            load_json_data(file_name)
        },
        AttractivenessModel::PowerLaw => {
            let file_name = "config_attractiveness_powerlaw";
            load_json_data(file_name)
        },
        AttractivenessModel::RandomUniform => {
            let file_name = "config_attractiveness_randomuniform";
            load_json_data(file_name)
        },
        AttractivenessModel::Uniform => {
            let file_name = "config_attractiveness_uniform";
            load_json_data(file_name)
        },
    }
}

pub fn select_rho_model(
    rho_enum: RhoDistributionModel
) -> HashMap<String, Value> {
    match rho_enum {
        RhoDistributionModel::Beta => {
            let file_name = "config_rho_beta";
            load_json_data(file_name)
        },
        RhoDistributionModel::DeltaBimodal => {
            let file_name = "config_rho_deltabimodal";
            load_json_data(file_name)
        },
        RhoDistributionModel::Exponential => {
            let file_name = "config_rho_exponential";
            load_json_data(file_name)
        },
        RhoDistributionModel::Gamma => {
            let file_name = "config_rho_gamma";
            load_json_data(file_name)
        },
        RhoDistributionModel::Gaussian => {
            let file_name = "config_rho_gaussian";
            load_json_data(file_name)
        },
        RhoDistributionModel::Homogeneous => {
            let file_name = "config_rho_homogeneous";
            load_json_data(file_name)
        },
        RhoDistributionModel::LogNormal => {
            let file_name = "config_rho_lognormal";
            load_json_data(file_name)
        },
        RhoDistributionModel::NegativeBinomial => {
            let file_name = "config_rho_negativebinomial";
            load_json_data(file_name)
        },
        RhoDistributionModel::Uniform => {
            let file_name = "config_rho_uniform";
            load_json_data(file_name)
        },
    }
}

pub fn select_tessellation_model(
    tess_enum: TessellationModel
) -> HashMap<String, Value> {
    match tess_enum {
        TessellationModel::BostonLattice => {
            let file_name = "config_tessellation_bostonlattice";
            load_json_data(file_name)
        }
        TessellationModel::BostonScatter => {
            let file_name = "config_tessellaton_bostonscatter";
            load_json_data(file_name)
        },
        TessellationModel::SyntheticLattice => {
            let file_name = "config_tessellation_regularlattice";
            load_json_data(file_name)
        },
    }
}

pub fn sir_prevalence(r0: f64) -> f64 {
    let mut r_inf = 0.0;
    let mut guess = 0.8;
    let mut escape = 0;
    let mut condition = true;
    while condition {
        r_inf = 1.0 - (-r0 * guess).exp();
        if r_inf == guess {
            condition = false;
        }
        guess = r_inf;
        escape += 1;
        if escape > 10000 {
            r_inf = 0.0;
            condition = false;
        }
    }
    r_inf
}

pub fn trim_string_header(string_literal: &String) -> String {
    let first_underscore_index = string_literal.find('_').unwrap();
    string_literal[first_underscore_index + 1..].to_string()
}

pub fn write_chosen_agents_file_name(
    mob_file_name: &String, 
    mpars: &MobilityPars
) -> String {
    let head = "mcage_".to_string(); // head is the same for all scenarios

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

pub fn write_grid_file_name(mob_file_name: &String) -> String {
    let timestamp = Local::now();
        let timestamp_string = format!("tm{:02}{:02}{:02}{:02}{:02}{:02}",
                                       timestamp.year() % 100,
                                       timestamp.month(),
                                       timestamp.day(),
                                       timestamp.hour(),
                                       timestamp.minute(),
                                       timestamp.second());

    let first_underscore_index = mob_file_name.find('_').unwrap();
    let new_file_name = &mob_file_name[first_underscore_index + 1..];
    let mgrid_chain = format!("mgrid_{}", new_file_name);
    mgrid_chain + &timestamp_string
}

pub fn write_digested_epidemic_string_identifier(
    epars: &EpidemicPars,
    mobility_str: &String, 
) -> String {

    let head = format!("edig_");

    let agent_seed_abbreviation = match epars.agent_seed {
        AgentSeedModel::Explorers => "Ex",
        AgentSeedModel::Random => "Ran",
        AgentSeedModel::Returners => "Ret",
        AgentSeedModel::TopExplorers => "TEx",
        AgentSeedModel::TopReturners => "TRe",
    };

    let location_seed_abbreviation = match epars.location_seed {
        LocationSeedModel::LeastAttractive => "LAt",
        LocationSeedModel::MostAttractive => "MAt",
        LocationSeedModel::Random => "Ran",
    };

    let vaccination_strategy_abbreviation = match epars.vaccination_strategy {
        VaccinationStrategy::Explorers => "Exp",
        VaccinationStrategy::Random => "Ran",
        VaccinationStrategy::Returners => "Ret",
        VaccinationStrategy::TopExplorers => "TEx",
        VaccinationStrategy::TopReturners => "TRe",
        VaccinationStrategy::Unmitigated => "Unm",
    };

    let chain = format!(
        "as{}_ls{}_ne{}_ns{}_rr{}_sf{}_te{}_tr{}_vs{}_vf{}_",
        agent_seed_abbreviation,
        location_seed_abbreviation,
        epars.nepicenters,
        epars.nsims,
        epars.removal_rate,
        epars.seed_fraction,
        epars.t_epidemic,
        epars.transmission_rate,
        vaccination_strategy_abbreviation,
        epars.vaccinated_fraction,
    );

    head + &chain + &mobility_str
}

pub fn write_epidemic_string_identifier(
    epars: &EpidemicPars,
    mobility_str: &String, 
) -> String {

    let head = format!("edyna_");

    let agent_seed_abbreviation = match epars.agent_seed {
        AgentSeedModel::Explorers => "Ex",
        AgentSeedModel::Random => "Ran",
        AgentSeedModel::Returners => "Ret",
        AgentSeedModel::TopExplorers => "TEx",
        AgentSeedModel::TopReturners => "TRe",
    };

    let location_seed_abbreviation = match epars.location_seed {
        LocationSeedModel::LeastAttractive => "LAt",
        LocationSeedModel::MostAttractive => "MAt",
        LocationSeedModel::Random => "Ran",
    };

    let vaccination_strategy_abbreviation = match epars.vaccination_strategy {
        VaccinationStrategy::Explorers => "Exp",
        VaccinationStrategy::Random => "Ran",
        VaccinationStrategy::Returners => "Ret",
        VaccinationStrategy::TopExplorers => "TEx",
        VaccinationStrategy::TopReturners => "TRe",
        VaccinationStrategy::Unmitigated => "Unm",
    };

    let chain = format!(
        "as{}_ls{}_ne{}_ns{}_rr{}_sf{}_te{}_tr{}_vs{}_vf{}_",
        agent_seed_abbreviation,
        location_seed_abbreviation,
        epars.nepicenters,
        epars.nsims,
        epars.removal_rate,
        epars.seed_fraction,
        epars.t_epidemic,
        epars.transmission_rate,
        vaccination_strategy_abbreviation,
        epars.vaccinated_fraction,
    );

    head + &chain + &mobility_str
}

pub fn write_epidemic_data_file_name(epi_file_name: &String) -> String {
    let first_underscore_index = epi_file_name.find('_').unwrap();
    let new_file_name = &epi_file_name[first_underscore_index + 1..];
    format!("emeta_{}", new_file_name)
}

pub fn write_mobility_metadata_file_name(
    mob_file_name: &String
) -> String {
    //let first_underscore_index = mob_file_name.find('_').unwrap();
    //let new_file_name = &mob_file_name[first_underscore_index + 1..];
    format!("meta_{}", mob_file_name)
}

pub fn write_mobility_string_identifier(
    mpars: &MobilityPars, 
    rho_hm: &HashMap<String, f64>,
    space_str: &String,
) -> String {

    let head = match mpars.selection {
        MobilitySelection::Pool => {
            format!("mpool_")
        },
        MobilitySelection::Real => {
            format!("mreal_")
        }
        MobilitySelection::Set => {
            format!("mset_")
        }
    };

    let subhead = match mpars.scenario {
        MobilityScenario::B1het => {
            format!("msb1het_")
        },
        MobilityScenario::B1hom => {
            format!("msb1hom_")
        },
        MobilityScenario::B2 => {
            format!("msb2_")
        },
        MobilityScenario::Depr => {
            format!("msdepr_")
        },
        MobilityScenario::Plain => {
            format!("msplain_")
        },
        MobilityScenario::Real => {
            format!("msreal_")
        }
        MobilityScenario::Uniform => {
            format!("msuniform_")
        },
    };

    let chain = format!(
        "gm{0}_hw{1}_t{2}_rm{3}_",
        mpars.gamma,
        mpars.home_weight,
        mpars.t_max,
        mpars.rho_model,
    );
        
    let rho_chain = match mpars.rho_model {
        RhoDistributionModel::Beta => { 
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("alpha").unwrap(),
                rho_hm.get("beta").unwrap()
            )
        },
        RhoDistributionModel::DeltaBimodal => {
            format!(
                "ra{0}_rb{1}_rc{2}_", 
                rho_hm.get("share").unwrap(), 
                rho_hm.get("mode1").unwrap(), 
                rho_hm.get("mode2").unwrap()
            )
        },
        RhoDistributionModel::Exponential => {
            format!(
                "ra{0}_", 
                rho_hm.get("rate").unwrap()
            )
        },
        RhoDistributionModel::Gamma => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("shape").unwrap(),
                rho_hm.get("scale").unwrap()
            )
        },
        RhoDistributionModel::Gaussian => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("mean").unwrap(),
                rho_hm.get("std_dev").unwrap()
            )
        },
        RhoDistributionModel::Homogeneous => {
            format!(
                "ra{0}_", 
                rho_hm.get("rho").unwrap()
            )
        },
        RhoDistributionModel::LogNormal => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("mean").unwrap(),
                rho_hm.get("variance").unwrap()
            )
        },
        RhoDistributionModel::NegativeBinomial => {
            format!(
                "ra{0}_rb{1}_", 
                rho_hm.get("mean").unwrap(),
                rho_hm.get("variance").unwrap()
            )
        },
        RhoDistributionModel::Uniform => {
            format!(
                "ra_",
            )
        },
    };

    let lockdown_abbreviation = match mpars.lockdown_strategy {
        LockdownStrategy::LeastAttractive => "LAt",
        LockdownStrategy::MostAttractive => "MAt",
        LockdownStrategy::Random => "Ran",
        LockdownStrategy::Unmitigated => "Unm",
    };

    let lock_chain = format!(
        "lm{0}_lf{1}_",
        lockdown_abbreviation,
        mpars.locked_fraction,
        );

    let timestamp = Local::now();
    let time_chain = format!(
        "tm{:02}{:02}{:02}{:02}{:02}{:02}_",
        timestamp.year() % 100,
        timestamp.month(),
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second(),
    );

    head + &subhead + &chain + &rho_chain + &lock_chain + &time_chain + &space_str
}

pub fn write_rho_vec_file_name(
    mob_file_name: &String, 
    mpars: &MobilityPars
) -> String {
    let head = format!("mrhod_");
    let nagent_string = format!(
                        "na{0}_",
                        mpars.nagents,
                        );
    let quarantine_abbreviation = match mpars.quarantine_strategy {
        QuarantineStrategy::Explorers => "Exp",
        QuarantineStrategy::Random => "Ran",
        QuarantineStrategy::Returners => "Ret",
        QuarantineStrategy::TopExplorers => "TEx",
        QuarantineStrategy::TopReturners => "TRe",
        QuarantineStrategy::Unmitigated => "Unm",
    };
    let int_string = format!(
                        "qm{0}_qf{1}_",
                        quarantine_abbreviation,
                        mpars.quarantined_fraction,
                        );
    
    let timestamp = Local::now();
    let timestamp_string = format!(
        "tm{:02}{:02}{:02}{:02}{:02}{:02}_",
        timestamp.year() % 100,
        timestamp.month(),
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second(),
    );

    let front_chain = head + &nagent_string + &int_string + &timestamp_string;
    
    let first_underscore_index = mob_file_name.find('_').unwrap();
    let new_file_name = &mob_file_name[first_underscore_index + 1..];
    let mrhod_file_name = front_chain + new_file_name;
    mrhod_file_name   
}

pub fn remove_mobility_file() {
    // Get the current working directory
    let current_dir = env::current_dir().expect("Failed to get current directory.");

    // Iterate over the entries in the directory
    for entry in fs::read_dir(current_dir).expect("Failed to read directory.") {
        if let Ok(entry) = entry {
            if let Some(file_name) = entry.file_name().to_str() {
                // Check if the file name contains both "mdyn" and "msbl2"
                if file_name.contains("mdyn") && file_name.contains("msb2") {
                    let file_path = entry.path();
                    if let Err(err) = fs::remove_file(&file_path) {
                        eprintln!("Error deleting file '{}': {}", file_name, err);
                    } else {
                        println!("File '{}' deleted successfully", file_name);
                    }
                }
            }
        }
    }
}

pub fn find_second_max_indices(matrix: &Vec<Vec<i32>>) -> Vec<u32> {
    let mut second_max_indices = Vec::with_capacity(matrix.len());

    for row in matrix.iter() {
        let mut max = i32::MIN;
        let mut second_max = i32::MIN;
        let mut max_index = 0;  // Initialize to zero
        let mut second_max_index = 0;  // Initialize to zero

        for (index, &value) in row.iter().enumerate() {
            if value >= max {
                second_max = max;
                second_max_index = max_index as u32;
                max = value;
                max_index = index;
            } else if value > second_max {
                second_max = value;
                second_max_index = index as u32;
            }
        }

        second_max_indices.push(second_max_index);
    }

    second_max_indices
}

