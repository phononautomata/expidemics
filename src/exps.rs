use clap::Parser;
use rgsl::types::rng::Rng as gsl_Rng;
use rgsl::types::rng::RngType;
use std::time::SystemTime;

use crate::analysis::digest_raw_output;
use crate::utils::build_mobility_retriever_file_name;
use crate::utils::chosen_ids_rho_to_pickle;
use crate::{
    mobility::{
        HomeModel,
        MobilitySelection,
        MobilityScenario,
        LockdownStrategy,
        MobileAgent,
        MobileAgentOutput,
        MobilityMetadata,
        MobilityPars,
        QuarantineStrategy,
        RhoDistributionModel,
        build_mobility_parameter_grid,
        get_chosen_agents,
    },
    epidemic::{
        AgentGrid,
        AgentSeedModel,
        EpidemicAgentEnsemble,
        EpidemicPars,
        LocationSeedModel,
        VaccinationStrategy,
        set_epicenters,
        sir_dynamics,
    },
    event::EventEnsemble,
    space::{
        AttractivenessModel,
        BoundaryModel,
        PoleModel,
        Space,
        SpaceFlags,
        TessellationModel,
    },
    utils::{
        build_chosen_ids_rho,
        retrieve_chosen_ids_rho,
        save_digested_epidemic_output,
        write_digested_epidemic_string_identifier,
        build_grid_retriever_file_name,
        mobility_metadata_from_pickle,
        retrieve_grid,
        retrieve_mobility,
        save_mobility_output_ensemble,
        build_space_retriever_file_name,
        write_mobility_string_identifier,
        convert_hm_value,
        load_json_data,
        merge_hashmaps, 
        retrieve_space, 
        save_epidemic_output_ensemble,
        select_attractiveness_model, 
        select_tessellation_model, 
        select_rho_model,
        write_epidemic_string_identifier,
    },
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(long, value_parser, default_value = "random")]
    pub agent_seed_model: AgentSeedModel,
    #[clap(long, value_parser, default_value = "gaussian")]
    pub attractiveness_model: AttractivenessModel,
    #[clap(long, value_parser, default_value = "finite")]
    pub boundary_model: BoundaryModel,
    #[clap(long, value_parser, default_value_t = false)]
    pub config_flag: bool,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub escape_condition: f64,
    #[clap(long, value_parser, default_value_t = 3)]
    pub exp_flag: usize,
    #[clap(long, value_parser, default_value_t = false)]
    pub expedited_escape_flag: bool,
    #[clap(long, value_parser, default_value_t = 0.21)]
    pub gamma: f64,
    #[clap(long, value_parser, default_value = "random")]
    pub home_model: HomeModel,
    #[clap(long, value_parser, default_value_t = 25)]
    pub home_weight: u32,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub location_threshold: f64,
    #[clap(long, value_parser, default_value = "unmitigated")]
    pub lockdown_model: LockdownStrategy,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub locked_fraction: f64,
    #[clap(long, value_parser, default_value = "most-attractive")]
    pub location_seed_model: LocationSeedModel,
    #[clap(long, value_parser, default_value = "set")]
    pub mobility_selection_flag: MobilitySelection,
    #[clap(long, value_parser, default_value = "depr")]
    pub mobility_scenario_model: MobilityScenario,
    #[clap(long, value_parser, default_value_t = 1000)]
    pub nagents: u32,
    #[clap(long, value_parser, default_value_t = 1)]
    pub nepicenters: u32,
    #[clap(long, value_parser, default_value_t = 50)]
    pub nsims: u32,
    #[clap(long, value_parser, default_value = "random-cartesian")]
    pub pole_model: PoleModel,
    #[clap(long, value_parser, default_value_t = 1.0)]
    pub pseudomass_exponent: f64,
    #[clap(long, value_parser, default_value = "unmitigated")]
    pub quarantine_model: QuarantineStrategy,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub quarantined_fraction: f64,
    #[clap(long, value_parser, default_value_t = false)]
    pub raw_output_flag: bool,
    #[clap(long, value_parser, default_value_t = 0.1)]
    pub removal_rate: f64,
    #[clap(long, value_parser, default_value_t = 0.6)]
    pub rho: f64,
    #[clap(long, value_parser, default_value = "beta")]
    pub rho_distribution_model: RhoDistributionModel,
    #[clap(long, value_parser, default_value_t = 0.7)]
    pub seed_fraction: f64,
    #[clap(long, value_parser, default_value_t = 0)]
    pub t_epidemic: u32,
    #[clap(long, value_parser, default_value_t = 1200)]
    pub t_max: u32,
    #[clap(long, value_parser, default_value = "boston-lattice")]
    pub tessellation_model: TessellationModel,
    #[clap(long, value_parser, default_value_t = 0.12)]
    pub transmission_rate: f64,
    #[clap(long, value_parser, default_value = "unmitigated")]
    pub vaccination_model: VaccinationStrategy,
    #[clap(long, value_parser, default_value_t = 0.7)]
    pub vaccinated_fraction: f64,
}

pub fn generate_space(args: Args) {
    let spars = SpaceFlags {
        tessellation: args.tessellation_model,
        pole: Some(args.pole_model),
        boundary: Some(args.boundary_model),
        attractiveness: Some(args.attractiveness_model),
    };

    match args.tessellation_model {
        TessellationModel::BostonLattice => {
            let file_name = "config_tessellation_bostonlattice";
            let space_hm = load_json_data(file_name);
            let space_hm = convert_hm_value(space_hm);
    
            let space = Space::new(&spars, &space_hm);

            space.to_pickle();
        },
        TessellationModel::BostonScatter => {
            let file_name = "config_tessellation_bostonscatter";
            let space_hm = load_json_data(file_name);
            let space_hm = convert_hm_value(space_hm);

            let space = Space::new(&spars, &space_hm);

            space.to_pickle();
        },
        TessellationModel::SyntheticLattice => {
    
            let tess_hm = select_tessellation_model(args.tessellation_model);

            let attr_hm = select_attractiveness_model(args.attractiveness_model);

            let space_hm = merge_hashmaps(tess_hm, attr_hm);
            let space_hm = convert_hm_value(space_hm);

            let mut space = Space::new(&spars, &space_hm);

            match args.pole_model {
                PoleModel::RandomCartesian => {
                    space.generate_random_cartesian_new_poles(&space_hm)
                }
                PoleModel::RandomPolar => {
                    space.generate_random_polar_new_poles(&space_hm)
                }
            }

            match args.boundary_model {
                BoundaryModel::Finite => {},
                BoundaryModel::Periodic => {
                    let center_coords = [0.0; 2];
                    space.get_pbc_coordinates(&space_hm, &center_coords);
                },
            }

            space.create_multipolar_field();

            space.to_pickle();
        },
    }

    println!("Space object generated.");
}

pub fn run_sequential_depr_dynamics(args: Args) {
    let space_cfn = match args.tessellation_model {
        TessellationModel::BostonLattice => {
            "config_space_bl_retriever"
        },
        TessellationModel::BostonScatter => {
            "config_space_bs_retriever"
        },
        TessellationModel::SyntheticLattice => {
            "config_space_rl_retriever"
        },
    };

    let space_str = build_space_retriever_file_name(args.tessellation_model, space_cfn);
    let mut space: Space = retrieve_space(&space_str);

    let space_hm = match args.tessellation_model {
        TessellationModel::BostonLattice => {
            let file_name = "config_tessellation_bostonlattice";
            let space_hm = load_json_data(file_name);
            convert_hm_value(space_hm)
        },
        TessellationModel::BostonScatter => {
            let file_name = "config_tessellation_bostonscatter";
            let space_hm = load_json_data(file_name);
            convert_hm_value(space_hm)
        },
        TessellationModel::SyntheticLattice => {
            let tess_hm = select_tessellation_model(args.tessellation_model);
            let attr_hm = select_attractiveness_model(args.attractiveness_model);
            let space_hm = merge_hashmaps(tess_hm, attr_hm);
            convert_hm_value(space_hm)
        },
    };

    let mut mpars = match args.config_flag {
        false => {
            MobilityPars::new(args.gamma, args.home_weight, Some(args.location_threshold), args.lockdown_model, args.locked_fraction, args.nagents, 0, args.quarantine_model, args.quarantined_fraction, args.rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, args.t_max)
        },
        true => {
            let mob_cfn = "config_mobility";
            let mob_hm = load_json_data(mob_cfn);
            let mob_hm = convert_hm_value(mob_hm);

            let gamma = *mob_hm.get("gamma").unwrap();
            let home_weight = *mob_hm.get("home_weight").unwrap() as u32;
            let location_threshold = Some(*mob_hm.get("location_threshold").unwrap());
            let locked_fraction = *mob_hm.get("locked_fraction").unwrap();
            let nagents = *mob_hm.get("nagents").unwrap() as u32;
            let nlocs = 0;
            let quarantined_fraction = *mob_hm.get("quarantined_fraction").unwrap();
            let rho = *mob_hm.get("rho").unwrap();
            let t_max = *mob_hm.get("t_max").unwrap() as u32;

            MobilityPars::new(gamma, home_weight, location_threshold, args.lockdown_model, locked_fraction, nagents, nlocs, args.quarantine_model, quarantined_fraction, rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, t_max)
        },
    };

    mpars.nlocs = match space.flags.tessellation {
        TessellationModel::BostonLattice => {
            *space.pars.get("x").unwrap() as u32 
            * *space.pars.get("y").unwrap() as u32
        }
        TessellationModel::BostonScatter => {
            *space.pars.get("nlocs").unwrap() as u32
        }
        TessellationModel::SyntheticLattice => {
            *space.pars.get("x_cells").unwrap() as u32 
            * *space.pars.get("y_cells").unwrap() as u32
        }
    };

    if args.mobility_scenario_model == MobilityScenario::Uniform {
        space.uniformize();
    }

    space.set_lockdowns(args.lockdown_model, mpars.locked_fraction);

    let od_rates = space.gravity_model_od_rate_matrix();

    let rho_hm = select_rho_model(args.rho_distribution_model);
    let rho_hm = convert_hm_value(rho_hm);

    let mut homogeneous_flag = true;
    if args.mobility_scenario_model == MobilityScenario::B1hom {
        homogeneous_flag = true;
    } else if args.mobility_scenario_model == MobilityScenario::B1het {
        homogeneous_flag = false;
    }

    let mut output_ensemble = Vec::new();

    let nagents = mpars.nagents;
    for a in 0..nagents {
        if a % 100 == 0 {
            println!("Agent {0} is moving", a);
        }

        let mut agent = MobileAgent::new();
        agent.set_id(a as u32);

        agent.sample_rho(args.rho_distribution_model, &rho_hm);

        agent.set_home(&space, &space_hm, mpars.home_weight, args.home_model);

        match args.mobility_scenario_model {
            MobilityScenario::B1het => {
                agent.run_baseline1_dynamics(&space, &od_rates, &mpars, homogeneous_flag);
            },
            MobilityScenario::B1hom => {
                agent.run_baseline1_dynamics(&space, &od_rates, &mpars, homogeneous_flag);
            },
            MobilityScenario::B2 => {},
            MobilityScenario::Depr => {
                agent.run_depr_dynamics(&space, &od_rates, &mpars);
            },
            MobilityScenario::Plain => {
                agent.run_depr_dynamics(&space, &od_rates, &mpars);
            },
            MobilityScenario::Real => {},
            MobilityScenario::Uniform => {
                agent.run_depr_dynamics(&space, &od_rates, &mpars);
            },
        }

        let agent_output = MobileAgentOutput::new(&agent);
        output_ensemble.push(agent_output);
    }

    println!("Mobility run finished. Now saving...");

    let mdyna_str = 
    write_mobility_string_identifier(&mpars, &rho_hm, &space_str);
    save_mobility_output_ensemble(&output_ensemble, &mdyna_str);

    let mmetadata = MobilityMetadata::new(&mpars);
    mmetadata.to_pickle(&mdyna_str);
    println!("Mobility results saved");
}

pub fn generate_space_time_agent_grid(args: Args) {
    let space_cfn = match args.tessellation_model {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(args.tessellation_model, space_cfn);

    let mpars = match args.config_flag {
        false => {
            MobilityPars::new(args.gamma, args.home_weight, Some(args.location_threshold), args.lockdown_model, args.locked_fraction, args.nagents, 0, args.quarantine_model, args.quarantined_fraction, args.rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, args.t_max)
        },
        true => {
            let mob_cfn = "config_mobility";
            let mob_hm = load_json_data(mob_cfn);
            let mob_hm = convert_hm_value(mob_hm);

            let gamma = *mob_hm.get("gamma").unwrap();
            let home_weight = *mob_hm.get("home_weight").unwrap() as u32;
            let location_threshold = Some(*mob_hm.get("location_threshold").unwrap());
            let locked_fraction = *mob_hm.get("locked_fraction").unwrap();
            let nagents = *mob_hm.get("nagents").unwrap() as u32;
            let nlocs = 0;
            let quarantined_fraction = *mob_hm.get("quarantined_fraction").unwrap();
            let rho = *mob_hm.get("rho").unwrap();
            let t_max = *mob_hm.get("t_max").unwrap() as u32;

            MobilityPars::new(gamma, home_weight, location_threshold, args.lockdown_model, locked_fraction, nagents, nlocs, args.quarantine_model, quarantined_fraction, rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, t_max)
        },
    };

    let rho_hm = select_rho_model(args.rho_distribution_model);
    let rho_hm = convert_hm_value(rho_hm);

    let mob_str = build_mobility_retriever_file_name(&mpars, &rho_hm, &space_str);
    write_mobility_string_identifier(&mpars, &rho_hm, &space_str);
    let mmetadata = mobility_metadata_from_pickle(&mob_str);
    let mobility_data = retrieve_mobility(&mob_str);
 
    let nchosen = mpars.nagents as usize;
    let chosen_agents = get_chosen_agents(&mobility_data, nchosen);

    let chosen_ids_rho = build_chosen_ids_rho(&mobility_data, &chosen_agents);

    let nlocs = mmetadata.pars.nlocs;
    let t_max = mpars.t_max;
    let agent_grid = AgentGrid::new(
        &mobility_data, 
        &chosen_ids_rho, 
        nlocs, 
        t_max
    );

    // TODO: Set quarantines (if enabled)

    agent_grid.to_pickle(&mob_str, &mpars);
    chosen_ids_rho_to_pickle(&chosen_ids_rho, &mob_str, &mpars);
    println!("Grid built and stored");

    if args.mobility_scenario_model == MobilityScenario::B2 {

        let space_cfn = match args.tessellation_model {
            TessellationModel::BostonLattice => {
                "config_space_bl_retriever"
            },
            TessellationModel::BostonScatter => {
                "config_space_bs_retriever"
            },
            TessellationModel::SyntheticLattice => {
                "config_space_rl_retriever"
            },
        };
    
        let space_str = build_space_retriever_file_name(args.tessellation_model, space_cfn);
        let mut space: Space = retrieve_space(&space_str);
    
        let space_hm = match args.tessellation_model {
            TessellationModel::BostonLattice => {
                let file_name = "config_tessellation_bostonlattice";
                let space_hm = load_json_data(file_name);
                convert_hm_value(space_hm)
            },
            TessellationModel::BostonScatter => {
                let file_name = "config_tessellation_bostonscatter";
                let space_hm = load_json_data(file_name);
                convert_hm_value(space_hm)
            },
            TessellationModel::SyntheticLattice => {
                let tess_hm = select_tessellation_model(args.tessellation_model);
                let attr_hm = select_attractiveness_model(args.attractiveness_model);
                let space_hm = merge_hashmaps(tess_hm, attr_hm);
                convert_hm_value(space_hm)
            },
        };
    
        let mut mpars = match args.config_flag {
            false => {
                MobilityPars::new(args.gamma, args.home_weight, Some(args.location_threshold), args.lockdown_model, args.locked_fraction, args.nagents, 0, args.quarantine_model, args.quarantined_fraction, args.rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, args.t_max)
            },
            true => {
                let mob_cfn = "config_mobility";
                let mob_hm = load_json_data(mob_cfn);
                let mob_hm = convert_hm_value(mob_hm);
    
                let gamma = *mob_hm.get("gamma").unwrap();
                let home_weight = *mob_hm.get("home_weight").unwrap() as u32;
                let location_threshold = Some(*mob_hm.get("location_threshold").unwrap());
                let locked_fraction = *mob_hm.get("locked_fraction").unwrap();
                let nagents = *mob_hm.get("nagents").unwrap() as u32;
                let nlocs = 0;
                let quarantined_fraction = *mob_hm.get("quarantined_fraction").unwrap();
                let rho = *mob_hm.get("rho").unwrap();
                let t_max = *mob_hm.get("t_max").unwrap() as u32;
    
                MobilityPars::new(gamma, home_weight, location_threshold, args.lockdown_model, locked_fraction, nagents, nlocs, args.quarantine_model, quarantined_fraction, rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, t_max)
            },
        };
    
        mpars.nlocs = match space.flags.tessellation {
            TessellationModel::BostonLattice => {
                *space.pars.get("x").unwrap() as u32 
                * *space.pars.get("y").unwrap() as u32
            }
            TessellationModel::BostonScatter => {
                *space.pars.get("nlocs").unwrap() as u32
            }
            TessellationModel::SyntheticLattice => {
                *space.pars.get("x_cells").unwrap() as u32 
                * *space.pars.get("y_cells").unwrap() as u32
            }
        };
        let rho_hm = select_rho_model(args.rho_distribution_model);
        let rho_hm = convert_hm_value(rho_hm);
        let _mob_file_name = write_mobility_string_identifier(&mpars, &rho_hm, &space_str);
        //let grid_cfn = match args.tessellation_model {
        //    TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
        //    TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
        //    TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
        //};

        let grid_str = build_grid_retriever_file_name(
            &mpars,
            &rho_hm,
            &space_str,
            args.tessellation_model, 
        );
        let agent_grid: AgentGrid = retrieve_grid(&grid_str);
    
        let mob_grid = build_mobility_parameter_grid(&agent_grid);
    
        space.set_lockdowns(args.lockdown_model, mpars.locked_fraction);
    
        let od_rates = space.gravity_model_od_rate_matrix();
    
        let rho_hm = select_rho_model(args.rho_distribution_model);
        let rho_hm = convert_hm_value(rho_hm);
    
        let mut output_ensemble = Vec::new();
    
        let nagents = mpars.nagents;
        for a in 0..nagents {
            if a % 100 == 0 {
                println!("Agent {0} is moving", a);
            }
    
            let mut agent = MobileAgent::new();
            agent.set_id(a as u32);
    
            agent.sample_rho(args.rho_distribution_model, &rho_hm);
    
            agent.set_home(&space, &space_hm, mpars.home_weight, args.home_model);
    
            agent.run_baseline2_dynamics(&space, &od_rates, &mob_grid, &mpars);
    
            let agent_output = MobileAgentOutput::new(&agent);
            output_ensemble.push(agent_output);
        }
    
        println!("Mobility run finished. Now saving...");
    
        mpars.scenario = MobilityScenario::B2;
        let mdyna_str = 
        write_mobility_string_identifier(&mpars, &rho_hm, &space_str);
        save_mobility_output_ensemble(&output_ensemble, &mdyna_str);
       
        let mmetadata = MobilityMetadata::new(&mpars);
        mmetadata.to_pickle(&mdyna_str);
        println!("Mobility results saved");
    
        let mob_str = write_mobility_string_identifier(&mpars, &rho_hm, &space_str);
       
        let nchosen = mpars.nagents as usize;
        let chosen_agents = get_chosen_agents(&output_ensemble, nchosen);
    
        let chosen_ids_rho = build_chosen_ids_rho(&output_ensemble, &chosen_agents);
    
        let nlocs = mmetadata.pars.nlocs;
        let t_max = mpars.t_max;
        let agent_grid = AgentGrid::new(
            &output_ensemble, 
            &chosen_ids_rho, 
            nlocs, 
            t_max
        );
    
        // TODO: Set quarantines (if enabled)
    
        agent_grid.to_pickle(&mob_str, &mpars);
        chosen_ids_rho_to_pickle(&chosen_ids_rho, &mob_str, &mpars);
        println!("Grid built and stored");
    }

}

pub fn run_sir_dynamics(args: Args) {
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize;
    let mut rngi = rand::thread_rng();
    let mut gsl_rngi = gsl_Rng::new(RngType::default()).unwrap();
    gsl_rngi.set(seed);
    let mut rngs = (&mut rngi, &mut gsl_rngi);

    let space_cfn = match args.tessellation_model {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(args.tessellation_model, space_cfn);
    let space: Space = retrieve_space(&space_str);
    let a_vec = space.collect_attractiveness();

    let mpars = match args.config_flag {
        false => {
            MobilityPars::new(args.gamma, args.home_weight, Some(args.location_threshold), args.lockdown_model, args.locked_fraction, args.nagents, 0, args.quarantine_model, args.quarantined_fraction, args.rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, args.t_max)
        },
        true => {
            let mob_cfn = "config_mobility";
            let mob_hm = load_json_data(mob_cfn);
            let mob_hm = convert_hm_value(mob_hm);

            let gamma = *mob_hm.get("gamma").unwrap();
            let home_weight = *mob_hm.get("home_weight").unwrap() as u32;
            let location_threshold = Some(*mob_hm.get("location_threshold").unwrap());
            let locked_fraction = *mob_hm.get("locked_fraction").unwrap();
            let nagents = *mob_hm.get("nagents").unwrap() as u32;
            let nlocs = 0;
            let quarantined_fraction = *mob_hm.get("quarantined_fraction").unwrap();
            let rho = *mob_hm.get("rho").unwrap();
            let t_max = *mob_hm.get("t_max").unwrap() as u32;

            MobilityPars::new(gamma, home_weight, location_threshold, args.lockdown_model, locked_fraction, nagents, nlocs, args.quarantine_model, quarantined_fraction, rho, args.rho_distribution_model, args.mobility_selection_flag, args.mobility_scenario_model, t_max)
        },
    };
    
    let rho_hm = select_rho_model(args.rho_distribution_model);
    let rho_hm = convert_hm_value(rho_hm);
    let mob_file_name = build_mobility_retriever_file_name(&mpars, &rho_hm, &space_str);
    //let grid_cfn = match args.tessellation_model {
    //    TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
    //    TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
    //    TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
    //};

    let grid_str = build_grid_retriever_file_name(
        &mpars, 
        &rho_hm,
        &space_str,
        args.tessellation_model,
    );
    let agent_grid: AgentGrid = retrieve_grid(&grid_str);
    let mmetadata = mobility_metadata_from_pickle(&mob_file_name);

    let mut chosen_ids_rho = retrieve_chosen_ids_rho(&grid_str);
    let nagents = chosen_ids_rho.len() as u32;
    mmetadata.pars.nagents;

    let epars = match args.config_flag {
        false => {
            EpidemicPars::new(args.agent_seed_model, args.escape_condition, args.expedited_escape_flag, args.location_seed_model, args.nagents, args.nepicenters, args.nsims, args.pseudomass_exponent, args.removal_rate, args.seed_fraction, args.t_epidemic, args.transmission_rate, args.vaccinated_fraction, args.vaccination_model)
        },
        true => {
            let config_epi_file_name = "config_epidemic";
            let epi_hm = load_json_data(config_epi_file_name);
            let epi_hm = convert_hm_value(epi_hm);

            let escape_condition = *epi_hm.get("escape_condition").unwrap();
            let expedited_escape_flag = {
                let expedited_escape = 
                *epi_hm.get("expedited_escape").unwrap();
                let converted_escape: bool = expedited_escape > 0.0;
                converted_escape
            };
            let nagents = *epi_hm.get("nagents").unwrap() as u32;
            let nepicenters = *epi_hm.get("nepicenters").unwrap() as u32;
            let nsims = *epi_hm.get("nsims").unwrap() as u32;
            let pseudomass_exponent = *epi_hm.get("pseudomass_exponent").unwrap();
            let removal_rate = *epi_hm.get("removal_rate").unwrap();
            let seed_fraction = *epi_hm.get("seed_fraction").unwrap();
            let t_epidemic = *epi_hm.get("t_epidemic").unwrap() as u32;
            let transmission_rate = *epi_hm.get("transmission_rate").unwrap();
            let vaccinated_fraction = *epi_hm.get("vaccinated_fraction").unwrap();
        
            EpidemicPars::new(args.agent_seed_model, escape_condition, expedited_escape_flag, args.location_seed_model, nagents, nepicenters, nsims, pseudomass_exponent, removal_rate, seed_fraction, t_epidemic, transmission_rate, vaccinated_fraction, args.vaccination_model)
        },
    };

    let nepicenters = epars.nepicenters;
    let epicenters = set_epicenters(
        args.location_seed_model, 
        nepicenters, 
        &a_vec,
    );
    println!("epicenter={} with A={}", epicenters[0], space.inner()[epicenters[0] as usize].attractiveness.unwrap());
    let seed_fraction = epars.seed_fraction;
    let t_epidemic = epars.t_epidemic;

    let mut output_ensemble = Vec::new();

    println!("Spreading starts");

    for _ in 0..epars.nsims {
        let mut agent_ensemble = 
        EpidemicAgentEnsemble::new(nagents, &chosen_ids_rho);
        let mut event_ensemble = EventEnsemble::new();

        agent_ensemble.rollout_vaccines(&epars);

        agent_grid.introduce_infections(
            args.agent_seed_model,
            seed_fraction, 
            t_epidemic, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &mut chosen_ids_rho, 
            &epicenters,
        );

        let output = sir_dynamics(
            &agent_grid, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &epars, 
            &mut rngs,
        );

        output_ensemble.push(output);
    }

    println!("Epidemic run finalized");

    if args.raw_output_flag {
        println!("Storing raw epidemic data");
        let epi_file_name = 
        write_epidemic_string_identifier(&epars, &grid_str);
        save_epidemic_output_ensemble(&output_ensemble, &epi_file_name);
        epars.to_pickle(&epi_file_name);
    } else {
        println!("Digesting raw epidemic data");
        let digested_output = digest_raw_output(&output_ensemble, &a_vec, &mpars, &rho_hm, &space_str);
        let epi_file_name = write_digested_epidemic_string_identifier(&epars, &grid_str);
        save_digested_epidemic_output(&digested_output, &epi_file_name);
        epars.to_pickle(&epi_file_name);
        println!("Epidemic run and output digestion finalized");
    }    
}