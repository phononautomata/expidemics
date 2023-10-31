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
pub struct ArgsSpace {
    #[clap(long, value_parser, default_value = "gaussian")]
    pub attractiveness_flag: AttractivenessModel,
    #[clap(long, value_parser, default_value = "finite")]
    pub boundary_flag: BoundaryModel,
    #[clap(long, value_parser, default_value = "random-cartesian")]
    pub pole_flag: PoleModel,
    #[clap(long, value_parser, default_value = "synthetic-lattice")]
    pub tessellation_flag: TessellationModel,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsMobility {
    #[clap(long, value_parser, default_value = "random")]
    pub home_flag: HomeModel,
    #[clap(long, value_parser, default_value = "unmitigated")]
    pub lockdown_flag: LockdownStrategy,
    #[clap(long, value_parser, default_value = "unmitigated")]
    pub quarantine_flag: QuarantineStrategy,
    #[clap(long, value_parser, default_value = "beta")]
    pub rho_flag: RhoDistributionModel,
    #[clap(long, value_parser, default_value = "set")]
    pub selection_flag: MobilitySelection,
    #[clap(long, value_parser, default_value = "depr")]
    pub scenario_flag: MobilityScenario,
    #[clap(long, value_parser, default_value = "synthetic-lattice")]
    pub tessellation_flag: TessellationModel,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsEpidemic {
    #[clap(long, value_parser, default_value = "random")]
    pub aseed_flag: AgentSeedModel,
    #[clap(long, value_parser, default_value = "most-attractive")]
    pub lseed_flag: LocationSeedModel,
    #[clap(long, value_parser, default_value_t = false)]
    pub raw_output: bool,
    #[clap(long, value_parser, default_value = "unmitigated")]
    pub vaccination_flag: VaccinationStrategy,
}

pub fn generate_space(args: ArgsSpace) {
    // Pack in space-related configurations
    let spars = SpaceFlags {
        tessellation: args.tessellation_flag,
        pole: Some(args.pole_flag),
        boundary: Some(args.boundary_flag),
        attractiveness: Some(args.attractiveness_flag),
    };

    // Space object generation
    match args.tessellation_flag {
        TessellationModel::BostonLattice => {
            // Load space parameter hashmap
            let file_name = "config_tessellation_bostonlattice";
            let space_hm = load_json_data(file_name);
            let space_hm = convert_hm_value(space_hm);
            // Invoke Space object
            let space = Space::new(&spars, &space_hm);
            // Serialize object & save
            space.to_pickle();
        },
        TessellationModel::BostonScatter => {
            // Load space parameter hashmap
            let file_name = "config_tessellation_bostonscatter";
            let space_hm = load_json_data(file_name);
            let space_hm = convert_hm_value(space_hm);
            // Invoke Space object
            let space = Space::new(&spars, &space_hm);
            // Serialize object & save
            space.to_pickle();
        },
        TessellationModel::SyntheticLattice => {
            // Load tessellation parameters
            let tess_hm = select_tessellation_model(args.tessellation_flag);
            // Load attractiveness parameters
            let attr_hm = select_attractiveness_model(args.attractiveness_flag);
            // Merge tessellation & attractiveness hashmaps
            let space_hm = merge_hashmaps(tess_hm, attr_hm);
            let space_hm = convert_hm_value(space_hm);

            // Invoke Space object
            let mut space = Space::new(&spars, &space_hm);

            // Generate attractiveness poles
            match args.pole_flag {
                PoleModel::RandomCartesian => {
                    space.generate_random_cartesian_new_poles(&space_hm)
                }
                PoleModel::RandomPolar => {
                    space.generate_random_polar_new_poles(&space_hm)
                }
            }
            // Set boundary conditions
            match args.boundary_flag {
                BoundaryModel::Finite => {},
                BoundaryModel::Periodic => {
                    let center_coords = [0.0; 2];
                    space.get_pbc_coordinates(&space_hm, &center_coords);
                },
            }
            // Generate attractiveness field
            space.create_multipolar_field();
            // Serialize object & save
            space.to_pickle();
        },
    }

    println!("Space object finished");
}

pub fn run_sequential_depr_dynamics(sargs: ArgsSpace, margs: ArgsMobility) {
    // Load Space object
    let space_cfn = match margs.tessellation_flag {
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

    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let mut space: Space = retrieve_space(&space_str);

    // Load Space hashmap
    let space_hm = match margs.tessellation_flag {
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
            let tess_hm = select_tessellation_model(sargs.tessellation_flag);
            let attr_hm = select_attractiveness_model(sargs.attractiveness_flag);
            let space_hm = merge_hashmaps(tess_hm, attr_hm);
            convert_hm_value(space_hm)
        },
    };

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let nagents = *mob_hm.get("nagents").unwrap() as usize;
    let mut mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        margs.scenario_flag,
    );

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

    // Uniformize space (if applies)
    if margs.scenario_flag == MobilityScenario::Uniform {
        space.uniformize();
    }

    // Introduce lockdowns (if applies)
    space.set_lockdowns(margs.lockdown_flag, mpars.locked_fraction);

    // Compute exploration OD matrix through gravity model
    let od_rates = space.gravity_model_od_rate_matrix();

    // Select rho distribution model
    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);

    // Prepare output structure: Vec<MobileAgentOutput>
    let mut output_ensemble = Vec::new();

    // Invoke sequential d-EPR dynamics
    for a in 0..nagents {
        if a % 100 == 0 {
            println!("Agent {0} is moving", a);
        }

        // Invoke Mobile Agent object
        let mut agent = MobileAgent::new();
        agent.set_id(a as u32);

        // Sample rho parameter for the agent
        agent.sample_rho(margs.rho_flag, &rho_hm);

        // Set home
        agent.set_home(&space, &space_hm, mpars.home_weight, margs.home_flag);

        // Run asynchronous d-EPR dynamics
        agent.run_depr_dynamics(&space, &od_rates, &mpars);

        // Add Mobile Agent Output to ensemble
        let agent_output = MobileAgentOutput::new(&agent);
        output_ensemble.push(agent_output);
    }

    println!("Mobility run finished. Now saving...");

    // Save mobility trajectories
    let mdyna_str = 
    write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    save_mobility_output_ensemble(&output_ensemble, &mdyna_str);
    // Save mobility metadata
    let mmetadata = MobilityMetadata::new(&mpars);
    mmetadata.to_pickle(&mdyna_str);
    println!("Mobility results saved");
}

pub fn generate_space_time_agent_grid(margs: ArgsMobility) {
    // Load Space object string
    let space_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);

    // Load Mobile Agent Output ensemble parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        margs.scenario_flag,
    );
    
    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);

    // Load Mobility object
    let mob_str = build_mobility_retriever_file_name(&mpars, &mob_hm, &rho_hm, &space_str);
    write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    let mmetadata = mobility_metadata_from_pickle(&mob_str);
    let mobility_data = retrieve_mobility(&mob_str);
 
    // Obtain a mobility sample (choose agents to include)
    let nchosen = mpars.nagents as usize;
    let chosen_agents = get_chosen_agents(&mobility_data, nchosen);

    // Get agent list with rho values
    let chosen_ids_rho = build_chosen_ids_rho(&mobility_data, &chosen_agents);

    // Rebuild micropopulations at every location and time step
    let nlocs = mmetadata.pars.nlocs;
    let t_max = mpars.t_max;
    let agent_grid = AgentGrid::new(
        &mobility_data, 
        &chosen_ids_rho, 
        nlocs, 
        t_max
    );

    // TODO: Set quarantines (if enabled)

    // Save data structures to pickle 
    agent_grid.to_pickle(&mob_str, &mpars);
    chosen_ids_rho_to_pickle(&chosen_ids_rho, &mob_str, &mpars);
    println!("Grid built and stored");
}

pub fn run_sir_dynamics(
    margs: ArgsMobility, 
    eargs: ArgsEpidemic,
) {
    // Set PRNGs stuff
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize;
    let mut rngi = rand::thread_rng(); // Rust rand
    let mut gsl_rngi = gsl_Rng::new(RngType::default()).unwrap(); // GSL rand
    gsl_rngi.set(seed);
    let mut rngs = (&mut rngi, &mut gsl_rngi);

    // Load Space object
    let space_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let space: Space = retrieve_space(&space_str);
    let a_vec = space.collect_attractiveness();

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        margs.scenario_flag,
    );
    
    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);
    let mob_file_name = 
    write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    let grid_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
        TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
    };

    // Load agent grid
    let grid_str = build_grid_retriever_file_name(
        &mpars,
        margs.tessellation_flag, 
        grid_cfn
    );
    let agent_grid: AgentGrid = retrieve_grid(&grid_str);
    let mmetadata = mobility_metadata_from_pickle(&mob_file_name);

    // Load agents ids and rho list
    let mut chosen_ids_rho = retrieve_chosen_ids_rho(&grid_str);
    let nagents = chosen_ids_rho.len() as u32;
    mmetadata.pars.nagents;

    // Load epidemic parameters
    let config_epi_file_name = "config_epidemic";
    let epi_hm = load_json_data(config_epi_file_name);
    let mut epi_hm = convert_hm_value(epi_hm);
    epi_hm.insert("nagents".to_string(), mpars.nagents as f64);
    let epars = EpidemicPars::new(
        eargs.aseed_flag, 
        eargs.lseed_flag, 
        eargs.vaccination_flag,
        &epi_hm,
    );

    // Configure seeding
    let nepicenters = *epi_hm.get("nepicenters").unwrap() as u32;
    let epicenters = set_epicenters(
        eargs.lseed_flag, 
        nepicenters, 
        &a_vec,
    );
    println!("epicenter={} with A={}", epicenters[0], space.inner()[epicenters[0] as usize].attractiveness.unwrap());
    let seed_fraction = *epi_hm.get("seed_fraction").unwrap();
    let t_epidemic = *epi_hm.get("t_epidemic").unwrap() as u32;

    // Prepare output structure
    let mut output_ensemble = Vec::new();

    println!("Spreading starts");

    // Loop over simulations
    for _ in 0..epars.nsims {
        // Prepare data structures
        let mut agent_ensemble = 
        EpidemicAgentEnsemble::new(nagents, &chosen_ids_rho);
        let mut event_ensemble = EventEnsemble::new();

        // Unroll vaccines
        agent_ensemble.rollout_vaccines(&epars);

        // Introduce patient zero // PROBLEM HERE
        agent_grid.introduce_infections(
            eargs.aseed_flag,
            seed_fraction, 
            t_epidemic, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &mut chosen_ids_rho, 
            &epicenters,
        );

        // Invoke SIR dynamics
        let output = sir_dynamics(
            &agent_grid, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &epars, 
            &mut rngs,
        );

        output_ensemble.push(output);
    }

    // Save epidemic results
    let epi_file_name = 
    write_epidemic_string_identifier(&epars, &grid_str);
    save_epidemic_output_ensemble(&output_ensemble, &epi_file_name);
    // Save metadata
    epars.to_pickle(&epi_file_name);
    println!("Epidemic run finalized");
}

pub fn run_sir_dynamics_and_digest(
    margs: ArgsMobility, 
    eargs: ArgsEpidemic,
) {
    // Set PRNGs stuff
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize;
    let mut rngi = rand::thread_rng(); // Rust rand
    let mut gsl_rngi = gsl_Rng::new(RngType::default()).unwrap(); // GSL rand
    gsl_rngi.set(seed);
    let mut rngs = (&mut rngi, &mut gsl_rngi);

    // Load Space object
    let space_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let space: Space = retrieve_space(&space_str);
    let a_vec = space.collect_attractiveness();

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        margs.scenario_flag,
    );

    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);
    //let mob_file_name = write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    let grid_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
        TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
    };
    let grid_str = build_grid_retriever_file_name(
        &mpars,
        margs.tessellation_flag, 
        grid_cfn
    );
    let agent_grid: AgentGrid = retrieve_grid(&grid_str);
    //let mmetadata = mobility_metadata_from_pickle(&mob_file_name);

    // Load agents ids and rho list
    let mut chosen_ids_rho = retrieve_chosen_ids_rho(&grid_str);
    let nagents = mpars.nagents; 

    // Load epidemic parameters
    let config_epi_file_name = "config_epidemic";
    let epi_hm = load_json_data(config_epi_file_name);
    let mut epi_hm = convert_hm_value(epi_hm);
    epi_hm.insert("nagents".to_string(), mpars.nagents as f64);
    let epars = EpidemicPars::new(
        eargs.aseed_flag, 
        eargs.lseed_flag, 
        eargs.vaccination_flag,
        &epi_hm,
    );

    // Configure seeding
    let nepicenters = *epi_hm.get("nepicenters").unwrap() as u32;
    let epicenters = set_epicenters(
        eargs.lseed_flag, 
        nepicenters, 
        &a_vec,
    );
    println!("epicenter={} with A={}", epicenters[0], space.inner()[epicenters[0] as usize].attractiveness.unwrap());
    let seed_fraction = *epi_hm.get("seed_fraction").unwrap();
    let t_epidemic = *epi_hm.get("t_epidemic").unwrap() as u32;

    // Prepare output structure
    let mut output_ensemble = Vec::new();

    println!("Spreading starts");

    // Loop over simulations
    for _ in 0..epars.nsims {
        // Prepare data structures
        let mut agent_ensemble = 
        EpidemicAgentEnsemble::new(nagents, &chosen_ids_rho);
        let mut event_ensemble = EventEnsemble::new();

        // Unroll vaccines
        agent_ensemble.rollout_vaccines(&epars);

        // Introduce patient zero // PROBLEM HERE
        agent_grid.introduce_infections(
            eargs.aseed_flag,
            seed_fraction, 
            t_epidemic, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &mut chosen_ids_rho, 
            &epicenters,
        );

        // Invoke SIR dynamics
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

    if eargs.raw_output {
        println!("Storing raw epidemic data");
        // Save epidemic results
        let epi_file_name = 
        write_epidemic_string_identifier(&epars, &grid_str);
        save_epidemic_output_ensemble(&output_ensemble, &epi_file_name);
        // Save metadata
        epars.to_pickle(&epi_file_name);
    } else {
        println!("Digesting raw epidemic data");
        let digested_output = digest_raw_output(&output_ensemble, &a_vec, &mpars, &mob_hm, &rho_hm, &space_str);
        let epi_file_name = write_digested_epidemic_string_identifier(&epars, &grid_str);
        save_digested_epidemic_output(&digested_output, &epi_file_name);
        epars.to_pickle(&epi_file_name);
        println!("Epidemic run and output digestion finalized");
    }    
}

pub fn run_sequential_baseline1_mobility(sargs: ArgsSpace, margs: ArgsMobility) {
    // Load space object
    let space_cfn = match margs.tessellation_flag {
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

    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let mut space: Space = retrieve_space(&space_str);

    // Load space hashmap
    let space_hm = match margs.tessellation_flag {
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
            let tess_hm = select_tessellation_model(sargs.tessellation_flag);
            let attr_hm = select_attractiveness_model(sargs.attractiveness_flag);
            let space_hm = merge_hashmaps(tess_hm, attr_hm);
            convert_hm_value(space_hm)
        },
    };

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let nagents = *mob_hm.get("nagents").unwrap() as usize;
    let mut mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        margs.scenario_flag,
    );

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

    // Uniformize space (if applies)
    if margs.scenario_flag == MobilityScenario::Plain {
        space.uniformize();
    }

    // Introduce lockdowns (if applies)
    space.set_lockdowns(margs.lockdown_flag, mpars.locked_fraction);

    // Compute exploration OD matrix through gravity model
    let od_rates = space.gravity_model_od_rate_matrix();

    // Select rho distribution model
    let hom_flag = true;
    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);

    // Prepare output structure: Vec<MobileAgentOutput>
    let mut output_ensemble = Vec::new();

    // Invoke single d-EPR dynamics
    for a in 0..nagents {
        if a % 100 == 0 {
            println!("Agent {0} is moving", a);
        }

        // Invoke Mobile Agent object
        let mut agent = MobileAgent::new();
        agent.set_id(a as u32);

        // Sample rho parameter for the agent
        agent.sample_rho(margs.rho_flag, &rho_hm);

        // Set home
        agent.set_home(&space, &space_hm, mpars.home_weight, margs.home_flag);

        // Run asynchronous 'baseline 1' dynamics
        agent.run_baseline1_dynamics(&space, &od_rates, &mpars, hom_flag);

        // Add Mobile Agent Output to ensemble
        let agent_output = MobileAgentOutput::new(&agent);
        output_ensemble.push(agent_output);
    }

    println!("Mobility run finished. Now saving...");

    // Save mobility trajectories
    let mdyna_str = 
    write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    save_mobility_output_ensemble(&output_ensemble, &mdyna_str);
    // Save mobility metadata
    let mmetadata = MobilityMetadata::new(&mpars);
    mmetadata.to_pickle(&mdyna_str);
    println!("Mobility results saved");
}

pub fn run_baseline1_sir_dynamics_and_digest(
    margs: ArgsMobility, 
    eargs: ArgsEpidemic,
) {
    // Set PRNGs stuff
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize;
    let mut rngi = rand::thread_rng(); // Rust rand
    let mut gsl_rngi = gsl_Rng::new(RngType::default()).unwrap(); // GSL rand
    gsl_rngi.set(seed);
    let mut rngs = (&mut rngi, &mut gsl_rngi);

    // Load Space object
    let space_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let space: Space = retrieve_space(&space_str);
    let a_vec = space.collect_attractiveness();

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        margs.scenario_flag,
    );

    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);
    //let mob_file_name = write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    let grid_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
        TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
    };
    let grid_str = build_grid_retriever_file_name(
        &mpars,
        margs.tessellation_flag, 
        grid_cfn
    );
    let agent_grid: AgentGrid = retrieve_grid(&grid_str);
    //let mmetadata = mobility_metadata_from_pickle(&mob_file_name);

    // Load agents ids and rho list
    let mut chosen_ids_rho = retrieve_chosen_ids_rho(&grid_str);
    let nagents = mpars.nagents; 
    //chosen_ids_rho.len() as u32;
    //mmetadata.pars.nagents;

    // Load epidemic parameters
    let config_epi_file_name = "config_epidemic";
    let epi_hm = load_json_data(config_epi_file_name);
    let mut epi_hm = convert_hm_value(epi_hm);
    epi_hm.insert("nagents".to_string(), mpars.nagents as f64);
    let epars = EpidemicPars::new(
        eargs.aseed_flag, 
        eargs.lseed_flag, 
        eargs.vaccination_flag,
        &epi_hm,
    );

    // Configure seeding
    let nepicenters = *epi_hm.get("nepicenters").unwrap() as u32;
    let epicenters = set_epicenters(
        eargs.lseed_flag, 
        nepicenters, 
        &a_vec,
    );
    println!("epicenter={} with A={}", epicenters[0], space.inner()[epicenters[0] as usize].attractiveness.unwrap());
    let seed_fraction = *epi_hm.get("seed_fraction").unwrap();
    let t_epidemic = *epi_hm.get("t_epidemic").unwrap() as u32;

    // Prepare output structure
    let mut output_ensemble = Vec::new();

    println!("Spreading starts");

    // Loop over simulations
    for _ in 0..epars.nsims {
        // Prepare data structures
        let mut agent_ensemble = 
        EpidemicAgentEnsemble::new(nagents, &chosen_ids_rho);
        let mut event_ensemble = EventEnsemble::new();

        // Unroll vaccines
        agent_ensemble.rollout_vaccines(&epars);

        // Introduce patient zero // PROBLEM HERE
        agent_grid.introduce_infections(
            eargs.aseed_flag,
            seed_fraction, 
            t_epidemic, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &mut chosen_ids_rho, 
            &epicenters,
        );

        // Invoke SIR dynamics
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

    if eargs.raw_output {
        println!("Storing raw epidemic data");
        // Save epidemic results
        let epi_file_name = 
        write_epidemic_string_identifier(&epars, &grid_str);
        save_epidemic_output_ensemble(&output_ensemble, &epi_file_name);
        // Save metadata
        epars.to_pickle(&epi_file_name);
    } else {
        println!("Digesting raw epidemic data");
        let digested_output = digest_raw_output(&output_ensemble, &a_vec, &mpars, &mob_hm, &rho_hm, &space_str);
        let epi_file_name = write_digested_epidemic_string_identifier(&epars, &grid_str);
        save_digested_epidemic_output(&digested_output, &epi_file_name);
        epars.to_pickle(&epi_file_name);
        println!("Epidemic run and output digestion finalized");
    }    
}

pub fn generate_baseline2_space_time_agent_grid(sargs: ArgsSpace, margs: ArgsMobility) {
    // Load space object
    let space_cfn = match margs.tessellation_flag {
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

    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let mut space: Space = retrieve_space(&space_str);

    // Load space hashmap
    let space_hm = match margs.tessellation_flag {
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
            let tess_hm = select_tessellation_model(sargs.tessellation_flag);
            let attr_hm = select_attractiveness_model(sargs.attractiveness_flag);
            let space_hm = merge_hashmaps(tess_hm, attr_hm);
            convert_hm_value(space_hm)
        },
    };

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let nagents = *mob_hm.get("nagents").unwrap() as usize;
    let mut mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        MobilityScenario::Depr,
    );

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
    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);
    let _mob_file_name = write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    let grid_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
        TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
    };

    // Load agent grid
    let grid_str = build_grid_retriever_file_name(
        &mpars,
        margs.tessellation_flag, 
        grid_cfn
    );
    let agent_grid: AgentGrid = retrieve_grid(&grid_str);

    // Compute mobility parameter grid
    let mob_grid = build_mobility_parameter_grid(&agent_grid);

    // Introduce lockdowns (if applies)
    space.set_lockdowns(margs.lockdown_flag, mpars.locked_fraction);

    // Compute exploration OD matrix through gravity model
    let od_rates = space.gravity_model_od_rate_matrix();

    // Select rho distribution model
    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);

    // Prepare output structure: Vec<MobileAgentOutput>
    let mut output_ensemble = Vec::new();

    // Invoke single d-EPR dynamics
    for a in 0..nagents {
        if a % 100 == 0 {
            println!("Agent {0} is moving", a);
        }

        // Invoke Mobile Agent object
        let mut agent = MobileAgent::new();
        agent.set_id(a as u32);

        // Sample rho parameter for the agent
        agent.sample_rho(margs.rho_flag, &rho_hm);

        // Set home
        agent.set_home(&space, &space_hm, mpars.home_weight, margs.home_flag);

        // Run asynchronous 'baseline 2' dynamics
        agent.run_baseline2_dynamics(&space, &od_rates, &mob_grid, &mpars);

        // Add Mobile Agent Output to ensemble
        let agent_output = MobileAgentOutput::new(&agent);
        output_ensemble.push(agent_output);
    }

    println!("Mobility run finished. Now saving...");

    // Save mobility trajectories
    mpars.scenario = MobilityScenario::B2;
    let mdyna_str = 
    write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    save_mobility_output_ensemble(&output_ensemble, &mdyna_str);
    // Save mobility metadata
    let mmetadata = MobilityMetadata::new(&mpars);
    mmetadata.to_pickle(&mdyna_str);
    println!("Mobility results saved");

    // Load Mobility object
    let mob_str = write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
   
    // Obtain a mobility sample (choose agents to include)
    let nchosen = mpars.nagents as usize;
    let chosen_agents = get_chosen_agents(&output_ensemble, nchosen);

    // Get agent list with rho values
    let chosen_ids_rho = build_chosen_ids_rho(&output_ensemble, &chosen_agents);

    // Rebuild micropopulations at every location and time step
    let nlocs = mmetadata.pars.nlocs;
    let t_max = mpars.t_max;
    let agent_grid = AgentGrid::new(
        &output_ensemble, 
        &chosen_ids_rho, 
        nlocs, 
        t_max
    );

    // TODO: Set quarantines (if enabled)

    // Save data structures to pickle 
    agent_grid.to_pickle(&mob_str, &mpars);
    chosen_ids_rho_to_pickle(&chosen_ids_rho, &mob_str, &mpars);
    println!("Grid built and stored");
}

pub fn run_baseline2_sir_dynamics_and_digest(
    margs: ArgsMobility, 
    eargs: ArgsEpidemic,
) {
    // Set PRNGs stuff
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize;
    let mut rngi = rand::thread_rng(); // Rust rand
    let mut gsl_rngi = gsl_Rng::new(RngType::default()).unwrap(); // GSL rand
    gsl_rngi.set(seed);
    let mut rngs = (&mut rngi, &mut gsl_rngi);

    // Load Space object
    let space_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_space_bl_retriever"},
        TessellationModel::BostonScatter => {"config_space_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_space_rl_retriever"},
    };
    let space_str = build_space_retriever_file_name(margs.tessellation_flag, space_cfn);
    let space: Space = retrieve_space(&space_str);
    let a_vec = space.collect_attractiveness();

    // Load general Mobility parameters
    let mob_cfn = "config_mobility";
    let mob_hm = load_json_data(mob_cfn);
    let mob_hm = convert_hm_value(mob_hm);
    let mpars = MobilityPars::new(
        margs.lockdown_flag, 
        &mob_hm, 
        margs.quarantine_flag, 
        margs.rho_flag, 
        margs.selection_flag, 
        MobilityScenario::B2,
    );

    let rho_hm = select_rho_model(margs.rho_flag);
    let rho_hm = convert_hm_value(rho_hm);
    //let mob_file_name = write_mobility_string_identifier(&mpars, &mob_hm, &rho_hm, &space_str);
    let grid_cfn = match margs.tessellation_flag {
        TessellationModel::BostonLattice => {"config_grid_bl_retriever"},
        TessellationModel::BostonScatter => {"config_grid_bs_retriever"},
        TessellationModel::SyntheticLattice => {"config_grid_rl_retriever"},
    };
    let grid_str = build_grid_retriever_file_name(
        &mpars,
        margs.tessellation_flag, 
        grid_cfn,
    );
    let agent_grid: AgentGrid = retrieve_grid(&grid_str);
    //let mmetadata = mobility_metadata_from_pickle(&mob_file_name);

    // Load agents ids and rho list
    let mut chosen_ids_rho = retrieve_chosen_ids_rho(&grid_str);
    let nagents = mpars.nagents; 
    //chosen_ids_rho.len() as u32;
    //mmetadata.pars.nagents;

    // Load epidemic parameters
    let config_epi_file_name = "config_epidemic";
    let epi_hm = load_json_data(config_epi_file_name);
    let mut epi_hm = convert_hm_value(epi_hm);
    epi_hm.insert("nagents".to_string(), mpars.nagents as f64);
    let epars = EpidemicPars::new(
        eargs.aseed_flag, 
        eargs.lseed_flag, 
        eargs.vaccination_flag,
        &epi_hm,
    );

    // Configure seeding
    let nepicenters = *epi_hm.get("nepicenters").unwrap() as u32;
    let epicenters = set_epicenters(
        eargs.lseed_flag, 
        nepicenters, 
        &a_vec,
    );
    println!("epicenter={} with A={}", epicenters[0], space.inner()[epicenters[0] as usize].attractiveness.unwrap());
    let seed_fraction = *epi_hm.get("seed_fraction").unwrap();
    let t_epidemic = *epi_hm.get("t_epidemic").unwrap() as u32;

    // Prepare output structure
    let mut output_ensemble = Vec::new();

    println!("Spreading starts");

    // Loop over simulations
    for _ in 0..epars.nsims {
        // Prepare data structures
        let mut agent_ensemble = 
        EpidemicAgentEnsemble::new(nagents, &chosen_ids_rho);
        let mut event_ensemble = EventEnsemble::new();

        // Unroll vaccines
        agent_ensemble.rollout_vaccines(&epars);

        // Introduce patient zero // PROBLEM HERE
        agent_grid.introduce_infections(
            eargs.aseed_flag,
            seed_fraction, 
            t_epidemic, 
            &mut agent_ensemble, 
            &mut event_ensemble,
            &mut chosen_ids_rho, 
            &epicenters,
        );

        // Invoke SIR dynamics
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

    if eargs.raw_output {
        println!("Storing raw epidemic data");
        // Save epidemic results
        let epi_file_name = 
        write_epidemic_string_identifier(&epars, &grid_str);
        save_epidemic_output_ensemble(&output_ensemble, &epi_file_name);
        // Save metadata
        epars.to_pickle(&epi_file_name);
    } else {
        println!("Digesting raw epidemic data");
        let digested_output = digest_raw_output(&output_ensemble, &a_vec, &mpars, &mob_hm, &rho_hm, &space_str);
        let epi_file_name = write_digested_epidemic_string_identifier(&epars, &grid_str);
        save_digested_epidemic_output(&digested_output, &epi_file_name);
        epars.to_pickle(&epi_file_name);
        println!("Epidemic run and output digestion finalized");
    }    
}