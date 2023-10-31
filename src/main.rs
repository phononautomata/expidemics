use clap::Parser;
use expidemics::{
    exps::{ 
    generate_space, 
    generate_space_time_agent_grid,
    run_sequential_depr_dynamics,
    run_sir_dynamics, 
    ArgsSpace, 
    ArgsMobility, 
    ArgsEpidemic, 
    run_sir_dynamics_and_digest, 
    run_sequential_baseline1_mobility, 
    run_baseline2_sir_dynamics_and_digest, 
    run_baseline1_sir_dynamics_and_digest, 
    generate_baseline2_space_time_agent_grid,
}, utils::{load_json_data, convert_hm_value}};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsExp {
    #[clap(long, value_parser, default_value_t = 0)]
    pub exp_flag: usize,
}

fn main() {
    let exp_flag = *convert_hm_value(load_json_data("config_experiment")).get("exp_flag").unwrap() as usize;
    
    //let input_args = ArgsExp::parse();
    match exp_flag {
        0 => {
            let sargs = ArgsSpace::parse();
            generate_space(sargs)
        },
        1 => {
            let sargs = ArgsSpace::parse();
            let margs = ArgsMobility::parse();
            run_sequential_depr_dynamics(sargs, margs)
        },
        2 => {
            let sargs = ArgsSpace::parse();
            let margs = ArgsMobility::parse();
            run_sequential_baseline1_mobility(sargs, margs);
        },
        3 => {
            let margs = ArgsMobility::parse();
            generate_space_time_agent_grid(margs)
        },
        4 => {
            let sargs = ArgsSpace::parse();
            let margs = ArgsMobility::parse();
            generate_baseline2_space_time_agent_grid(sargs, margs);
        },
        5 => {
            let margs = ArgsMobility::parse();
            let eargs = ArgsEpidemic::parse();
            run_sir_dynamics(margs, eargs)
        },
        6 => {
            let margs = ArgsMobility::parse();
            let eargs = ArgsEpidemic::parse();
            run_sir_dynamics_and_digest(margs, eargs);
        },
        7 => {
            let margs = ArgsMobility::parse();
            let eargs = ArgsEpidemic::parse();
            run_baseline1_sir_dynamics_and_digest(margs, eargs);
        }
        8 => {
            let margs = ArgsMobility::parse();
            let eargs = ArgsEpidemic::parse();
            run_baseline2_sir_dynamics_and_digest(margs, eargs);
        },
        _ => panic!(),
    };
}