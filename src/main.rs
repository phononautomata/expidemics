use clap::Parser;
use expidemics::
    exps::{ 
        Args,
        generate_space, 
        generate_space_time_agent_grid,
        run_sequential_depr_dynamics,
        run_sir_dynamics, 
    };

fn main() {
    let args = Args::parse();

    match args.exp_flag {
        0 => {
            generate_space(args)
        },
        1 => {
            run_sequential_depr_dynamics(args)
        },
        2 => {
            generate_space_time_agent_grid(args)
        },
        3 => {
            run_sir_dynamics(args)
        },
        _ => panic!(),
    };
}