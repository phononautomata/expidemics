import os
import time
import utils as ut
import argparse

cwd_path = os.getcwd()

parser = argparse.ArgumentParser(description='Process some arguments.')

parser.add_argument('--ngrids', type=int, default=2, help='Number of grid realizations to generate')
parser.add_argument('--epidemic_flag', type=bool, default=True, help='Set False to skip epidemic spreading simulations')

parser.add_argument('--agent_seed_model', type=str, default="random", choices=['explorers', 'random', 'returners', 'top-explorers', 'top-returners'], help="Model for zero patient assignation")
parser.add_argument('--attractiveness_model', type=str, default="uniform", choices=['data-based', 'exponential', 'gaussian', 'inverse-square', 'inverse-square-root', 'linear', 'power-law', 'random-uniform', 'uniform'], help="Attractiveness field model")
parser.add_argument('--boundary_model', type=str, default="finite", choices=['finite', 'periodic'], help="Model to define the boundaries of the lattice (in case of)")
parser.add_argument('--config_flag', type=bool, default=False, help="Set True to read (some) parameters from config files")
parser.add_argument('--escape_condition', type=float, default=0.0, help="Prevalence fraction threshold to finish dynamics")
parser.add_argument('--exp_flag', type=int, default=0, choices=[0, 1, 2, 3], help="Experiment number to launch")
parser.add_argument('--expedited_escape_flag', type=bool, default=False, help="Set True for an expedited end of the dynamics")
parser.add_argument('--gamma', type=float, default=0.21, help="Exploration's probability gamma parameter")
parser.add_argument('--home_model', type=str, default="random", choices=['attractiveness', 'random', 'census'], help="Model to distribute agents in home locations")
parser.add_argument('--home_weight', type=float, default=25, help="Weight of the agents' initial position as a home location")
parser.add_argument('--location_threshold', type=float, default=0.0, help="Do not remember what the heck was that for")
parser.add_argument('--lockdown_model', type=str, default="unmitigated", choices=['least-attractive', 'most-attractive', 'random', 'unmitigated'], help="Strategy to active lockdowns on the population")
parser.add_argument('--locked_fraction', type=float, default=0.0, help="Fraction of agents under lockdown")
parser.add_argument('--location_seed_model', type=str, default="most-attractive", choices=['least-attractive', 'most-attractive', 'random'], help="Model to choose epicenter")
parser.add_argument('--mobility_scenario_model', type=str, default='depr', choices=['b1het', 'b1hom', 'b2', 'depr', 'plain', 'uniform'], help='Mobility scenario flag')
parser.add_argument('--mobility_selection_flag', type=str, default='set', choices=['set', 'pool'], help='Selection flag')
parser.add_argument('--nagents', type=int, default=1000, help="Number of agents in the system")
parser.add_argument('--nepicenters', type=int, default=1, help="Number of epicenters for the outbreak")
parser.add_argument('--nsims', type=int, default=50, help="Number of stochastic realizations for the epidemic spreading")
parser.add_argument('--pole_model', type=str, default="random-cartesian", choices=['random-cartesian', 'random-polar'], help="Model to generate attractiveness poles in the lattice")
parser.add_argument('--pseudomass_exponent', type=float, default=1.0, help="Pseudo-mass-action exponent acting on the location's population (force of infection)")
parser.add_argument('--quarantine_model', type=str, default="unmitigated", choices=['explorers', 'random', 'returners', 'top-explorers', 'top-returners', 'unmitigated'], help="Strategy to active quarantines on the population")
parser.add_argument('--quarantined_fraction', type=float, default=0.0, help="Fraction of agents under quarantine")
parser.add_argument('--raw_output_flag', type=bool, default=True, help='Set False to obtain epidemic results without statistical post-processing')
parser.add_argument('--removal_rate', type=float, default=0.1, help="SIR model removal rate (inverse of the infectious period)")
parser.add_argument('--rho', type=float, default=0.6, help="Exploration's probability rho parameter")
parser.add_argument('--rho_distribution_model', type=str, default='beta', choices=['beta', 'delta-bimodal', 'exponential', 'gamma', 'gaussian', 'homogeneous', 'log-normal', 'negative-binomial', 'uniform'], help="Exploration's probability rho parameter distribution model")
parser.add_argument('--rho_distribution_model_parameter_a', type=float, default=2.0, help="Parameter 'a' for exploration's probability rho parameter distribution model")
parser.add_argument('--rho_distribution_model_parameter_b', type=float, default=2.0, help="Parameter 'b' for exploration's probability rho parameter distribution model")
parser.add_argument('--seed_fraction', type=float, default=0.7, help="Fraction of the population to be infected (patient zero) in every epicenter")
parser.add_argument('--t_max', type=int, default=1200, help="Maximum simulation time for agents mobility trajectories")
parser.add_argument('--tessellation_model', type=str, default='boston-lattice', choices=['boston-lattice', 'boston-scatter', 'synthetic-lattice'], help='Tessellation flag')
parser.add_argument('--transmission_rate', type=float, default=0.12, help="SIR model transmission rate")
parser.add_argument('--vaccination_model', type=str, default="unmitigated", choices=['explorers', 'random', 'returners', 'top-explorers', 'top-returners', 'unmitigated'], help="Vaccination strategy on the population")

args = parser.parse_args()

cwd_path = os.getcwd()

if args.tessellation_model == 'boston-lattice':
    config_space_retriever_file_name = 'config_space_bl_retriever.json'
    space_filename = ut.build_boston_lattice_file_name(cwd_path, 'config', config_space_retriever_file_name)
    config_grid_retriever_file_name = 'config_grid_bl_retriever.json'
elif args.tessellation_model == 'boston-scatter':
    config_space_retriever_file_name = 'config_space_bs_retriever.json'
    space_filename = ut.build_boston_scatter_file_name(cwd_path, 'config', config_space_retriever_file_name)
    config_grid_retriever_file_name = 'config_grid_bs_retriever.json'
elif args.tessellation_model == 'synthetic-lattice':
    config_space_retriever_file_name = 'config_space_rl_retriever.json'
    space_filename = ut.build_regular_lattice_file_name(cwd_path, 'config', config_space_retriever_file_name)
    config_grid_retriever_file_name = 'config_grid_rl_retriever.json'
space_fullname = os.path.join(cwd_path, 'data', space_filename)

space_exists = os.path.exists(space_fullname)

if space_exists == False:
    time.sleep(30)
    # Space object created with default parameters
    ut.call_rust_file(cwd_path, exp_flag=0)

ngrids = args.ngrids

if args.mobility_selection_flag == 'pool':
    ngrids = 1

mob_exp_flag = 1
grid_exp_flag = 2
epi_exp_flag = 3

rdmp_dict = ut.build_rho_distribution_model_parameter_dictionary(args.rho_distribution_model)

for i in range(ngrids):
    print(f"Calling mobility set builder {i + 1}")

    # Run mobility dynamics
    ut.call_rust_file(
        cwd_path, 
        exp_flag=mob_exp_flag, 
        home_model=args.home_model, 
        home_weight=args.home_weight, 
        mobility_scenario_model=args.mobility_scenario_model, 
        nagents=args.nagents, t_max=args.t_max, 
        rho_distribution_model=args.rho_distribution_model,
        tessellation_model=args.tessellation_model,
        )

    print(f"Calling agent space-time grid builder {i + 1}")

    # Find mobility file created & rewrite config_mobility_retriever.json accordingly
    mobility_timestamp = ut.find_latest_file_with_timestamp(cwd_path, lower_path='data', mobility_scenario_model=args.mobility_scenario_model)
    config_mobility_retriever_fullname = os.path.join(cwd_path, 'config', 'config_mobility_retriever.json')
    ut.modify_json_file(config_mobility_retriever_fullname, key='tm', value=mobility_timestamp)
    ut.modify_json_file(config_mobility_retriever_fullname, key='rm', value=args.rho_distribution_model.capitalize())
    ut.modify_json_file(config_mobility_retriever_fullname, key='ra', value=rdmp_dict['ra'])
    ut.modify_json_file(config_mobility_retriever_fullname, key='rb', value=rdmp_dict['rb'])

    # Generate space-time agent grid
    ut.call_rust_file(
        cwd_path, 
        exp_flag=grid_exp_flag, 
        mobility_scenario_model=args.mobility_scenario_model, 
        nagents=args.nagents,
        rho_distribution_model=args.rho_distribution_model,
        )

    print(f"Calling epidemics")

    # Modify grid retriever
    config_grid_retriever_fullname = os.path.join(cwd_path, 'config', config_grid_retriever_file_name)
    ut.modify_json_file(config_grid_retriever_fullname, key='tm', value=mobility_timestamp)
    ut.modify_json_file(config_grid_retriever_fullname, key='rm', value=args.rho_distribution_model.capitalize())
    ut.modify_json_file(config_grid_retriever_fullname, key='ra', value=rdmp_dict['ra'])
    ut.modify_json_file(config_grid_retriever_fullname, key='rb', value=rdmp_dict['rb'])

    # Invoke epidemic
    ut.call_rust_file(
        cwd_path, 
        exp_flag=epi_exp_flag, 
        mobility_scenario_model=args.mobility_scenario_model, 
        nagents=args.nagents,
        rho_distribution_model=args.rho_distribution_model,
        )

    time.sleep(10)

    # Delete mobility data
    mobility_filename = ut.build_mobility_file_name(
        cwd_path=cwd_path,
        lower_path='config',
        mobility_config_file_name='config_mobility_retriever.json', 
        mobility_selection_flag=args.mobility_selection_flag, 
        mobility_scenario_model=args.mobility_scenario_model, 
        space_config_file_name=config_space_retriever_file_name, 
        tessellation_model=args.tessellation_model,
        )
    
    mobility_fullname = os.path.join(cwd_path, 'data', mobility_filename)
    if os.path.exists(mobility_fullname):
        os.remove(mobility_fullname)
        print(f"Deleted mobility file: {mobility_fullname}")
    else:
        print(f"File not found: {mobility_fullname}")

print('Calls finished. Agent space-time grids generated. Epidemics launched.')