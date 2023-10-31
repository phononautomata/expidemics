import os

import utils as ut
import analysis as an

cwd_path = os.getcwd()


def main():

    boston_regular = True
    norm = False
    
    # Load raw space object
    raw_filename = 'boston_space_object.csv'
    lower_path = 'data'
    full_path = os.path.join(cwd_path, lower_path, raw_filename)
    space_df = ut.open_file(full_path)

    # Load curation parameters
    filename = 'config_space_curator.json'
    lower_path = 'config'
    full_path = os.path.join(cwd_path, lower_path, filename)
    cur_pars = ut.read_json_file(fullname=full_path)

    # Curate space object
    space_df = an.curate_space_df(space_df, cur_pars, limit_flag=True, round_flag=True)

    if boston_regular == True:

        br_filename = 'config_space_bl_retriever.json'
        lower_path = 'config'
        fullname = os.path.join(cwd_path, lower_path, br_filename)
        reg_pars = ut.read_json_file(fullname=fullname)
        reg_pars['rd'] = cur_pars['rd']

        space_df = an.build_databased_regular_lattice_space_df(space_df, reg_pars, cur_pars, norm=norm)

        filename = 'bl_' + ut.dict_to_string(reg_pars) + '.json'
        lower_path = 'data'
        fullname = os.path.join(cwd_path, lower_path, filename)
        ut.space_object_to_rust_as_json(space_df, fullname=fullname)

        print("Boston space dataframe to Rust is ready")

if __name__ == "__main__":
    main()
