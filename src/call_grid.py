import os
import time
import utils as ut

cwd_path = os.getcwd()
lower_path = 'data'
fullpath = os.path.join(cwd_path, lower_path)

distribution_flag = 'Beta'
selection_flag = 'mset'
scenario_flag = 'depr' # 'b1hom', 'b1het'

if (scenario_flag == 'depr') or (scenario_flag == 'uniform'):
    experiment_flag = 3
    mob_timestamps = ut.collect_mobility_set_timestamps(fullpath, scenario_flag, distribution_flag)
elif (scenario_flag == 'b1hom') or (scenario_flag == 'b1het') or (scenario_flag == 'plain'):
    experiment_flag = 3
    mob_timestamps = ut.collect_mobility_set_timestamps(fullpath, scenario_flag, distribution_flag)
elif scenario_flag == 'b2':
    experiment_flag = 4
    mob_timestamps = ut.collect_mobility_grid_timestamps(fullpath, 'depr', distribution_flag)

lower_path = 'config/'
filename = 'config_experiment.json'
json_fullname = os.path.join(cwd_path, lower_path, filename)
ut.modify_json_file(json_fullname, key='exp_flag', value=experiment_flag)

# Loop for 
for i, timestamp in zip(range(len(mob_timestamps)), mob_timestamps):
    print("Calling agent grid builder {0}".format(i+1))

    lower_path = 'config/'

    if scenario_flag == 'b2':
        filename = 'config_grid_bl_retriever.json'
        json_fullname = os.path.join(cwd_path, lower_path, filename)
        ut.modify_json_file(json_fullname, key='tm', value=timestamp)

    # Agent grid
    filename = 'config_mobility_retriever.json'
    json_fullname = os.path.join(cwd_path, lower_path, filename)
    ut.modify_json_file(json_fullname, key='tm', value=timestamp)
    ut.call_rust_file(cwd_path)

    time.sleep(5)

print('Calls finished')