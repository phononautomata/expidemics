import os
import time
import utils as ut

cwd_path = os.getcwd()
lower_path = 'data'
fullpath = os.path.join(cwd_path, lower_path)

selection_flag = 'mset'
scenario_flag = 'depr' # 'b1hom', 'b1het'
distribution_flag = 'Beta'

if (scenario_flag == 'depr') or (scenario_flag == 'uniform'):
    experiment_flag = 6
elif (scenario_flag == 'b1hom') or (scenario_flag == 'b1het') or (scenario_flag == 'plain'):
    experiment_flag = 7
elif scenario_flag == 'b2':
    experiment_flag = 8

lower_path = 'config/'
filename = 'config_experiment.json'
json_fullname = os.path.join(cwd_path, lower_path, filename)
ut.modify_json_file(json_fullname, key='exp_flag', value=experiment_flag)

mob_timestamps = ut.collect_mobility_grid_timestamps(fullpath, scenario_flag, distribution_flag=distribution_flag)

for i, timestamp in zip(range(len(mob_timestamps)), mob_timestamps):
    print("Calling epidemics {0}".format(i+1))

    lower_path = 'config/'
    filename = 'config_grid_rl_retriever.json'
    json_fullname = os.path.join(cwd_path, lower_path, filename)
    ut.modify_json_file(json_fullname, key='tm', value=timestamp)
    ut.modify_json_file(json_fullname, key='ms', value=scenario_flag)
    
    filename = 'config_mobility_retriever.json'
    json_fullname = os.path.join(cwd_path, lower_path, filename)
    ut.modify_json_file(json_fullname, key='tm', value=timestamp)
    
    ut.call_rust_file(cwd_path)
    time.sleep(10)

print('Calls finished')