import os
import time
import utils as ut

cwd_path = os.getcwd()
lower_path = 'data'
fullpath = os.path.join(cwd_path, lower_path)

ngrids = 35
selection_flag = 'set'
scenario_flag = 'depr' # 'b1hom', 'b1het'

if selection_flag == 'pool':
    ngrids = 1

if scenario_flag == 'depr' or scenario_flag == 'uniform':
    experiment_flag = 1
elif (scenario_flag == 'b1hom') or (scenario_flag == 'b1het') or (scenario_flag == 'plain'):
    experiment_flag = 2

lower_path = 'config/'
filename = 'config_experiment.json'
json_fullname = os.path.join(cwd_path, lower_path, filename)
ut.modify_json_file(json_fullname, key='exp_flag', value=experiment_flag)

# Loop for sequential mobility
for i in range(ngrids):
    print("Calling mobility set builder {0}".format(i+1))

    # Sequential mobility
    ut.call_rust_file(cwd_path)

    time.sleep(5)

print('Calls finished')