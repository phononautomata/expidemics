import os
import collections
import itertools
import numpy as np
import pickle as pk
import ruptures as rpt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from collections import Counter, OrderedDict, defaultdict
from itertools import chain
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statistics import mode
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

import utils as ut
import analysis as an

cwd_path = os.getcwd()

def plot_202307D1_macro_time_evolution():
    lower_path = 'config/'
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    low_rho = [0.0, 0.05]
    mid_rho = [0.45, 0.55] 
    hig_rho = [0.95, 1.0]
    tot_rho = [0.0, 1.0]

    # Define empty lists for storing the curves
    extended_low_rho_inc_st = []
    extended_low_rho_pre_st = []
    extended_mid_rho_inc_st = []
    extended_mid_rho_pre_st = []
    extended_hig_rho_inc_st = []
    extended_hig_rho_pre_st = []
    extended_tot_inc_st = []
    extended_tot_pre_st = []

    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))
        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        
        # Filter dataframe by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        
        # Compute population in rho intervals
        low_rho_pop_s = an.number_of_agents_in_rho_interval(agent_df, low_rho)
        mid_rho_pop_s = an.number_of_agents_in_rho_interval(agent_df, mid_rho)
        hig_rho_pop_s = an.number_of_agents_in_rho_interval(agent_df, hig_rho)
        tot_pop_s = an.number_of_agents_in_rho_interval(agent_df, tot_rho)
        
        # Filter dataframe by health status
        agent_df = an.health_status_filter_agent_data_frame(agent_df, health_status='Removed')
        
        # Rebuild epidemic curves
        low_rho_inc_st = an.rebuild_incidence_time_series_for_rho_group(agent_df, low_rho, t_max)
        low_rho_pre_st = an.rebuild_prevalence_time_series_for_rho_group(agent_df, low_rho, t_max)
        mid_rho_inc_st = an.rebuild_incidence_time_series_for_rho_group(agent_df, mid_rho, t_max)
        mid_rho_pre_st = an.rebuild_prevalence_time_series_for_rho_group(agent_df, mid_rho, t_max)
        hig_rho_inc_st = an.rebuild_incidence_time_series_for_rho_group(agent_df, hig_rho, t_max)
        hig_rho_pre_st = an.rebuild_prevalence_time_series_for_rho_group(agent_df, hig_rho, t_max)
        tot_inc_st = an.rebuild_incidence_time_series_for_rho_group(agent_df, tot_rho, t_max)
        tot_pre_st = an.rebuild_prevalence_time_series_for_rho_group(agent_df, tot_rho, t_max)
        
        N_inf = np.sum(tot_pre_st[:,-1])
        N_tot = np.sum(tot_pop_s)
        print("N_inf={0}".format(N_inf))
        print("N_tot={0}".format(N_tot))
        print("ratio={0}".format(N_inf/N_tot))

        # Normalize the curves by dividing by population size
        low_rho_inc_st = low_rho_inc_st / np.expand_dims(low_rho_pop_s, axis=1)
        low_rho_pre_st = low_rho_pre_st / np.expand_dims(low_rho_pop_s, axis=1)
        mid_rho_inc_st = mid_rho_inc_st / np.expand_dims(mid_rho_pop_s, axis=1)
        mid_rho_pre_st = mid_rho_pre_st / np.expand_dims(mid_rho_pop_s, axis=1)
        hig_rho_inc_st = hig_rho_inc_st / np.expand_dims(hig_rho_pop_s, axis=1)
        hig_rho_pre_st = hig_rho_pre_st / np.expand_dims(hig_rho_pop_s, axis=1)
        tot_inc_st = tot_inc_st / np.expand_dims(tot_pop_s, axis=1)
        tot_pre_st = tot_pre_st / np.expand_dims(tot_pop_s, axis=1)
    
        # Append the curves to the respective lists
        extended_low_rho_inc_st.append(low_rho_inc_st)
        extended_low_rho_pre_st.append(low_rho_pre_st)
        extended_mid_rho_inc_st.append(mid_rho_inc_st)
        extended_mid_rho_pre_st.append(mid_rho_pre_st)
        extended_hig_rho_inc_st.append(hig_rho_inc_st)
        extended_hig_rho_pre_st.append(hig_rho_pre_st)
        extended_tot_inc_st.append(tot_inc_st)
        extended_tot_pre_st.append(tot_pre_st)

    # Stack the curves
    extended_low_rho_inc_st = np.vstack(extended_low_rho_inc_st)
    extended_low_rho_pre_st = np.vstack(extended_low_rho_pre_st)
    extended_mid_rho_inc_st = np.vstack(extended_mid_rho_inc_st)
    extended_mid_rho_pre_st = np.vstack(extended_mid_rho_pre_st)
    extended_hig_rho_inc_st = np.vstack(extended_hig_rho_inc_st)
    extended_hig_rho_pre_st = np.vstack(extended_hig_rho_pre_st)
    extended_tot_inc_st = np.vstack(extended_tot_inc_st)
    extended_tot_pre_st = np.vstack(extended_tot_pre_st)


    #N_inf = np.sum(extended_tot_pre_st[:,-1] * tot_pop_s) 
    #N_tot = np.sum(tot_pop_s) * len(timestamps)
    #print("N_inf={0}, N_tot={1}, r={2}".format(N_inf, N_tot, N_inf/N_tot))
    

    # Average curves
    average_low_rho_inc_t = np.mean(extended_low_rho_inc_st, axis=0)
    average_low_rho_pre_t = np.mean(extended_low_rho_pre_st, axis=0)
    average_mid_rho_inc_t = np.mean(extended_mid_rho_inc_st, axis=0)
    average_mid_rho_pre_t = np.mean(extended_mid_rho_pre_st, axis=0)
    average_hig_rho_inc_t = np.mean(extended_hig_rho_inc_st, axis=0)
    average_hig_rho_pre_t = np.mean(extended_hig_rho_pre_st, axis=0)
    average_tot_inc_t = np.mean(extended_tot_inc_st, axis=0)
    average_tot_pre_t = np.mean(extended_tot_pre_st, axis=0)
    print("obtained r={0}".format(average_tot_pre_t[-1]))

    # Confidence intervals
    #ci_low_rho_inc_t = np.percentile(extended_low_rho_inc_st, [2.5, 97.5], axis=0)
    #ci_low_rho_pre_t = np.percentile(extended_low_rho_pre_st, [2.5, 97.5], axis=0)
    #ci_mid_rho_inc_t = np.percentile(extended_mid_rho_inc_st, [2.5, 97.5], axis=0)
    #ci_mid_rho_pre_t = np.percentile(extended_mid_rho_pre_st, [2.5, 97.5], axis=0)
    #ci_hig_rho_inc_t = np.percentile(extended_hig_rho_inc_st, [2.5, 97.5], axis=0)
    #ci_hig_rho_pre_t = np.percentile(extended_hig_rho_pre_st, [2.5, 97.5], axis=0)
    #ci_tot_inc_t = np.percentile(extended_tot_inc_st, [2.5, 97.5], axis=0)
    #ci_tot_pre_t = np.percentile(extended_tot_pre_st, [2.5, 97.5], axis=0)

    # Compute confidence intervals
    ci_low_rho_inc_t = stats.t.interval(0.95, len(extended_low_rho_inc_st)-1, loc=average_low_rho_inc_t, scale=stats.sem(extended_low_rho_inc_st))
    ci_low_rho_pre_t = stats.t.interval(0.95, len(extended_low_rho_pre_st)-1, loc=average_low_rho_pre_t, scale=stats.sem(extended_low_rho_pre_st))
    ci_mid_rho_inc_t = stats.t.interval(0.95, len(extended_mid_rho_inc_st)-1, loc=average_mid_rho_inc_t, scale=stats.sem(extended_mid_rho_inc_st))
    ci_mid_rho_pre_t = stats.t.interval(0.95, len(extended_mid_rho_pre_st)-1, loc=average_mid_rho_pre_t, scale=stats.sem(extended_mid_rho_pre_st))
    ci_hig_rho_inc_t = stats.t.interval(0.95, len(extended_hig_rho_inc_st)-1, loc=average_hig_rho_inc_t, scale=stats.sem(extended_hig_rho_inc_st))
    ci_hig_rho_pre_t = stats.t.interval(0.95, len(extended_hig_rho_pre_st)-1, loc=average_hig_rho_pre_t, scale=stats.sem(extended_hig_rho_pre_st))
    ci_tot_inc_t = stats.t.interval(0.95, len(extended_tot_inc_st)-1, loc=average_tot_inc_t, scale=stats.sem(extended_tot_inc_st))
    ci_tot_pre_t = stats.t.interval(0.95, len(extended_tot_pre_st)-1, loc=average_tot_pre_t, scale=stats.sem(extended_tot_pre_st))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # SUBPLOT 0
    ax[0].plot(average_low_rho_inc_t, color='dodgerblue', label=r'top ret')
    ax[0].plot(average_mid_rho_inc_t, color='slateblue', label=r'middle $\rho$')
    ax[0].plot(average_hig_rho_inc_t, color='firebrick', label=r'top exp')
    ax[0].plot(average_tot_inc_t, color='black', linestyle='dashed', label=r'total')
    
    ax[0].fill_between(np.arange(len(average_tot_inc_t)), ci_tot_inc_t[0], ci_tot_inc_t[1], color='black', alpha=0.3)
    ax[0].fill_between(np.arange(len(average_low_rho_inc_t)), ci_low_rho_inc_t[0], ci_low_rho_inc_t[1], color='dodgerblue', alpha=0.3)
    ax[0].fill_between(np.arange(len(average_mid_rho_inc_t)), ci_mid_rho_inc_t[0], ci_mid_rho_inc_t[1], color='slateblue', alpha=0.3)
    ax[0].fill_between(np.arange(len(average_hig_rho_inc_t)), ci_hig_rho_inc_t[0], ci_hig_rho_inc_t[1], color='firebrick', alpha=0.3)
    
    ax[0].set_xlabel(r'$t$', fontsize=25)
    ax[0].set_ylabel(r'$i_{\rho}(t)$', fontsize=25)
    ax[0].legend()

    # SUBPLOT 1
    ax[1].plot(average_low_rho_pre_t, color='dodgerblue', label=r'top ret')
    ax[1].plot(average_mid_rho_pre_t, color='slateblue', label=r'middle $\rho$')
    ax[1].plot(average_hig_rho_pre_t, color='firebrick', label=r'top exp')
    ax[1].plot(average_tot_pre_t, color='black', linestyle='dashed', label=r'total')
    
    ax[1].fill_between(np.arange(len(average_tot_inc_t)), ci_tot_pre_t[0], ci_tot_pre_t[1], color='black', alpha=0.3)
    ax[1].fill_between(np.arange(len(average_low_rho_pre_t)), ci_low_rho_pre_t[0], ci_low_rho_pre_t[1], color='dodgerblue', alpha=0.3)
    ax[1].fill_between(np.arange(len(average_mid_rho_pre_t)), ci_mid_rho_pre_t[0], ci_mid_rho_pre_t[1], color='slateblue', alpha=0.3)
    ax[1].fill_between(np.arange(len(average_hig_rho_pre_t)), ci_hig_rho_pre_t[0], ci_hig_rho_pre_t[1], color='firebrick', alpha=0.3)
    
    ax[1].set_xlabel(r'$t$', fontsize=25)
    ax[1].set_ylabel(r'$r_{\rho}(t)$', fontsize=25)
    ax[1].legend()

    # Plot classical SIR analytical solution
    r_inf = ut.sir_prevalence(R0, r_0)
    ax[1].axhline(r_inf, color='steelblue', linestyle='--', label=r'$r_{hom}(\infty)$')

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'macro_t_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D1_dynamics_with_vaccination():
    lower_path = 'config/'
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    header = 'edyna'
    merged_dict = {}
    merged_dict.update(epi_pars)
    merged_dict.update(grid_pars) 
    string_segments = ['vsUnm', 'vf0_']
    filenames = ut.collect_pickle_filenames_by_exclusion(fullpath, header, string_segments)

    nsims = 30
    t_max = 1200

    rho_interval = [0.0, 1.0]

    # Define empty lists for storing the curves
    tev_inc_st = []
    tev_pre_st = []
    trv_inc_st = []
    trv_pre_st = []
    rav_inc_st = []
    rav_pre_st = []

    for filename, i in zip(filenames, range(len(filenames))):
        print("Loop {0}, timestamp: {1}".format(i+1, filename))
        # Build fullname
        fullname = os.path.join(cwd_path, lower_path, filename)

        # Build agent dataframe
        agent_df = an.build_agent_data_frame(fullname, nsims_load=nsims)

        # Filter dataframe by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.025)

        # Compute population in rho intervals
        pop_s = an.number_of_agents_in_rho_interval(agent_df, rho_interval)
        pop_s = an.number_of_agents_in_rho_interval(agent_df, rho_interval)
        
        # Filter dataframe by health status
        agent_df = an.health_status_filter_agent_data_frame(agent_df, health_status='Removed')

        # Rebuild epidemic curves
        inc_st = an.rebuild_incidence_time_series_for_rho_group(agent_df, rho_interval, t_max)
        pre_st = an.rebuild_prevalence_time_series_for_rho_group(agent_df, rho_interval, t_max)
        
        # Normalize the curves by dividing by population size
        inc_st = inc_st / np.expand_dims(pop_s, axis=1)
        pre_st = pre_st / np.expand_dims(pop_s, axis=1)
    
        # Append the curves to the respective lists
        if 'vsTEx' in filename:
            tev_inc_st.append(inc_st)
            tev_pre_st.append(pre_st)
        elif 'vsTRe' in filename:
            trv_inc_st.append(inc_st)
            trv_pre_st.append(pre_st)
        elif 'vsRan' in filename:
            rav_inc_st.append(inc_st)
            rav_pre_st.append(pre_st)

    # Stack the curves
    tev_inc_st = np.vstack(tev_inc_st)
    tev_pre_st = np.vstack(tev_pre_st)
    trv_inc_st = np.vstack(trv_inc_st)
    trv_pre_st = np.vstack(trv_pre_st)
    rav_inc_st = np.vstack(rav_inc_st)
    rav_pre_st = np.vstack(rav_pre_st)

    # Average curves
    average_tev_inc_t = np.mean(tev_inc_st, axis=0)
    average_tev_pre_t = np.mean(tev_pre_st, axis=0)
    average_trv_inc_t = np.mean(trv_inc_st, axis=0)
    average_trv_pre_t = np.mean(trv_pre_st, axis=0)
    average_rav_inc_t = np.mean(rav_inc_st, axis=0)
    average_rav_pre_t = np.mean(rav_pre_st, axis=0)

    # Compute confidence intervals
    ci_tev_inc_t = stats.t.interval(0.95, len(tev_inc_st)-1, loc=average_tev_inc_t, scale=stats.sem(tev_inc_st))
    ci_tev_pre_t = stats.t.interval(0.95, len(tev_pre_st)-1, loc=average_tev_pre_t, scale=stats.sem(tev_pre_st))
    ci_trv_inc_t = stats.t.interval(0.95, len(trv_inc_st)-1, loc=average_trv_inc_t, scale=stats.sem(trv_inc_st))
    ci_trv_pre_t = stats.t.interval(0.95, len(trv_pre_st)-1, loc=average_trv_pre_t, scale=stats.sem(trv_pre_st))
    ci_rav_inc_t = stats.t.interval(0.95, len(rav_inc_st)-1, loc=average_rav_inc_t, scale=stats.sem(rav_inc_st))
    ci_rav_pre_t = stats.t.interval(0.95, len(rav_pre_st)-1, loc=average_rav_pre_t, scale=stats.sem(rav_pre_st))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # SUBPLOT 0
    ax[0].plot(average_tev_inc_t, color='tomato', label='top exp')
    ax[0].plot(average_trv_inc_t, color='lightskyblue', label='top ret')
    ax[0].plot(average_rav_inc_t, color='goldenrod', label= 'random')

    ax[0].fill_between(np.arange(len(average_tev_inc_t)), ci_tev_inc_t[0], ci_tev_inc_t[1], color='black', alpha=0.3)
    ax[0].fill_between(np.arange(len(average_trv_inc_t)), ci_trv_inc_t[0], ci_trv_inc_t[1], color='dodgerblue', alpha=0.3)
    ax[0].fill_between(np.arange(len(average_rav_inc_t)), ci_rav_inc_t[0], ci_rav_inc_t[1], color='slateblue', alpha=0.3)

    ax[0].set_xlabel(r'$t$', fontsize=25)
    ax[0].set_ylabel(r'$i_{\rho}(t)$', fontsize=25)
    ax[0].legend()

    # SUBPLOT 1
    ax[1].plot(average_tev_pre_t, color='tomato', label='top exp')
    ax[1].plot(average_trv_pre_t, color='lightskyblue', label='top ret')
    ax[1].plot(average_rav_pre_t, color='goldenrod', label='random')

    ax[1].fill_between(np.arange(len(average_tev_pre_t)), ci_tev_pre_t[0], ci_tev_pre_t[1], color='black', alpha=0.3)
    ax[1].fill_between(np.arange(len(average_trv_pre_t)), ci_trv_pre_t[0], ci_trv_pre_t[1], color='dodgerblue', alpha=0.3)
    ax[1].fill_between(np.arange(len(average_rav_pre_t)), ci_rav_pre_t[0], ci_rav_pre_t[1], color='slateblue', alpha=0.3)

    ax[1].set_xlabel(r'$t$', fontsize=25)
    ax[1].set_ylabel(r'$r_{\rho}(t)$', fontsize=25)
    ax[1].legend()

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    base_name = 'macro_vaccine_comp'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D1_invader_infected_rho():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    nlocs = 2500
    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001
    nlocs_eff = len(attr_l[attr_l > attr_cutoff])

    nsims = 25
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    # Define empty lists for storing the curves
    extended_invader_rho_sl = []
    extended_infected_rho_a = []
    extended_rho_a = []
    nlocs_inv = 0

    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))
        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)

        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        rho_a = an.get_rho_values(agent_df)
        nsims_eff = an.number_of_simulations(agent_df)

        invader_rho_sl = an.collect_location_invaders_rho(event_df, space_df, nlocs, nlocs_eff, t_max)
        sum_nan_elements = np.sum(np.isnan(invader_rho_sl))
        nlocs_inv += nlocs_eff * nsims_eff - sum_nan_elements

        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        inf_rho_a = an.get_rho_values(agent_df)

        N_inf = len(inf_rho_a)
        N_tot = len(rho_a)
        print("N_inf={0}".format(N_inf))
        print("N_tot={0}".format(N_tot))
        print("ratio={0}", N_inf/N_tot)

        extended_invader_rho_sl.extend(invader_rho_sl.flatten())
        extended_rho_a.extend(rho_a)
        extended_infected_rho_a.extend(inf_rho_a)

    N_inf = len(extended_infected_rho_a)
    N_tot = len(extended_rho_a)
    print("N_inf={0}, N_tot={1}, r={2}".format(N_inf, N_tot, N_inf/N_tot))

    # Build invasion & infected histograms
    bins_rho = 30
    hist_rho, bin_edges_rho = np.histogram(extended_rho_a, bins_rho)
    hist_invader, bin_invader_edges = np.histogram(extended_invader_rho_sl, bin_edges_rho)
    invader_mid_points = np.asarray([bin_edges_rho[i] + (bin_edges_rho[i+1] - bin_edges_rho[i]) / 2.0 
                                    for i in range(len(bin_edges_rho) -1)])
    hist_infected, bin_infected_edges = np.histogram(extended_infected_rho_a, bin_edges_rho)
    infected_mid_points = np.asarray([bin_edges_rho[i] + (bin_edges_rho[i+1] - bin_edges_rho[i]) / 2.0 
                                    for i in range(len(bin_edges_rho) -1)])
    infected_fraction = hist_infected / hist_rho

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))

    # SUBPLOT 0: HISTOGRAM. INVADER'S RHO DISTRIBUTION
    density = True
    ax[0].hist(extended_invader_rho_sl, bin_edges_rho, density=density, color='lightskyblue', alpha=0.5)
    ax[0].axvline(0.5, color='indigo', linestyle='--')

    # Right y-axis: Invasion expectation
    ax_r0 = ax[0].twinx()
    invader_fraction = hist_invader / (nlocs_inv)
    ax_r0.scatter(invader_mid_points, invader_fraction, marker='o', color='black', label=r'$N_{inv,\rho}/V_{{eff}}$')
    expected_share = hist_rho / len(extended_rho_a)
    ax_r0.plot(invader_mid_points, expected_share, linestyle='--', color='indigo', label=r'null: $N_{inv,\rho}/N_{\rho}$')
    
    # Subplot 0 settings
    title = r'invasion share by $\rho$ profile'
    ax[0].set_title(title, fontsize=30)
    ax[0].set_xlabel(r"$\rho$", fontsize=25)
    ax[0].set_ylabel(r"$P_{{inv},\rho}$", fontsize=25)
    ax[0].tick_params(axis='both', labelsize=15)
    #ax_r21.set_ylim(0.0, 0.2)
    handles, labels = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax_r0.get_legend_handles_labels()
    handles += handles2
    labels += labels2
    ax[0].legend(handles, labels, loc=0, fontsize=20)

    # SUBPLOT 1: HISTOGRAM. INVADER'S RHO DISTRIBUTION
    density = True
    ax[1].hist(extended_infected_rho_a, bin_edges_rho, density=density, color='lightskyblue', alpha=0.5)
    ax[1].axvline(0.5, color='indigo', linestyle='--')

    # Right y-axis: Invasion expectation
    ax_r1 = ax[1].twinx()
    ax_r1.scatter(infected_mid_points, infected_fraction, marker='o', color='black', label=r'$n_{inf,\rho}$')
    expected_share = hist_rho / len(extended_rho_a)
    # Plot classical SIR analytical solution
    r_inf = ut.sir_prevalence(R0, r_0)
    ax_r1.axhline(r_inf, color='steelblue', linestyle='--', label=r'$r_{hom}(\infty)$')
    
    # Subplot 0 settings
    title = r'infection share by $\rho$ profile'
    ax[1].set_title(title, fontsize=30)
    ax[1].set_xlabel(r"$\rho$", fontsize=25)
    ax[1].set_ylabel(r"$P_{{inf},\rho}$", fontsize=25)
    ax[1].tick_params(axis='both', labelsize=15)
    #ax_r21.set_ylim(0.0, 0.2)
    handles, labels = ax[1].get_legend_handles_labels()
    handles2, labels2 = ax_r1.get_legend_handles_labels()
    handles += handles2
    labels += labels2
    ax[1].legend(handles, labels, loc=0, fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'inv_inf_t_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D1_micro_time_with_rho():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    nlocs = 2500
    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001
    nlocs_eff = len(attr_l[attr_l > attr_cutoff])

    nsims = 25
    t_max = 1200

    # Define empty lists for storing the curves
    invader_rho_a = []
    invasion_time_a = []
    infected_rho_a = []
    infection_time_a = []

    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))
        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build epidemic data fullname
        lower_path = 'data'
        fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build agent & event dataframes
        agent_df = an.build_agent_data_frame(fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(fullname, nsims_load=nsims)
        
        # Filter event dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        # Filter agent dataframe by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        
        invader_rho_sl = an.collect_location_invaders_rho(event_df, space_df, nlocs, nlocs_eff, t_max)
        invasion_time_sl = an.collect_invasion_times(event_df, space_df, nlocs, nlocs_eff, t_max)
        invader_rho_a.extend(invader_rho_sl.flatten())
        invasion_time_a.extend(invasion_time_sl.flatten())

        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        infected_rho_a.extend(an.get_rho_values(agent_df))
        infection_time_a.extend(an.get_infection_times(agent_df))

    # Prepare figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))

    # Filter out NaN values
    invader_rho_a = np.array(invader_rho_a)
    invasion_time_a = np.array(invasion_time_a)
    infected_rho_a = np.array(infected_rho_a)
    infection_time_a = np.array(infection_time_a)
    valid_indices = np.logical_not(np.isnan(invader_rho_a) | np.isnan(invasion_time_a))
    filt_invader_rho_a = invader_rho_a[valid_indices]
    filt_invasion_time_a = invasion_time_a[valid_indices]

    # Define the number of bins for rho values & calculate bin edges for rho values
    num_bins = 20
    rho_bins = np.linspace(np.min(filt_invader_rho_a), np.max(filt_invader_rho_a), num_bins + 1)

    # Compute the average invasion time and other statistics for each rho bin
    average_invasion_times = []
    std_dev_invasion_times = []
    lower_95CI_invasion_times = []
    upper_95CI_invasion_times = []

    for i in range(num_bins):
        # Find the indices of invasion times within the current rho bin
        indices = np.where((filt_invader_rho_a >= rho_bins[i]) & (filt_invader_rho_a < rho_bins[i + 1]))
        # Calculate average invasion time
        average_time = np.mean(filt_invasion_time_a[indices])
        average_invasion_times.append(average_time)
        # Calculate standard deviation of invasion times
        std_dev_time = np.std(filt_invasion_time_a[indices], ddof=1)  # ddof=1 for sample standard deviation
        std_dev_invasion_times.append(std_dev_time)
        # Calculate 95% confidence intervals
        num_samples = len(filt_invasion_time_a[indices])
        if num_samples > 1:
            t_value = stats.t.ppf(0.975, df=num_samples-1)  # t-value for 95% CI
            standard_error = std_dev_time / np.sqrt(num_samples)
            lower_95CI = average_time - t_value * standard_error
            upper_95CI = average_time + t_value * standard_error
        else:
            # For small sample sizes (n <= 1), we cannot calculate confidence intervals
            lower_95CI = np.nan
            upper_95CI = np.nan

        lower_95CI_invasion_times.append(lower_95CI)
        upper_95CI_invasion_times.append(upper_95CI)

    # Convert the lists to numpy arrays for convenience (optional)
    average_invasion_times = np.array(average_invasion_times)
    std_dev_invasion_times = np.array(std_dev_invasion_times)
    lower_95CI_invasion_times = np.array(lower_95CI_invasion_times)
    upper_95CI_invasion_times = np.array(upper_95CI_invasion_times)

    # Plot binned rho values against average invasion times
    ax[0].plot(rho_bins[:-1], average_invasion_times, marker='o', color='teal', label=r'$\langle t_{{inv}}\rangle $')
    ax[0].fill_between(rho_bins[:-1], lower_95CI_invasion_times, upper_95CI_invasion_times, color='teal', alpha=0.2, label='95% CI')

    # Subplot 1 settings
    title = r'Invasion times'
    ax[0].set_title(title, fontsize=30)
    ax[0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0].set_ylabel(r"$t_{{inv}}$", fontsize=30)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].legend(fontsize=25)

    # Define the number of bins for rho values & calculate bin edges for rho values
    num_bins = 20
    rho_bins = np.linspace(np.min(infected_rho_a), np.max(infected_rho_a), num_bins + 1)

    # Compute the average infection time and other statistics for each rho bin
    average_infection_times = []
    std_dev_infection_times = []
    lower_95CI_infection_times = []
    upper_95CI_infection_times = []

    for i in range(num_bins):
        # Find the indices of infection times within the current rho bin
        indices = np.where((infected_rho_a >= rho_bins[i]) & (infected_rho_a < rho_bins[i + 1]))
        # Calculate average infection time
        average_time = np.mean(infection_time_a[indices])
        average_infection_times.append(average_time)
        # Calculate standard deviation of infection times
        std_dev_time = np.std(infection_time_a[indices], ddof=1)  # ddof=1 for sample standard deviation
        std_dev_infection_times.append(std_dev_time)
        # Calculate 95% confidence intervals
        num_samples = len(infection_time_a[indices])
        if num_samples > 1:
            t_value = stats.t.ppf(0.975, df=num_samples-1)  # t-value for 95% CI
            standard_error = std_dev_time / np.sqrt(num_samples)
            lower_95CI = average_time - t_value * standard_error
            upper_95CI = average_time + t_value * standard_error
        else:
            # For small sample sizes (n <= 1), we cannot calculate confidence intervals
            lower_95CI = np.nan
            upper_95CI = np.nan
        lower_95CI_infection_times.append(lower_95CI)
        upper_95CI_infection_times.append(upper_95CI)

    # Convert the lists to numpy arrays for convenience (optional)
    average_infection_times = np.array(average_infection_times)
    std_dev_infection_times = np.array(std_dev_infection_times)
    lower_95CI_infection_times = np.array(lower_95CI_infection_times)
    upper_95CI_infection_times = np.array(upper_95CI_infection_times)

    # Plot binned rho values against average invasion times
    ax[1].plot(rho_bins[:-1], average_infection_times, marker='o', color='teal', label=r'$\langle t_{{inv}}\rangle $')
    ax[1].fill_between(rho_bins[:-1], lower_95CI_infection_times, upper_95CI_infection_times, color='teal', alpha=0.2, label='95% CI')

    # Subplot 1 settings
    title = r'Infection times'
    ax[1].set_title(title, fontsize=30)
    ax[1].set_xlabel(r"$\rho$", fontsize=30)
    ax[1].set_ylabel(r"$t_{{inf}}$", fontsize=30)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].legend(fontsize=25)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'micro_t_rho_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D1_force_of_infection_contribution():
    lower_path = 'config/'
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    t_max = 1200

    # Define empty lists for storing the curves
    extended_results = []
    infected_rho_a = []
    t_inf_a = []
    cum_size_a = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        print("Counting")
        results = an.count_event_sizes_experienced_in_rho_t_inf_groups(agent_df, event_df, trajectory_df, nlocs, t_max)

        extended_results.append(results)

    for results in extended_results:
        for triad in results:
            infected_rho_a.append(triad[0])
            t_inf_a.append(triad[1])
            cum_size_a.append(triad[2])

    infected_rho_a = np.array(infected_rho_a)
    t_inf_a = np.array(t_inf_a)
    cum_size_a = np.array(cum_size_a)

    # Filter special cases
    t_inf_lower = 4
    t_inf_upper = 21
    infected_rho_a = np.array(infected_rho_a)
    filt_infected_rho_a = infected_rho_a[np.logical_and(t_inf_upper > t_inf_a, t_inf_a > t_inf_lower)]
    filt_cum_size_a = cum_size_a[np.logical_and(t_inf_upper > t_inf_a, t_inf_a > t_inf_lower)]
    t_inf_a = t_inf_a[np.logical_and(t_inf_upper > t_inf_a, t_inf_a > t_inf_lower)]

    rho_bins = 20
    t_inf_bins = t_inf_upper - t_inf_lower + 1
    t_inf_bins_array = np.arange(t_inf_lower - 0.5, t_inf_upper + 1.5)
    hist, rho_bins_array, t_inf_bins_array = np.histogram2d(filt_infected_rho_a, t_inf_a, bins=[rho_bins, t_inf_bins_array])
    hist_w, rho_bins_array, t_inf_bins_array = np.histogram2d(filt_infected_rho_a, t_inf_a, bins=[rho_bins, t_inf_bins_array], weights=filt_cum_size_a)

    #hist, rho_bins_array, t_inf_bins_array = np.histogram2d(filt_infected_rho_a, t_inf_a, bins=[rho_bins, t_inf_bins])
    #hist_w, rho_bins_array, t_inf_bins_array = np.histogram2d(filt_infected_rho_a, t_inf_a, bins=[rho_bins, t_inf_bins], weights=filt_cum_size_a)
    
    for i in range(rho_bins):
        for j in range(t_inf_bins):
            bin_sum = np.sum(hist[i, j])
            if bin_sum != 0:
                hist_w[i, j] /= bin_sum
    
    # Plotting the 2D histogram
    fig, ax = plt.subplots()

    im = ax.imshow(
            hist_w.T, 
            origin='lower', 
            extent=[rho_bins_array[0], rho_bins_array[-1], t_inf_bins_array[0], t_inf_bins_array[-1]], 
            aspect='auto', 
            cmap='Blues'
        )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('mean new cases contribution', fontsize=15)

    ax.axvline(0.5, color='crimson', linestyle='--')

    # Subplot 11 settings
    ax.set_yticks(np.arange(t_inf_bins_array[0], t_inf_bins_array[-1]+1))
   
    # Manually adjust the y-axis tick positions
    yticks_adjusted = np.arange(t_inf_bins_array[0] - 0.5, t_inf_bins_array[-1] + 0.5)
    ax.set_yticks(yticks_adjusted)
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_ylabel(r"$t_{{inf}}$", fontsize=20)
    ax.tick_params(axis='both', labelsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'foi_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D1_visits_until_infection():
    lower_path = 'config/'
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)
    #timestamps = timestamps[:2]

    nsims = 25

    # Define empty lists for storing the curves
    extended_results = []
    visits_until_a = []
    infected_where_a = []
    home_a = []
    rho_a = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)
    
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        event_df = an.simulations_filter_event_data_frame(event_df, nsims_load=nsims)

        print("Counting")
        results = an.count_visits_until_infection_and_where(agent_df, trajectory_df)

        extended_results.append(results)

    for results in extended_results:
        visits_until_a.extend(results[0])
        infected_where_a.extend(results[1])
        home_a.extend(results[2])
        rho_a.extend(results[3])

    rho_a = np.array(rho_a)
    home_a = np.array(home_a)
    infected_where_a = np.array(infected_where_a)
    visits_until_a = np.array(visits_until_a)

    same_location_indices = np.where(infected_where_a == home_a)
    same_explorers_indices = np.where((infected_where_a == home_a) & (rho_a > 0.5))
    same_returners_indices = np.where((infected_where_a == home_a) & (rho_a < 0.5))
    same_count = len(same_location_indices[0])
    same_exp_dens = np.round(100.0 * len(same_explorers_indices[0]) / same_count, 2)
    same_ret_dens = np.round(100.0 * len(same_returners_indices[0]) / same_count, 2)
    diff_location_indices = np.where(infected_where_a != home_a)
    diff_explorers_indices = np.where((infected_where_a != home_a) & (rho_a > 0.5))
    diff_returners_indices = np.where((infected_where_a != home_a) & (rho_a < 0.5))
    diff_count = len(diff_location_indices[0])
    diff_exp_dens = np.round(100.0 * len(diff_explorers_indices[0]) / diff_count, 2)
    diff_ret_dens = np.round(100.0 * len(diff_returners_indices[0]) / diff_count, 2)
    same_dens = same_count / (same_count + diff_count)
    diff_dens = diff_count / (same_count + diff_count)
    same_dens = np.round(100.0 * same_dens, 2)
    diff_dens = np.round(100.0 * diff_dens, 2)

    print("Agents infected at home: {0}".format(same_dens))
    print("Explorers infected at home: {0}".format(same_exp_dens))
    print("Returners infected at home: {0}".format(same_ret_dens))
    print("Agents infected outside: {0}".format(diff_dens))
    print("Explorers infected outside: {0}".format(diff_exp_dens))
    print("Returners infected outside: {0}".format(diff_ret_dens))

    # Plot
    fig, ax = plt.subplots()
    
    ax.scatter(
        rho_a[same_location_indices], 
        visits_until_a[same_location_indices], 
        c='dodgerblue', 
        alpha=0.5,
        label='at home: {0}%'.format(same_dens),
        )
    ax.scatter(
        rho_a[diff_location_indices], 
        visits_until_a[diff_location_indices], 
        c='firebrick', 
        alpha=0.5,
        label='outside: {0}%'.format(diff_dens)
        )
    
    ax.text(
        0.85, 
        0.55, 
        f'home exp: {same_exp_dens}%', 
        ha='right', 
        va='top', 
        color='black',
        transform=ax.transAxes, 
        fontsize=10,
        zorder=10,
        )
    ax.text(
        0.35, 
        0.55, 
        f'home ret: {same_ret_dens}%', 
        ha='right', 
        va='top', 
        color='black',
        transform=ax.transAxes, 
        fontsize=10,
        zorder=10,
        )
    ax.text(
        0.85, 
        0.15, 
        f'out exp: {diff_exp_dens}%', 
        ha='right', 
        va='top',
        color='black',
        transform=ax.transAxes, 
        fontsize=10,
        zorder=10,
        )
    ax.text(
        0.35, 
        0.15,
        f'out ret: {diff_ret_dens}%', 
        ha='right', 
        va='top',
        color='black',
        transform=ax.transAxes, 
        fontsize=10,
        zorder=10,
        )

    ax.axvline(0.5, color='teal', linestyle='--')
    ax.set_xlabel(r'$\rho$', fontsize=25)
    ax.set_ylabel(r'visits until infection', fontsize=25)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'visits_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D2_invader_infected_rho_map():
    lower_path = 'config/'
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    # Define empty lists for storing the curves
    app_INV_RH = []
    app_INF_RH = [] 

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))
        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        INV_RHO = an.compute_invader_average_rho_map(event_df, nlocs, t_max)
        INF_RHO = an.compute_infected_average_rho_map(agent_df, nlocs)

        # Append
        app_INV_RH.append(INV_RHO)
        app_INF_RH.append(INF_RHO)

    # Average
    avg_INV_RH = np.nanmean(app_INV_RH, axis=0)
    avg_INF_RH = np.nanmean(app_INF_RH, axis=0)

    # Prepare figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))

    # SUBPLOT 0: IMSHOW. INVADER RHO SPATIAL MAP
    im0 = ax[0].imshow(avg_INV_RH.T, cmap='coolwarm')
    im0.set_clim(vmin=0.0, vmax=1.0)
    cbar0 = fig.colorbar(im0, ax=ax[0], shrink=0.7)
    cbar0.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

    # Settings 00
    ax[0].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    ax[0].set_ylabel("latitude (\u00b0 N)", fontsize=25)
    title = "invader average mobility profile"
    ax[0].set_title(title, fontsize=25)
    ax[0].invert_yaxis()
    ax[0].tick_params(axis='both', labelsize=18)

    new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
    new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
    x_ticks_pos = range(0, 51, 10)
    y_ticks_pos = range(0, 51, 10)
    ax[0].set_xticks(x_ticks_pos)
    ax[0].set_yticks(y_ticks_pos)
    ax[0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    ax[0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
    
    # SUBPLOT 1: IMSHOW. INFECTED'S AVERAGE RHO SPATIAL MAP
    im1 = ax[1].imshow(avg_INF_RH.T, cmap='coolwarm')
    im1.set_clim(vmin=0, vmax=1)
    cbar1 = fig.colorbar(im1, ax=ax[1], shrink=0.7)
    cbar1.set_label(r'infected $\langle\rho\rangle$', fontsize=25)

    # Settings 01
    title = "infected average mobility profile"
    ax[1].set_title(title, fontsize=25)
    #ax[1].set_ylabel("latitude (\u00b0 N)", fontsize=25)
    ax[1].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    ax[1].invert_yaxis()
    ax[1].tick_params(axis='both', labelsize=18)
    
    ax[1].set_xticks(x_ticks_pos)
    ax[1].set_yticks(y_ticks_pos)
    ax[1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    ax[1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'rho_map_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D2_exploration_return_step():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_inf_epr_a = []
    ext_inf_rho_a = []
    ext_infected_where_a = []
    ext_infected_when_a = []
    ext_infected_where_freq_a = []
    ext_inv_epr_a = []
    ext_inv_rho_a = []
    ext_invaded_where_a = []
    ext_invaded_when_a = []
    ext_invaded_where_freq_a = []

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        # Infection process        
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        inv_results, inf_results = an.collect_infection_epr_steps(agent_df, trajectory_df, nsims, nlocs)

        # Infection results
        inf_epr_a, inf_rho_a, infected_where_a, infected_when_a, infected_where_freq_a = inf_results
        inv_epr_a, inv_rho_a, invaded_where_a, invaded_when_a, invaded_where_freq_a = inv_results
        ext_inf_epr_a.extend(inf_epr_a)
        ext_inf_rho_a.extend(inf_rho_a)
        ext_infected_where_a.extend(infected_where_a)
        ext_infected_when_a.extend(infected_when_a)
        ext_infected_where_freq_a.extend(infected_where_freq_a)
        ext_inv_epr_a.extend(inv_epr_a)
        ext_inv_rho_a.extend(inv_rho_a)
        ext_invaded_where_a.extend(invaded_where_a)
        ext_invaded_when_a.extend(invaded_when_a)
        ext_invaded_where_freq_a.extend(invaded_where_freq_a)

    ext_inf_epr_a = np.array(ext_inf_epr_a)
    ext_inf_rho_a = np.array(ext_inf_rho_a)
    ext_infected_when_a = np.array(ext_infected_when_a)
    ext_infected_where_freq_a = np.array(ext_infected_where_freq_a)
    ext_infected_where_a = np.array(ext_infected_where_a)
    ext_inv_epr_a = np.array(ext_inv_epr_a)
    ext_inv_rho_a = np.array(ext_inv_rho_a)
    ext_invaded_when_a = np.array(ext_invaded_when_a)
    ext_invaded_where_freq_a = np.array(ext_invaded_where_freq_a)
    ext_invaded_where_a = np.array(ext_invaded_where_a)

    # Filter non-invasion data
    ext_inv_epr_a = ext_inv_epr_a[ext_invaded_where_freq_a != 0]
    ext_inv_rho_a = ext_inv_rho_a[ext_invaded_where_freq_a != 0]
    ext_invaded_when_a = ext_invaded_when_a[ext_invaded_where_freq_a != 0]
    ext_invaded_where_a = ext_invaded_where_a[ext_invaded_where_freq_a != 0]
    ext_invaded_where_freq_a = ext_invaded_where_freq_a[ext_invaded_where_freq_a != 0]

    # Create the figure and axes objects
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    # SUBPLOT 0: Hexbin
    hb0 = ax[0, 0].hexbin(
        ext_inv_rho_a[ext_inv_epr_a == 1],
        ext_invaded_where_freq_a[ext_inv_epr_a == 1],
        gridsize=30,
        cmap='Reds',
    )
    ax[0, 0].axvline(0.5, color='teal', linestyle='--')
    ax[0, 0].set_title(r"invasions under exploration", fontsize=25)
    ax[0, 0].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    ax[0, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 0].tick_params(axis='both', labelsize=20)
    #ax[0,0].legend(fontsize=20)
    fig.colorbar(hb0, ax=ax[0, 0])

    # SUBPLOT 1: Hexbin
    hb1 = ax[0, 1].hexbin(
        ext_inv_rho_a[ext_inv_epr_a != 1],
        ext_invaded_where_freq_a[ext_inv_epr_a != 1],
        gridsize=30,
        cmap='Blues',
    )
    ax[0, 1].axvline(0.5, color='teal', linestyle='--')
    ax[0, 1].set_title(r"invasions under preferential return", fontsize=25)
    ax[0, 1].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    ax[0, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 1].tick_params(axis='both', labelsize=20)
    #ax[0,1].legend(fontsize=20)
    fig.colorbar(hb1, ax=ax[0, 1])

    # SUBPLOT 2: Hexbin
    hb2 = ax[1, 0].hexbin(
        ext_inf_rho_a[ext_inf_epr_a == 1],
        ext_infected_where_freq_a[ext_inf_epr_a == 1],
        gridsize=30,
        cmap='Reds',
    )
    ax[1, 0].axvline(0.5, color='teal', linestyle='--')
    ax[1, 0].set_title(r"infections under exploration", fontsize=25)
    ax[1, 0].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    ax[1, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 0].tick_params(axis='both', labelsize=20)
    #ax[1,0].legend(fontsize=20)
    fig.colorbar(hb2, ax=ax[1, 0])

    # SUBPLOT 3: Hexbin
    hb3 = ax[1, 1].hexbin(
        ext_inf_rho_a[ext_inf_epr_a != 1],
        ext_infected_where_freq_a[ext_inf_epr_a != 1],
        gridsize=30,
        cmap='Blues',
    )
    ax[1, 1].axvline(0.5, color='teal', linestyle='--')
    ax[1, 1].set_title(r"infections under preferential return", fontsize=25)
    ax[1, 1].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    ax[1, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 1].tick_params(axis='both', labelsize=20)
    #ax[1,1].legend(fontsize=20)
    fig.colorbar(hb3, ax=ax[1, 1])

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'epr_step_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D2_epr_step_attractiveness_and_rho():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_inf_epr_a = []
    ext_inf_rho_a = []
    ext_infected_where_a = []
    ext_infected_when_a = []
    ext_attract_inf_a = []
    ext_infected_where_freq_a = []
    ext_inv_epr_a = []
    ext_inv_rho_a = []
    ext_invaded_where_a = []
    ext_invaded_when_a = []
    ext_attract_inv_a = []
    ext_invaded_where_freq_a = []

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        # Infection process        
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        inv_results, inf_results = an.collect_infection_epr_steps(agent_df, trajectory_df, nsims, nlocs)

        # Infection results
        inf_epr_a, inf_rho_a, infected_where_a, infected_when_a, infected_where_freq_a = inf_results
        inv_epr_a, inv_rho_a, invaded_where_a, invaded_when_a, invaded_where_freq_a = inv_results
        ext_inf_epr_a.extend(inf_epr_a)
        ext_inf_rho_a.extend(inf_rho_a)
        ext_infected_where_a.extend(infected_where_a)
        ext_infected_when_a.extend(infected_when_a)
        ext_infected_where_freq_a.extend(infected_where_freq_a)
        attract_inf_a = [space_df.loc[space_df['id'] == l, 'attractiveness'].values[0] for l in infected_where_a]
        ext_attract_inf_a.extend(attract_inf_a)
        ext_inv_epr_a.extend(inv_epr_a)
        ext_inv_rho_a.extend(inv_rho_a)
        ext_invaded_where_a.extend(invaded_where_a)
        ext_invaded_when_a.extend(invaded_when_a)
        ext_invaded_where_freq_a.extend(invaded_where_freq_a)
        attract_inv_a = [space_df.loc[space_df['id'] == l, 'attractiveness'].values[0] for l in invaded_where_a]
        ext_attract_inv_a.extend(attract_inv_a)
    
    ext_inf_epr_a = np.array(ext_inf_epr_a)
    ext_inf_rho_a = np.array(ext_inf_rho_a)
    ext_infected_when_a = np.array(ext_infected_when_a)
    ext_infected_where_freq_a = np.array(ext_infected_where_freq_a)
    ext_infected_where_a = np.array(ext_infected_where_a)
    ext_attract_inf_a = np.array(ext_attract_inf_a)
    ext_inv_epr_a = np.array(ext_inv_epr_a)
    ext_inv_rho_a = np.array(ext_inv_rho_a)
    ext_invaded_when_a = np.array(ext_invaded_when_a)
    ext_invaded_where_freq_a = np.array(ext_invaded_where_freq_a)
    ext_invaded_where_a = np.array(ext_invaded_where_a)
    ext_attract_inv_a = np.array(ext_attract_inv_a)

    # Filter non-invasion data
    ext_inv_epr_a = ext_inv_epr_a[ext_invaded_where_freq_a != 0]
    ext_inv_rho_a = ext_inv_rho_a[ext_invaded_where_freq_a != 0]
    ext_invaded_when_a = ext_invaded_when_a[ext_invaded_where_freq_a != 0]
    ext_invaded_where_a = ext_invaded_where_a[ext_invaded_where_freq_a != 0]
    ext_attract_inv_a = ext_attract_inv_a[ext_invaded_where_freq_a != 0]
    ext_invaded_where_freq_a = ext_invaded_where_freq_a[ext_invaded_where_freq_a != 0]

    # Create the figure and axes objects
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    # SUBPLOT 00: Hexbin
    hb00 = ax[0, 0].hexbin(
    ext_inv_rho_a[ext_inv_epr_a == 1],
    ext_invaded_where_freq_a[ext_inv_epr_a == 1],
    C=ext_attract_inv_a[ext_inv_epr_a == 1],
    gridsize=50,
    cmap='viridis',
    reduce_C_function=np.mean
    )
    ax[0, 0].axvline(0.5, color='teal', linestyle='--')
    ax[0, 0].set_title("Invasions under exploration", fontsize=25)
    ax[0, 0].set_ylabel(r'$f_{inf}$', fontsize=25)
    ax[0, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 0].tick_params(axis='both', labelsize=20)
    cbar00 = fig.colorbar(hb00, ax=ax[0, 0])
    cbar00.set_label(r'$A$', fontsize=25)

    #sc00 = ax[0, 0].scatter(
    #    ext_inv_rho_a[ext_inv_epr_a == 1],
    #    ext_invaded_where_freq_a[ext_inv_epr_a == 1],
    #    c= ext_attract_inv_a[ext_inv_epr_a == 1],
    #    cmap='viridis'
    #    )
    #ax[0, 0].axvline(0.5, color='teal', linestyle='--')
    #ax[0, 0].set_title(r"invasions under exploration", fontsize=25)
    #ax[0, 0].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    #ax[0, 0].set_xlabel(r'$\rho$', fontsize=25)
    #ax[0, 0].tick_params(axis='both', labelsize=20)
    ##ax[0,0].legend(fontsize=20)
    #cbar00 = fig.colorbar(sc00, ax=ax[0, 0])
    #cbar00.set_label(r'$A$', fontsize=25)

    # SUBPLOT 01: Hexbin
    hb01 = ax[0, 1].hexbin(
    ext_inv_rho_a[ext_inv_epr_a != 1],
    ext_invaded_where_freq_a[ext_inv_epr_a != 1],
    C=ext_attract_inv_a[ext_inv_epr_a != 1],
    gridsize=50,
    cmap='viridis',
    reduce_C_function=np.mean
    )
    ax[0, 1].axvline(0.5, color='teal', linestyle='--')
    ax[0, 1].set_title("Invasions under preferential return", fontsize=25)
    ax[0, 1].set_ylabel(r'$f_{inf}$', fontsize=25)
    ax[0, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 1].tick_params(axis='both', labelsize=20)
    cbar01 = fig.colorbar(hb01, ax=ax[0, 1])
    cbar01.set_label(r'$A$', fontsize=25)

    #sc1 = ax[0, 1].scatter(
    #    ext_inv_rho_a[ext_inv_epr_a != 1],
    #    ext_invaded_where_freq_a[ext_inv_epr_a != 1],
    #    c=ext_attract_inv_a[ext_inv_epr_a != 1],
    #    cmap='viridis'
    #    )
    #ax[0, 1].axvline(0.5, color='teal', linestyle='--')
    #ax[0, 1].set_title(r"invasions under preferential return", fontsize=25)
    #ax[0, 1].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    #ax[0, 1].set_xlabel(r'$\rho$', fontsize=25)
    #ax[0, 1].tick_params(axis='both', labelsize=20)
    ##ax[0,1].legend(fontsize=20)
    #cbar1 = fig.colorbar(sc1, ax=ax[0, 1])
    #cbar1.set_label(r'$A$', fontsize=25)

    # SUBPLOT 10: Hexbin
    hb10 = ax[1, 0].hexbin(
    ext_inf_rho_a[ext_inf_epr_a == 1],
    ext_infected_where_freq_a[ext_inf_epr_a == 1],
    C=ext_attract_inf_a[ext_inf_epr_a == 1],
    gridsize=50,
    cmap='viridis',
    reduce_C_function=np.mean
    )
    ax[1, 0].axvline(0.5, color='teal', linestyle='--')
    ax[1, 0].set_title("Invasions under preferential return", fontsize=25)
    ax[1, 0].set_ylabel(r'$f_{inf}$', fontsize=25)
    ax[1, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 0].tick_params(axis='both', labelsize=20)
    cbar10 = fig.colorbar(hb10, ax=ax[1, 0])
    cbar10.set_label(r'$A$', fontsize=25)

    #sc2 = ax[1, 0].scatter(
    #    ext_inf_rho_a[ext_inf_epr_a == 1],
    #    ext_infected_where_freq_a[ext_inf_epr_a == 1],
    #    c= ext_attract_inf_a[ext_inf_epr_a == 1],
    #    cmap='viridis'
    #    )
    #ax[1, 0].axvline(0.5, color='teal', linestyle='--')
    #ax[1, 0].set_title(r"infections under exploration", fontsize=25)
    #ax[1, 0].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    #ax[1, 0].set_xlabel(r'$\rho$', fontsize=25)
    #ax[1, 0].tick_params(axis='both', labelsize=20)
    ##ax[1,0].legend(fontsize=20)
    #cbar2 = fig.colorbar(sc2, ax=ax[1, 0])
    #cbar2.set_label(r'$A$', fontsize=25)

    # SUBPLOT 11: Hexbin
    hb11 = ax[1, 1].hexbin(
    ext_inf_rho_a[ext_inf_epr_a != 1],
    ext_infected_where_freq_a[ext_inf_epr_a != 1],
    C=ext_attract_inf_a[ext_inf_epr_a != 1],
    gridsize=50,
    cmap='viridis',
    reduce_C_function=np.mean
    )
    ax[1, 1].axvline(0.5, color='teal', linestyle='--')
    ax[1, 1].set_title("Invasions under preferential return", fontsize=25)
    ax[1, 1].set_ylabel(r'$f_{inf}$', fontsize=25)
    ax[1, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 1].tick_params(axis='both', labelsize=20)
    cbar11 = fig.colorbar(hb11, ax=ax[1, 1])
    cbar11.set_label(r'$A$', fontsize=25)

    # SUBPLOT 3: Hexbin
    #sc3 = ax[1, 1].scatter(
    #    ext_inf_rho_a[ext_inf_epr_a != 1],
    #    ext_infected_where_freq_a[ext_inf_epr_a != 1],
    #    c= ext_attract_inf_a[ext_inf_epr_a != 1],
    #    cmap='viridis'
    #    )
    #ax[1, 1].axvline(0.5, color='teal', linestyle='--')
    #ax[1, 1].set_title(r"infections under preferential return", fontsize=25)
    #ax[1, 1].set_ylabel(r'$f_{{inf}}$', fontsize=25)
    #ax[1, 1].set_xlabel(r'$\rho$', fontsize=25)
    #ax[1, 1].tick_params(axis='both', labelsize=20)
    ##ax[1,1].legend(fontsize=20)
    #cbar3 = fig.colorbar(sc3, ax=ax[1, 1])
    #cbar3.set_label(r'$A$', fontsize=25)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'epr_step_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D2_dynamics_with_lockdown():
    lower_path = 'config/'
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    header = 'edyna'
    merged_dict = {}
    merged_dict.update(epi_pars)
    merged_dict.update(grid_pars) 
    string_segments = ['lmUnm', 'lf0_']
    filenames = ut.collect_pickle_filenames_by_exclusion(fullpath, header, string_segments)

    nsims = 25
    t_max = 1200

    rho_interval = [0.0, 1.0]

    # Define empty lists for storing the curves
    mal_inc_st = []
    mal_pre_st = []
    ral_inc_st = []
    ral_pre_st = []

    for filename, i in zip(filenames, range(len(filenames))):
        print("Loop {0}, timestamp: {1}".format(i+1, filename))
        # Build fullname
        fullname = os.path.join(cwd_path, lower_path, filename)

        # Build agent dataframe
        agent_df = an.build_agent_data_frame(fullname, nsims_load=nsims)

        # Filter dataframe by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.025)

        # Compute population in rho intervals
        pop_s = an.number_of_agents_in_rho_interval(agent_df, rho_interval)
        pop_s = an.number_of_agents_in_rho_interval(agent_df, rho_interval)
        
        # Filter dataframe by health status
        agent_df = an.health_status_filter_agent_data_frame(agent_df, health_status='Removed')

        # Rebuild epidemic curves
        inc_st = an.rebuild_incidence_time_series_for_rho_group(agent_df, rho_interval, t_max)
        pre_st = an.rebuild_prevalence_time_series_for_rho_group(agent_df, rho_interval, t_max)
        
        # Normalize the curves by dividing by population size
        inc_st = inc_st / np.expand_dims(pop_s, axis=1)
        pre_st = pre_st / np.expand_dims(pop_s, axis=1)
    
        # Append the curves to the respective lists
        if 'lmMAt' in filename:
            mal_inc_st.append(inc_st)
            mal_pre_st.append(pre_st)
        elif 'lmRan' in filename:
            ral_inc_st.append(inc_st)
            ral_pre_st.append(pre_st)

    # Stack the curves
    mal_inc_st = np.vstack(mal_inc_st)
    mal_pre_st = np.vstack(mal_pre_st)
    ral_inc_st = np.vstack(ral_inc_st)
    ral_pre_st = np.vstack(ral_pre_st)

    # Average curves
    average_mal_inc_st = np.mean(mal_inc_st, axis=0)
    average_mal_pre_st = np.mean(mal_pre_st, axis=0)
    average_ral_inc_st = np.mean(ral_inc_st, axis=0)
    average_ral_pre_st = np.mean(ral_pre_st, axis=0)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # SUBPLOT 0
    ax[0].plot(average_mal_inc_st, color='tomato', label='most attr')
    ax[0].plot(average_ral_inc_st, color='goldenrod', label= 'random')
    ax[0].set_xlabel(r'$t$', fontsize=25)
    ax[0].set_ylabel(r'$i_{\rho}(t)$', fontsize=25)
    ax[0].legend()

    # SUBPLOT 1
    ax[1].plot(average_mal_pre_st, color='tomato', label='most attr')
    ax[1].plot(average_ral_pre_st, color='goldenrod', label='random')
    ax[1].set_xlabel(r'$t$', fontsize=25)
    ax[1].set_ylabel(r'$r_{\rho}(t)$', fontsize=25)
    ax[1].legend()

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    base_name = 'macro_lockdown_comp'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D3_visits_frequency():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    top_k = 4
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_rho_a = []
    ext_S_a = []
    ext_t_avg_k_a = []
    ext_attr_a = []
    ext_rho_inf_a = []
    ext_S_inf_a = []
    ext_t_avg_k_inf_a = []
    ext_attr_inf_a = []
    ext_t_inf_a = []

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        # Collect stay times for all agents
        all_res, inf_res = an.collect_stay_times(agent_df, trajectory_df, space_df, nlocs, t_max, top_k)
        
        rho_a, S_a, t_avg_k_a, attr_a = all_res
        rho_inf_a, S_inf_a, t_avg_k_inf_a, t_inf_a, attr_inf_a = inf_res

        ext_rho_a.extend(rho_a)
        ext_S_a.extend(S_a)
        ext_t_avg_k_a.extend(t_avg_k_a)
        ext_attr_a.extend(attr_a)
        ext_rho_inf_a.extend(rho_inf_a)
        ext_S_inf_a.extend(S_inf_a)
        ext_t_avg_k_inf_a.extend(t_avg_k_inf_a)
        ext_attr_inf_a.extend(attr_inf_a)
        ext_t_inf_a.extend(t_inf_a)

    ext_rho_a = np.array(ext_rho_a)
    ext_S_a = np.array(ext_S_a)
    ext_t_avg_k_a = np.array(ext_t_avg_k_a)
    ext_attr_a = np.array(ext_attr_a)
    ext_rho_inf_a = np.array(ext_rho_inf_a)
    ext_S_inf_a = np.array(ext_S_inf_a)
    ext_t_avg_k_inf_a = np.array(ext_t_avg_k_inf_a)
    ext_attr_inf_a = np.array(ext_attr_inf_a)
    ext_t_inf_a = np.array(ext_t_inf_a)

    print("Computations finished. Time to plot")

    # Create the figure and axes objects
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    # SUBPLOT 0: Hexbin
    sc0 = ax[0, 0].scatter(
        t_max / ext_S_a,
        t_max * ext_t_avg_k_a,
        c= ext_rho_a,
        cmap='coolwarm'
        )
    x = np.linspace(np.min(t_max / ext_S_a), np.max(t_max / ext_S_a), 1000)
    ax[0, 0].plot(x, x, color='black', linestyle='dashed')
    #ax[0, 0].axvline(0.5, color='teal', linestyle='--')
    ax[0, 0].set_title(r"stay times ($k$={0})".format(top_k), fontsize=25)
    ax[0, 0].set_xlabel(r'$\langle t_{{stay}}\rangle_{t_{{max}}}$', fontsize=25)
    ax[0, 0].set_ylabel(r'$\langle t_{{stay}}\rangle_{t_{{max}}}^k$', fontsize=25)
    ax[0, 0].tick_params(axis='both', labelsize=20)
    #ax[0,0].legend(fontsize=20)
    cbar0 = fig.colorbar(sc0, ax=ax[0, 0])
    cbar0.set_label(r'$\rho$', fontsize=25)

    # SUBPLOT 01: Hexbin
    sc01 = ax[0, 1].scatter(
        t_inf_a / ext_S_inf_a,
        t_inf_a * ext_t_avg_k_inf_a,
        c= ext_rho_inf_a,
        cmap='coolwarm'
        )
    x = np.linspace(np.min(t_inf_a / ext_S_inf_a), np.max(t_inf_a / ext_S_inf_a), 1000)
    ax[0, 1].plot(x, x, color='black', linestyle='dashed')
    #ax[0, 1].axvline(0.5, color='teal', linestyle='--')
    ax[0, 1].set_title(r"stay times while infected ($k$={0})".format(top_k), fontsize=25)
    ax[0, 1].set_xlabel(r'$\langle t_{{stay}}\rangle_{t_{{T_I}}}$', fontsize=25)
    ax[0, 1].set_ylabel(r'$\langle t_{{stay}}\rangle_{t_{{T_I}}}^k$', fontsize=25)
    ax[0, 1].tick_params(axis='both', labelsize=20)
    #ax[0,1].legend(fontsize=20)
    cbar01 = fig.colorbar(sc01, ax=ax[0, 1])
    cbar01.set_label(r'$\rho$', fontsize=25)

    # SUBPLOT 10: Hexbin
    #sc10 = ax[1, 0].scatter(
    #    ext_rho_a,
    #    ext_t_avg_k_a * ext_S_a,
    #    c=ext_rho_a,
    #    cmap='coolwarm'
    #    )
    hb10 = ax[1, 0].hexbin(
        ext_rho_a,
        ext_t_avg_k_a * ext_S_a,
        gridsize=30,
        cmap='Blues',
        )
    ax[1, 0].set_title(r"stay times ratio ($k$={0})".format(top_k), fontsize=25)
    ax[1, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 0].set_ylabel(r'$\langle t_{{stay}}\rangle_{t_{{max}}}^k/\langle t_{{stay}}\rangle_{t_{{max}}}$', fontsize=25)
    ax[1, 0].tick_params(axis='both', labelsize=20)
    #cbar10 = fig.colorbar(sc10, ax=ax[1, 0])
    cbar10 = fig.colorbar(hb10, ax=ax[1, 0])
    cbar10.set_label(r'count', fontsize=25)

    # SUBPLOT 11: Hexbin
    #sc11 = ax[1, 1].scatter(
    #    ext_rho_inf_a[ext_t_inf_a > 4],
    #    ext_t_avg_k_inf_a[ext_t_inf_a > 4] * ext_S_inf_a[ext_t_inf_a > 4],
    #    c=ext_rho_inf_a[ext_t_inf_a > 4],
    #    cmap='coolwarm'
    #    )
    hb11 = ax[1, 1].hexbin(
        ext_rho_inf_a[ext_t_inf_a > 4],
        ext_t_avg_k_inf_a[ext_t_inf_a > 4] * ext_S_inf_a[ext_t_inf_a > 4],
        gridsize=30,
        cmap='Blues',
        )
    #ax10,01].axvline(0.5, color='teal', linestyle='--')
    ax[1, 1].set_title(r"stay times ratio while infected ($k$={0})".format(top_k), fontsize=25)
    ax[1, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 1].set_ylabel(r'$\langle t_{{stay}}\rangle_{T_{{I}}}^k /\langle t_{{stay}}\rangle_{T_{{I}}}$', fontsize=25)
    ax[1, 1].tick_params(axis='both', labelsize=20)
    #cbar11 = fig.colorbar(sc11, ax=ax[1, 1])
    cbar11 = fig.colorbar(hb11, ax=ax[1, 1])
    cbar11.set_label(r'count', fontsize=25)
   
    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'visits_freq_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D3_newcomers():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_visits_dict = {}
    ext_rho_dict = {}
    ext_freq_dict = {}
    nsims_eff = 0
    nagents_s = 0

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        nsims_eff += an.number_of_simulations(agent_df)
        nagents_s += len(agent_df)

        # Compute stuff
        visit_dict, rho_dict, freq_dict = an.collect_recurrency(agent_df, trajectory_df, space_df, nlocs, t_max)

        # Extend or merge dicts for the rest of simulations
        for key, value in visit_dict.items():
            ext_visits_dict.setdefault(key, []).extend(value)

        for key, value in rho_dict.items():
            ext_rho_dict.setdefault(key, []).extend(value)

        for key, value in freq_dict.items():
            ext_freq_dict.setdefault(key, []).extend(value)
        
        #ext_visits_dict.update(visit_dict)
        #ext_rho_dict.update(rho_dict)
        #ext_freq_dict.update(freq_dict)

    # Prepare output structure
    x_cells = int(np.sqrt(nlocs))
    y_cells = int(np.sqrt(nlocs))    
    visits_avg_array = np.zeros((x_cells, y_cells))
    rho_avg_array = np.zeros((x_cells, y_cells))
    freq_avg_array = np.zeros((x_cells, y_cells))
    newcomer_array = np.zeros((x_cells, y_cells))

    # Loop over locations
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            if l in ext_visits_dict:
                visits_a = ext_visits_dict[l]
                avg_visits = sum(visits_a) / ((nlocs - 1) * len(visits_a))
                rho_a = ext_rho_dict[l]
                avg_rho = sum(rho_a) / len(rho_a)
                freq_a = ext_freq_dict[l]
                avg_freq = sum(freq_a) / len(freq_a)
                avg_newcomers = len(rho_a) / nagents_s
                
                visits_avg_array[j, i] = avg_visits
                rho_avg_array[j, i] = avg_rho
                freq_avg_array[j, i] = avg_freq
                newcomer_array[j, i] = avg_newcomers
            l += 1

    # Prepare figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    # SUBPLOT 00: IMSHOW. VISIT MAP
    im00 = ax[0, 0].imshow(visits_avg_array, cmap='viridis')
    cbar00 = fig.colorbar(im00, ax=ax[0, 0], shrink=0.8)
    cbar00.set_label(r'$\frac{1}{N_{{uv,l}}}\sum_{a}(S_{a}-1)/(V-1)$', fontsize=20)

    # Settings 00
    title = r"Locations' connectedness from visitor's $S$"
    ax[0, 0].set_title(title, fontsize=25)
    ax[0, 0].invert_yaxis()
    ax[0, 0].tick_params(axis='both', labelsize=18)

    # SUBPLOT 01: IMSHOW. VISIT MAP
    im01 = ax[0, 1].imshow(freq_avg_array, cmap='viridis')
    cbar01 = fig.colorbar(im01, ax=ax[0, 1], shrink=0.8)
    cbar01.set_label(r'$\frac{1}{N_{{uv,l}}}\sum_{a}f_l^a$', fontsize=20)

    # Settings 01
    title = r"Location's recurrency"
    ax[0, 1].set_title(title, fontsize=25)
    ax[0, 1].invert_yaxis()
    ax[0, 1].tick_params(axis='both', labelsize=18)

    # SUBPLOT 10: IMSHOW. VISIT MAP
    im10 = ax[1, 0].imshow(rho_avg_array, cmap='coolwarm')
    cbar10 = fig.colorbar(im10, ax=ax[1, 0], shrink=0.8)
    cbar10.set_label(r'$\langle\rho\rangle_{{uv}}$', fontsize=20)

    # Settings 10
    title = r"Unique visitor typical mobility profile"
    ax[1, 0].set_title(title, fontsize=25)
    ax[1, 0].invert_yaxis()
    ax[1, 0].tick_params(axis='both', labelsize=18)

    # SUBPLOT 11: IMSHOW. VISIT MAP
    im11 = ax[1, 1].imshow(newcomer_array, cmap='viridis')
    cbar10 = fig.colorbar(im11, ax=ax[1, 1], shrink=0.8)
    cbar10.set_label(r'$N_{{uv}}/N$', fontsize=20)

    # Settings 11
    title = r"Share of unique visitors"
    ax[1, 1].set_title(title, fontsize=25)
    ax[1, 1].invert_yaxis()
    ax[1, 1].tick_params(axis='both', labelsize=18)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'newcomers_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D3_infected_newcomers():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_visits_dict = {}
    ext_rho_dict = {}
    ext_freq_dict = {}
    nsims_eff = 0
    nagents_s = 0

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        nsims_eff += an.number_of_simulations(agent_df)
        nagents_s += len(agent_df)

        # Compute stuff
        visit_dict, rho_dict, freq_dict = an.collect_infected_recurrency(agent_df, trajectory_df, t_max)

        # Extend or merge dicts for the rest of simulations
        for key, value in visit_dict.items():
            ext_visits_dict.setdefault(key, []).extend(value)

        for key, value in rho_dict.items():
            ext_rho_dict.setdefault(key, []).extend(value)

        for key, value in freq_dict.items():
            ext_freq_dict.setdefault(key, []).extend(value)
        
        #ext_visits_dict.update(visit_dict)
        #ext_rho_dict.update(rho_dict)
        #ext_freq_dict.update(freq_dict)

    # Prepare output structure
    x_cells = int(np.sqrt(nlocs))
    y_cells = int(np.sqrt(nlocs))    
    visits_avg_array = np.zeros((x_cells, y_cells))
    rho_avg_array = np.zeros((x_cells, y_cells))
    freq_avg_array = np.zeros((x_cells, y_cells))
    newcomer_array = np.zeros((x_cells, y_cells))

    # Loop over locations
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            if l in ext_visits_dict:
                visits_a = ext_visits_dict[l]
                avg_visits = sum(visits_a) / ((nlocs - 1) * len(visits_a))
                rho_a = ext_rho_dict[l]
                avg_rho = sum(rho_a) / len(rho_a)
                freq_a = ext_freq_dict[l]
                avg_freq = sum(freq_a) / len(freq_a)
                avg_newcomers = len(rho_a) / nagents_s
                
                visits_avg_array[j, i] = avg_visits
                rho_avg_array[j, i] = avg_rho
                freq_avg_array[j, i] = avg_freq
                newcomer_array[j, i] = avg_newcomers
            l += 1

    # Prepare figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    # SUBPLOT 00: IMSHOW. VISIT MAP
    im00 = ax[0, 0].imshow(visits_avg_array, cmap='viridis')
    cbar00 = fig.colorbar(im00, ax=ax[0, 0], shrink=0.8)
    cbar00.set_label(r'$\frac{1}{N_{{uv,l}}}\sum_{a}(S_{a}-1)/(V-1)$', fontsize=20)

    # Settings 00
    title = r"Locations' connectedness from visitor's $S$"
    ax[0, 0].set_title(title, fontsize=25)
    ax[0, 0].invert_yaxis()
    ax[0, 0].tick_params(axis='both', labelsize=18)

    # SUBPLOT 01: IMSHOW. VISIT MAP
    im01 = ax[0, 1].imshow(freq_avg_array, cmap='viridis')
    cbar01 = fig.colorbar(im01, ax=ax[0, 1], shrink=0.8)
    cbar01.set_label(r'$\frac{1}{N_{{uv,l}}}\sum_{a}f_l^a$', fontsize=20)

    # Settings 01
    title = r"Location's recurrency"
    ax[0, 1].set_title(title, fontsize=25)
    ax[0, 1].invert_yaxis()
    ax[0, 1].tick_params(axis='both', labelsize=18)

    # SUBPLOT 10: IMSHOW. VISIT MAP
    im10 = ax[1, 0].imshow(rho_avg_array, cmap='coolwarm')
    cbar10 = fig.colorbar(im10, ax=ax[1, 0], shrink=0.8)
    cbar10.set_label(r'$\langle\rho\rangle_{{uv}}$', fontsize=20)

    # Settings 10
    title = r"Unique visitor typical mobility profile"
    ax[1, 0].set_title(title, fontsize=25)
    ax[1, 0].invert_yaxis()
    ax[1, 0].tick_params(axis='both', labelsize=18)

    # SUBPLOT 11: IMSHOW. VISIT MAP
    im11 = ax[1, 1].imshow(newcomer_array, cmap='viridis')
    cbar10 = fig.colorbar(im11, ax=ax[1, 1], shrink=0.8)
    cbar10.set_label(r'$N_{{uv}}/N$', fontsize=20)

    # Settings 11
    title = r"Share of unique visitors"
    ax[1, 1].set_title(title, fontsize=25)
    ax[1, 1].invert_yaxis()
    ax[1, 1].tick_params(axis='both', labelsize=18)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'INF_newcomers_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D3_connectivity():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_weight_dict = {}
    ext_rho_dict = {}
    ext_inf_weight_dict = {}
    ext_inf_rho_dict = {}
    nsims_eff = 0
    nagents_sims = 0

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        nsims_eff += an.number_of_simulations(agent_df)
        nagents_sims += len(agent_df)

        # Compute stuff
        weight_dict, rho_dict, inf_weight_dict, inf_rho_dict = an.collect_connectivity(agent_df, trajectory_df)
        # Extend or merge dicts for the rest of simulations
        for key, value in weight_dict.items():
            if key in ext_weight_dict:
                ext_weight_dict[key] += value
                ext_rho_dict[key].extend(rho_dict[key])
            else:
                ext_weight_dict[key] = value
                ext_rho_dict[key] = rho_dict[key]
        
        for key, value in inf_weight_dict.items():
            if key in ext_inf_weight_dict:
                ext_inf_weight_dict[key] += value
                ext_inf_rho_dict[key].extend(inf_rho_dict[key])
            else:
                ext_inf_weight_dict[key] = value
                ext_inf_rho_dict[key] = inf_rho_dict[key]

    keys = list(ext_weight_dict.keys())
    max_i = max(keys, key=lambda x: x[0])[0]
    max_j = max(keys, key=lambda x: x[1])[1]
    shape = (max_i + 1, max_j + 1)

    # Create empty NumPy array
    ext_weight_array = np.zeros(shape, dtype=int)
    ext_rho_array = np.zeros(shape, dtype=float)
    ext_inf_weight_array = np.zeros(shape, dtype=int)
    ext_inf_rho_array = np.zeros(shape, dtype=float)

    # Assign values from dictionary to array
    for key, value in ext_weight_dict.items():
        ext_weight_array[key] = value
        ext_rho_array[key] = np.mean(ext_rho_dict[key])
    for key, value in ext_inf_weight_dict.items():
        ext_inf_weight_array[key] = value
        ext_inf_rho_array[key] = np.mean(ext_inf_rho_dict[key])

    # Get the dimensions of the array
    n, m = ext_weight_array.shape
    # Upper diagonal
    upper_indices = np.triu_indices(n, k=1)  # Get indices of upper triangle excluding the main diagonal
    upper_weight = ext_weight_array[upper_indices]
    upper_rho = ext_rho_array[upper_indices]
    # Lower diagonal
    lower_indices = np.tril_indices(n, k=-1)  # Get indices of lower triangle excluding the main diagonal
    lower_weight = ext_weight_array[lower_indices]
    lower_rho = ext_rho_array[lower_indices]

    # Get the dimensions of the array
    n, m = ext_inf_weight_array.shape
    # Upper diagonal
    upper_indices = np.triu_indices(n, k=1)  # Get indices of upper triangle excluding the main diagonal
    upper_inf_weight = ext_inf_weight_array[upper_indices]
    upper_inf_rho = ext_inf_rho_array[upper_indices]
    # Lower diagonal
    lower_indices = np.tril_indices(n, k=-1)  # Get indices of lower triangle excluding the main diagonal
    lower_inf_weight = ext_inf_weight_array[lower_indices]
    lower_inf_rho = ext_inf_rho_array[lower_indices]

    # Calculate the average for each key in ext_rho_dict and ext_inf_rho_dict
    rho_means = []
    inf_rho_means = []
    for key in ext_rho_dict:
        rho_means.append(np.mean(ext_rho_dict[key]))
    for key in ext_inf_rho_dict:
        inf_rho_means.append(np.mean(ext_inf_rho_dict[key]))
    rho_array = np.array(rho_means)
    inf_rho_array = np.array(inf_rho_means)

    # Prepare figure
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))

    # SUBPLOT 00: WEIGHT DISTRIBUTION
    #hb0 = ax[0, 0].hexbin(
    #    rho_ij[(rho_ij != 0) & (rho_ji != 0)],
    #    rho_ji[(rho_ij != 0) & (rho_ji != 0)],
    #    gridsize=30,
    #    cmap='Reds',
    #)
    ax[0, 0].scatter(upper_rho, lower_rho)
    ax[0, 0].axvline(0.5, color='teal', linestyle='--')
    ax[0, 0].set_title(r"$\langle\rho\rangle$ scatter for full trajectories", fontsize=25)
    ax[0, 0].set_ylabel(r'$\langle\rho\rangle_{ji}$', fontsize=25)
    ax[0, 0].set_xlabel(r'$\langle\rho\rangle_{ij}$', fontsize=25)
    ax[0, 0].tick_params(axis='both', labelsize=20)
    #fig.colorbar(hb0, ax=ax[0, 0])

    # SUBPLOT 01: RHO DISTRIBUTION
    bins = 30
    density = False
    ax[0, 1].hist(rho_array, bins=bins, density=density, color='lightskyblue')

    # Settings 01
    title = r"$\langle\rho\rangle$ distribution"
    ax[0, 1].set_title(title, fontsize=25)
    ax[0, 1].tick_params(axis='both', labelsize=18)

    # SUBPLOT 10: WEIGHT DISTRIBUTION
    ax[1, 0].scatter(upper_inf_rho, lower_inf_rho)
    #hb1 = ax[1, 0].hexbin(
    #    inf_rho_ij[(inf_rho_ij != 0) & (inf_rho_ji != 0)],
    #    inf_rho_ji[(inf_rho_ij != 0) & (inf_rho_ji != 0)],
    #    gridsize=30,
    #    cmap='Reds',
    #)
    ax[1, 0].axvline(0.5, color='teal', linestyle='--')
    ax[1, 0].set_title(r"$\langle\rho\rangle$ scatter while infected", fontsize=25)
    ax[1, 0].set_ylabel(r'$\langle\rho\rangle_{ji}$', fontsize=25)
    ax[1, 0].set_xlabel(r'$\langle\rho\rangle_{ij}$', fontsize=25)
    ax[1, 0].tick_params(axis='both', labelsize=20)
    #ax[0,0].legend(fontsize=20)
    #fig.colorbar(hb1, ax=ax[1, 0])

    # SUBPLOT 01: RHO DISTRIBUTION
    bins = 30
    density = False
    ax[1, 1].hist(inf_rho_array, bins=bins, density=density)

    # Settings 00
    title = r"infected $\langle\rho\rangle$ distribution"
    ax[1, 1].set_title(title, fontsize=25)
    ax[1, 1].tick_params(axis='both', labelsize=18)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'connect_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D3_location_times():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200
    rho_interval = [0.0, 1.0]
    attr_cutoff = 0.00000001

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    arrival_times_dict = {}
    peak_times_dict = {}
    nsims_eff = 0
    nagents_sims = 0

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        nsims_eff += an.number_of_simulations(agent_df)
        nagents_sims += len(agent_df)

        # Filter by health status
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        # Compute stuff
        inc_slt = an.rebuild_local_incidence_time_series_for_rho_group(agent_df, rho_interval, trajectory_df, nlocs, t_max)

        # Iterate over simulation, location, and time dimensions
        for s in range(nsims):
            for l in range(nlocs):
                attr = space_df.loc[space_df['id'] == l, 'attractiveness'].values[0]
                if attr >= attr_cutoff:
                    for t in range(t_max):
                        if inc_slt[s, l, t] > 0:
                            # Find the arrival time (first non-zero time)
                            #arrival_times_sl[s, l] = t
                            if l in arrival_times_dict:
                                arrival_times_dict[l].append(t)
                            else:
                                arrival_times_dict[l] = [t]
                            break
                    
                    # Find the peak time (time with the maximum infected population)
                    #peak_times_sl[s, l] = np.argmax(inc_slt[s, l])
                    if l in peak_times_dict:
                        peak_times_dict[l].append(np.argmax(inc_slt[s, l]))
                    else:
                        peak_times_dict[l] = [np.argmax(inc_slt[s, l])]

    # Get attractiveness list
    attr_l = np.zeros(nlocs)
    arri_l = np.zeros(nlocs)
    peti_l = np.zeros(nlocs)

    # Initialize average arrays
    AT = np.zeros((x_cells, y_cells))
    PT = np.zeros((x_cells, y_cells))
    # Loop over locations
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            attr = space_df.loc[space_df['id'] == l, 'attractiveness'].values[0]
            attr_l[l] = attr

            if l in arrival_times_dict:
                mat = np.mean(arrival_times_dict[l])
                AT[y_cells - 1 - j, i] = mat
                arri_l[l] = mat
            else:
                AT[y_cells - 1 - j, i] = np.nan
                arri_l[l] = np.nan
            if l in peak_times_dict:
                mpt = np.mean(peak_times_dict[l])
                PT[y_cells - 1 - j, i] = mpt
                peti_l[l] = mpt
            else:
                PT[y_cells - 1 - j, i] = np.nan
                peti_l[l] = np.nan
            l += 1

    # Prepare figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    # SUBPLOT 00: IMSHOW. ARRIVAL TIMES
    im0 = ax[0, 0].imshow(AT.T, cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=ax[0, 0], shrink=0.7)
    cbar0.set_label(r'time', fontsize=25)

    # Settings 00
    title = r"arrival times"
    ax[0, 0].set_title(title, fontsize=25)
    ax[0, 0].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    ax[0, 0].set_ylabel("latitude (\u00b0 N)", fontsize=25)
    ax[0, 0].invert_yaxis()
    ax[0, 0].tick_params(axis='both', labelsize=18)

    new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
    new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
    x_ticks_pos = range(0, 51, 10)
    y_ticks_pos = range(0, 51, 10)
    ax[0, 0].set_xticks(x_ticks_pos)
    ax[0, 0].set_yticks(y_ticks_pos)
    ax[0, 0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    ax[0, 0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
    
    # SUBPLOT 01: IMSHOW. PEAK TIMES
    im1 = ax[0, 1].imshow(PT.T, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=ax[0, 1], shrink=0.7)
    cbar1.set_label(r'time', fontsize=25)

    # Settings 01
    title = r"peak times"
    ax[0, 1].set_title(title, fontsize=25)
    ax[0, 1].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    ax[0, 1].invert_yaxis()
    ax[0, 1].tick_params(axis='both', labelsize=18)
    
    ax[0, 1].set_xticks(x_ticks_pos)
    ax[0, 1].set_yticks(y_ticks_pos)
    ax[0, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    ax[0, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

    # Filter zero attractiveess
    arri_l = arri_l[attr_l > attr_cutoff]
    peti_l = peti_l[attr_l > attr_cutoff]
    attr_l = attr_l[attr_l > attr_cutoff]

    # Filter nan
    attr_l_nonnan = attr_l[~np.isnan(arri_l)]
    arri_l_nonnan = arri_l[~np.isnan(arri_l)]

    # SUBPLOT 10: SCATTER. 
    ax[1, 0].scatter(arri_l, np.log10(attr_l), color='teal')

    # Fit linear regression and plot regression line
    model_10 = LinearRegression()
    model_10.fit(arri_l_nonnan.reshape(-1, 1), np.log10(attr_l_nonnan))
    y_pred_10 = model_10.predict(arri_l_nonnan.reshape(-1, 1))
    ax[1, 0].plot(arri_l_nonnan, y_pred_10, color='crimson', linestyle='--', linewidth=2)
    
    # Calculate and display R-squared value
    # Get R-squared value and display it
    r2_10 = model_10.score(arri_l_nonnan.reshape(-1, 1), np.log10(attr_l_nonnan))
    ax[1, 0].text(0.75, 0.75, r'$R^2$={0}'.format(np.round(r2_10, 2)), transform=ax[1, 0].transAxes, fontsize=25, color='black')

    # Settings 10
    title = r"arrival times vs. attractiveness"
    ax[1, 0].set_title(title, fontsize=25)
    ax[1, 0].set_xlabel(r"$t$", fontsize=25)
    ax[1, 0].set_ylabel(r"$\log A$", fontsize=25)
    ax[1, 0].tick_params(axis='both', labelsize=18)
    
    # SUBPLOT 11: SCATTER.
    ax[1, 1].scatter(peti_l, np.log10(attr_l), color='teal')

    # Fit linear regression and plot regression line
    attr_l_nonnan = attr_l[~np.isnan(peti_l)]
    peti_l_nonnan = peti_l[~np.isnan(peti_l)]

    model_11 = LinearRegression()
    model_11.fit(peti_l_nonnan.reshape(-1, 1), np.log10(attr_l_nonnan))
    y_pred_11 = model_11.predict(peti_l_nonnan.reshape(-1, 1))
    ax[1, 1].plot(peti_l_nonnan, y_pred_11, color='crimson', linestyle='--', linewidth=2)
    
    # Calculate and display R-squared value
    # Get R-squared value and display it
    r2_11 = model_11.score(peti_l_nonnan.reshape(-1, 1), np.log10(attr_l_nonnan))
    ax[1, 1].text(0.75, 0.75, r'$R^2$={0}'.format(np.round(r2_11, 2)), transform=ax[1, 1].transAxes, fontsize=25, color='black')
    
    # Settings 10
    title = r"peak times vs. attractiveness"
    ax[1, 1].set_title(title, fontsize=25)
    ax[1, 1].set_xlabel(r"$t$", fontsize=25)
    ax[1, 1].set_ylabel(r"$\log A$", fontsize=25)
    ax[1, 1].tick_params(axis='both', labelsize=18)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'loc_times_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D3_case_flow():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define structures
    ext_case_imports = {}
    ext_rho_imports = {}
    ext_case_exports = {}
    ext_rho_exports = {}
    ext_case_sources = {}
    ext_rho_sources = {}
    ext_case_sinks = {}
    ext_rho_sinks = {}
    nsims_eff = 0
    nagents_sims = 0

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
    
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)

        nsims_eff += an.number_of_simulations(agent_df)
        nagents_sims += len(agent_df)

        # Filter dataframe by health status
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        # Compute stuff
        imports, exports, sources, sinks = an.collect_case_flow(agent_df, trajectory_df)

        # Extend or merge dicts for the rest of simulations
        case_imports, rho_imports = imports 
        for key, value in case_imports.items():
            if key in ext_case_imports:
                ext_case_imports[key] += value
                ext_rho_imports[key].extend(rho_imports[key])
            else:
                ext_case_imports[key] = value
                ext_rho_imports[key] = rho_imports[key]
        
        case_exports, rho_exports = exports
        for key, value in case_exports.items():
            if key in ext_case_exports:
                ext_case_exports[key] += value
                ext_rho_exports[key].extend(rho_exports[key])
            else:
                ext_case_exports[key] = value
                ext_rho_exports[key] = rho_exports[key]
        
        case_sources, rho_sources = sources
        for key, value in case_sources.items():
            if key in ext_case_sources:
                ext_case_sources[key] += value
                ext_rho_sources[key].extend(rho_sources[key])
            else:
                ext_case_sources[key] = value
                ext_rho_sources[key] = rho_sources[key]
        
        case_sinks, rho_sinks = sinks
        for key, value in case_sinks.items():
            if key in ext_case_sinks:
                ext_case_sinks[key] += value
                ext_rho_sinks[key].extend(rho_sinks[key])
            else:
                ext_case_sinks[key] = value
                ext_rho_sinks[key] = rho_sinks[key]

    # Prepare output structure
    x_cells = int(np.sqrt(nlocs))
    y_cells = int(np.sqrt(nlocs))    
    case_imports = np.zeros((x_cells, y_cells))
    rho_imports = np.zeros((x_cells, y_cells))
    case_exports = np.zeros((x_cells, y_cells))
    rho_exports = np.zeros((x_cells, y_cells))
    case_sources = np.zeros((x_cells, y_cells))
    rho_sources = np.zeros((x_cells, y_cells))
    case_sinks = np.zeros((x_cells, y_cells))
    rho_sinks = np.zeros((x_cells, y_cells))

    # Loop over locations
    l = 0
    for i in range(x_cells):
        for j in range(y_cells):
            if l in ext_case_imports:
                case_imports[j, i] = ext_case_imports[l]
                rho_imports[j, i] = np.mean(ext_rho_imports[l])
            if l in ext_case_exports:
                case_exports[j, i] = ext_case_exports[l]
                rho_exports[j, i] = np.mean(ext_rho_exports[l])
            if l in ext_case_sources:
                case_sources[j, i] = ext_case_sources[l]
                rho_sources[j, i] = np.mean(ext_rho_sources[l])
            if l in ext_case_sinks:
                case_sinks[j, i] = ext_case_sinks[l]
                rho_sinks[j, i] = np.mean(ext_rho_sinks[l])

            l += 1

    # Prepare figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    # SUBPLOT 00: IMSHOW. VISIT MAP
    im00 = ax[0, 0].imshow(case_exports - case_imports, cmap='viridis')
    cbar00 = fig.colorbar(im00, ax=ax[0, 0], shrink=0.8)
    cbar00.set_label(r'infected case net flow', fontsize=20)

    # Settings 00
    title = r"Infected case net flow (exports - imports)"
    ax[0, 0].set_title(title, fontsize=25)
    ax[0, 0].invert_yaxis()
    ax[0, 0].tick_params(axis='both', labelsize=18)

    # SUBPLOT 01: IMSHOW. VISIT MAP
    im01 = ax[0, 1].imshow(case_sources, cmap='viridis')
    cbar01 = fig.colorbar(im01, ax=ax[0, 1], shrink=0.8)
    cbar01.set_label(r'infected case produced', fontsize=20)

    # Settings 01
    title = r"Infected case production"
    ax[0, 1].set_title(title, fontsize=25)
    ax[0, 1].invert_yaxis()
    ax[0, 1].tick_params(axis='both', labelsize=18)

    # SUBPLOT 10: IMSHOW. VISIT MAP
    im10 = ax[1, 0].imshow(rho_exports - rho_imports, cmap='coolwarm')
    cbar10 = fig.colorbar(im10, ax=ax[1, 0], shrink=0.8)
    cbar10.set_label(r'$\langle\rho\rangle$', fontsize=20)

    # Settings 10
    title = r"$\langle\rho\rangle$ from net flow"
    ax[1, 0].set_title(title, fontsize=25)
    ax[1, 0].invert_yaxis()
    ax[1, 0].tick_params(axis='both', labelsize=18)

    # SUBPLOT 11: IMSHOW. VISIT MAP
    im11 = ax[1, 1].imshow(rho_sources, cmap='coolwarm')
    cbar11 = fig.colorbar(im11, ax=ax[1, 1], shrink=0.8)
    cbar11.set_label(r'$\langle\rho\rangle$', fontsize=20)

    # Settings 11
    title = r"$\langle\rho\rangle$ from production"
    ax[1, 1].set_title(title, fontsize=25)
    ax[1, 1].invert_yaxis()
    ax[1, 1].tick_params(axis='both', labelsize=18)

    # SUBPLOT 02: IMSHOW. VISIT MAP
    #im02 = ax[0, 2].imshow(case_sources, cmap='viridis')
    #cbar02 = fig.colorbar(im02, ax=ax[0, 2], shrink=0.8)
    #cbar02.set_label(r'infected cases', fontsize=20)

    ## Settings 12
    #title = r"Infected cases production"
    #ax[0, 2].set_title(title, fontsize=25)
    #ax[0, 2].invert_yaxis()
    #ax[0, 2].tick_params(axis='both', labelsize=18)

    # SUBPLOT 12: IMSHOW. VISIT MAP
    #im12 = ax[1, 2].imshow(rho_sources, cmap='coolwarm')
    #cbar12 = fig.colorbar(im12, ax=ax[1, 2], shrink=0.8)
    #cbar12.set_label(r'$\langle\rho\rangle$', fontsize=20)

    ## Settings 12
    #title = r"$\langle\rho\rangle$ from production"
    #ax[1, 2].set_title(title, fontsize=25)
    #ax[1, 2].invert_yaxis()
    #ax[1, 2].tick_params(axis='both', labelsize=18)
   
    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'case_flow_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D4_contact():
    lower_path = 'config/'
    # Load grid parameters from grid retriever json file
    grbl_filename = 'config_grid_bl_retriever'
    grbl_fullname = os.path.join(cwd_path, lower_path, grbl_filename)
    grid_pars = ut.read_json_file(grbl_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/boston_4_na50000_x50'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)[0:1]

    nsims = 1
    t_max = 1200
    low_cutoff = 0.05
    mid_cutoff_l = 0.45
    mid_cutoff_h = 0.55
    hig_cutoff = 0.95

    # Define structures
    ext_hig_k_avg_st = []
    ext_mid_k_avg_st = []
    ext_low_k_avg_st = []

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grbl_fullname)
        grid_pars['tm'] = timestamp
        grid_filename = ut.build_grid_results_filename(grid_pars)
        grid_filename += '.pickle'
        grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
        grid = ut.open_file(grid_fullname)['inner']
        
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build epidemic filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build epidemic fullname
        lower_path = 'data/boston_4_na50000_x50'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Build dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)

        # Collect trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=1)
        #agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        # Compute degree
        hig_agent_df = agent_df[agent_df['mobility'] >= hig_cutoff]
        hig_k_avg_st = an.compute_average_degree_in_time(hig_agent_df, trajectory_df, grid, t_max)
        mid_agent_df = agent_df[(mid_cutoff_l <= agent_df['mobility']) & (agent_df['mobility'] <= mid_cutoff_h)]
        mid_k_avg_st = an.compute_average_degree_in_time(mid_agent_df, trajectory_df, grid, t_max)
        low_agent_df = agent_df[agent_df['mobility'] <= low_cutoff]
        low_k_avg_st = an.compute_average_degree_in_time(low_agent_df, trajectory_df, grid, t_max)

        # Append the curves to the respective lists
        ext_low_k_avg_st.append(low_k_avg_st)
        ext_mid_k_avg_st.append(mid_k_avg_st)
        ext_hig_k_avg_st.append(hig_k_avg_st)

    # Stack the curves
    ext_low_k_avg_st = np.vstack(ext_low_k_avg_st)
    ext_mid_k_avg_st = np.vstack(ext_mid_k_avg_st)
    ext_hig_k_avg_st = np.vstack(ext_hig_k_avg_st)

    # Average curves
    average_low_k_avg_t = np.mean(ext_low_k_avg_st, axis=0)
    average_mid_k_avg_t = np.mean(ext_mid_k_avg_st, axis=0)
    average_hig_k_avg_t = np.mean(ext_hig_k_avg_st, axis=0)
 
    # Prepare figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # SUBPLOT 0
    ax.plot(average_hig_k_avg_t, color='firebrick', label='top exp')
    ax.plot(average_mid_k_avg_t, color='slateblue', label=r'mid $\rho$')
    ax.plot(average_low_k_avg_t, color='dodgerblue', label='top ret')

    # Settings 0
    title = r"instantaneous average degree by mobility profile"
    ax.set_title(title, fontsize=25)
    ax.set_xlabel(r"$t$", fontsize=25)
    ax.set_ylabel(r"$\langle k(t)\rangle$", fontsize=25)
    ax.tick_params(axis='both', labelsize=22)
    ax.legend(fontsize=20)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'contact_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D5_visits_until_infection():
    lower_path = 'config/'
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)
    #timestamps = timestamps[:2]

    nsims = 25

    # Define empty lists for storing the curves
    extended_results = []
    visits_until_a = []
    infected_where_a = []
    home_a = []
    rho_a = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)
    
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        event_df = an.simulations_filter_event_data_frame(event_df, nsims_load=nsims)

        print("Counting")
        results = an.count_visits_until_infection_and_where(agent_df, trajectory_df)

        extended_results.append(results)

    for results in extended_results:
        visits_until_a.extend(results[0])
        infected_where_a.extend(results[1])
        home_a.extend(results[2])
        rho_a.extend(results[3])

    rho_a = np.array(rho_a)
    home_a = np.array(home_a)
    infected_where_a = np.array(infected_where_a)
    visits_until_a = np.array(visits_until_a)

    same_location_indices = np.where(infected_where_a == home_a)
    same_explorers_indices = np.where((infected_where_a == home_a) & (rho_a > 0.5))
    same_returners_indices = np.where((infected_where_a == home_a) & (rho_a < 0.5))
    same_count = len(same_location_indices[0])
    same_exp_dens = np.round(100.0 * len(same_explorers_indices[0]) / same_count, 2)
    same_ret_dens = np.round(100.0 * len(same_returners_indices[0]) / same_count, 2)
    diff_location_indices = np.where(infected_where_a != home_a)
    diff_explorers_indices = np.where((infected_where_a != home_a) & (rho_a > 0.5))
    diff_returners_indices = np.where((infected_where_a != home_a) & (rho_a < 0.5))
    diff_count = len(diff_location_indices[0])
    diff_exp_dens = np.round(100.0 * len(diff_explorers_indices[0]) / diff_count, 2)
    diff_ret_dens = np.round(100.0 * len(diff_returners_indices[0]) / diff_count, 2)
    same_dens = same_count / (same_count + diff_count)
    diff_dens = diff_count / (same_count + diff_count)
    same_dens = np.round(100.0 * same_dens, 2)
    diff_dens = np.round(100.0 * diff_dens, 2)

    print("Agents infected at home: {0}".format(same_dens))
    print("Explorers infected at home: {0}".format(same_exp_dens))
    print("Returners infected at home: {0}".format(same_ret_dens))
    print("Agents infected outside: {0}".format(diff_dens))
    print("Explorers infected outside: {0}".format(diff_exp_dens))
    print("Returners infected outside: {0}".format(diff_ret_dens))

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    hb1 = ax[0].hexbin(
        rho_a[same_location_indices],
        visits_until_a[same_location_indices],
        gridsize=(30, 30),
        cmap='Blues',
        alpha=1.0,
        label='at home: {0}%'.format(same_dens),
    )

    hb2 = ax[1].hexbin(
        rho_a[diff_location_indices],
        visits_until_a[diff_location_indices],
        gridsize=(30, 30),
        cmap='Reds',
        alpha=1.0,
        label='outside: {0}%'.format(diff_dens),
    )

    # Add colorbars for the hexbins
    cbar1 = plt.colorbar(hb1, ax=ax[0])
    cbar1.set_label('Count', fontsize=15)

    cbar2 = plt.colorbar(hb2, ax=ax[1])
    cbar2.set_label('Count', fontsize=15)
    
    #ax.scatter(
    #    rho_a[same_location_indices], 
    #    visits_until_a[same_location_indices], 
    #    c='dodgerblue', 
    #    alpha=0.5,
    #    label='at home: {0}%'.format(same_dens),
    #    )
    #ax.scatter(
    #    rho_a[diff_location_indices], 
    #    visits_until_a[diff_location_indices], 
    #    c='firebrick', 
    #    alpha=0.5,
    #    label='outside: {0}%'.format(diff_dens)
    #    )
    
    ax[0].text(
        0.95, 
        0.55, 
        f'home exp: {same_exp_dens}%', 
        ha='right', 
        va='top', 
        color='black',
        transform=ax[0].transAxes, 
        fontsize=25,
        zorder=10,
        )
    ax[0].text(
        0.45, 
        0.55, 
        f'home ret: {same_ret_dens}%', 
        ha='right', 
        va='top', 
        color='black',
        transform=ax[0].transAxes, 
        fontsize=25,
        zorder=10,
        )
    ax[1].text(
        0.95, 
        0.65, 
        f'out exp: {diff_exp_dens}%', 
        ha='right', 
        va='top',
        color='black',
        transform=ax[1].transAxes, 
        fontsize=25,
        zorder=10,
        )
    ax[1].text(
        0.45, 
        0.65,
        f'out ret: {diff_ret_dens}%', 
        ha='right', 
        va='top',
        color='black',
        transform=ax[1].transAxes, 
        fontsize=25,
        zorder=10,
        )

    ax[0].axvline(0.5, color='teal', linestyle='--')
    ax[0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0].set_ylabel(r'visits until infection', fontsize=25)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(fontsize=20)

    ax[1].axvline(0.5, color='teal', linestyle='--')
    ax[1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1].set_ylabel(r'visits until infection', fontsize=25)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'visits_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D5_offsprings_by_event():
    lower_path = 'config/'
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)[0:1]

    nsims = 5
    nlocs = 2500
    t_max = 1200

    # Define empty lists for storing the curves
    extended_results = []
    infected_rho_a = []
    t_inf_a = []
    cum_size_a = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        print("Counting")
        results = an.collect_offsprings_by_event(event_df)
    
        extended_results.extend(results)

    # Compute results for full dynamics
    infector_rho_a = []
    offspring_a = []
    last_time_a = []
    for results in extended_results:
        infector_rho_a.append(results[0])
        offspring_a.append(np.sum(np.array([size for size in results[1][0]])))
        last_time_a.append(results[1][1][-1])
    infector_rho_a = np.array(infector_rho_a)
    offspring_a = np.array(offspring_a)
    last_time_a = np.array(last_time_a)

    # Calculate bin edges for rho values
    num_bins = 30
    cutoff_time = 1200
    infector_rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    # Compute the average value of offspring for each rho bin
    offspring_avg = []
    lower_95CI_offspring = []
    upper_95CI_offspring = []
    for i in range(num_bins):
        mask = (infector_rho_a >= infector_rho_bins[i]) & (infector_rho_a < infector_rho_bins[i + 1])
        mask &= (last_time_a <= cutoff_time)
        if np.any(mask):
            avg_offspring = np.mean(offspring_a[mask])
            std_dev_offspring = np.std(offspring_a[mask], ddof=1)  # ddof=1 for sample standard deviation
            num_samples = np.sum(mask)
            t_value = stats.t.ppf(0.975, df=num_samples-1)  # t-value for 95% CI
            standard_error = std_dev_offspring / np.sqrt(num_samples)
            lower_95CI = avg_offspring - t_value * standard_error
            upper_95CI = avg_offspring + t_value * standard_error
        else:
            avg_offspring = 0
            lower_95CI = 0
            upper_95CI = 0
        offspring_avg.append(avg_offspring)
        lower_95CI_offspring.append(lower_95CI)
        upper_95CI_offspring.append(upper_95CI)

    # Prepare figure
    fig, ax = plt.subplots()

    # Plot the results
    ax.plot(infector_rho_bins[:-1], offspring_avg, marker='o', color='teal', label='Average Offspring')
    ax.fill_between(infector_rho_bins[:-1], lower_95CI_offspring, upper_95CI_offspring, color='teal', alpha=0.2, label='95% CI')
    ax.set_xlabel(r'$\rho$', fontsize=25)
    ax.set_ylabel('Number of offsprings', fontsize=25)
    ax.set_title('New cases generated', fontsize=30)
    ax.tick_params(axis='both', labelsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'offsprings_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202307D5_cases_in_location():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    grid_pars = ut.read_json_file(grid_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    x_cells = int(np.sqrt(2500))
    y_cells = int(np.sqrt(2500))
    t_max = 1200
    rho_interval = [0.0, 1.0]
    attr_cutoff = 0.00000001

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    # Define empty lists for storing the curves
    extended_results = []
    infected_rho_a = []
    t_inf_a = []
    cum_size_a = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        print("Counting")
        results = an.collect_cases_by_location(event_df)
    
        extended_results.extend(results)

    attr_l = [space_df.loc[space_df['id'] == location, 'attractiveness'].values[0] for location, total_size, _ in results]
    total_sizes = [total_size for _, total_size, _ in results]
    mean_rho_values = [mean_rho for _, _, mean_rho in results]
    attr_l = np.array(attr_l)
    total_sizes = np.array(total_sizes)

    # Prepare figure
    fig, ax = plt.subplots()

    # Plot the results
    #sc = ax.scatter(attr_l, total_sizes, c=mean_rho_values, cmap='coolwarm')
    #cbar0 = fig.colorbar(sc, ax=ax)
    #cbar0.set_label(r'infector $\langle\rho\rangle$', fontsize=25)

    # Create the hexbin plot
    #fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(attr_l, total_sizes, C=mean_rho_values, cmap='coolwarm', gridsize=30, mincnt=1)
    hb.set_clim(vmin=0.0, vmax=1.0)
    cbar0 = fig.colorbar(hb, ax=ax)
    cbar0.set_label(r'infector $\langle\rho\rangle$', fontsize=25)

    # Compute the mean value for each hexbin
    xbins = hb.get_offsets()[:, 0]
    ybins = hb.get_offsets()[:, 1]
    mean_values = hb.get_array()
    mean_rho_for_hexbins = []

    for i in range(len(mean_values)):
        if i == len(mean_values) - 1:  # Handle the last bin separately
            condition = np.logical_and(attr_l >= xbins[i], total_sizes >= ybins[i])
        else:
            condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_sizes >= ybins[i], total_sizes < ybins[i + 1]))

        indices = np.where(condition)
        if len(indices[0]) > 0:
            mean_rho_for_hexbins.append(np.mean(np.array(mean_rho_values)[indices]))
        else:
            mean_rho_for_hexbins.append(0.0)

    model_1 = LinearRegression()
    model_1.fit(attr_l.reshape(-1, 1), total_sizes)
    y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
    ax.plot(attr_l, y_pred_11, color='crimson', linestyle='--', linewidth=2)
    
    # Calculate and display R-squared value
    # Get R-squared value and display it
    r2_1 = model_1.score(attr_l.reshape(-1, 1), total_sizes)
    ax.text(0.15, 0.75, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax.transAxes, fontsize=20, color='black')

    ax.set_xlabel(r'$A$', fontsize=25)
    ax.set_ylabel('mean total cases', fontsize=25)
    ax.set_title('', fontsize=30)
    ax.tick_params(axis='both', labelsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'cases_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_invader_infected_rho():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    filename = 'config_grid_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    grid_pars = ut.read_json_file(fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)
    
    nlocs = 2500
    attr_l = space_df['attractiveness'].to_numpy()
    attr_cutoff = 0.000000001
    nlocs_eff = len(attr_l[attr_l > attr_cutoff])

    nsims = 10
    t_max = 1200
    R0 = 1.2
    r_0 = 0.0

    # Define empty lists for storing the curves
    #extended_invader_rho_sl = []
    #extended_infected_rho_a = []
    #extended_rho_a = []
    nlocs_inv = 0
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)
    inv_rho_hist_s = []
    inf_rho_hist_s = []
    rho_hist_s = []
    global_prev_s = []

    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))
        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)

        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        nagents_eff = len(agent_df)
        rho_a = an.get_rho_values(agent_df)
        nsims_eff = an.number_of_simulations(agent_df)

        invader_rho_sl = an.collect_location_invaders_rho(event_df, space_df, nlocs, nlocs_eff, t_max)
        sum_nan_elements = np.sum(np.isnan(invader_rho_sl))
        nlocs_inv = nlocs_eff * nsims_eff - sum_nan_elements

        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        inf_rho_a = an.get_rho_values(agent_df)

        global_prev_s.append(len(agent_df) / nagents_eff)
        rho_hist, _ = np.histogram(rho_a, bins=rho_bins)
        rho_hist_s.append(rho_hist)
        inv_rho_hist, _ = np.histogram(invader_rho_sl.flatten(), bins=rho_bins)
        norm_inv_rho_hist = np.array(inv_rho_hist) / nlocs_inv
        inv_rho_hist_s.append(norm_inv_rho_hist)
        inf_rho_hist, _ = np.histogram(inf_rho_a, bins=rho_bins)
        norm_inf_rho_hist = np.array(inf_rho_hist) / np.array(rho_hist)
        inf_rho_hist_s.append(norm_inf_rho_hist)

    # Compute mid points
    mid_points = np.asarray([rho_bins[i] + (rho_bins[i+1] - rho_bins[i]) / 2.0 
                                    for i in range(len(rho_bins) - 1)])

    # Convert the lists of histograms to numpy arrays
    inv_rho_hist_s = np.array(inv_rho_hist_s)
    inf_rho_hist_s = np.array(inf_rho_hist_s)
    rho_hist_s = np.array(rho_hist_s)

    # Compute total rho hist
    ext_rho_hist_r = np.sum(rho_hist_s, axis=0)
    expected_share = ext_rho_hist_r / np.sum(ext_rho_hist_r)

    # Compute average global prevalence
    mean_global_prev = np.mean(np.array(global_prev_s))

    # Compute the average histograms
    avg_inv_rho_hist = np.mean(inv_rho_hist_s, axis=0)
    avg_inf_rho_hist = np.mean(inf_rho_hist_s, axis=0)
    avg_rho_hist = np.mean(rho_hist_s, axis=0)

    # Compute the standard deviations
    std_inv_rho_hist = np.std(inv_rho_hist_s, axis=0)
    std_inf_rho_hist = np.std(inf_rho_hist_s, axis=0)
    std_rho_hist = np.std(rho_hist_s, axis=0)

    # Compute the upper and lower 95% CI
    upper_95CI_inv_rho_hist = avg_inv_rho_hist + 1.96 * std_inv_rho_hist / np.sqrt(len(inv_rho_hist_s))
    lower_95CI_inv_rho_hist = avg_inv_rho_hist - 1.96 * std_inv_rho_hist / np.sqrt(len(inv_rho_hist_s))
    
    upper_95CI_inf_rho_hist = avg_inf_rho_hist + 1.96 * std_inf_rho_hist / np.sqrt(len(inf_rho_hist_s))
    lower_95CI_inf_rho_hist = avg_inf_rho_hist - 1.96 * std_inf_rho_hist / np.sqrt(len(inf_rho_hist_s))
   
    #upper_95CI_rho_hist = avg_rho_hist + 1.96 * std_rho_hist / np.sqrt(len(rho_hist_s))
    #lower_95CI_rho_hist = avg_rho_hist - 1.96 * std_rho_hist / np.sqrt(len(rho_hist_s))

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))

    # SUBPLOT 0: INVASION BY RHO PROFILE
    ax[0].scatter(rho_bins[:-1], avg_inv_rho_hist, marker='o', color='teal', label=r'$mean$')
    ax[0].fill_between(rho_bins[:-1], lower_95CI_inv_rho_hist, upper_95CI_inv_rho_hist, color='teal', alpha=0.2, label='95% CI')
    ax[0].plot(rho_bins[:-1], expected_share, linestyle='--', color='indigo', label=r'null: $N_{inv,\rho}/N_{\rho}$')

    # Subplot 0 settings
    title = r'Invasion share by profile'
    ax[0].set_title(title, fontsize=30)
    ax[0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0].set_ylabel(r"$N_{{inv,\rho}}/N_{l, inv}$", fontsize=30)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].legend(fontsize=20)

    # SUBPLOT 0: INVASION BY RHO PROFILE
    ax[1].scatter(rho_bins[:-1], avg_inf_rho_hist, marker='o', color='teal', label=r'$mean$')
    ax[1].fill_between(rho_bins[:-1], lower_95CI_inf_rho_hist, upper_95CI_inf_rho_hist, color='teal', alpha=0.2, label='95% CI')
    r_inf = ut.sir_prevalence(R0, r_0)
    ax[1].axhline(r_inf, color='steelblue', linestyle='--', label=r'$r_{hom}(\infty)$')
    ax[1].axhline(mean_global_prev, color='crimson', linestyle='--', label='global sim')

    # Subplot 0 settings
    title = r'Infection share by profile'
    ax[1].set_title(title, fontsize=30)
    ax[1].set_xlabel(r"$\rho$", fontsize=30)
    ax[1].set_ylabel(r"$N_{{inf,\rho}}/N_{\rho}$", fontsize=30)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].legend(fontsize=25)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'inv_inf_t_shade_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_home_infected():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)
    attr_l = space_df['attractiveness'].to_numpy()

    nsims = 25
    nlocs = 2500

    # Define empty lists for storing the curves
    extended_results = []
    visits_until_a = []
    infected_where_a = []
    infected_where_freq_binf_a = []
    infected_where_freq_tmax_a = []
    home_a = []
    rho_a = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)
    
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        event_df = an.simulations_filter_event_data_frame(event_df, nsims_load=nsims)

        print("Counting")
        results = an.count_visits_until_infection_where_and_freq(agent_df, trajectory_df)

        extended_results.append(results)

    for results in extended_results:
        visits_until_a.extend(results[0])
        infected_where_a.extend(results[1])
        infected_where_freq_binf_a.extend(results[2])
        infected_where_freq_tmax_a.extend(results[3])
        home_a.extend(results[4])
        rho_a.extend(results[5])

    rho_a = np.array(rho_a)
    home_a = np.array(home_a)
    infected_where_a = np.array(infected_where_a)
    infected_where_freq_binf_a = np.array(infected_where_freq_binf_a)
    infected_where_freq_tmax_a = np.array(infected_where_freq_tmax_a)
    visits_until_a = np.array(visits_until_a)
    attr_a = np.array([attr_l[infected_where] for infected_where in infected_where_a])

    # Home selection
    home_infected_rho = rho_a[infected_where_a == home_a]
    out_infected_rho = rho_a[infected_where_a != home_a]
    home_freq_binf = infected_where_freq_binf_a[infected_where_a == home_a]
    home_freq_tmax = infected_where_freq_tmax_a[infected_where_a == home_a]
    out_freq_binf = infected_where_freq_binf_a[infected_where_a != home_a]
    out_freq_tmax = infected_where_freq_tmax_a[infected_where_a != home_a]
    home_visits = visits_until_a[infected_where_a == home_a]
    home_attr = attr_a[infected_where_a == home_a]
    out_attr = attr_a[infected_where_a != home_a]

    # Prepare figure
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

    nbins = 30
    rho_bins = np.linspace(0.0, 1.0, nbins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    counts_rho, _ = np.histogram(rho_a, bins=rho_bins)
    
    home_counts_rho, _ = np.histogram(home_infected_rho, bins=rho_bins)
    home_infected = np.sum(home_infected_rho) / np.sum(rho_a)
    total_home_elements = len(home_infected_rho)
    norm_home_counts_rho = home_counts_rho / counts_rho

    out_counts_rho, _ = np.histogram(out_infected_rho, bins=rho_bins)
    out_infected = np.sum(out_infected_rho) / np.sum(rho_a)
    total_out_elements = len(out_infected_rho)
    norm_out_counts_rho = out_counts_rho / counts_rho

    # Use `numpy.digitize` to get the bin indices for home_infected_rho and out_infected_rho
    home_bin_indices = np.digitize(home_infected_rho, rho_bins) - 1
    out_bin_indices = np.digitize(out_infected_rho, rho_bins) - 1

    # Calculate the average values for home_freq_binf, home_freq_tmax, out_freq_binf, and out_freq_tmax by rho bin category
    home_avg_binf = [home_freq_binf[home_bin_indices == i].mean() for i in range(nbins)]
    home_avg_tmax = [home_freq_tmax[home_bin_indices == i].mean() for i in range(nbins)]
    out_avg_binf = [out_freq_binf[out_bin_indices == i].mean() for i in range(nbins)]
    out_avg_tmax = [out_freq_tmax[out_bin_indices == i].mean() for i in range(nbins)]

    home_avg_attr = [home_attr[home_bin_indices == i].mean() for i in range(nbins)]
    out_avg_attr = [out_attr[out_bin_indices == i].mean() for i in range(nbins)]

    # SUBPLOT 0
    ax[0].scatter(rho_midpoints, norm_home_counts_rho, marker='o', color='dodgerblue', label=r'home mean')
    ax[0].scatter(rho_midpoints, norm_out_counts_rho, marker='o', color='firebrick', label=r'out mean')
    ax[0].axhline(home_infected, color='dodgerblue', linestyle='--', label='global home')
    ax[0].axhline(out_infected, color='firebrick', linestyle='--', label='global out')

    title = 'infected where?'
    ax[0].set_title(title, fontsize=30)
    ax[0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0].set_ylabel(r'$N_{inf,\rho, home(out)}/N_{inf,\rho}$', fontsize=25)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(fontsize=20)

    # SUBPLOT 1 
    ax[1].scatter(rho_midpoints, home_avg_tmax, marker='o', color='dodgerblue', label=r'home mean')
    ax[1].scatter(rho_midpoints, out_avg_tmax, marker='o', color='firebrick', label=r'out mean')

    title = 'recurrence to infection location'
    ax[1].set_title(title, fontsize=30)
    ax[1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1].set_ylabel(r'$f_{inf,l,\rho}$', fontsize=25)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(fontsize=20)

    # SUBPLOT 2
    attr_cutoff = 0.000000001
    attr_l = space_df['attractiveness'].to_numpy()
    attr_l = attr_l[attr_l > attr_cutoff]
    attr_max = np.max(attr_l)
    attr_min = np.min(attr_l)
    ax[2].scatter(rho_midpoints, home_avg_attr, marker='o', color='dodgerblue', label=r'home mean')
    ax[2].scatter(rho_midpoints, out_avg_attr, marker='o', color='firebrick', label=r'out mean')

    title = 'how attractive?'
    ax[2].set_title(title, fontsize=30)
    ax[2].set_xlabel(r'$\rho$', fontsize=25)
    ax[2].set_ylabel(r'$A_{inf,l,\rho}$', fontsize=25)
    ax[2].set_xlim(0.0, 1.0)
    ax[2].set_ylim(attr_min, attr_max)
    ax[2].tick_params(axis='both', labelsize=20)
    ax[2].legend(fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'home_infected_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_home_infector():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    nsims = 25
    nlocs = 2500
    t_max = 1200

    nbins = 30
    rho_bins = np.linspace(0.0, 1.0, nbins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    # Define empty lists for storing the curves
    ext_inf_rho_by_rho_bin = []
    ext_home_ce_by_rho_bin = []
    ext_home_cum_size_by_rho_bin = []
    ext_home_freq_by_rho_bin = []
    ext_home_attr_by_rho_bin = []
    ext_out_ce_by_rho_bin = []
    ext_out_cum_size_by_rho_bin = []
    ext_out_freq_by_rho_bin = []
    ext_out_attr_by_rho_bin = []
    ext_tot_cum_size_by_rho_bin = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)
    
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        event_df = an.simulations_filter_event_data_frame(event_df, nsims_load=nsims)

        print("Counting")
        results = an.count_home_recurrence_and_attractiveness_for_infectors(agent_df, event_df, trajectory_df, space_df, nlocs, t_max)

        inf_rho_a = np.zeros(len(results))
        home_ce_a = np.zeros(len(results))
        home_cum_size_a = np.zeros(len(results))
        home_freq_a = np.zeros(len(results))
        home_attr_a = np.zeros(len(results))
        out_ce_a = np.zeros(len(results))
        out_cum_size_a = np.zeros(len(results))
        out_freq_a = np.zeros(len(results))
        out_attr_a = np.zeros(len(results))

        for result, a in zip(results, range(len(results))):
            inf_rho_a[a] = result[0]
            home_ce_a[a] = result[2]
            out_ce_a[a] = result[3]
            home_cum_size_a[a] = result[4]
            out_cum_size_a[a] = result[5]
            home_freq_a[a] = result[6]
            out_freq_a[a] = result[7]
            home_attr_a[a] = result[8]
            out_attr_a[a] = result[9]

        inf_rho_by_rho_bin = np.zeros(len(rho_bins) - 1)
        home_ce_by_rho_bin = np.zeros(len(rho_bins) - 1)
        home_cum_size_by_rho_bin = np.zeros(len(rho_bins) - 1)
        home_freq_by_rho_bin = np.zeros(len(rho_bins) - 1)
        home_attr_by_rho_bin = np.zeros(len(rho_bins) - 1)
        out_ce_by_rho_bin = np.zeros(len(rho_bins) -1 )
        out_cum_size_by_rho_bin = np.zeros(len(rho_bins) - 1)
        out_freq_by_rho_bin = np.zeros(len(rho_bins) - 1)
        out_attr_by_rho_bin = np.zeros(len(rho_bins) - 1)
        total_ce_by_rho_bin = np.zeros(len(rho_bins) - 1)
        total_cum_size_by_rho_bin = np.zeros(len(rho_bins) - 1)
        
        for rho, a in zip(inf_rho_a, range(len(inf_rho_a))):
            # Determine the corresponding rho bin for each inf_rho_a value
            bin_index = np.digitize(rho, rho_bins)

            inf_rho_by_rho_bin[bin_index - 1] += 1
            
            home_ce_by_rho_bin[bin_index - 1] += home_ce_a[a]
            home_cum_size_by_rho_bin[bin_index - 1] += home_cum_size_a[a]
            home_freq_by_rho_bin[bin_index - 1] += home_freq_a[a]
            home_attr_by_rho_bin[bin_index - 1] += home_attr_a[a]
            
            out_ce_by_rho_bin[bin_index - 1] += out_ce_a[a]
            out_cum_size_by_rho_bin[bin_index - 1] += out_cum_size_a[a]
            out_freq_by_rho_bin[bin_index - 1] += out_freq_a[a]
            out_attr_by_rho_bin[bin_index - 1] += out_attr_a[a]

            total_ce_by_rho_bin[bin_index - 1] += (home_ce_a[a] + out_ce_a[a])
            total_cum_size_by_rho_bin[bin_index - 1] += (home_cum_size_a[a] + out_cum_size_a[a])

        ext_inf_rho_by_rho_bin.append(inf_rho_by_rho_bin)

        ext_home_ce_by_rho_bin.append(home_ce_by_rho_bin / total_ce_by_rho_bin)
        ext_out_ce_by_rho_bin.append(out_ce_by_rho_bin / total_ce_by_rho_bin)

        ext_home_cum_size_by_rho_bin.append(home_cum_size_by_rho_bin / total_cum_size_by_rho_bin)
        ext_out_cum_size_by_rho_bin.append(out_cum_size_by_rho_bin / total_cum_size_by_rho_bin)

        ext_tot_cum_size_by_rho_bin.append(total_cum_size_by_rho_bin / inf_rho_by_rho_bin)
        
        ext_home_freq_by_rho_bin.append(home_freq_by_rho_bin / inf_rho_by_rho_bin)
        ext_home_attr_by_rho_bin.append(home_attr_by_rho_bin / inf_rho_by_rho_bin)
        
        ext_out_freq_by_rho_bin.append(out_freq_by_rho_bin / inf_rho_by_rho_bin)
        ext_out_attr_by_rho_bin.append(out_attr_by_rho_bin / inf_rho_by_rho_bin)
        
    # Convert the lists of histograms to numpy arrays
    ext_inf_rho_by_rho_bin = np.array(ext_inf_rho_by_rho_bin)
    ext_home_ce_by_rho_bin = np.array(ext_home_ce_by_rho_bin)
    ext_home_cum_size_by_rho_bin = np.array(ext_home_cum_size_by_rho_bin)
    ext_home_freq_by_rho_bin = np.array(ext_home_freq_by_rho_bin)
    ext_home_attr_by_rho_bin = np.array(ext_home_attr_by_rho_bin)
    ext_out_ce_by_rho_bin = np.array(ext_out_ce_by_rho_bin)
    ext_out_cum_size_by_rho_bin = np.array(ext_out_cum_size_by_rho_bin)
    ext_out_freq_by_rho_bin = np.array(ext_out_freq_by_rho_bin)
    ext_out_attr_by_rho_bin = np.array(ext_out_attr_by_rho_bin)
    ext_tot_cum_size_by_rho_bin = np.array(ext_tot_cum_size_by_rho_bin)

    # Compute the average histograms
    avg_home_ce_hist = np.mean(ext_home_ce_by_rho_bin, axis=0)
    avg_out_ce_hist = np.mean(ext_out_ce_by_rho_bin, axis=0)
    avg_home_cum_size_hist = np.mean(ext_home_cum_size_by_rho_bin, axis=0)
    avg_out_cum_size_hist = np.mean(ext_out_cum_size_by_rho_bin, axis=0)
    avg_home_freq_hist = np.mean(ext_home_freq_by_rho_bin, axis=0)
    avg_out_freq_hist = np.mean(ext_out_freq_by_rho_bin, axis=0)
    avg_home_attr_hist = np.mean(ext_home_attr_by_rho_bin, axis=0)
    avg_out_attr_hist = np.mean(ext_out_attr_by_rho_bin, axis=0)
    avg_tot_cum_size_hist = np.mean(ext_tot_cum_size_by_rho_bin, axis=0)

    # Compute the standard deviations
    std_home_ce_hist = np.std(ext_home_ce_by_rho_bin, axis=0)
    std_out_ce_hist = np.std(ext_out_ce_by_rho_bin, axis=0)
    std_home_cum_size_hist = np.std(ext_home_cum_size_by_rho_bin, axis=0)
    std_out_cum_size_hist = np.std(ext_out_cum_size_by_rho_bin, axis=0)
    std_home_freq_hist = np.std(ext_home_freq_by_rho_bin, axis=0)
    std_out_freq_hist = np.std(ext_out_freq_by_rho_bin, axis=0)
    std_home_attr_hist = np.std(ext_home_attr_by_rho_bin, axis=0)
    std_out_attr_hist = np.std(ext_out_attr_by_rho_bin, axis=0)
    std_tot_cum_size_hist = np.std(ext_tot_cum_size_by_rho_bin, axis=0)

    # Compute the upper and lower 95% CI
    u95CI_home_ce_hist = avg_home_ce_hist + 1.96 * std_home_ce_hist / np.sqrt(len(ext_home_ce_by_rho_bin))
    l95CI_home_ce_hist = avg_home_ce_hist - 1.96 * std_home_ce_hist / np.sqrt(len(ext_home_ce_by_rho_bin))
    u95CI_out_ce_hist = avg_out_ce_hist + 1.96 * std_out_ce_hist / np.sqrt(len(ext_out_ce_by_rho_bin))
    l95CI_out_ce_hist = avg_out_ce_hist - 1.96 * std_out_ce_hist / np.sqrt(len(ext_out_ce_by_rho_bin))
    u95CI_home_cum_size_hist = avg_home_cum_size_hist + 1.96 * std_home_cum_size_hist / np.sqrt(len(ext_home_cum_size_by_rho_bin))
    l95CI_home_cum_size_hist = avg_home_cum_size_hist - 1.96 * std_home_cum_size_hist / np.sqrt(len(ext_home_cum_size_by_rho_bin))
    u95CI_out_cum_size_hist = avg_out_cum_size_hist + 1.96 * std_out_cum_size_hist / np.sqrt(len(ext_out_cum_size_by_rho_bin))
    l95CI_out_cum_size_hist = avg_out_cum_size_hist - 1.96 * std_out_cum_size_hist / np.sqrt(len(ext_out_cum_size_by_rho_bin))
    u95CI_home_freq_hist = avg_home_freq_hist + 1.96 * std_home_freq_hist / np.sqrt(len(ext_home_freq_by_rho_bin))
    l95CI_home_freq_hist = avg_home_freq_hist - 1.96 * std_home_freq_hist / np.sqrt(len(ext_home_freq_by_rho_bin))
    u95CI_out_freq_hist = avg_out_freq_hist + 1.96 * std_out_freq_hist / np.sqrt(len(ext_out_freq_by_rho_bin))
    l95CI_out_freq_hist = avg_out_freq_hist - 1.96 * std_out_freq_hist / np.sqrt(len(ext_out_freq_by_rho_bin))
    u95CI_home_attr_hist = avg_home_attr_hist + 1.96 * std_home_attr_hist / np.sqrt(len(ext_home_attr_by_rho_bin))
    l95CI_home_attr_hist = avg_home_attr_hist - 1.96 * std_home_attr_hist / np.sqrt(len(ext_home_attr_by_rho_bin))
    u95CI_out_attr_hist = avg_out_attr_hist + 1.96 * std_out_attr_hist / np.sqrt(len(ext_out_attr_by_rho_bin))
    l95CI_out_attr_hist = avg_out_attr_hist - 1.96 * std_out_attr_hist / np.sqrt(len(ext_out_attr_by_rho_bin))
    u95CI_tot_cum_size_hist = avg_tot_cum_size_hist + 1.96 * std_tot_cum_size_hist / np.sqrt(len(ext_tot_cum_size_by_rho_bin))
    l95CI_tot_cum_size_hist = avg_tot_cum_size_hist - 1.96 * std_tot_cum_size_hist / np.sqrt(len(ext_tot_cum_size_by_rho_bin))

    # PREPARE FIGURE 1 - HOME-INFECTOR
    fig1, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

    # SUBPLOT 0
    ax[0].scatter(rho_midpoints, avg_home_ce_hist, marker='o', color='dodgerblue', label=r'home mean')
    ax[0].scatter(rho_midpoints, avg_out_ce_hist, marker='o', color='firebrick', label=r'out mean')
    total = np.sum(ext_home_ce_by_rho_bin) + np.sum(ext_out_ce_by_rho_bin)
    home_fraction = np.sum(ext_home_ce_by_rho_bin) / total
    out_fraction = np.sum(ext_out_ce_by_rho_bin) / total
    ax[0].axhline(home_fraction, color='dodgerblue', linestyle='--', label='global home')
    ax[0].axhline(out_fraction, color='firebrick', linestyle='--', label='global out')

    title = 'event triggered where?'
    ax[0].set_title(title, fontsize=30)
    ax[0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0].set_ylabel(r'$share of events at home (outside)$', fontsize=25)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(fontsize=20)

    # SUBPLOT 1 
    ax[1].scatter(rho_midpoints, avg_home_freq_hist, marker='o', color='dodgerblue', label=r'home mean')
    ax[1].scatter(rho_midpoints, avg_out_freq_hist, marker='o', color='firebrick', label=r'out mean')

    title = 'recurrence to event location'
    ax[1].set_title(title, fontsize=30)
    ax[1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1].set_ylabel(r'$f_{inf,l,\rho}$', fontsize=25)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(fontsize=20)

    # SUBPLOT 2
    attr_cutoff = 0.000000001
    attr_l = space_df['attractiveness'].to_numpy()
    attr_l = attr_l[attr_l > attr_cutoff]
    attr_max = np.max(attr_l)
    attr_min = np.min(attr_l)

    ax[2].scatter(rho_midpoints, avg_home_attr_hist, marker='o', color='dodgerblue', label=r'home mean')
    ax[2].scatter(rho_midpoints, avg_out_attr_hist, marker='o', color='firebrick', label=r'out mean')

    title = 'how attractive?'
    ax[2].set_ylim(attr_min, attr_max)
    ax[2].set_title(title, fontsize=30)
    ax[2].set_xlabel(r'$\rho$', fontsize=25)
    ax[2].set_ylabel(r'$A_{inf,l,\rho}$', fontsize=25)
    ax[2].set_xlim(0.0, 1.0)
    ax[2].tick_params(axis='both', labelsize=20)
    ax[2].legend(fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'home_infector_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

    # Prepare figure
    fig2, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    ax[0].scatter(rho_midpoints, avg_home_cum_size_hist, marker='o', color='dodgerblue', label=r'$mean$')
    ax[0].fill_between(rho_midpoints, l95CI_home_cum_size_hist, u95CI_home_cum_size_hist, color='dodgerblue', alpha=0.2, label='95% CI')
    ax[0].scatter(rho_midpoints, avg_out_cum_size_hist, marker='o', color='firebrick', label=r'$mean$')
    ax[0].fill_between(rho_midpoints, l95CI_out_cum_size_hist, u95CI_out_cum_size_hist, color='firebrick', alpha=0.2, label='95% CI')
    
    ax[1].scatter(rho_midpoints, avg_tot_cum_size_hist, marker='o', color='teal', label=r'$mean$')
    ax[1].fill_between(rho_midpoints, l95CI_tot_cum_size_hist, u95CI_tot_cum_size_hist, color='teal', alpha=0.2, label='95% CI')

    # Subplot 0 settings
    #ax[0].set_title(title, fontsize=30)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0].set_ylabel(r"share of new cases at home/outside", fontsize=30)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].legend(fontsize=20)

    #ax[1].set_title(title, fontsize=30)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 2.0)
    ax[1].set_xlabel(r"$\rho$", fontsize=30)
    ax[1].set_ylabel(r"new cases/$N_{inf,\rho}$", fontsize=30)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].legend(fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'foi_rho_inf_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_force_of_infection_contribution():
    lower_path = 'config/'
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    nsims = 25
    nlocs = 2500
    t_max = 1200

    nbins = 30
    rho_bins = np.linspace(0.0, 1.0, nbins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    # Define empty lists for storing the curves
    ext_cum_sizes_by_rho_bin = []
    ext_norm_inf_rho_cum_sizes_by_rho_bin = []
    ext_norm_rho_cum_sizes_by_rho_bin = []
   
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build fullname
        lower_path = 'data'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        
        nagents_eff = len(agent_df)
        rho_a = an.get_rho_values(agent_df)
        nsims_eff = an.number_of_simulations(agent_df)
        rho_hist, _ = np.histogram(rho_a, bins=rho_bins)

        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        print("Counting")
        results = an.count_event_sizes_experienced_in_rho_t_inf_groups(agent_df, event_df, trajectory_df, nlocs, t_max)
        
        inf_rho_a = np.zeros(len(results))
        cum_size_a = np.zeros(len(results))
        for result, a in zip(results, range(len(results))):
            inf_rho_a[a] = result[0]
            cum_size_a[a] = result[2]

        cum_sizes_by_rho_bin = np.zeros(len(rho_bins) - 1)
        for rho, cum_size in zip(inf_rho_a, cum_size_a):
            # Determine the corresponding rho bin for each inf_rho_a value
            bin_index = np.digitize(rho, rho_bins)
            cum_sizes_by_rho_bin[bin_index - 1] += cum_size

        ext_cum_sizes_by_rho_bin.append(cum_sizes_by_rho_bin)

        inf_rho_hist, _ = np.histogram(inf_rho_a, bins=rho_bins)
        norm_inf_rho_cum_sizes_by_rho_bin = cum_sizes_by_rho_bin / inf_rho_hist
        ext_norm_inf_rho_cum_sizes_by_rho_bin.append(norm_inf_rho_cum_sizes_by_rho_bin)

        norm_rho_cum_sizes_by_rho_bin = cum_sizes_by_rho_bin / rho_hist
        ext_norm_rho_cum_sizes_by_rho_bin.append(norm_rho_cum_sizes_by_rho_bin)

    # Convert the lists of histograms to numpy arrays
    ext_norm_inf_rho_cum_sizes_by_rho_bin = np.array(ext_norm_inf_rho_cum_sizes_by_rho_bin)
    ext_norm_rho_cum_sizes_by_rho_bin = np.array(ext_norm_rho_cum_sizes_by_rho_bin)

    # Compute the average histograms
    avg_norm_inf_rho_nc_hist = np.mean(ext_norm_inf_rho_cum_sizes_by_rho_bin, axis=0)
    avg_norm_rho_nc_hist = np.mean(ext_norm_rho_cum_sizes_by_rho_bin, axis=0)

    # Compute the standard deviations
    std_norm_inf_rho_nc_hist = np.std(ext_norm_inf_rho_cum_sizes_by_rho_bin, axis=0)
    std_norm_rho_nc_hist = np.std(ext_norm_rho_cum_sizes_by_rho_bin, axis=0)

    # Compute the upper and lower 95% CI
    u95CI_norm_inf_rho_nc_hist = avg_norm_inf_rho_nc_hist + 1.96 * std_norm_inf_rho_nc_hist / np.sqrt(len(ext_norm_inf_rho_cum_sizes_by_rho_bin))
    l95CI_norm_inf_rho_nc_hist = avg_norm_inf_rho_nc_hist - 1.96 * std_norm_inf_rho_nc_hist / np.sqrt(len(ext_norm_inf_rho_cum_sizes_by_rho_bin))
    u95CI_norm_rho_nc_hist = avg_norm_rho_nc_hist + 1.96 * std_norm_rho_nc_hist / np.sqrt(len(ext_norm_rho_cum_sizes_by_rho_bin))
    l95CI_norm_rho_nc_hist = avg_norm_rho_nc_hist - 1.96 * std_norm_rho_nc_hist / np.sqrt(len(ext_norm_rho_cum_sizes_by_rho_bin))

    # Prepare figure
    fig, ax = plt.subplots()

    ax.scatter(rho_midpoints, avg_norm_inf_rho_nc_hist, marker='o', color='teal', label=r'$mean$')
    ax.fill_between(rho_midpoints, l95CI_norm_inf_rho_nc_hist, u95CI_norm_inf_rho_nc_hist, color='teal', alpha=0.2, label='95% CI')

    #ax.scatter(rho_midpoints, avg_norm_rho_nc_hist, marker='o', color='teal', label=r'$mean$')
    #ax.fill_between(rho_midpoints, l95CI_norm_rho_nc_hist, u95CI_norm_rho_nc_hist, color='teal', alpha=0.2, label='95% CI')

    # Subplot 0 settings
    title = r'contribution to infections'
    ax.set_title(title, fontsize=30)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$\rho$", fontsize=30)
    ax.set_ylabel(r"new cases/$N_{inf,\rho}$", fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    ax.legend(fontsize=20)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'foi_rho_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_contact_time_averaged():
    lower_path = 'config/'
    # Load grid parameters from grid retriever json file
    grbl_filename = 'config_grid_bl_retriever'
    grbl_fullname = os.path.join(cwd_path, lower_path, grbl_filename)
    grid_pars = ut.read_json_file(grbl_fullname)
    # Delete grid time stamp key
    del grid_pars['tm']
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)[0:5]

    nsims = 2
    t_ini = 30
    t_fin = 500
    low_cutoff = 0.05
    mid_cutoff_l = 0.45
    mid_cutoff_h = 0.55
    hig_cutoff = 0.95

    nbins = 30
    rho_bins = np.linspace(0.0, 1.0, nbins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    # Define structures
    ext_k_avg_sr = np.zeros((len(timestamps), len(rho_bins) - 1))

    # Loop over mobility realizations
    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grbl_fullname)
        grid_pars['tm'] = timestamp
        grid_filename = ut.build_grid_results_filename(grid_pars)
        grid_filename += '.pickle'
        grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
        grid = ut.open_file(grid_fullname)['inner']
        
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build epidemic filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)

        # Build epidemic fullname
        lower_path = 'data'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Build dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)

        # Collect trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)

        # Filter dataframes by outbreak size
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=1)
        #agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')

        for j in range(len(rho_bins) - 1):
            print("{0}".format(j))
            # Define the cutoff values for the current rho bin range
            low_cutoff = rho_bins[j]
            high_cutoff = rho_bins[j + 1]

            # Filter the agent_df based on the current rho bin range
            agent_df_in_range = agent_df[(low_cutoff <= agent_df['mobility']) & (agent_df['mobility'] <= high_cutoff)]

            # Compute the average degree for the agents in the current rho bin range
            k_avg_s = an.compute_average_degree(agent_df_in_range, trajectory_df, grid, t_ini, t_fin)
            ext_k_avg_sr[i, j] = k_avg_s

    # Average curve
    avg_k_avg_r = np.mean(ext_k_avg_sr, axis=0)

    # Compute the standard deviation
    std_k_avg_r = np.std(ext_k_avg_sr, axis=0)

    # Compute the upper and lower 95% CI
    u95CI_k_avg_r = avg_k_avg_r + 1.96 * std_k_avg_r / np.sqrt(len(ext_k_avg_sr))
    l95CI_k_avg_r = avg_k_avg_r - 1.96 * std_k_avg_r / np.sqrt(len(ext_k_avg_sr))
 
    # Prepare figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # SUBPLOT 0
    ax.scatter(rho_midpoints, avg_k_avg_r, marker='o', color='teal', label=r'$mean$')
    ax.fill_between(rho_midpoints, l95CI_k_avg_r, u95CI_k_avg_r, color='teal', alpha=0.2, label='95% CI')

    # Settings 0
    title = r""
    ax.set_title(title, fontsize=25)
    ax.set_xlabel(r"$\rho$", fontsize=30)
    ax.set_ylabel(r"$\langle k\rangle_{\Delta t, \rho}$", fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    ax.legend(fontsize=25)

    # General settings. Font, font sizes, layout...
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'contact_tavg_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_top_attractiveness():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    nsims = 2
    nlocs = 2500
    t_max = 1200

    nbins = 30
    rho_bins = np.linspace(0.0, 1.0, nbins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    # Define empty lists for storing the curves
    ext_inf_rho_by_rho_bin = []
    ext_top_freq_by_rho_bin = []
    ext_top2_freq_by_rho_bin = []

    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=nsims)
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)
    
        print("Counting")
        results = an.get_top_locations_attractiveness(agent_df, trajectory_df, space_df, t_max)

        inf_rho_a = np.zeros(len(results))
        top_freq_a = np.zeros(len(results))
        top2_freq_a = np.zeros(len(results))
        
        for result, a in zip(results, range(len(results))):
            inf_rho_a[a] = result[0]
            top_freq_a[a] = result[1]
            top2_freq_a[a] = result[2]

        inf_rho_by_rho_bin = np.zeros(len(rho_bins) - 1)
        top_freq_by_rho_bin = np.zeros(len(rho_bins) - 1)
        top2_freq_by_rho_bin = np.zeros(len(rho_bins) - 1)
 
        for rho, a in zip(inf_rho_a, range(len(inf_rho_a))):
            # Determine the corresponding rho bin for each inf_rho_a value
            bin_index = np.digitize(rho, rho_bins)

            inf_rho_by_rho_bin[bin_index - 1] += 1
            
            top_freq_by_rho_bin[bin_index - 1] += top_freq_a[a]
            top2_freq_by_rho_bin[bin_index - 1] += top2_freq_a[a]
        
        ext_inf_rho_by_rho_bin.append(inf_rho_by_rho_bin)
        ext_top_freq_by_rho_bin.append(top_freq_by_rho_bin / inf_rho_by_rho_bin)
        ext_top2_freq_by_rho_bin.append(top2_freq_by_rho_bin / inf_rho_by_rho_bin)
    
    # Convert the lists of histograms to numpy arrays
    ext_inf_rho_by_rho_bin = np.array(ext_inf_rho_by_rho_bin)
    ext_top_freq_by_rho_bin = np.array(ext_top_freq_by_rho_bin)
    ext_top2_freq_by_rho_bin = np.array(ext_top2_freq_by_rho_bin)

    # Compute the average histograms
    avg_top_freq_hist = np.mean(ext_top_freq_by_rho_bin, axis=0)
    avg_top2_freq_hist = np.mean(ext_top2_freq_by_rho_bin, axis=0)

    # Compute the standard deviations
    std_top_freq_hist = np.std(ext_top_freq_by_rho_bin, axis=0)
    std_top2_freq_hist = np.std(ext_top2_freq_by_rho_bin, axis=0)

    # Compute the upper and lower 95% CI
    u95CI_top_freq_hist = avg_top_freq_hist + 1.96 * std_top_freq_hist / np.sqrt(len(ext_top_freq_by_rho_bin))
    l95CI_top_freq_hist = avg_top_freq_hist - 1.96 * std_top_freq_hist / np.sqrt(len(ext_top_freq_by_rho_bin))
    u95CI_top2_freq_hist = avg_top2_freq_hist + 1.96 * std_top2_freq_hist / np.sqrt(len(ext_top2_freq_by_rho_bin))
    l95CI_top2_freq_hist = avg_top2_freq_hist - 1.96 * std_top2_freq_hist / np.sqrt(len(ext_top2_freq_by_rho_bin))

    # Prepare figure
    fig, ax = plt.subplots()

    ax.scatter(rho_midpoints, avg_top_freq_hist, marker='o', color='firebrick', label=r'top mean')
    ax.fill_between(rho_midpoints, l95CI_top_freq_hist, u95CI_top_freq_hist , color='firebrick', alpha=0.2, label='95% CI')

    ax.scatter(rho_midpoints, avg_top2_freq_hist, marker='o', color='dodgerblue', label=r'2nd loc mean')
    ax.fill_between(rho_midpoints, l95CI_top2_freq_hist, u95CI_top2_freq_hist , color='dodgerblue', alpha=0.2, label='95% CI')

    # Subplot 0 settings
    title = r'dominant locations attractiveness'
    ax.set_title(title, fontsize=30)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\rho$", fontsize=30)
    ax.set_ylabel(r"$A$", fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    ax.legend(fontsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'attr_rho_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_202308D1_home_and_out():
    lower_path = 'config/'
    # Load space parameters from space retriever json file
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    # Load grid parameters from mobility retriever json file
    filename = 'config_mobility_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    mob_pars = ut.read_json_file(fullname)
    # Load grid parameters from grid retriever json file
    grid_filename = 'config_grid_bl_retriever'
    grid_fullname = os.path.join(cwd_path, lower_path, grid_filename)
    # Load epidemic parameters from epidemic retriever json file
    filename = 'config_epidemic_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    epi_pars = ut.read_json_file(fullname)
    # Collect all mobility time stamps
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path)
    timestamps = ut.collect_mobility_timestamps(fullpath)[0:17]

    # Load space data
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    nsims = 1
    nlocs = 2500
    t_max = 1200

    nbins = 30
    rho_bins = np.linspace(0.0, 1.0, nbins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    # Define empty lists for storing the curves
    ext_inf_rho_by_rho_bin = []
    ext_home_home_by_rho_bin = []
    ext_home_out_by_rho_bin = []
    ext_out_home_by_rho_bin = []
    ext_out_out_by_rho_bin = []

    for timestamp, i in zip(timestamps, range(len(timestamps))):
        print("Loop {0}, timestamp: {1}".format(i+1, timestamp))

        grid_pars = ut.read_json_file(grid_fullname)
        del grid_pars['tm']
        grid_pars_copy = grid_pars.copy()

        # Build filename
        epi_filename = ut.build_epidemic_results_filename(epi_pars, grid_pars, timestamp)
        
        # Build fullname
        lower_path = 'data'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)
        
        # Build  agent dataframe
        agent_df = an.build_agent_data_frame(epi_fullname, nsims_load=nsims)
        agent_df = an.outbreak_filter_agent_data_frame(agent_df, prevalence_threshold=0.01)
        agent_df = an.health_status_filter_agent_data_frame(agent_df, 'Removed')
        agent_df = an.simulations_filter_agent_data_frame(agent_df, nsims_load=nsims)
        event_df = an.build_event_data_frame(epi_fullname, nsims_load=nsims)
        
        # Collect infected trajectories
        mob_filename = ut.build_mobility_results_filename(grid_pars)
        mob_fullname = os.path.join(cwd_path, lower_path, mob_filename)
        age_filename = ut.build_chosen_agents_filename(grid_pars_copy, timestamp)
        fullname_chosen = os.path.join(cwd_path, lower_path, age_filename)
        trajectory_df = an.collect_trajectories(mob_fullname, fullname_chosen)
    
        # Filter dataframe by outbreak size
        event_df = an.outbreak_filter_event_data_frame(event_df, agent_df, prevalence_threshold=0.01)
        event_df = an.simulations_filter_event_data_frame(event_df, nsims_load=nsims)

        print("Counting")
        results = an.count_home_and_out(agent_df, event_df, trajectory_df, space_df, nlocs, t_max)

        inf_rho_a = np.zeros(len(results))
        home_home_a = np.zeros(len(results))
        home_out_a = np.zeros(len(results))
        out_home_a = np.zeros(len(results))
        out_out_a = np.zeros(len(results))
    
        for result, a in zip(results, range(len(results))):
            inf_rho_a[a] = result[0]
            home_home_a[a] = result[1]
            home_out_a[a] = result[2]
            out_home_a[a] = result[3]
            out_out_a[a] = result[4]

        inf_rho_by_rho_bin = np.zeros(len(rho_bins) - 1)
        home_home_by_rho_bin = np.zeros(len(rho_bins) - 1)
        home_out_by_rho_bin = np.zeros(len(rho_bins) - 1)
        out_home_by_rho_bin = np.zeros(len(rho_bins) - 1)
        out_out_by_rho_bin = np.zeros(len(rho_bins) - 1)
        total_by_rho_bin = np.zeros(len(rho_bins) - 1)
        
        for rho, a in zip(inf_rho_a, range(len(inf_rho_a))):
            # Determine the corresponding rho bin for each inf_rho_a value
            bin_index = np.digitize(rho, rho_bins)

            inf_rho_by_rho_bin[bin_index - 1] += 1
            
            home_home_by_rho_bin[bin_index - 1] += home_home_a[a]
            home_out_by_rho_bin[bin_index - 1] += home_out_a[a]
            out_home_by_rho_bin[bin_index - 1] += out_home_a[a]
            out_out_by_rho_bin[bin_index - 1] += out_out_a[a]
            total_by_rho_bin[bin_index - 1] += (home_home_a[a] + home_out_a[a] + out_home_a[a] + out_out_a[a])
            
        ext_inf_rho_by_rho_bin.append(inf_rho_by_rho_bin)

        ext_home_home_by_rho_bin.append(home_home_by_rho_bin / total_by_rho_bin)
        ext_home_out_by_rho_bin.append(home_out_by_rho_bin / total_by_rho_bin)
        ext_out_home_by_rho_bin.append(out_home_by_rho_bin / total_by_rho_bin)
        ext_out_out_by_rho_bin.append(out_out_by_rho_bin / total_by_rho_bin)
    
    # Convert the lists of histograms to numpy arrays
    ext_inf_rho_by_rho_bin = np.array(ext_inf_rho_by_rho_bin)
    ext_home_home_by_rho_bin = np.array(ext_home_home_by_rho_bin) 
    ext_home_out_by_rho_bin = np.array(ext_home_out_by_rho_bin) 
    ext_out_home_by_rho_bin = np.array(ext_out_home_by_rho_bin) 
    ext_out_out_by_rho_bin = np.array(ext_out_out_by_rho_bin)

    # Compute the average histograms
    avg_home_home_hist = np.mean(ext_home_home_by_rho_bin, axis=0)
    avg_home_out_hist = np.mean(ext_home_out_by_rho_bin, axis=0)
    avg_out_home_hist = np.mean(ext_out_home_by_rho_bin, axis=0)
    avg_out_out_hist = np.mean(ext_out_out_by_rho_bin, axis=0)
   
    # Compute the standard deviations
    std_home_home_hist = np.std(ext_home_home_by_rho_bin, axis=0)
    std_home_out_hist = np.std(ext_home_out_by_rho_bin, axis=0)
    std_out_home_hist = np.std(ext_out_home_by_rho_bin, axis=0)
    std_out_out_hist = np.std(ext_out_out_by_rho_bin, axis=0)
    
    # Compute the upper and lower 95% CI
    u95CI_home_home_hist = avg_home_home_hist + 1.96 * std_home_home_hist / np.sqrt(len(ext_home_home_by_rho_bin))
    l95CI_home_home_hist = avg_home_home_hist - 1.96 * std_home_home_hist / np.sqrt(len(ext_home_home_by_rho_bin))
    u95CI_home_out_hist = avg_home_out_hist + 1.96 * std_home_out_hist / np.sqrt(len(ext_home_out_by_rho_bin))
    l95CI_home_out_hist = avg_home_out_hist - 1.96 * std_home_out_hist / np.sqrt(len(ext_home_out_by_rho_bin))
    u95CI_out_home_hist = avg_out_home_hist + 1.96 * std_out_home_hist / np.sqrt(len(ext_out_home_by_rho_bin))
    l95CI_out_home_hist = avg_out_home_hist - 1.96 * std_out_home_hist / np.sqrt(len(ext_out_home_by_rho_bin))
    u95CI_out_out_hist = avg_out_out_hist + 1.96 * std_out_out_hist / np.sqrt(len(ext_out_out_by_rho_bin))
    l95CI_out_out_hist = avg_out_out_hist - 1.96 * std_out_out_hist / np.sqrt(len(ext_out_out_by_rho_bin))
    

    # PREPARE FIGURE 1 - HOME-INFECTOR
    fig, ax = plt.subplots()

    # SUBPLOT 0
    ax.scatter(rho_midpoints, avg_home_home_hist, marker='o', color='dodgerblue', label=r'h-h mean')
    ax.fill_between(rho_midpoints, l95CI_home_home_hist, u95CI_home_home_hist , color='dodgerblue', alpha=0.2)

    ax.scatter(rho_midpoints, avg_home_out_hist, marker='.', color='slateblue', label=r'h-o mean')
    ax.fill_between(rho_midpoints, l95CI_home_out_hist, u95CI_home_out_hist , color='slateblue', alpha=0.2)

    ax.scatter(rho_midpoints, avg_out_home_hist, marker='s', color='deeppink', label=r'o-h mean')
    ax.fill_between(rho_midpoints, l95CI_out_home_hist, u95CI_out_home_hist , color='deeppink', alpha=0.2)

    ax.scatter(rho_midpoints, avg_out_out_hist, marker='^', color='firebrick', label=r'o-o mean')
    ax.fill_between(rho_midpoints, l95CI_out_out_hist, u95CI_out_out_hist , color='firebrick', alpha=0.2)

    title = 'origin-destination infections'
    ax.set_title(title, fontsize=30)
    ax.set_xlabel(r'$\rho$', fontsize=25)
    ax.set_ylabel(r'event fraction', fontsize=25)
    #ax.set_xlim(0.0, 1.0)
    #ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'home_and_out_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def main():

    #plot_202307D1_macro_time_evolution()
    #plot_202307D1_invader_infected_rho()
    #plot_202307D1_visits_until_infection()
    #plot_202307D1_micro_time_with_rho()
    #plot_202307D2_invader_infected_rho_map()
    #plot_202307D2_exploration_return_step()
    #plot_202307D2_epr_step_attractiveness_and_rho()
    #plot_202307D2_dynamics_with_lockdown()
    #plot_202307D3_visits_frequency()
    #plot_202307D3_newcomers()
    #plot_202307D3_infected_newcomers()
    #plot_202307D3_connectivity()
    #plot_202307D3_location_times()
    #plot_202307D3_case_flow()
    #plot_202307D4_contact()
    #plot_202307D5_visits_until_infection()
    #plot_202307D5_offsprings_by_event()
    #plot_202307D5_cases_in_location()
    #plot_202307D1_force_of_infection_contribution()
    #plot_202308D1_invader_infected_rho()
    #plot_202308D1_home_infected()
    plot_202308D1_force_of_infection_contribution()
    #plot_202308D1_contact_time_averaged()
    plot_202308D1_home_infector()
    #plot_202308D1_top_attractiveness()
    plot_202308D1_home_and_out()

    print("Take that!")

if __name__ == "__main__":
    main()
