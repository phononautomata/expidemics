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

import analysis as an
import utils as ut

cwd_path = os.getcwd()


def plot_figure1(
        focal_dist, 
        comp_flag=None, 
        bl_flag=None, 
        stats_flag=False, 
        t_inv_flag=False, 
        t_inf_flag=False, 
        ):
    """ 2x2 panel
    f00: invasion profile
    f01: invasion time profile
    f10: infection profile
    f11: infection time profile
    """

    # Load space parameters from space retriever json file
    lower_path = 'config/'
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)

    # Collect all digested epidemic file names
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    # Load space data
    lower_path = 'data/'
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    prevalence_cutoff = 0.05
    R0 = 1.2
    r_0 = 0.0

    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    agents_per_rho_sim = []
    infected_per_rho_sim = []
    invaders_per_rho_sim = []
    nlocs_invaded_sim = []
    total_cases_loc_sim = []
    t_inv_stats_per_rho_sim = []
    t_inf_stats_per_rho_sim = []
    t_inv_dist_per_rho_sim = []
    t_inf_dist_per_rho_sim = []

    comp_agents_per_rho_sim = []
    comp_infected_per_rho_sim = []
    comp_invaders_per_rho_sim = []
    comp_nlocs_invaded_sim = []
    comp_total_cases_loc_sim = []
    comp_t_inv_stats_per_rho_sim = []
    comp_t_inf_stats_per_rho_sim = []
    comp_t_inv_dist_per_rho_sim = []
    comp_t_inf_dist_per_rho_sim = []

    bl_agents_per_rho_sim = []
    bl_infected_per_rho_sim = []
    bl_invaders_per_rho_sim = []
    bl_nlocs_invaded_sim = []
    bl_total_cases_loc_sim = []
    bl_t_inv_stats_per_rho_sim = []
    bl_t_inf_stats_per_rho_sim = []
    bl_t_inv_dist_per_rho_sim = []
    bl_t_inf_dist_per_rho_sim = []

    # Loop over the collected file names
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))
        # Build the full path
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Check if the file exists
        if os.path.exists(epi_fullname):
            print(f"File {epi_fullname} accepted.")
            # Load digested epidemic data
            out_sim_data = ut.load_chapter_figure1_data(epi_fullname, 
                                                        stats_flag=stats_flag, 
                                                        t_inv_flag=t_inv_flag,
                                                        t_inf_flag=t_inf_flag,
                                                        )
            
            if focal_dist in epi_filename and bl_flag in epi_filename and 'depr' not in epi_filename:
                # Collect data from every realization in structure
                bl_agents_per_rho_sim.extend(out_sim_data['agents'])
                bl_infected_per_rho_sim.extend(out_sim_data['infected'])
                bl_invaders_per_rho_sim.extend(out_sim_data['invaders'])
                bl_nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
                bl_total_cases_loc_sim.extend(out_sim_data['total_cases'])
                
                if stats_flag:
                    bl_t_inv_stats_per_rho_sim.extend(out_sim_data['t_inv_stats'])
                    bl_t_inf_stats_per_rho_sim.extend(out_sim_data['t_inf_stats'])
                if t_inv_flag:
                    bl_t_inv_dist_per_rho_sim.extend(out_sim_data['t_inv_dist'])
                if t_inf_flag:
                    bl_t_inf_dist_per_rho_sim.extend(out_sim_data['t_inf_dist'])
            elif comp_flag is not None and comp_flag in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                comp_agents_per_rho_sim.extend(out_sim_data['agents'])
                comp_infected_per_rho_sim.extend(out_sim_data['infected'])
                comp_invaders_per_rho_sim.extend(out_sim_data['invaders'])
                comp_nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
                comp_total_cases_loc_sim.extend(out_sim_data['total_cases'])
                
                if stats_flag:
                    comp_t_inv_stats_per_rho_sim.extend(out_sim_data['t_inv_stats'])
                    comp_t_inf_stats_per_rho_sim.extend(out_sim_data['t_inf_stats'])
                if t_inv_flag:
                    comp_t_inv_dist_per_rho_sim.extend(out_sim_data['t_inv_dist'])
                if t_inf_flag:
                    comp_t_inf_dist_per_rho_sim.extend(out_sim_data['t_inf_dist'])
            elif focal_dist in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                agents_per_rho_sim.extend(out_sim_data['agents'])
                infected_per_rho_sim.extend(out_sim_data['infected'])
                invaders_per_rho_sim.extend(out_sim_data['invaders'])
                nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
                total_cases_loc_sim.extend(out_sim_data['total_cases'])
                
                if stats_flag:
                    t_inv_stats_per_rho_sim.extend(out_sim_data['t_inv_stats'])
                    t_inf_stats_per_rho_sim.extend(out_sim_data['t_inf_stats'])
                if t_inv_flag:
                    t_inv_dist_per_rho_sim.extend(out_sim_data['t_inv_dist'])
                if t_inf_flag:
                    t_inf_dist_per_rho_sim.extend(out_sim_data['t_inf_dist'])
    
            else:
                print(f"File {epi_fullname} exists but no matching case found.")
        else:
            # File doesn't exist, skip the rest of the loop
            print(f"File {epi_fullname} does not exist. Skipping this iteration.")
            continue

    if focal_dist != None:
        # Convert into numpy arrays
        agents_per_rho_sim = np.array(agents_per_rho_sim)
        infected_per_rho_sim = np.array(infected_per_rho_sim)
        invaders_per_rho_sim = np.array(invaders_per_rho_sim)
        nlocs_invaded_sim = np.array(nlocs_invaded_sim)
        total_cases_loc_sim = np.array(total_cases_loc_sim)

        # Compute final observables
        infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
        infected_fraction_per_rho_sim = infected_per_rho_sim / agents_per_rho_sim
        invaded_fraction_per_rho_sim = invaders_per_rho_sim / nlocs_invaded_sim[:, np.newaxis]

        # Filter failed outbreaks
        failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

        infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
        agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
        invaders_per_rho_sim = np.delete(invaders_per_rho_sim, failed_outbreaks, axis=0)
        nlocs_invaded_sim = np.delete(nlocs_invaded_sim, failed_outbreaks, axis=0)
        total_cases_loc_sim = np.delete(total_cases_loc_sim, failed_outbreaks, axis=0)

        if stats_flag:
            t_inv_stats_per_loc_sim = [sim for i, sim in enumerate(t_inv_stats_per_loc_sim) if i not in failed_outbreaks]
            t_inf_stats_per_loc_sim = [sim for i, sim in enumerate(t_inf_stats_per_loc_sim) if i not in failed_outbreaks]

            t_inv_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in t_inv_stats_per_rho_sim]
            avg_t_inv_avg_per_rho = np.nanmean(np.array(t_inv_avg_dist_per_rho), axis=0)
            std_t_inv_avg_per_rho = np.nanstd(np.array(t_inv_avg_dist_per_rho), axis=0)
            z = 1.96
            moe = z * (std_t_inv_avg_per_rho / np.sqrt(len(avg_t_inv_avg_per_rho)))
            u95_t_inv_avg_per_rho = avg_t_inv_avg_per_rho + moe
            l95_t_inv_avg_per_rho = avg_t_inv_avg_per_rho - moe

            #t_inv_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in t_inv_stats_per_rho_sim]
            #u95_t_inv_avg_per_rho = np.nanmean(np.array(t_inv_u95_dist_per_rho), axis=0)
            #t_inv_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in t_inv_stats_per_rho_sim]
            #l95_t_inv_avg_per_rho = np.nanmean(np.array(t_inv_l95_dist_per_rho), axis=0)

            t_inf_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in t_inf_stats_per_rho_sim]
            avg_t_inf_avg_per_rho = np.nanmean(np.array(t_inf_avg_dist_per_rho), axis=0)
            std_t_inf_avg_per_rho = np.nanstd(np.array(t_inf_avg_dist_per_rho), axis=0)
            z = 1.96
            moe = z * (std_t_inf_avg_per_rho / np.sqrt(len(avg_t_inf_avg_per_rho)))
            u95_t_inf_avg_per_rho = avg_t_inf_avg_per_rho + moe
            l95_t_inf_avg_per_rho = avg_t_inf_avg_per_rho - moe

            #t_inf_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in t_inf_stats_per_rho_sim]
            #u95_t_inf_avg_per_rho = np.nanmean(np.array(t_inf_u95_dist_per_rho), axis=0)
            #t_inf_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in t_inf_stats_per_rho_sim]
            #l95_t_inf_avg_per_rho = np.nanmean(np.array(t_inf_l95_dist_per_rho), axis=0)

        if t_inv_flag:
            t_inv_dist_per_rho_sim = [sim for i, sim in enumerate(t_inv_dist_per_rho_sim) if i not in failed_outbreaks]

            nbins = len(t_inv_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            t_inv_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(t_inv_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    t_inv_values = t_inv_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    t_inv_dist_per_rho[rho_idx].extend(t_inv_values)

            t_inv_avg_per_rho = np.array([np.nanmean(sublist) for sublist in t_inv_dist_per_rho])
            t_inv_std_per_rho = np.array([np.nanstd(sublist) for sublist in t_inv_dist_per_rho])
            z = 1.96
            moe = z * (t_inv_std_per_rho / np.sqrt([len(sublist) for sublist in t_inv_dist_per_rho]))
            t_inv_u95_per_rho = t_inv_avg_per_rho + moe
            t_inv_l95_per_rho = t_inv_avg_per_rho - moe

        if t_inf_flag:
            t_inf_dist_per_rho_sim = [sim for i, sim in enumerate(t_inf_dist_per_rho_sim) if i not in failed_outbreaks]

            nbins = len(t_inf_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            t_inf_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(t_inf_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    t_inf_values = t_inf_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    t_inf_dist_per_rho[rho_idx].extend(t_inf_values)

            t_inf_avg_per_rho = np.array([np.nanmean(sublist) for sublist in t_inf_dist_per_rho])
            t_inf_std_per_rho = np.array([np.nanstd(sublist) for sublist in t_inf_dist_per_rho])
            z = 1.96
            moe = z * (t_inf_std_per_rho / np.sqrt([len(sublist) for sublist in t_inf_dist_per_rho]))
            t_inf_u95_per_rho = t_inf_avg_per_rho + moe
            t_inf_l95_per_rho = t_inf_avg_per_rho - moe

        # Perform stats
        infected_fraction_avg_per_rho = np.nanmean(infected_fraction_per_rho_sim, axis=0)
        infected_fraction_avg = np.nanmean(infected_fraction_sim)
        std = np.std(infected_fraction_per_rho_sim, axis=0)
        nsims = len(infected_fraction_per_rho_sim)
        z = 1.96
        moe = z * (std / np.sqrt(nsims))
        infected_fraction_u95_per_rho = infected_fraction_avg_per_rho + moe
        infected_fraction_l95_per_rho = infected_fraction_avg_per_rho - moe

        invaded_fraction_avg_per_rho = np.nanmean(invaded_fraction_per_rho_sim, axis=0)
        std = np.std(invaded_fraction_per_rho_sim, axis=0)
        nsims = len(invaded_fraction_per_rho_sim)
        z = 1.96
        moe = z * (std / np.sqrt(nsims))
        invaded_fraction_u95_per_rho = invaded_fraction_avg_per_rho + moe
        invaded_fraction_l95_per_rho = invaded_fraction_avg_per_rho - moe

        total_cases_avg_loc = np.nanmean(total_cases_loc_sim, axis=0)
        attr_l = space_df['attractiveness'].to_numpy()

    if comp_flag != None:
        # Convert into numpy arrays
        comp_agents_per_rho_sim = np.array(comp_agents_per_rho_sim)
        comp_infected_per_rho_sim = np.array(comp_infected_per_rho_sim)
        comp_invaders_per_rho_sim = np.array(comp_invaders_per_rho_sim)
        comp_nlocs_invaded_sim = np.array(comp_nlocs_invaded_sim)
        comp_total_cases_loc_sim = np.array(comp_total_cases_loc_sim)

        # Filter failed outbreaks
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_failed_outbreaks = np.where(comp_infected_fraction_sim < prevalence_cutoff)[0]

        comp_infected_per_rho_sim = np.delete(comp_infected_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_agents_per_rho_sim = np.delete(comp_agents_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_invaders_per_rho_sim = np.delete(comp_invaders_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_nlocs_invaded_sim = np.delete(comp_nlocs_invaded_sim, comp_failed_outbreaks, axis=0)
        comp_total_cases_loc_sim = np.delete(comp_total_cases_loc_sim, comp_failed_outbreaks, axis=0)
        
        if stats_flag:
            comp_t_inv_stats_per_loc_sim = [sim for i, sim in enumerate(comp_t_inv_stats_per_loc_sim) if i not in comp_failed_outbreaks]
            comp_t_inf_stats_per_loc_sim = [sim for i, sim in enumerate(comp_t_inf_stats_per_loc_sim) if i not in comp_failed_outbreaks]

            # Compute stats
            comp_t_inv_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in comp_t_inv_stats_per_rho_sim]
            comp_avg_t_inv_avg_per_rho = np.nanmean(np.array(comp_t_inv_avg_dist_per_rho), axis=0)
            comp_t_inv_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in comp_t_inv_stats_per_rho_sim]
            comp_u95_t_inv_avg_per_rho = np.nanmean(np.array(comp_t_inv_u95_dist_per_rho), axis=0)
            comp_t_inv_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in comp_t_inv_stats_per_rho_sim]
            comp_l95_t_inv_avg_per_rho = np.nanmean(np.array(comp_t_inv_l95_dist_per_rho), axis=0)

            comp_t_inf_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in comp_t_inf_stats_per_rho_sim]
            comp_avg_t_inf_avg_per_rho = np.nanmean(np.array(comp_t_inf_avg_dist_per_rho), axis=0)
            comp_t_inf_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in comp_t_inf_stats_per_rho_sim]
            comp_u95_t_inf_avg_per_rho = np.nanmean(np.array(comp_t_inf_u95_dist_per_rho), axis=0)
            comp_t_inf_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in comp_t_inf_stats_per_rho_sim]
            comp_l95_t_inf_avg_per_rho = np.nanmean(np.array(comp_t_inf_l95_dist_per_rho), axis=0)

        if t_inv_flag:
            comp_t_inv_dist_per_rho_sim = [sim for i, sim in enumerate(comp_t_inv_dist_per_rho_sim) if i not in comp_failed_outbreaks]

            nbins = len(comp_t_inv_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            comp_t_inv_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_t_inv_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_t_inv_values = comp_t_inv_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_t_inv_dist_per_rho[rho_idx].extend(comp_t_inv_values)

            comp_t_inv_avg_per_rho = np.array([np.nanmean(sublist) for sublist in comp_t_inv_dist_per_rho])
            comp_t_inv_std_per_rho = np.array([np.nanstd(sublist) for sublist in comp_t_inv_dist_per_rho])
            z = 1.96
            comp_moe = z * (comp_t_inv_std_per_rho / np.sqrt([len(sublist) for sublist in comp_t_inv_dist_per_rho]))
            comp_t_inv_u95_per_rho = comp_t_inv_avg_per_rho + comp_moe
            comp_t_inv_l95_per_rho = comp_t_inv_avg_per_rho - comp_moe
    
        if t_inf_flag:
            comp_t_inf_dist_per_rho_sim = [sim for i, sim in enumerate(comp_t_inf_dist_per_rho_sim) if i not in comp_failed_outbreaks]

            nbins = len(comp_t_inf_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            comp_t_inf_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_t_inf_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_t_inf_values = comp_t_inf_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_t_inf_dist_per_rho[rho_idx].extend(comp_t_inf_values)

            comp_t_inf_avg_per_rho = np.array([np.nanmean(sublist) for sublist in comp_t_inf_dist_per_rho])
            comp_t_inf_std_per_rho = np.array([np.nanstd(sublist) for sublist in comp_t_inf_dist_per_rho])
            z = 1.96
            comp_moe = z * (comp_t_inf_std_per_rho / np.sqrt([len(sublist) for sublist in comp_t_inf_dist_per_rho]))
            comp_t_inf_u95_per_rho = comp_t_inf_avg_per_rho + comp_moe
            comp_t_inf_l95_per_rho = comp_t_inf_avg_per_rho - comp_moe

        # Compute final observables
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_infected_fraction_per_rho_sim = comp_infected_per_rho_sim / comp_agents_per_rho_sim
        comp_invaded_fraction_per_rho_sim = comp_invaders_per_rho_sim / comp_nlocs_invaded_sim[:, np.newaxis]
    
        # Perform stats
        comp_infected_fraction_avg_per_rho = np.nanmean(comp_infected_fraction_per_rho_sim, axis=0)
        comp_infected_fraction_avg = np.nanmean(comp_infected_fraction_sim)
        comp_std = np.std(comp_infected_fraction_per_rho_sim, axis=0)
        nsims = len(comp_infected_fraction_per_rho_sim)
        z = 1.96
        comp_moe = z * (comp_std / np.sqrt(nsims))
        comp_infected_fraction_u95_per_rho = comp_infected_fraction_avg_per_rho + comp_moe
        comp_infected_fraction_l95_per_rho = comp_infected_fraction_avg_per_rho - comp_moe

        comp_invaded_fraction_avg_per_rho = np.nanmean(comp_invaded_fraction_per_rho_sim, axis=0)
        comp_std = np.std(comp_invaded_fraction_per_rho_sim, axis=0)
        nsims = len(comp_invaded_fraction_per_rho_sim)
        z = 1.96
        comp_moe = z * (comp_std / np.sqrt(nsims))
        comp_invaded_fraction_u95_per_rho = comp_invaded_fraction_avg_per_rho + comp_moe
        comp_invaded_fraction_l95_per_rho = comp_invaded_fraction_avg_per_rho - comp_moe

    if bl_flag != None:
        # Convert into numpy arrays
        bl_agents_per_rho_sim = np.array(bl_agents_per_rho_sim)
        bl_infected_per_rho_sim = np.array(bl_infected_per_rho_sim)
        bl_invaders_per_rho_sim = np.array(bl_invaders_per_rho_sim)
        bl_nlocs_invaded_sim = np.array(bl_nlocs_invaded_sim)
        bl_total_cases_loc_sim = np.array(bl_total_cases_loc_sim)

        # Filter failed outbreaks
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_failed_outbreaks = np.where(bl_infected_fraction_sim < prevalence_cutoff)[0]

        bl_infected_per_rho_sim = np.delete(bl_infected_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_agents_per_rho_sim = np.delete(bl_agents_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_invaders_per_rho_sim = np.delete(bl_invaders_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_nlocs_invaded_sim = np.delete(bl_nlocs_invaded_sim, bl_failed_outbreaks, axis=0)
        bl_total_cases_loc_sim = np.delete(bl_total_cases_loc_sim, bl_failed_outbreaks, axis=0)
        
        if stats_flag:
            bl_t_inv_stats_per_loc_sim = [sim for i, sim in enumerate(bl_t_inv_stats_per_loc_sim) if i not in bl_failed_outbreaks]
            bl_t_inf_stats_per_loc_sim = [sim for i, sim in enumerate(bl_t_inf_stats_per_loc_sim) if i not in bl_failed_outbreaks]

            # Compute final stats for distributed variables
            bl_t_inv_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in bl_t_inv_stats_per_rho_sim]
            bl_avg_t_inv_avg_per_rho = np.nanmean(np.array(bl_t_inv_avg_dist_per_rho), axis=0)
            bl_t_inv_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in bl_t_inv_stats_per_rho_sim]
            bl_u95_t_inv_avg_per_rho = np.nanmean(np.array(bl_t_inv_u95_dist_per_rho), axis=0)
            bl_t_inv_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in bl_t_inv_stats_per_rho_sim]
            bl_l95_t_inv_avg_per_rho = np.nanmean(np.array(bl_t_inv_l95_dist_per_rho), axis=0)
    
            bl_t_inf_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in bl_t_inf_stats_per_rho_sim]
            bl_avg_t_inf_avg_per_rho = np.nanmean(np.array(bl_t_inf_avg_dist_per_rho), axis=0)
            bl_t_inf_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in bl_t_inf_stats_per_rho_sim]
            bl_u95_t_inf_avg_per_rho = np.nanmean(np.array(bl_t_inf_u95_dist_per_rho), axis=0)
            bl_t_inf_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in bl_t_inf_stats_per_rho_sim]
            bl_l95_t_inf_avg_per_rho = np.nanmean(np.array(bl_t_inf_l95_dist_per_rho), axis=0)

        if t_inv_flag:
            bl_t_inv_dist_per_rho_sim = [sim for i, sim in enumerate(bl_t_inv_dist_per_rho_sim) if i not in bl_failed_outbreaks]

            nbins = len(bl_t_inv_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            bl_t_inv_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_t_inv_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_t_inv_values = bl_t_inv_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_t_inv_dist_per_rho[rho_idx].extend(bl_t_inv_values)

            bl_t_inv_avg_per_rho = np.array([np.nanmean(sublist) for sublist in bl_t_inv_dist_per_rho])
            bl_t_inv_std_per_rho = np.array([np.nanstd(sublist) for sublist in bl_t_inv_dist_per_rho])
            z = 1.96
            bl_moe = z * (bl_t_inv_std_per_rho / np.sqrt([len(sublist) for sublist in bl_t_inv_dist_per_rho]))
            bl_t_inv_u95_per_rho = bl_t_inv_avg_per_rho + bl_moe
            bl_t_inv_l95_per_rho = bl_t_inv_avg_per_rho - bl_moe
    
        if t_inf_flag:
            bl_t_inf_dist_per_rho_sim = [sim for i, sim in enumerate(bl_t_inf_dist_per_rho_sim) if i not in bl_failed_outbreaks]

            nbins = len(bl_t_inf_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            bl_t_inf_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_t_inf_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_t_inf_values = bl_t_inf_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_t_inf_dist_per_rho[rho_idx].extend(bl_t_inf_values)

            bl_t_inf_avg_per_rho = np.array([np.nanmean(sublist) for sublist in bl_t_inf_dist_per_rho])
            bl_t_inf_std_per_rho = np.array([np.nanstd(sublist) for sublist in bl_t_inf_dist_per_rho])
            z = 1.96
            bl_moe = z * (bl_t_inf_std_per_rho / np.sqrt([len(sublist) for sublist in bl_t_inf_dist_per_rho]))
            bl_t_inf_u95_per_rho = bl_t_inf_avg_per_rho + bl_moe
            bl_t_inf_l95_per_rho = bl_t_inf_avg_per_rho - bl_moe

        # Compute final observables
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_infected_fraction_per_rho_sim = bl_infected_per_rho_sim / bl_agents_per_rho_sim
        bl_invaded_fraction_per_rho_sim = bl_invaders_per_rho_sim / bl_nlocs_invaded_sim[:, np.newaxis]
    
        # Compute stats
        bl_infected_fraction_avg_per_rho = np.nanmean(bl_infected_fraction_per_rho_sim, axis=0)
        bl_infected_fraction_avg = np.nanmean(bl_infected_fraction_sim)
        bl_std = np.std(bl_infected_fraction_per_rho_sim, axis=0)
        nsims = len(bl_infected_fraction_per_rho_sim)
        z = 1.96
        bl_moe = z * (bl_std / np.sqrt(nsims))
        bl_infected_fraction_u95_per_rho = bl_infected_fraction_avg_per_rho + bl_moe
        bl_infected_fraction_l95_per_rho = bl_infected_fraction_avg_per_rho - bl_moe

        bl_invaded_fraction_avg_per_rho = np.nanmean(bl_invaded_fraction_per_rho_sim, axis=0)
        bl_std = np.std(bl_invaded_fraction_per_rho_sim, axis=0)
        nsims = len(bl_invaded_fraction_per_rho_sim)
        z = 1.96
        bl_moe = z * (std / np.sqrt(nsims))
        bl_invaded_fraction_u95_per_rho = bl_invaded_fraction_avg_per_rho + bl_moe
        bl_invaded_fraction_l95_per_rho = bl_invaded_fraction_avg_per_rho - bl_moe

        # Compute stats for collapsed observales
        bl_infected_fraction_avg = np.mean(bl_infected_fraction_sim)

        bl_total_cases_avg_loc = np.nanmean(bl_total_cases_loc_sim, axis=0)

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 14))

    # SUBPLOT 00: INVASION FRACTION PROFILE
    if focal_dist != None:
        ax[0, 0].scatter(rho_bins, invaded_fraction_avg_per_rho, marker='o', color='teal', label=r'beta mean')
        ax[0, 0].fill_between(rho_bins, invaded_fraction_l95_per_rho, invaded_fraction_u95_per_rho, color='teal', alpha=0.2)
        expected_share = np.sum(agents_per_rho_sim, axis=0) / np.sum(agents_per_rho_sim)
        ax[0, 0].plot(rho_bins, expected_share, linestyle='--', color='indigo', label=r'null: $N_{inv,\rho}/N_{\rho}$')

    if comp_flag != None:
        ax[0, 0].scatter(rho_bins, comp_invaded_fraction_avg_per_rho, marker='o', color='seagreen', label=r'gaussian mean')
        ax[0, 0].fill_between(rho_bins, comp_invaded_fraction_l95_per_rho, comp_invaded_fraction_u95_per_rho, color='seagreen', alpha=0.2)
        comp_expected_share = np.sum(comp_agents_per_rho_sim, axis=0) / np.sum(comp_agents_per_rho_sim)
        ax[0, 0].plot(rho_bins, comp_expected_share, linestyle='--', color='indigo')

    if bl_flag != None:
        ax[0, 0].scatter(rho_bins, bl_invaded_fraction_avg_per_rho, marker='o', color='steelblue', label=r'baseline mean')
        ax[0, 0].fill_between(rho_bins, bl_invaded_fraction_l95_per_rho, bl_invaded_fraction_u95_per_rho, color='steelblue', alpha=0.2)
        bl_expected_share = np.sum(bl_agents_per_rho_sim, axis=0) / np.sum(bl_agents_per_rho_sim)
        ax[0, 0].plot(rho_bins, bl_expected_share, linestyle='--', color='navy')

    # Subplot 00 settings
    title = r'Invasion share by $\rho$ profile'
    ax[0, 0].set_title(title, fontsize=30)
    ax[0, 0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0, 0].set_ylabel(r"$N_{{inv,\rho}}/N_{l, inv}$", fontsize=30)
    ax[0, 0].tick_params(axis='both', labelsize=25)
    ax[0, 0].set_xlim(0.0, 1.0)
    ax[0, 0].legend(loc='center', fontsize=15)

    # SUBPLOT 01: INVASION TIME PROFILE
    if focal_dist != None:
        if stats_flag:
            ax[0, 1].plot(rho_bins, avg_t_inv_avg_per_rho, marker='o', color='teal', label=r'beta $\langle t_{{inv}}\rangle $')
            ax[0, 1].fill_between(rho_bins, l95_t_inv_avg_per_rho, u95_t_inv_avg_per_rho, color='teal', alpha=0.2)
        if t_inv_flag:
            ax[0, 1].plot(rho_bins, t_inv_avg_per_rho, marker='o', color='teal', label=r'beta $\langle t_{{inv}}\rangle $')
            ax[0, 1].fill_between(rho_bins, t_inv_l95_per_rho, t_inv_u95_per_rho, color='teal', alpha=0.2)

    if comp_flag != None:
        if stats_flag:
            ax[0, 1].plot(rho_bins, comp_avg_t_inv_avg_per_rho, marker='o', color='seagreen', label=r'gaussian')
            ax[0, 1].fill_between(rho_bins, comp_l95_t_inv_avg_per_rho, comp_u95_t_inv_avg_per_rho, color='seagreen', alpha=0.2)
        if t_inv_flag:
            ax[0, 1].plot(rho_bins, comp_t_inv_avg_per_rho, marker='o', color='seagreen', label=r'gaussian')
            ax[0, 1].fill_between(rho_bins, comp_t_inv_l95_per_rho, comp_t_inv_u95_per_rho, color='seagreen', alpha=0.2)

    if bl_flag != None:
        if stats_flag:
            ax[0, 1].plot(rho_bins, bl_avg_t_inv_avg_per_rho, marker='o', color='steelblue', label=r'baseline')
            ax[0, 1].fill_between(rho_bins, bl_l95_t_inv_avg_per_rho, bl_u95_t_inv_avg_per_rho, color='steelblue', alpha=0.2)
        if t_inv_flag:
            ax[0, 1].plot(rho_bins, bl_t_inv_avg_per_rho, marker='o', color='steelblue', label=r'baseline')
            ax[0, 1].fill_between(rho_bins, bl_t_inv_l95_per_rho, bl_t_inv_u95_per_rho, color='steelblue', alpha=0.2)

    # Subplot 01 settings
    title = r'Invasion times by $\rho$ profile'
    ax[0, 1].set_title(title, fontsize=30)
    ax[0, 1].set_xlabel(r"$\rho$", fontsize=30)
    ax[0, 1].set_ylabel(r"$t_{{inv}}$", fontsize=30)
    ax[0, 1].tick_params(axis='both', labelsize=25)
    ax[0, 1].set_xlim(0.0, 1.0)
    ax[0, 1].legend(fontsize=25)

    # SUBPLOT 10: INFECTION FRACTION PROFILE
    if focal_dist != None:
        ax[1, 0].scatter(rho_bins, infected_fraction_avg_per_rho, marker='o', color='teal', label=r'beta mean')
        ax[1, 0].fill_between(rho_bins, infected_fraction_l95_per_rho, infected_fraction_u95_per_rho, color='teal', alpha=0.2)
        r_inf = ut.sir_prevalence(R0, r_0)
        ax[1, 0].axhline(r_inf, color='steelblue', linestyle='--', label=r'$r_{hom}(\infty)$')
        ax[1, 0].axhline(infected_fraction_avg, color='crimson', linestyle='--', label='global sim')

    if comp_flag != None:
        ax[1, 0].scatter(rho_bins, comp_infected_fraction_avg_per_rho, marker='o', color='seagreen', label=r'gaussian')
        ax[1, 0].fill_between(rho_bins, comp_infected_fraction_l95_per_rho, comp_infected_fraction_u95_per_rho, color='seagreen', alpha=0.2)
        ax[1, 0].axhline(comp_infected_fraction_avg, color='crimson', linestyle='--', label='global sim')
    
    if bl_flag != None:
        ax[1, 0].scatter(rho_bins, bl_infected_fraction_avg_per_rho, marker='o', color='steelblue', label=r'baseline')
        ax[1, 0].fill_between(rho_bins, bl_infected_fraction_l95_per_rho, bl_infected_fraction_u95_per_rho, color='steelblue', alpha=0.2)
        ax[1, 0].axhline(bl_infected_fraction_avg, color='deeppink', linestyle='--')

    # Subplot 10 settings
    title = r'Infection share by $\rho$ profile'
    ax[1, 0].set_title(title, fontsize=30)
    ax[1, 0].set_xlabel(r"$\rho$", fontsize=30)
    ax[1, 0].set_ylabel(r"$N_{{inf,\rho}}/N_{\rho}$", fontsize=30)
    ax[1, 0].tick_params(axis='both', labelsize=25)
    ax[1, 0].set_xlim(0.0, 1.0)
    ax[1, 0].legend(loc='upper center', fontsize=15)

    # SUBPLOT 11: INFECTION TIME PROFILE
    if focal_dist != None:
        if stats_flag:
            ax[1, 1].plot(rho_bins, avg_t_inf_avg_per_rho, marker='o', color='teal', label=r'beta $\langle t_{{inf}}\rangle $')
            ax[1, 1].fill_between(rho_bins, l95_t_inf_avg_per_rho, u95_t_inf_avg_per_rho, color='teal', alpha=0.2,)
        if t_inf_flag:
            ax[1, 1].plot(rho_bins, t_inf_avg_per_rho, marker='o', color='teal', label=r'beta $\langle t_{{inf}}\rangle $')
            ax[1, 1].fill_between(rho_bins, t_inf_l95_per_rho, t_inf_u95_per_rho, color='teal', alpha=0.2)

    if comp_flag != None:
        if stats_flag:
            ax[1, 1].plot(rho_bins, comp_avg_t_inf_avg_per_rho, marker='o', color='seagreen', label=r'gaussian')
            ax[1, 1].fill_between(rho_bins, comp_l95_t_inf_avg_per_rho, comp_u95_t_inf_avg_per_rho, color='seagreen', alpha=0.2)
        if t_inf_flag:
            ax[1, 1].plot(rho_bins, comp_t_inf_avg_per_rho, marker='o', color='seagreen', label=r'gaussian')
            ax[1, 1].fill_between(rho_bins, comp_t_inf_l95_per_rho, comp_t_inf_u95_per_rho, color='seagreen', alpha=0.2)

    if bl_flag != None:
        if stats_flag:
            ax[1, 1].plot(rho_bins, bl_avg_t_inf_avg_per_rho, marker='o', color='steelblue', label=r'baseline')
            ax[1, 1].fill_between(rho_bins, bl_l95_t_inf_avg_per_rho, bl_u95_t_inf_avg_per_rho, color='steelblue', alpha=0.2)
        if t_inf_flag:
            ax[1, 1].plot(rho_bins, bl_t_inf_avg_per_rho, marker='o', color='steelblue', label=r'baseline')
            ax[1, 1].fill_between(rho_bins, bl_t_inf_l95_per_rho, bl_t_inf_u95_per_rho, color='steelblue', alpha=0.2)

    # Subplot 11 settings
    title = r'Infection times by $\rho$ profile'
    ax[1, 1].set_title(title, fontsize=30)
    ax[1, 1].set_xlabel(r"$\rho$", fontsize=30)
    ax[1, 1].set_ylabel(r"$t_{{inf}}$", fontsize=30)
    ax[1, 1].tick_params(axis='both', labelsize=25)
    ax[1, 1].set_xlim(0.0, 1.0)
    ax[1, 1].legend(fontsize=25)

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
    base_name = 'chf1_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_figure2(
        focal_dist, 
        comp_flag=None, 
        bl_flag=None, 
        stats_flag=False, 
        r_inv_flag=False, 
        r_inf_flag=False,
        ):
    """ 1x2 panel
    f01: invasion map
    f02: infection vs. attractiveness scatter
    """

    # Load space parameters from space retriever json file
    lower_path = 'config/'
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
  
    # Collect all digested epidemic file names
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    # Load space data
    lower_path = 'data/'
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    prevalence_cutoff = 0.05
    R0 = 1.2
    r_0 = 0.0

    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    agents_per_rho_sim = []
    infected_per_rho_sim = []
    invaders_per_rho_sim = []
    nlocs_invaded_sim = []
    total_cases_loc_sim = []
    r_inv_stats_per_loc_sim = []
    r_inf_stats_per_loc_sim = []
    r_inv_dist_per_loc_sim = []
    r_inf_dist_per_loc_sim = []

    comp_agents_per_rho_sim = []
    comp_infected_per_rho_sim = []
    comp_invaders_per_rho_sim = []
    comp_nlocs_invaded_sim = []
    comp_total_cases_loc_sim = []
    comp_r_inv_stats_per_loc_sim = []
    comp_r_inf_stats_per_loc_sim = []
    comp_r_inv_dist_per_loc_sim = []
    comp_r_inf_dist_per_loc_sim = []

    bl_agents_per_rho_sim = []
    bl_infected_per_rho_sim = []
    bl_invaders_per_rho_sim = []
    bl_nlocs_invaded_sim = []
    bl_total_cases_loc_sim = []
    bl_r_inv_stats_per_loc_sim = []
    bl_r_inf_stats_per_loc_sim = []
    bl_r_inv_dist_per_loc_sim = []
    bl_r_inf_dist_per_loc_sim = []

    # Loop over the collected file names
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))
        # Build the full path
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Check if the file exists
        if os.path.exists(epi_fullname):
            print(f"File {epi_fullname} accepted.")
            # Load digested epidemic data
            out_sim_data = ut.load_chapter_figure2_data(epi_fullname, 
                                                        stats_flag=stats_flag, 
                                                        r_inv_flag=r_inf_flag,
                                                        r_inf_flag=r_inf_flag,
                                                        )
            
            if focal_dist in epi_filename and bl_flag in epi_filename and 'depr' not in epi_filename:
                # Collect data from every realization in structure
                bl_agents_per_rho_sim.extend(out_sim_data['agents'])
                bl_infected_per_rho_sim.extend(out_sim_data['infected'])
                bl_invaders_per_rho_sim.extend(out_sim_data['invaders'])
                bl_nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
                bl_total_cases_loc_sim.extend(out_sim_data['total_cases'])
                
                if stats_flag:
                    bl_r_inv_stats_per_loc_sim.extend(out_sim_data['r_inv_stats'])
                    bl_r_inf_stats_per_loc_sim.extend(out_sim_data['r_inf_stats'])
                if r_inv_flag:
                    bl_r_inv_dist_per_loc_sim.extend(out_sim_data['r_inv_dist'])
                if r_inf_flag:
                    bl_r_inf_dist_per_loc_sim.extend(out_sim_data['r_inf_dist'])
            elif comp_flag is not None and comp_flag in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                comp_agents_per_rho_sim.extend(out_sim_data['agents'])
                comp_infected_per_rho_sim.extend(out_sim_data['infected'])
                comp_invaders_per_rho_sim.extend(out_sim_data['invaders'])
                comp_nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
                comp_total_cases_loc_sim.extend(out_sim_data['total_cases'])
                
                if stats_flag:
                    comp_r_inv_stats_per_loc_sim.extend(out_sim_data['r_inv_stats'])
                    comp_r_inf_stats_per_loc_sim.extend(out_sim_data['r_inf_stats'])
                if r_inv_flag:
                    comp_r_inv_dist_per_loc_sim.extend(out_sim_data['r_inv_dist'])
                if r_inf_flag:
                    comp_r_inf_dist_per_loc_sim.extend(out_sim_data['r_inf_dist'])
            elif focal_dist in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                agents_per_rho_sim.extend(out_sim_data['agents'])
                infected_per_rho_sim.extend(out_sim_data['infected'])
                invaders_per_rho_sim.extend(out_sim_data['invaders'])
                nlocs_invaded_sim.extend(out_sim_data['nlocs_invaded'])
                total_cases_loc_sim.extend(out_sim_data['total_cases'])
                
                if stats_flag:
                    r_inv_stats_per_loc_sim.extend(out_sim_data['r_inv_stats'])
                    r_inf_stats_per_loc_sim.extend(out_sim_data['r_inf_stats'])
                if r_inv_flag:
                    r_inv_dist_per_loc_sim.extend(out_sim_data['r_inv_dist'])
                if r_inf_flag:
                    r_inf_dist_per_loc_sim.extend(out_sim_data['r_inf_dist'])
            else:
                print(f"File {epi_fullname} exists but no matching case found.")
        else:
            # File doesn't exist, skip the rest of the loop
            print(f"File {epi_fullname} does not exist. Skipping this iteration.")
            continue

    # Convert into numpy arrays
    agents_per_rho_sim = np.array(agents_per_rho_sim)
    infected_per_rho_sim = np.array(infected_per_rho_sim)
    invaders_per_rho_sim = np.array(invaders_per_rho_sim)
    nlocs_invaded_sim = np.array(nlocs_invaded_sim)
    total_cases_loc_sim = np.array(total_cases_loc_sim)

    # Compute final observables
    infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
    infected_fraction_per_rho_sim = infected_per_rho_sim / agents_per_rho_sim
    invaded_fraction_per_rho_sim = invaders_per_rho_sim / nlocs_invaded_sim[:, np.newaxis]

    # Filter failed outbreaks
    failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

    infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
    agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
    invaders_per_rho_sim = np.delete(invaders_per_rho_sim, failed_outbreaks, axis=0)
    nlocs_invaded_sim = np.delete(nlocs_invaded_sim, failed_outbreaks, axis=0)
    total_cases_loc_sim = np.delete(total_cases_loc_sim, failed_outbreaks, axis=0)

    if stats_flag:
        r_inv_stats_per_loc_sim = [sim for i, sim in enumerate(r_inv_stats_per_loc_sim) if i not in failed_outbreaks]
        r_inf_stats_per_loc_sim = [sim for i, sim in enumerate(r_inf_stats_per_loc_sim) if i not in failed_outbreaks]

        r_inv_avg_dist_per_loc = [[loc_data['mean'] for loc_data in sim_data] for sim_data in r_inv_stats_per_loc_sim]
        avg_r_inv_avg_per_loc = np.nanmean(np.array(r_inv_avg_dist_per_loc), axis=0)

        r_inf_avg_dist_per_loc = [[loc_data['mean'] for loc_data in sim_data] for sim_data in r_inf_stats_per_loc_sim]
        avg_r_inf_avg_per_loc = np.nanmean(np.array(r_inf_avg_dist_per_loc), axis=0)

        nlocs = 2500
        x_cells = int(np.sqrt(nlocs))
        y_cells = x_cells
        rho_avg_lattice = np.zeros((x_cells, y_cells))
        l = 0
        for i in range(x_cells):
            for j in range(y_cells):
                rho_avg_lattice[y_cells - 1 -j, i] = avg_r_inv_avg_per_loc[l]
                l += 1

    if r_inv_flag:
        r_inv_dist_per_loc_sim = [sim for i, sim in enumerate(r_inv_dist_per_loc_sim) if i not in failed_outbreaks]
        
        nlocs = len(r_inv_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
        r_inv_dist_per_loc = [[] for _ in range(nlocs)]

        # Iterate through simulations and rhos to aggregate t_inv values
        for sim_idx in range(len(r_inv_dist_per_loc_sim)):
            for loc_idx in range(nlocs):
                # Extract t_inv values for sim_idx and rho_idx
                r_inv_values = r_inv_dist_per_loc_sim[sim_idx][loc_idx]
                # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                r_inv_dist_per_loc[loc_idx].extend(r_inv_values)
        
        r_inv_avg_per_loc = np.array([np.nanmean(sublist) for sublist in r_inv_dist_per_loc])
        r_inv_std_per_loc = np.array([np.nanstd(sublist) for sublist in r_inv_dist_per_loc])
        z = 1.96
        moe = z * (r_inv_std_per_loc / np.sqrt(len(r_inv_avg_per_loc)))
        r_inv_u95_per_loc = r_inv_avg_per_loc + moe
        r_inv_l95_per_loc = r_inv_avg_per_loc - moe

        nlocs = 2500
        x_cells = int(np.sqrt(nlocs))
        y_cells = x_cells
        rho_avg_lattice = np.zeros((x_cells, y_cells))
        l = 0
        for i in range(x_cells):
            for j in range(y_cells):
                rho_avg_lattice[y_cells - 1 -j, i] = r_inv_avg_per_loc[l]
                l += 1

    if r_inf_flag:
        r_inf_dist_per_loc_sim = [sim for i, sim in enumerate(r_inf_dist_per_loc_sim) if i not in failed_outbreaks]
        
        nlocs = len(r_inf_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
        r_inf_dist_per_loc = [[] for _ in range(nlocs)]

        # Iterate through simulations and rhos to aggregate t_inv values
        for sim_idx in range(len(r_inf_dist_per_loc_sim)):
            for loc_idx in range(nlocs):
                # Extract t_inv values for sim_idx and rho_idx
                r_inf_values = r_inf_dist_per_loc_sim[sim_idx][loc_idx]
                # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                r_inf_dist_per_loc[loc_idx].extend(r_inf_values)
        
        r_inf_avg_per_loc = np.array([np.nanmean(sublist) for sublist in r_inf_dist_per_loc])
        r_inf_std_per_loc = np.array([np.nanstd(sublist) for sublist in r_inf_dist_per_loc])
        z = 1.96
        moe = z * (r_inf_std_per_loc / np.sqrt(len(r_inf_avg_per_loc)))
        r_inf_u95_per_loc = r_inf_avg_per_loc + moe
        r_inf_l95_per_loc = r_inf_avg_per_loc - moe

    # Perform stats
    infected_fraction_avg_per_rho = np.nanmean(infected_fraction_per_rho_sim, axis=0)
    infected_fraction_avg = np.nanmean(infected_fraction_sim)
    std = np.std(infected_fraction_per_rho_sim, axis=0)
    nsims = len(infected_fraction_per_rho_sim)
    z = 1.96
    moe = z * (std / np.sqrt(nsims))
    infected_fraction_u95_per_rho = infected_fraction_avg_per_rho + moe
    infected_fraction_l95_per_rho = infected_fraction_avg_per_rho - moe
    
    invaded_fraction_avg_per_rho = np.nanmean(invaded_fraction_per_rho_sim, axis=0)
    std = np.std(invaded_fraction_per_rho_sim, axis=0)
    nsims = len(invaded_fraction_per_rho_sim)
    z = 1.96
    moe = z * (std / np.sqrt(nsims))
    invaded_fraction_u95_per_rho = invaded_fraction_avg_per_rho + moe
    invaded_fraction_l95_per_rho = invaded_fraction_avg_per_rho - moe

    total_cases_avg_loc = np.nanmean(total_cases_loc_sim, axis=0)

    attr_l = space_df['attractiveness'].to_numpy()
    

    if comp_flag != None:
        # Convert into numpy arrays
        comp_agents_per_rho_sim = np.array(comp_agents_per_rho_sim)
        comp_infected_per_rho_sim = np.array(comp_infected_per_rho_sim)
        comp_invaders_per_rho_sim = np.array(comp_invaders_per_rho_sim)
        comp_nlocs_invaded_sim = np.array(comp_nlocs_invaded_sim)
        comp_total_cases_loc_sim = np.array(comp_total_cases_loc_sim)

        # Filter failed outbreaks
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_failed_outbreaks = np.where(comp_infected_fraction_sim < prevalence_cutoff)[0]

        comp_infected_per_rho_sim = np.delete(comp_infected_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_agents_per_rho_sim = np.delete(comp_agents_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_invaders_per_rho_sim = np.delete(comp_invaders_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_nlocs_invaded_sim = np.delete(comp_nlocs_invaded_sim, comp_failed_outbreaks, axis=0)
        comp_total_cases_loc_sim = np.delete(comp_total_cases_loc_sim, comp_failed_outbreaks, axis=0)
        
        if stats_flag:
            comp_r_inv_stats_per_loc_sim = [sim for i, sim in enumerate(comp_r_inv_stats_per_loc_sim) if i not in comp_failed_outbreaks]
            comp_r_inf_stats_per_loc_sim = [sim for i, sim in enumerate(comp_r_inf_stats_per_loc_sim) if i not in comp_failed_outbreaks]

        if r_inv_flag:
            comp_r_inv_dist_per_loc_sim = [sim for i, sim in enumerate(comp_r_inv_dist_per_loc_sim) if i not in comp_failed_outbreaks]

            nlocs = len(comp_r_inv_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
            comp_r_inv_dist_per_loc = [[] for _ in range(nlocs)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_r_inv_dist_per_loc_sim)):
                for loc_idx in range(nlocs):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_r_inv_values = comp_r_inv_dist_per_loc_sim[sim_idx][loc_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_r_inv_dist_per_loc[loc_idx].extend(comp_r_inv_values)

            comp_r_inv_avg_per_loc = np.array([np.nanmean(sublist) for sublist in comp_r_inv_dist_per_loc])
            comp_r_inv_std_per_loc = np.array([np.nanstd(sublist) for sublist in comp_r_inv_dist_per_loc])
            z = 1.96
            comp_moe = z * (comp_r_inv_std_per_loc / np.sqrt(len(comp_r_inv_avg_per_loc)))
            comp_r_inv_u95_per_loc = comp_r_inv_avg_per_loc + comp_moe
            comp_r_inv_l95_per_loc = comp_r_inv_avg_per_loc - comp_moe

        if r_inf_flag:
            comp_r_inf_dist_per_loc_sim = [sim for i, sim in enumerate(comp_r_inf_dist_per_loc_sim) if i not in comp_failed_outbreaks]

            nlocs = len(comp_r_inf_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
            comp_r_inf_dist_per_loc = [[] for _ in range(nlocs)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_r_inf_dist_per_loc_sim)):
                for loc_idx in range(nlocs):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_r_inf_values = comp_r_inf_dist_per_loc_sim[sim_idx][loc_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_r_inf_dist_per_loc[loc_idx].extend(comp_r_inf_values)

            comp_r_inf_avg_per_loc = np.array([np.nanmean(sublist) for sublist in comp_r_inf_dist_per_loc])
            comp_r_inf_std_per_loc = np.array([np.nanstd(sublist) for sublist in comp_r_inf_dist_per_loc])
            z = 1.96
            comp_moe = z * (comp_r_inf_std_per_loc / np.sqrt(len(comp_r_inf_avg_per_loc)))
            comp_r_inf_u95_per_loc = comp_r_inf_avg_per_loc + comp_moe
            comp_r_inf_l95_per_loc = comp_r_inf_avg_per_loc - comp_moe

        # Compute final observables
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_infected_fraction_per_rho_sim = comp_infected_per_rho_sim / comp_agents_per_rho_sim
        comp_invaded_fraction_per_rho_sim = comp_invaders_per_rho_sim / comp_nlocs_invaded_sim[:, np.newaxis]
    
        # Perform stats
        comp_infected_fraction_avg_per_rho = np.nanmean(comp_infected_fraction_per_rho_sim, axis=0)
        comp_infected_fraction_avg = np.nanmean(comp_infected_fraction_sim)
        comp_std = np.std(comp_infected_fraction_per_rho_sim, axis=0)
        nsims = len(comp_infected_fraction_per_rho_sim)
        z = 1.96
        comp_moe = z * (comp_std / np.sqrt(nsims))
        comp_infected_fraction_u95_per_rho = comp_infected_fraction_avg_per_rho + comp_moe
        comp_infected_fraction_l95_per_rho = comp_infected_fraction_avg_per_rho - comp_moe

        comp_invaded_fraction_avg_per_rho = np.nanmean(comp_invaded_fraction_per_rho_sim, axis=0)
        comp_std = np.std(comp_invaded_fraction_per_rho_sim, axis=0)
        nsims = len(comp_invaded_fraction_per_rho_sim)
        z = 1.96
        comp_moe = z * (comp_std / np.sqrt(nsims))
        comp_invaded_fraction_u95_per_rho = comp_invaded_fraction_avg_per_rho + comp_moe
        comp_invaded_fraction_l95_per_rho = comp_invaded_fraction_avg_per_rho - comp_moe

    if bl_flag != None:
        # Convert into numpy arrays
        bl_agents_per_rho_sim = np.array(bl_agents_per_rho_sim)
        bl_infected_per_rho_sim = np.array(bl_infected_per_rho_sim)
        bl_invaders_per_rho_sim = np.array(bl_invaders_per_rho_sim)
        bl_nlocs_invaded_sim = np.array(bl_nlocs_invaded_sim)
        bl_total_cases_loc_sim = np.array(bl_total_cases_loc_sim)

        # Filter failed outbreaks
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_failed_outbreaks = np.where(bl_infected_fraction_sim < prevalence_cutoff)[0]

        bl_infected_per_rho_sim = np.delete(bl_infected_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_agents_per_rho_sim = np.delete(bl_agents_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_invaders_per_rho_sim = np.delete(bl_invaders_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_nlocs_invaded_sim = np.delete(bl_nlocs_invaded_sim, bl_failed_outbreaks, axis=0)
        bl_total_cases_loc_sim = np.delete(bl_total_cases_loc_sim, bl_failed_outbreaks, axis=0)
        
        if stats_flag:
            bl_r_inv_stats_per_loc_sim = [sim for i, sim in enumerate(bl_r_inv_stats_per_loc_sim) if i not in bl_failed_outbreaks]
            bl_r_inf_stats_per_loc_sim = [sim for i, sim in enumerate(bl_r_inf_stats_per_loc_sim) if i not in bl_failed_outbreaks]
    
    
        if r_inv_flag:
            bl_r_inv_dist_per_loc_sim = [sim for i, sim in enumerate(bl_r_inv_dist_per_loc_sim) if i not in bl_failed_outbreaks]

            nlocs = len(bl_r_inv_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
            bl_r_inv_dist_per_loc = [[] for _ in range(nlocs)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_r_inv_dist_per_loc_sim)):
                for loc_idx in range(nlocs):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_r_inv_values = bl_r_inv_dist_per_loc_sim[sim_idx][loc_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_r_inv_dist_per_loc[loc_idx].extend(bl_r_inv_values)

            bl_r_inv_avg_per_loc = np.array([np.nanmean(sublist) for sublist in bl_r_inv_dist_per_loc])
            bl_r_inv_std_per_loc = np.array([np.nanstd(sublist) for sublist in bl_r_inv_dist_per_loc])
            z = 1.96
            bl_moe = z * (bl_r_inv_std_per_loc / np.sqrt(len(bl_r_inv_avg_per_loc)))
            bl_r_inv_u95_per_loc = bl_r_inv_avg_per_loc + bl_moe
            bl_r_inv_l95_per_loc = bl_r_inv_avg_per_loc - bl_moe

        if r_inf_flag:
            bl_r_inf_dist_per_loc_sim = [sim for i, sim in enumerate(bl_r_inf_dist_per_loc_sim) if i not in bl_failed_outbreaks]

            nlocs = len(bl_r_inf_dist_per_loc_sim[0])  # Assuming all inner lists have the same size
            bl_r_inf_dist_per_loc = [[] for _ in range(nlocs)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_r_inf_dist_per_loc_sim)):
                for loc_idx in range(nsims):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_r_inf_values = bl_r_inf_dist_per_loc_sim[sim_idx][loc_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_r_inf_dist_per_loc[loc_idx].extend(bl_r_inf_values)

            bl_r_inf_avg_per_loc = np.array([np.nanmean(sublist) for sublist in bl_r_inf_dist_per_loc])
            bl_r_inf_std_per_loc = np.array([np.nanstd(sublist) for sublist in bl_r_inf_dist_per_loc])
            z = 1.96
            bl_moe = z * (bl_r_inf_std_per_loc / np.sqrt(len(bl_r_inf_avg_per_loc)))
            bl_r_inf_u95_per_loc = bl_r_inf_avg_per_loc + bl_moe
            bl_r_inf_l95_per_loc = bl_r_inf_avg_per_loc - bl_moe

        # Compute final observables
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_infected_fraction_per_rho_sim = bl_infected_per_rho_sim / bl_agents_per_rho_sim
        bl_invaded_fraction_per_rho_sim = bl_invaders_per_rho_sim / bl_nlocs_invaded_sim[:, np.newaxis]
    
        # Compute stats
        bl_infected_fraction_avg_per_rho = np.nanmean(bl_infected_fraction_per_rho_sim, axis=0)
        bl_infected_fraction_avg = np.nanmean(bl_infected_fraction_sim)
        bl_std = np.std(bl_infected_fraction_per_rho_sim, axis=0)
        nsims = len(bl_infected_fraction_per_rho_sim)
        z = 1.96
        bl_moe = z * (bl_std / np.sqrt(nsims))
        bl_infected_fraction_u95_per_rho = bl_infected_fraction_avg_per_rho + bl_moe
        bl_infected_fraction_l95_per_rho = bl_infected_fraction_avg_per_rho - bl_moe

        bl_invaded_fraction_avg_per_rho = np.nanmean(bl_invaded_fraction_per_rho_sim, axis=0)
        bl_std = np.std(bl_invaded_fraction_per_rho_sim, axis=0)
        nsims = len(bl_invaded_fraction_per_rho_sim)
        z = 1.96
        bl_moe = z * (std / np.sqrt(nsims))
        bl_invaded_fraction_u95_per_rho = bl_invaded_fraction_avg_per_rho + bl_moe
        bl_invaded_fraction_l95_per_rho = bl_invaded_fraction_avg_per_rho - bl_moe

        # Compute stats for collapsed observales
        bl_infected_fraction_avg = np.mean(bl_infected_fraction_sim)

        bl_r_inf_avg_dist_per_loc = [[loc_data['mean'] for loc_data in sim_data] for sim_data in bl_r_inf_stats_per_loc_sim]
        bl_avg_r_inf_avg_per_loc = np.nanmean(np.array(bl_r_inf_avg_dist_per_loc), axis=0)

        bl_total_cases_avg_loc = np.nanmean(bl_total_cases_loc_sim, axis=0)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 14))

    # SUBPLOT 0: INVADER RHO SPATIAL MAP
    im0 = ax[0].imshow(rho_avg_lattice.T, cmap='coolwarm')
    im0.set_clim(vmin=0.0, vmax=1.0)
    cbar0 = fig.colorbar(im0, ax=ax[0], shrink=0.7)
    cbar0.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

    # Settings 0
    ax[0].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    ax[0].set_ylabel("latitude (\u00b0 N)", fontsize=25)
    title = r"Invader average $\rho$ profile"
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

    # SUBPLOT 1: CASES VS. ATTRACTIVENESS REGRESSION
    if stats_flag:
        hb1 = ax[1].hexbin(attr_l, total_cases_avg_loc, C=avg_r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
    if r_inf_flag:
        hb1 = ax[1].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
    hb1.set_clim(vmin=0.0, vmax=1.0)
    cbar1 = fig.colorbar(hb1, ax=ax[1])
    cbar1.set_label(r'infector $\langle\rho\rangle$', fontsize=25)

    # Compute the mean value for each hexbin
    xbins = hb1.get_offsets()[:, 0]
    ybins = hb1.get_offsets()[:, 1]
    mean_values = hb1.get_array()
    mean_rho_for_hexbins = []

    for i in range(len(mean_values)):
        if i == len(mean_values) - 1:  # Handle the last bin separately
            condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
        else:
            condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

        indices = np.where(condition)
        if len(indices[0]) > 0:
            if stats_flag:
                mean_rho_for_hexbins.append(np.nanmean(np.array(avg_r_inf_avg_per_loc)[indices]))
            if r_inf_flag:
                mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
        else:
            mean_rho_for_hexbins.append(0.0)

    model_1 = LinearRegression()
    model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
    y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
    ax[1].plot(attr_l, y_pred_11, color='crimson', linestyle='--', linewidth=2)
    
    # Calculate and display R-squared value
    # Get R-squared value and display it
    r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
    ax[1].text(0.15, 0.75, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1].transAxes, fontsize=20, color='black')

    if bl_flag != None:
        #hb = ax[1, 2].hexbin(attr_l, total_cases_avg_loc, C=avg_r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
        #hb.set_clim(vmin=0.0, vmax=1.0)
        #cbar0 = fig.colorbar(hb, ax=ax[1, 2])
        #cbar0.set_label(r'infector $\langle\rho\rangle$', fontsize=25)

        # Compute the mean value for each hexbin
        xbins = hb1.get_offsets()[:, 0]
        ybins = hb1.get_offsets()[:, 1]
        mean_values = hb1.get_array()
        mean_rho_for_hexbins = []

        for i in range(len(mean_values)):
            if i == len(mean_values) - 1:  # Handle the last bin separately
                condition = np.logical_and(attr_l >= xbins[i], bl_total_cases_avg_loc >= ybins[i])
            else:
                condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], bl_total_cases_avg_loc >= ybins[i], bl_total_cases_avg_loc < ybins[i + 1]))

            indices = np.where(condition)
            if len(indices[0]) > 0:
                if stats_flag:
                    mean_rho_for_hexbins.append(np.nanmean(np.array(bl_avg_r_inf_avg_per_loc)[indices]))
                if r_inf_flag:
                    mean_rho_for_hexbins.append(np.nanmean(np.array(bl_r_inf_avg_per_loc)[indices]))
            else:
                mean_rho_for_hexbins.append(0.0)

        model_1 = LinearRegression()
        model_1.fit(attr_l.reshape(-1, 1), bl_total_cases_avg_loc)
        bl_y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
        ax[1].plot(attr_l, bl_y_pred_11, color='navy', linestyle='--', linewidth=2)
        # Calculate and display R-squared value
        # Get R-squared value and display it
        r2_1 = model_1.score(attr_l.reshape(-1, 1), bl_total_cases_avg_loc)
        ax[1].text(0.15, 0.95, r'bl $R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1].transAxes, fontsize=20, color='black')

    ax[1].set_xlabel(r'$A$', fontsize=25)
    ax[1].set_ylabel('mean total cases', fontsize=25)
    ax[1].set_title('', fontsize=30)
    ax[1].tick_params(axis='both', labelsize=15)

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
    base_name = 'chf2_' + epi_filename
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_figure3(
        focal_dist, 
        comp_flag=None, 
        bl_flag=None, 
        stats_flag=False, 
        f_inf_flag=False, 
        a_inf_flag=False,
        ):
    """ 1x3 panel
    f0: home-outside infection profile
    f1: home-outside frequency profile
    f2: home-outside attractiveness profile
    
    """

    # Load space parameters from space retriever json file
    lower_path = 'config/'
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)

    # Collect all digested epidemic file names
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    # Load space data
    lower_path = 'data/'
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    prevalence_cutoff = 0.05

    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    agents_per_rho_sim = []
    infected_per_rho_sim = []
    infected_h_per_rho_sim = []
    infected_o_per_rho_sim = []
    f_inf_h_stats_per_rho_sim = []
    f_inf_o_stats_per_rho_sim = []
    a_inf_h_stats_per_rho_sim = []
    a_inf_o_stats_per_rho_sim = []
    f_inf_h_dist_per_rho_sim = []
    f_inf_o_dist_per_rho_sim = []
    a_inf_h_dist_per_rho_sim = []
    a_inf_o_dist_per_rho_sim = []

    comp_agents_per_rho_sim = []
    comp_infected_per_rho_sim = []
    comp_infected_h_per_rho_sim = []
    comp_infected_o_per_rho_sim = []
    comp_f_inf_h_stats_per_rho_sim = []
    comp_f_inf_o_stats_per_rho_sim = []
    comp_a_inf_h_stats_per_rho_sim = []
    comp_a_inf_o_stats_per_rho_sim = []
    comp_f_inf_h_dist_per_rho_sim = []
    comp_f_inf_o_dist_per_rho_sim = []
    comp_a_inf_h_dist_per_rho_sim = []
    comp_a_inf_o_dist_per_rho_sim = []

    bl_agents_per_rho_sim = []
    bl_infected_per_rho_sim = []
    bl_infected_h_per_rho_sim = []
    bl_infected_o_per_rho_sim = []
    bl_f_inf_h_stats_per_rho_sim = []
    bl_f_inf_o_stats_per_rho_sim = []
    bl_a_inf_h_stats_per_rho_sim = []
    bl_a_inf_o_stats_per_rho_sim = []
    bl_f_inf_h_dist_per_rho_sim = []
    bl_f_inf_o_dist_per_rho_sim = []
    bl_a_inf_h_dist_per_rho_sim = []
    bl_a_inf_o_dist_per_rho_sim = []

    # Loop over the collected file names
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))
        # Build the full path
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Build fullname
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Check if the file exists
        if os.path.exists(epi_fullname):
            # Load digested epidemic data
            out_sim_data = ut.load_chapter_figure3_data(
                epi_fullname,
                stats_flag=stats_flag, 
                f_inf_flag=f_inf_flag, 
                a_inf_flag=a_inf_flag,
                )
            
            if focal_dist in epi_filename and bl_flag in epi_filename and 'depr' not in epi_filename:
                # Collect data from every realization in structure
                bl_agents_per_rho_sim.extend(out_sim_data['agents'])
                bl_infected_per_rho_sim.extend(out_sim_data['infected'])
                bl_infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
                bl_infected_o_per_rho_sim.extend(out_sim_data['infected_o'])
                
                if stats_flag:
                    bl_f_inf_h_stats_per_rho_sim.extend(out_sim_data['f_inf_h_stats'])
                    bl_f_inf_o_stats_per_rho_sim.extend(out_sim_data['f_inf_o_stats'])
                    bl_a_inf_h_stats_per_rho_sim.extend(out_sim_data['a_inf_h_stats'])
                    bl_a_inf_o_stats_per_rho_sim.extend(out_sim_data['a_inf_o_stats'])
                if f_inf_flag:
                    bl_f_inf_h_dist_per_rho_sim.extend(out_sim_data['f_inf_h_dist'])
                    bl_f_inf_o_dist_per_rho_sim.extend(out_sim_data['f_inf_o_dist'])
                if a_inf_flag:
                    bl_a_inf_h_dist_per_rho_sim.extend(out_sim_data['a_inf_h_dist'])
                    bl_a_inf_o_dist_per_rho_sim.extend(out_sim_data['a_inf_o_dist'])
            
            elif comp_flag is not None and comp_flag in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                comp_agents_per_rho_sim.extend(out_sim_data['agents'])
                comp_infected_per_rho_sim.extend(out_sim_data['infected'])
                comp_infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
                comp_infected_o_per_rho_sim.extend(out_sim_data['infected_o'])
                if stats_flag:
                    comp_f_inf_h_stats_per_rho_sim.extend(out_sim_data['f_inf_h_stats'])
                    comp_f_inf_o_stats_per_rho_sim.extend(out_sim_data['f_inf_o_stats'])
                    comp_a_inf_h_stats_per_rho_sim.extend(out_sim_data['a_inf_h_stats'])
                    comp_a_inf_o_stats_per_rho_sim.extend(out_sim_data['a_inf_o_stats'])
                if f_inf_flag:
                    comp_f_inf_h_dist_per_rho_sim.extend(out_sim_data['f_inf_h_dist'])
                    comp_f_inf_o_dist_per_rho_sim.extend(out_sim_data['f_inf_o_dist'])
                if a_inf_flag:
                    comp_a_inf_h_dist_per_rho_sim.extend(out_sim_data['a_inf_h_dist'])
                    comp_a_inf_o_dist_per_rho_sim.extend(out_sim_data['a_inf_o_dist'])
            
            elif focal_dist in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                agents_per_rho_sim.extend(out_sim_data['agents'])
                infected_per_rho_sim.extend(out_sim_data['infected'])
                infected_h_per_rho_sim.extend(out_sim_data['infected_h'])
                infected_o_per_rho_sim.extend(out_sim_data['infected_o'])
            
                if stats_flag:
                    f_inf_h_stats_per_rho_sim.extend(out_sim_data['f_inf_h_stats'])
                    f_inf_o_stats_per_rho_sim.extend(out_sim_data['f_inf_o_stats'])
                    a_inf_h_stats_per_rho_sim.extend(out_sim_data['a_inf_h_stats'])
                    a_inf_o_stats_per_rho_sim.extend(out_sim_data['a_inf_o_stats'])
                if f_inf_flag:
                    f_inf_h_dist_per_rho_sim.extend(out_sim_data['f_inf_h_dist'])
                    f_inf_o_dist_per_rho_sim.extend(out_sim_data['f_inf_o_dist'])
                if a_inf_flag:
                    a_inf_h_dist_per_rho_sim.extend(out_sim_data['a_inf_h_dist'])
                    a_inf_o_dist_per_rho_sim.extend(out_sim_data['a_inf_o_dist'])

        else:
            # File doesn't exist, skip the rest of the loop
            print(f"File {epi_fullname} does not exist. Skipping this iteration.")
            continue

    if focal_dist != None:

        # Convert into numpy arrays
        agents_per_rho_sim = np.array(agents_per_rho_sim)
        infected_per_rho_sim = np.array(infected_per_rho_sim)
        infected_h_per_rho_sim = np.array(infected_h_per_rho_sim)
        infected_o_per_rho_sim = np.array(infected_o_per_rho_sim)

        # Filter failed outbreaks
        infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
        failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

        agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
        infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
        infected_h_per_rho_sim = np.delete(infected_h_per_rho_sim, failed_outbreaks, axis=0)
        infected_o_per_rho_sim = np.delete(infected_o_per_rho_sim, failed_outbreaks, axis=0)

        if stats_flag:
            f_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(f_inf_h_stats_per_rho_sim) if i not in failed_outbreaks]
            f_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(f_inf_o_stats_per_rho_sim) if i not in failed_outbreaks]
            a_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(a_inf_h_stats_per_rho_sim) if i not in failed_outbreaks]
            a_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(a_inf_o_stats_per_rho_sim) if i not in failed_outbreaks]

            # Compute stats
            f_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in f_inf_h_stats_per_rho_sim]
            avg_f_inf_h_avg_per_rho = np.mean(np.array(f_inf_h_avg_dist_per_rho), axis=0)
            f_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in f_inf_h_stats_per_rho_sim]
            u95_f_inf_h_avg_per_rho = np.mean(np.array(f_inf_h_u95_dist_per_rho), axis=0)
            f_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in f_inf_h_stats_per_rho_sim]
            l95_f_inf_h_avg_per_rho = np.mean(np.array(f_inf_h_l95_dist_per_rho), axis=0)

            f_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in f_inf_o_stats_per_rho_sim]
            avg_f_inf_o_avg_per_rho = np.mean(np.array(f_inf_o_avg_dist_per_rho), axis=0)
            f_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in f_inf_o_stats_per_rho_sim]
            u95_f_inf_o_avg_per_rho = np.mean(np.array(f_inf_o_u95_dist_per_rho), axis=0)
            f_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in f_inf_o_stats_per_rho_sim]
            l95_f_inf_o_avg_per_rho = np.mean(np.array(f_inf_o_l95_dist_per_rho), axis=0)

            a_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in a_inf_h_stats_per_rho_sim]
            avg_a_inf_h_avg_per_rho = np.mean(np.array(a_inf_h_avg_dist_per_rho), axis=0)
            a_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in a_inf_h_stats_per_rho_sim]
            u95_a_inf_h_avg_per_rho = np.mean(np.array(a_inf_h_u95_dist_per_rho), axis=0)
            a_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in a_inf_h_stats_per_rho_sim]
            l95_a_inf_h_avg_per_rho = np.mean(np.array(a_inf_h_l95_dist_per_rho), axis=0)

            a_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in a_inf_o_stats_per_rho_sim]
            avg_a_inf_o_avg_per_rho = np.mean(np.array(a_inf_o_avg_dist_per_rho), axis=0)
            a_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in a_inf_o_stats_per_rho_sim]
            u95_a_inf_o_avg_per_rho = np.mean(np.array(a_inf_o_u95_dist_per_rho), axis=0)
            a_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in a_inf_o_stats_per_rho_sim]
            l95_a_inf_o_avg_per_rho = np.mean(np.array(a_inf_o_l95_dist_per_rho), axis=0)

        if f_inf_flag:
            f_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_h_dist_per_rho_sim) if i not in failed_outbreaks]
            nbins = len(f_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            f_inf_h_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(f_inf_h_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    f_inf_h_values = f_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    f_inf_h_dist_per_rho[rho_idx].extend(f_inf_h_values)

            f_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_h_dist_per_rho])
            f_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_h_dist_per_rho])
            z = 1.96
            moe = z * (f_inf_h_std_per_rho / np.sqrt(len(f_inf_h_avg_per_rho)))
            f_inf_h_u95_per_rho = f_inf_h_avg_per_rho + moe
            f_inf_h_l95_per_rho = f_inf_h_avg_per_rho - moe

            f_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(f_inf_o_dist_per_rho_sim) if i not in failed_outbreaks]
            nbins = len(f_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            f_inf_o_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(f_inf_o_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    f_inf_o_values = f_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    f_inf_o_dist_per_rho[rho_idx].extend(f_inf_o_values)

            f_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in f_inf_o_dist_per_rho])
            f_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in f_inf_o_dist_per_rho])
            z = 1.96
            moe = z * (f_inf_o_std_per_rho / np.sqrt(len(f_inf_o_avg_per_rho)))
            f_inf_o_u95_per_rho = f_inf_o_avg_per_rho + moe
            f_inf_o_l95_per_rho = f_inf_o_avg_per_rho - moe

        if a_inf_flag:
            a_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(a_inf_h_dist_per_rho_sim) if i not in failed_outbreaks]
            nbins = len(f_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            a_inf_h_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(a_inf_h_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    a_inf_h_values = a_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    a_inf_h_dist_per_rho[rho_idx].extend(a_inf_h_values)

            a_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in a_inf_h_dist_per_rho])
            a_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in a_inf_h_dist_per_rho])
            z = 1.96
            moe = z * (a_inf_h_std_per_rho / np.sqrt(len(a_inf_h_avg_per_rho)))
            a_inf_h_u95_per_rho = a_inf_h_avg_per_rho + moe
            a_inf_h_l95_per_rho = a_inf_h_avg_per_rho - moe

            a_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(a_inf_o_dist_per_rho_sim) if i not in failed_outbreaks]
            nbins = len(a_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            a_inf_o_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(a_inf_o_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    a_inf_o_values = a_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    a_inf_o_dist_per_rho[rho_idx].extend(a_inf_o_values)

            a_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in a_inf_o_dist_per_rho])
            a_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in a_inf_o_dist_per_rho])
            z = 1.96
            moe = z * (a_inf_o_std_per_rho / np.sqrt(len(a_inf_o_avg_per_rho)))
            a_inf_o_u95_per_rho = a_inf_o_avg_per_rho + moe
            a_inf_o_l95_per_rho = a_inf_o_avg_per_rho - moe

        # Compute final observables
        infected_fraction_per_rho_sim = infected_per_rho_sim / agents_per_rho_sim
        infected_h_fraction_per_rho_sim = infected_h_per_rho_sim / infected_per_rho_sim
        infected_o_fraction_per_rho_sim = infected_o_per_rho_sim / infected_per_rho_sim

        # Compute stats
        infected_h_fraction_avg_per_rho = np.mean(infected_h_fraction_per_rho_sim, axis=0)
        inf_h_frac_std_per_rho = np.std(infected_h_fraction_per_rho_sim, axis=0)
        z = 1.96
        moe = z * (inf_h_frac_std_per_rho / np.sqrt(len(infected_h_fraction_per_rho_sim)))
        infected_h_fraction_u95_per_rho = infected_h_fraction_avg_per_rho + moe
        infected_h_fraction_l95_per_rho = infected_h_fraction_avg_per_rho - moe

        infected_o_fraction_avg_per_rho = np.mean(infected_o_fraction_per_rho_sim, axis=0)
        inf_o_frac_std_per_rho = np.std(infected_o_fraction_per_rho_sim, axis=0)
        moe = z * (inf_o_frac_std_per_rho / np.sqrt(len(infected_o_fraction_per_rho_sim)))
        infected_o_fraction_u95_per_rho = infected_o_fraction_avg_per_rho + moe
        infected_o_fraction_l95_per_rho = infected_o_fraction_avg_per_rho - moe

    if comp_flag != None:
        # Convert into numpy arrays
        comp_agents_per_rho_sim = np.array(comp_agents_per_rho_sim)
        comp_infected_per_rho_sim = np.array(comp_infected_per_rho_sim)
        comp_infected_h_per_rho_sim = np.array(comp_infected_h_per_rho_sim)
        comp_infected_o_per_rho_sim = np.array(comp_infected_o_per_rho_sim)

        # Filter failed outbreaks
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_failed_outbreaks = np.where(comp_infected_fraction_sim < prevalence_cutoff)[0]

        comp_agents_per_rho_sim = np.delete(comp_agents_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_infected_per_rho_sim = np.delete(comp_infected_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_infected_h_per_rho_sim = np.delete(comp_infected_h_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_infected_o_per_rho_sim = np.delete(comp_infected_o_per_rho_sim, comp_failed_outbreaks, axis=0)
        
        if stats_flag:
            comp_f_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(comp_f_inf_h_stats_per_rho_sim) if i not in comp_failed_outbreaks]
            comp_f_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(comp_f_inf_o_stats_per_rho_sim) if i not in comp_failed_outbreaks]
            comp_a_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(comp_a_inf_h_stats_per_rho_sim) if i not in comp_failed_outbreaks]
            comp_a_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(comp_a_inf_o_stats_per_rho_sim) if i not in comp_failed_outbreaks]

            # Compute stats
            comp_f_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in comp_f_inf_h_stats_per_rho_sim]
            comp_avg_f_inf_h_avg_per_rho = np.mean(np.array(comp_f_inf_h_avg_dist_per_rho), axis=0)
            comp_f_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in comp_f_inf_h_stats_per_rho_sim]
            comp_u95_f_inf_h_avg_per_rho = np.mean(np.array(comp_f_inf_h_u95_dist_per_rho), axis=0)
            comp_f_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in comp_f_inf_h_stats_per_rho_sim]
            comp_l95_f_inf_h_avg_per_rho = np.mean(np.array(comp_f_inf_h_l95_dist_per_rho), axis=0)

            comp_f_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in comp_f_inf_o_stats_per_rho_sim]
            comp_avg_f_inf_o_avg_per_rho = np.mean(np.array(comp_f_inf_o_avg_dist_per_rho), axis=0)
            comp_f_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in comp_f_inf_o_stats_per_rho_sim]
            comp_u95_f_inf_o_avg_per_rho = np.mean(np.array(comp_f_inf_o_u95_dist_per_rho), axis=0)
            comp_f_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in comp_f_inf_o_stats_per_rho_sim]
            comp_l95_f_inf_o_avg_per_rho = np.mean(np.array(comp_f_inf_o_l95_dist_per_rho), axis=0)

            comp_a_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in comp_a_inf_h_stats_per_rho_sim]
            comp_avg_a_inf_h_avg_per_rho = np.mean(np.array(comp_a_inf_h_avg_dist_per_rho), axis=0)
            comp_a_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in comp_a_inf_h_stats_per_rho_sim]
            comp_u95_a_inf_h_avg_per_rho = np.mean(np.array(comp_a_inf_h_u95_dist_per_rho), axis=0)
            comp_a_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in comp_a_inf_h_stats_per_rho_sim]
            comp_l95_a_inf_h_avg_per_rho = np.mean(np.array(comp_a_inf_h_l95_dist_per_rho), axis=0)

            comp_a_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in comp_a_inf_o_stats_per_rho_sim]
            comp_avg_a_inf_o_avg_per_rho = np.mean(np.array(comp_a_inf_o_avg_dist_per_rho), axis=0)
            comp_a_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in comp_a_inf_o_stats_per_rho_sim]
            comp_u95_a_inf_o_avg_per_rho = np.mean(np.array(comp_a_inf_o_u95_dist_per_rho), axis=0)
            comp_a_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in comp_a_inf_o_stats_per_rho_sim]
            comp_l95_a_inf_o_avg_per_rho = np.mean(np.array(comp_a_inf_o_l95_dist_per_rho), axis=0)

        if f_inf_flag:
            comp_f_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(comp_f_inf_h_dist_per_rho_sim) if i not in comp_failed_outbreaks]
            nbins = len(comp_f_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            comp_f_inf_h_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_f_inf_h_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_f_inf_h_values = comp_f_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_f_inf_h_dist_per_rho[rho_idx].extend(comp_f_inf_h_values)

            comp_f_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in comp_f_inf_h_dist_per_rho])
            comp_f_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in comp_f_inf_h_dist_per_rho])
            z = 1.96
            comp_moe = z * (comp_f_inf_h_std_per_rho / np.sqrt(len(comp_f_inf_h_avg_per_rho)))
            comp_f_inf_h_u95_per_rho = comp_f_inf_h_avg_per_rho + comp_moe
            comp_f_inf_h_l95_per_rho = comp_f_inf_h_avg_per_rho - comp_moe

            comp_f_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(comp_f_inf_o_dist_per_rho_sim) if i not in comp_failed_outbreaks]
            nbins = len(comp_f_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            comp_f_inf_o_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_f_inf_o_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_f_inf_o_values = comp_f_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_f_inf_o_dist_per_rho[rho_idx].extend(comp_f_inf_o_values)

            comp_f_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in comp_f_inf_o_dist_per_rho])
            comp_f_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in comp_f_inf_o_dist_per_rho])
            z = 1.96
            comp_moe = z * (comp_f_inf_o_std_per_rho / np.sqrt(len(comp_f_inf_o_avg_per_rho)))
            comp_f_inf_o_u95_per_rho = comp_f_inf_o_avg_per_rho + comp_moe
            comp_f_inf_o_l95_per_rho = comp_f_inf_o_avg_per_rho - comp_moe
    
        if a_inf_flag:
            comp_a_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(comp_a_inf_h_dist_per_rho_sim) if i not in comp_failed_outbreaks]
            nbins = len(comp_a_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            comp_a_inf_h_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_a_inf_h_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_a_inf_h_values = comp_a_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_a_inf_h_dist_per_rho[rho_idx].extend(comp_a_inf_h_values)

            comp_a_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in comp_a_inf_h_dist_per_rho])
            comp_a_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in comp_a_inf_h_dist_per_rho])
            z = 1.96
            comp_moe = z * (comp_a_inf_h_std_per_rho / np.sqrt(len(comp_a_inf_h_avg_per_rho)))
            comp_a_inf_h_u95_per_rho = comp_a_inf_h_avg_per_rho + comp_moe
            comp_a_inf_h_l95_per_rho = comp_a_inf_h_avg_per_rho - comp_moe

            comp_a_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(comp_a_inf_o_dist_per_rho_sim) if i not in comp_failed_outbreaks]
            nbins = len(comp_a_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            comp_a_inf_o_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(comp_a_inf_o_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    comp_a_inf_o_values = comp_a_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    comp_a_inf_o_dist_per_rho[rho_idx].extend(comp_a_inf_o_values)

            comp_a_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in comp_a_inf_o_dist_per_rho])
            comp_a_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in comp_a_inf_o_dist_per_rho])
            z = 1.96
            comp_moe = z * (comp_a_inf_o_std_per_rho / np.sqrt(len(comp_a_inf_o_avg_per_rho)))
            comp_a_inf_o_u95_per_rho = comp_a_inf_o_avg_per_rho + moe
            comp_a_inf_o_l95_per_rho = comp_a_inf_o_avg_per_rho - moe

        # Compute final observables
        comp_infected_fraction_per_rho_sim = comp_infected_per_rho_sim / comp_agents_per_rho_sim
        comp_infected_h_fraction_per_rho_sim = comp_infected_h_per_rho_sim / comp_infected_per_rho_sim
        comp_infected_o_fraction_per_rho_sim = comp_infected_o_per_rho_sim / comp_infected_per_rho_sim

        # Compute stats
        comp_infected_h_fraction_avg_per_rho = np.mean(comp_infected_h_fraction_per_rho_sim, axis=0)
        comp_infected_o_fraction_avg_per_rho = np.mean(comp_infected_o_fraction_per_rho_sim, axis=0)
    
    if bl_flag != None:
        # Convert into numpy arrays
        bl_agents_per_rho_sim = np.array(bl_agents_per_rho_sim)
        bl_infected_per_rho_sim = np.array(bl_infected_per_rho_sim)
        bl_infected_h_per_rho_sim = np.array(bl_infected_h_per_rho_sim)
        bl_infected_o_per_rho_sim = np.array(bl_infected_o_per_rho_sim)

        # Filter failed outbreaks
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_failed_outbreaks = np.where(bl_infected_fraction_sim < prevalence_cutoff)[0]

        bl_agents_per_rho_sim = np.delete(bl_agents_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_infected_per_rho_sim = np.delete(bl_infected_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_infected_h_per_rho_sim = np.delete(bl_infected_h_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_infected_o_per_rho_sim = np.delete(bl_infected_o_per_rho_sim, bl_failed_outbreaks, axis=0)

        if stats_flag:
            bl_f_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(bl_f_inf_h_stats_per_rho_sim) if i not in bl_failed_outbreaks]
            bl_f_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(bl_f_inf_o_stats_per_rho_sim) if i not in bl_failed_outbreaks]
            bl_a_inf_h_stats_per_rho_sim = [sim for i, sim in enumerate(bl_a_inf_h_stats_per_rho_sim) if i not in bl_failed_outbreaks]
            bl_a_inf_o_stats_per_rho_sim = [sim for i, sim in enumerate(bl_a_inf_o_stats_per_rho_sim) if i not in bl_failed_outbreaks]
    
            # Compute stats
            bl_f_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in bl_f_inf_h_stats_per_rho_sim]
            bl_avg_f_inf_h_avg_per_rho = np.mean(np.array(bl_f_inf_h_avg_dist_per_rho), axis=0)
            bl_f_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in bl_f_inf_h_stats_per_rho_sim]
            bl_u95_f_inf_h_avg_per_rho = np.mean(np.array(bl_f_inf_h_u95_dist_per_rho), axis=0)
            bl_f_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in bl_f_inf_h_stats_per_rho_sim]
            bl_l95_f_inf_h_avg_per_rho = np.mean(np.array(bl_f_inf_h_l95_dist_per_rho), axis=0)

            bl_f_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in bl_f_inf_o_stats_per_rho_sim]
            bl_avg_f_inf_o_avg_per_rho = np.mean(np.array(bl_f_inf_o_avg_dist_per_rho), axis=0)
            bl_f_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in bl_f_inf_o_stats_per_rho_sim]
            bl_u95_f_inf_o_avg_per_rho = np.mean(np.array(bl_f_inf_o_u95_dist_per_rho), axis=0)
            bl_f_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in bl_f_inf_o_stats_per_rho_sim]
            bl_l95_f_inf_o_avg_per_rho = np.mean(np.array(bl_f_inf_o_l95_dist_per_rho), axis=0)

            bl_a_inf_h_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in bl_a_inf_h_stats_per_rho_sim]
            bl_avg_a_inf_h_avg_per_rho = np.mean(np.array(bl_a_inf_h_avg_dist_per_rho), axis=0)
            bl_a_inf_h_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in bl_a_inf_h_stats_per_rho_sim]
            bl_u95_a_inf_h_avg_per_rho = np.mean(np.array(bl_a_inf_h_u95_dist_per_rho), axis=0)
            bl_a_inf_h_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in bl_a_inf_h_stats_per_rho_sim]
            bl_l95_a_inf_h_avg_per_rho = np.mean(np.array(bl_a_inf_h_l95_dist_per_rho), axis=0)

            bl_a_inf_o_avg_dist_per_rho = [[rho_data['mean'] for rho_data in sim_data] for sim_data in bl_a_inf_o_stats_per_rho_sim]
            bl_avg_a_inf_o_avg_per_rho = np.mean(np.array(bl_a_inf_o_avg_dist_per_rho), axis=0)
            bl_a_inf_o_u95_dist_per_rho = [[rho_data['u95'] for rho_data in sim_data] for sim_data in bl_a_inf_o_stats_per_rho_sim]
            bl_u95_a_inf_o_avg_per_rho = np.mean(np.array(bl_a_inf_o_u95_dist_per_rho), axis=0)
            bl_a_inf_o_l95_dist_per_rho = [[rho_data['l95'] for rho_data in sim_data] for sim_data in bl_a_inf_o_stats_per_rho_sim]
            bl_l95_a_inf_o_avg_per_rho = np.mean(np.array(bl_a_inf_o_l95_dist_per_rho), axis=0)

        if f_inf_flag:
            bl_f_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(bl_f_inf_h_dist_per_rho_sim) if i not in bl_failed_outbreaks]
            nbins = len(bl_f_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            bl_f_inf_h_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_f_inf_h_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_f_inf_h_values = bl_f_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_f_inf_h_dist_per_rho[rho_idx].extend(bl_f_inf_h_values)

            bl_f_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in bl_f_inf_h_dist_per_rho])
            bl_f_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in bl_f_inf_h_dist_per_rho])
            z = 1.96
            bl_moe = z * (bl_f_inf_h_std_per_rho / np.sqrt(len(bl_f_inf_h_avg_per_rho)))
            bl_f_inf_h_u95_per_rho = bl_f_inf_h_avg_per_rho + bl_moe
            bl_f_inf_h_l95_per_rho = bl_f_inf_h_avg_per_rho - bl_moe

            bl_f_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(bl_f_inf_o_dist_per_rho_sim) if i not in bl_failed_outbreaks]
            nbins = len(bl_f_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            bl_f_inf_o_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_f_inf_o_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_f_inf_o_values = bl_f_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_f_inf_o_dist_per_rho[rho_idx].extend(bl_f_inf_o_values)

            bl_f_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in bl_f_inf_o_dist_per_rho])
            bl_f_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in bl_f_inf_o_dist_per_rho])
            z = 1.96
            bl_moe = z * (bl_f_inf_o_std_per_rho / np.sqrt(len(bl_f_inf_o_avg_per_rho)))
            bl_f_inf_o_u95_per_rho = bl_f_inf_o_avg_per_rho + bl_moe
            bl_f_inf_o_l95_per_rho = bl_f_inf_o_avg_per_rho - bl_moe
    
        if a_inf_flag:
            bl_a_inf_h_dist_per_rho_sim = [sim for i, sim in enumerate(bl_a_inf_h_dist_per_rho_sim) if i not in bl_failed_outbreaks]
            nbins = len(bl_a_inf_h_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            bl_a_inf_h_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_a_inf_h_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_a_inf_h_values = bl_a_inf_h_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_a_inf_h_dist_per_rho[rho_idx].extend(bl_a_inf_h_values)

            bl_a_inf_h_avg_per_rho = np.array([np.nanmean(sublist) for sublist in bl_a_inf_h_dist_per_rho])
            bl_a_inf_h_std_per_rho = np.array([np.nanstd(sublist) for sublist in bl_a_inf_h_dist_per_rho])
            z = 1.96
            bl_moe = z * (bl_a_inf_h_std_per_rho / np.sqrt(len(bl_a_inf_h_avg_per_rho)))
            bl_a_inf_h_u95_per_rho = bl_a_inf_h_avg_per_rho + bl_moe
            bl_a_inf_h_l95_per_rho = bl_a_inf_h_avg_per_rho - bl_moe

            bl_a_inf_o_dist_per_rho_sim = [sim for i, sim in enumerate(bl_a_inf_o_dist_per_rho_sim) if i not in bl_failed_outbreaks]
            nbins = len(bl_a_inf_o_dist_per_rho_sim[0])  # Assuming all inner lists have the same size
            bl_a_inf_o_dist_per_rho = [[] for _ in range(nbins)]

            # Iterate through simulations and rhos to aggregate t_inv values
            for sim_idx in range(len(bl_a_inf_o_dist_per_rho_sim)):
                for rho_idx in range(nbins):
                    # Extract t_inv values for sim_idx and rho_idx
                    bl_a_inf_o_values = bl_a_inf_o_dist_per_rho_sim[sim_idx][rho_idx]
                    # Append t_inv values to the corresponding element in t_inv_dist_per_rho
                    bl_a_inf_o_dist_per_rho[rho_idx].extend(bl_a_inf_o_values)

            bl_a_inf_o_avg_per_rho = np.array([np.nanmean(sublist) for sublist in bl_a_inf_o_dist_per_rho])
            bl_a_inf_o_std_per_rho = np.array([np.nanstd(sublist) for sublist in bl_a_inf_o_dist_per_rho])
            z = 1.96
            bl_moe = z * (bl_a_inf_o_std_per_rho / np.sqrt(len(bl_a_inf_o_avg_per_rho)))
            bl_a_inf_o_u95_per_rho = bl_a_inf_o_avg_per_rho + bl_moe
            bl_a_inf_o_l95_per_rho = bl_a_inf_o_avg_per_rho - bl_moe

        # Compute final observables
        bl_infected_fraction_per_rho_sim = bl_infected_per_rho_sim / bl_agents_per_rho_sim
        bl_infected_h_fraction_per_rho_sim = bl_infected_h_per_rho_sim / bl_infected_per_rho_sim
        bl_infected_o_fraction_per_rho_sim = bl_infected_o_per_rho_sim / bl_infected_per_rho_sim

        bl_infected_h_fraction_sim = np.sum(bl_infected_h_per_rho_sim, axis=1) / np.sum(bl_infected_per_rho_sim, axis=1)
        bl_infected_o_fraction_sim = np.sum(bl_infected_o_per_rho_sim, axis=1) / np.sum(bl_infected_per_rho_sim, axis=1)

        # Compute stats
        bl_infected_h_fraction_avg_per_rho = np.mean(bl_infected_h_fraction_per_rho_sim, axis=0)
        bl_infected_o_fraction_avg_per_rho = np.mean(bl_infected_o_fraction_per_rho_sim, axis=0)
        bl_infected_h_fraction_avg = np.mean(bl_infected_h_fraction_sim)
        bl_infected_o_fraction_avg = np.mean(bl_infected_o_fraction_sim)
            
    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 12))

    # SUBPLOT 0
    if focal_dist != None:
        ax[0].scatter(rho_bins, infected_h_fraction_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
        ax[0].fill_between(rho_bins, infected_h_fraction_l95_per_rho, infected_h_fraction_u95_per_rho, color='dodgerblue', alpha=0.2)
        ax[0].scatter(rho_bins, infected_o_fraction_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
        ax[0].fill_between(rho_bins, infected_o_fraction_l95_per_rho, infected_o_fraction_u95_per_rho, color='firebrick', alpha=0.2)

        total = np.sum(infected_h_per_rho_sim) + np.sum(infected_o_per_rho_sim)
        home_fraction = np.sum(infected_h_per_rho_sim) / total
        out_fraction = np.sum(infected_o_per_rho_sim) / total
        ax[0].axhline(home_fraction, color='dodgerblue', linestyle='--', label='global home')
        ax[0].axhline(out_fraction, color='firebrick', linestyle='--', label='global out')

    if comp_flag != None:
        ax[0].scatter(rho_bins, comp_infected_h_fraction_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
        ax[0].scatter(rho_bins, comp_infected_o_fraction_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
        comp_total = np.sum(comp_infected_h_per_rho_sim) + np.sum(comp_infected_o_per_rho_sim)
        comp_home_fraction = np.sum(comp_infected_h_per_rho_sim) / comp_total
        comp_out_fraction = np.sum(comp_infected_o_per_rho_sim) / comp_total
        ax[0].axhline(comp_home_fraction, color='dodgerblue', linestyle='--', label='global home')
        ax[0].axhline(comp_out_fraction, color='firebrick', linestyle='--', label='global out')
    
    if bl_flag != None:
        #ax[0].axhline(bl_infected_h_fraction_avg, color='steelblue', linestyle='--', label='bl1 home')
        ax[0].axhline(bl_infected_o_fraction_avg, color='darkred', linestyle='--', label='bl1 out')

    title = 'event triggered where?'
    ax[0].set_title(title, fontsize=30)
    ax[0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0].set_ylabel(r'share of events at home (outside)', fontsize=25)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(fontsize=20)

    # SUBPLOT 1 
    if focal_dist != None:
        if stats_flag:
            ax[1].scatter(rho_bins, avg_f_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[1].scatter(rho_bins, avg_f_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
        if f_inf_flag:
            ax[1].scatter(rho_bins, f_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[1].fill_between(rho_bins, f_inf_h_l95_per_rho, f_inf_h_u95_per_rho, color='dodgerblue', alpha=0.2)
            ax[1].scatter(rho_bins, f_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[1].fill_between(rho_bins, f_inf_o_l95_per_rho, f_inf_o_u95_per_rho, color='firebrick', alpha=0.2)

    if comp_flag != None:
        if stats_flag:
            ax[1].scatter(rho_bins, comp_avg_f_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[1].fill_between(rho_bins, comp_l95_f_inf_h_avg_per_rho, comp_u95_f_inf_h_avg_per_rho, color='dodgerblue', alpha=0.2)
            ax[1].scatter(rho_bins, comp_avg_f_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[1].fill_between(rho_bins, comp_l95_f_inf_o_avg_per_rho, comp_u95_f_inf_o_avg_per_rho, color='firebrick', alpha=0.2)
        if f_inf_flag:
            ax[1].scatter(rho_bins, comp_f_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[1].fill_between(rho_bins, comp_f_inf_h_l95_per_rho, comp_f_inf_h_u95_per_rho, color='dodgerblue', alpha=0.2)
            ax[1].scatter(rho_bins, comp_f_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[1].fill_between(rho_bins, comp_f_inf_o_l95_per_rho, comp_f_inf_o_u95_per_rho, color='firebrick', alpha=0.2)

    if bl_flag != None:
        #ax[1].scatter(rho_bins, bl_avg_f_inf_h_avg_per_rho, marker='o', color='steelblue', label=r'home mean')
        if stats_flag:
            ax[1].scatter(rho_bins, bl_avg_f_inf_o_avg_per_rho, marker='_', color='darkred', label=r'bl out mean')
        if f_inf_flag:
            ax[1].scatter(rho_bins, bl_f_inf_o_avg_per_rho, marker='_', color='darkred', label=r'bl out mean')

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

    if focal_dist != None:
        if stats_flag:
            ax[2].scatter(rho_bins, avg_a_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[2].fill_between(rho_bins, l95_a_inf_h_avg_per_rho, u95_a_inf_h_avg_per_rho, color='dodgerblue', alpha=0.2)
            ax[2].scatter(rho_bins, avg_a_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[2].fill_between(rho_bins, l95_a_inf_o_avg_per_rho, u95_a_inf_o_avg_per_rho, color='firebrick', alpha=0.2)

        if a_inf_flag:
            ax[2].scatter(rho_bins, a_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[2].fill_between(rho_bins, a_inf_h_l95_per_rho, a_inf_h_u95_per_rho, color='dodgerblue', alpha=0.2)
            ax[2].scatter(rho_bins, a_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[2].fill_between(rho_bins, a_inf_o_l95_per_rho, a_inf_o_u95_per_rho, color='firebrick', alpha=0.2)

    if comp_flag != None:
        if stats_flag:
            ax[2].scatter(rho_bins, comp_avg_a_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[2].fill_between(rho_bins, comp_l95_a_inf_h_avg_per_rho, comp_u95_a_inf_h_avg_per_rho, color='dodgerblue', alpha=0.2)
            ax[2].scatter(rho_bins, comp_avg_a_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[2].fill_between(rho_bins, comp_l95_a_inf_o_avg_per_rho, comp_u95_a_inf_o_avg_per_rho, color='firebrick', alpha=0.2)

        if a_inf_flag:
            ax[2].scatter(rho_bins, comp_a_inf_h_avg_per_rho, marker='o', color='dodgerblue', label=r'home mean')
            ax[2].fill_between(rho_bins, comp_a_inf_h_l95_per_rho, comp_a_inf_h_u95_per_rho, color='dodgerblue', alpha=0.2)
            ax[2].scatter(rho_bins, comp_a_inf_o_avg_per_rho, marker='o', color='firebrick', label=r'out mean')
            ax[2].fill_between(rho_bins, comp_a_inf_o_l95_per_rho, comp_a_inf_o_u95_per_rho, color='firebrick', alpha=0.2)

    if bl_flag != None:
        #ax[2].scatter(rho_bins, bl_avg_a_inf_h_avg_per_rho, marker='o', color='steelblue', label=r'home mean')
        if stats_flag:
            ax[2].scatter(rho_bins, bl_avg_a_inf_o_avg_per_rho, marker='_', color='darkred', label=r'bl out mean')
        if a_inf_flag:
            ax[2].scatter(rho_bins, bl_a_inf_o_avg_per_rho, marker='_', color='darkred', label=r'bl out mean')

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
    base_name = 'chf2_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_figure4(focal_dist, comp_flag=None, bl_flag=None):
    """ 1 figure: origin-destination infection profile
    """
 
    # Collect all digested epidemic file names
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05

    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    agents_per_rho_sim = []
    infected_per_rho_sim = []
    events_hh_per_rho_sim = []
    events_ho_per_rho_sim = []
    events_oh_per_rho_sim = []
    events_oo_per_rho_sim = []

    comp_agents_per_rho_sim = []
    comp_infected_per_rho_sim = []
    comp_events_hh_per_rho_sim = []
    comp_events_ho_per_rho_sim = []
    comp_events_oh_per_rho_sim = []
    comp_events_oo_per_rho_sim = []

    bl_agents_per_rho_sim = []
    bl_infected_per_rho_sim = []
    bl_events_hh_per_rho_sim = []
    bl_events_ho_per_rho_sim = []
    bl_events_oh_per_rho_sim = []
    bl_events_oo_per_rho_sim = []
    
    # Loop over the collected file names
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))
        # Build the full path
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Check if the file exists
        if os.path.exists(epi_fullname):
            # Load digested epidemic data
            out_sim_data = ut.load_chapter_figure4_data(epi_fullname)

            if focal_dist in epi_filename and bl_flag in epi_filename and 'depr' not in epi_filename:
                # Collect data from every realization in structure
                bl_agents_per_rho_sim.extend(out_sim_data['agents'])
                bl_infected_per_rho_sim.extend(out_sim_data['infected'])
                bl_events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
                bl_events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
                bl_events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
                bl_events_oo_per_rho_sim.extend(out_sim_data['events_oo'])
            
            elif comp_flag is not None and comp_flag in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                comp_agents_per_rho_sim.extend(out_sim_data['agents'])
                comp_infected_per_rho_sim.extend(out_sim_data['infected'])
                comp_events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
                comp_events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
                comp_events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
                comp_events_oo_per_rho_sim.extend(out_sim_data['events_oo'])
            
            elif focal_dist in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                agents_per_rho_sim.extend(out_sim_data['agents'])
                infected_per_rho_sim.extend(out_sim_data['infected'])
                events_hh_per_rho_sim.extend(out_sim_data['events_hh'])
                events_ho_per_rho_sim.extend(out_sim_data['events_ho'])
                events_oh_per_rho_sim.extend(out_sim_data['events_oh'])
                events_oo_per_rho_sim.extend(out_sim_data['events_oo'])

        else:
            # File doesn't exist, skip the rest of the loop
            print(f"File {epi_fullname} does not exist. Skipping this iteration.")
            continue

    if focal_dist!= None:
        # Convert into numpy arrays
        agents_per_rho_sim = np.array(agents_per_rho_sim)
        infected_per_rho_sim = np.array(infected_per_rho_sim)
        events_hh_per_rho_sim = np.array(events_hh_per_rho_sim)
        events_ho_per_rho_sim = np.array(events_ho_per_rho_sim)
        events_oh_per_rho_sim = np.array(events_oh_per_rho_sim)
        events_oo_per_rho_sim = np.array(events_oo_per_rho_sim)

        # Filter failed outbreaks
        infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
        failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

        agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
        infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
        events_hh_per_rho_sim = np.delete(events_hh_per_rho_sim, failed_outbreaks, axis=0)
        events_ho_per_rho_sim = np.delete(events_ho_per_rho_sim, failed_outbreaks, axis=0)
        events_oh_per_rho_sim = np.delete(events_oh_per_rho_sim, failed_outbreaks, axis=0) 
        events_oo_per_rho_sim = np.delete(events_oo_per_rho_sim, failed_outbreaks, axis=0)

        # Create a new list to store the normalized values
        normalized_data = []

        # Iterate through each simulation and group
        for sim in range(len(events_hh_per_rho_sim)):
            normalized_sim_data = []
            for group in range(len(events_hh_per_rho_sim[sim])):
                # Calculate the sum of hh, ho, oh, oo for this simulation and group
                total_events = (
                    events_hh_per_rho_sim[sim][group] +
                    events_ho_per_rho_sim[sim][group] +
                    events_oh_per_rho_sim[sim][group] +
                    events_oo_per_rho_sim[sim][group]
                )

                # Normalize hh, ho, oh, oo values and append to the normalized_sim_data list
                if total_events != 0:
                    normalized_hh = events_hh_per_rho_sim[sim][group] / total_events
                    normalized_ho = events_ho_per_rho_sim[sim][group] / total_events
                    normalized_oh = events_oh_per_rho_sim[sim][group] / total_events
                    normalized_oo = events_oo_per_rho_sim[sim][group] / total_events
                else:
                    # Handle the case where total_events is 0 to avoid division by zero
                    normalized_hh = 0
                    normalized_ho = 0
                    normalized_oh = 0
                    normalized_oo = 0

                normalized_sim_data.append([normalized_hh, normalized_ho, normalized_oh, normalized_oo])

            # Append the normalized data for this simulation to the overall list
            normalized_data.append(normalized_sim_data)

        normalized_data = np.array(normalized_data)

        # Compute the average histograms for each variable separately
        avg_hh_hist = np.mean(normalized_data[:, :, 0], axis=0)
        avg_ho_hist = np.mean(normalized_data[:, :, 1], axis=0)
        avg_oh_hist = np.mean(normalized_data[:, :, 2], axis=0)
        avg_oo_hist = np.mean(normalized_data[:, :, 3], axis=0)

        # Compute the standard deviations for each variable separately
        std_hh_hist = np.std(normalized_data[:, :, 0], axis=0)
        std_ho_hist = np.std(normalized_data[:, :, 1], axis=0)
        std_oh_hist = np.std(normalized_data[:, :, 2], axis=0)
        std_oo_hist = np.std(normalized_data[:, :, 3], axis=0)

        # Compute the upper and lower 95% CI for each variable separately
        nsims = len(normalized_data)
        u95CI_hh_hist = avg_hh_hist + 1.96 * std_hh_hist / np.sqrt(nsims)
        l95CI_hh_hist = avg_hh_hist - 1.96 * std_hh_hist / np.sqrt(nsims)
        u95CI_ho_hist = avg_ho_hist + 1.96 * std_ho_hist / np.sqrt(nsims)
        l95CI_ho_hist = avg_ho_hist - 1.96 * std_ho_hist / np.sqrt(nsims)
        u95CI_oh_hist = avg_oh_hist + 1.96 * std_oh_hist / np.sqrt(nsims)
        l95CI_oh_hist = avg_oh_hist - 1.96 * std_oh_hist / np.sqrt(nsims)
        u95CI_oo_hist = avg_oo_hist + 1.96 * std_oo_hist / np.sqrt(nsims)
        l95CI_oo_hist = avg_oo_hist - 1.96 * std_oo_hist / np.sqrt(nsims)

    if comp_flag != None:
        # Convert into numpy arrays
        comp_agents_per_rho_sim = np.array(comp_agents_per_rho_sim)
        comp_infected_per_rho_sim = np.array(comp_infected_per_rho_sim)
        comp_events_hh_per_rho_sim = np.array(comp_events_hh_per_rho_sim)
        comp_events_ho_per_rho_sim = np.array(comp_events_ho_per_rho_sim)
        comp_events_oh_per_rho_sim = np.array(comp_events_oh_per_rho_sim)
        comp_events_oo_per_rho_sim = np.array(comp_events_oo_per_rho_sim)

        # Filter failed outbreaks
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_failed_outbreaks = np.where(comp_infected_fraction_sim < prevalence_cutoff)[0]

        comp_agents_per_rho_sim = np.delete(comp_agents_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_infected_per_rho_sim = np.delete(comp_infected_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_events_hh_per_rho_sim = np.delete(comp_events_hh_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_events_ho_per_rho_sim = np.delete(comp_events_ho_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_events_oh_per_rho_sim = np.delete(comp_events_oh_per_rho_sim, comp_failed_outbreaks, axis=0) 
        comp_events_oo_per_rho_sim = np.delete(comp_events_oo_per_rho_sim, comp_failed_outbreaks, axis=0)

        # Create a new list to store the normalized values
        comp_normalized_data = []

        # Iterate through each simulation and group
        for sim in range(len(comp_events_hh_per_rho_sim)):
            comp_normalized_sim_data = []
            for group in range(len(comp_events_hh_per_rho_sim[sim])):
                # Calculate the sum of hh, ho, oh, oo for this simulation and group
                comp_total_events = (
                    comp_events_hh_per_rho_sim[sim][group] +
                    comp_events_ho_per_rho_sim[sim][group] +
                    comp_events_oh_per_rho_sim[sim][group] +
                    comp_events_oo_per_rho_sim[sim][group]
                )

                # Normalize hh, ho, oh, oo values and append to the normalized_sim_data list
                if comp_total_events != 0:
                    comp_normalized_hh = comp_events_hh_per_rho_sim[sim][group] / comp_total_events
                    comp_normalized_ho = comp_events_ho_per_rho_sim[sim][group] / comp_total_events
                    comp_normalized_oh = comp_events_oh_per_rho_sim[sim][group] / comp_total_events
                    comp_normalized_oo = comp_events_oo_per_rho_sim[sim][group] / comp_total_events
                else:
                    # Handle the case where total_events is 0 to avoid division by zero
                    comp_normalized_hh = 0
                    comp_normalized_ho = 0
                    comp_normalized_oh = 0
                    comp_normalized_oo = 0

                comp_normalized_sim_data.append([comp_normalized_hh, comp_normalized_ho, comp_normalized_oh, comp_normalized_oo])

            # Append the normalized data for this simulation to the overall list
            comp_normalized_data.append(comp_normalized_sim_data)

        comp_normalized_data = np.array(comp_normalized_data)

        # Compute the average histograms for each variable separately
        comp_avg_hh_hist = np.mean(comp_normalized_data[:, :, 0], axis=0)
        comp_avg_ho_hist = np.mean(comp_normalized_data[:, :, 1], axis=0)
        comp_avg_oh_hist = np.mean(comp_normalized_data[:, :, 2], axis=0)
        comp_avg_oo_hist = np.mean(comp_normalized_data[:, :, 3], axis=0)

        # Compute the standard deviations for each variable separately
        comp_std_hh_hist = np.std(comp_normalized_data[:, :, 0], axis=0)
        comp_std_ho_hist = np.std(comp_normalized_data[:, :, 1], axis=0)
        comp_std_oh_hist = np.std(comp_normalized_data[:, :, 2], axis=0)
        comp_std_oo_hist = np.std(comp_normalized_data[:, :, 3], axis=0)

        # Compute the upper and lower 95% CI for each variable separately
        nsims = len(normalized_data)
        comp_u95CI_hh_hist = comp_avg_hh_hist + 1.96 * comp_std_hh_hist / np.sqrt(nsims)
        comp_l95CI_hh_hist = comp_avg_hh_hist - 1.96 * comp_std_hh_hist / np.sqrt(nsims)
        comp_u95CI_ho_hist = comp_avg_ho_hist + 1.96 * comp_std_ho_hist / np.sqrt(nsims)
        comp_l95CI_ho_hist = comp_avg_ho_hist - 1.96 * comp_std_ho_hist / np.sqrt(nsims)
        comp_u95CI_oh_hist = comp_avg_oh_hist + 1.96 * comp_std_oh_hist / np.sqrt(nsims)
        comp_l95CI_oh_hist = comp_avg_oh_hist - 1.96 * comp_std_oh_hist / np.sqrt(nsims)
        comp_u95CI_oo_hist = comp_avg_oo_hist + 1.96 * comp_std_oo_hist / np.sqrt(nsims)
        comp_l95CI_oo_hist = comp_avg_oo_hist - 1.96 * comp_std_oo_hist / np.sqrt(nsims)

    if bl_flag != None:
        # Convert into numpy arrays
        bl_agents_per_rho_sim = np.array(bl_agents_per_rho_sim)
        bl_infected_per_rho_sim = np.array(bl_infected_per_rho_sim)
        bl_events_hh_per_rho_sim = np.array(bl_events_hh_per_rho_sim)
        bl_events_ho_per_rho_sim = np.array(bl_events_ho_per_rho_sim)
        bl_events_oh_per_rho_sim = np.array(bl_events_oh_per_rho_sim)
        bl_events_oo_per_rho_sim = np.array(bl_events_oo_per_rho_sim)

        # Filter failed outbreaks
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_failed_outbreaks = np.where(bl_infected_fraction_sim < prevalence_cutoff)[0]

        bl_agents_per_rho_sim = np.delete(bl_agents_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_infected_per_rho_sim = np.delete(bl_infected_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_events_hh_per_rho_sim = np.delete(bl_events_hh_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_events_ho_per_rho_sim = np.delete(bl_events_ho_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_events_oh_per_rho_sim = np.delete(bl_events_oh_per_rho_sim, bl_failed_outbreaks, axis=0) 
        bl_events_oo_per_rho_sim = np.delete(bl_events_oo_per_rho_sim, bl_failed_outbreaks, axis=0)

        # Collapse observables
        bl_agents_sim = np.sum(agents_per_rho_sim, axis=1)
        bl_infected_sim = np.sum(infected_per_rho_sim, axis=1)
        bl_events_hh_sim = np.sum(bl_events_hh_per_rho_sim, axis=1)
        bl_events_ho_sim = np.sum(bl_events_ho_per_rho_sim, axis=1)
        bl_events_oh_sim = np.sum(bl_events_oh_per_rho_sim, axis=1)
        bl_events_oo_sim = np.sum(bl_events_oo_per_rho_sim, axis=1)

        bl_total_events_sim = bl_events_hh_sim + bl_events_ho_sim + bl_events_oh_sim + bl_events_oo_sim

        # Compute stats for collapsed observables
        bl_hh_fraction_sim = bl_events_hh_sim / bl_total_events_sim
        bl_ho_fraction_sim = bl_events_ho_sim / bl_total_events_sim
        bl_oh_fraction_sim = bl_events_oh_sim / bl_total_events_sim
        bl_oo_fraction_sim = bl_events_oo_sim / bl_total_events_sim

        bl_hh_fraction_avg = np.mean(bl_hh_fraction_sim)
        bl_ho_fraction_avg = np.mean(bl_ho_fraction_sim)
        bl_oh_fraction_avg = np.mean(bl_oh_fraction_sim)
        bl_oo_fraction_avg = np.mean(bl_oo_fraction_sim)

        # Create a new list to store the normalized values
        bl_normalized_data = []

        # Iterate through each simulation and group
        for sim in range(len(bl_events_hh_per_rho_sim)):
            bl_normalized_sim_data = []
            for group in range(len(bl_events_hh_per_rho_sim[sim])):
                # Calculate the sum of hh, ho, oh, oo for this simulation and group
                bl_total_events = (
                    bl_events_hh_per_rho_sim[sim][group] +
                    bl_events_ho_per_rho_sim[sim][group] +
                    bl_events_oh_per_rho_sim[sim][group] +
                    bl_events_oo_per_rho_sim[sim][group]
                )

                # Normalize hh, ho, oh, oo values and append to the normalized_sim_data list
                if bl_total_events != 0:
                    bl_normalized_hh = bl_events_hh_per_rho_sim[sim][group] / bl_total_events
                    bl_normalized_ho = bl_events_ho_per_rho_sim[sim][group] / bl_total_events
                    bl_normalized_oh = bl_events_oh_per_rho_sim[sim][group] / bl_total_events
                    bl_normalized_oo = bl_events_oo_per_rho_sim[sim][group] / bl_total_events
                else:
                    # Handle the case where total_events is 0 to avoid division by zero
                    bl_normalized_hh = 0
                    bl_normalized_ho = 0
                    bl_normalized_oh = 0
                    bl_normalized_oo = 0

                bl_normalized_sim_data.append([bl_normalized_hh, bl_normalized_ho, bl_normalized_oh, bl_normalized_oo])

            # Append the normalized data for this simulation to the overall list
            bl_normalized_data.append(bl_normalized_sim_data)

        bl_normalized_data = np.array(bl_normalized_data)

        # Compute the average histograms for each variable separately
        bl_avg_hh_hist = np.mean(bl_normalized_data[:, :, 0], axis=0)
        bl_avg_ho_hist = np.mean(bl_normalized_data[:, :, 1], axis=0)
        bl_avg_oh_hist = np.mean(bl_normalized_data[:, :, 2], axis=0)
        bl_avg_oo_hist = np.mean(bl_normalized_data[:, :, 3], axis=0)

        # Compute the standard deviations for each variable separately
        bl_std_hh_hist = np.std(bl_normalized_data[:, :, 0], axis=0)
        bl_std_ho_hist = np.std(bl_normalized_data[:, :, 1], axis=0)
        bl_std_oh_hist = np.std(bl_normalized_data[:, :, 2], axis=0)
        bl_std_oo_hist = np.std(bl_normalized_data[:, :, 3], axis=0)

        # Compute the upper and lower 95% CI for each variable separately
        nsims = len(normalized_data)
        bl_u95CI_hh_hist = bl_avg_hh_hist + 1.96 * bl_std_hh_hist / np.sqrt(nsims)
        bl_l95CI_hh_hist = bl_avg_hh_hist - 1.96 * bl_std_hh_hist / np.sqrt(nsims)
        bl_u95CI_ho_hist = bl_avg_ho_hist + 1.96 * bl_std_ho_hist / np.sqrt(nsims)
        bl_l95CI_ho_hist = bl_avg_ho_hist - 1.96 * bl_std_ho_hist / np.sqrt(nsims)
        bl_u95CI_oh_hist = bl_avg_oh_hist + 1.96 * bl_std_oh_hist / np.sqrt(nsims)
        bl_l95CI_oh_hist = bl_avg_oh_hist - 1.96 * bl_std_oh_hist / np.sqrt(nsims)
        bl_u95CI_oo_hist = bl_avg_oo_hist + 1.96 * bl_std_oo_hist / np.sqrt(nsims)
        bl_l95CI_oo_hist = bl_avg_oo_hist - 1.96 * bl_std_oo_hist / np.sqrt(nsims)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if focal_dist != None:
        # Plot hh
        ax.scatter(rho_bins, avg_hh_hist, marker='o', color='dodgerblue', label=r'h-h mean')
        ax.fill_between(rho_bins, l95CI_hh_hist, u95CI_hh_hist, color='dodgerblue', alpha=0.2)
        # Plot ho
        ax.scatter(rho_bins, avg_ho_hist, marker='o', color='slateblue', label=r'h-o mean')
        ax.fill_between(rho_bins, l95CI_ho_hist, u95CI_ho_hist, color='slateblue', alpha=0.2)
        # Plot oh
        ax.scatter(rho_bins, avg_oh_hist, marker='o', color='deeppink', label=r'o-h mean')
        ax.fill_between(rho_bins, l95CI_oh_hist, u95CI_oh_hist, color='deeppink', alpha=0.2)
        # Plot oo
        ax.scatter(rho_bins, avg_oo_hist, marker='o', color='firebrick', label=r'o-o mean')
        ax.fill_between(rho_bins, l95CI_oo_hist, u95CI_oo_hist, color='firebrick', alpha=0.2)

    if comp_flag != None:
        # Plot hh
        ax.scatter(rho_bins, comp_avg_hh_hist, marker='s', color='dodgerblue', label=r'gaussian h-h')
        ax.fill_between(rho_bins, comp_l95CI_hh_hist, comp_u95CI_hh_hist, color='dodgerblue', alpha=0.2)
        # Plot ho
        ax.scatter(rho_bins, comp_avg_ho_hist, marker='s', color='slateblue', label=r'gaussian h-o')
        ax.fill_between(rho_bins, comp_l95CI_ho_hist, comp_u95CI_ho_hist, color='slateblue', alpha=0.2)
        # Plot oh
        ax.scatter(rho_bins, comp_avg_oh_hist, marker='s', color='deeppink', label=r'gaussian o-h')
        ax.fill_between(rho_bins, comp_l95CI_oh_hist, comp_u95CI_oh_hist, color='deeppink', alpha=0.2)
        # Plot oo
        ax.scatter(rho_bins, comp_avg_oo_hist, marker='s', color='firebrick', label=r'gaussian o-o')
        ax.fill_between(rho_bins, comp_l95CI_oo_hist, comp_u95CI_oo_hist, color='firebrick', alpha=0.2)

    if bl_flag != None:
        #ax.axhline(bl_hh_fraction_avg, color='steelblue', linestyle='--', label=r'bl1 hh')
        #ax.axhline(bl_ho_fraction_avg, color='darkslateblue', linestyle='--', label=r'bl1 ho')
        #ax.axhline(bl_oh_fraction_avg, color='fuchsia', linestyle='--', label=r'bl1 oh')
        ax.axhline(bl_oo_fraction_avg, color='darkred', linestyle='--', label=r'baseline oo')


    title = 'origin-destination infections'
    ax.set_title(title, fontsize=30)
    ax.set_xlabel(r'$\rho$', fontsize=25)
    ax.set_ylabel(r'event fraction', fontsize=25)
    #ax.set_xlim(0.0, 1.0)
    #ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=20, loc='upper center')

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
    base_name = 'chf4_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_figure5(focal_dist, comp_flag=None, bl_flag=None):
    """ 1 figure: contribution to new cases profile
    """

    # Collect all digested epidemic file names
    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)
    rho_midpoints = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    agents_per_rho_sim = []
    infected_per_rho_sim = []
    new_cases_per_rho_sim = []

    comp_agents_per_rho_sim = []
    comp_infected_per_rho_sim = []
    comp_new_cases_per_rho_sim = []

    bl_agents_per_rho_sim = []
    bl_infected_per_rho_sim = []
    bl_new_cases_per_rho_sim = []
    
    # Loop over the collected file names
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))
        # Build the full path
        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        # Check if the file exists
        if os.path.exists(epi_fullname):
            # Load digested epidemic data
            out_sim_data = ut.load_chapter_figure5_data(epi_fullname)

            if focal_dist in epi_filename and bl_flag in epi_filename and 'depr' not in epi_filename:
                # Collect data from every realization in structure
                bl_agents_per_rho_sim.extend(out_sim_data['agents'])
                bl_infected_per_rho_sim.extend(out_sim_data['infected'])
                bl_new_cases_per_rho_sim.extend(out_sim_data['new_cases'])
            elif comp_flag is not None and comp_flag in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                comp_agents_per_rho_sim.extend(out_sim_data['agents'])
                comp_infected_per_rho_sim.extend(out_sim_data['infected'])
                comp_new_cases_per_rho_sim.extend(out_sim_data['new_cases'])
            elif focal_dist in epi_filename and 'depr' in epi_filename:
                # Collect data from every realization in structure
                agents_per_rho_sim.extend(out_sim_data['agents'])
                infected_per_rho_sim.extend(out_sim_data['infected'])
                new_cases_per_rho_sim.extend(out_sim_data['new_cases'])

        else:
            # File doesn't exist, skip the rest of the loop
            print(f"File {epi_fullname} does not exist. Skipping this iteration.")
            continue

    if focal_dist != None:
        # Convert into numpy arrays
        agents_per_rho_sim = np.array(agents_per_rho_sim)
        infected_per_rho_sim = np.array(infected_per_rho_sim)
        new_cases_per_rho_sim = np.array(new_cases_per_rho_sim)

        # Filter failed outbreaks
        infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
        failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

        agents_per_rho_sim = np.delete(agents_per_rho_sim, failed_outbreaks, axis=0)
        infected_per_rho_sim = np.delete(infected_per_rho_sim, failed_outbreaks, axis=0)
        new_cases_per_rho_sim = np.delete(new_cases_per_rho_sim, failed_outbreaks, axis=0)

        # Collapse observales
        agents_sim = np.sum(agents_per_rho_sim, axis=1)
        infected_sim = np.sum(infected_per_rho_sim, axis=1)
        new_cases_sim = np.sum(new_cases_per_rho_sim, axis=1)
        new_cases_fraction_sim = new_cases_sim / infected_sim

        # Compute final observables
        new_cases_fraction_per_rho_sim = new_cases_per_rho_sim / infected_per_rho_sim

        # Compute stats
        new_cases_fraction_avg_per_rho = np.mean(new_cases_fraction_per_rho_sim, axis=0)
        new_cases_fraction_std_per_rho = np.std(new_cases_fraction_per_rho_sim, axis=0)
        new_cases_fraction_u95_per_rho = new_cases_fraction_avg_per_rho + 1.96 * new_cases_fraction_std_per_rho / np.sqrt(len(new_cases_fraction_per_rho_sim))
        new_cases_fraction_l95_per_rho = new_cases_fraction_avg_per_rho - 1.96 * new_cases_fraction_std_per_rho / np.sqrt(len(new_cases_fraction_per_rho_sim))

        new_cases_fraction_avg = np.mean(new_cases_fraction_sim, axis=0)
        new_cases_fraction_std = np.std(new_cases_fraction_sim, axis=0)
        new_cases_fraction_u95 = new_cases_fraction_sim + 1.96 * new_cases_fraction_std / np.sqrt(len(new_cases_fraction_sim))
        new_cases_fraction_l95 = new_cases_fraction_sim - 1.96 * new_cases_fraction_std / np.sqrt(len(new_cases_fraction_sim))

    if comp_flag != None:
        # Convert into numpy arrays
        comp_infected_per_rho_sim = np.array(comp_infected_per_rho_sim)
        comp_new_cases_per_rho_sim = np.array(comp_new_cases_per_rho_sim)

        # Filter failed outbreaks
        comp_infected_fraction_sim = np.sum(comp_infected_per_rho_sim, axis=1) / np.sum(comp_agents_per_rho_sim, axis=1)
        comp_failed_outbreaks = np.where(comp_infected_fraction_sim < prevalence_cutoff)[0]

        comp_agents_per_rho_sim = np.delete(comp_agents_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_infected_per_rho_sim = np.delete(comp_infected_per_rho_sim, comp_failed_outbreaks, axis=0)
        comp_new_cases_per_rho_sim = np.delete(comp_new_cases_per_rho_sim, comp_failed_outbreaks, axis=0)
    
        # Compute final observables
        comp_new_cases_fraction_per_rho_sim = comp_new_cases_per_rho_sim / comp_infected_per_rho_sim

        # Compute stats
        comp_new_cases_fraction_avg_per_rho = np.mean(comp_new_cases_fraction_per_rho_sim, axis=0)
        comp_new_cases_fraction_std_per_rho = np.std(comp_new_cases_fraction_per_rho_sim, axis=0)
        comp_new_cases_fraction_u95_per_rho = comp_new_cases_fraction_avg_per_rho + 1.96 * comp_new_cases_fraction_std_per_rho / np.sqrt(len(comp_new_cases_fraction_per_rho_sim))
        comp_new_cases_fraction_l95_per_rho = comp_new_cases_fraction_avg_per_rho - 1.96 * comp_new_cases_fraction_std_per_rho / np.sqrt(len(comp_new_cases_fraction_per_rho_sim))

    if bl_flag != None:
        # Convert into numpy arrays
        bl_agents_per_rho_sim = np.array(bl_agents_per_rho_sim)
        bl_infected_per_rho_sim = np.array(bl_infected_per_rho_sim)
        bl_new_cases_per_rho_sim = np.array(bl_new_cases_per_rho_sim)

        # Filter failed outbreaks
        bl_infected_fraction_sim = np.sum(bl_infected_per_rho_sim, axis=1) / np.sum(bl_agents_per_rho_sim, axis=1)
        bl_failed_outbreaks = np.where(bl_infected_fraction_sim < prevalence_cutoff)[0]

        bl_agents_per_rho_sim = np.delete(bl_agents_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_infected_per_rho_sim = np.delete(bl_infected_per_rho_sim, bl_failed_outbreaks, axis=0)
        bl_new_cases_per_rho_sim = np.delete(bl_new_cases_per_rho_sim, bl_failed_outbreaks, axis=0)

        # Collapse observales
        bl_agents_sim = np.sum(bl_agents_per_rho_sim, axis=1)
        bl_infected_sim = np.sum(bl_infected_per_rho_sim, axis=1)
        bl_new_cases_sim = np.sum(bl_new_cases_per_rho_sim, axis=1)
        bl_new_cases_fraction_sim = bl_new_cases_sim / bl_infected_sim
    
        # Compute final observables
        bl_new_cases_fraction_per_rho_sim = bl_new_cases_per_rho_sim / bl_infected_per_rho_sim
    
        # Compute stats
        bl_new_cases_fraction_avg_per_rho = np.mean(bl_new_cases_fraction_per_rho_sim, axis=0)
        bl_new_cases_fraction_std_per_rho = np.std(bl_new_cases_fraction_per_rho_sim, axis=0)
        bl_new_cases_fraction_u95_per_rho = bl_new_cases_fraction_avg_per_rho + 1.96 * bl_new_cases_fraction_std_per_rho / np.sqrt(len(bl_new_cases_fraction_per_rho_sim))
        bl_new_cases_fraction_l95_per_rho = bl_new_cases_fraction_avg_per_rho - 1.96 * bl_new_cases_fraction_std_per_rho / np.sqrt(len(bl_new_cases_fraction_per_rho_sim))

        bl_new_cases_fraction_avg = np.mean(bl_new_cases_fraction_sim, axis=0)
        bl_new_cases_fraction_std = np.std(bl_new_cases_fraction_sim, axis=0)
        bl_new_cases_fraction_u95 = bl_new_cases_fraction_sim + 1.96 * bl_new_cases_fraction_std / np.sqrt(len(bl_new_cases_fraction_sim))
        bl_new_cases_fraction_l95 = bl_new_cases_fraction_sim - 1.96 * bl_new_cases_fraction_std / np.sqrt(len(bl_new_cases_fraction_sim))

    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 8))

    if focal_dist != None:
        ax.scatter(rho_bins, new_cases_fraction_avg_per_rho, marker='o', color='teal', label=r'beta mean')
        ax.fill_between(rho_bins, new_cases_fraction_l95_per_rho, new_cases_fraction_u95_per_rho, color='teal', alpha=0.2)
        ax.axhline(new_cases_fraction_avg, color='darkslategrey', linestyle='--', label=r'global avg')

    if comp_flag != None:
        ax.scatter(rho_bins, comp_new_cases_fraction_avg_per_rho, marker='o', color='seagreen', label=r'gaussian mean')
        ax.fill_between(rho_bins, comp_new_cases_fraction_l95_per_rho, comp_new_cases_fraction_u95_per_rho, color='teal', alpha=0.2)

    if bl_flag != None:
        ax.axhline(bl_new_cases_fraction_avg, color='steelblue', linestyle='--', label=r'baseline')

    # Subplot 0 settings
    title = r'contribution to infections'
    ax.set_title(title, fontsize=30)
    ax.set_xlim(0.0, 1.0)
    #ax.set_ylim(0.0, 2.2)
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
    base_name = 'chf5_' + epi_filename
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def main():

    focal_dist = 'Beta'
    comp_flag = 'Gaussian'
    bl_flag = 'b1hom'

    plot_figure1(focal_dist=focal_dist, comp_flag=comp_flag, bl_flag=bl_flag, t_inv_flag=True, t_inf_flag=True)
    plot_figure2(focal_dist=focal_dist, comp_flag=comp_flag, bl_flag=bl_flag, r_inv_flag=True, r_inf_flag=True)
    plot_figure3(focal_dist=focal_dist, comp_flag=comp_flag, bl_flag=bl_flag, f_inf_flag=True, a_inf_flag=True)
    plot_figure4(focal_dist=focal_dist, comp_flag=comp_flag, bl_flag=bl_flag)
    plot_figure5(focal_dist=focal_dist, comp_flag=comp_flag, bl_flag=bl_flag)

if __name__ == '__main__':
    main()
