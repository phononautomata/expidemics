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
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statistics import mode
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

import analysis as an
import utils as ut

cwd_path = os.getcwd()

def plot_depr_panel3A(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 4x3 panel
    f00: typical invader rho spatial map for beta depr
    f01: typical invader rho spatial map for beta uniform
    f02: typical invader rho spatial map for gaussian depr
    f10: invasion fraction spatial map for beta depr
    f11: invasion fraction spatial map for beta uniform
    f12: invasion fraction spatial map for gaussian depr
    f20: invasion times map for beta depr
    f21: invasion times map for beta uniform
    f22: invasion times map for gaussian depr
    f30: invasion time distribution for beta depr
    f31: invasion time distribution for beta uniform
    f32: invasion time distribution for gaussian depr
    """

    lower_path = 'config/'
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    lower_path = 'data/'
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    R0 = 1.2
    r_0 = 0.0

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel3A_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    label_dict = ut.build_label_dictionary()
    custom_cmap = ListedColormap(['red'])

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 17))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]

        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            r_inv_dist_per_loc_sim = []
            t_inv_dist_per_loc_sim = []
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel3A_results(
                    out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    r_inv_dist_per_loc_sim=r_inv_dist_per_loc_sim,
                    t_inv_dist_per_loc_sim=t_inv_dist_per_loc_sim,
                )

            processed_results = ut.compute_depr_chapter_panel3A_stats(
                agents_per_rho_sim=agents_per_rho_sim, 
                infected_per_rho_sim=infected_per_rho_sim,
                space_df=space_df,
                prevalence_cutoff=prevalence_cutoff,
                r_inv_dist_per_loc_sim=r_inv_dist_per_loc_sim,
                t_inv_dist_per_loc_sim=t_inv_dist_per_loc_sim,
                )

            inv_rho_avg_lattice = processed_results['inv_rho_avg_lattice']
            inv_t_avg_lattice = processed_results['t_inv_avg_lattice']
            r_inv_avg_per_loc = processed_results['inv_rho_avg_loc']
            t_inv_avg_per_loc = processed_results['t_inv_avg_loc']
            invasion_fraction_avg = processed_results['invasion_fraction_avg']
            inv_rate_avg_lattice = processed_results['inv_rate_avg_lattice']
            invasion_fraction_avg_loc = processed_results['invasion_fraction_avg_loc']

            if dist_key == distribution_flags[0] and sce_key == 'depr':
                # SUBPLOT 00
                im00 = ax[0, 0].imshow(inv_rho_avg_lattice.T, cmap='coolwarm', aspect='auto')
                im00.set_clim(vmin=0.0, vmax=1.0)
                cbar00 = fig.colorbar(im00, ax=ax[0, 0], shrink=0.9)
                cbar00.ax.tick_params(labelsize=18)
                #cbar01.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_rho_avg_lattice.T), inv_rho_avg_lattice)
                masked_data[28, 28] = 1.0
                im00 = ax[0, 0].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)

                #ax[0, 0].set_xlabel("longitude (\u00b0 W)", fontsize=25)
                ax[0, 0].set_ylabel("latitude (\u00b0 N)", fontsize=35)
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
                ax[0, 0].set_title("Beta d-EPR", fontsize=40)

                # SUBPLOT 10
                im10 = ax[1, 0].imshow(inv_rate_avg_lattice.T, cmap='Blues', aspect='auto')
                im10.set_clim(vmin=0.0, vmax=1.0)
                cbar10 = fig.colorbar(im10, ax=ax[1, 0], shrink=0.9)
                cbar10.ax.tick_params(labelsize=18)
                #cbar10.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_rate_avg_lattice.T), inv_rate_avg_lattice)
                masked_data[28, 28] = 1.0
                im10 = ax[1, 0].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)

                #ax[1, 0].set_xlabel('longitude (\u00b0 W)', fontsize=25)
                ax[1, 0].set_ylabel('latitude (\u00b0 N)', fontsize=35)
                ax[1, 0].invert_yaxis()
                ax[1, 0].tick_params(axis='both', labelsize=18)

                ax[1, 0].text(0.05, 0.05, r"$D(\infty)/V=${0:.2f}".format(np.round(invasion_fraction_avg, 2)), transform=ax[1, 0].transAxes, fontsize=20, color='red', weight='bold')

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[1, 0].set_xticks(x_ticks_pos)
                ax[1, 0].set_yticks(y_ticks_pos)
                ax[1, 0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[1, 0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                # SUBPLOT 20
                vmin = np.nanmin(inv_t_avg_lattice)
                vmax = 350 #np.nanmax(inv_t_avg_lattice)
                im20 = ax[2, 0].imshow(inv_t_avg_lattice.T, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                #im20.set_clim(vmin=0.0, vmax=1.0)
                cbar20 = fig.colorbar(im20, ax=ax[2, 0], shrink=0.9)
                cbar20.ax.tick_params(labelsize=18)
                #cbar20.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_t_avg_lattice.T), inv_t_avg_lattice)
                masked_data[28, 28] = 1.0
                im20 = ax[2, 0].imshow(masked_data, cmap=custom_cmap)

                ax[2, 0].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                ax[2, 0].set_ylabel('latitude (\u00b0 N)', fontsize=35)
                ax[2, 0].invert_yaxis()
                ax[2, 0].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[2, 0].set_xticks(x_ticks_pos)
                ax[2, 0].set_yticks(y_ticks_pos)
                ax[2, 0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[2, 0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                # SUBPLOT 30
                #ax[3, 0].hist(t_inv_avg_per_loc, color='slateblue')
                #ax[3, 0].set_xlabel(r'$\langle t_{{inv}}\rangle$', fontsize=25)
                #ax[3, 0].set_ylabel(r'counts', fontsize=25)
                #ax[3, 0].set_title('', fontsize=30)
                #ax[3, 0].tick_params(axis='both', labelsize=18)

                ax[0, 0].text(0.025, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 0].text(0.025, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
                ax[2, 0].text(0.025, 0.9, r"A3", transform=ax[2, 0].transAxes, fontsize=40, color='black', weight="bold")
                #ax[3, 0].text(0.05, 0.9, r"A4", transform=ax[3, 0].transAxes, fontsize=40, color='black', weight="bold")

            elif dist_key == distribution_flags[0] and sce_key == 'b1het':
                # SUBPLOT 01
                im01 = ax[0, 1].imshow(inv_rho_avg_lattice.T, cmap='coolwarm', aspect='auto')
                im01.set_clim(vmin=0.0, vmax=1.0)
                cbar01 = fig.colorbar(im01, ax=ax[0, 1], shrink=0.9)
                cbar01.ax.tick_params(labelsize=18)
                #cbar01.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_rho_avg_lattice.T), inv_rho_avg_lattice)
                masked_data[28, 28] = 1.0
                im01 = ax[0, 1].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)

                #ax[0, 1].set_xlabel("longitude (\u00b0 W)", fontsize=25)
                #ax[0, 1].set_ylabel("latitude (\u00b0 N)", fontsize=25)
                ax[0, 1].invert_yaxis()
                ax[0, 1].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[0, 1].set_xticks(x_ticks_pos)
                ax[0, 1].set_yticks(y_ticks_pos)
                ax[0, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[0, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                ax[0, 1].set_title("Beta memory-less", fontsize=40)

                # SUBPLOT 11
                im11 = ax[1, 1].imshow(inv_rate_avg_lattice.T, cmap='Blues', aspect='auto')
                im11.set_clim(vmin=0.0, vmax=1.0)
                cbar11 = fig.colorbar(im11, ax=ax[1, 1], shrink=0.9)
                cbar11.ax.tick_params(labelsize=18)
                #cbar11.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_rate_avg_lattice.T), inv_rate_avg_lattice)
                masked_data[28, 28] = 1.0
                im11 = ax[1, 1].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)

                #ax[1, 1].set_xlabel('longitude (\u00b0 W)', fontsize=25)
                #ax[1, 1].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[1, 1].invert_yaxis()
                ax[1, 1].tick_params(axis='both', labelsize=18)

                ax[1, 1].text(0.05, 0.05, r"$D(\infty)/V=${0:.2f}".format(np.round(invasion_fraction_avg, 2)), transform=ax[1, 1].transAxes, fontsize=20, color='red', weight='bold')

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[1, 1].set_xticks(x_ticks_pos)
                ax[1, 1].set_yticks(y_ticks_pos)
                ax[1, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[1, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                # SUBPLOT 21
                vmin = np.nanmin(inv_t_avg_lattice)
                vmax = 350 #np.nanmax(inv_t_avg_lattice)
                im21 = ax[2, 1].imshow(inv_t_avg_lattice.T, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                #im21.set_clim(vmin=0.0, vmax=1.0)
                cbar21 = fig.colorbar(im21, ax=ax[2, 1], shrink=0.9)
                cbar21.ax.tick_params(labelsize=18)
                #cbar21.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_t_avg_lattice.T), inv_t_avg_lattice)
                masked_data[28, 28] = 1.0
                im21 = ax[2, 1].imshow(masked_data, cmap=custom_cmap)

                ax[2, 1].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                #ax[2, 1].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[2, 1].invert_yaxis()
                ax[2, 1].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[2, 1].set_xticks(x_ticks_pos)
                ax[2, 1].set_yticks(y_ticks_pos)
                ax[2, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[2, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                # SUBPLOT 31
                #ax[3, 1].hist(t_inv_avg_per_loc, color='slateblue')
                #ax[3, 1].set_xlabel(r'$\langle t_{{inv}}\rangle$', fontsize=25)
                #ax[3, 1].set_ylabel(r'counts', fontsize=25)
                #ax[3, 1].set_title('', fontsize=30)
                #ax[3, 1].tick_params(axis='both', labelsize=18)

                ax[0, 1].text(0.025, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 1].text(0.025, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
                ax[2, 1].text(0.025, 0.9, r"B3", transform=ax[2, 1].transAxes, fontsize=40, color='black', weight="bold")

            elif dist_key == distribution_flags[1] and sce_key == 'depr':
                # SUBPLOT 02
                im02 = ax[0, 2].imshow(inv_rho_avg_lattice.T, cmap='coolwarm', aspect='auto')
                im02.set_clim(vmin=0.0, vmax=1.0)
                cbar02 = fig.colorbar(im02, ax=ax[0, 2], shrink=0.9)
                cbar02.ax.tick_params(labelsize=18)
                cbar02.set_label(r'invader $\langle\rho\rangle$', fontsize=35)

                masked_data = np.ma.masked_where(np.ones_like(inv_rho_avg_lattice.T), inv_rho_avg_lattice)
                masked_data[28, 28] = 1.0
                im02 = ax[0, 2].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)

                #ax[0, 2].set_xlabel("longitude (\u00b0 W)", fontsize=25)
                #ax[0, 2].set_ylabel("latitude (\u00b0 N)", fontsize=25)
                ax[0, 2].invert_yaxis()
                ax[0, 2].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[0, 2].set_xticks(x_ticks_pos)
                ax[0, 2].set_yticks(y_ticks_pos)
                ax[0, 2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[0, 2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                ax[0, 2].set_title("Gaussian d-EPR", fontsize=40)

                # SUBPLOT 12
                im12 = ax[1, 2].imshow(inv_rate_avg_lattice.T, cmap='Blues', aspect='auto')
                im12.set_clim(vmin=0.0, vmax=1.0)
                cbar12 = fig.colorbar(im12, ax=ax[1, 2], shrink=0.9)
                cbar12.ax.tick_params(labelsize=18)
                cbar12.set_label(r'local invasion probability', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inv_rate_avg_lattice.T), inv_rate_avg_lattice)
                masked_data[28, 28] = 1.0
                im12 = ax[1, 2].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)

                #ax[1, 2].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                #ax[1, 2].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[1, 2].invert_yaxis()
                ax[1, 2].tick_params(axis='both', labelsize=18)

                ax[1, 2].text(0.05, 0.05, r"$D(\infty)/V=${0:.2f}".format(np.round(invasion_fraction_avg, 2)), transform=ax[1, 2].transAxes, fontsize=20, color='red', weight='bold')

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[1, 2].set_xticks(x_ticks_pos)
                ax[1, 2].set_yticks(y_ticks_pos)
                ax[1, 2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[1, 2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                # SUBPLOT 22
                vmin = np.nanmin(inv_t_avg_lattice)
                vmax = 350 #np.nanmax(inv_t_avg_lattice)
                im22 = ax[2, 2].imshow(inv_t_avg_lattice.T, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                #im22.set_clim(vmin=0.0, vmax=1.0)
                cbar22 = fig.colorbar(im22, ax=ax[2, 2], shrink=0.9)
                cbar22.ax.tick_params(labelsize=18)
                cbar22.set_label(r'$\langle t_{{inv}}\rangle$', fontsize=35)

                masked_data = np.ma.masked_where(np.ones_like(inv_t_avg_lattice.T), inv_t_avg_lattice)
                masked_data[28, 28] = 1.0
                im22 = ax[2, 2].imshow(masked_data, cmap=custom_cmap)

                ax[2, 2].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                #ax[2, 2].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[2, 2].invert_yaxis()
                ax[2, 2].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[2, 2].set_xticks(x_ticks_pos)
                ax[2, 2].set_yticks(y_ticks_pos)
                ax[2, 2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[2, 2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                # SUBPLOT 32
                #ax[3, 2].hist(t_inv_avg_per_loc, color='slateblue')
                #ax[3, 2].set_xlabel(r'$\langle t_{{inv}}\rangle$', fontsize=25)
                #ax[3, 2].set_ylabel(r'counts', fontsize=25)
                #ax[3, 2].tick_params(axis='both', labelsize=18)

                ax[0, 2].text(0.025, 0.9, r"C1", transform=ax[0, 2].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 2].text(0.025, 0.9, r"C2", transform=ax[1, 2].transAxes, fontsize=40, color='black', weight="bold")
                ax[2, 2].text(0.025, 0.9, r"C3", transform=ax[2, 2].transAxes, fontsize=40, color='black', weight="bold")
                #ax[3, 2].text(0.05, 0.9, r"C4", transform=ax[3, 2].transAxes, fontsize=40, color='black', weight="bold")

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
    base_name = 'deprf3A'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel3B(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 4x3 panel
    f00: typical infected rho spatial map for beta depr
    f01: typical infected rho spatial map for beta uniform
    f02: typical infected rho spatial map for gaussian depr
    f10: local new cases vs. attractiveness scatter for beta depr
    f11: local new cases vs. attractiveness scatter for beta uniform
    f12: local new cases vs. attractiveness scatter for gaussian depr
    f20: peak times map for beta depr
    f21: peak times map for beta uniform
    f22: peak times map for gaussian depr
    f30: peak time distribution for beta depr
    f31: peak time distribution for beta uniform
    f32: peak time distribution for gaussian depr
    """

    lower_path = 'config/'
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    lower_path = 'data/'
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    attr_cutoff = 0.000000001
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel3B_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    label_dict = ut.build_label_dictionary()
    custom_cmap = ListedColormap(['red'])
    custom_cmap2 = ListedColormap(['black', 'viridis'])

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 17))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]

        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            total_cases_loc_sim = []
            r_inf_dist_per_loc_sim = []
            pt_dist_per_loc_sim = []
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel3B_results(
                    out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    total_cases_loc_sim=total_cases_loc_sim,
                    r_inf_dist_per_loc_sim=r_inf_dist_per_loc_sim,
                    pt_dist_per_loc_sim=pt_dist_per_loc_sim,
                )

            processed_results = ut.compute_depr_chapter_panel3B_stats(
                agents_per_rho_sim=agents_per_rho_sim, 
                infected_per_rho_sim=infected_per_rho_sim,
                total_cases_loc_sim=total_cases_loc_sim,
                space_df=space_df,
                prevalence_cutoff=prevalence_cutoff,
                r_inf_dist_per_loc_sim=r_inf_dist_per_loc_sim,
                pt_dist_per_loc_sim=pt_dist_per_loc_sim,
                )
            
            total_cases_avg_loc = processed_results['total_cases_avg_loc']
            attr_l = processed_results['attractiveness_l']
            inf_rho_avg_lattice = processed_results['inf_rho_avg_lattice']
            r_inf_avg_per_loc = processed_results['inf_rho_avg_loc']
            pt_avg_lattice = processed_results['pt_avg_lattice']
            pt_avg_per_loc = processed_results['pt_avg_loc']

            if dist_key == distribution_flags[0] and sce_key == 'depr':
                # SUBPLOT 00
                im00 = ax[0, 0].imshow(inf_rho_avg_lattice.T, cmap='coolwarm', aspect='auto')
                im00.set_clim(vmin=0.0, vmax=1.0)
                cbar00 = fig.colorbar(im00, ax=ax[0, 0], shrink=1.0)
                cbar00.ax.tick_params(labelsize=18)
                #cbar00.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inf_rho_avg_lattice.T), inf_rho_avg_lattice)
                masked_data[28, 28] = 1.0
                im00 = ax[0, 0].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0, aspect='auto') # extent=extent)

                ax[0, 0].set_xlabel("longitude (\u00b0 W)", fontsize=35)
                ax[0, 0].set_ylabel("latitude (\u00b0 N)", fontsize=35)
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
                ax[0, 0].set_title("Beta d-EPR", fontsize=40)

                # SUBPLOT 10
                hb10 = ax[1, 0].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
                hb10.set_clim(vmin=0.0, vmax=1.0)
                cbar10 = fig.colorbar(hb10, ax=ax[1, 0])
                cbar10.ax.tick_params(labelsize=18)
                #cbar10.set_label(r'infected $\langle\rho\rangle$', fontsize=25)

                # Compute the mean value for each hexbin
                xbins = hb10.get_offsets()[:, 0]
                ybins = hb10.get_offsets()[:, 1]
                mean_values = hb10.get_array()
                mean_rho_for_hexbins = []

                for i in range(len(mean_values)):
                    if i == len(mean_values) - 1:  # Handle the last bin separately
                        condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
                    else:
                        condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

                    indices = np.where(condition)
                    if len(indices[0]) > 0:
                        mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
                    else:
                        mean_rho_for_hexbins.append(0.0)

                model_1 = LinearRegression()
                model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
                y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
                ax[1, 0].plot(attr_l, y_pred_11, color='indigo', linestyle='--', linewidth=2)
                r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
                ax[1, 0].text(0.5, 0.85, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 0].transAxes, fontsize=30, color='black')

                ax[1, 0].set_xlabel(r'$A$', fontsize=35)
                ax[1, 0].set_ylabel('mean total cases', fontsize=35)
                ax[1, 0].tick_params(axis='both', labelsize=18)

                # SUBPLOT 20
                vmin = 200 #np.nanmin(pt_avg_lattice)
                vmax = 350 #np.nanmax(pt_avg_lattice)
                im20 = ax[2, 0].imshow(pt_avg_lattice.T, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                #im20.set_clim(vmin=0.0, vmax=1.0)
                cbar20 = fig.colorbar(im20, ax=ax[2, 0], shrink=1.0)
                cbar20.ax.tick_params(labelsize=18)
                #cbar20.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(pt_avg_lattice.T), pt_avg_lattice)
                masked_data[28, 28] = 1.0
                im20 = ax[2, 0].imshow(masked_data, cmap=custom_cmap, aspect='auto')

                ax[2, 0].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                ax[2, 0].set_ylabel('latitude (\u00b0 N)', fontsize=35)
                ax[2, 0].invert_yaxis()
                ax[2, 0].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[2, 0].set_xticks(x_ticks_pos)
                ax[2, 0].set_yticks(y_ticks_pos)
                ax[2, 0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[2, 0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                ax[0, 0].text(0.025, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 0].text(0.025, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
                ax[2, 0].text(0.025, 0.9, r"A3", transform=ax[2, 0].transAxes, fontsize=40, color='black', weight="bold")

            elif dist_key == distribution_flags[0] and sce_key == 'b1het':
                # SUBPLOT 01
                im01 = ax[0, 1].imshow(inf_rho_avg_lattice.T, cmap='coolwarm', aspect='auto')
                im01.set_clim(vmin=0.0, vmax=1.0)
                cbar01 = fig.colorbar(im01, ax=ax[0, 1], shrink=1.0)
                cbar01.ax.tick_params(labelsize=18)
                #cbar01.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                masked_data = np.ma.masked_where(np.ones_like(inf_rho_avg_lattice.T), inf_rho_avg_lattice)
                masked_data[28, 28] = 1.0
                im01 = ax[0, 1].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0, aspect='auto') # extent=extent)

                ax[0, 1].set_xlabel("longitude (\u00b0 W)", fontsize=35)
                #ax[0, 1].set_ylabel("latitude (\u00b0 N)", fontsize=25)
                ax[0, 1].invert_yaxis()
                ax[0, 1].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[0, 1].set_xticks(x_ticks_pos)
                ax[0, 1].set_yticks(y_ticks_pos)
                ax[0, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[0, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                ax[0, 1].set_title("Beta memory-less", fontsize=40)

                # SUBPLOT 11
                hb11 = ax[1, 1].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
                hb11.set_clim(vmin=0.0, vmax=1.0)
                cbar11 = fig.colorbar(hb11, ax=ax[1, 1])
                cbar11.ax.tick_params(labelsize=18)
                #cbar11.set_label(r'infected $\langle\rho\rangle$', fontsize=25)

                # Compute the mean value for each hexbin
                xbins = hb11.get_offsets()[:, 0]
                ybins = hb11.get_offsets()[:, 1]
                mean_values = hb11.get_array()
                mean_rho_for_hexbins = []

                for i in range(len(mean_values)):
                    if i == len(mean_values) - 1:  # Handle the last bin separately
                        condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
                    else:
                        condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

                    indices = np.where(condition)
                    if len(indices[0]) > 0:
                        mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
                    else:
                        mean_rho_for_hexbins.append(0.0)

                model_1 = LinearRegression()
                model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
                y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
                ax[1, 1].plot(attr_l, y_pred_11, color='indigo', linestyle='--', linewidth=2)
                r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
                ax[1, 1].text(0.5, 0.85, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 1].transAxes, fontsize=30, color='black')

                ax[1, 1].set_xlabel(r'$A$', fontsize=35)
                #ax[1, 1].set_ylabel('mean total cases', fontsize=25)
                ax[1, 1].tick_params(axis='both', labelsize=18)

                # SUBPLOT 21
                #masked_data = np.ma.masked_where(pt_avg_lattice > threshold, pt_avg_lattice)
                # Plot the masked data
                #im21 = ax[2, 1].imshow(masked_data.T, cmap=custom_cmap, aspect='auto')
                # Optionally, set colorbar for the custom colormap
                #cbar21 = fig.colorbar(im21, ax=ax[2, 1], shrink=1.0)
                #cbar21.ax.tick_params(labelsize=18)
                
                vmin = 200 #np.nanmin(pt_avg_lattice)
                vmax = 350 #np.nanmax(pt_avg_lattice)
                im21 = ax[2, 1].imshow(pt_avg_lattice.T, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                #im21.set_clim(vmin=0.0, vmax=1.0)
                cbar21 = fig.colorbar(im21, ax=ax[2, 1], shrink=1.0)
                cbar21.ax.tick_params(labelsize=18)
                #cbar21.set_label(r'invader $\langle\rho\rangle$', fontsize=25)
                masked_data = np.ma.masked_where(np.ones_like(pt_avg_lattice.T), pt_avg_lattice)
                masked_data[28, 28] = 1.0
                im21 = ax[2, 1].imshow(masked_data, cmap=custom_cmap, aspect='auto')

                ax[2, 1].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                #ax[2, 1].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[2, 1].invert_yaxis()
                ax[2, 1].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[2, 1].set_xticks(x_ticks_pos)
                ax[2, 1].set_yticks(y_ticks_pos)
                ax[2, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[2, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                ax[0, 1].text(0.025, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 1].text(0.025, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
                ax[2, 1].text(0.025, 0.9, r"B3", transform=ax[2, 1].transAxes, fontsize=40, color='black', weight="bold")

            elif dist_key == distribution_flags[1] and sce_key == 'depr':
                # SUBPLOT 02
                im02 = ax[0, 2].imshow(inf_rho_avg_lattice.T, cmap='coolwarm', aspect='auto')
                im02.set_clim(vmin=0.0, vmax=1.0)
                cbar02 = fig.colorbar(im02, ax=ax[0, 2], shrink=1.0)
                cbar02.ax.tick_params(labelsize=18)
                cbar02.set_label(r'infected $\langle\rho\rangle$', fontsize=35)

                masked_data = np.ma.masked_where(np.ones_like(inf_rho_avg_lattice.T), inf_rho_avg_lattice)
                masked_data[28, 28] = 1.0
                im02 = ax[0, 2].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0, aspect='auto') # extent=extent)

                ax[0, 2].set_xlabel("longitude (\u00b0 W)", fontsize=35)
                #ax[0, 2].set_ylabel("latitude (\u00b0 N)", fontsize=25)
                ax[0, 2].invert_yaxis()
                ax[0, 2].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[0, 2].set_xticks(x_ticks_pos)
                ax[0, 2].set_yticks(y_ticks_pos)
                ax[0, 2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[0, 2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                ax[0, 2].set_title("Gaussian d-EPR", fontsize=40)

                # SUBPLOT 12
                hb12 = ax[1, 2].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
                hb12.set_clim(vmin=0.0, vmax=1.0)
                cbar12 = fig.colorbar(hb12, ax=ax[1, 2])
                cbar12.ax.tick_params(labelsize=18)
                cbar12.set_label(r'infected $\langle\rho\rangle$', fontsize=35)

                # Compute the mean value for each hexbin
                xbins = hb12.get_offsets()[:, 0]
                ybins = hb12.get_offsets()[:, 1]
                mean_values = hb12.get_array()
                mean_rho_for_hexbins = []

                for i in range(len(mean_values)):
                    if i == len(mean_values) - 1:  # Handle the last bin separately
                        condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
                    else:
                        condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

                    indices = np.where(condition)
                    if len(indices[0]) > 0:
                        mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
                    else:
                        mean_rho_for_hexbins.append(0.0)

                model_1 = LinearRegression()
                model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
                y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
                ax[1, 2].plot(attr_l, y_pred_11, color='indigo', linestyle='--', linewidth=2)
                r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
                ax[1, 2].text(0.5, 0.85, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 2].transAxes, fontsize=30, color='black')

                ax[1, 2].set_xlabel(r'$A$', fontsize=35)
                #ax[1, 2].set_ylabel('mean total cases', fontsize=25)
                ax[1, 2].tick_params(axis='both', labelsize=18)

                # SUBPLOT 22
                vmin = 200 #np.nanmin(pt_avg_lattice)
                vmax = 350 #np.nanmax(pt_avg_lattice)
                im22 = ax[2, 2].imshow(pt_avg_lattice.T, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                #im22.set_clim(vmin=0.0, vmax=1.0)
                cbar22 = fig.colorbar(im22, ax=ax[2, 2], shrink=1.0)
                cbar22.ax.tick_params(labelsize=18)
                cbar22.set_label(r'$\langle t_{{peak}}\rangle$', fontsize=35)

                masked_data = np.ma.masked_where(np.ones_like(pt_avg_lattice.T), pt_avg_lattice)
                masked_data[28, 28] = 1.0
                im22 = ax[2, 2].imshow(masked_data, cmap=custom_cmap, aspect='auto')

                ax[2, 2].set_xlabel('longitude (\u00b0 W)', fontsize=35)
                #ax[2, 2].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[2, 2].invert_yaxis()
                ax[2, 2].tick_params(axis='both', labelsize=18)

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[2, 2].set_xticks(x_ticks_pos)
                ax[2, 2].set_yticks(y_ticks_pos)
                ax[2, 2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[2, 2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

                ax[0, 2].text(0.025, 0.9, r"C1", transform=ax[0, 2].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 2].text(0.025, 0.9, r"C2", transform=ax[1, 2].transAxes, fontsize=40, color='black', weight="bold")
                ax[2, 2].text(0.025, 0.9, r"C3", transform=ax[2, 2].transAxes, fontsize=40, color='black', weight="bold")

    #fig.subplots_adjust(hspace=0.4, wspace=0.4)

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
    base_name = 'deprf3B'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel5(
        distribution_flags=None,
        scenario_flags=None,
        stats_flag=False,
        f_inf_flag=False,
        a_inf_flag=False,
        ):
    """ 2x3 panel
    f00: home (outside) infection fraction rho profile for beta
    f01: home (outside) visitation frequency rho profile for beta
    f02: home (outside) attractiveness rho profile for beta
    f10: home (outside) infection fraction rho profile for beta
    f11: home (outside) visitation frequency rho profile for beta
    f12: home (outside) attractiveness rho profile for beta
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    R0 = 1.2
    r_0 = 0.0

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            out_sim_data = ut.load_depr_chapter_panel5_data(
                epi_fullname, 
                f_inf_flag=f_inf_flag,
                a_inf_flag=a_inf_flag,
                )
    
            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    ch_dict = {'b1hom': 'teal', 'b1het': 'deepskyblue', 'depr': 'dodgerblue', 'uniform': 'royalblue', 'plain': 'slateblue'}
    co_dict = {'b1hom': 'lightcoral', 'b1het': 'maroon', 'depr': 'firebrick', 'uniform': 'orangered', 'plain': 'sienna'}

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25, 14))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            infected_h_per_rho_sim = []
            infected_o_per_rho_sim = []

            if stats_flag:
                f_inf_h_stats_per_rho_sim = []
                f_inf_o_stats_per_rho_sim = []
                a_inf_h_stats_per_rho_sim = []
                a_inf_o_stats_per_rho_sim = []
            if f_inf_flag:
                f_inf_h_dist_per_rho_sim = []
                f_inf_o_dist_per_rho_sim = []
            if a_inf_flag:
                a_inf_h_dist_per_rho_sim = []
                a_inf_o_dist_per_rho_sim = []
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel5_results(
                    out_sim_data=out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    infected_h_per_rho_sim=infected_h_per_rho_sim,
                    infected_o_per_rho_sim=infected_o_per_rho_sim,
                    f_inf_flag=f_inf_flag,
                    f_inf_h_dist_per_rho_sim=f_inf_h_dist_per_rho_sim,
                    f_inf_o_dist_per_rho_sim=f_inf_o_dist_per_rho_sim,
                    a_inf_flag=a_inf_flag,
                    a_inf_h_dist_per_rho_sim=a_inf_h_dist_per_rho_sim,
                    a_inf_o_dist_per_rho_sim=a_inf_o_dist_per_rho_sim,
                )

            processed_results = ut.compute_depr_chapter_panel5_stats(
                agents_per_rho_sim,
                infected_per_rho_sim,
                infected_h_per_rho_sim=infected_h_per_rho_sim,
                infected_o_per_rho_sim=infected_o_per_rho_sim,
                prevalence_cutoff=prevalence_cutoff, 
                f_inf_flag=f_inf_flag, 
                f_inf_h_dist_per_rho_sim=f_inf_h_dist_per_rho_sim,
                f_inf_o_dist_per_rho_sim=f_inf_o_dist_per_rho_sim,
                a_inf_flag=a_inf_flag,
                a_inf_h_dist_per_rho_sim=a_inf_h_dist_per_rho_sim,
                a_inf_o_dist_per_rho_sim=a_inf_o_dist_per_rho_sim,
                )
            
            infected_h_fraction_avg_per_rho = processed_results['inf_h_frac_avg_per_rho']
            infected_h_fraction_l95_per_rho = processed_results['inf_h_frac_l95_per_rho']
            infected_h_fraction_u95_per_rho = processed_results['inf_h_frac_u95_per_rho']
            infected_o_fraction_avg_per_rho = processed_results['inf_o_frac_avg_per_rho']
            infected_o_fraction_l95_per_rho = processed_results['inf_o_frac_l95_per_rho']
            infected_o_fraction_u95_per_rho = processed_results['inf_o_frac_u95_per_rho']
            home_global_fraction = processed_results['inf_h_frac_global']
            out_global_fraction = processed_results['inf_o_frac_global']

            if f_inf_flag:
                f_inf_h_avg_per_rho = processed_results['f_inf_h_avg_per_rho']
                f_inf_h_l95_per_rho = processed_results['f_inf_h_l95_per_rho']
                f_inf_h_u95_per_rho = processed_results['f_inf_h_u95_per_rho']
                f_inf_h_avg_global = processed_results['f_inf_h_avg_global']
                f_inf_h_l95_global = processed_results['f_inf_h_l95_global']
                f_inf_h_u95_global = processed_results['f_inf_h_u95_global']
                f_inf_o_avg_per_rho = processed_results['f_inf_o_avg_per_rho']
                f_inf_o_l95_per_rho = processed_results['f_inf_o_l95_per_rho']
                f_inf_o_u95_per_rho = processed_results['f_inf_o_u95_per_rho']
                f_inf_o_avg_global = processed_results['f_inf_o_avg_global']
                f_inf_o_l95_global = processed_results['f_inf_o_l95_global']
                f_inf_o_u95_global = processed_results['f_inf_o_u95_global']

            if a_inf_flag:
                a_inf_h_avg_per_rho = processed_results['a_inf_h_avg_per_rho']
                a_inf_h_l95_per_rho = processed_results['a_inf_h_l95_per_rho']
                a_inf_h_u95_per_rho = processed_results['a_inf_h_u95_per_rho']
                a_inf_h_avg_global = processed_results['a_inf_h_avg_global']
                a_inf_h_l95_global = processed_results['a_inf_h_l95_global']
                a_inf_h_u95_global = processed_results['a_inf_h_u95_global']
                a_inf_o_avg_per_rho = processed_results['a_inf_o_avg_per_rho']
                a_inf_o_l95_per_rho = processed_results['a_inf_o_l95_per_rho']
                a_inf_o_u95_per_rho = processed_results['a_inf_o_u95_per_rho']
                a_inf_o_avg_global = processed_results['a_inf_o_avg_global']
                a_inf_o_l95_global = processed_results['a_inf_o_l95_global']
                a_inf_o_u95_global = processed_results['a_inf_o_u95_global']

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':

                    ax[0, 0].scatter(rho_bins, infected_h_fraction_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[0, 0].fill_between(rho_bins, infected_h_fraction_l95_per_rho, infected_h_fraction_u95_per_rho, color=ch_dict[sce_key], alpha=0.2,)
                    ax[0, 0].scatter(rho_bins, infected_o_fraction_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[0, 0].fill_between(rho_bins, infected_o_fraction_l95_per_rho, infected_o_fraction_u95_per_rho, color=co_dict[sce_key], alpha=0.2,)

                    ax[0, 0].axhline(home_global_fraction, color=ch_dict[sce_key], linestyle='--')
                    ax[0, 0].axhline(out_global_fraction, color=co_dict[sce_key], linestyle='--')

                    ax[0, 1].scatter(rho_bins, f_inf_h_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[0, 1].fill_between(rho_bins, f_inf_h_l95_per_rho, f_inf_h_u95_per_rho, color=ch_dict[sce_key], alpha=0.2)
                    ax[0, 1].scatter(rho_bins, f_inf_o_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[0, 1].fill_between(rho_bins, f_inf_o_l95_per_rho, f_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)

                    ax[0, 2].scatter(rho_bins, a_inf_h_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[0, 2].fill_between(rho_bins, a_inf_h_l95_per_rho, a_inf_h_u95_per_rho, color=ch_dict[sce_key], alpha=0.2)
                    ax[0, 2].scatter(rho_bins, a_inf_o_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[0, 2].fill_between(rho_bins, a_inf_o_l95_per_rho, a_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[0, 0].axhline(out_global_fraction, color=co_dict[sce_key], linestyle='--', label=r'out {0}'.format(label_dict[sce_key]))

                    ax[0, 1].axhline(f_inf_o_avg_global, color=co_dict[sce_key], linestyle='--', label=r'out {0}'.format(label_dict[sce_key]))
                    ax[0, 1].fill_between(rho_bins, f_inf_o_l95_per_rho, f_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)

                    ax[0, 2].axhline(a_inf_h_avg_global, color=ch_dict[sce_key], linestyle='--', label=r'home {0}'.format(label_dict[sce_key]))
                    ax[0, 2].fill_between(rho_bins, a_inf_h_l95_global, a_inf_h_u95_global, color=ch_dict[sce_key], alpha=0.2)
                    ax[0, 2].axhline(a_inf_o_avg_global, color=co_dict[sce_key], linestyle='--', label=r'out {0}'.format(label_dict[sce_key]))
                    ax[0, 2].fill_between(rho_bins, a_inf_o_l95_global, a_inf_o_u95_global, color=co_dict[sce_key], alpha=0.2)

            else:
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':

                    ax[1, 0].scatter(rho_bins, infected_h_fraction_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[1, 0].fill_between(rho_bins, infected_h_fraction_l95_per_rho, infected_h_fraction_u95_per_rho, color=ch_dict[sce_key], alpha=0.2,)
                    ax[1, 0].scatter(rho_bins, infected_o_fraction_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[1, 0].fill_between(rho_bins, infected_o_fraction_l95_per_rho, infected_o_fraction_u95_per_rho, color=co_dict[sce_key], alpha=0.2,)

                    ax[1, 0].axhline(home_global_fraction, color=ch_dict[sce_key], linestyle='--')
                    ax[1, 0].axhline(out_global_fraction, color=co_dict[sce_key], linestyle='--')

                    ax[1, 1].scatter(rho_bins, f_inf_h_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[1, 1].fill_between(rho_bins, f_inf_h_l95_per_rho, f_inf_h_u95_per_rho, color=ch_dict[sce_key], alpha=0.2)
                    ax[1, 1].scatter(rho_bins, f_inf_o_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[1, 1].fill_between(rho_bins, f_inf_o_l95_per_rho, f_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)

                    ax[1, 2].scatter(rho_bins, a_inf_h_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[1, 2].fill_between(rho_bins, a_inf_h_l95_per_rho, a_inf_h_u95_per_rho, color=ch_dict[sce_key], alpha=0.2)
                    ax[1, 2].scatter(rho_bins, a_inf_o_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[1, 2].fill_between(rho_bins, a_inf_o_l95_per_rho, a_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)
                
                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[1, 0].axhline(out_global_fraction, color=co_dict[sce_key], linestyle='--', label='out {0}'.format(label_dict[sce_key]))

                    ax[1, 1].axhline(f_inf_o_avg_global, color=co_dict[sce_key], linestyle='--', label='out {0}'.format(label_dict[sce_key]))
                    ax[1, 1].fill_between(rho_bins, f_inf_o_l95_global, f_inf_o_u95_global, color=co_dict[sce_key], alpha=0.2)

                    ax[1, 2].axhline(a_inf_h_avg_global, color=ch_dict[sce_key], linestyle='--', label='home {0}'.format(label_dict[sce_key]))
                    ax[1, 2].fill_between(rho_bins, a_inf_h_l95_global, a_inf_h_u95_global, color=ch_dict[sce_key], alpha=0.2)
                    ax[1, 2].axhline(a_inf_o_avg_global, color=co_dict[sce_key], linestyle='--', label='out {0}'.format(label_dict[sce_key]))
                    ax[1, 2].fill_between(rho_bins, a_inf_o_l95_global, a_inf_o_u95_global, color=co_dict[sce_key], alpha=0.2)

    #ax[0, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 0].text(0.05, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0, 0].set_ylabel(r'$I_{h(o),\rho}/I_{\rho}$', fontsize=25)
    ax[0, 0].set_xlim(0.0, 1.0)
    ax[0, 0].set_ylim(0.0, 1.0)
    ax[0, 0].tick_params(axis='both', labelsize=20)
    ax[0, 0].legend(loc='lower center', fontsize=15)

    ax[0, 1].text(0.05, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 1].set_ylabel(r'$f_{inf,l,\rho}$', fontsize=25)
    ax[0, 1].set_xlim(0.0, 1.0)
    ax[0, 1].set_ylim(0.0, 1.0)
    ax[0, 1].tick_params(axis='both', labelsize=20)
    ax[0, 1].legend(loc='upper right', fontsize=15)

    ax[0, 2].text(0.05, 0.9, r"C1", transform=ax[0, 2].transAxes, fontsize=40, color='black', weight="bold")
    ax[0, 2].set_ylim(0.0, 0.007)
    #ax[0, 2].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 2].set_ylabel(r'$A_{inf,l,\rho}$', fontsize=25)
    ax[0, 2].set_xlim(0.0, 1.0)
    ax[0, 2].tick_params(axis='both', labelsize=20)
    ax[0, 2].legend(loc='upper right', fontsize=15)

    ax[1, 0].text(0.9, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 0].set_ylabel(r'$I_{h(o),\rho}/I_{\rho}$', fontsize=25)
    ax[1, 0].set_xlim(0.0, 1.0)
    ax[1, 0].set_ylim(0.0, 1.0)
    ax[1, 0].tick_params(axis='both', labelsize=20)
    ax[1, 0].legend(loc='lower left', fontsize=15)

    ax[1, 1].text(0.05, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 1].set_ylabel(r'$f_{inf,l,\rho}$', fontsize=25)
    ax[1, 1].set_xlim(0.0, 1.0)
    ax[1, 1].set_ylim(0.0, 1.0)
    ax[1, 1].tick_params(axis='both', labelsize=20)
    ax[1, 1].legend(fontsize=15)

    ax[1, 2].text(0.05, 0.9, r"C2", transform=ax[1, 2].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 2].set_ylim(0.0, 0.007)
    ax[1, 2].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 2].set_ylabel(r'$A_{inf,l,\rho}$', fontsize=25)
    ax[1, 2].set_xlim(0.0, 1.0)
    ax[1, 2].tick_params(axis='both', labelsize=20)
    ax[1, 2].legend(loc='upper right', fontsize=15)

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
    base_name = 'deprf5_extra'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel6(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: od-flows rho profile for beta
    f1: od-flows rho profile for gaussian
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel6extra_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    chh_dict = {'b1hom': 'cornflowerblue', 'b1het': 'deepskyblue', 'depr': 'dodgerblue', 'plain': 'royalblue', 'uniform': 'mediumblue'}
    cho_dict = {'b1hom': 'darkseagreen', 'b1het': 'olivedrab', 'depr': 'teal', 'plain': 'lightseagreen', 'uniform': 'seagreen'}
    coh_dict = {'b1hom': 'crimson', 'b1het': 'orchid', 'depr': 'deeppink', 'plain': 'violet', 'uniform': 'darkorchid'}
    coo_dict = {'b1hom': 'lightcoral', 'b1het': 'salmon', 'depr': 'firebrick', 'plain': 'maroon', 'uniform': 'orangered'}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 12))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            events_hh_per_rho_sim = []
            events_ho_per_rho_sim = []
            events_oh_per_rho_sim = []
            events_oo_per_rho_sim = []
            
            f_trip_hh_dist_per_rho_sim = []
            f_trip_ho_dist_per_rho_sim = []
            f_trip_oh_dist_per_rho_sim = []
            f_trip_oo_dist_per_rho_sim = []
            da_trip_hh_dist_per_rho_sim = []
            da_trip_ho_dist_per_rho_sim = []
            da_trip_oh_dist_per_rho_sim = []
            da_trip_oo_dist_per_rho_sim = []

            a_exp_dist_per_rho_sim = []
            sum_p_exp_per_rho_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel6extra_results(
                    out_sim_data, 
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    events_hh_per_rho_sim=events_hh_per_rho_sim,
                    events_ho_per_rho_sim=events_ho_per_rho_sim,
                    events_oh_per_rho_sim=events_oh_per_rho_sim,
                    events_oo_per_rho_sim=events_oo_per_rho_sim,
                    f_trip_hh_dist_per_rho_sim=f_trip_hh_dist_per_rho_sim,
                    f_trip_ho_dist_per_rho_sim=f_trip_ho_dist_per_rho_sim,
                    f_trip_oh_dist_per_rho_sim=f_trip_oh_dist_per_rho_sim,
                    f_trip_oo_dist_per_rho_sim=f_trip_oo_dist_per_rho_sim,
                    da_trip_hh_dist_per_rho_sim=da_trip_hh_dist_per_rho_sim,
                    da_trip_ho_dist_per_rho_sim=da_trip_ho_dist_per_rho_sim,
                    da_trip_oh_dist_per_rho_sim=da_trip_oh_dist_per_rho_sim,
                    da_trip_oo_dist_per_rho_sim=da_trip_oo_dist_per_rho_sim,
                    a_exp_dist_per_rho_sim=a_exp_dist_per_rho_sim,
                    sum_p_exp_per_rho_sim=sum_p_exp_per_rho_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel6extra_stats(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim,
                events_hh_per_rho_sim=events_hh_per_rho_sim,
                events_ho_per_rho_sim=events_ho_per_rho_sim,
                events_oh_per_rho_sim=events_oh_per_rho_sim,
                events_oo_per_rho_sim=events_oo_per_rho_sim,
                f_trip_hh_dist_per_rho_sim=f_trip_hh_dist_per_rho_sim,
                f_trip_ho_dist_per_rho_sim=f_trip_ho_dist_per_rho_sim,
                f_trip_oh_dist_per_rho_sim=f_trip_oh_dist_per_rho_sim,
                f_trip_oo_dist_per_rho_sim=f_trip_oo_dist_per_rho_sim,
                da_trip_hh_dist_per_rho_sim=da_trip_hh_dist_per_rho_sim,
                da_trip_ho_dist_per_rho_sim=da_trip_ho_dist_per_rho_sim,
                da_trip_oh_dist_per_rho_sim=da_trip_oh_dist_per_rho_sim,
                da_trip_oo_dist_per_rho_sim=da_trip_oo_dist_per_rho_sim,
                a_exp_dist_per_rho_sim=a_exp_dist_per_rho_sim,
                sum_p_exp_per_rho_sim=sum_p_exp_per_rho_sim,
                prevalence_cutoff=0.025, 
                )
            
            events_hh_avg_per_rho = processed_results['hh_avg_per_rho']
            events_hh_l95_per_rho = processed_results['hh_l95_per_rho']
            events_hh_u95_per_rho = processed_results['hh_u95_per_rho']
            events_ho_avg_per_rho = processed_results['ho_avg_per_rho']
            events_ho_l95_per_rho = processed_results['ho_l95_per_rho']
            events_ho_u95_per_rho = processed_results['ho_u95_per_rho']
            events_oh_avg_per_rho = processed_results['oh_avg_per_rho']
            events_oh_l95_per_rho = processed_results['oh_l95_per_rho']
            events_oh_u95_per_rho = processed_results['oh_u95_per_rho']
            events_oo_avg_per_rho = processed_results['oo_avg_per_rho']
            events_oo_l95_per_rho = processed_results['oo_l95_per_rho']
            events_oo_u95_per_rho = processed_results['oo_u95_per_rho']

            hh_avg_global = processed_results['hh_avg_global']
            hh_l95_global = processed_results['hh_l95_global']
            hh_u95_global = processed_results['hh_u95_global']
            ho_avg_global = processed_results['ho_avg_global']
            ho_l95_global = processed_results['ho_l95_global']
            ho_u95_global = processed_results['ho_u95_global']
            oh_avg_global = processed_results['oh_avg_global']
            oh_l95_global = processed_results['oh_l95_global']
            oh_u95_global = processed_results['oh_u95_global']
            oo_avg_global = processed_results['oo_avg_global']
            oo_l95_global = processed_results['oo_l95_global']
            oo_u95_global = processed_results['oo_u95_global']

            f_trip_hh_avg_per_rho = processed_results['f_trip_hh_avg_per_rho']
            f_trip_hh_l95_per_rho = processed_results['f_trip_hh_l95_per_rho']
            f_trip_hh_u95_per_rho = processed_results['f_trip_hh_u95_per_rho']
            f_trip_ho_avg_per_rho = processed_results['f_trip_ho_avg_per_rho']
            f_trip_ho_l95_per_rho = processed_results['f_trip_ho_l95_per_rho']
            f_trip_ho_u95_per_rho = processed_results['f_trip_ho_u95_per_rho']
            f_trip_oh_avg_per_rho = processed_results['f_trip_oh_avg_per_rho']
            f_trip_oh_l95_per_rho = processed_results['f_trip_oh_l95_per_rho']
            f_trip_oh_u95_per_rho = processed_results['f_trip_oh_u95_per_rho']
            f_trip_oo_avg_per_rho = processed_results['f_trip_oo_avg_per_rho']
            f_trip_oo_l95_per_rho = processed_results['f_trip_oo_l95_per_rho']
            f_trip_oo_u95_per_rho = processed_results['f_trip_oo_u95_per_rho']
            da_trip_hh_avg_per_rho = processed_results['da_trip_hh_avg_per_rho']
            da_trip_hh_l95_per_rho = processed_results['da_trip_hh_l95_per_rho']
            da_trip_hh_u95_per_rho = processed_results['da_trip_hh_u95_per_rho']
            da_trip_ho_avg_per_rho = processed_results['da_trip_ho_avg_per_rho']
            da_trip_ho_l95_per_rho = processed_results['da_trip_ho_l95_per_rho']
            da_trip_ho_u95_per_rho = processed_results['da_trip_ho_u95_per_rho']
            da_trip_oh_avg_per_rho = processed_results['da_trip_oh_avg_per_rho']
            da_trip_oh_l95_per_rho = processed_results['da_trip_oh_l95_per_rho']
            da_trip_oh_u95_per_rho = processed_results['da_trip_oh_u95_per_rho']
            da_trip_oo_avg_per_rho = processed_results['da_trip_oo_avg_per_rho']
            da_trip_oo_l95_per_rho = processed_results['da_trip_oo_l95_per_rho']
            da_trip_oo_u95_per_rho = processed_results['da_trip_oo_u95_per_rho']

            a_exp_dist_per_rho = processed_results['a_exp_dist_per_rho']
            a_exp_avg_per_rho = processed_results['a_exp_avg_per_rho']
            a_exp_l95_per_rho = processed_results['a_exp_l95_per_rho']
            a_exp_u95_per_rho = processed_results['a_exp_u95_per_rho']
            p_exp_avg_per_rho = processed_results['p_exp_avg_per_rho']
            p_exp_l95_per_rho = processed_results['p_exp_l95_per_rho']
            p_exp_u95_per_rho = processed_results['p_exp_u95_per_rho']

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    ax[0].scatter(rho_bins, events_hh_avg_per_rho, marker='o', color=chh_dict[sce_key], label=r'h-h {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, events_hh_l95_per_rho, events_hh_u95_per_rho, color=chh_dict[sce_key], alpha=0.2)
                    ax[0].scatter(rho_bins, events_ho_avg_per_rho, marker='s', color=cho_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, events_ho_l95_per_rho, events_ho_u95_per_rho, color=cho_dict[sce_key], alpha=0.2)
                    ax[0].scatter(rho_bins, events_oh_avg_per_rho, marker='v', color=coh_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, events_oh_l95_per_rho, events_oh_u95_per_rho, color=coh_dict[sce_key], alpha=0.2)
                    ax[0].scatter(rho_bins, events_oo_avg_per_rho, marker='P', color=coo_dict[sce_key], label=r'o-o {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, events_oo_l95_per_rho, events_oo_u95_per_rho, color=coo_dict[sce_key], alpha=0.2)

                    intersection_index = np.argmin(np.abs(events_hh_avg_per_rho + events_oh_avg_per_rho - events_ho_avg_per_rho - events_oo_avg_per_rho))
                    intersection_rho = rho_bins[intersection_index]
                    ax[0].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                    ax[0].axhline(0.25, color='gray', linestyle='dotted', alpha=1.0)
                    print("Intersection rho={0}".format(intersection_rho))

                    #ax[1].scatter(rho_bins, p_exp_avg_per_rho, marker='o', color='slateblue', label=r'$P_{{exp}}$ {0}'.format(label_dict[sce_key]))
                    #ax[1].fill_between(rho_bins, p_exp_l95_per_rho, p_exp_u95_per_rho, color='slateblue', alpha=0.2)

                    #bins = 40
                    #density = True
                    #ax[2].hist(a_exp_dist_per_rho[1], color='firebrick', bins=bins, density=density, label='top explorers', alpha=0.5)
                    #ax[2].hist(a_exp_dist_per_rho[0], color='dodgerblue', bins=bins, density=density, label='top returners', alpha=0.5)

                    ax[1].scatter(rho_bins, f_trip_hh_avg_per_rho, marker=marker_dict[sce_key], color=chh_dict[sce_key], label=r'h-h {0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, f_trip_hh_l95_per_rho, f_trip_hh_u95_per_rho, color=chh_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, f_trip_ho_avg_per_rho, marker=marker_dict[sce_key], color=cho_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, f_trip_ho_l95_per_rho, f_trip_ho_u95_per_rho, color=cho_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, f_trip_oh_avg_per_rho, marker=marker_dict[sce_key], color=coh_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, f_trip_oh_l95_per_rho, f_trip_oh_u95_per_rho, color=coh_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, f_trip_oo_avg_per_rho, marker=marker_dict[sce_key], color=coo_dict[sce_key], label=r'o-o {0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, f_trip_oo_l95_per_rho, f_trip_oo_u95_per_rho, color=coo_dict[sce_key], alpha=0.2)

                    intersection_index = np.argmin(np.abs(f_trip_hh_avg_per_rho + f_trip_oh_avg_per_rho - f_trip_ho_avg_per_rho - f_trip_oo_avg_per_rho))
                    intersection_rho = rho_bins[intersection_index]
                    ax[1].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                    ax[1].axhline(0.25, color='gray', linestyle='dotted', alpha=1.0)
                    print("Intersection rho={0}".format(intersection_rho))

                    ax[2].scatter(rho_bins, da_trip_hh_avg_per_rho, marker=marker_dict[sce_key], color=chh_dict[sce_key], label=r'h-h {0}'.format(label_dict[sce_key]),)
                    ax[2].fill_between(rho_bins, da_trip_hh_l95_per_rho, da_trip_hh_u95_per_rho, color=chh_dict[sce_key], alpha=0.2)
                    ax[2].scatter(rho_bins, da_trip_ho_avg_per_rho, marker=marker_dict[sce_key], color=cho_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]),)
                    ax[2].fill_between(rho_bins, da_trip_ho_l95_per_rho, da_trip_ho_u95_per_rho, color=cho_dict[sce_key], alpha=0.2)
                    ax[2].scatter(rho_bins, da_trip_oh_avg_per_rho, marker=marker_dict[sce_key], color=coh_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]),)
                    ax[2].fill_between(rho_bins, da_trip_oh_l95_per_rho, da_trip_oh_u95_per_rho, color=coh_dict[sce_key], alpha=0.2)
                    ax[2].scatter(rho_bins, da_trip_oo_avg_per_rho, marker=marker_dict[sce_key], color=coo_dict[sce_key], label=r'o-o {0}'.format(label_dict[sce_key]),)
                    ax[2].fill_between(rho_bins, da_trip_oo_l95_per_rho, da_trip_oo_u95_per_rho, color=coo_dict[sce_key], alpha=0.2)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[0].axhline(hh_avg_global, color=chh_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, hh_l95_global, hh_u95_global, color=chh_dict[sce_key], alpha=0.2)
                    ax[0].axhline(ho_avg_global, color=cho_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, ho_l95_global, ho_u95_global, color=cho_dict[sce_key], alpha=0.2)
                    ax[0].axhline(oh_avg_global, color=coh_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, oh_l95_global, oh_u95_global, color=coh_dict[sce_key], alpha=0.2)
                    ax[0].axhline(oo_avg_global, color=coo_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[0].fill_between(rho_bins, oo_l95_global, oo_u95_global, color=coo_dict[sce_key], alpha=0.2)

            elif dist_key == distribution_flags[1]:
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    ax[1].scatter(rho_bins, events_hh_avg_per_rho, marker='o', color=chh_dict[sce_key], label=r'h-h {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, events_hh_l95_per_rho, events_hh_u95_per_rho, color=chh_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, events_ho_avg_per_rho, marker='s', color=cho_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, events_ho_l95_per_rho, events_ho_u95_per_rho, color=cho_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, events_oh_avg_per_rho, marker='v', color=coh_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, events_oh_l95_per_rho, events_oh_u95_per_rho, color=coh_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, events_oo_avg_per_rho, marker='P', color=coo_dict[sce_key], label=r'o-o {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, events_oo_l95_per_rho, events_oo_u95_per_rho, color=coo_dict[sce_key], alpha=0.2)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[1].axhline(hh_avg_global, color=chh_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, hh_l95_global, hh_u95_global, color=chh_dict[sce_key], alpha=0.2)
                    ax[1].axhline(ho_avg_global, color=cho_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, ho_l95_global, ho_u95_global, color=cho_dict[sce_key], alpha=0.2)
                    ax[1].axhline(oh_avg_global, color=coh_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, oh_l95_global, oh_u95_global, color=coh_dict[sce_key], alpha=0.2)
                    ax[1].axhline(oo_avg_global, color=coo_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax[1].fill_between(rho_bins, oo_l95_global, oo_u95_global, color=coo_dict[sce_key], alpha=0.2)

    ax[0].text(0.05, 0.95, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_xlabel(r'$\rho$', fontsize=35)
    ax[0].set_ylabel(r'event fraction', fontsize=35)
    #ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(fontsize=20, loc='upper center')

    ax[1].text(0.05, 0.95, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1].set_xlabel(r'$\rho$', fontsize=35)
    ax[1].set_ylabel(r'trip frequency while infected', fontsize=35)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(fontsize=20, loc='upper center')

    ax[2].text(0.05, 0.95, r"B", transform=ax[2].transAxes, fontsize=40, color='black', weight="bold")
    ax[2].set_xlabel(r'$\rho$', fontsize=35)
    ax[2].set_ylabel(r'$\Delta A$', fontsize=35)
    ax[2].tick_params(axis='both', labelsize=20)
    ax[2].legend(fontsize=20, loc='upper center')

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
    base_name = 'deprf8'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panelfinal(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: od-flows rho profile for beta
    f1: od-flows rho profile for gaussian
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panelfinal_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    chh_dict = {'b1hom': 'cornflowerblue', 'b1het': 'deepskyblue', 'depr': 'dodgerblue', 'plain': 'royalblue', 'uniform': 'mediumblue'}
    cho_dict = {'b1hom': 'darkseagreen', 'b1het': 'olivedrab', 'depr': 'teal', 'plain': 'lightseagreen', 'uniform': 'seagreen'}
    coh_dict = {'b1hom': 'crimson', 'b1het': 'orchid', 'depr': 'deeppink', 'plain': 'violet', 'uniform': 'darkorchid'}
    coo_dict = {'b1hom': 'lightcoral', 'b1het': 'salmon', 'depr': 'firebrick', 'plain': 'maroon', 'uniform': 'orangered'}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 8))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            events_hh_per_rho_sim = []
            events_ho_per_rho_sim = []
            events_oh_per_rho_sim = []
            events_oo_per_rho_sim = []
            
            f_trip_hh_dist_per_rho_sim = []
            f_trip_ho_dist_per_rho_sim = []
            f_trip_oh_dist_per_rho_sim = []
            f_trip_oo_dist_per_rho_sim = []
            da_trip_hh_dist_per_rho_sim = []
            da_trip_ho_dist_per_rho_sim = []
            da_trip_oh_dist_per_rho_sim = []
            da_trip_oo_dist_per_rho_sim = []

            a_exp_dist_per_rho_sim = []
            sum_p_exp_per_rho_sim = []

            infected_h_per_rho_sim = []
            infected_o_per_rho_sim = []
            sum_avg_foi_per_rho_sim = []
            sum_avg_pc_foi_per_rho_sim = []
            sum_avg_shared_per_rho_sim = []
            sum_avg_size_per_rho_sim = []
            sum_avg_t_pop_per_rho_sim = []
            sum_cum_i_pop_per_rho_sim = []
            sum_cum_shared_per_rho_sim = []
            sum_cum_size_per_rho_sim = []
            sum_cum_t_pop_per_rho_sim = []

            event_attractiveness_sim = []
            event_inf_pop_avg_rho_sim = []
            event_size_sim = []
            event_tot_pop_sim = []

            f_inf_tr_h_dist_per_rho_sim = []
            f_inf_tr_o_dist_per_rho_sim = []

            nevents_eff_per_rho_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panelfinal_results(
                    out_sim_data, 
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    events_hh_per_rho_sim=events_hh_per_rho_sim,
                    events_ho_per_rho_sim=events_ho_per_rho_sim,
                    events_oh_per_rho_sim=events_oh_per_rho_sim,
                    events_oo_per_rho_sim=events_oo_per_rho_sim,
                    f_trip_hh_dist_per_rho_sim=f_trip_hh_dist_per_rho_sim,
                    f_trip_ho_dist_per_rho_sim=f_trip_ho_dist_per_rho_sim,
                    f_trip_oh_dist_per_rho_sim=f_trip_oh_dist_per_rho_sim,
                    f_trip_oo_dist_per_rho_sim=f_trip_oo_dist_per_rho_sim,
                    da_trip_hh_dist_per_rho_sim=da_trip_hh_dist_per_rho_sim,
                    da_trip_ho_dist_per_rho_sim=da_trip_ho_dist_per_rho_sim,
                    da_trip_oh_dist_per_rho_sim=da_trip_oh_dist_per_rho_sim,
                    da_trip_oo_dist_per_rho_sim=da_trip_oo_dist_per_rho_sim,
                    a_exp_dist_per_rho_sim=a_exp_dist_per_rho_sim,
                    sum_p_exp_per_rho_sim=sum_p_exp_per_rho_sim,
                    infected_h_per_rho_sim=infected_h_per_rho_sim,
                    infected_o_per_rho_sim=infected_o_per_rho_sim,
                    sum_avg_foi_per_rho_sim=sum_avg_foi_per_rho_sim,
                    sum_avg_pc_foi_per_rho_sim=sum_avg_pc_foi_per_rho_sim,
                    sum_avg_shared_per_rho_sim=sum_avg_shared_per_rho_sim,
                    sum_avg_size_per_rho_sim=sum_avg_size_per_rho_sim,
                    sum_avg_t_pop_per_rho_sim=sum_avg_t_pop_per_rho_sim,
                    sum_cum_i_pop_per_rho_sim=sum_cum_i_pop_per_rho_sim,
                    sum_cum_shared_per_rho_sim=sum_cum_shared_per_rho_sim,
                    sum_cum_size_per_rho_sim=sum_cum_size_per_rho_sim,
                    sum_cum_t_pop_per_rho_sim=sum_cum_t_pop_per_rho_sim,
                    event_attractiveness_sim=event_attractiveness_sim,
                    event_inf_pop_avg_rho_sim=event_inf_pop_avg_rho_sim,
                    event_size_sim=event_size_sim,
                    event_tot_pop_sim=event_tot_pop_sim,
                    f_inf_tr_h_dist_per_rho_sim=f_inf_tr_h_dist_per_rho_sim,
                    f_inf_tr_o_dist_per_rho_sim=f_inf_tr_o_dist_per_rho_sim,
                    nevents_eff_per_rho_sim=nevents_eff_per_rho_sim,
                    )

            #processed_results = ut.compute_depr_chapter_panelfinal_stats(
            #    agents_per_rho_sim=agents_per_rho_sim,
            #    infected_per_rho_sim=infected_per_rho_sim,
            #    events_hh_per_rho_sim=events_hh_per_rho_sim,
            #    events_ho_per_rho_sim=events_ho_per_rho_sim,
            #    events_oh_per_rho_sim=events_oh_per_rho_sim,
            #    events_oo_per_rho_sim=events_oo_per_rho_sim,
            #    f_trip_hh_dist_per_rho_sim=f_trip_hh_dist_per_rho_sim,
            #    f_trip_ho_dist_per_rho_sim=f_trip_ho_dist_per_rho_sim,
            #    f_trip_oh_dist_per_rho_sim=f_trip_oh_dist_per_rho_sim,
            #    f_trip_oo_dist_per_rho_sim=f_trip_oo_dist_per_rho_sim,
            #    da_trip_hh_dist_per_rho_sim=da_trip_hh_dist_per_rho_sim,
            #    da_trip_ho_dist_per_rho_sim=da_trip_ho_dist_per_rho_sim,
            #    da_trip_oh_dist_per_rho_sim=da_trip_oh_dist_per_rho_sim,
            #    da_trip_oo_dist_per_rho_sim=da_trip_oo_dist_per_rho_sim,
            #    a_exp_dist_per_rho_sim=a_exp_dist_per_rho_sim,
            #    sum_p_exp_per_rho_sim=sum_p_exp_per_rho_sim,
            #    infected_h_per_rho_sim=infected_h_per_rho_sim,
            #    infected_o_per_rho_sim=infected_o_per_rho_sim,
            #    sum_avg_foi_per_rho_sim=sum_avg_foi_per_rho_sim,
            #    sum_avg_pc_foi_per_rho_sim=sum_avg_pc_foi_per_rho_sim,
            #    sum_avg_shared_per_rho_sim=sum_avg_shared_per_rho_sim,
            #    sum_avg_size_per_rho_sim=sum_avg_size_per_rho_sim,
            #    sum_avg_t_pop_per_rho_sim=sum_avg_t_pop_per_rho_sim,
            #    sum_cum_i_pop_per_rho_sim=sum_cum_i_pop_per_rho_sim,
            #    sum_cum_shared_per_rho_sim=sum_cum_shared_per_rho_sim,
            #    sum_cum_size_per_rho_sim=sum_cum_size_per_rho_sim,
            #    sum_cum_t_pop_per_rho_sim=sum_cum_t_pop_per_rho_sim,
            #    f_inf_tr_h_dist_per_rho_sim=f_inf_tr_h_dist_per_rho_sim,
            #    f_inf_tr_o_dist_per_rho_sim=f_inf_tr_o_dist_per_rho_sim,
            #    nevents_eff_per_rho_sim=nevents_eff_per_rho_sim,
            #    prevalence_cutoff=0.025, 
            #    )
    
            #a_exp_dist_per_rho = processed_results['a_exp_dist_per_rho']
            #a_exp_avg_per_rho = processed_results['a_exp_avg_per_rho']
            #a_exp_l95_per_rho = processed_results['a_exp_l95_per_rho']
            #a_exp_u95_per_rho = processed_results['a_exp_u95_per_rho']
            #p_exp_avg_per_rho = processed_results['p_exp_avg_per_rho']
            #p_exp_l95_per_rho = processed_results['p_exp_l95_per_rho']
            #p_exp_u95_per_rho = processed_results['p_exp_u95_per_rho']

            event_attractiveness_sim = list(chain.from_iterable(event_attractiveness_sim))
            event_tot_pop_sim = list(chain.from_iterable(event_tot_pop_sim))
            event_inf_pop_avg_rho_sim = list(chain.from_iterable(event_inf_pop_avg_rho_sim))
            event_size_sim = list(chain.from_iterable(event_size_sim))

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':

                    hb = ax[0].hexbin(event_attractiveness_sim, event_inf_pop_avg_rho_sim, gridsize=20, cmap='Blues')
                    cb = plt.colorbar(hb, ax=ax[0])
                    cb.set_label(label='counts', fontsize=25)
                    cb.ax.tick_params(labelsize=25)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    pass

            elif dist_key == distribution_flags[1]:
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    pass

                elif sce_key == 'b1hom' or sce_key == 'plain':
                   pass

    ax[0].text(0.05, 0.95, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0].set_ylabel(r'population', fontsize=25)
    #ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(fontsize=20, loc='lower center')

    ax[1].text(0.05, 0.95, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1].set_xlabel(r'$\rho$', fontsize=25)
    ax[1].set_ylabel(r'$P_{{exp}}$', fontsize=25)
    #ax[1].set_xlim(0.0, 1.0)
    #ax[1].set_ylim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(fontsize=20, loc='upper center')

    #ax[2].text(0.05, 0.95, r"C", transform=ax[2].transAxes, fontsize=40, color='black', weight="bold")
    #ax[2].set_xlabel(r'$A$', fontsize=25)
    #ax[2].set_ylabel(r'P(A)', fontsize=25)
    #ax[2].tick_params(axis='both', labelsize=20)
    #ax[2].legend(fontsize=20, loc='upper center')

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
    base_name = 'deprf_final'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel8_slides(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: od-flows rho profile for beta
    f1: od-flows rho profile for gaussian
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel6extra_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    chh_dict = {'b1hom': 'cornflowerblue', 'b1het': 'deepskyblue', 'depr': 'dodgerblue', 'plain': 'royalblue', 'uniform': 'mediumblue'}
    cho_dict = {'b1hom': 'darkseagreen', 'b1het': 'olivedrab', 'depr': 'teal', 'plain': 'lightseagreen', 'uniform': 'seagreen'}
    coh_dict = {'b1hom': 'crimson', 'b1het': 'orchid', 'depr': 'deeppink', 'plain': 'violet', 'uniform': 'darkorchid'}
    coo_dict = {'b1hom': 'lightcoral', 'b1het': 'salmon', 'depr': 'firebrick', 'plain': 'maroon', 'uniform': 'orangered'}

    fig, ax = plt.subplots(figsize=(15, 15))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            events_hh_per_rho_sim = []
            events_ho_per_rho_sim = []
            events_oh_per_rho_sim = []
            events_oo_per_rho_sim = []
            
            f_trip_hh_dist_per_rho_sim = []
            f_trip_ho_dist_per_rho_sim = []
            f_trip_oh_dist_per_rho_sim = []
            f_trip_oo_dist_per_rho_sim = []
            da_trip_hh_dist_per_rho_sim = []
            da_trip_ho_dist_per_rho_sim = []
            da_trip_oh_dist_per_rho_sim = []
            da_trip_oo_dist_per_rho_sim = []

            a_exp_dist_per_rho_sim = []
            sum_p_exp_per_rho_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel6extra_results(
                    out_sim_data, 
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    events_hh_per_rho_sim=events_hh_per_rho_sim,
                    events_ho_per_rho_sim=events_ho_per_rho_sim,
                    events_oh_per_rho_sim=events_oh_per_rho_sim,
                    events_oo_per_rho_sim=events_oo_per_rho_sim,
                    f_trip_hh_dist_per_rho_sim=f_trip_hh_dist_per_rho_sim,
                    f_trip_ho_dist_per_rho_sim=f_trip_ho_dist_per_rho_sim,
                    f_trip_oh_dist_per_rho_sim=f_trip_oh_dist_per_rho_sim,
                    f_trip_oo_dist_per_rho_sim=f_trip_oo_dist_per_rho_sim,
                    da_trip_hh_dist_per_rho_sim=da_trip_hh_dist_per_rho_sim,
                    da_trip_ho_dist_per_rho_sim=da_trip_ho_dist_per_rho_sim,
                    da_trip_oh_dist_per_rho_sim=da_trip_oh_dist_per_rho_sim,
                    da_trip_oo_dist_per_rho_sim=da_trip_oo_dist_per_rho_sim,
                    a_exp_dist_per_rho_sim=a_exp_dist_per_rho_sim,
                    sum_p_exp_per_rho_sim=sum_p_exp_per_rho_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel6extra_stats(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim,
                events_hh_per_rho_sim=events_hh_per_rho_sim,
                events_ho_per_rho_sim=events_ho_per_rho_sim,
                events_oh_per_rho_sim=events_oh_per_rho_sim,
                events_oo_per_rho_sim=events_oo_per_rho_sim,
                f_trip_hh_dist_per_rho_sim=f_trip_hh_dist_per_rho_sim,
                f_trip_ho_dist_per_rho_sim=f_trip_ho_dist_per_rho_sim,
                f_trip_oh_dist_per_rho_sim=f_trip_oh_dist_per_rho_sim,
                f_trip_oo_dist_per_rho_sim=f_trip_oo_dist_per_rho_sim,
                da_trip_hh_dist_per_rho_sim=da_trip_hh_dist_per_rho_sim,
                da_trip_ho_dist_per_rho_sim=da_trip_ho_dist_per_rho_sim,
                da_trip_oh_dist_per_rho_sim=da_trip_oh_dist_per_rho_sim,
                da_trip_oo_dist_per_rho_sim=da_trip_oo_dist_per_rho_sim,
                a_exp_dist_per_rho_sim=a_exp_dist_per_rho_sim,
                sum_p_exp_per_rho_sim=sum_p_exp_per_rho_sim,
                prevalence_cutoff=0.025, 
                )
            
            events_hh_avg_per_rho = processed_results['hh_avg_per_rho']
            events_hh_l95_per_rho = processed_results['hh_l95_per_rho']
            events_hh_u95_per_rho = processed_results['hh_u95_per_rho']
            events_ho_avg_per_rho = processed_results['ho_avg_per_rho']
            events_ho_l95_per_rho = processed_results['ho_l95_per_rho']
            events_ho_u95_per_rho = processed_results['ho_u95_per_rho']
            events_oh_avg_per_rho = processed_results['oh_avg_per_rho']
            events_oh_l95_per_rho = processed_results['oh_l95_per_rho']
            events_oh_u95_per_rho = processed_results['oh_u95_per_rho']
            events_oo_avg_per_rho = processed_results['oo_avg_per_rho']
            events_oo_l95_per_rho = processed_results['oo_l95_per_rho']
            events_oo_u95_per_rho = processed_results['oo_u95_per_rho']

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    marker_size = 120
                    ax.scatter(rho_bins, events_hh_avg_per_rho, marker='o', color=chh_dict[sce_key], s=marker_size, label=r'h-h {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_hh_l95_per_rho, events_hh_u95_per_rho, color=chh_dict[sce_key], alpha=0.2)
                    ax.scatter(rho_bins, events_ho_avg_per_rho, marker='s', color=cho_dict[sce_key], s=marker_size, label=r'h-o {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_ho_l95_per_rho, events_ho_u95_per_rho, color=cho_dict[sce_key], alpha=0.2)
                    ax.scatter(rho_bins, events_oh_avg_per_rho, marker='v', color=coh_dict[sce_key], s=marker_size, label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_oh_l95_per_rho, events_oh_u95_per_rho, color=coh_dict[sce_key], alpha=0.2)
                    ax.scatter(rho_bins, events_oo_avg_per_rho, marker='P', color=coo_dict[sce_key], s=marker_size, label=r'o-o {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_oo_l95_per_rho, events_oo_u95_per_rho, color=coo_dict[sce_key], alpha=0.2)

                    intersection_index = np.argmin(np.abs(events_hh_avg_per_rho + events_oh_avg_per_rho - events_ho_avg_per_rho - events_oo_avg_per_rho))
                    intersection_rho = rho_bins[intersection_index]
                    ax.axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                    ax.axhline(0.25, color='gray', linestyle='dotted', alpha=1.0)
                    print("Intersection rho={0}".format(intersection_rho))

    #ax.text(0.05, 0.95, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax.set_xlabel(r'$\rho$', fontsize=35)
    ax.set_ylabel(r'OD event fraction', fontsize=35)
    #a].set_xlim(0.0, 1.0)
    #a].set_ylim(0.0, 1.0)
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
    base_name = 'deprf8_slides'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel_event(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: od-flows rho profile for beta
    f1: od-flows rho profile for gaussian
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel_event_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []

            event_attractiveness_sim = []
            event_inf_pop_avg_rho_sim = []
            event_location_sim = []
            event_size_sim = []
            event_time_sim = []
            event_tot_pop_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel_events_results(
                    out_sim_data, 
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    event_attractiveness_sim=event_attractiveness_sim,
                    event_inf_pop_avg_rho_sim=event_inf_pop_avg_rho_sim,
                    event_location_sim=event_location_sim,
                    event_size_sim=event_size_sim,
                    event_time_sim=event_time_sim,
                    event_tot_pop_sim=event_tot_pop_sim,
                    )
                
            event_sl_dict = ut.classify_events_by_location(
                attractiveness_se=event_attractiveness_sim, 
                inf_pop_avg_rho_se=event_inf_pop_avg_rho_sim, 
                location_se=event_location_sim, 
                size_se=event_size_sim, 
                time_se=event_time_sim, 
                tot_pop_se=event_tot_pop_sim,
                )
            
            ut.compute_additional_statistics(event_sl_dict)

            # Get the number of simulations and locations
            nsims = len(event_sl_dict)
  
            #locations = sorted(list(locations))
            locations = range(2500)
            nlocs = len(locations)

            # Create empty NumPy arrays to store the data
            attractiveness_sl = np.empty((nsims, nlocs))
            avg_inf_rho_sl = np.empty((nsims, nlocs))
            avg_size_sl = np.empty((nsims, nlocs))
            inter_time_sl = np.empty((nsims, nlocs))
            number_of_events_sl = np.empty((nsims, nlocs))
            time_sl = np.empty((nsims, nlocs))
            time_inv_sl = np.empty((nsims, nlocs))
            event_rate_sl = np.empty((nsims, nlocs))
            time_window_sl = np.empty((nsims, nlocs))
            rho_inv_sl = np.empty((nsims, nlocs))

            # Iterate through simulations and locations to populate the arrays
            for sim_idx in range(nsims):
                for loc_idx, location in enumerate(locations):
                    data = event_sl_dict[sim_idx].get(location, None)
                    if data:
                        attractiveness_sl[sim_idx, loc_idx] = data['attractiveness']
                        avg_inf_rho_sl[sim_idx, loc_idx] = data['inf_rho_avg']
                        avg_size_sl[sim_idx, loc_idx] = data['size_avg']
                        inter_time_sl[sim_idx, loc_idx] = data['inter_time_avg']
                        number_of_events_sl[sim_idx, loc_idx] = data['number_of_events']
                        time_sl[sim_idx, loc_idx] = data['time_avg']
                        time_inv_sl[sim_idx, loc_idx] = data['time_inv']
                        event_rate_sl[sim_idx, loc_idx] = data['event_rate']
                        time_window_sl[sim_idx, loc_idx] = data['time_window']
                        rho_inv_sl[sim_idx, loc_idx] = data['rho_inv']
                    else:
                        # Handle the case when data is missing for a location in a simulation
                        attractiveness_sl[sim_idx, loc_idx] = np.nan
                        avg_inf_rho_sl[sim_idx, loc_idx] = np.nan
                        avg_size_sl[sim_idx, loc_idx] = np.nan
                        inter_time_sl[sim_idx, loc_idx] = np.nan
                        number_of_events_sl[sim_idx, loc_idx] = np.nan
                        time_sl[sim_idx, loc_idx] = np.nan
                        time_inv_sl[sim_idx, loc_idx] = np.nan
                        event_rate_sl[sim_idx, loc_idx] = np.nan
                        time_window_sl[sim_idx, loc_idx] = np.nan
                        rho_inv_sl[sim_idx, loc_idx] = np.nan

            infected_fraction_sim = np.sum(infected_per_rho_sim, axis=1) / np.sum(agents_per_rho_sim, axis=1)
            failed_outbreaks = np.where(infected_fraction_sim < prevalence_cutoff)[0]

            attractiveness_sl = np.delete(attractiveness_sl, failed_outbreaks, axis=0)
            avg_inf_rho_sl = np.delete(avg_inf_rho_sl, failed_outbreaks, axis=0)
            avg_size_sl = np.delete(avg_size_sl, failed_outbreaks, axis=0)
            inter_time_sl = np.delete(inter_time_sl, failed_outbreaks, axis=0)
            number_of_events_sl = np.delete(number_of_events_sl, failed_outbreaks, axis=0)
            time_sl = np.delete(time_sl, failed_outbreaks, axis=0)
            time_inv_sl = np.delete(time_inv_sl, failed_outbreaks, axis=0)
            event_rate_sl = np.delete(event_rate_sl, failed_outbreaks, axis=0)
            time_window_sl = np.delete(time_window_sl, failed_outbreaks, axis=0)
            rho_inv_sl = np.delete(rho_inv_sl, failed_outbreaks, axis=0)

            attractiveness_l = np.nanmean(attractiveness_sl, axis=0)
            avg_avg_inf_rho_l = np.nanmean(avg_inf_rho_sl, axis=0)
            avg_avg_size_l = np.nanmean(avg_size_sl, axis=0)
            avg_inter_time_l = np.nanmean(inter_time_sl, axis=0)
            avg_number_of_events_l = np.nanmean(number_of_events_sl, axis=0)
            avg_time_l = np.nanmean(time_sl, axis=0)
            avg_time_inv_l = np.nanmean(time_inv_sl, axis=0)
            avg_event_rate_l = np.nanmean(event_rate_sl, axis=0)
            avg_time_window_l = np.nanmean(time_window_sl, axis=0)
            avg_rho_inv_l = np.nanmean(rho_inv_sl, axis=0)

            x_cells = int(np.sqrt(nlocs))
            y_cells = x_cells
            avg_inter_time_lattice = np.zeros((x_cells, y_cells))
            avg_time_lattice = np.zeros((x_cells, y_cells))
            avg_time_inv_lattice = np.zeros((x_cells, y_cells))
            avg_event_rate_lattice = np.zeros((x_cells, y_cells))
            avg_time_window_lattice = np.zeros((x_cells, y_cells))
            avg_rho_inv_lattice = np.zeros((x_cells, y_cells))
            l = 0
            for i in range(x_cells):
                for j in range(y_cells):
                    avg_inter_time_lattice[y_cells - 1 - j, i] = avg_inter_time_l[l]
                    avg_time_lattice[y_cells - 1 - j, i] = avg_time_l[l]
                    avg_time_inv_lattice[y_cells - 1 - j, i] = avg_time_inv_l[l]
                    avg_event_rate_lattice[y_cells - 1 - j, i] = avg_event_rate_l[l]
                    avg_time_window_lattice[y_cells - 1 - j, i] = avg_time_window_l[l]
                    avg_rho_inv_lattice[y_cells - 1 - j, i] = avg_rho_inv_l[l]
                    l += 1

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    
                    cmap = 'coolwarm'
                    vmin = 0.0
                    vmax = 1.0
                    
                    avg_avg_inf_rho_l = avg_avg_inf_rho_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_number_of_events_l = avg_number_of_events_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_avg_size_l = avg_avg_size_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_time_inv_l = avg_time_inv_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_rho_inv_l = avg_rho_inv_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_time_l = avg_time_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_time_window_l = avg_time_window_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    avg_inter_time_l = avg_inter_time_l[attractiveness_l != np.nanmax(attractiveness_l)]
                    attractiveness_l = attractiveness_l[attractiveness_l != np.nanmax(attractiveness_l)]

                    # Plot 00
                    sc00 = ax[0, 0].scatter(attractiveness_l, avg_number_of_events_l, c=avg_avg_inf_rho_l, cmap=cmap) #, vmin=vmin, vmax=vmax)

                    cbar00 = fig.colorbar(sc00, ax=ax[0, 0], shrink=1.0)
                    cbar00.ax.tick_params(labelsize=18)
                    #cbar0.set_label(r'$\langle \rho \rangle(I_{\ell})$', fontsize=35)

                    mask = ~np.isnan(attractiveness_l) & ~np.isnan(avg_number_of_events_l)
                    attractiveness_cleaned = attractiveness_l[mask]
                    avg_number_of_events_cleaned = avg_number_of_events_l[mask]

                    model_0 = LinearRegression()
                    model_0.fit(attractiveness_cleaned.reshape(-1, 1), avg_number_of_events_cleaned)
                    y_pred_0 = model_0.predict(attractiveness_cleaned.reshape(-1, 1))
                    r2_0 = model_0.score(attractiveness_cleaned.reshape(-1, 1), avg_number_of_events_cleaned)
                    ax[0, 0].plot(attractiveness_cleaned, y_pred_0, color='indigo', linestyle='--', linewidth=2)
                    ax[0, 0].text(0.55, 0.05, r'$R^2$={0}'.format(np.round(r2_0, 2)), transform=ax[0, 0].transAxes, fontsize=30, color='black')

                    # Plot 01
                    sc01 = ax[0, 1].scatter(attractiveness_l, avg_avg_size_l, c=avg_avg_inf_rho_l, cmap=cmap) #, vmin=vmin, vmax=vmax)

                    cbar01 = fig.colorbar(sc01, ax=ax[0, 1], shrink=1.0)
                    cbar01.ax.tick_params(labelsize=18)
                    #cbar1.set_label(r'$\langle \rho \rangle(I_{\ell})$', fontsize=35)

                    mask = ~np.isnan(attractiveness_l) & ~np.isnan(avg_avg_size_l)
                    attractiveness_cleaned = attractiveness_l[mask]
                    avg_avg_size_cleaned = avg_avg_size_l[mask]

                    model_1 = LinearRegression()
                    model_1.fit(attractiveness_cleaned.reshape(-1, 1), avg_avg_size_cleaned)
                    y_pred_1 = model_1.predict(attractiveness_cleaned.reshape(-1, 1))
                    r2_1 = model_1.score(attractiveness_cleaned.reshape(-1, 1), avg_avg_size_cleaned)    
                    ax[0, 1].plot(attractiveness_cleaned, y_pred_1, color='indigo', linestyle='--', linewidth=2)
                    ax[0, 1].text(0.55, 0.05, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[0, 1].transAxes, fontsize=30, color='black')

                    # Plot 02
                    sc02 = ax[0, 2].scatter(attractiveness_l, avg_time_inv_l, c=avg_rho_inv_l, cmap=cmap) #, vmin=vmin, vmax=vmax)

                    cbar02 = fig.colorbar(sc02, ax=ax[0, 2], shrink=1.0)
                    cbar02.ax.tick_params(labelsize=18)
                    cbar02.set_label(r'infected $\langle \rho \rangle$', fontsize=35)
                    #cbar2.set_label(r'$\langle \rho \rangle(I_{\ell})$', fontsize=35)

                    # Plot 10
                    sc10 = ax[1, 0].scatter(attractiveness_l, avg_time_l, c=avg_avg_inf_rho_l, cmap=cmap) #, vmin=vmin, vmax=vmax)
                    cbar10 = fig.colorbar(sc10, ax=ax[1, 0], shrink=1.0)
                    cbar10.ax.tick_params(labelsize=18)

                    mask = ~np.isnan(attractiveness_l) & ~np.isnan(avg_time_l)
                    attractiveness_cleaned = attractiveness_l[mask]
                    avg_time_cleaned = avg_time_l[mask]

                    model_1 = LinearRegression()
                    model_1.fit(attractiveness_cleaned.reshape(-1, 1), avg_time_cleaned)
                    y_pred_1 = model_1.predict(attractiveness_cleaned.reshape(-1, 1))
                    r2_1 = model_1.score(attractiveness_cleaned.reshape(-1, 1), avg_time_cleaned)    
                    ax[1, 0].plot(attractiveness_cleaned, y_pred_1, color='indigo', linestyle='--', linewidth=2)
                    ax[1, 0].text(0.15, 0.05, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 0].transAxes, fontsize=30, color='black')

                    # Plot 11
                    sc11 = ax[1, 1].scatter(attractiveness_l, avg_time_window_l, c=avg_avg_inf_rho_l, cmap=cmap) #, vmin=vmin, vmax=vmax)
                    cbar11 = fig.colorbar(sc11, ax=ax[1, 1], shrink=1.0)
                    cbar11.ax.tick_params(labelsize=18)

                    # Plot 12
                    sc12 = ax[1, 2].scatter(attractiveness_l, avg_inter_time_l, c=avg_avg_inf_rho_l, cmap=cmap) #, vmin=vmin, vmax=vmax)
                    cbar12 = fig.colorbar(sc12, ax=ax[1, 2], shrink=1.0)
                    cbar12.ax.tick_params(labelsize=18)
                    cbar12.set_label(r'infected $\langle \rho \rangle$', fontsize=35)

                    #im = ax[3].imshow(avg_inter_time_lattice.T, cmap='viridis', aspect='auto')
                    #cbar = fig.colorbar(im, ax=ax[3], shrink=1.0)
                    #cbar.ax.tick_params(labelsize=18)
                    #cbar.set_label(r'$\langle \Delta t_{{e,\ell}}\rangle$', fontsize=35)

                    #masked_data = np.ma.masked_where(np.ones_like(avg_inter_time_lattice.T), avg_inter_time_lattice)
                    #masked_data[28, 28] = 1.0
                    #im = ax[3].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0, aspect='auto') # extent=extent)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    pass

            elif dist_key == distribution_flags[1]:
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    pass

                elif sce_key == 'b1hom' or sce_key == 'plain':
                   pass
    
    ax[0, 0].text(0.05, 0.85, r"A", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 0].set_xlabel(r'$A_{\ell}$', fontsize=35)
    ax[0, 0].set_ylabel(r'$\langle E_{\ell}\rangle$', fontsize=35)
    #ax[0, 0].set_xlim(0.0, 1.0)
    #ax[0, 0].set_ylim(0.0, 1.0)
    ax[0, 0].tick_params(axis='both', labelsize=20)
    #ax[0, 0].legend(fontsize=20, loc='upper center')

    ax[0, 1].text(0.05, 0.85, r"B", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 1].set_xlabel(r'$A_{\ell}$', fontsize=35)
    ax[0, 1].set_ylabel(r'$\langle \Sigma_{e,\ell}\rangle$', fontsize=35)
    #ax[1].set_xlim(0.0, 1.0)
    #ax[1].set_ylim(0.0, 1.0)
    ax[0, 1].tick_params(axis='both', labelsize=20)
    #ax[1].legend(fontsize=20, loc='upper center')

    ax[0, 2].text(0.85, 0.85, r"C", transform=ax[0, 2].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 2].set_xlabel(r'$A_{\ell}$', fontsize=35)
    ax[0, 2].set_ylabel(r'$\langle t_{{inv},\ell}\rangle$', fontsize=35)
    #ax[2].set_xlim(0.0, 1.0)
    #ax[2].set_ylim(0.0, 1.0)
    ax[0, 2].tick_params(axis='both', labelsize=20)
    #ax[2].legend(fontsize=20, loc='upper center')

    ax[1, 0].text(0.85, 0.85, r"D", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 0].set_xlabel(r'$A_{\ell}$', fontsize=35)
    ax[1, 0].set_ylabel(r'$\langle t_{e,\ell}\rangle$', fontsize=35)
    #ax[2].set_xlim(0.0, 1.0)
    #ax[2].set_ylim(0.0, 1.0)
    ax[1, 0].tick_params(axis='both', labelsize=20)
    #ax[2].legend(fontsize=20, loc='upper center')

    ax[1, 1].text(0.05, 0.85, r"E", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 1].set_xlabel(r'$A_{\ell}$', fontsize=35)
    ax[1, 1].set_ylabel(r'$\langle t_{{end},\ell}-t_{{inv},\ell}\rangle$', fontsize=35)
    #ax[2].set_xlim(0.0, 1.0)
    #ax[2].set_ylim(0.0, 1.0)
    ax[1, 1].tick_params(axis='both', labelsize=20)
    #ax[2].legend(fontsize=20, loc='upper center')

    ax[1, 2].text(0.85, 0.85, r"F", transform=ax[1, 2].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 2].set_xlabel(r'$A_{\ell}$', fontsize=35)
    ax[1, 2].set_ylabel(r'$\langle \Delta t_{e,\ell}\rangle$', fontsize=35)
    #ax[2].set_xlim(0.0, 1.0)
    #ax[2].set_ylim(0.0, 1.0)
    ax[1, 2].tick_params(axis='both', labelsize=20)
    #ax[2].legend(fontsize=20, loc='upper center')

    #ax[3].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    #ax[3].set_ylabel("latitude (\u00b0 N)", fontsize=25)
    #ax[3].invert_yaxis()
    #ax[3].tick_params(axis='both', labelsize=18)

    #new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
    #new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
    #x_ticks_pos = range(0, 51, 10)
    #y_ticks_pos = range(0, 51, 10)
    #ax[3].set_xticks(x_ticks_pos)
    #ax[3].set_yticks(y_ticks_pos)
    #ax[3].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    #ax[3].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

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
    base_name = 'depr_events'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel_locs(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: od-flows rho profile for beta
    f1: od-flows rho profile for gaussian
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel_locs_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []

            inf_rho_dist_per_loc_sim = []
            inf_rho_h_dist_per_loc_sim = []
            inf_rho_o_dist_per_loc_sim = []
        
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel_locs_results(
                    out_sim_data, 
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    inf_rho_dist_per_loc_sim=inf_rho_dist_per_loc_sim,
                    inf_rho_h_dist_per_loc_sim=inf_rho_h_dist_per_loc_sim,
                    inf_rho_o_dist_per_loc_sim=inf_rho_o_dist_per_loc_sim,
                    )
                
            processed_results = ut.compute_depr_chapter_panel_locs(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim,
                inf_rho_dist_per_loc_sim=inf_rho_dist_per_loc_sim,
                inf_rho_h_dist_per_loc_sim=inf_rho_h_dist_per_loc_sim,
                inf_rho_o_dist_per_loc_sim=inf_rho_o_dist_per_loc_sim,
                prevalence_cutoff=0.025,
            )
            
            avg_inf_rho_lattice = processed_results['avg_inf_rho_lattice']
            avg_inf_rho_h_lattice = processed_results['avg_inf_rho_h_lattice']
            avg_inf_rho_o_lattice = processed_results['avg_inf_rho_o_lattice']
            avg_fra_inf_h_lattice = processed_results['avg_fra_inf_h_lattice']
            avg_fra_inf_o_lattice = processed_results['avg_fra_inf_o_lattice']

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    
                    im0 = ax[0].imshow(avg_fra_inf_h_lattice.T, cmap='viridis', aspect='auto')
                    #im0 = ax[0].imshow(avg_inf_rho_h_lattice.T, cmap='coolwarm', aspect='auto')
                    im0.set_clim(vmin=0.0, vmax=1.0)
                    cbar0 = fig.colorbar(im0, ax=ax[0], shrink=1.0)
                    cbar0.ax.tick_params(labelsize=18)

                    im1 = ax[1].imshow(avg_fra_inf_o_lattice.T, cmap='viridis', aspect='auto')
                    #im1 = ax[1].imshow(avg_inf_rho_o_lattice.T, cmap='coolwarm', aspect='auto')
                    im1.set_clim(vmin=0.0, vmax=1.0)
                    cbar1 = fig.colorbar(im1, ax=ax[1], shrink=1.0)
                    cbar1.ax.tick_params(labelsize=18)
                    #cbar1.set_label(r'infected $\langle\rho\rangle$', fontsize=35)
                    cbar1.set_label(r'population fraction', fontsize=35)
        
                    #im2 = ax[2].imshow(avg_inf_rho_o_lattice.T, cmap='coolwarm', aspect='auto')
                    #cbar2 = fig.colorbar(im2, ax=ax[2], shrink=1.0)
                    #cbar2.ax.tick_params(labelsize=18)
                    #cbar2.set_label(r'$\langle \rho\rangle$', fontsize=35)

                    #masked_data = np.ma.masked_where(np.ones_like(avg_inter_time_lattice.T), avg_inter_time_lattice)
                    #masked_data[28, 28] = 1.0
                    #im = ax[3].imshow(masked_data, cmap=custom_cmap, vmin=0.0, vmax=1.0, aspect='auto') # extent=extent)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    pass

            elif dist_key == distribution_flags[1]:
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    pass

                elif sce_key == 'b1hom' or sce_key == 'plain':
                   pass
    
    ax[0].set_title("infected at home", fontsize=35)
    ax[0].set_xlabel("longitude (\u00b0 W)", fontsize=30)
    ax[0].set_ylabel("latitude (\u00b0 N)", fontsize=30)
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

    ax[1].set_title("infected outside", fontsize=35)
    ax[1].set_xlabel("longitude (\u00b0 W)", fontsize=30)
    #ax[1].set_ylabel("latitude (\u00b0 N)", fontsize=35)
    ax[1].invert_yaxis()
    ax[1].tick_params(axis='both', labelsize=18)

    new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
    new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
    x_ticks_pos = range(0, 51, 10)
    y_ticks_pos = range(0, 51, 10)
    ax[1].set_xticks(x_ticks_pos)
    ax[1].set_yticks(y_ticks_pos)
    ax[1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    ax[1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

    #ax[2].set_xlabel("longitude (\u00b0 W)", fontsize=35)
    #ax[2].set_ylabel("latitude (\u00b0 N)", fontsize=35)
    #ax[2].invert_yaxis()
    #ax[2].tick_params(axis='both', labelsize=18)
#
    #new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
    #new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
    #x_ticks_pos = range(0, 51, 10)
    #y_ticks_pos = range(0, 51, 10)
    #ax[2].set_xticks(x_ticks_pos)
    #ax[2].set_yticks(y_ticks_pos)
    #ax[2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    #ax[2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

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
    base_name = 'depr_locs'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel_netmob(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: od-flows rho profile for beta
    f1: od-flows rho profile for gaussian
    """

    lower_path = 'config/'
    filename = 'config_space_bl_retriever'
    fullname = os.path.join(cwd_path, lower_path, filename)
    space_pars = ut.read_json_file(fullname)
    lower_path = 'data/'
    space_filename = 'space_' + ut.dict_to_string(space_pars) + '.pickle'
    space_fullname = os.path.join(cwd_path, lower_path, space_filename)
    space_df = an.build_spatial_data_frame(space_fullname)

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    
    num_bins = 30
    rho_bins = np.linspace(0.0, 1.0, num_bins + 1)

    collected_output = {}
  
    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'data/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        if os.path.exists(epi_fullname):

            distribution_flag = epi_filename.split('_rm')[1].split('_')[0]
            scenario_flag = epi_filename.split('_ms')[1].split('_')[0]

            if (distribution_flag in distribution_flags) and (scenario_flag in scenario_flags):

                out_sim_data = ut.load_depr_chapter_panel_netmob_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            returner_netmob_sim = []
            commuter_netmob_sim = []
            explorer_netmob_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel_netmob_results(
                    out_sim_data=out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    returner_netmob_sim=returner_netmob_sim,
                    commuter_netmob_sim=commuter_netmob_sim,
                    explorer_netmob_sim=explorer_netmob_sim,
                    )
                
            processed_results = ut.compute_depr_chapter_panel_netmob(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim,
                returner_netmob_sim=returner_netmob_sim,
                commuter_netmob_sim=commuter_netmob_sim,
                explorer_netmob_sim=explorer_netmob_sim,
                space_df=space_df,
                prevalence_cutoff=0.025,
                attr_cutoff=0.000000001,
            )
            
            avg_returner_netmob = processed_results['avg_returner_netmob']
            avg_commuter_netmob = processed_results['avg_commuter_netmob']
            avg_explorer_netmob = processed_results['avg_explorer_netmob']

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    
                    colors = ['lavender', 'slateblue']
                    cmap = ListedColormap(colors)

                    im0 = ax[0].imshow(avg_returner_netmob, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                    im1 = ax[1].imshow(avg_commuter_netmob, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                    im2 = ax[2].imshow(avg_explorer_netmob, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                    cbar2 = fig.colorbar(im2, ax=ax[2], shrink=1.0)
                    cbar2.ax.tick_params(labelsize=18)
                    cbar2.set_ticks([0, 1])

                    # Set the tick labels to '0' and '1'
                    #cbar2.set_ticklabels(['0', '1'], fontsize=18)
                    #cbar2.ax.set_yticklabels(['0', '1'], fontsize=18)
    
    #ax[0].invert_yaxis()
    ax[0].set_title(r'top returners ($\rho<0.05$)', fontsize=30)
    #ax[0].tick_params(axis='both', labelsize=18)
    #ax[1].invert_yaxis()
    ax[1].set_title(r'top commuters ($\rho\approx 0.33$)', fontsize=30)
    #ax[1].tick_params(axis='both', labelsize=18)
    #ax[2].invert_yaxis()
    ax[2].set_title(r'top explorers ($\rho\approx 0.95$)', fontsize=30)
    #ax[2].tick_params(axis='both', labelsize=18)

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
    base_name = 'depr_netmob'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def main():

    focal_dist = 'Beta'
    comp_dist = 'Gaussian'
    distribution_flags = [focal_dist] #, comp_dist] 
    scenario_flags = ['depr'] #, 'b1het']

    #plot_depr_panel3A(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panel3B(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panel6(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panelfinal(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panel8_slides(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    plot_depr_panel_event(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panel_locs(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panel_netmob(distribution_flags=distribution_flags, scenario_flags=scenario_flags)

if __name__ == '__main__':
    main()