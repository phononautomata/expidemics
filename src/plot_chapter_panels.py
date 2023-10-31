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

def plot_depr_panel2(
        distribution_flags=None,
        scenario_flags=None,
        stats_flag=False,
        t_inv_flag=False,
        ):
    """ 2x2 panel
    f00: invasion rho profile for beta
    f01: invasion time rho profile for beta
    f10: invasion rho profile for Gaussian
    f11: invasion time rho profile for Gaussian
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

            out_sim_data = ut.load_depr_chapter_panel2_data(
                epi_fullname, 
                t_inv_flag=t_inv_flag
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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 14))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            invaders_per_rho_sim = []
            nlocs_invaded_sim = []
            
            if stats_flag:
                t_inv_stats_per_rho_sim = []
            if t_inv_flag:
                t_inv_dist_per_rho_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel2_results(
                    out_sim_data=out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    invaders_per_rho_sim=invaders_per_rho_sim,
                    nlocs_invaded_sim=nlocs_invaded_sim,
                    t_inv_flag=t_inv_flag,
                    t_inv_dist_per_rho_sim=t_inv_dist_per_rho_sim,
                )

            processed_results = ut.compute_depr_chapter_panel2_stats(
                agents_per_rho_sim=agents_per_rho_sim, 
                infected_per_rho_sim=infected_per_rho_sim, 
                invaders_per_rho_sim=invaders_per_rho_sim, 
                nlocs_invaded_sim=nlocs_invaded_sim, 
                prevalence_cutoff=prevalence_cutoff, 
                t_inv_flag=t_inv_flag, 
                t_inv_dist_per_rho_sim=t_inv_dist_per_rho_sim,
                )
            
            invaded_fraction_avg_per_rho = processed_results['inv_avg_per_rho']
            invaded_fraction_l95_per_rho = processed_results['inv_l95_per_rho']
            invaded_fraction_u95_per_rho = processed_results['inv_u95_per_rho']
            invaded_fraction_avg = processed_results['inv_avg']
            
            if t_inv_flag:
                t_inv_avg_per_rho = processed_results['t_inv_avg_per_rho']
                t_inv_l95_per_rho = processed_results['t_inv_l95_per_rho']
                t_inv_u95_per_rho = processed_results['t_inv_u95_per_rho']
                t_inv_avg_global = processed_results['t_inv_avg_global']
                t_inv_l95_global = processed_results['t_inv_l95_global']
                t_inv_u95_global = processed_results['t_inv_u95_global']

            if dist_key == distribution_flags[0]:
                ax[0, 1].scatter(rho_bins, invaded_fraction_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                ax[0, 1].fill_between(rho_bins, invaded_fraction_l95_per_rho, invaded_fraction_u95_per_rho, color=color_dict[sce_key], alpha=0.2,)

                expected_share = np.sum(agents_per_rho_sim, axis=0) / np.sum(agents_per_rho_sim)

                if sce_key == 'depr':
                    ax[0, 1].plot(rho_bins, expected_share, linestyle='--', color='indigo', label=r'null: $N_{\rho}/N$',)
                else:
                    ax[0, 1].plot(rho_bins, expected_share, linestyle='--', color='indigo', )

                intersection_index = np.argmin(np.abs(invaded_fraction_avg_per_rho - expected_share))
                intersection_rho = rho_bins[intersection_index]
                print("Intersection rho={0}".format(intersection_rho))
                
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    ax[1, 1].plot(rho_bins, t_inv_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    ax[1, 1].fill_between(rho_bins, t_inv_l95_per_rho, t_inv_u95_per_rho, color=color_dict[sce_key], alpha=0.2,)
                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[1, 1].axhline(t_inv_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=label_dict[sce_key],)
                    ax[1, 1].fill_between(rho_bins, t_inv_l95_global, t_inv_u95_global, color=color_dict[sce_key], alpha=0.2,)

            else:
                ax[0, 0].scatter(rho_bins, invaded_fraction_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                ax[0, 0].fill_between(rho_bins, invaded_fraction_l95_per_rho, invaded_fraction_u95_per_rho, color=color_dict[sce_key], alpha=0.2,)
                
                expected_share = np.sum(agents_per_rho_sim, axis=0) / np.sum(agents_per_rho_sim)
                
                if sce_key == 'depr':
                    ax[0, 0].plot(rho_bins, expected_share, linestyle='--', color='indigo', label=r'null: $N_{\rho}/N$',)
                else:
                    ax[0, 0].plot(rho_bins, expected_share, linestyle='--', color='indigo', )

                intersection_index = np.argmin(np.abs(invaded_fraction_avg_per_rho - expected_share))
                intersection_rho = rho_bins[intersection_index]
                print("Intersection rho={0}".format(intersection_rho))
            
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    ax[1, 0].plot(rho_bins[5:-5], t_inv_avg_per_rho[5:-5], marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    ax[1, 0].fill_between(rho_bins[5:-5], t_inv_l95_per_rho[5:-5], t_inv_u95_per_rho[5:-5], color=color_dict[sce_key], alpha=0.2,)
                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[1, 0].axhline(t_inv_avg_global, color=color_dict[sce_key], linestyle='--', label=label_dict[sce_key],)
                    ax[1, 0].fill_between(rho_bins, t_inv_l95_global, t_inv_u95_global, color=color_dict[sce_key], alpha=0.2,)

    ax[0, 0].set_title("Gaussian setting", fontsize=40)
    ax[0, 0].text(0.05, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0, 0].set_ylabel(r"$N_{{inv,\rho}}/V_{inv}$", fontsize=35)
    ax[0, 0].tick_params(axis='both', labelsize=25)
    ax[0, 0].set_xlim(0.0, 1.0)
    ax[0, 0].legend(loc='upper right', fontsize=25, labelspacing=0.5, handletextpad=0.5, handlelength=3.0)

    ax[0, 1].set_title("Beta setting", fontsize=40)
    #ax[0, 1].set_xlabel(r"$\rho$", fontsize=30)
    ax[0, 1].text(0.05, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 1].set_ylabel(r"$N_{{inv,\rho}}/V_{inv}$", fontsize=30)
    ax[0, 1].tick_params(axis='both', labelsize=25)
    ax[0, 1].set_xlim(0.0, 1.0)
    ax[0, 1].legend(loc='lower center', fontsize=25, labelspacing=0.5, handletextpad=0.5, handlelength=3.0)

    ax[1, 0].text(0.05, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 0].set_xlabel(r"$\rho$", fontsize=35)
    ax[1, 0].set_ylabel(r"$t_{{inv}}$", fontsize=35)
    ax[1, 0].tick_params(axis='both', labelsize=25)
    ax[1, 0].set_xlim(0.0, 1.0)
    ax[1, 0].legend(loc='upper center', fontsize=25, labelspacing=0.5, handletextpad=0.5, handlelength=3.0)

    ax[1, 1].set_xlabel(r"$\rho$", fontsize=35)
    ax[1, 1].text(0.05, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[1, 1].set_ylabel(r"$t_{{inv}}$", fontsize=30)
    ax[1, 1].tick_params(axis='both', labelsize=25)
    ax[1, 1].set_xlim(0.0, 1.0)
    ax[1, 1].legend(loc='upper right', fontsize=25, labelspacing=0.5, handletextpad=0.5, handlelength=3.0)

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
    base_name = 'deprf2'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel3(
        distribution_flags=None,
        scenario_flags=None,
        stats_flag=False,
        r_inv_flag=False,
        r_inf_flag=False,
        ):
    """ 2x3 panel
    f00: invasion spatial map for beta depr
    f01: invasion spatial map for beta uniform
    f02: invasion spatial map for gaussian depr
    f10: local new cases vs. attractiveness scatter for beta depr
    f11: local new cases vs. attractiveness scatter for beta uniform
    f12: local new cases vs. attractiveness scatter for gaussian depr
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

            out_sim_data = ut.load_depr_chapter_panel3_data(
                epi_fullname, 
                r_inv_flag=r_inv_flag, 
                r_inf_flag=r_inf_flag,
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
    label_dict = ut.build_label_dictionary()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25, 14))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            total_cases_loc_sim = []

            if r_inv_flag:
                r_inv_dist_per_loc_sim = []
                r_inv_dist_per_loc_sim = []
            if r_inf_flag:
                r_inf_dist_per_loc_sim = []
                r_inf_dist_per_loc_sim = []
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel3_results(
                    out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    total_cases_loc_sim=total_cases_loc_sim,
                    r_inv_flag=r_inv_flag,
                    r_inv_dist_per_loc_sim=r_inv_dist_per_loc_sim,
                    r_inf_flag=r_inf_flag,
                    r_inf_dist_per_loc_sim=r_inf_dist_per_loc_sim,
                )

            processed_results = ut.compute_depr_chapter_panel3_stats(
                agents_per_rho_sim=agents_per_rho_sim, 
                infected_per_rho_sim=infected_per_rho_sim,
                total_cases_loc_sim=total_cases_loc_sim,
                space_df=space_df,
                prevalence_cutoff=0.025,
                r_inv_flag=r_inv_flag,
                r_inv_dist_per_loc_sim=r_inv_dist_per_loc_sim,
                r_inf_flag=r_inf_flag,
                r_inf_dist_per_loc_sim=r_inf_dist_per_loc_sim,
                )
            
            total_cases_avg_loc = processed_results['total_cases_avg_loc']
            attr_l = processed_results['attractiveness_l']
            if r_inv_flag:
                inv_rho_avg_lattice = processed_results['inv_rho_avg_lattice']
                r_inv_avg_per_loc = processed_results['inv_rho_avg_loc']            
                invasion_fraction_avg = processed_results['invasion_fraction_avg']
                inv_rate_avg_lattice = processed_results['inv_rate_avg_lattice']
                invasion_fraction_avg_loc = processed_results['invasions_fraction_avg_loc']
            if r_inf_flag:
                inf_rho_avg_lattice = processed_results['inf_rho_avg_lattice']
                r_inf_avg_per_loc = processed_results['inv_rho_avg_loc']

            if dist_key == distribution_flags[0] and sce_key == 'depr':

                im0 = ax[0, 0].imshow(inv_rho_avg_lattice.T, cmap='coolwarm')
                im0.set_clim(vmin=0.0, vmax=1.0)
                cbar0 = fig.colorbar(im0, ax=ax[0, 0], shrink=1.0)
                #cbar0.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

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

                """
                im1 = ax[1, 0].imshow(inv_rate_avg_lattice.T, cmap='viridis')
                im1.set_clim(vmin=0.0, vmax=1.0)
                cbar1 = fig.colorbar(im1, ax=ax[1, 0], shrink=1.0)
                #cbar1.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                ax[1, 0].set_xlabel('longitude (\u00b0 W)', fontsize=25)
                ax[1, 0].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[1, 0].invert_yaxis()
                ax[1, 0].tick_params(axis='both', labelsize=18)

                ax[1, 0].text(0.05, 0.05, r"{0:.2f}$\%$".format(np.round(invasion_fraction_avg * 100.0, 2)), transform=ax[1, 0].transAxes, fontsize=20, color='white')

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[1, 0].set_xticks(x_ticks_pos)
                ax[1, 0].set_yticks(y_ticks_pos)
                ax[1, 0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[1, 0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                """

                hb2 = ax[1, 0].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
                hb2.set_clim(vmin=0.0, vmax=1.0)
                cbar2 = fig.colorbar(hb2, ax=ax[1, 0])
                #cbar3.set_label(r'infected $\langle\rho\rangle$', fontsize=25)

                # Compute the mean value for each hexbin
                xbins = hb2.get_offsets()[:, 0]
                ybins = hb2.get_offsets()[:, 1]
                mean_values = hb2.get_array()
                mean_rho_for_hexbins = []

                for i in range(len(mean_values)):
                    if i == len(mean_values) - 1:  # Handle the last bin separately
                        condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
                    else:
                        condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

                    indices = np.where(condition)
                    if len(indices[0]) > 0:
                        if r_inf_flag:
                            mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
                    else:
                        mean_rho_for_hexbins.append(0.0)

                model_1 = LinearRegression()
                model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
                y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
                ax[1, 0].plot(attr_l, y_pred_11, color='indigo', linestyle='--', linewidth=2)
    
                r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
                ax[1, 0].text(0.5, 0.85, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 0].transAxes, fontsize=30, color='black')

                ax[1, 0].set_xlabel(r'$A$', fontsize=25)
                ax[1, 0].set_ylabel('mean total cases', fontsize=25)
                ax[1, 0].tick_params(axis='both', labelsize=15)

                ax[0, 0].text(0.05, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 0].text(0.05, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
                #ax[2, 0].text(0.05, 0.9, r"A2", transform=ax[2, 0].transAxes, fontsize=40, color='black', weight="bold")
            
            elif dist_key == distribution_flags[0] and sce_key == 'b1het':

                im0 = ax[0, 1].imshow(inv_rho_avg_lattice.T, cmap='coolwarm')
                im0.set_clim(vmin=0.0, vmax=1.0)
                cbar0 = fig.colorbar(im0, ax=ax[0, 1], shrink=1.0)
                #cbar0.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                ax[0, 1].set_xlabel("longitude (\u00b0 W)", fontsize=25)
                ax[0, 1].set_ylabel("latitude (\u00b0 N)", fontsize=25)
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

                """
                im1 = ax[1, 1].imshow(inv_rate_avg_lattice.T, cmap='viridis')
                im1.set_clim(vmin=0.0, vmax=1.0)
                cbar1 = fig.colorbar(im1, ax=ax[1, 1], shrink=1.0)
                #cbar1.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                ax[1, 1].set_xlabel('longitude (\u00b0 W)', fontsize=25)
                ax[1, 1].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[1, 1].invert_yaxis()
                ax[1, 1].tick_params(axis='both', labelsize=18)

                ax[1, 1].text(0.05, 0.05, r"{0:.2f}$\%$".format(np.round(invasion_fraction_avg * 100.0, 2)), transform=ax[1, 1].transAxes, fontsize=20, color='white')

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[1, 1].set_xticks(x_ticks_pos)
                ax[1, 1].set_yticks(y_ticks_pos)
                ax[1, 1].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[1, 1].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                """

                hb2 = ax[1, 1].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
                hb2.set_clim(vmin=0.0, vmax=1.0)
                cbar2 = fig.colorbar(hb2, ax=ax[1, 1])
                #cbar2.set_label(r'infected $\langle\rho\rangle$', fontsize=25)

                # Compute the mean value for each hexbin
                xbins = hb2.get_offsets()[:, 0]
                ybins = hb2.get_offsets()[:, 1]
                mean_values = hb2.get_array()
                mean_rho_for_hexbins = []

                for i in range(len(mean_values)):
                    if i == len(mean_values) - 1:  # Handle the last bin separately
                        condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
                    else:
                        condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

                    indices = np.where(condition)
                    if len(indices[0]) > 0:
                        if r_inf_flag:
                            mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
                    else:
                        mean_rho_for_hexbins.append(0.0)

                model_1 = LinearRegression()
                model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
                y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
                ax[1, 1].plot(attr_l, y_pred_11, color='indigo', linestyle='--', linewidth=2)
    
                r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
                ax[1, 1].text(0.5, 0.85, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 1].transAxes, fontsize=30, color='black')

                ax[1, 1].set_xlabel(r'$A$', fontsize=25)
                ax[1, 1].set_ylabel('mean total cases', fontsize=25)
                ax[1, 1].tick_params(axis='both', labelsize=15)

                ax[0, 1].text(0.05, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 1].text(0.05, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
                #ax[2, 1].text(0.05, 0.9, r"B2", transform=ax[2, 1].transAxes, fontsize=40, color='black', weight="bold")

            elif dist_key == distribution_flags[1] and sce_key == 'depr':

                im0 = ax[0, 2].imshow(inv_rho_avg_lattice.T, cmap='coolwarm')
                im0.set_clim(vmin=0.0, vmax=1.0)
                cbar0 = fig.colorbar(im0, ax=ax[0, 2], shrink=1.0)
                cbar0.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                ax[0, 2].set_xlabel("longitude (\u00b0 W)", fontsize=25)
                ax[0, 2].set_ylabel("latitude (\u00b0 N)", fontsize=25)
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

                """
                im1 = ax[1, 2].imshow(inv_rate_avg_lattice.T, cmap='viridis')
                im1.set_clim(vmin=0.0, vmax=1.0)
                cbar1 = fig.colorbar(im1, ax=ax[1, 2], shrink=1.0)
                #cbar1.set_label(r'invader $\langle\rho\rangle$', fontsize=25)

                ax[1, 2].set_xlabel('longitude (\u00b0 W)', fontsize=25)
                ax[1, 2].set_ylabel('latitude (\u00b0 N)', fontsize=25)
                ax[1, 2].invert_yaxis()
                ax[1, 2].tick_params(axis='both', labelsize=18)

                ax[1, 2].text(0.05, 0.05, r"{0:.2f}$\%$".format(np.round(invasion_fraction_avg * 100.0, 2)), transform=ax[1, 2].transAxes, fontsize=20, color='white')

                new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
                new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
                x_ticks_pos = range(0, 51, 10)
                y_ticks_pos = range(0, 51, 10)
                ax[1, 2].set_xticks(x_ticks_pos)
                ax[1, 2].set_yticks(y_ticks_pos)
                ax[1, 2].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
                ax[1, 2].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])
                """

                hb2 = ax[1, 2].hexbin(attr_l, total_cases_avg_loc, C=r_inf_avg_per_loc, cmap='coolwarm', gridsize=30, mincnt=1)
                hb2.set_clim(vmin=0.0, vmax=1.0)
                cbar2 = fig.colorbar(hb2, ax=ax[1, 2])
                cbar2.set_label(r'infected $\langle\rho\rangle$', fontsize=25)

                # Compute the mean value for each hexbin
                xbins = hb2.get_offsets()[:, 0]
                ybins = hb2.get_offsets()[:, 1]
                mean_values = hb2.get_array()
                mean_rho_for_hexbins = []

                for i in range(len(mean_values)):
                    if i == len(mean_values) - 1:  # Handle the last bin separately
                        condition = np.logical_and(attr_l >= xbins[i], total_cases_avg_loc >= ybins[i])
                    else:
                        condition = np.logical_and.reduce((attr_l >= xbins[i], attr_l < xbins[i + 1], total_cases_avg_loc >= ybins[i], total_cases_avg_loc < ybins[i + 1]))

                    indices = np.where(condition)
                    if len(indices[0]) > 0:
                        if r_inf_flag:
                            mean_rho_for_hexbins.append(np.nanmean(np.array(r_inf_avg_per_loc)[indices]))
                    else:
                        mean_rho_for_hexbins.append(0.0)

                model_1 = LinearRegression()
                model_1.fit(attr_l.reshape(-1, 1), total_cases_avg_loc)
                y_pred_11 = model_1.predict(attr_l.reshape(-1, 1))
                ax[1, 2].plot(attr_l, y_pred_11, color='indigo', linestyle='--', linewidth=2)
    
                r2_1 = model_1.score(attr_l.reshape(-1, 1), total_cases_avg_loc)
                ax[1, 2].text(0.5, 0.85, r'$R^2$={0}'.format(np.round(r2_1, 2)), transform=ax[1, 2].transAxes, fontsize=30, color='black')

                ax[1, 2].set_xlabel(r'$A$', fontsize=25)
                ax[1, 2].set_ylabel('mean total cases', fontsize=25)
                ax[1, 2].tick_params(axis='both', labelsize=15)

                ax[0, 2].text(0.05, 0.9, r"C1", transform=ax[0, 2].transAxes, fontsize=40, color='black', weight="bold")
                ax[1, 2].text(0.05, 0.9, r"C2", transform=ax[1, 2].transAxes, fontsize=40, color='black', weight="bold")
                #ax[2, 2].text(0.05, 0.9, r"C2", transform=ax[2, 2].transAxes, fontsize=40, color='black', weight="bold")

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
    base_name = 'deprf3'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel4(
        distribution_flags=None,
        scenario_flags=None,
        stats_flag=False,
        t_inf_flag=False,
        ):
    """ 2x2 panel
    f00: infection rho profile for beta
    f01: infection time rho profile for beta
    f10: infection rho profile for Gaussian
    f11: infection time rho profile for Gaussian
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

            out_sim_data = ut.load_depr_chapter_panel4_data(
                epi_fullname, 
                t_inf_flag=t_inf_flag
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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 14))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []

            if stats_flag:
                t_inf_stats_per_rho_sim = []
            if t_inf_flag:
                t_inf_dist_per_rho_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel4_results(
                    out_sim_data=out_sim_data,
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    t_inf_flag=t_inf_flag,
                    t_inf_dist_per_rho_sim=t_inf_dist_per_rho_sim,
                )

            processed_results = ut.compute_depr_chapter_panel4_stats(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim, 
                prevalence_cutoff=prevalence_cutoff, 
                t_inf_flag=t_inf_flag, 
                t_inf_dist_per_rho_sim=t_inf_dist_per_rho_sim,
                )
            
            infected_fraction_avg_per_rho = processed_results['inf_avg_per_rho']
            infected_fraction_l95_per_rho = processed_results['inf_l95_per_rho']
            infected_fraction_u95_per_rho = processed_results['inf_u95_per_rho']
            infected_fraction_avg = processed_results['inf_avg']
            infected_fraction_l95 = processed_results['inf_l95']
            infected_fraction_u95 = processed_results['inf_u95']
            
            if t_inf_flag:
                t_inf_avg_per_rho = processed_results['t_inf_avg_per_rho']
                t_inf_l95_per_rho = processed_results['t_inf_l95_per_rho']
                t_inf_u95_per_rho = processed_results['t_inf_u95_per_rho']
                t_inf_avg_global = processed_results['t_inf_avg_global']
                t_inf_l95_global = processed_results['t_inf_l95_global']
                t_inf_u95_global = processed_results['t_inf_u95_global']

            if dist_key == distribution_flags[0]:
    
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    ax[0, 1].scatter(rho_bins, infected_fraction_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    ax[0, 1].fill_between(rho_bins, infected_fraction_l95_per_rho, infected_fraction_u95_per_rho, color=color_dict[sce_key], alpha=0.2,)
                    
                    r_inf = ut.sir_prevalence(R0, r_0)
                    ax[0, 1].axhline(r_inf, color='steelblue', linestyle='--', )
                    
                    label_text = r'$r_{hom}(\infty)$'
                    x_coord = 0.8
                    y_coord = r_inf - 0.004
                    ax[0, 1].text(x_coord, y_coord, label_text, color='black', fontsize=25)
                    
                    ax[0, 1].axhline(infected_fraction_avg, color=color_dict[sce_key], linestyle='--',)

                    ax[1, 1].plot(rho_bins, t_inf_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    ax[1, 1].fill_between(rho_bins, t_inf_l95_per_rho, t_inf_u95_per_rho, color=color_dict[sce_key], alpha=0.2,)

                    intersection_index = np.argmin(np.abs(infected_fraction_avg_per_rho - infected_fraction_avg))
                    intersection_rho = rho_bins[intersection_index]
                    print("Intersection rho={0}".format(intersection_rho))

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[0, 1].axhline(infected_fraction_avg, color=color_dict[sce_key], linestyle='--', label=r'{0}'.format(label_dict[sce_key]),)
                    ax[0, 1].fill_between(rho_bins, infected_fraction_l95, infected_fraction_u95, color=color_dict[sce_key], alpha=0.2,)

                    ax[1, 1].axhline(t_inf_avg_global, color=color_dict[sce_key], linestyle='--', label=label_dict[sce_key],)
                    ax[1, 1].fill_between(rho_bins, t_inf_l95_global, t_inf_u95_global, color=color_dict[sce_key], alpha=0.2,)
    
            else:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':

                    ax[0, 0].scatter(rho_bins[5:-5], infected_fraction_avg_per_rho[5:-5], marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    ax[0, 0].fill_between(rho_bins[5:-5], infected_fraction_l95_per_rho[5:-5], infected_fraction_u95_per_rho[5:-5], color=color_dict[sce_key], alpha=0.2,)

                    r_inf = ut.sir_prevalence(R0, r_0)

                    ax[0, 0].axhline(r_inf, color='steelblue', linestyle='--', )
                    
                    label_text = r'$r_{hom}(\infty)$'
                    x_coord = 0.5
                    y_coord = r_inf - 0.006
                    ax[0, 0].text(x_coord, y_coord, label_text, color='black', fontsize=25)
                    
                    ax[0, 0].axhline(infected_fraction_avg, color=color_dict[sce_key], linestyle='--',)

                    ax[1, 0].plot(rho_bins[5:-5], t_inf_avg_per_rho[5:-5], marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    ax[1, 0].fill_between(rho_bins[5:-5], t_inf_l95_per_rho[5:-5], t_inf_u95_per_rho[5:-5], color=color_dict[sce_key], alpha=0.2,)

                    intersection_index = np.argmin(np.abs(infected_fraction_avg_per_rho - infected_fraction_avg))
                    intersection_rho = rho_bins[intersection_index]
                    print("Intersection rho={0}".format(intersection_rho))

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    ax[0, 0].axhline(infected_fraction_avg, color=color_dict[sce_key], linestyle='--', label=r'{0}'.format(label_dict[sce_key]),)
                    ax[0, 0].fill_between(rho_bins, infected_fraction_l95, infected_fraction_u95, color=color_dict[sce_key], alpha=0.2,)

                    ax[1, 0].axhline(t_inf_avg_global, color=color_dict[sce_key], linestyle='--', label=label_dict[sce_key],)
                    ax[1, 0].fill_between(rho_bins, t_inf_l95_global, t_inf_u95_global, color=color_dict[sce_key], alpha=0.2,)

    ax[0, 0].set_title("Gaussian setting", fontsize=40)
    ax[0, 0].text(0.05, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=40, color='black', weight="bold")
    #ax[0, 0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0, 0].set_ylim(0.255, 0.32)
    ax[0, 0].set_ylabel(r"$R_{\rho}(\infty)/N_{\rho}$", fontsize=35)
    ax[0, 0].tick_params(axis='both', labelsize=25)
    ax[0, 0].set_xlim(0.0, 1.0)
    ax[0, 0].legend(loc='center', fontsize=25)

    ax[0, 1].set_title("Beta setting", fontsize=40)
    #ax[0, 1].set_xlabel(r"$\rho$", fontsize=30)
    ax[0, 1].text(0.05, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=40, color='black', weight="bold")
    ax[0, 1].set_ylim(0.248, 0.32)
    #ax[0, 1].set_ylabel(r"$R_{\rho}(\infty)/N_{\rho}$", fontsize=30)
    ax[0, 1].tick_params(axis='both', labelsize=25)
    ax[0, 1].set_xlim(0.0, 1.0)
    ax[0, 1].legend(loc='upper center', fontsize=25, labelspacing=0.5, handletextpad=0.5)

    ax[1, 0].text(0.05, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=40, color='black', weight="bold")
    ax[1, 0].set_xlabel(r"$\rho$", fontsize=35)
    ax[1, 0].set_ylim(279, 310)
    ax[1, 0].set_ylabel(r"$t_{{inf}}$", fontsize=35)
    ax[1, 0].tick_params(axis='both', labelsize=25)
    ax[1, 0].set_xlim(0.0, 1.0)
    ax[1, 0].legend(loc='center', fontsize=25, bbox_to_anchor=(0.5, 0.35))

    ax[1, 1].set_xlabel(r"$\rho$", fontsize=35)
    ax[1, 1].text(0.05, 0.9, r"A3", transform=ax[1, 1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[1, 1].set_ylabel(r"$t_{{inf}}$", fontsize=30)
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
    base_name = 'deprf4'
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

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

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

                    ax[0].scatter(rho_bins, infected_h_fraction_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[0].fill_between(rho_bins, infected_h_fraction_l95_per_rho, infected_h_fraction_u95_per_rho, color=ch_dict[sce_key], alpha=0.2,)
                    ax[0].scatter(rho_bins, infected_o_fraction_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[0].fill_between(rho_bins, infected_o_fraction_l95_per_rho, infected_o_fraction_u95_per_rho, color=co_dict[sce_key], alpha=0.2,)

                    ax[0].axhline(home_global_fraction, color=ch_dict[sce_key], linestyle='--')
                    ax[0].axhline(out_global_fraction, color=co_dict[sce_key], linestyle='--')

                    ax[0].text(0.7, 0.6, r"$\langle R_{o}(\infty)/R(\infty)\rangle$", transform=ax[0].transAxes, fontsize=15, color='black')
                    ax[0].text(0.7, 0.42, r"$\langle R_{h}(\infty)/R(\infty)\rangle$", transform=ax[0].transAxes, fontsize=15, color='black')

                    intersection_index = np.argmin(np.abs(infected_h_fraction_avg_per_rho - infected_o_fraction_avg_per_rho))
                    intersection_rho = rho_bins[intersection_index]
                    ax[0].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                    ax[0].axhline(0.5, color='gray', linestyle='dotted', alpha=1.0)
                    print("Intersection rho={0}".format(intersection_rho))

                    ax[1].scatter(rho_bins, f_inf_h_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, f_inf_h_l95_per_rho, f_inf_h_u95_per_rho, color=ch_dict[sce_key], alpha=0.2)
                    ax[1].scatter(rho_bins, f_inf_o_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, f_inf_o_l95_per_rho, f_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)

                    intersection_index = np.argmin(np.abs(f_inf_h_avg_per_rho - 0.5))
                    intersection_rho = rho_bins[intersection_index]
                    ax[1].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                    ax[1].axhline(0.5, color='gray', linestyle='dotted', alpha=1.0)
                    print("Intersection rho={0}".format(intersection_rho))

                    ax[2].scatter(rho_bins, a_inf_h_avg_per_rho, marker=marker_dict[sce_key], color=ch_dict[sce_key], label=r'home {0}'.format(label_dict[sce_key]),)
                    ax[2].fill_between(rho_bins, a_inf_h_l95_per_rho, a_inf_h_u95_per_rho, color=ch_dict[sce_key], alpha=0.2)
                    ax[2].scatter(rho_bins, a_inf_o_avg_per_rho, marker=marker_dict[sce_key], color=co_dict[sce_key], label=r'out {0}'.format(label_dict[sce_key]),)
                    ax[2].fill_between(rho_bins, a_inf_o_l95_per_rho, a_inf_o_u95_per_rho, color=co_dict[sce_key], alpha=0.2)

                    a_avg = 0.00075
                    ax[2].axhline(a_avg, color='indigo', linestyle='dashed', alpha=1.0)
                    ax[2].text(0.7, 0.23, r"$\langle A\rangle=\sum_{{\ell}}A_{{\ell}}$", transform=ax[2].transAxes, fontsize=20, color='black')

    ax[0].set_xlabel(r'$\rho$', fontsize=40)
    ax[0].text(0.05, 0.9, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_ylabel(r'$R_{h(o),\rho}(\infty)/R_{\rho}(\infty)$', fontsize=40)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(-0.01, 1.0)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].legend(loc='upper right', fontsize=20)

    ax[1].text(0.05, 0.9, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1].set_xlabel(r'$\rho$', fontsize=40)
    ax[1].set_ylabel(r'$f_{\ell{\mathrm{\;inf}},\rho}$', fontsize=40)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(-0.01, 1.0)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].legend(loc='upper right', fontsize=20)

    ax[2].text(0.05, 0.9, r"C", transform=ax[2].transAxes, fontsize=40, color='black', weight="bold")
    ax[2].set_xlabel(r'$\rho$', fontsize=40)
    ax[2].set_ylabel(r'$A_{\ell{\mathrm{\;inf}},\rho}$', fontsize=40)
    ax[2].set_xlim(0.0, 1.0)
    ax[2].set_ylim(-0.00005, 0.0025)
    ax[2].tick_params(axis='both', labelsize=20)
    ax[2].legend(loc='upper right', fontsize=20)

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
    base_name = 'deprf5'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel6_old(
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

            out_sim_data = ut.load_depr_chapter_panel6_data_old(epi_fullname)
    
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

    chh_dict = {'b1hom': 'cornflowerblue', 'b1het': 'deepskyblue', 'depr': 'dodgerblue', 'plain': 'royalblue', 'uniform': 'mediumblue'}
    cho_dict = {'b1hom': 'darkseagreen', 'b1het': 'olivedrab', 'depr': 'teal', 'plain': 'lightseagreen', 'uniform': 'seagreen'}
    coh_dict = {'b1hom': 'crimson', 'b1het': 'orchid', 'depr': 'deeppink', 'plain': 'violet', 'uniform': 'darkorchid'}
    coo_dict = {'b1hom': 'lightcoral', 'b1het': 'salmon', 'depr': 'firebrick', 'plain': 'maroon', 'uniform': 'orangered'}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

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
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel6_results_old(
                    out_sim_data, 
                    agents_per_rho_sim=agents_per_rho_sim,
                    infected_per_rho_sim=infected_per_rho_sim,
                    events_hh_per_rho_sim=events_hh_per_rho_sim,
                    events_ho_per_rho_sim=events_ho_per_rho_sim,
                    events_oh_per_rho_sim=events_oh_per_rho_sim,
                    events_oo_per_rho_sim=events_oo_per_rho_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel6_stats_old(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim,
                events_hh_per_rho_sim=events_hh_per_rho_sim,
                events_ho_per_rho_sim=events_ho_per_rho_sim,
                events_oh_per_rho_sim=events_oh_per_rho_sim,
                events_oo_per_rho_sim=events_oo_per_rho_sim,
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

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    ax.scatter(rho_bins, events_hh_avg_per_rho, marker='o', color=chh_dict[sce_key], label=r'h-h {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_hh_l95_per_rho, events_hh_u95_per_rho, color=chh_dict[sce_key], alpha=0.2)
                    ax.scatter(rho_bins, events_ho_avg_per_rho, marker='s', color=cho_dict[sce_key], label=r'h-o {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_ho_l95_per_rho, events_ho_u95_per_rho, color=cho_dict[sce_key], alpha=0.2)
                    ax.scatter(rho_bins, events_oh_avg_per_rho, marker='v', color=coh_dict[sce_key], label=r'o-h {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_oh_l95_per_rho, events_oh_u95_per_rho, color=coh_dict[sce_key], alpha=0.2)
                    ax.scatter(rho_bins, events_oo_avg_per_rho, marker='P', color=coo_dict[sce_key], label=r'o-o {0}'.format(label_dict[sce_key]))
                    ax.fill_between(rho_bins, events_oo_l95_per_rho, events_oo_u95_per_rho, color=coo_dict[sce_key], alpha=0.2)

    #ax.text(0.05, 0.9, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax.set_xlabel(r'$\rho$', fontsize=25)
    ax.set_ylabel(r'event fraction', fontsize=25)
    #ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 1.0)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=25, loc='upper center')

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
    base_name = 'deprf6'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel7_old(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: new cases fraction rho profile for all
    f1: avg foi rho profile for all
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    R0 = 1.2
    mu = 0.1
    beta = R0 * mu
    
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

                out_sim_data = ut.load_depr_chapter_panel7_data_old(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 14))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            infected_h_per_rho_sim = []
            infected_o_per_rho_sim = []
            sum_avg_a_h_per_rho_sim = []
            sum_avg_a_o_per_rho_sim = []
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
            event_infector_rho_sim = []
            event_size_sim = []
            event_tot_pop_sim = []
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel7_results_old(
                    out_sim_data, 
                    agents_per_rho_sim,
                    infected_per_rho_sim,
                    infected_h_per_rho_sim,
                    infected_o_per_rho_sim,
                    sum_avg_a_h_per_rho_sim,
                    sum_avg_a_o_per_rho_sim,
                    sum_avg_foi_per_rho_sim,
                    sum_avg_pc_foi_per_rho_sim,
                    sum_avg_shared_per_rho_sim,
                    sum_avg_size_per_rho_sim,
                    sum_avg_t_pop_per_rho_sim,
                    sum_cum_i_pop_per_rho_sim,
                    sum_cum_shared_per_rho_sim,
                    sum_cum_size_per_rho_sim,
                    sum_cum_t_pop_per_rho_sim,
                    event_attractiveness_sim,
                    event_inf_pop_avg_rho_sim,
                    event_infector_rho_sim,
                    event_size_sim,
                    event_tot_pop_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel7_stats_old(
                agents_per_rho_sim=agents_per_rho_sim,
                infected_per_rho_sim=infected_per_rho_sim,
                infected_h_per_rho_sim=infected_h_per_rho_sim,
                infected_o_per_rho_sim=infected_o_per_rho_sim,
                sum_avg_a_h_per_rho_sim=sum_avg_a_h_per_rho_sim,
                sum_avg_a_o_per_rho_sim=sum_avg_a_o_per_rho_sim,
                sum_avg_foi_per_rho_sim=sum_avg_foi_per_rho_sim,
                sum_avg_pc_foi_per_rho_sim=sum_avg_pc_foi_per_rho_sim,
                sum_avg_shared_per_rho_sim=sum_avg_shared_per_rho_sim,
                sum_avg_size_per_rho_sim=sum_avg_size_per_rho_sim,
                sum_avg_t_pop_per_rho_sim=sum_avg_t_pop_per_rho_sim,
                sum_cum_i_pop_per_rho_sim=sum_cum_i_pop_per_rho_sim,
                sum_cum_shared_per_rho_sim=sum_cum_shared_per_rho_sim,
                sum_cum_size_per_rho_sim=sum_cum_size_per_rho_sim,
                sum_cum_t_pop_per_rho_sim=sum_cum_t_pop_per_rho_sim,
                prevalence_cutoff=0.025, 
                )
            
            fra_avg_a_h_avg_per_rho = processed_results['fra_avg_a_h_avg_per_rho']
            fra_avg_a_h_l95_per_rho = processed_results['fra_avg_a_h_l95_per_rho']
            fra_avg_a_h_u95_per_rho = processed_results['fra_avg_a_h_u95_per_rho']
            fra_avg_a_o_avg_per_rho = processed_results['fra_avg_a_o_avg_per_rho']
            fra_avg_a_o_l95_per_rho = processed_results['fra_avg_a_o_l95_per_rho']
            fra_avg_a_o_u95_per_rho = processed_results['fra_avg_a_o_u95_per_rho']
            fra_avg_a_avg_per_rho = processed_results['fra_avg_a_avg_per_rho']
            fra_avg_a_l95_per_rho = processed_results['fra_avg_a_l95_per_rho']
            fra_avg_a_u95_per_rho = processed_results['fra_avg_a_u95_per_rho']
            fra_avg_foi_avg_per_rho = processed_results['fra_avg_foi_avg_per_rho']
            fra_avg_foi_l95_per_rho = processed_results['fra_avg_foi_l95_per_rho']
            fra_avg_foi_u95_per_rho = processed_results['fra_avg_foi_u95_per_rho']
            fra_avg_pc_foi_avg_per_rho = processed_results['fra_avg_pc_foi_avg_per_rho']
            fra_avg_pc_foi_l95_per_rho = processed_results['fra_avg_pc_foi_l95_per_rho']
            fra_avg_pc_foi_u95_per_rho = processed_results['fra_avg_pc_foi_u95_per_rho']
            fra_avg_shared_avg_per_rho = processed_results['fra_avg_shared_avg_per_rho']
            fra_avg_shared_l95_per_rho = processed_results['fra_avg_shared_l95_per_rho']
            fra_avg_shared_u95_per_rho = processed_results['fra_avg_shared_u95_per_rho']
            fra_avg_size_avg_per_rho = processed_results['fra_avg_size_avg_per_rho']
            fra_avg_size_l95_per_rho = processed_results['fra_avg_size_l95_per_rho']
            fra_avg_size_u95_per_rho = processed_results['fra_avg_size_u95_per_rho']
            fra_avg_t_pop_avg_per_rho = processed_results['fra_avg_t_pop_avg_per_rho']
            fra_avg_t_pop_l95_per_rho = processed_results['fra_avg_t_pop_l95_per_rho']
            fra_avg_t_pop_u95_per_rho = processed_results['fra_avg_t_pop_u95_per_rho']
            fra_cum_shared_avg_per_rho = processed_results['fra_cum_shared_avg_per_rho']
            fra_cum_shared_l95_per_rho = processed_results['fra_cum_shared_l95_per_rho']
            fra_cum_shared_u95_per_rho = processed_results['fra_cum_shared_u95_per_rho']
            fra_cum_size_avg_per_rho = processed_results['fra_cum_size_avg_per_rho']
            fra_cum_size_l95_per_rho = processed_results['fra_cum_size_l95_per_rho']
            fra_cum_size_u95_per_rho = processed_results['fra_cum_size_u95_per_rho']
            fra_cum_t_pop_avg_per_rho = processed_results['fra_cum_t_pop_avg_per_rho']
            fra_cum_t_pop_l95_per_rho = processed_results['fra_cum_t_pop_l95_per_rho']
            fra_cum_t_pop_u95_per_rho = processed_results['fra_cum_t_pop_u95_per_rho']

            event_attractiveness_sim = list(chain(*event_attractiveness_sim))
            event_inf_pop_avg_rho_sim = list(chain(*event_inf_pop_avg_rho_sim))
            event_infector_rho_sim = list(chain(*event_infector_rho_sim))
            event_size_sim = list(chain(*event_size_sim))
            event_tot_pop_sim = list(chain(*event_tot_pop_sim))
        
            if (dist_key == 'Beta') and (sce_key == 'depr'):
                
                ax[0].scatter(rho_bins, fra_cum_size_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                ax[0].fill_between(rho_bins, fra_cum_size_l95_per_rho, fra_cum_size_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[0].axhline(new_cases_fraction_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[0].fill_between(rho_bins, new_cases_fraction_l95_global, new_cases_fraction_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, avg_foi_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, avg_foi_l95_per_rho, avg_foi_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_foi_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_foi_l95_global, avg_foi_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, fra_avg_foi_avg_per_rho, marker=marker_dict[sce_key], color='dodgerblue', label=r'home {0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, fra_cum_t_pop_l95_per_rho, fra_cum_t_pop_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_a_h_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_a_h_l95_global, avg_a_h_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, avg_a_o_avg_per_rho, marker=marker_dict[sce_key], color='firebrick', label=r'out {0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, avg_a_o_l95_per_rho, avg_a_o_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_a_o_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_a_o_l95_global, avg_a_o_u95_global, color=color_dict[sce_key], alpha=0.2)

                hb = ax[1].hexbin(event_attractiveness_sim, event_inf_pop_avg_rho_sim, gridsize=20, cmap='Blues')
                cb = plt.colorbar(hb, ax=ax[1])
                cb.set_label(label='counts', fontsize=25)
                cb.ax.tick_params(labelsize=25)

                #hb = ax[1].hexbin(
                #    event_attractiveness_sim,
                #    event_tot_pop_sim,
                #    C=event_inf_pop_avg_rho_sim, 
                #    gridsize=40,
                #    cmap='coolwarm',
                #)
                #cb = plt.colorbar(hb, ax=ax[1])
                #cb.set_label(label=r'$\langle\rho\rangle_{I_{{\ell}}}$', fontsize=25)
                #cb.ax.tick_params(labelsize=25)

            elif (dist_key == 'Beta') and (sce_key == 'uniform'):

                ax[0].scatter(rho_bins, fra_cum_size_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                ax[0].fill_between(rho_bins, fra_cum_size_l95_per_rho, fra_cum_size_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[0].axhline(new_cases_fraction_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[0].fill_between(rho_bins, new_cases_fraction_l95_global, new_cases_fraction_u95_global, color=color_dict[sce_key], alpha=0.2)

                ax[1].scatter(rho_bins, fra_avg_a_avg_per_rho, marker=marker_dict[sce_key], color='dodgerblue', label=r'home {0} {1}'.format(label_dict[sce_key], dist_key))
                ax[1].fill_between(rho_bins, fra_avg_a_l95_per_rho, fra_avg_a_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_a_h_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_a_h_l95_global, avg_a_h_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, avg_foi_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, avg_foi_l95_per_rho, avg_foi_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_foi_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_foi_l95_global, avg_foi_u95_global, color=color_dict[sce_key], alpha=0.2)

                hb = ax[1].hexbin(event_attractiveness_sim, event_inf_pop_avg_rho_sim, gridsize=20, cmap='Blues')
                cb = plt.colorbar(hb, ax=ax[1])
                cb.set_label(label='counts', fontsize=25)
                cb.ax.tick_params(labelsize=25)

            elif (dist_key == 'Gaussian') and (sce_key == 'depr'):

                ax[0].scatter(rho_bins, fra_cum_size_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                ax[0].fill_between(rho_bins, fra_cum_size_l95_per_rho, fra_cum_size_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[0].axhline(new_cases_fraction_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Gaussian {0}'.format(label_dict[sce_key]))
                #ax[0].fill_between(rho_bins, new_cases_fraction_l95_global, new_cases_fraction_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, fra_avg_a_avg_per_rho, marker=marker_dict[sce_key], color='dodgerblue', label=r'home {0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, fra_avg_a_l95_per_rho, fra_avg_a_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_a_h_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_a_h_l95_global, avg_a_h_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, avg_foi_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, avg_foi_l95_per_rho, avg_foi_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_foi_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Gaussian {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_foi_l95_global, avg_foi_u95_global, color=color_dict[sce_key], alpha=0.2)

                hb = ax[1].hexbin(event_attractiveness_sim, event_inf_pop_avg_rho_sim, gridsize=20, cmap='Blues')
                cb = plt.colorbar(hb, ax=ax[1])
                cb.set_label(label='counts', fontsize=25)
                cb.ax.tick_params(labelsize=25)
            
            elif (dist_key == 'Gaussian') and (sce_key == 'uniform'):

                ax[0].scatter(rho_bins, fra_cum_size_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                ax[0].fill_between(rho_bins, fra_cum_size_l95_per_rho, fra_cum_size_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[0].axhline(new_cases_fraction_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Gaussian {0}'.format(label_dict[sce_key]))
                #ax[0].fill_between(rho_bins, new_cases_fraction_l95_global, new_cases_fraction_u95_global, color=color_dict[sce_key], alpha=0.2)

                ax[1].scatter(rho_bins, fra_avg_a_avg_per_rho, marker=marker_dict[sce_key], color='dodgerblue', label=r'home {0} {1}'.format(label_dict[sce_key], dist_key))
                ax[1].fill_between(rho_bins, fra_avg_a_l95_per_rho, fra_avg_a_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_a_h_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Beta {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_a_h_l95_global, avg_a_h_u95_global, color=color_dict[sce_key], alpha=0.2)

                #ax[1].scatter(rho_bins, avg_foi_avg_per_rho, marker=marker_dict[sce_key], color=color_dict[sce_key], label=r'{0} {1}'.format(label_dict[sce_key], dist_key))
                #ax[1].fill_between(rho_bins, avg_foi_l95_per_rho, avg_foi_u95_per_rho, color=color_dict[sce_key], alpha=0.2)
                #ax[1].axhline(avg_foi_avg_global, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'global avg Gaussian {0}'.format(label_dict[sce_key]))
                #ax[1].fill_between(rho_bins, avg_foi_l95_global, avg_foi_u95_global, color=color_dict[sce_key], alpha=0.2)

                hb = ax[1].hexbin(event_attractiveness_sim, event_inf_pop_avg_rho_sim, gridsize=20, cmap='Blues')
                cb = plt.colorbar(hb, ax=ax[1])
                cb.set_label(label='counts', fontsize=25)
                cb.ax.tick_params(labelsize=25)

    ax[0].text(0.05, 0.9, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 2.2)
    ax[0].set_xlabel(r"$\rho$", fontsize=30)
    ax[0].set_ylabel(r"$\Delta I_{\rho}/I_{inf,\rho}$", fontsize=30)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].legend(fontsize=20)

    ax[1].text(0.05, 0.9, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[1].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 2.2)
    ax[1].set_xlabel(r"$A$ at event", fontsize=30)
    ax[1].set_ylabel(r"infected population $\langle\rho\rangle$ at event", fontsize=30)
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
    base_name = 'deprf7'
    extension_list = ['png']
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

                out_sim_data = ut.load_depr_chapter_panel6_data(epi_fullname)

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

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():

            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            infected_per_rho_sim = []
            nevents_eff_per_rho_sim = []
            sum_avg_foi_per_rho_sim = []
            sum_avg_pc_foi_per_rho_sim = []
            sum_avg_shared_per_rho_sim = []
            sum_avg_size_per_rho_sim = []
            sum_avg_t_pop_per_rho_sim = []
            sum_cum_i_pop_per_rho_sim = []
            sum_cum_shared_per_rho_sim = []
            sum_cum_size_per_rho_sim = []
            sum_cum_t_pop_per_rho_sim = []

            event_inf_pop_avg_rho_per_rho_sim = []
            event_infector_rho_per_rho_sim = []
            event_size_from_ipar_per_rho_sim = []
            event_size_from_ir_per_rho_sim = []
            event_tot_pop_from_ipar_per_rho_sim = []
            event_tot_pop_from_ir_per_rho_sim = []

            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel6_results(
                    out_sim_data, 
                    agents_per_rho_sim, 
                    infected_per_rho_sim, 
                    nevents_eff_per_rho_sim, 
                    sum_avg_foi_per_rho_sim, 
                    sum_avg_pc_foi_per_rho_sim, 
                    sum_avg_shared_per_rho_sim, 
                    sum_avg_size_per_rho_sim, 
                    sum_avg_t_pop_per_rho_sim, 
                    sum_cum_i_pop_per_rho_sim, 
                    sum_cum_shared_per_rho_sim, 
                    sum_cum_size_per_rho_sim, 
                    sum_cum_t_pop_per_rho_sim, 
                    event_inf_pop_avg_rho_per_rho_sim,
                    event_infector_rho_per_rho_sim,
                    event_size_from_ipar_per_rho_sim,
                    event_size_from_ir_per_rho_sim,
                    event_tot_pop_from_ipar_per_rho_sim,
                    event_tot_pop_from_ir_per_rho_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel6_stats(
                agents_per_rho_sim, 
                infected_per_rho_sim, 
                nevents_eff_per_rho_sim, 
                sum_avg_foi_per_rho_sim, 
                sum_avg_pc_foi_per_rho_sim, 
                sum_avg_shared_per_rho_sim, 
                sum_avg_size_per_rho_sim, 
                sum_avg_t_pop_per_rho_sim, 
                sum_cum_i_pop_per_rho_sim, 
                sum_cum_shared_per_rho_sim, 
                sum_cum_size_per_rho_sim, 
                sum_cum_t_pop_per_rho_sim, 
                event_inf_pop_avg_rho_per_rho_sim,
                event_infector_rho_per_rho_sim,
                event_size_from_ipar_per_rho_sim,
                event_size_from_ir_per_rho_sim,
                event_tot_pop_from_ipar_per_rho_sim,
                event_tot_pop_from_ir_per_rho_sim,
                prevalence_cutoff=0.025, 
                )

            #agents_avg_per_rho = processed_results['agents_avg_per_rho']
            #agents_l95_per_rho = processed_results['agents_l95_per_rho']
            #agents_u95_per_rho = processed_results['agents_u95_per_rho']
#
            #fra_infected_avg_per_rho = processed_results['fra_infected_avg_per_rho']
            #fra_infected_l95_per_rho = processed_results['fra_infected_l95_per_rho']
            #fra_infected_u95_per_rho = processed_results['fra_infected_u95_per_rho']

            fra_avg_shared_avg_per_rho = processed_results['fra_avg_shared_avg_per_rho']
            fra_avg_shared_l95_per_rho = processed_results['fra_avg_shared_l95_per_rho']
            fra_avg_shared_u95_per_rho = processed_results['fra_avg_shared_u95_per_rho']

            fra_avg_size_avg_per_rho = processed_results['fra_avg_size_avg_per_rho']
            fra_avg_size_l95_per_rho = processed_results['fra_avg_size_l95_per_rho']
            fra_avg_size_u95_per_rho = processed_results['fra_avg_size_u95_per_rho']

            fra_cum_size_avg_per_rho = processed_results['fra_cum_size_avg_per_rho']
            fra_cum_size_l95_per_rho = processed_results['fra_cum_size_l95_per_rho']
            fra_cum_size_u95_per_rho = processed_results['fra_cum_size_u95_per_rho']

            fra_cum_shared_avg_per_rho = processed_results['fra_cum_shared_avg_per_rho']
            fra_cum_shared_l95_per_rho = processed_results['fra_cum_shared_l95_per_rho']
            fra_cum_shared_u95_per_rho = processed_results['fra_cum_shared_u95_per_rho']

            fra_nevents_eff_avg_per_rho = processed_results['fra_nevents_eff_avg_per_rho']
            fra_nevents_eff_l95_per_rho = processed_results['fra_nevents_eff_l95_per_rho']
            fra_nevents_eff_u95_per_rho = processed_results['fra_nevents_eff_u95_per_rho']

            fra_ev_inf_pop_avg_rho_avg_per_rho = processed_results['fra_ev_inf_pop_avg_rho_avg_per_rho']
            fra_ev_inf_pop_avg_rho_l95_per_rho = processed_results['fra_ev_inf_pop_avg_rho_l95_per_rho']
            fra_ev_inf_pop_avg_rho_u95_per_rho = processed_results['fra_ev_inf_pop_avg_rho_u95_per_rho']

            fra_ev_infector_rho_avg_per_rho = processed_results['fra_ev_infector_rho_avg_per_rho']
            fra_ev_infector_rho_l95_per_rho = processed_results['fra_ev_infector_rho_l95_per_rho']
            fra_ev_infector_rho_u95_per_rho = processed_results['fra_ev_infector_rho_u95_per_rho']

            #infected_avg_per_rho = processed_results['infected_avg_per_rho']
            #infected_l95_per_rho = processed_results['infected_l95_per_rho']
            #infected_u95_per_rho = processed_results['infected_u95_per_rho']

            #nevents_eff_avg_per_rho = processed_results['nevents_eff_avg_per_rho']
            #nevents_eff_l95_per_rho = processed_results['nevents_eff_l95_per_rho']
            #nevents_eff_u95_per_rho = processed_results['nevents_eff_u95_per_rho']

            #sum_avg_size_avg_per_rho = processed_results['sum_avg_size_avg_per_rho']
            #sum_avg_size_l95_per_rho = processed_results['sum_avg_size_l95_per_rho']
            #sum_avg_size_u95_per_rho = processed_results['sum_avg_size_u95_per_rho']
            
            #sum_cum_size_avg_per_rho = processed_results['sum_cum_size_avg_per_rho']
            #sum_cum_size_l95_per_rho = processed_results['sum_cum_size_l95_per_rho']
            #sum_cum_size_u95_per_rho = processed_results['sum_cum_size_u95_per_rho']

            #sum_avg_shared_avg_per_rho = processed_results['sum_avg_shared_avg_per_rho']
            #sum_avg_shared_l95_per_rho = processed_results['sum_avg_shared_l95_per_rho']
            #sum_avg_shared_u95_per_rho = processed_results['sum_avg_shared_u95_per_rho']

            #fra_ev_inf_pop_avg_rho_avg = processed_results['fra_ev_inf_pop_avg_rho_avg']
            #fra_ev_inf_pop_avg_rho_l95 = processed_results['fra_ev_inf_pop_avg_rho_l95']
            #fra_ev_inf_pop_avg_rho_u95 = processed_results['fra_ev_inf_pop_avg_rho_u95']

            fra_cum_shared_avg = processed_results['fra_cum_shared_avg']
            fra_cum_shared_l95 = processed_results['fra_cum_shared_l95']
            fra_cum_shared_u95 = processed_results['fra_cum_shared_u95']

            fra_ev_infector_rho_avg = processed_results['fra_ev_infector_rho_avg']
            fra_ev_infector_rho_l95 = processed_results['fra_ev_infector_rho_l95']
            fra_ev_infector_rho_u95 = processed_results['fra_ev_infector_rho_u95']

            fra_ev_size_from_ipar_avg_per_rho = processed_results['fra_ev_size_from_ipar_avg_per_rho']
            fra_ev_size_from_ipar_l95_per_rho = processed_results['fra_ev_size_from_ipar_l95_per_rho']
            fra_ev_size_from_ipar_u95_per_rho = processed_results['fra_ev_size_from_ipar_u95_per_rho']

            fra_ev_size_from_ir_avg_per_rho = processed_results['fra_ev_size_from_ir_avg_per_rho']
            fra_ev_size_from_ir_l95_per_rho = processed_results['fra_ev_size_from_ir_l95_per_rho']
            fra_ev_size_from_ir_u95_per_rho = processed_results['fra_ev_size_from_ir_u95_per_rho']

            if dist_key == distribution_flags[0]:

                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    #ax[1].scatter(rho_bins, fra_cum_shared_avg_per_rho, linestyle='dashed', color='slateblue', alpha=1.0, label='event')
                    #ax[1].fill_between(rho_bins, fra_cum_shared_l95_per_rho, fra_cum_shared_u95_per_rho, color='slateblue', alpha=0.2)
                    #ax[1].scatter(rho_bins, fra_ev_size_from_ir_avg_per_rho, linestyle=linestyle_dict[sce_key], color=color_dict[sce_key], alpha=1.0, label=label_dict[sce_key])
                    #ax[1].fill_between(rho_bins, fra_ev_size_from_ir_l95_per_rho, fra_ev_size_from_ir_u95_per_rho, color=color_dict[sce_key], alpha=0.2)

                    ax[1].scatter(rho_bins, fra_nevents_eff_avg_per_rho, linestyle=linestyle_dict[sce_key], color=color_dict[sce_key], alpha=1.0, label=label_dict[sce_key])
                    ax[1].fill_between(rho_bins, fra_nevents_eff_l95_per_rho, fra_nevents_eff_u95_per_rho, color=color_dict[sce_key], alpha=0.2)

                    ax[1].axhline(1.0, color='gray', linestyle='dashed', alpha=0.25)
                    ax[1].axvline(0.5, color='gray', linestyle='dashed', alpha=0.25)

                elif sce_key == 'b1hom' or sce_key == 'plain':
                    #ax[1].axhline(fra_cum_shared_avg, color=color_dict[sce_key], linestyle='--', label=r'{0}'.format(label_dict[sce_key]),)
                    #ax[1].fill_between(rho_bins, fra_cum_shared_l95, fra_cum_shared_u95, color=color_dict[sce_key], alpha=0.2,)

                    ax[1].axhline(fra_ev_infector_rho_avg, color=color_dict[sce_key], linestyle='--', label=r'{0}'.format(label_dict[sce_key]),)
                    ax[1].fill_between(rho_bins, fra_ev_infector_rho_l95, fra_ev_infector_rho_u95, color=color_dict[sce_key], alpha=0.2,)

            elif dist_key == distribution_flags[1]:
                if sce_key == 'depr' or sce_key == 'uniform' or sce_key == 'b1het':
                    #ax[0].scatter(rho_bins, fra_cum_shared_avg_per_rho, linestyle='dashed', color='slateblue', alpha=1.0, label='event')
                    #ax[0].fill_between(rho_bins, fra_cum_shared_l95_per_rho, fra_cum_shared_u95_per_rho, color='slateblue', alpha=0.2)
                    #ax[0].scatter(rho_bins, fra_ev_size_from_ir_avg_per_rho, linestyle=linestyle_dict[sce_key], color=color_dict[sce_key], alpha=1.0, label=label_dict[sce_key])
                    #ax[0].fill_between(rho_bins, fra_ev_size_from_ir_l95_per_rho, fra_ev_size_from_ir_u95_per_rho, color=color_dict[sce_key], alpha=0.2)

                    ax[0].scatter(rho_bins, fra_nevents_eff_avg_per_rho, linestyle=linestyle_dict[sce_key], color=color_dict[sce_key], alpha=1.0, label=label_dict[sce_key])
                    ax[0].fill_between(rho_bins, fra_nevents_eff_l95_per_rho, fra_nevents_eff_u95_per_rho, color=color_dict[sce_key], alpha=0.2)

                    ax[0].axhline(1.0, color='gray', linestyle='dashed', alpha=0.25)
                    ax[0].axvline(0.5, color='gray', linestyle='dashed', alpha=0.25)
    
                elif sce_key == 'b1hom' or sce_key == 'plain':
                    #ax[0].axhline(fra_cum_shared_avg, color=color_dict[sce_key], linestyle=linestyle_dict[sce_key], label=r'{0}'.format(label_dict[sce_key]),)
                    #ax[0].fill_between(rho_bins, fra_cum_shared_l95, fra_cum_shared_u95, color=color_dict[sce_key], alpha=0.2,)

                    ax[0].axhline(fra_ev_infector_rho_avg, color=color_dict[sce_key], linestyle='--', label=r'{0}'.format(label_dict[sce_key]),)
                    ax[0].fill_between(rho_bins, fra_ev_infector_rho_l95, fra_ev_infector_rho_u95, color=color_dict[sce_key], alpha=0.2,)

    ax[0].set_title("Gaussian setting", fontsize=40)
    ax[0].text(0.05, 0.90, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_xlabel(r'$\rho$', fontsize=35)
    ax[0].set_ylabel(r'$(\Sigma_{\rho}/\Sigma)/(R_{\rho}(\infty)/R(\infty))$', fontsize=35)
    #ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].legend(fontsize=20, loc='lower center')

    ax[1].set_title("Beta setting", fontsize=40)
    ax[1].text(0.05, 0.90, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1].set_xlabel(r'$\rho$', fontsize=35)
    #ax[1].set_xlim(0.0, 1.0)
    #ax[1].set_ylim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].legend(fontsize=20, loc='lower center')

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
    base_name = 'deprf6'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel7(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: new cases fraction rho profile for all
    f1: avg foi rho profile for all
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    R0 = 1.2
    mu = 0.1
    beta = R0 * mu
    
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

                out_sim_data = ut.load_depr_chapter_panel7_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            avg_a_h_dist_per_rho_sim = []
            avg_a_o_dist_per_rho_sim = []
            events_hh_per_rho_sim = []
            events_ho_per_rho_sim = []
            events_oh_per_rho_sim = []
            events_oo_per_rho_sim = []
            f_inf_tr_h_dist_per_rho_sim = []
            f_inf_tr_o_dist_per_rho_sim = []
            infected_per_rho_sim = []
            infected_h_per_rho_sim = []
            infected_o_per_rho_sim = []
    
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel7_results(
                    out_sim_data, 
                    agents_per_rho_sim,
                    avg_a_h_dist_per_rho_sim,
                    avg_a_o_dist_per_rho_sim,
                    events_hh_per_rho_sim,
                    events_ho_per_rho_sim,
                    events_oh_per_rho_sim,
                    events_oo_per_rho_sim,
                    f_inf_tr_h_dist_per_rho_sim,
                    f_inf_tr_o_dist_per_rho_sim,
                    infected_per_rho_sim,
                    infected_h_per_rho_sim,
                    infected_o_per_rho_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel7_stats(
                agents_per_rho_sim,
                avg_a_h_dist_per_rho_sim,
                avg_a_o_dist_per_rho_sim,
                events_hh_per_rho_sim,
                events_ho_per_rho_sim,
                events_oh_per_rho_sim,
                events_oo_per_rho_sim,
                f_inf_tr_h_dist_per_rho_sim,
                f_inf_tr_o_dist_per_rho_sim,
                infected_per_rho_sim,
                infected_h_per_rho_sim,
                infected_o_per_rho_sim,
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

            events_h_avg_per_rho = events_hh_avg_per_rho + events_oh_avg_per_rho
            events_o_avg_per_rho = events_ho_avg_per_rho + events_oo_avg_per_rho
            events_h_l95_per_rho = events_hh_l95_per_rho + events_oh_l95_per_rho
            events_h_u95_per_rho = events_hh_u95_per_rho + events_oh_u95_per_rho
            events_o_l95_per_rho = events_oo_l95_per_rho + events_ho_l95_per_rho
            events_o_u95_per_rho = events_oo_u95_per_rho + events_ho_u95_per_rho

            f_inf_tr_h_avg_per_rho = processed_results['f_inf_tr_h_avg_per_rho']
            f_inf_tr_h_l95_per_rho = processed_results['f_inf_tr_h_l95_per_rho']
            f_inf_tr_h_u95_per_rho = processed_results['f_inf_tr_h_u95_per_rho']

            f_inf_tr_o_avg_per_rho = processed_results['f_inf_tr_o_avg_per_rho']
            f_inf_tr_o_l95_per_rho = processed_results['f_inf_tr_o_l95_per_rho']
            f_inf_tr_o_u95_per_rho = processed_results['f_inf_tr_o_u95_per_rho']

            avg_a_h_avg_per_rho = processed_results['avg_a_h_avg_per_rho']
            avg_a_h_l95_per_rho = processed_results['avg_a_h_l95_per_rho']
            avg_a_h_u95_per_rho = processed_results['avg_a_h_u95_per_rho']
            
            avg_a_o_avg_per_rho = processed_results['avg_a_o_avg_per_rho']
            avg_a_o_l95_per_rho = processed_results['avg_a_o_l95_per_rho']
            avg_a_o_u95_per_rho = processed_results['avg_a_o_u95_per_rho']

            if (dist_key == 'Beta') and (sce_key == 'depr'):

                ax[0].scatter(rho_bins, events_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[0].fill_between(rho_bins, events_h_l95_per_rho, events_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[0].scatter(rho_bins, events_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[0].fill_between(rho_bins, events_o_l95_per_rho, events_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(events_h_avg_per_rho - events_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[0].axvline(rho_bins[np.argmax(events_h_avg_per_rho)], color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axvline(rho_bins[np.argmax(events_o_avg_per_rho)], color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                ax[0].axhline(0.5, color='gray', linestyle='dotted', alpha=1.0)
                print("Intersection rho={0}".format(intersection_rho))

                ax[1].scatter(rho_bins, f_inf_tr_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[1].fill_between(rho_bins, f_inf_tr_h_l95_per_rho, f_inf_tr_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[1].scatter(rho_bins, f_inf_tr_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[1].fill_between(rho_bins, f_inf_tr_o_l95_per_rho, f_inf_tr_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(f_inf_tr_h_avg_per_rho - f_inf_tr_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[1].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                ax[1].axhline(0.5, color='gray', linestyle='dotted', alpha=1.0)
                print("Intersection rho={0}".format(intersection_rho))

                ax[2].scatter(rho_bins, avg_a_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[2].fill_between(rho_bins, avg_a_h_l95_per_rho, avg_a_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[2].scatter(rho_bins, avg_a_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[2].fill_between(rho_bins, avg_a_o_l95_per_rho, avg_a_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(avg_a_h_avg_per_rho - avg_a_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                #åax[2].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                print("Intersection rho={0}".format(intersection_rho))

                a_avg = 0.00075
                ax[2].axhline(a_avg, color='indigo', linestyle='dashed', alpha=1.0)
                ax[2].text(0.7, 0.17, r"$\langle A\rangle=\sum_{{\ell}}A_{{\ell}}$", transform=ax[2].transAxes, fontsize=20, color='black')

            elif (dist_key == 'Gaussian') and (sce_key == 'depr'):

                ax[0].scatter(rho_bins, events_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[0].fill_between(rho_bins, events_h_l95_per_rho, events_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[0].scatter(rho_bins, events_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[0].fill_between(rho_bins, events_o_l95_per_rho, events_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(events_h_avg_per_rho - events_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[0].axvline(rho_bins[np.argmax(events_h_avg_per_rho)], color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axvline(rho_bins[np.argmax(events_o_avg_per_rho)], color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                ax[0].axhline(0.5, color='gray', linestyle='dotted', alpha=1.0)
                print("Intersection rho={0}".format(intersection_rho))

                ax[1].scatter(rho_bins, f_inf_tr_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[1].fill_between(rho_bins, f_inf_tr_h_l95_per_rho, f_inf_tr_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[1].scatter(rho_bins, f_inf_tr_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[1].fill_between(rho_bins, f_inf_tr_o_l95_per_rho, f_inf_tr_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(f_inf_tr_h_avg_per_rho - f_inf_tr_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[1].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                ax[1].axhline(0.5, color='gray', linestyle='dotted', alpha=1.0)
                print("Intersection rho={0}".format(intersection_rho))

                ax[2].scatter(rho_bins, avg_a_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[2].fill_between(rho_bins, avg_a_h_l95_per_rho, avg_a_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[2].scatter(rho_bins, avg_a_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[2].fill_between(rho_bins, avg_a_o_l95_per_rho, avg_a_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(avg_a_h_avg_per_rho - avg_a_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                #åax[2].axvline(intersection_rho, color='gray', linestyle='dotted', alpha=1.0)
                print("Intersection rho={0}".format(intersection_rho))

                a_avg = 0.00075
                ax[2].axhline(a_avg, color='indigo', linestyle='dashed', alpha=1.0)
                ax[2].text(0.7, 0.17, r"$\langle A\rangle=\sum_{{\ell}}A_{{\ell}}$", transform=ax[2].transAxes, fontsize=20, color='black')

    ax[0].text(0.05, 0.9, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 2.2)
    ax[0].set_xlabel(r"$\rho$", fontsize=35)
    ax[0].set_ylabel(r"$E_{h(o),\rho}$", fontsize=35)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].legend(fontsize=20)

    ax[1].text(0.05, 0.9, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    ax[1].set_xlim(0.0, 1.0)
    #ax[1].set_ylim(0.0, 2.2)
    ax[1].set_xlabel(r"$\rho$", fontsize=35)
    ax[1].set_ylabel(r"$f_{h(o),T_I}$", fontsize=35)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].legend(fontsize=20)

    ax[2].text(0.05, 0.9, r"C", transform=ax[2].transAxes, fontsize=40, color='black', weight="bold")
    ax[2].set_xlim(0.0, 1.0)
    #ax[2].set_ylim(0.0, 2.2)
    ax[2].set_xlabel(r"$\rho$", fontsize=35)
    ax[2].set_ylabel(r"$\langle A_{h(o),\rho}\rangle_{T_I}$", fontsize=35)
    ax[2].tick_params(axis='both', labelsize=25)
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
    base_name = 'deprf7'
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_depr_panel8(
        distribution_flags=None,
        scenario_flags=None,
        ):
    """ 1x2 panel
    f0: new cases fraction rho profile for all
    f1: avg foi rho profile for all
    """

    lower_path = 'data/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_digested_epidemic_file_names(fullpath)

    prevalence_cutoff = 0.05
    R0 = 1.2
    mu = 0.1
    beta = R0 * mu
    
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

                out_sim_data = ut.load_depr_chapter_panel7_data(epi_fullname)

                if distribution_flag not in collected_output:
                    collected_output[distribution_flag] = {}

                if scenario_flag not in collected_output[distribution_flag]:
                    collected_output[distribution_flag][scenario_flag] = []

                collected_output[distribution_flag][scenario_flag].append(out_sim_data)

    color_dict = ut.build_color_dictionary()
    marker_dict = ut.build_marker_dictionary()
    linestyle_dict = ut.build_linestyle_dictionary()
    label_dict = ut.build_label_dictionary()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    for dist_key in collected_output.keys():
        
        inner_dict = collected_output[dist_key]
        
        for sce_key in inner_dict.keys():
            
            output_list = inner_dict[sce_key]

            agents_per_rho_sim = []
            events_hh_per_rho_sim = []
            events_ho_per_rho_sim = []
            events_oh_per_rho_sim = []
            events_oo_per_rho_sim = []
            f_inf_tr_h_dist_per_rho_sim = []
            f_inf_tr_o_dist_per_rho_sim = []
            infected_per_rho_sim = []
            infected_h_per_rho_sim = []
            infected_o_per_rho_sim = []
            sum_avg_a_h_per_rho_sim = []
            sum_avg_a_o_per_rho_sim = []
            
            for out_sim_data in output_list:
                ut.extend_depr_chapter_panel7_results(
                    out_sim_data, 
                    agents_per_rho_sim,
                    events_hh_per_rho_sim,
                    events_ho_per_rho_sim,
                    events_oh_per_rho_sim,
                    events_oo_per_rho_sim,
                    f_inf_tr_h_dist_per_rho_sim,
                    f_inf_tr_o_dist_per_rho_sim,
                    infected_per_rho_sim,
                    infected_h_per_rho_sim,
                    infected_o_per_rho_sim,
                    sum_avg_a_h_per_rho_sim,
                    sum_avg_a_o_per_rho_sim,
                    )

            processed_results = ut.compute_depr_chapter_panel8_stats(
                agents_per_rho_sim,
                events_hh_per_rho_sim,
                events_ho_per_rho_sim,
                events_oh_per_rho_sim,
                events_oo_per_rho_sim,
                f_inf_tr_h_dist_per_rho_sim,
                f_inf_tr_o_dist_per_rho_sim,
                infected_per_rho_sim,
                infected_h_per_rho_sim,
                infected_o_per_rho_sim,
                sum_avg_a_h_per_rho_sim,
                sum_avg_a_o_per_rho_sim,
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

            events_h_avg_per_rho = events_hh_avg_per_rho + events_oh_avg_per_rho
            events_o_avg_per_rho = events_ho_avg_per_rho + events_oo_avg_per_rho
            events_h_l95_per_rho = events_hh_l95_per_rho + events_oh_l95_per_rho
            events_h_u95_per_rho = events_hh_u95_per_rho + events_oh_u95_per_rho
            events_o_l95_per_rho = events_oo_l95_per_rho + events_ho_l95_per_rho
            events_o_u95_per_rho = events_oo_u95_per_rho + events_ho_u95_per_rho

            f_inf_tr_h_avg_per_rho = processed_results['f_inf_tr_h_avg_per_rho']
            f_inf_tr_h_l95_per_rho = processed_results['f_inf_tr_h_l95_per_rho']
            f_inf_tr_h_u95_per_rho = processed_results['f_inf_tr_h_u95_per_rho']

            f_inf_tr_o_avg_per_rho = processed_results['f_inf_tr_o_avg_per_rho']
            f_inf_tr_o_l95_per_rho = processed_results['f_inf_tr_o_l95_per_rho']
            f_inf_tr_o_u95_per_rho = processed_results['f_inf_tr_o_u95_per_rho']

            fra_avg_a_h_avg_per_rho = processed_results['fra_avg_a_h_avg_per_rho']
            fra_avg_a_h_l95_per_rho = processed_results['fra_avg_a_h_l95_per_rho']
            fra_avg_a_h_u95_per_rho = processed_results['fra_avg_a_h_u95_per_rho']
            
            fra_avg_a_o_avg_per_rho = processed_results['fra_avg_a_o_avg_per_rho']
            fra_avg_a_o_l95_per_rho = processed_results['fra_avg_a_o_l95_per_rho']
            fra_avg_a_o_u95_per_rho = processed_results['fra_avg_a_o_u95_per_rho']
            
            fra_avg_a_avg_per_rho = processed_results['fra_avg_a_avg_per_rho']
            fra_avg_a_l95_per_rho = processed_results['fra_avg_a_l95_per_rho']
            fra_avg_a_u95_per_rho = processed_results['fra_avg_a_u95_per_rho']
        
            if (dist_key == 'Beta') and (sce_key == 'depr'):

                ax[0].scatter(rho_bins, events_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[0].fill_between(rho_bins, events_h_l95_per_rho, events_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[0].scatter(rho_bins, events_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[0].fill_between(rho_bins, events_o_l95_per_rho, events_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(events_h_avg_per_rho - events_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[0].axvline(rho_bins[np.argmax(events_h_avg_per_rho)], color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axvline(rho_bins[np.argmax(events_o_avg_per_rho)], color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axvline(intersection_rho, color='gray', linestyle='dashed', alpha=0.25)
                ax[0].axhline(0.5, color='gray', linestyle='dotted', alpha=0.25)
                print("Intersection rho={0}".format(intersection_rho))

                ax[1].scatter(rho_bins, f_inf_tr_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[1].fill_between(rho_bins, f_inf_tr_h_l95_per_rho, f_inf_tr_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[1].scatter(rho_bins, f_inf_tr_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[1].fill_between(rho_bins, f_inf_tr_o_l95_per_rho, f_inf_tr_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(f_inf_tr_h_avg_per_rho - f_inf_tr_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[1].axvline(intersection_rho, color='gray', linestyle='dashed', alpha=0.25)
                ax[1].axhline(0.5, color='gray', linestyle='dotted', alpha=0.25)
                print("Intersection rho={0}".format(intersection_rho))

                ax[2].scatter(rho_bins, fra_avg_a_h_avg_per_rho, marker='o', color='dodgerblue', alpha=1.0, label=r'home')
                ax[2].fill_between(rho_bins, fra_avg_a_h_l95_per_rho, fra_avg_a_h_u95_per_rho, color='dodgerblue', alpha=0.2)
                ax[2].scatter(rho_bins, fra_avg_a_o_avg_per_rho, marker='o', color='firebrick', alpha=1.0, label=r'out')
                ax[2].fill_between(rho_bins, fra_avg_a_o_l95_per_rho, fra_avg_a_o_u95_per_rho, color='firebrick', alpha=0.2)

                intersection_index = np.argmin(np.abs(fra_avg_a_h_avg_per_rho - fra_avg_a_o_avg_per_rho))
                intersection_rho = rho_bins[intersection_index]
                ax[2].axvline(intersection_rho, color='gray', linestyle='dashed', alpha=0.25)
                print("Intersection rho={0}".format(intersection_rho))

                a_avg = 0.00075
                ax[2].axhline(a_avg, color='indigo', linestyle='dashed', alpha=1.0)
                ax[2].text(0.7, 0.9, r"$\langle A\rangle=\sum_{{\ell}}A_{{\ell}}$", transform=ax[2].transAxes, fontsize=20, color='black')

    ax[0].text(0.05, 0.9, r"A", transform=ax[0].transAxes, fontsize=40, color='black', weight="bold")
    ax[0].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 2.2)
    ax[0].set_xlabel(r"$\rho$", fontsize=35)
    ax[0].set_ylabel(r"$E_{h(o),\rho}$", fontsize=35)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].legend(fontsize=20)

    ax[1].text(0.05, 0.9, r"B", transform=ax[1].transAxes, fontsize=40, color='black', weight="bold")
    #ax[1].set_xlim(0.0, 1.0)
    #ax[0].set_ylim(0.0, 2.2)
    ax[1].set_xlabel(r"$\rho$", fontsize=35)
    ax[1].set_ylabel(r"$f_{h(o),T_I}$", fontsize=35)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].legend(fontsize=20)

    ax[2].text(0.05, 0.9, r"C", transform=ax[2].transAxes, fontsize=40, color='black', weight="bold")
    #ax[2].set_xlim(0.0, 1.0)
    #ax[2].set_ylim(0.0, 2.2)
    ax[2].set_xlabel(r"$\rho$", fontsize=35)
    ax[2].set_ylabel(r"$\langle A_{o,\rho}\rangle_{T_I}$", fontsize=35)
    ax[2].tick_params(axis='both', labelsize=25)
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
    base_name = 'deprf8'
    extension_list = ['png']
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
    scenario_flags = ['depr'] #, 'b1het', 'uniform', 'plain']

    #plot_depr_panel2(distribution_flags=distribution_flags, scenario_flags=scenario_flags, stats_flag=False, t_inv_flag=True)
    #plot_depr_panel3(distribution_flags=distribution_flags, scenario_flags=scenario_flags, stats_flag=False, r_inv_flag=True, r_inf_flag=True)
    #plot_depr_panel4(distribution_flags=distribution_flags, scenario_flags=scenario_flags, stats_flag=False, t_inf_flag=True)
    #plot_depr_panel5(distribution_flags=distribution_flags, scenario_flags=scenario_flags, stats_flag=False, f_inf_flag=True, a_inf_flag=True)
    plot_depr_panel6(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    plot_depr_panel7(distribution_flags=distribution_flags, scenario_flags=scenario_flags)
    #plot_depr_panel8(distribution_flags=distribution_flags, scenario_flags=scenario_flags)

if __name__ == '__main__':
    main()
