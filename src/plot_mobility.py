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

def plot_visit_trajectory_outline(fullpath, nlocs, nagents_load=1000):
    """
    Subplot 0: Scatter. Number of different visits vs. rho parameter.
    Subplot 1: Histogram. Number of different visits.
    Subplot 2: Scatter. Visit frequency for every location vs. rho parameter.
    Subplot 3: Histogram. Visit frequency.
    Subplot 4: Scatter. Top location visit frequency vs. rho parameter.
    Subplot 5: Histogram. Top location visit frequency.
    Subplot 6: Scatter. Home location visit frequency vs. rho parameter.
    Subplot 7: Histogram. Home location visit frequency.
    """

    # Build trajectory data frame
    mob_df = an.build_trajectory_data_frame(fullname=fullpath, nagents_load=nagents_load)

    # Prepare data structures for plots
    nagents = nagents_load
    rho_a = np.zeros(nagents)
    freq_diff_visits_a = np.zeros(nagents)
    toploc_visits_a = np.zeros(nagents)
    home_visit_freq_a = np.zeros(nagents)
    loc_freq_visits_a = []

    # Compute data to be plotted
    for a in range(nagents):
        rho_a[a] = an.get_agent_rho(mob_df, a)
        
        traj_a = an.get_agent_trajectory(mob_df, a)
        t_max = len(traj_a)

        loc_freq_visits = []
        for l in traj_a:
            loc_freq_visits.append(an.compute_location_visits(traj_a, l))
        loc_freq_visits_a.append(loc_freq_visits)

        different_visits = len(set(traj_a))
        freq_diff_visits_a[a] = different_visits / t_max
        
        toploc = an.compute_most_visited_location(trajectory=traj_a)
        toploc_visits_a[a] = toploc[1] / t_max
        
        home = traj_a[0]
        home_visit_freq_a[a] = an.compute_location_visits(traj_a, home) / t_max

    # Prepare figure template
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,8))
    title = 'Visits outline'
    fig.suptitle(title, fontsize=30)

    # SUBPLOT 00: Scatter. rho vs. freq_diff_visits_a
    ax[0, 0].scatter(rho_a, freq_diff_visits_a, color='teal')
    ax[0, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 0].set_ylabel(r'different visit rate $S/t_{{max}}$', fontsize=25)
    ax[0, 0].tick_params(axis='both', labelsize=15)

    # SUBPLOT 01: Histogram. freq_diff_visits_a
    density = True
    ax[0, 1].hist(freq_diff_visits_a, bins='auto', density=density, color='teal')
    ax[0, 1].set_xlabel(r'different visit rate $S/t_{{max}}$', fontsize=25)
    ax[0, 1].set_ylabel('norm. count', fontsize=25)
    ax[0, 1].tick_params(axis='both', labelsize=15)

    # SUBPLOT 02: Scatter. rho vs. loc_freq_visits_a
    max_locs = max(len(loc_freq_visits) for loc_freq_visits in loc_freq_visits_a)
    for a in range(nagents):
        rho_extend = np.full(max_locs, rho_a[a])
        ax[0, 2].scatter(rho_extend, loc_freq_visits_a[a])
    ax[0, 2].set_xlabel(r'$\rho$', fontsize=25)
    ax[0, 2].set_ylabel(r'$f$', fontsize=25)
    ax[0, 2].tick_params(axis='both', labelsize=15)

    # SUBPLOT 03: Histogram. loc_frequ_visits_a
    flattened_loc_freq_visits = [visit for sublist in loc_freq_visits_a for visit in sublist]
    ax[0, 3].hist(flattened_loc_freq_visits, bins='auto', density=density, color='teal')
    ax[0, 3].set_xlabel(r'$f$', fontsize=25)
    ax[0, 3].set_ylabel('norm. count', fontsize=25)
    ax[0, 3].tick_params(axis='both', labelsize=15)

    # SUBPLOT 10: Scatter. rho vs. toploc_visits_a
    ax[1, 0].scatter(rho_a, toploc_visits_a, color='teal')
    ax[1, 0].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 0].set_ylabel(r'$f_{{top}}$', fontsize=25)
    ax[1, 0].tick_params(axis='both', labelsize=15)

    # SUBPLOT 11: Histogram. toploc_visits_a
    ax[1, 1].hist(toploc_visits_a, bins='auto', density=density, color='teal')
    ax[1, 1].set_xlabel(r'$f_{{top}}$', fontsize=25)
    ax[1, 1].set_ylabel('norm. count', fontsize=25)
    ax[1, 1].tick_params(axis='both', labelsize=15)

    # SUBPLOT 12: Scatter. rho vs. home_visits_a
    ax[1, 2].scatter(rho_a, home_visit_freq_a, color='teal')
    ax[1, 2].set_xlabel(r'$\rho$', fontsize=25)
    ax[1, 2].set_ylabel(r'$f_{{home}}$', fontsize=25)
    ax[1, 2].tick_params(axis='both', labelsize=15)

    # SUBPLOT 13: Histogram. home_visits_a  
    ax[1, 3].hist(home_visit_freq_a, bins='auto', density=density, color='teal')
    ax[1, 3].set_xlabel(r'$f_{{home}}$', fontsize=25)
    ax[1, 3].set_ylabel('norm. count', fontsize=25)
    ax[1, 3].tick_params(axis='both', labelsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    plt.show()

def plot_visit_number_frequency(pars):

    # Load grid data

    # Transform grid data into trajectory dataframe

    # Load epidemic data
    pass

def plot_total_visits(fullpath, nagents_load=1000):
    """
    Subplot 0: Total visits in every location (lattice structure preserved).
    Subplot 1: How many agents have a every location as top location. (lsp).
    Subplot 2: Unique visits to every location (how many different agents).
    """
    
    # Prepare figure

    # Figure settings
    pass
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_visit_frequency_and_attractiveness(full_path):
    """
    Subplot 00: f_top-distribution for every agent
    Subplot 01: f_home-distribution for every agent
    Subplot 10: A_top-distribution for every agent
    Subplot 11: A_home-distribution for every agent
    """
    
    pass
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_first_time_visits(full_path):
    """
    Subplot 00: Average visit time for every location
    Subplot 01: Average visit time of 25% visitors
    Subplot 02: Average visit time of 50% visitors
    Subplot 10: P25-time
    Subplot 11: P50-time
    Subplot 12: P75-time
    """
    pass
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_time_to_poles(full_path):
    """
    Subplot 00: How many time to reach what became top location
    Subplot 01: How many time to reach 2-most visited
    Subplot 10: How many time to reach 50%-most visited
    Subplot 12: How many time to reach 75%-most visited
    """
    pass
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_rho_spatial_distribution(full_path):
    """
    Subplot 0: <rho> spatial distribution
    Subplot 1: <rho> lower 95% CI
    Subplot 2: <rho> upper 95% CI
    """
    pass
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()


def main():

    lower_path = 'data'
    filename = "mdyna_gm0.21_hw25_t1200_rmBeta_ra2_rb2_space_amGaussian_aa0_ab10_bmFinite_np50_pmRandomCartesian_x50_y50_ts230626184935.pickle"
    fullpath = os.path.join(cwd_path, lower_path, filename)
    #mob_df = an.build_trajectory_data_frame(fullname=fullpath, nagents_load=50000)
    
    plot_visit_trajectory_outline(fullpath, nlocs=2500, nagents_load=1000)

    
if __name__ == "__main__":
    main()






