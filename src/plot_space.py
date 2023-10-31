import os
import collections
import itertools
import numpy as np
import pickle as pk
import pandas as pd
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

def plot_space_outline(lower_path, filename):
    # Build full path name
    fullpath = os.path.join(cwd_path, lower_path, filename)
    # Load space dataframe
    space_df = an.build_spatial_data_frame(fullpath)
    # Load OD rates
    od_rates = an.build_gravity_law_od_rates(space_df)

     # Extract the attractiveness values
    attractiveness_values = space_df['attractiveness'].values

    # Reshape the attractiveness values as a regular lattice
    i_indices = space_df['i_index'].values
    j_indices = space_df['j_index'].values
    max_i_index = np.max(i_indices)
    max_j_index = np.max(j_indices)
    attractiveness_lattice = np.zeros((max_j_index + 1, max_i_index + 1))
    attractiveness_lattice[j_indices, i_indices] = attractiveness_values
    
    # Create the figure and subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot the attractiveness values as a regular lattice
    img1 = ax[0, 0].imshow(attractiveness_lattice, cmap='viridis')
    ax[0, 0].set_title('Attractiveness Values (Lattice)')
    fig.colorbar(img1, ax=ax[0, 0])

    # Plot the histogram of attractiveness values
    ax[0, 1].hist(attractiveness_values, bins=20, color='teal')
    ax[0, 1].set_title('Attractiveness Histogram')
    
    # Plot the OD rates matrix
    img2 = ax[1, 0].imshow(od_rates, cmap='viridis', vmin=0, vmax=np.max(od_rates))
    ax[1, 0].set_title('OD Rates Matrix')
    fig.colorbar(img2, ax=ax[1, 0])

    # Plot the histogram of OD rates values
    ax[1, 1].hist(od_rates.flatten(), bins=20, color='teal')
    ax[1, 1].set_title('OD Rates Histogram')

    # Settings
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Build full path name
    lower_path = 'results'
    fullpath = os.path.join(cwd_path, lower_path, filename)
    fullpath = ut.trim_file_extension(fullpath)

    # Save plot
    extension_list = ['pdf', 'png']
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
    for ext in extension_list:
        full_name = os.path.join(fullpath + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()


def plot_scatter_locations(lower_path, filename):
    fullpath = os.path.join(cwd_path, lower_path, filename)
    #space_df = pd.read_csv(fullpath)
    space_df = ut.open_file(fullpath)

    # Extract arrays from columns
    attractiveness_array = space_df['counts'].values
    lon_array = space_df['lon_medoid'].values
    lat_array = space_df['lat_medoid'].values

    # Create a scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(lon_array, lat_array, c=attractiveness_array, cmap='viridis', alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter)

    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=25)
    ax.set_ylabel('Latitude', fontsize=25)
    cbar.set_label('Attractiveness', fontsize=25)

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    base_name = 'location_scatter_'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_attractiveness_distribution(lower_path, filename):
    fullpath = os.path.join(cwd_path, lower_path, filename)
    space_df = ut.open_file(fullpath)
    #space_df = pd.read_csv(fullpath)

    # Extract arrays from columns
    attractiveness_array = space_df['counts'].values

    # Create a histogram
    fig, ax = plt.subplots()
    bins = 30
    density = False
    ax.hist(attractiveness_array, bins=bins, density=density)

    # Set labels and title
    ax.set_xlabel('Attractiveness', fontsize=25)
    ax.set_ylabel('Frequency', fontsize=25)
    ax.set_title('Attractiveness Distribution', fontsize=25)

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    base_name = 'attractiveness_distribution_'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def main():

    lower_path = 'data'
    #filename = 'space_amGaussian_aa0_ab10_bmFinite_np50_pmRandomCartesian_x50_y50_ts230626190046.pickle'
    filename = 'bl_DX50_DY50_LN070.8_LT042.1_rd4_x50_y50_ts230725140842_DMSFalse_LNE70.8_LNW71.4_LTN42.6_LTS42.1_nl1589550_rd4.pickle'
    path = os.path.join(cwd_path, lower_path)
    plot_space_outline(path, filename)

    #filename = 'boston_space_df_d3.pickle'
    path = os.path.join(cwd_path, lower_path)
    plot_attractiveness_distribution(path, filename)
    plot_scatter_locations(path, filename)

    #fullpath = os.path.join(cwd_path, lower_path, filename)
    #space_df = pd.read_csv(fullpath)
    #space_df = an.round_coordinates(space_df, decimals=3)
    #space_df = an.coarse_grain_locations(space_df)
    #filename = 'boston_space_df_d3.pickle'
    #fullpath = os.path.join(cwd_path, lower_path, filename)
    #ut.save_to_pickle(space_df, fullpath)


  

if __name__ == "__main__":
    main()


