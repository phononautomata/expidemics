import os

import utils as ut
import numpy as np
import matplotlib.pyplot as plt

import analysis as an

cwd_path = os.getcwd()

# Plotting functions
def plot_space_outline(filename=None, space_df=None, value_id='count', norm=True):
    if  space_df.empty:
        # Build full path name
        lower_path = 'data'
        fullpath = os.path.join(cwd_path, lower_path, filename)
        space_df = ut.open_file(fullpath)

    # Extract the attractiveness values
    attractiveness_values = space_df[value_id].values
    if norm == True:
        attractiveness_values = 1000.0 * attractiveness_values / np.sum(attractiveness_values)
        space_df[value_id] = attractiveness_values
    lon_array = space_df['lon_medoid'].values
    lat_array = space_df['lat_medoid'].values

    # Load OD rates
    nlocs = len(lon_array)
    od_rates = np.empty((nlocs, nlocs))
    od_rates = an.build_lonlat_gravity_law_od_rates(space_df, value_id)
    
    # Create the figure and subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot the attractiveness values as a regular lattice
    sc = ax[0, 0].scatter(lon_array, lat_array, c=attractiveness_values, cmap='viridis', alpha=0.5)
    ax[0, 0].set_title('Location scatter')
    ax[0, 0].set_xlabel('longitude', fontsize=20)
    ax[0, 0].set_ylabel('latitude', fontsize=20)
    cbar = fig.colorbar(sc, ax=ax[0, 0])
    cbar.set_label(r'$A$', fontsize=20)

    # Plot the histogram of attractiveness values
    bins = 30
    density = True
    ax[0, 1].hist(attractiveness_values, bins=bins, density=density, color='teal')
    ax[0, 1].set_xlabel(r'$A$', fontsize=20)
    ax[0, 1].set_ylabel(r'count', fontsize=20)
    ax[0, 1].set_title(r'Attractiveness distribution')
    
    # Plot the OD rates matrix
    img2 = ax[1, 0].imshow(od_rates, cmap='viridis', vmin=0, vmax=np.max(od_rates))
    ax[1, 0].set_xlabel(r'$A$', fontsize=20)
    ax[1, 0].set_title('OD rates matrix')
    cbar2 = fig.colorbar(img2, ax=ax[1, 0])
    cbar2.set_label(r'OD rate', fontsize=20)

    # Plot the histogram of OD rates values
    bins = 30
    density = True
    ax[1, 1].hist(od_rates.flatten(), bins=bins, density=density, color='teal')
    ax[1, 1].set_xlabel(r'OD rate', fontsize=20)
    ax[1, 1].set_ylabel(r'count', fontsize=20)
    ax[1, 1].set_title('OD rates distribution')

    # Settings
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Build full path name
    lower_path = 'results'
    filename = 'space_plot_' + filename
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

# Plotting functions
def plot_regular_lattice_outline(filename=None, value_id='attractiveness', norm=False):
    #if  space_df.empty:
    # Build full path name
    lower_path = 'data'
    fullpath = os.path.join(cwd_path, lower_path, filename)
    space_df = ut.open_file(fullpath)

    # Extract the attractiveness values
    attractiveness_values = space_df[value_id].values
    if norm == True:
        attractiveness_values = 1000.0 * attractiveness_values / np.sum(attractiveness_values)
        space_df[value_id] = attractiveness_values
    x_array = space_df['x'].values
    y_array = space_df['y'].values

    # Build lattice matrix
    Lx = np.max(space_df['j_index'].values) + 1
    Ly = np.max(space_df['i_index'].values) + 1
    A = np.empty((Ly, Lx))
    for i in range(Lx):
        for j in range(Ly):
            row = space_df[(space_df['i_index'] == i) & (space_df['j_index'] == j)]
            if len(row) > 0:
                A[Ly - 1 - j][i] = row[value_id].values[0]
            else:
                A[Ly - 1 - j][i] = 0
    
    # Load OD rates
    nlocs = len(x_array)
    od_rates = np.zeros((nlocs, nlocs))
    #od_rates = an.build_lonlat_gravity_law_od_rates(space_df, value_id)

    # Create the figure and subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    cutoff = 0.0000001  
    A_masked = np.ma.masked_where(A < cutoff, A)    

    # Create the plot
    im = ax[0].imshow(A_masked.T, cmap='viridis', origin='lower', interpolation='none')
    # Set color for masked elements (in this case, white)
    im.set_clim(vmin=np.min(A), vmax=np.max(A))
    # Add colorbar and labels
    cbar = fig.colorbar(im, ax=ax[0])
    cbar.set_label(r'$A$', fontsize=25)
    cbar.ax.yaxis.set_tick_params(labelsize=25)
    #ax[0].set_title(r'Boston reconstructed attractiveness field', fontsize=30)
    ax[0].set_xlabel("longitude (\u00b0 W)", fontsize=25)
    ax[0].set_ylabel("latitude (\u00b0 N)", fontsize=25)
    ax[0].tick_params(axis='both', labelsize=16)
   
    new_xticklabels = ['71.40', '71.28', '71.16', '71.04', '70.92', '70.80']
    new_yticklabels = ['42.1', '42.2', '42.3', '42.4', '42.5', '42.6']
    x_ticks_pos = range(0, 51, 10)
    y_ticks_pos = range(0, 51, 10)
    ax[0].set_xticks(x_ticks_pos)
    ax[0].set_yticks(y_ticks_pos)
    ax[0].set_xticklabels([new_xticklabels[pos] for pos in range(len(x_ticks_pos))])
    ax[0].set_yticklabels([new_yticklabels[pos] for pos in range(len(y_ticks_pos))])

    # Add label "A" to subplot 0
    ax[0].text(0.92, 0.9, "A", fontsize=40, color='red', transform=ax[0].transAxes)

    # Plot the log-log plot of attractiveness values
    bins = 30
    density = False
    attractiveness_values_filtered = np.array(attractiveness_values[attractiveness_values > cutoff])
    hist, bin_edges = np.histogram(attractiveness_values_filtered, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_centers = bin_centers.astype(float)
    hist = hist.astype(float)

    #ax[1].loglog(bin_centers, hist / np.sum(hist), color='teal', marker='o')
    ax[1].loglog(bin_centers, hist / np.sum(hist), 'o', color='teal', markersize=10, linestyle='dotted')
    #ax[1].plot(bin_centers, hist / np.sum(hist), color='teal', marker='o')
    #ax[1].plot(log_bin_centers, log_hist)

    a_avg = np.round(np.mean(attractiveness_values_filtered), 5)
    exponent = int(np.log10(a_avg) - 1)
    ax[1].axvline(a_avg, color='indigo', linestyle='--', alpha=0.2)
    ax[1].text(
        0.3, 0.1, r"$\langle A\rangle={:.1f} \times 10^{{{}}}$".format(7.5, exponent),
        transform=ax[1].transAxes, fontsize=20, color='black',
    )
    print(np.max(attractiveness_values_filtered))

    #ax[1].set_title('Attractiveness distribution', fontsize=30)
    ax[1].set_xlabel(r'$A$', fontsize=25)
    ax[1].set_ylabel(r'$P(A)$', fontsize=25)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].set_ylim(10e-5, 1.0)
    ax[1].set_xlim(10e-5, 10e-3)

    ax[1].text(0.92, 0.9, "B", fontsize=40, transform=ax[1].transAxes)

    # Settings
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Build full path name
    lower_path = 'results'
    filename = 'space_plot_' + filename
    fullpath = os.path.join(cwd_path, lower_path)
    filename = ut.trim_file_extension(filename)

    # Save plot
    extension_list = ['pdf', 'png']
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
    for ext in extension_list:
        full_name = os.path.join(fullpath, filename + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def main():
    # Build path
    #lower_path = 'data'
    #raw_filename = 'boston_space_object.csv'
    #full_path = os.path.join(cwd_path, lower_path, raw_filename)
    #space_df = ut.open_file(full_path)
    
    # Curation settings
    #curation_dict = {}
    #curation_dict['dms'] = False
    #curation_dict['north_lat'] = 42.6
    #curation_dict['south_lat'] = 42.1
    #curation_dict['west_lon'] = 71.4
    #curation_dict['east_lon'] = 70.8
    #rounding_decimals = 4
    #curation_dict['round'] = rounding_decimals
    #value_id = 'counts'
    #curation_dict['value'] = value_id

    # Curate
    #space_df = an.curate_space_df(space_df, curation_dict, limit_flag=True, round_flag=True)

    # Load curated space df
    #lower_path = 'data'
    #cur_str = ut.dict_to_string(curation_dict)
    #ext = '.pickle'
    #curated_filename = 'boston_df_' + cur_str

    # Plot
    #plot_space_outline(curated_filename + ext, space_df, value_id, norm=True)

    # Regular lattice
    #regular_dict = {}
    #regular_dict['DX'] = 50.0
    #regular_dict['DY'] = 50.0
    #regular_dict['dx'] = 1000.0
    #regular_dict['dy'] = 1000.0
    #regular_dict['LON0'] = 70.8
    #regular_dict['LAT0'] = 42.1
#
    #space_df = an.build_databased_regular_lattice_space_df(space_df, regular_dict, curation_dict)
#
    #ut.space_object_to_rust_as_json(space_df)

    # Load regular space df
    #lower_path = 'data'
    #reg_str = ut.dict_to_string(regular_dict)
    #cur_str = ut.dict_to_string(curation_dict)
    ext = '.pickle'
    #regular_filename = 'boston_rl_df_' + reg_str + '_' + cur_str

    #regular_filename = 'bl_DX50_DY50_LN070.8_LT042.1_rd3_x50_y50_DMSFalse_LNE70.8_LNW71.4_LTN42.6_LTS42.1_nl65366_rd3'
    regular_filename = 'bl_DX50_DY50_LN070.8_LT042.1_rd4_x50_y50_ts230721202506_DMSFalse_LNE70.8_LNW71.4_LTN42.6_LTS42.1_nl1589550_rd4'
    #regular_filename = 'bl_DX50_DY50_LN070.8_LT042.1_rd4_x100_y100_ts230725140842_DMSFalse_LNE70.8_LNW71.4_LTN42.6_LTS42.1_nl1589550_rd4'
    #regular_filename = 'bl_DX50_DY50_LN070.8_LT042.1_rd4_x50_y50_ts230725140842_DMSFalse_LNE70.8_LNW71.4_LTN42.6_LTS42.1_nl1589550_rd4'

    # Plot
    plot_regular_lattice_outline(regular_filename + ext)

if __name__ == "__main__":
    main()
