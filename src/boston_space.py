import os
import sys
import pickle
import pandas as pd

cwd = os.getcwd()

file_path = '/data/stays/stays2_14460.csv.gz'
df = pd.read_csv(file_path, compression='gzip')

filtered_df = df[['user', 'duration', 'lon_medoid', 'lat_medoid', 'lon_home', 'lat_home']]

decimals = 4
filtered_df['lon_medoid'] = filtered_df['lon_medoid'].apply(lambda x: round(x, decimals))
filtered_df['lat_medoid'] = filtered_df['lat_medoid'].apply(lambda x: round(x, decimals))

lon_mapping = {value: index for index, value in enumerate(sorted(filtered_df['lon_medoid'].unique()))}
lat_mapping = {value: index for index, value in enumerate(sorted(filtered_df['lat_medoid'].unique()))}

filtered_df['i_index'] = filtered_df['lon_medoid'].map(lon_mapping)
filtered_df['j_index'] = filtered_df['lat_medoid'].map(lat_mapping)

#max_i_index = filtered_df['i_index'].max()
#max_j_index = filtered_df['j_index'].max()

sorted_df = filtered_df.sort_values(['i_index', 'j_index'], ascending=[True, True])
sorted_df['loc_id'] = pd.factorize(list(zip(sorted_df['i_index'], sorted_df['j_index'])))[0]
#max_l_index = sorted_df['loc_id'].max()
total_rows = len(sorted_df)
nlocs = sorted_df['loc_id'].nunique()

appended_loc_ids = {}

space_rows = []

progress_interval = 50000
count = 0
loc_count = 0

for idx, row in sorted_df.iterrows():
    loc_id = row['loc_id']

    if loc_id not in appended_loc_ids:
        lon = row['lon_medoid']
        lat = row['lat_medoid']
        i_index = row['i_index']
        j_index = row['j_index']

        counts = sorted_df[sorted_df['loc_id'] == loc_id].shape[0]
        unique_visitor_counts = len(sorted_df[sorted_df['loc_id'] == loc_id]['user'].unique())
        cum_duration = sorted_df[sorted_df['loc_id'] == loc_id]['duration'].sum()

        space_row = {
            'loc_id': loc_id,
            'lon_medoid': lon,
            'lat_medoid': lat,
            'i_index': i_index,
            'j_index': j_index,
            'counts': counts,
            'unique_counts': unique_visitor_counts,
            'cum_duration': cum_duration
        }

        space_rows.append(space_row)

        appended_loc_ids[loc_id] = True
        loc_count += 1

    if count % progress_interval == 0:
        print("Process {} rows of total {}... ".format(count, total_rows))
        print("Added {} locations of total {}".format(loc_count, nlocs))
        sys.stdout.flush()
    count += 1

space_df = pd.DataFrame(space_rows)

filename = 'boston_space_object' # filename = 'boston_home_census'
ext = '.csv'
fullname = os.path.join(cwd, filename + ext)
space_df.to_csv(fullname)
