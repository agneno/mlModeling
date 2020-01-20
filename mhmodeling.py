import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import glob
import os

WINDOW_SIZE = 2160

# csv working files are saved in folder data/2018_srautai/
# check if combined file already exists
exists = os.path.isfile(r'data/2018_srautai/combined_csv.csv')
if exists:
    df = pd.read_csv(r'data/2018_srautai/combined_csv.csv', ';')
    print(f'Combined file exists')
else:
    os.chdir(r'data/2018_srautai')
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combine all files in the list
    df = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    df.to_csv(r'combined_csv.csv',index=False)
    df = pd.read_csv(r'combined_csv.csv', ';')
    print(f'Combined file generated')

# Vol (volume) – traffic volume (auto/h); 
# Occ (occupancy) – detector occupancy (% of measuring interval) – how much time of measuring
# interval (1 h) detector was occupied by cars; Spd (speed) – average speed (km/h).
df.columns = ['name', 'time', 'vol_orig', 'occ_orig',
              'spd_orig', 'vol_proc', 'occ_proc', 'spd_proc']

df.time = pd.to_datetime(df.time)
df.sort_values(['time'])
df.groupby(['time']).mean()
df.set_index('time', inplace=True)

# creating folder and changing path for results
os.makedirs(r'results', exist_ok=True)
os.chdir(r'results')

# identifying trends in the data
volume = df[['vol_orig']]
# taking a rolling average
volume.rolling(WINDOW_SIZE).mean().plot(figsize=(50, 10), linewidth=1, fontsize=30)
plt.xlabel('time', fontsize=20)
plt.savefig('volume')
occupation = df[['occ_orig']]
occupation.rolling(WINDOW_SIZE).mean().plot(figsize=(60, 10), linewidth=1, fontsize=30)
plt.xlabel('time', fontsize=20)
plt.savefig('occupation')
df_rm = pd.concat([volume.rolling(WINDOW_SIZE).mean(), occupation.rolling(WINDOW_SIZE).mean()], axis=1)
df_rm.plot(figsize=(60,10), linewidth=1, fontsize=30)
plt.xlabel('time', fontsize=20)
plt.savefig('volumeAndOccupationRolling')

#remove the trend
volume.diff(axis = 0, periods = 1).plot(figsize=(60, 10), linewidth=1, fontsize=30)
plt.xlabel('time', fontsize=20)
plt.savefig('volumeTrend')
occupation.diff(axis = 0, periods = 1).plot(figsize=(60, 10), linewidth=1, fontsize=30)
plt.xlabel('time', fontsize=20)
plt.savefig('occupationTrend')

# seasonal and periodicity patterns in volume
plot_acf(volume, lags=10000)
plt.savefig('volumeAutocorr')
print(f'Graphs are plotted and saved. Analysis is finished successfully.')
