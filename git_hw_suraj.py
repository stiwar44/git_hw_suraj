import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import s3fs
import warnings
warnings.filterwarnings("ignore")
import geopandas as gpd
import os

# navigate to the path where corresponding NWM feature ids are stored for USGS gages
path1 = r'\\en4143590nas.sebe.dhcp.asu.edu\SHARED\NWM_RetrospectiveAnalysis'

# read the USGS gages and NWM feature id listed file as a dataframe
USGS_NWM = pd.read_csv(os.path.join(path1, 'USGS_gages_corresponding_NWM_ST.txt'),dtype = 'str')
USGS_NWM = USGS_NWM.set_index('USGS_gages')

# navigate to the path where USGS HUC8 gages discharges are stored
path2 = r'\\en4143590nas.sebe.dhcp.asu.edu\SHARED\US_Discharge\USGS\HUC08_gages_streamflow\Daily_in_cms'

# make a list of HUC8 gages whose discharges are present
HUC8_gages = [items[:-4] for items in os.listdir(path2)]

# subset the dataframe of USGS gage ID and NWM feature ID based on the HUC8 gages whose discharge are downloaded
USGS_NWM_HUC8 = USGS_NWM[USGS_NWM.index.isin(HUC8_gages)]

# nagivate to path where NWM simulated discharge corresponding to these HUC8 gages are stored
path3 = r'\\en4143590nas.sebe.dhcp.asu.edu\SHARED\US_Discharge\NWM\v_2_1_streamflow_output\NWM_corresponding_to_HUC8'

# make a list of NWM feature ids for which simulated flow are stored in the directory
NWM_HUC8 = [items[:-4] for items in os.listdir(path3) if items[-3:] =='txt']

# Initialize an empty dataframe with dates from 1830-01-01 to 2022-12-31 as index for USGS data
usgs_df = pd.DataFrame(index=pd.date_range('1830-01-01', '2022-12-31')).rename_axis("Date")

# Initialize an empty dataframe with hourly dates from 1979-02-01 to 2020-12-31 as index for NWM data
nwm_df = pd.DataFrame(index=pd.date_range('1979-02-01 01:00:00', '2020-12-31 23:00:00', freq='H')).rename_axis("Date")

# Placeholder list to store NWM feature IDs
nwm_feature_ids = []

# Iterate over rows in the mapping dataframe named 'USGS_NWM_HUC8'
for idx, row in USGS_NWM_HUC8.iterrows():
    
    usgs_gage_id = idx
    nwm_feature_ids.append(row['feature_id'])
    
    # Construct the file path for the USGS data
    usgs_file_path = os.path.join(path2, f'{usgs_gage_id}.txt')
    
    # Load USGS data from the constructed path and rename columns to match the exact USGS gage ID
    usgs_data = pd.read_csv(usgs_file_path, index_col='Date', parse_dates=True).rename(columns={'Q_cms': usgs_gage_id})
    
    # Remove timezone information from the index for merging
    usgs_data.index = usgs_data.index.tz_localize(None)
    
    # Left join the USGS data to the main USGS dataframe
    usgs_df = usgs_df.join(usgs_data, how='left')

# Extract unique (non-repeating) NWM feature IDs for loading data
unique_nwm_id = np.unique(nwm_feature_ids)

# Iterate over the unique NWM feature IDs to load and join data
for nwm_feature_id in unique_nwm_id:
    # Construct the file path for the NWM data
    nwm_file_path = os.path.join(path3, f'{nwm_feature_id}.txt')
    
    # Load NWM data from the constructed path and rename columns to match the feature ID
    nwm_data = pd.read_csv(nwm_file_path, index_col='Date', parse_dates=True).rename(
                columns={f'{nwm_feature_id}_Qcms': nwm_feature_id})
    
    # Remove timezone information from the index for merging
    nwm_data.index = nwm_data.index.tz_localize(None)
    
    # Left join the NWM data to the main NWM dataframe
    nwm_df = nwm_df.join(nwm_data, how='left')

# Resample the hourly NWM data to daily averages
nwm_daily_avg = nwm_df.resample('D').mean()

# define a function to calculate the Nash Sutcliffe Efficiency between observed and simulated data
def calculate_nse(obs, sim):
    # Calculate the Nash-Sutcliffe Efficiency (NSE).
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    return nse

# define pearson correlation function 
def pearson_r(obs,sim):
    r = np.sum((obs-np.mean(obs))*(sim-np.mean(sim)))/np.sqrt(np.sum((obs-np.mean(obs))**2)*np.sum((sim-np.mean(sim))**2))
    return r

# define a function to calculate the percentage bias between observed and simulated data
def calculate_pbias(obs, sim):
    # Calculate the Percent Bias (Pbias).
    return 100*np.sum(obs-sim)/np.sum(obs)

# define a function to calculate the relative bias between the observed and simulated data
def calculate_RB(obs, sim):
    return np.sum((sim-obs)*100/obs)/len(obs)

# define a function to return relative root mean square error
def RRMSE(obs, sim):
    return 100*np.sqrt((np.sum((obs-sim)**2))/len(obs))/np.sum(obs)

# define a function to return root mean squared relative error 
def RMSRE(obs, sim):
    return np.sqrt(np.sum(((obs-sim)*100/obs)**2)/len(obs))

# Lists to store calculated metrics
nse = []
r_squared = []
rb = []
rmsre = []
usgs_gages = []

# Iterate over the columns in the USGS dataframe
for usgs_col in usgs_df.columns:
    # Find the corresponding NWM feature ID for the given USGS column from the mapping dataframe
    nwm_col = USGS_NWM_HUC8.loc[usgs_col, 'feature_id']
    
    # Drop NA values from both USGS and NWM datasets to determine common date range
    usgs_data_after_drop = usgs_df[usgs_col].dropna()
    nwm_data_after_drop = nwm_daily_avg[nwm_col].dropna()
    
    # Continue only if both datasets after dropping NA values are non-empty
    if not usgs_data_after_drop.empty and not nwm_data_after_drop.empty:
        # Find the common date range between the two datasets
        common_start = max(usgs_data_after_drop.index[0], nwm_data_after_drop.index[0])
        common_end = min(usgs_data_after_drop.index[-1], nwm_data_after_drop.index[-1])
        
        # Skip current iteration if there's no overlapping date range
        if common_start >= common_end:
            continue
        
        # Append current USGS gage to the list
        usgs_gages.append(usgs_col)
       
        # Extract overlapping data for both datasets
        overlapping_usgs_data = usgs_df.loc[common_start:common_end, usgs_col]
        overlapping_nwm_data = nwm_daily_avg.loc[common_start:common_end, nwm_col]
        
        # Combine the overlapping data into a single dataframe
        combined_df = pd.DataFrame({
            f'{usgs_col}': overlapping_usgs_data,
            f'{nwm_col}': overlapping_nwm_data
        })
        
        # Extract non-NA values for calculations and scatter plots
        valid_data = combined_df.dropna()
        x = valid_data[usgs_col].values
        y_NWM = valid_data[nwm_col].values
        
        # Compute metrics (assuming the functions are defined elsewhere in your code)
        nse_NWM = calculate_nse(x, y_NWM)
        rb_NWM = calculate_RB(x, y_NWM)
        r_NWM = pearson_r(x, y_NWM)
        rmsre_NWM = RMSRE(x, y_NWM)
        
        # Append computed metrics to respective lists
        nse.append(nse_NWM)
        r_squared.append(r_NWM**2)
        rb.append(rb_NWM)
        rmsre.append(rmsre_NWM)
        
        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot the time series data in the first subplot
        ax[0].plot(combined_df.index, combined_df[usgs_col], label=f'USGS: {usgs_col}', lw=1.2)
        ax[0].plot(combined_df.index, combined_df[nwm_col], label=f'NWM feature: {nwm_col}', lw=0.75)
        ax[0].legend(frameon=False)
        ax[0].set_xlabel("Daily timeseries")
        ax[0].set_ylabel("Discharge (m$^3$/s)")
        
        # Scatter plot in the second subplot
        ax[1].scatter(x, y_NWM, marker='o', edgecolor='b', facecolor='none', s=5)
        ax[1].plot([0, max(x.max(), y_NWM.max())], [0, max(x.max(), y_NWM.max())], lw=0.5, c='gray')
        ax[1].set_xlabel("USGS discharge (m$^3$/s)")
        ax[1].set_ylabel("NWM discharge (m$^3$/s)")
        ax[1].grid(lw=0.3)
        ax[1].text(0.05, 0.845, f"NSE = {nse_NWM:.2f}  RB = {rb_NWM:.2f}\nR$^2$ = {r_NWM**2:.2f}  RMSRE = {rmsre_NWM:.2f}", 
            fontsize=12, color='b', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.5'),
                   transform=ax[1].transAxes)
        
        # Adjust the layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(fr'\\en4143590nas.sebe.dhcp.asu.edu\SHARED\US_Discharge\NWM\v_2_1_streamflow_output'
                    fr'\NWM_corresponding_to_HUC8\comparison_plot\USGS_{usgs_col}_vs_NWM_{nwm_col}.png')
        plt.show()

# define a target CRS to work on same coordinate system
target_crs = "EPSG:4269"
# Define the path to the shapefile containing USGS streamflow data
shapefile_path = r'D:\\1._Ph.D._Research_work\\ArcGIS_works\\GageLoc\\GageLoc.shp'
gages = gpd.read_file(shapefile_path)
gages = gages.set_index("SOURCE_FEA")

# # read US boundary shapefile
# read the USA state map's shapefile and project it to the target crs.
US_boundary = gpd.read_file(r"C:\Users\stiwar44.ASURITE\Dropbox (ASU)\Hydrology_Mascaro_Lab\Suraj\GIS_files\US_territory"
                           r"\CONUS_territory.shp")
# project the US boundary shapefile to target CRS
US_boundary = US_boundary.to_crs(target_crs)

# Subset the 'gages' dataframe to only include the gages that match the HUC8 USGS gages with available discharge data
gages_subset = gages.loc[usgs_gages]

# Add the Nash-Sutcliffe Efficiency (NSE) values to the subsetted dataframe
gages_subset['nse'] = nse

# Add the R-squared values to the subsetted dataframe
gages_subset['r_squared'] = r_squared

# Add the Relative Bias (RB) values to the subsetted dataframe
gages_subset['rb'] = rb

# Add the Root Mean Squared Relative Error (RMSRE) values to the subsetted dataframe
gages_subset['rmsre'] = rmsre

# Define condition1: Both NSE and R-squared values should be greater than 0.55
condition1 = (gages_subset['nse'] > 0.55) & (gages_subset['r_squared'] > 0.55)

# Define condition2: Absolute values of NSE should be >= 0.29, R-squared should be >= 0.3, 
# Relative Bias should be <= 500, and RMSRE should be <= 600
condition2 = (
    (gages_subset['nse'].abs() >= 0.29) &
    (gages_subset['r_squared'].abs() >= 0.3) &
    (gages_subset['rb'].abs() <= 500) &
    (gages_subset['rmsre'].abs() <= 600)
)

# Create a new column "classification" in the dataframe
# If either condition1 or condition2 is satisfied, classify as 'good', otherwise classify as 'bad'
gages_subset['classification'] = np.where(condition1 | condition2, 'good', 'bad')


# Initialize a figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the US boundary on the initialized axis
US_boundary.boundary.plot(ax=ax, color='gray', lw=0.5)

# Dictionary mapping classifications to their respective plot settings
plot_settings = {
    'good': {'facecolor': 'green', 'label': 'good correlation'},
    'bad': {'facecolor': 'red', 'label': 'bad correlation'}
}

# Iterate through the plot settings and plot each classification
for classification, settings in plot_settings.items():
    subset = gages_subset[gages_subset['classification'] == classification]
    subset.plot(ax=ax, marker='o', edgecolor='k', markersize=50, legend=True, **settings)

# Set title and labels for the axis
ax.set_title('Good vs Bad Performance', fontsize=16)
ax.set_xlabel("Lon (deg)", fontsize=14)
ax.set_ylabel("Lat (deg)", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)  # Set fontsize for tick labels
ax.legend()

# Display the plot
plt.show()

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_performance_metrics(data_subset, title):
    """
    Plots the performance metrics for a given data subset.

    Parameters:
    - data_subset: DataFrame containing the data subset to be plotted.
    - title: Title of the plot.
    """

    # Initialize a 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # List of columns to be plotted
    columns = ['nse', 'r_squared', 'rb', 'rmsre']
    
    # Iterate over each axis and column to plot
    for ax, col in zip(axs.flatten(), columns):
        # Plot the US boundary on the current axis
        US_boundary.boundary.plot(ax=ax, lw=0.25, color='gray')
        
        # Create a divider for the current axis for colorbar placement
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=-0.4)
        
        data_subset_finite = data_subset[[col,'geometry']].replace([np.inf, -np.inf], np.nan).dropna()
        data_subset_nan = data_subset[data_subset[col].isin([np.nan])]
        
        # Plot the geodata based on the current column
        data_subset_finite.plot(ax=ax, column=col, legend=True, cax=cax, cmap=cmap)
        
        if not data_subset_nan.empty:
            data_subset_nan.plot(ax= ax, marker='P',facecolor='blue',
                                edgecolor='k',markersize=35,lw=0.5)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(col, fontsize=12)
        
        # Create dummy artists with labels for filtered gages above
        legend_nan = plt.Line2D([0], [0], color= 'none', marker='P', markerfacecolor='blue',markeredgecolor = 'k',
                                 markersize=6, label='NaN')
        
        # Add all legends to the plot
        if not data_subset_nan.empty:
            ax.legend(frameon=False,ncols=1,handles=[legend_nan], fontsize=10, loc = 'lower left',
                      bbox_to_anchor=(-0.91, 0.01, 0, 0))
            
    # Adjust the layout for better visuals
    fig.subplots_adjust(hspace=-0.9, wspace=-0.25)
    plt.suptitle(title, y=0.74)
    plt.tight_layout()
    plt.show()

# Using the function for different subsets
plot_performance_metrics(gages_subset, "Performance metrics for entire dataset")