# git_hw_suraj
--------------------------------------------------------------------------------------------------------------
This project hosts a comprehensive Python-based tool for analyzing and visualizing streamflow data. It primarily focuses on evaluating and comparing streamflow data from US Geological Survey (USGS) gages and National Water Model (NWM) simulations. The tool is designed to assist hydrologists, environmental scientists, and data analysts in understanding and interpreting streamflow patterns, assessing model accuracy, and visualizing data both temporally and spatially.

**Features**
--------------------------------------------------------------------------------------------------------------
**Data Integration:** Efficiently reads and merges streamflow data from USGS gages and NWM feature IDs.
**Performance Metrics Calculation:** Calculates various hydrological performance metrics such as Nash Sutcliffe Efficiency, Pearson correlation, Percent Bias, Relative Bias, and Root Mean Squared Relative Error.
**Temporal and Spatial Visualization:** Generates time series and scatter plots to compare USGS and NWM data. Utilizes geospatial data for mapping the locations and performance of gages.
**Customizable Analysis:** The tool allows for the analysis of data across different time periods and spatial scales.

**Installation**
--------------------------------------------------------------------------------------------------------------
Before installing, ensure you have Python installed on your system. The tool requires Python 3.x.

**Dependencies**
--------------------------------------------------------------------------------------------------------------
The tool depends on several Python libraries including Pandas, NumPy, Matplotlib, Geopandas, and S3FS. These can be installed via pip:
pip install pandas numpy matplotlib geopandas s3fs

**Installing the Tool**
--------------------------------------------------------------------------------------------------------------
**Clone the Repository:** First, clone the repository to your local machine:
git clone https://github.com/stiwar44/git_hw_suraj
**Navigate to the Tool's Directory:** Change directory to the cloned repository:
cd [repository name]
**Install Additional Dependencies:** If there are any additional dependencies listed in a requirements file, install them using:
pip install -r requirements.txt

**Contact**
--------------------------------------------------------------------------------------------------------------
For any queries or further assistance, please reach out to stiwar44@asu.edu.

**Note:** Replace placeholder [repository name] with your repository directory.
