```python
# Firstly, some necessary imports

# Jupyter notebook related
%reload_ext autoreload
%autoreload 2
%matplotlib inline

# Built-in modules
import pickle
import sys
import os
import datetime
import itertools
from aenum import MultiValueEnum

# Basics of Python data handling and visualization
import numpy as np
np.random.seed(42)
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from tqdm.auto import tqdm

# Machine learning
import lightgbm as lgb
import joblib
from sklearn import metrics
from sklearn import preprocessing

# Imports from eo-learn and sentinelhub-py
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor, ExtractBandsTask, MergeFeatureTask
from eolearn.io import SentinelHubInputTask, ExportToTiff
from eolearn.mask import AddMultiCloudMaskTask, AddValidDataMaskTask
from eolearn.geometry import VectorToRaster, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolation, SimpleFilterTask, NormalizedDifferenceIndexTask
from sentinelhub import UtmZoneSplitter, BBox, CRS, DataSource, UtmGridSplitter
```


```python
# Folder where data for running the notebook is stored
DATA_FOLDER = os.path.join('/Users/admin/Desktop', 'ArcGIS Data')

# Load geojson file
country_india = gpd.read_file(os.path.join(DATA_FOLDER, 'intersection_aqueduct_india.shp'))
country_india.crs
```




    <Geographic 2D CRS: EPSG:4326>
    Name: WGS 84
    Axis Info [ellipsoidal]:
    - Lat[north]: Geodetic latitude (degree)
    - Lon[east]: Geodetic longitude (degree)
    Area of Use:
    - name: World
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich




```python
country_india = country_india.to_crs(epsg=32643)
country_india.crs
```




    <Projected CRS: EPSG:32643>
    Name: WGS 84 / UTM zone 43N
    Axis Info [cartesian]:
    - E[east]: Easting (metre)
    - N[north]: Northing (metre)
    Area of Use:
    - name: World - N hemisphere - 72°E to 78°E - by country
    - bounds: (72.0, 0.0, 78.0, 84.0)
    Coordinate Operation:
    - name: UTM zone 43N
    - method: Transverse Mercator
    Datum: World Geodetic System 1984
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich




```python
country_india.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GU</th>
      <th>Shape_Leng</th>
      <th>Shape_Area</th>
      <th>BasinID</th>
      <th>COUNTRY</th>
      <th>BASIN_NAME</th>
      <th>WITHDRAWAL</th>
      <th>CONSUMPTIO</th>
      <th>BA</th>
      <th>BWS</th>
      <th>...</th>
      <th>W_POWER</th>
      <th>W_MINE</th>
      <th>W_OILGAS</th>
      <th>DEF_PQUANT</th>
      <th>W_AGR</th>
      <th>W_FOODBV</th>
      <th>W_TEX</th>
      <th>OWR_cat</th>
      <th>st_nm</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7807</td>
      <td>2.571923</td>
      <td>0.073414</td>
      <td>4569</td>
      <td>Bangladesh</td>
      <td>None</td>
      <td>307930400.0</td>
      <td>1.753866e+08</td>
      <td>6.763807e+09</td>
      <td>0.045526</td>
      <td>...</td>
      <td>1.513997</td>
      <td>3.281757</td>
      <td>3.411456</td>
      <td>1.931001</td>
      <td>1.954313</td>
      <td>2.639725</td>
      <td>2.236996</td>
      <td>Medium to high risk (2-3)</td>
      <td>Tripura</td>
      <td>MULTIPOLYGON (((2194194.564 2668633.150, 21920...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7809</td>
      <td>6.234207</td>
      <td>0.538332</td>
      <td>4571</td>
      <td>Bangladesh</td>
      <td>None</td>
      <td>149308368.0</td>
      <td>8.258270e+07</td>
      <td>1.882137e+10</td>
      <td>0.007933</td>
      <td>...</td>
      <td>1.555468</td>
      <td>3.206916</td>
      <td>3.299842</td>
      <td>1.389035</td>
      <td>1.681410</td>
      <td>2.265866</td>
      <td>1.916721</td>
      <td>Medium to high risk (2-3)</td>
      <td>Tripura</td>
      <td>MULTIPOLYGON (((2275734.064 2726525.554, 22756...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7822</td>
      <td>2.575702</td>
      <td>0.093155</td>
      <td>4583</td>
      <td>China</td>
      <td>BRAHMAPUTRA</td>
      <td>1432244.0</td>
      <td>8.325534e+05</td>
      <td>7.304932e+08</td>
      <td>0.001961</td>
      <td>...</td>
      <td>1.074833</td>
      <td>1.707919</td>
      <td>1.856967</td>
      <td>0.696584</td>
      <td>1.066773</td>
      <td>1.475756</td>
      <td>1.181928</td>
      <td>Low to medium risk (1-2)</td>
      <td>Sikkim</td>
      <td>POLYGON ((1869311.164 3174260.247, 1869300.136...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8009</td>
      <td>2.235312</td>
      <td>0.099651</td>
      <td>4991</td>
      <td>China</td>
      <td>TARIM</td>
      <td>455921.0</td>
      <td>4.426724e+04</td>
      <td>2.571421e+07</td>
      <td>0.017730</td>
      <td>...</td>
      <td>3.385144</td>
      <td>3.184017</td>
      <td>2.678851</td>
      <td>4.252955</td>
      <td>3.747108</td>
      <td>3.154807</td>
      <td>3.426857</td>
      <td>High risk (3-4)</td>
      <td>Jammu &amp; Kashmir</td>
      <td>MULTIPOLYGON (((836885.167 3975795.030, 836892...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8262</td>
      <td>4.061245</td>
      <td>0.302299</td>
      <td>5205</td>
      <td>China</td>
      <td>TARIM</td>
      <td>338800.0</td>
      <td>3.288652e+04</td>
      <td>5.436473e+07</td>
      <td>0.006232</td>
      <td>...</td>
      <td>3.329202</td>
      <td>3.277878</td>
      <td>2.647033</td>
      <td>4.453084</td>
      <td>3.697437</td>
      <td>3.176306</td>
      <td>3.458442</td>
      <td>High risk (3-4)</td>
      <td>Jammu &amp; Kashmir</td>
      <td>POLYGON ((939404.212 3947257.592, 939669.181 3...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>




```python
karnataka = country_india[country_india.st_nm=='Karnataka']
karnataka.crs
```




    <Projected CRS: EPSG:32643>
    Name: WGS 84 / UTM zone 43N
    Axis Info [cartesian]:
    - E[east]: Easting (metre)
    - N[north]: Northing (metre)
    Area of Use:
    - name: World - N hemisphere - 72°E to 78°E - by country
    - bounds: (72.0, 0.0, 78.0, 84.0)
    Coordinate Operation:
    - name: UTM zone 43N
    - method: Transverse Mercator
    Datum: World Geodetic System 1984
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich




```python
# Get Karnataka's shape in polygon format
karnataka_shape = karnataka.geometry.values[-1]
# Plot state
karnataka.plot()
plt.axis('off');

# Print size 
print('Dimension of the area is {0:.0f} x {1:.0f} m2'.format(karnataka_shape.bounds[2] - karnataka_shape.bounds[0],
                                                             karnataka_shape.bounds[3] - karnataka_shape.bounds[1]))
```

    Dimension of the area is 255938 x 163168 m2



![png](output_5_1.png)



```python
# Create the splitter to obtain a list of bboxes
bbox_splitter = UtmZoneSplitter([karnataka_shape], karnataka.crs, 5000)

bbox_list = np.array(bbox_splitter.get_bbox_list())
info_list = np.array(bbox_splitter.get_info_list())
#show_splitter(bbox_splitter, show_legend=True)
```


```python
len(info_list)
```




    914




```python

# Prepare info of selected EOPatches
geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
idxs = [info['index'] for info in info_list]
idxs_x = [info['index_x'] for info in info_list]
idxs_y = [info['index_y'] for info in info_list]

gdf = gpd.GeoDataFrame({'index': idxs, 'index_x': idxs_x, 'index_y': idxs_y},
                           crs=karnataka.crs,
                           geometry=geometry)
```


```python
# select a 5x5 area (id of center patch)
ID = 407

# Obtain surrounding 5x5 patches
patchIDs = []
for idx, [bbox, info] in enumerate(zip(bbox_list, info_list)):
    if (abs(info['index_x'] - info_list[ID]['index_x']) <= 2 and
        abs(info['index_y'] - info_list[ID]['index_y']) <= 2):
        patchIDs.append(idx)

# Check if final size is 5x5
if len(patchIDs) != 5*5:
    print('Warning! Use a different central patch ID, this one is on the border.')

# Change the order of the patches (used for plotting later)
patchIDs = np.transpose(np.fliplr(np.array(patchIDs).reshape(5, 5))).ravel()
```


```python
# figure
fig, ax = plt.subplots(figsize=(30, 30))
gdf.plot(ax=ax,facecolor='w',edgecolor='r',alpha=0.5)
karnataka.plot(ax=ax, facecolor='w',edgecolor='b',alpha=0.5)
ax.set_title('Selected 5x5  tiles from Karnataka', fontsize=25);
for bbox, info in zip(bbox_list, info_list):
    geo = bbox.geometry
    ax.text(geo.centroid.x, geo.centroid.y, info['index'], ha='center', va='center')
    
gdf[gdf.index.isin(patchIDs)].plot(ax=ax,facecolor='g',edgecolor='r',alpha=0.5)

plt.axis('off');
```


![png](output_10_0.png)



```python
gdf.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1379799e8>




![png](output_11_1.png)

