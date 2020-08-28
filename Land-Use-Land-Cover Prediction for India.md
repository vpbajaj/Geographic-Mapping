
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

