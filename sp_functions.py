#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree  
import numpy as np

import geopandas as gpd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import json
import requests
from shapely.geometry import Point, MultiPoint, Polygon, shape, LineString



def extract_poly_coords(geom):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return {'exterior_coords': exterior_coords}

def get_bldg_vertex(geodf, column_id):
    bldg_vertex_data = gpd.GeoDataFrame()
    for row in tqdm(geodf[str(column_id)].index):
        tmp_geo = extract_poly_coords(geodf['geometry'].loc[row])
        for pt in list(tmp_geo['exterior_coords']):
            pt = gpd.GeoDataFrame(geometry = gpd.GeoSeries(Point(pt)), 
                                  data= pd.DataFrame([geodf[column_id].loc[row]], 
                                                     columns=['UID']))
            bldg_vertex_data = bldg_vertex_data.append(pt)
            
    return bldg_vertex_data

def get_line_centroid(vertice_df):
    bldg_centroid = gpd.GeoDataFrame()
    for idx in vertice_df['UID'].index:
        line_center = LineString([vertice_df['geometry'].loc[idx], \
                                  vertice_df['geometry2'].loc[idx]]).centroid
        pt = gpd.GeoDataFrame(geometry = gpd.GeoSeries(line_center), 
                              crs = {'init':'epsg4326'},
                              data= pd.DataFrame([vertice_df['UID'].loc[idx]], columns=['UID']))
        bldg_centroid = bldg_centroid.append(pt)
    return(bldg_centroid)

def ckdnearest(gdA, gdB, bcol):   
    '''
    find id for nearest spatial point
    '''
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)) )
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)) )
    btree = cKDTree(nB)
    dist, idx = btree.query(nA,k=1)
    df = pd.DataFrame.from_dict({'distance': dist.astype(float),
                                 'to' : gdB.iloc[idx][bcol].values})
    return df
        
def get_routing(row, api_id, api_code):
    start_lat, start_lon = row['lat_x'], row['lon_x']
    end_lat, end_lon = row['lat_y'], row['lon_y']
    
    url = (
        'https://route.api.here.com/routing/7.2/calculateroute.json?'
        'waypoint0={}%2C{}'
        '&waypoint1={}%2C{}'
        '&mode=shortest;pedestrian'
        '&app_id={}&app_code={}'
    ).format(start_lat, start_lon, end_lat, end_lon, api_id, api_code)
    obj = requests.get(url).json()
    return obj

def buffer_area(centroids, category, radius):
    centroids.crs = {'init' :'epsg:4326'}
    centroids = centroids.to_crs({'init' :'epsg:32637'})
    centroids_buffer = centroids.copy()
    centroids_buffer['geometry'] = centroids_buffer.buffer(distance=radius)
    centroids_buffer['category'] = category
    
    centroids_buffer = centroids_buffer.to_crs({'init' :'epsg:4326'})
    centroids_buffer = centroids_buffer.dissolve(by='category')
    return centroids_buffer

def to_point(bldg_vertex):
    bldg_vertex['lon'] = bldg_vertex.geometry.x
    bldg_vertex['lat'] = bldg_vertex.geometry.y

    bldg_vertex['list_coord'] = list(zip(bldg_vertex['lon'],bldg_vertex['lat']))
    bldg_vertex = bldg_vertex.reset_index(drop=True).reset_index()

    bldg_vertex['v_id'] = bldg_vertex['index'].astype(str) +'_' +  bldg_vertex['UID'].astype(str)
    bldg_vertex = bldg_vertex.drop(axis=1, columns=['index'])
    
    return(bldg_vertex)

def from_point(bldg_centroid, bldg_vertex):
    bldg_centroid = bldg_centroid.to_crs({'init' :'epsg:4326'})
    from_p = bldg_centroid.copy()
    from_p['lon'] = from_p.geometry.x
    from_p['lat'] = from_p.geometry.y
    from_p = from_p.rename(columns={'UID':'UID_from'})
    from_p['list_coord'] = list(zip(from_p['lon'],from_p['lat']))
    from_p['v_id'] = ckdnearest(from_p, bldg_vertex, 'v_id')['to']
    return(from_p)


def classify_bldg(all_bldg, t_bldg, category, radius, api_id, api_code):
    point_cat = ['SUBWAY_EXIT', 'BUS_STOP']
    if category not in point_cat:
        target_object = gpd.sjoin(all_bldg, t_bldg, how = 'inner')

        bldg_vertex_target = get_bldg_vertex(target_object, 'UID')
        tt = pd.DataFrame(bldg_vertex_target)
        tt = tt.groupby('UID', group_keys=False).apply(lambda d: d.iloc[:-1].assign(geometry2=d.geometry.values[1:]))
        tt = tt.reset_index()
        bldg_centroid = get_line_centroid(tt)
        
    else:
        bldg_centroid = t_bldg
        
    
    #building buffer
    bldg_centroid_buffer_dsvld = buffer_area(bldg_centroid, category, radius)
    
    #intersects buffer & polygons
    print('intersect buffer & polygons')
    intersected_bldg = gpd.sjoin(all_bldg, bldg_centroid_buffer_dsvld, how='left')
    intersected_bldg['restriction'] = intersected_bldg['index_right']
    intersected_bldg['restriction'] = intersected_bldg['restriction'].fillna('green')

    intersected_bldg = intersected_bldg.rename(columns={'index_right':'category', 'UID_left':'UID'})
    intersected_bldg.loc[intersected_bldg['restriction']!='green', 'restriction'] = 'red'
    
    other_bldg= intersected_bldg[intersected_bldg['restriction']=='red']
    
    #get vertices for other polygons
    other_bldg_vertex = get_bldg_vertex(other_bldg, 'UID')
        
    bldg_vertex = to_point(other_bldg_vertex) 
    print('for routing')
    #for routing
    from_p = from_point(bldg_centroid, bldg_vertex)
    
    print('send routing')
    data_for_routing = pd.merge(from_p, bldg_vertex[['v_id', 'UID','lon', 'lat']], on = 'v_id', how='inner')

    yellow_bldg = pd.DataFrame()
    red_bldg = pd.DataFrame()

    for idx, row in tqdm(data_for_routing.iterrows()):
        start_lat, start_lon = row['lat_x'], row['lon_x']
        end_lat, end_lon = row['lat_y'], row['lon_y']
    
        obj = get_routing(row, api_id, api_code)
        
        shortest_dst = obj['response']['route'][0]['leg'][0]['length']
        if shortest_dst < radius:
            tmp = pd.DataFrame([row['UID']], columns = ['UID'])
            red_bldg = red_bldg.append(tmp)
        else:
            tmp = pd.DataFrame([row['UID']], columns = ['UID'])
            yellow_bldg = yellow_bldg.append(tmp)
            
    ok = yellow_bldg[~yellow_bldg['UID'].isin(red_bldg['UID'])].drop_duplicates()
    neok = red_bldg['UID'].drop_duplicates()

    green= intersected_bldg[intersected_bldg['restriction']=='green']
    yellow = all_bldg[all_bldg['UID'].isin(ok['UID'])]
    red = all_bldg[~all_bldg['UID'].isin(yellow['UID'])]
    red = red[~red['UID'].isin(green['UID'])]

    #data for export
    red['restriction'] = 'red'
    red['category'] = category
    red = red[['UID', 'ADDRESS', 'category', 'restriction']]
    yellow['restriction'] = 'yellow'
    yellow['category'] = category
    yellow = yellow[['UID', 'ADDRESS', 'category', 'restriction']]
    green = green[['UID', 'ADDRESS', 'category', 'restriction']]
    green['category'] = category
    all_bldg = pd.concat([green, yellow, red])
    return all_bldg