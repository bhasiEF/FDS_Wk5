# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:42:46 2023

@author: bhasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely as shapely
from scipy.ndimage import gaussian_filter1d

import gspread
from google.colab import auth
from google.auth import default

# a bit of code needed to import a google sheet
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

# grab shape within which to sample
url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_nation_20m.zip"
us = gpd.read_file(url).explode(index_parts = True)
us = us.reset_index().drop(columns=['level_0','level_1'])
mainland = us.iloc[35].geometry
bounds = np.reshape(mainland.exterior.bounds, (2,2))

sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1hzpZUUckx2xqi1UqqSJ7Kgf4SGVCiOUkucnxFrdejng/edit?usp=sharing")
worksheet = sh.worksheet("US_County_Level_Presidential_Results_12-16")  #the name of the sheet
rows = worksheet.get_all_values()

#We would be working with this dataframe now onwards
votes = pd.DataFrame.from_records(rows)

# Setting the first row as the column names
votes.columns = votes.iloc[0]
votes = votes.drop(votes.index[0])

# Load the json file with county coordinates
geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

# clean up the dataframes and merge them
votes_continental = votes[votes['state_abbr'].apply(lambda x: x !='AK' and x != 'HI')]
votes_continental['FIPS'] = votes_continental['FIPS'].apply(lambda x: int(x))
votes_continental = votes_continental[['FIPS', 'votes_dem_2012', 'votes_gop_2012', 'votes_dem_2016', 'votes_gop_2016']]

geoData = geoData.rename(columns={'id':'FIPS','NAME':'county','CENSUSAREA':'area'})
geoData['FIPS'] = geoData['FIPS'].apply(lambda x: int(x))
geoData = geoData[['county','FIPS','area','geometry']]

df = geoData.merge(votes_continental, how='inner', on='FIPS')
df['votes_dem_2012'] = df['votes_dem_2012'].astype(int)
df['votes_gop_2012'] = df['votes_gop_2012'].astype(int)
df['p_2012'] = df['votes_dem_2012']/(df['votes_dem_2012'] + df['votes_gop_2012'])

df['votes_dem_2016'] = df['votes_dem_2016'].astype(int)
df['votes_gop_2016'] = df['votes_gop_2016'].astype(int)
df['p_2016'] = df['votes_dem_2016']/(df['votes_dem_2016'] + df['votes_gop_2016'])

df = df.sort_values(by=['p_2012'])
df = df.reset_index()

df['cum_area'] = df.area.cumsum()
df['cum_area_prop'] = df['cum_area']/df['cum_area'].iloc[-1]

y = np.array(df['cum_area_prop'])
y = (np.concatenate(([0],y[:-1])) + y)/2
y = np.concatenate(([0],y,[1]))

x = np.array(df['p_2012'])
x = np.concatenate(([0],x,[1]))

x0 = np.linspace(0,1,1001)
y0 = np.interp(x0, x, y)
y_sm = gaussian_filter1d(y0, sigma=31)

dydx = np.diff(y_sm)/(x0[1]-x0[0])
x_ = (x0[:-1] + x0[1:])/2

plt.plot(x_, dydx)

def crash_land():
  on_land = False

  while(not on_land):
    point = shapely.Point(np.diff(bounds, axis=0)*np.random.rand(2) + bounds[0,:])
    on_land = point.within(mainland)

  return point

def closest(poly, point):
  if type(poly) is shapely.geometry.polygon.Polygon:
    return poly.exterior.distance(point)
  elif type(poly) is shapely.geometry.multipolygon.MultiPolygon:
    polys = list(poly.geoms)
    return min([x.exterior.distance(point) for x in polys])

def prop(loc):
  mask = df.geometry.apply(lambda x: loc.within(x))
  if mask.any():
    p = df[mask].iloc[0].p_2016
  else:
    p = df.iloc[df.geometry.apply(lambda x: closest(x, loc)).argmin()].p_2016

  return p

def prop_2012(loc):
  mask = df.geometry.apply(lambda x: loc.within(x))
  if mask.any():
    p = df[mask].iloc[0].p_2012
  else:
    p = df.iloc[df.geometry.apply(lambda x: closest(x, loc)).argmin()].p_2012

  return p

def poll(loc,n):
  return ['Hillary Clinton' if x else 'Donald Trump' for x in (np.random.rand(n)<prop(loc))]

def prior_2012(hz):
  prior = np.interp(hz,x_,dydx)
  prior /= np.sum(prior)
  return prior

def prior_loc_2012(loc, hz, sigma):
  p = prop_2012(loc)
  prior = np.exp(-0.5*((hz-p)/sigma)**2)
  prior /= np.sum(prior)
  return prior