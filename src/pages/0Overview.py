from __future__ import annotations

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from helpers import Data, StreamlitHelpers

# Loading the data
if 'df_idf' not in st.session_state:
    st.session_state['df_idf'] = Data.load_shape_file()

if 'df_mut' not in st.session_state:
    if 'df' in st.session_state:
        st.session_state['df_mut'] = st.session_state['df'].copy()
    else:
        st.session_state['df_mut'] = Data.load_df(explode=False)
    # st.session_state['df_mut'] = Data.load_df(explode=False)
    st.session_state['df_mut'] = Data.turn_mutations_df_into_geodf(st.session_state['df_mut'],
                                                                   crs=st.session_state['df_idf'].crs)

df_idf = st.session_state['df_idf']
df_mut = st.session_state['df_mut']


if 'df_idf_mut' not in st.session_state:
    st.session_state['df_idf_mut'] = gpd.sjoin(df_idf, df_mut, how='inner', predicate='intersects')

df_idf_mut = st.session_state['df_idf_mut']

columns = st.columns(3)
with columns[0]:
    st.selectbox('Select a district', options=['All'] + list(df_idf_mut['nomcom'].unique()), key='commune')
with columns[1]:
    st.selectbox('Select an aggregation method', options=['median', 'mean', 'max', 'min'], key='agg_func')
with columns[2]:
    price_in_district = Data.calculate_price_by_district(df=df_idf_mut,
                                                         district=st.session_state['commune'],
                                                         aggregation_func=st.session_state['agg_func'])
    st.metric(label='Price', value=f"{price_in_district:,.{2}f} €")

if st.session_state['commune'] == 'All':
    df_agg = df_idf_mut.groupby('nomcom').agg({'valeurfonc': st.session_state['agg_func']}).reset_index()
else:
    df_filtered = df_idf_mut.loc[df_idf_mut['nomcom'] == st.session_state['commune'], :]
    df_agg = df_filtered.groupby('nomcom').agg({'valeurfonc': st.session_state['agg_func']}).reset_index()

df_com = gpd.GeoDataFrame(pd.merge(df_agg, df_idf[['nomcom', 'geometry']], on='nomcom', how='left'))
df_dep = df_idf[['numdep', 'geometry']].dissolve(by='numdep', aggfunc='sum')

# Defining the center of the map
if st.session_state['commune'] == 'All':
    longitude = 2.3522219
    latitude = 48.856614
else:
    longitude = df_com.geometry.iloc[0].centroid.coords.xy[0][0]
    latitude = df_com.geometry.iloc[0].centroid.coords.xy[1][0]
# Create the empty map
m = folium.Map(location=[latitude, longitude],
               zoom_start=10, tiles='cartodbpositron')

# Add the legend to the map
colors = ['#00ae53', '#86dc76', '#daf8aa', '#ffe6a4', '#ff9a61', '#ee0028']

# Add the department outline to the map
folium.GeoJson(df_dep,
               style_function=lambda x: {
                   'color': 'black',
                   'weight': 2.5,
                   'fillOpacity': 0
               },
               name='Departement').add_to(m)

# Defining the colormap for the legend and the communes
values = np.linspace(df_com['valeurfonc'].min(), df_com['valeurfonc'].max(), num=7)
rounded_vals = np.around(values / 100_000) * 100_000
colormap_dept = cm.StepColormap(colors=colors,
                                vmin=min(df_com['valeurfonc']),
                                vmax=max(df_com['valeurfonc']),
                                index=rounded_vals)

style_function = lambda x: {'fillColor': colormap_dept(x['properties']['valeurfonc']),
                            'color': '',
                            'weight': 0.0001,
                            'fillOpacity': 0.6}

# Add the commune data to the map
folium.GeoJson(df_com,
               style_function=style_function,
               tooltip=folium.GeoJsonTooltip(fields=['nomcom', 'valeurfonc'],
                                             aliases=['Commune/Arrondisement', 'Valeur fonciere'],
                                             localize=False),
               name='Community').add_to(m)

m.get_root().html.add_child(folium.Element(StreamlitHelpers.get_title_html()))
m.get_root().html.add_child(folium.Element(StreamlitHelpers.get_legend_html(df_com, colors=colors)))

st_folium(m, width=724)
