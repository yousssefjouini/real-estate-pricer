from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from shapely.geometry import Point
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer, MinMaxScaler

import os
import pickle
import re
from typing import Union, Any
import warnings

warnings.simplefilter('ignore', FutureWarning)


class Data:
    @staticmethod
    def load_csv_files(path: str = None) -> dict[str, pd.DataFrame]:
        """This function loads the csv files Eleven provided us with."""
        if path is None:
            for root, folders, files in os.walk('.'):
                if 'mutations_d75_train_localized.csv' in files:
                    path = root
                    break
        data = {}
        for file in os.listdir(path):
            # We are only interested in csv files
            if not file.endswith('.csv'):
                continue
            data[file.split('.', 1)[0]] = pd.read_csv(f'{path}/{file}')
            # Dropping 'codservch' and 'refdoc' since they only contain NaNs along with 'Unnamed: 0.1' and 'Unnamed: 0'
            # that are useless
            data[file.split('.', 1)[0]] = data[file.split('.', 1)[0]].drop(columns=['codservch', 'refdoc',
                                                                                    'Unnamed: 0.1', 'Unnamed: 0'])
            # Filling the NaN values in 'valeurfonc' with 0
            data[file.split('.', 1)[0]] = data[file.split('.', 1)[0]].fillna(value={'valeurfonc': 0})

        return data

    @staticmethod
    def infer_dtypes(df: pd.DataFrame) -> dict[Union[str, int], str]:
        """This function infers the data types of each column"""
        df = df.copy()
        # Define the lower and upper thresholds for each data type
        t_lower = [-128, 0, -32_768, 0, -2_147_483_648, 0, -9_223_372_036_854_775_808, 0]
        t_upper = [127, 255, 32_767, 65_535, 2_147_483_647, 4_294_967_295, 9_223_372_036_854_775_807,
                   18_446_744_073_709_551_615]
        t_name = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        dtypes = dict()
        # The string columns remain the same
        for col in df.select_dtypes(include=[object]):
            try:
                df[col].astype('datetime64')
                dtypes[col] = 'datetime64'
            except ValueError:
                dtypes[col] = 'object'
        # The datetime columns remain the same
        for col in df.select_dtypes(include=['datetime64']):
            dtypes[col] = 'datetime64'
        # The boolean columns remain the same
        for col in df.select_dtypes(include=[bool]):
            dtypes[col] = bool
        # The numeric type depends on the values
        for col in df.select_dtypes(exclude=[object, 'datetime64', bool]):
            try:
                # Check if a column only contains integers
                if ((df[col] - df[col].astype(int)).sum() == 0) & (df[col].isna().sum() == 0):
                    # Get the lower bound index
                    l_possible = [bound for bound in t_lower if (bound - df[col].min()) <= 0]
                    l_index = t_lower.index(l_possible[0])
                    (df[col].max(), abs(df[col].min()))
                    # Get the upper bound index
                    u_possible = [bound for bound in t_upper if bound - df[col].max() >= 0]
                    u_index = t_upper.index(u_possible[0])
                    # Take the higher value between the lower and upper bounds index
                    dtypes[col] = t_name[max(l_index, u_index)]
                # For floating point numbers we always use float64 to ensure good precision
                else:
                    dtypes[col] = 'float64'
            except pd.errors.IntCastingNaNError:
                warnings.warn(f"{col} might be an integer if the wasn't for the NA values")
                dtypes[col] = 'float64'
        return dtypes

    @staticmethod
    def eval_columns(df: pd.DataFrame, columns: list[Union[str, int]] = None) -> pd.DataFrame:
        df = df.copy()
        if columns is None:
            columns = ['l_codinsee', 'l_section', 'l_idpar', 'l_idparmut', 'l_idlocmut']
        # Turning the columns from string to lists (applying it all at once does not work because of the NaNs)
        for col in columns:
            df.loc[:, col] = df.loc[:, col].apply(eval)
        return df

    @staticmethod
    def explode_df(df: pd.DataFrame, columns: list[Union[str, int]] = None) -> pd.DataFrame:
        """This function explodes (=expands) the DataFrame based on the provided columns."""
        df = df.copy()
        if columns is None:
            # We use 'l_codinsee' and 'l_section' as default values since the other have too many unique values and are
            # therefore not useful
            columns = ['l_codinsee', 'l_section']

        # Exploding the columns individually for the same reason as above
        for col in columns:
            df = df.explode(column=col)
        return df

    @classmethod
    def load_df(cls, path: str = None, explode: bool = True, drop_useless: bool = True) -> pd.DataFrame:
        """This is the main data loading method"""
        data = cls.load_csv_files(path=path)
        df = pd.concat([data[key] for key in data], ignore_index=True)
        df = df.astype(dtype=cls.infer_dtypes(df))
        if explode:
            df = cls.eval_columns(df)
            df = cls.explode_df(df)
        if drop_useless:
            # These columns are categorical and have mostly unique values
            useless_columns = ['idmutinvar', 'idopendata', 'l_idpar', 'l_idparmut', 'l_idlocmut']
            relevant_columns = set(df.columns).difference(useless_columns)
            df = df.loc[:, [col for col in df.columns if col in relevant_columns]]
        return df

    @classmethod
    def load_data_for_model(cls, path: str = None, keep_lists: bool = False) -> pd.DataFrame:
        """This method loads the data without some (for the modeling) useless columns. While the list columns might
        be useful, encoding them would put major emphasis on them, one therefore has the option to keep them.
        This method should be extended once feature engineering has been done."""
        df = cls.load_df(path=path)
        # We only want to keep the columns that don't have too many unique values and are in a format that lends itself
        # to a model and that doesn't duplicate information.
        # 'idmutation' has too many unique values, 'datemut' is in datetime format and the year and month have already
        # been extracted to distinct columns, the information of 'idnatmut' and 'codtypbien' is contained in other
        # columns already.
        relevant_columns = set(df.columns).difference(['idmutation', 'datemut', 'idnatmut', 'codtypbien'])
        if not keep_lists:
            relevant_columns = relevant_columns.difference(['l_codinsee', 'l_section'])
        df = df.loc[:, [col for col in df.columns if col in relevant_columns]]

        return df

    @classmethod
    def load_data_for_lgbm(cls, path: str = None) -> pd.DataFrame:
        df = cls.load_df(path=path, explode=False)
        df = df.loc[df['libtypbien'].isin(['UN APPARTEMENT', 'UNE MAISON']), :]
        df = df.loc[(df['valeurfonc'] > 1e4) & (df['valeurfonc'] <= 5 * 1e6), :]
        return df

    @classmethod
    def load_sample_df(cls, path: str = None) -> pd.DataFrame:
        file_name = 'mutations_d77_train_localized.csv'
        if path is None:
            for root, folders, files in os.walk('.'):
                if file_name in files:
                    path = root
                    break
        df = pd.read_csv(f'{path}/{file_name}')
        df = cls.eval_columns(df)
        df = cls.explode_df(df)
        useless_columns = ['idmutinvar', 'idopendata', 'l_idpar', 'l_idparmut', 'l_idlocmut', 'codservch', 'refdoc',
                           'Unnamed: 0.1', 'Unnamed: 0']
        relevant_columns = set(df.columns).difference(useless_columns)
        df = df.loc[:, [col for col in df.columns if col in relevant_columns]]
        return df

    @staticmethod
    def remove_outliers_q_based(df: pd.DataFrame, q_lower: float = 0.01, q_upper: float = 0.99) -> pd.DataFrame:
        """This method filters out records with 'valeurfonc' outliers based on the provided quantiles"""
        df = df.copy()
        t_lower = df['valeurfonc'].quantile(q_lower)
        t_upper = df['valeurfonc'].quantile(q_upper)
        return df.loc[(df['valeurfonc'] > t_lower) & (df['valeurfonc'] < t_upper), :]

    @staticmethod
    def calculate_distance(df: pd.DataFrame, latitude, longitude) -> pd.DataFrame:
        df = df.copy()
        df.loc[:, 'distance'] = ((df.loc[:, 'latitude'] - latitude) ** 2
                                 + (df.loc[:, 'longitude'] - longitude) ** 2) ** 0.5
        return df

    @staticmethod
    def load_shape_file(path: str = None,
                        file_name: str = 'communes-dile-de-france-au-01-janvier.shp') -> gpd.GeoDataFrame:
        """This method loads shape files in a GeoPandas DataFrame. By default the data of Île-de-France is loaded."""
        if path is None:
            for root, folders, files in os.walk('.'):
                if file_name in files:
                    path = root
        gdf = gpd.read_file(f'{path}/{file_name}')
        gdf['nomcom'] = gdf['nomcom'].str.encode('ISO-8859-1').str.decode('utf-8')
        return gdf

    @staticmethod
    def turn_mutations_df_into_geodf(df: pd.DataFrame, crs: str = 'epsg:4326') -> gpd.GeoDataFrame:
        df = df.copy()
        df['long_lat'] = gpd.GeoSeries(map(Point, zip(df['longitude'], df['latitude'])))
        return gpd.GeoDataFrame(df, geometry='long_lat', crs=crs)

    @staticmethod
    def calculate_price_by_district(df: pd.DataFrame,
                                    district: str = 'Épône',
                                    aggregation_func: str = 'mean') -> float:
        """This function aggregates the price for the desired district based on the provided aggregation_func."""
        # Handling the input
        if not {'nomcom', 'valeurfonc'}.intersection(df.columns):
            raise ValueError('The dataframe needs the have the following columns: "nomcom" and "valeurfonc"')

        if aggregation_func not in ['mean', 'median', 'min', 'max']:
            raise ValueError('Value computation does not exist')

        if district == 'All':
            return eval(f"df['valeurfonc'].{aggregation_func}()")

        agg = df.groupby('nomcom').agg({'valeurfonc': aggregation_func})
        return agg.loc[district, 'valeurfonc']


class Model:
    @staticmethod
    def create_pricing_model(df: pd.DataFrame = None, save: bool = True, saving_path: str = '.') -> Pipeline:
        """This method trains an ExtraTreesRegressor based on df."""
        if df is None:
            df = Data.load_data_for_model()
        target = 'valeurfonc'
        features = set(df.columns).difference([target])

        X = df.loc[:, [col for col in df.columns if col in features]]
        y = df.loc[:, target]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Some features need engineering before we can use them
        # The date features get a spline transformation
        # String columns and 'coddep' get a one-hot encoding
        # Numerical features (except 'coddep') are scaled
        categorical_preprocessor = Pipeline([('one_hot', OneHotEncoder(sparse=False))])
        numerical_preprocessor = Pipeline([('scaler', StandardScaler())])
        date_preprocessor = Pipeline([('spline', SplineTransformer())])

        spline_columns = ['anneemut', 'moismut']
        one_hot_columns = list(X.select_dtypes('object').columns) + ['libnatmut']
        coordinate_columns = ['latitude', 'longitude']
        scaling_columns = [col for col in X.select_dtypes(np.number).columns if col not in spline_columns
                           + one_hot_columns + coordinate_columns]

        preprocessor = ColumnTransformer([('cat_pre', categorical_preprocessor, one_hot_columns),
                                          ('num_pre', numerical_preprocessor, scaling_columns),
                                          ('spline_pre', date_preprocessor, spline_columns)],
                                         remainder='passthrough')
        pipe = Pipeline([('preprocessor', preprocessor), ('model', ExtraTreesRegressor(max_depth=35))])

        pipe.fit(X_train, y_train)

        if save:
            with open(f'{saving_path}/pricing_model.pkl', 'wb') as f:
                pickle.dump(pipe, f)
        return pipe

    @staticmethod
    def create_lgbm_pricing_model(df: pd.DataFrame = None, save: bool = True, saving_path: str = '.') -> Pipeline:
        """This method trains an LGBMRegressor based on df."""
        if df is None:
            df = Data.load_data_for_lgbm()

        numerical_features = ['sbati', 'latitude', 'longitude', 'anneemut']
        categorical_features = ['coddep', 'libnatmut', 'libtypbien']
        passthrough = ['nbapt1pp', 'nbapt2pp', 'nbapt3pp', 'nbapt4pp', 'nbapt5pp', 'nbmai1pp', 'nbmai2pp', 'nbmai3pp',
                       'nbmai4pp', 'nbmai5pp']
        considered_cols = numerical_features + categorical_features + passthrough

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = MinMaxScaler()

        y = df['valeurfonc']
        X = df[considered_cols]

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = ColumnTransformer([('cat', categorical_transformer, categorical_features),
                                          ('num', numerical_transformer, numerical_features)],
                                         remainder="passthrough")

        pipe = Pipeline([('preprocessor', preprocessor),
                         ('regressor', RandomForestRegressor()),
                         ])

        pipe.fit(train_x, train_y)

        if save:
            with open(f'{saving_path}/lgbm_pricing_model.pkl', 'wb') as f:
                pickle.dump(pipe, f)
        return pipe

    @staticmethod
    def load_pricing_model(path: str = None, file_name: str = 'pricing_model.pkl') -> Any:
        """This method loads the trained pricing model from the disk."""
        for root, folders, files in os.walk('.'):
            if 'pricing_model.pkl' in files:
                path = root
        with open(f'{path}/{file_name}', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_lgbm_pricing_model(path: str = '.', file_name: str = 'lgbm_pricing_model.pkl') -> Any:
        """This method loads the trained LGBM pricing model from the disk."""
        for root, folders, files in os.walk('.'):
            if 'lgbm_pricing_model.pkl' in files:
                path = root
        with open(f'{path}/{file_name}', 'rb') as f:
            return pickle.load(f)


class Scraper:
    google_maps_url = 'https://www.google.com/maps'

    def __init__(self, webdriver_path: str = None):
        if webdriver_path is None:
            for root, folders, files in os.walk('.'):
                if 'geckodriver' in files:
                    webdriver_path = f'{root}/geckodriver'
                    break
        webdriver_options = Options()
        webdriver_options.add_argument('--headless')
        self.driver = webdriver.Firefox(service=Service(executable_path=webdriver_path), options=webdriver_options)
        self.driver.get(self.google_maps_url)
        self.remove_cookie_window()

    def remove_cookie_window(self) -> None:
        """This method removes the cookie window."""
        reject_button_xpath = '/html/body/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[1]'
        reject_button = self.driver.find_element(By.XPATH, reject_button_xpath)
        reject_button.click()

    def get_coordinates(self) -> tuple[float, float]:
        """This method returns the x, y coordinates from the URL."""
        coordinates_found = False
        while not coordinates_found:
            coordinates_found = '@' in self.driver.current_url
        latitude, longitude = re.search(r'@([\d.-]+),([\d.-]+)', self.driver.current_url).groups()
        return eval(latitude), eval(longitude)

    def search_place_with_url(self, place: str) -> None:
        url = f"{self.google_maps_url}/place/{place.replace(' ', '+')}"
        self.driver.get(url)

    def type_search(self, place: str) -> None:
        search_box = self.driver.find_element(By.ID, 'searchboxinput')
        search_box.clear()
        search_box.send_keys(place)

    def get_suggestions(self) -> list[str]:
        class_name = 'DAdBuc'
        try:
            WebDriverWait(self.driver, 1).until(EC.presence_of_element_located((By.CLASS_NAME, class_name)))
        except:
            self.driver.find_element(By.ID, 'searchboxinput').click()
            self.get_suggestions()
        suggestions = self.driver.find_element(By.CLASS_NAME, class_name)
        suggestions_list = suggestions.text.split('\n')
        return suggestions_list

    def search(self) -> None:
        self.driver.find_element(By.ID, 'searchbox-searchbutton').click()


class StreamlitHelpers:
    @staticmethod
    def get_title_html() -> str:
        """
        <div id='maplegend' class='maplegend'
            style='position: absolute; z-index:9999; border:0px; background-color:rgba(255, 255, 255, 0.8);
             border-radius:6px; padding: 10px; font-size:25px; left: 0px; top: 0px;'>

        <div class='legend-title'>Visualization of price of mutation per communes & arrondissements </div>
        <div class='legend-scale'><font size="3">Per commune / Île-de-France / between 2014 and 2021</font></div>
        </div>
        """

    @staticmethod
    def get_legend_html(df: Union[pd.DataFrame, gpd.GeoDataFrame], colors: list[str] = None) -> str:
        if colors is None:
            colors = ['#00ae53', '#86dc76', '#daf8aa',
                      '#ffe6a4', '#ff9a61', '#ee0028']
        values = np.linspace(df['valeurfonc'].min(), df['valeurfonc'].max(), num=7)
        rounded_vals = np.around(values / 100_000) * 100_000

        legend_html = "<div id='maplegend' class='maplegend' style='position: absolute; z-index:9999; " \
                      "border:2px solid grey; background-color:rgba(255, 255, 255, 0.8); border-radius:6px; " \
                      "padding: 10px; font-size:14px; right: 20px; top: 20px;'>"
        legend_html += "<div class='legend-title'>Valeur fonciere</div>"
        legend_html += "<div class='legend-scale'>"
        legend_html += "<ul class='legend-labels'>"
        legend_html += "<li><span style='background:{0};opacity:0.7;'></span> < {1}k € </li>".format(
            colors[0], int(rounded_vals[1] / 1000))
        for i in range(1, len(values) - 2):
            legend_html += "<li><span style='background:{0};opacity:0.7;'></span>{1}k € - {2}k €</li>".format(
                colors[i],
                int(rounded_vals[i] / 1000),
                int(rounded_vals[i + 1] / 1000))
        legend_html += "<li><span style='background:{0};opacity:0.7;'></span> > {1}k € </li>".format(
            colors[i + 1], int(rounded_vals[i + 1] / 1000))
        legend_html += """</ul>
        </div>
        </div>
        <style type='text/css'>
          .maplegend .legend-title {
            text-align: left;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 90%;
            }
          .maplegend .legend-scale ul {
            margin: 0;
            margin-bottom: 5px;
            padding: 0;
            float: left;
            list-style: none;
            }
          .maplegend .legend-scale ul li {
            font-size: 80%;
            list-style: none;
            margin-left: 0;
            line-height: 18px;
            margin-bottom: 2px;
            }
          .maplegend ul.legend-labels li span {
            display: block;
            float: left;
            height: 16px;
            width: 30px;
            margin-right: 5px;
            margin-left: 0;
            border: 1px solid #999;
            }
          .maplegend .legend-source {
            font-size: 80%;
            color: #777;
            clear: both;
            }
          .maplegend a {
            color: #777;
            }
        </style>"""
        return legend_html

    @staticmethod
    def create_price_distirbution_graph(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Any:
        ...
