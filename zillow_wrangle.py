import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from env import get_db_url
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#Function to import the SQL database into jupyter notebook
def zillow_data():
    filename = 'zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        zillow_df = pd.read_sql('''SELECT 
    bedroomcnt,
    bathroomcnt,
    calculatedfinishedsquarefeet,
    taxvaluedollarcnt,
    yearbuilt,
    latitude,
    longitude,
    taxamount,
    lotsizesquarefeet,
    rawcensustractandblock,
    roomcnt,
    poolcnt,
    regionidcounty,
    garagecarcnt,
    pred.logerror as logerror17,
    pred16.logerror as logerror16,
    pred.transactiondate as transaction17,
    pred16.transactiondate as transaction16
FROM
    properties_2017 AS prop
  
        LEFT JOIN
    predictions_2017 AS pred USING (parcelid)
    left join
    predictions_2016 as pred16 using (parcelid)
     left join
    propertylandusetype AS plut on prop.propertylandusetypeid = plut.propertylandusetypeid
    
WHERE
    (pred16.transactiondate > '2016-08-31' or pred.transactiondate LIKE '2017%%')
    
    
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
        AND (prop.unitcnt = 1
        OR plut.propertylandusetypeid = 261)''', get_db_url('zillow'))
        # zillow_df.to_csv(filename)

        return zillow_df



def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)



def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['parcelid', 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)


def summarize(df):
    '''
    This function will take in a single argument (a pandas dataframe) and 
    output to console various statistices on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('----------------------')
    print('Dataframe head')
    print(df.head(3))
    print('----------------------')
    print('Dataframe Info ')
    print(df.info())
    print('----------------------')
    print('Dataframe Description')
    print(df.describe())
    print('----------------------')
    num_cols = [col for col in df.columns if df[col].dtypes != 'object']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('----------------------')
    print('Dataframe value counts ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            # define bins for continuous columns and don't sort them
            print(df[col].value_counts(bins=10, sort=False))
    print('----------------------')
    print('nulls in df by column')
    print(nulls_by_col(df))
    print('----------------------')
    print('null in df by row')
    print(nulls_by_row(df))
    print('----------------------')


def keep_col(df):
    #Making pools boolean despite the amount of nulls, if a house has a pool it will be listed in the features becuase it is a high ticket item
    df['poolcnt'] = np.where((df['poolcnt'] == 1.0) , True , False)
    
    # Assigning the value of the car garage to the dataset if its above 1 and making the nulls to 0 doing this because garages are important enough to list and there are as many nulls with garage sq
    df['garagecarcnt'] = np.where((df['garagecarcnt'] >= 1.0) , df['garagecarcnt'] , 0)  

    #Feature engineering new variables to combat multicolinearty and test to see if new features help the model

    df['poolcnt'] = df.poolcnt.map({True:1, False:0})
    return df 



def trim_zillow(df):
  
    # If the lot size is smaller than the finished square feet, it's probably bad data or not a single family home
    df = df[~(df.lotsizesquarefeet < df.calculatedfinishedsquarefeet)]
    # If the finished square feet is less than 500 it is likeley an apartment, or bad data
    df = df[~(df.calculatedfinishedsquarefeet < 500)]
    # Do not want to include studios or lofts into the data set
    df = df[~(df.bathroomcnt < 1)]
    # If there are no bathrooms, it is not a home
    df = df[~(df.bedroomcnt < 1)]
    # remove duplicate parcelids
    # df = df.sort_values('transaction17').drop_duplicates('parcelid',keep='last')
    # df = df.sort_values('transaction16').drop_duplicates('parcelid',keep='last')
    return df

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df


def series_upper_outliers(s, k=1.5):
  
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))


def split_scale(df):
    # Copy a new dataframe to perform feature engineering
    scaled_df = df.copy()
    scaler = RobustScaler()
    # Split the scaled data into train, validate, test
    train, validate, test = zillow_wrangle.split_data(scaled_df)
    # Columns to scale
    cols = ['calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'rawcensustractandblock', 'garagecarcnt', 'house_age', 'polar_combo', 'weird_ratio', 'cen_ratio', 'raw_cen_bin']
    # Fit numerical features to scaler
    scaler.fit(train[cols])
    # Set the features to transformed value
    train[cols] = scaler.transform(train[cols])
    validate[cols] = scaler.transform(validate[cols])
    test[cols] = scaler.transform(test[cols])
    return train, validate, test


def df_upper_outliers(df, k, cols):
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        upper_bound = q3 + k * iqr
    return df.apply(lambda x: max([x - upper_bound, 0]))

def df_lower_outliers(df, k, cols):
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
    return df.apply(lambda x: min([x - lower_bound, 0]))    



def remove_outliers(df, k, cols):
    # df = df_upper_outliers()
    # df = df_lower_outliers()

     # return dataframe without outliers
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr


        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
    return df    
    
def feature_engineering(df):
    #Feature engineering new variables to combat multicolinearty and test to see if new features help the model

    # df['pool_encoded'] = df.poolcnt.map({True:1, False:0})
    # Making lat and long usuable by dividing by 1 000 000
    df['lat'] = df.latitude / 1_000_000
    df['long'] = df.longitude / 1_000_000
    
    # I will drop bed and bath in exchange for the ratio which accomplishes the same task without multicollinearity
    df['bed_bath_ratio'] = df['bedroomcnt'] / df['bathroomcnt']
  
    #dropping year built and turning it into house age which will then be scaled
    df['house_age'] = 2017 - df['yearbuilt']
    return df
    
def latlong_to_cart(df):
    df['coslat'] = df.lat.apply(math.cos)
    df['coslong'] = df.long.apply(math.cos)
    df['sinlong'] = df.long.apply(math.sin)
    df['x'] = df.coslong * df.coslat
    df['y'] = df.coslat * df.sinlong
    df['z'] = df.latitude.apply(math.sin) 
    return df

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def add_upper_outlier_columns(df, k=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('float64'):
        df[col + '_outliers_upper'] = df_upper_outliers(df[col], k)
    return df


def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.70):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) #1, or ‘columns’ : Drop columns which contain missing value
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) #0, or ‘index’ : Drop rows which contain missing values.
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

def combine_transactions(df):
    df['logerror'] = 0
    df.loc[~df['logerror17'].isnull(), 'logerror'] = df.loc[~df['logerror17'].isnull(), 'logerror17']
    df.loc[~df['logerror16'].isnull(), 'logerror'] = df.loc[~df['logerror16'].isnull(), 'logerror16']
    df['transactiondate'] = None
    df.loc[~df['transaction17'].isnull(), 'transactiondate'] = df.loc[~df['transaction17'].isnull(), 'transaction17']
    df.loc[~df['transaction16'].isnull(), 'transactiondate'] = df.loc[~df['transaction16'].isnull(), 'transaction16']   

    df = df.drop(columns= ['transaction16','transaction17', 'logerror16', 'logerror17'])

    return df

def add_date_features(df):
    
    df['transactiondate'] = pd.to_datetime(df['transactiondate'], format= '%Y/%m/%d')

    df["transaction_month"] = df["transactiondate"].dt.month
    
    df["transaction_quarter"] = df["transactiondate"].dt.quarter

    df.drop(["transactiondate"], inplace=True, axis=1)
    
    return df

def county_name(county):
    if county == 6037:
        return 'Los Angeles'
    elif county == 6059:
        return 'Orange'
    elif county == 6111:
        return 'Ventura'

def split_data(df):

    # split the data
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                       random_state=123)
    return train, validate, test   




# This function is used to plot categorical and continuous variables to help the exploration process and see how each variable compares to each other
def plot_categorical_and_continuous_vars(df, cat_cols, cont_cols):
    #designated coloumns to explore in my notebook. this for loops cycles through both continuous and categorical variables
    for cont in cont_cols:
        for cat in cat_cols:
            #setting the chart size and titles
            fig = plt.figure(figsize= (20, 10))
            fig.suptitle(f'{cont} vs {cat}')
            
            # using a violin plot for the first graph
            plt.subplot(131)
            sns.violinplot(data=df, x = cat, y = cont)
           
            # makes a histogram for the third graph
            plt.subplot(1, 3, 3)
            sns.histplot(data = df, x = cont, bins = 50, hue = cat)
            
            #makes a bar chart for the second graph
            plt.subplot(1, 3, 2)
            sns.barplot(data = df, x = cat, y = cont)


def wrangle_zillow():

    df = zillow_data()

    df = keep_col(df)

    df = feature_engineering(df)

    df = trim_zillow(df)



    # df.to_csv("zillow.csv", index=False)

    return df
