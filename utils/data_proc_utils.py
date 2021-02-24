
import os
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
from utils.analysis_utils import check_missing

def get_files(folder = os.path.join(os.getcwd(), 'data')):
    container = {}
    for i in os.listdir(folder):
        container[i] = (pd.read_csv(os.path.join(folder,i)))
    return container


def proc_calendar(df):
    
    out = df.copy()
    set_price = lambda x: float(str(x).replace('$','').replace(',',''))
    
    out['date'] = pd.to_datetime(out['date'], format = '%Y-%m-%d')
    out['price'] = out['price'].apply(set_price)
    out['available'] = out['available'].map({'f' : False, 't' : True})
    out['month'] = out['date'].dt.month
    out['day_of_week'] = out['date'].apply(lambda x: calendar.day_name[x.weekday()])
    
    return out[~out['price'].isna()]

def listing_proc(df, columns_to_drop = None, columns_to_dummy = None, get_dummies_kwargs = {}):
    
    #date where data was compiled 
    COMPILE_DATE = datetime(2020, 12, 16) 
    
    def review_score_bucket(col):
        if col >= 80:
            return 'good'
        elif col >= 60:
            return 'satisfactory'
        elif col < 60:
            return 'bad'
        else:
            return np.nan

    def get_dummies_drop_original(columns_to_dummy):
        for i in columns_to_dummy:
            pd.concat(out.drop(i, axis = 1), pd.get_dummies(out[i], prefix = i, **get_dummies_kwargs))
    
    remove_pc = lambda x: float(str(x).replace('%', ''))
    to_numeric = lambda x: pd.to_numeric(x, errors = 'coerce')
    get_len = lambda x: len(str(x))
    cleanse_price = lambda x: float(str(x).replace('$', '').replace(',', ''))

    #get rid of columns with more than 90% of values missing
    out = df.copy()
    missing_cols = check_missing(out)
    out = out.drop(missing_cols, axis=1)
    
    # create derived fields
    out['verification_types'] = out['host_verifications'].apply(get_len)
    out['number_of_amenities'] = out['amenities'].apply(lambda x: len(x.split(',')))
    out['price'] = out['price'].apply(cleanse_price)
    out['cleaning_fee'] = out['cleaning_fee'].apply(cleanse_price).fillna(0)
    out['security_deposit'] = out['security_deposit'].apply(cleanse_price).fillna(0)
    out['extra_people'] = out['extra_people'].apply(cleanse_price).fillna(0)
    out['host_response_rate'] = out['host_response_rate'].apply(remove_pc)
    out['host_acceptance_rate'] = out['host_acceptance_rate'].apply(remove_pc)
    out['host_listings_count'] = out['host_listings_count'].apply(to_numeric)
    out['host_total_listings_count'] = out['host_total_listings_count'].apply(to_numeric)
    out['accommodates'] = out['accommodates'].apply(to_numeric)
    out['host_tenure_days'] =  (COMPILE_DATE - pd.to_datetime(out['host_since'], format = '%Y-%m-%d')).dt.days
    out['name_length'] = out['name'].apply(get_len)
    out['description_length'] = out['description'].apply(get_len)
    out['neighbourhood_overview_length'] = out['neighborhood_overview'].apply(get_len)
    out['host_about_length'] = out['host_about'].apply(get_len)
    out['review_bucket'] = out['review_scores_rating'].apply(lambda x: review_score_bucket(x))
    out['cleaning_fee_flag'] = out['cleaning_fee'].apply(lambda x: 1 if x > 0 else 0)
    out['security_deposit_flag'] = out['security_deposit'].apply(lambda x: 1 if x > 0 else 0)
    out['instant_bookable'] = out['instant_bookable'].map({'f': 0, 't': 1}) 
    
    #get_dummies_drop_orginal(columns_to_dummy)
    
    #drop out additional columns
    if columns_to_drop:
        out = out.drop(columns_to_drop, axis = 1)
    
    return out


def listing_proc_dummies(data, columns_to_dummy, dummy_na=False):
    df = data.copy()
    
    for col in columns_to_dummy:
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col] ,dummy_na=dummy_na, prefix = col, prefix_sep = '_', drop_first=True)], axis=1)
    
    return df
    
def listing_proc_drop(data, columns_to_drop):
    df = data.copy()

    for col in columns_to_drop:
        df = df.drop([col], axis = 1)
    
    return df


def listing_proc_dummies(data, columns_to_dummy, dummy_na=False):
    df = data.copy()
    
    for col in columns_to_dummy:
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col] ,dummy_na=dummy_na, prefix = col, prefix_sep = '_', drop_first=True)], axis=1)
    
    return df
    
def listing_proc_dummy_drop(data, columns_to_drop):
    df = data.copy()

    for col in columns_to_drop:
        df = df.drop([col], axis = 1)
    
    return df