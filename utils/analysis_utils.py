import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

def identify_outliers(df, col, threshold, method = 'sd'):
    
    if method == 'sd':
        z = (df[col] - np.mean(df[col]))/np.std(df[col])
        return np.abs(z) > threshold
    
    elif method == 'iqr':
        q3 = np.quantile(df[col], .75)
        q1 = np.quantile(df[col], .25)
        iqr = q3 - q1
        ub = q3 + threshold*iqr
        ub_gt = np.greater(df[col] - ub,0)
        lb = q1 - threshold*iqr
        lb_gt = np.less(df[col] - lb,0)
        return np.logical_or(ub_gt, lb_gt)

def check_missing(df, threshold=0.9):
    df_length = len(df)
    missing_dict = {col: np.round(sum(df[col].isna())/df_length,3) for col in df}
    
    missing_cols = []
    for k, v in missing_dict.items():
        if v > threshold:
            missing_cols.append(k)
    return missing_cols

def check_missing_dict(df, threshold=0.9):
    df_length = len(df)
    missing_dict = {col: np.round(sum(df[col].isna())/df_length,5) for col in df}
    
    missing_cols = {}
    for k, v in missing_dict.items():
        if v > threshold:
            missing_cols[k] = v
    return missing_cols

def find_redundent_columns(df):
    uniques = {col: df[col].unique() for col in df}
    redundent = []
    for k, v in uniques.items():
        if len(v) == 1:
            redundent.append(k)
    return redundent

class CustomImputer:
    def __init__(self):
        pass
    def fit(self, X_train):
        self.columns = X_train.columns[X_train.isna().any()]
        self.medians = {}
        for i in self.columns:
            self.medians[i] = X_train[i].median()
    def transform(self, X):
        return X.fillna(self.medians)

def get_ccp_path(dt_reg, X_train, y_train):
    c = CustomImputer()
    c.fit(X_train)
    X_train_imp = c.transform(X_train)
    path = dt_reg.cost_complexity_pruning_path(X_train_imp, y_train)
    return path

def optimise_dt(ccp_alpha, model, folds, X_train, y_train):
    
    ccp_average_error = []
    ccp_average_r2 = []
    for i in ccp_alpha:
        ccp_error = []
        ccp_r2 = []
        dt = model(ccp_alpha = i, random_state = 1)
        for train_idx, val_idx in folds.split(X_train, y_train):
            _X_train, _X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            _y_train, _y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            _X_train, _X_val = impute(_X_train, _X_val)
            dt.fit(_X_train, _y_train)
            pred = dt.predict(_X_val)
            ccp_error.append([np.mean(mean_squared_error(_y_val, pred))])
            ccp_r2.append([np.mean(r2_score(_y_val, pred))])
            
        ccp_average_error.append([i, np.mean(ccp_error), np.std(ccp_error)])
        ccp_average_r2.append([i, np.mean(ccp_r2), np.std(ccp_r2)])
    return ccp_average_error, ccp_average_r2


def impute(X_train, X_val):
    c = CustomImputer()
    c.fit(X_train)
    return c.transform(X_train), c.transform(X_val)

def normalise(X_train, X_val):
    s = StandardScaler()
    return s.fit_transform(X_train), s.transform(X_val)


def optimal_regularisation(params, model, folds, X_train, y_train):
    mse_average = []
    r2_average = []

    for i in params:
        m = model(alpha = i, random_state = 1)
        mse = []
        r2 = []
        for train_idx, val_idx in folds.split(X_train, y_train):
            _X_train, _X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            _y_train, _y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            _X_train, _X_val = impute(_X_train, _X_val)
            _X_train, _X_val = normalise(_X_train, _X_val)
            m.fit(_X_train, _y_train)
            pred = m.predict(_X_val)
            mse.append([np.mean(mean_squared_error(_y_val, pred))])
            r2.append([np.mean(r2_score(_y_val, pred))])
        mse_average.append([i, np.mean(mse), np.std(mse)])
        r2_average.append([i, np.mean(r2), np.std(r2)])
    return mse_average, r2_average


def errors_to_df(r2, mse):
    r2_summary_df = pd.DataFrame(r2, columns = ['param', 'mean_r2', 'std_r2'])
    mse_summary_df = pd.DataFrame(mse, columns = ['param', 'mean_mse', 'std_mse'])
    return r2_summary_df, mse_summary_df


def get_optimal_model_params(summary_df, metric = 'mse', optimisation= '_min'):
    if optimisation == '_min':
        filt_min = min(summary_df[f'mean_{metric}'])
        return summary_df[summary_df[f'mean_{metric}'] == filt_min]
    elif optimisation == '_max':
        filt_max = max(summary_df[f'mean_{metric}'])
        return summary_df[summary_df[f'mean_{metric}'] == filt_max]
