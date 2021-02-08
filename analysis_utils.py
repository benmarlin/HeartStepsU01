import pandas as pd
import numpy as np
import math
import collections

import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import pacf, plot_pacf
from scipy.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import IterativeImputer, KNNImputer

from . import data_utils


def process_morning_survey(df):
    #Get Mood categories and create new columns
    df['Mood'] = pd.Categorical(df['Mood'])
    df['Mood Code'] = df['Mood'].cat.codes
    categories = dict(enumerate(df['Mood'].cat.categories))

    for key, value in categories.items():
        df[value] = df['Mood Code'].apply(lambda x: True if x == key else False)

    column_list = ['Busy', 'Committed', 'Rested']    
    for key, value in categories.items():
        column_list.append(value)
    df_selected = df[column_list]

    return df_selected

def process_daily_metrics(df):
    return df[['Fitbit Step Count', 'Fitbit Minutes Worn']]

def process_previous_fitbit_data(df):
    df['Previous Count'] = df['Fitbit Step Count'].shift()
    df['Previous Worn']  = df['Fitbit Minutes Worn'].shift()
    df.loc[df.groupby('Subject ID')['Previous Count'].head(1).index, 'Previous Count'] = 0
    df.loc[df.groupby('Subject ID')['Previous Worn'].head(1).index,  'Previous Worn']  = 0    
    return df

def convert_column_to_binary(df, column_name):
    #Set to 0 if value = 0, and 1 if value > 0
    df[column_name] = df[column_name].apply(lambda x: 1 if x > 0 else 0)
    return df

def plot_time_series(df, y_name, count):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    plt.figure(figsize=(12,2))
    previous = ''
    for row, value in enumerate(list(df.index)):
        subject_id = value[0]
        date = value[1]
        step = df.loc[subject_id, date][y_name]        
        if previous != subject_id:
            previous = subject_id
            steps = list(df.loc[subject_id,:][y_name])
            plt.plot(steps, label=subject_id)
            count -= 1
        if count == 0:
            break
    plt.legend(loc=1)
    plt.ylabel(y_name)
    plt.xlabel('number of days')
    plt.title(y_name)

def check_stationary(df, names):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    for name in names:
        adf_test = adfuller(df[name])
        adf_stat = adf_test[0]
        p_value  = adf_test[1]
        critical_values = adf_test[4]
        print('Check if %s is stationary?' % name)
        print('ADF statistic = %f' % adf_stat)
        print('p-value = %f' % p_value)
        print('critical values:')
        for key, value in critical_values.items():
            print('\t%s: %.3f' % (key, value))
        # Ho: time series is non-stationary
        if adf_stat < critical_values['5%']: # reject Ho
            print('%s time series is stationary' % name)             
        else: # failed to reject Ho
            print('%s time series is non-stationary' % name)
        print()

def impute(df, methods):
    #Impute missing data using method = IterativeImputer, KNNImputer:2 (using 2 neighbors)...
    #methods is a dictionary of key, value, where key contains the method 
    #and value contains all the columns where the method is applied.
    df = df.replace({True: 1, False: 0})   
    df_group = df.groupby(by='Subject ID')
    participant_ids = list(set(df.index.get_level_values(0)))
    frames = []
    #Impute for each participant
    for participant_id in participant_ids:
        df_i = df_group.get_group(participant_id)        
        df_i = df_i.reset_index()
        df_i = df_i.set_index(['Date'])
        for method, columns in methods.items():
            if method == 'IterativeImputer':
                imputer = IterativeImputer()
            elif method.find('KNNImputer:') >= 0:
                method_array = method.split(':')
                n_neighbors = int(method_array[1])
                imputer = KNNImputer(n_neighbors=n_neighbors, copy=False)
            else:
                #sklearn KNNImputer default uses 5 neighbors
                imputer = KNNImputer(copy=False)                
            X = df_i[columns].copy()
            X = X.reset_index()
            #sklearn imputer returns an array, so store and add back column names
            all_column_names = X.columns
            X_imputed = pd.DataFrame(imputer.fit_transform(X[columns]))           
            if X_imputed.shape[1] != X[columns].shape[1]:
                print('cannot impute participant %s' % participant_id)
                continue           
            X[columns] = X_imputed
            X.columns = all_column_names
            X = X.set_index(['Date'])           
            df_i[columns] = X
        df_i = df_i.reset_index().set_index(['Subject ID', 'Date'])        
        frames.append(df_i)
    df = pd.concat(frames)
    df = df.sort_index(level='Subject ID')
    return df
        
def get_pacf(df, names):   
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df_group = df.groupby(by='Subject ID')
    participant_ids = list(set(df.index.get_level_values(0)))
    
    #Compute the pacf values for each participant, and aggregate the pacf values
    for name in names:
        pacf_averages = collections.defaultdict(list)
        confidence_intervals = collections.defaultdict(list)
        
        for index, participant_id in enumerate(participant_ids):
            df = df_group.get_group(participant_id)
            current_threshold_lags = math.floor(df.shape[0]//2)-1
            max_lags = 15
            if (current_threshold_lags > max_lags) and (int(df[name].mean()) > 0):       
                pacf_values, confidence_interval = pacf(df[name], nlags=max_lags, alpha=0.05)
                lags = np.arange(max_lags+1)
                #Plot for sanity test
                #plot_pacf(df[name], lags=max_lags)
                #plt.title(participant_id + ' ' + name + ' Partial Autocorrelation')
                for i, lag in enumerate(lags):
                    pacf_averages[lag].append(pacf_values[i])
                    confidence_intervals[lag].append(confidence_interval[i])
                    
        lags = pacf_averages.keys()
        pacf_values = np.array(list(pacf_averages.values()))
        if len(pacf_values) > 0:
            plt.figure()
            pacf_values = pacf_values.mean(axis=1)
            confidence_interval = np.array(list(confidence_intervals.values())).mean(axis=1)            
            plt.fill_between(lags,
                             confidence_interval[:, 0] - pacf_values,
                             confidence_interval[:, 1] - pacf_values, alpha=0.25)
            plt.scatter(lags, pacf_values, color='C0', zorder=2)
            plt.bar(lags, pacf_values, width=0.1, color='black', zorder=1)
            plt.axhline(y=0)
            title = 'Average ' + name + ' Partial Autocorrelation'
            plt.title(title)
        else:
            print('found no pacf values for', name)
            
def get_pearsonr(df, name, lagged_name, max_lag):
    print('correlation between %s and lagged %s:' % (name, lagged_name))
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    for lag in range(1,max_lag):
        series1 = df[name].iloc[lag:]
        series2 = df[lagged_name].iloc[:-lag]
        results = pearsonr(series1, series2)
        corr = results[0]
        p_value = results[1]
        detail = ''
        if p_value < 0.05:
            detail = 'significant'
        print('lag=%d   corr=%.5f   p_value=%s \t%s' % (lag, corr, p_value, detail))
    print()

def compute_VAR(df, names, max_lag):
    df = df.dropna()
    df = df.reset_index()
    df = df.replace({True: 1, False: 0})
    df = df[names]
    model = VAR(df)
    results = model.fit(maxlags=max_lag)
    print(results.summary())

def get_correlations(df):
    df = df.replace({True: 1, False: 0})    
    correlations = df.corr()
    figsize = (9,8)
    if df.shape[1] > 12:
        figsize = (10,9)
    plt.figure(figsize=figsize)
    sn.heatmap(correlations, cmap=cm.seismic, annot=True, vmin=-1, vmax=1)

def update_name(old_name):
    return str(old_name).replace(' ', '_').lower()
    
def perform_gee(df, y_name, x_array, groups_name, fixed_effect='',
                family='Gaussian', cov_struct='Exchangeable', x_lim=None):
    #Perform GEE Linear Regression (additional options are fixed_effect, family and cov_struct)
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df = df.reset_index()

    if family == 'Gaussian':
        fam = sm.families.Gaussian()
    elif family == 'Poisson':
        fam = sm.families.Poisson()
    elif family == 'Binomial':
        fam = sm.families.Binomial()
    else:
        print('undefined family = %s, family is set to Gaussian' % family)
        family = 'Gaussian'
        fam = sm.families.Gaussian()

    if cov_struct == 'Independence':
        cov = sm.cov_struct.Independence() 
    elif cov_struct == 'Exchangeable':
        cov = sm.cov_struct.Exchangeable()
    else:
        print('undefined cov_struct = %s, cov_struct is set to Exchangeable' % cov_struct)
        cov_struct = 'Exchangeable'
        cov = sm.cov_struct.Exchangeable()

    df.columns = [update_name(x) for x in df.columns]
    y_name = update_name(y_name)
    groups = update_name(groups_name)

    equation = y_name + " ~ "
    for index, covariate in enumerate(x_array):
        covariate = update_name(covariate)
        if index == 0:
            equation += covariate
        else:
            equation += ' + ' + covariate

    figsize = (10,4)
    if fixed_effect != '':
        fixed_effect = update_name(fixed_effect)
        equation += " + C(" + fixed_effect + ")"
        figsize = (10,14)
    
    if family == 'Binomial':
        x_array = [update_name(x) for x in x_array]        
        model = sm.GEE(endog=df[y_name], exog=df[x_array], groups=df[groups], family=fam, cov_struct=cov)
    else:
        model = smf.gee(equation, data=df, groups=groups, family=fam, cov_struct=cov)

    results = model.fit()    
    (QIC, QICu) = results.qic(results.scale)
    print('%s =\n%s\n%s\n%s %s QIC = %.4f, QICu = %.4f\n\n\n' % (
           y_name, results.summary(), cov.summary(), family, cov_struct, QIC, QICu))
        
    df_coef = results.params.to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    title = y_name + ' using GEE ' + family + ' ' + cov_struct
    plt.title(title)

def perform_linear_regression(df, y_name, b_fixed_effect=False, x_lim=None):
    #Perform three linear regressions: OLS, GEE, Mixed Linear Model
    
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df = df.reset_index()

    plt.figure(figsize=(5,4))
    plt.hist(df['Fitbit Step Count']) 
    plt.xlabel('Fitbit Step Count')
    plt.title('Fitbit Step Count histogram')

    df['Fitbit Log Step Count'] = np.log(df['Fitbit Step Count'] + 1e-7)

    df.columns = [update_name(x) for x in df.columns]
    y_display = y_name
    y_name = update_name(y_name)
    
    equation  = " ~ busy + committed + rested + energetic"
    equation += " + fatigued + happy + relaxed + sad + stressed + tense"

    figsize = (10,4)
    if b_fixed_effect:
        #Add unconditional fixed effect on Subject ID
        equation += " + C(subject_id)"
        figsize = (10,14)
              
    model = y_name + equation
    
    mod  = smf.ols(model, data=df)
    res0 = mod.fit()
    print('%s =\n%s\n\n\n' % (y_display, res0.summary()))

    fam_display = 'Gaussian'
    cov_display = 'Exchangeable'
    cov = sm.cov_struct.Exchangeable()
    mod = smf.gee(model, "subject_id", data=df, cov_struct=cov)    
    res1 = mod.fit()    
    (QIC, QICu) = res1.qic(res1.scale)
    print('%s =\n%s\n%s\n%s %s QIC = %.4f, QICu = %.4f\n\n\n' % (
           y_name, res1.summary(), cov.summary(), fam_display, cov_display, QIC, QICu))

    mod = smf.mixedlm(model, df, groups=df['subject_id'])    
    res2 = mod.fit()
    print('%s =\n%s\n\n\n' % (y_display, res2.summary()))
    
    df_coef = res0.params.to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    plt.title(y_display + ' using OLS')
    
    df_coef = res1.params.to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    plt.title(y_display + ' using GEE Regression')

    df_coef = res2.params[:len(res2.params)-1].to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    plt.title(y_display + ' using Mixed Linear Model Regression')

def split_data(df, y_name, b_split_per_participant, split_percent=0.8, b_display=True):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    
    if b_split_per_participant:
        df = df.drop(columns=['Fitbit Minutes Worn'])       
        X = df.drop(columns=y_name)
        y = df[y_name]        
        X_train = X.groupby('Subject ID').apply(lambda x: x.head(round(split_percent * len(x))))
        y_train = y.groupby('Subject ID').apply(lambda x: x.head(round(split_percent * len(x))))
        X_test  = X.groupby('Subject ID').apply(lambda x: x.tail(len(x) - round(split_percent * len(x))))
        y_test  = y.groupby('Subject ID').apply(lambda x: x.tail(len(x) - round(split_percent * len(x))))
        X_train = X_train.values
        y_train = y_train.values
    else:
        df = df.reset_index()
        df = df.drop(columns=['Fitbit Minutes Worn', 'Subject ID', 'Date'])
        X = df.drop(columns=y_name).values
        y = df[y_name].values
        split = int(len(X) * split_percent)
        X_train = X[:split]
        y_train = y[:split]
        X_test  = X[split:]
        y_test  = y[split:]

    if b_display:
        print('Classification for ' + y_name)
        print('\ntrain split   = {}%'.format(int(split_percent*100)))
        print('X_train.shape =', X_train.shape)
        print('y_train.shape =', y_train.shape)
        print('X_test.shape  =', X_test.shape)
        print('y_test.shape  =', y_test.shape)
        
    return (X_train, y_train, X_test, y_test)

def perform_classification(df, y_name, b_split_per_participant):
    
    #Split the data and targets into training/testing sets
    split_data_tuple = split_data(df, y_name, b_split_per_participant=True)
    (X_train, y_train, X_test, y_test) = split_data_tuple

    model = LogisticRegression(solver='lbfgs', random_state=0)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('\nmodel =', model)
    print('\ntrain accuracy = %s' % accuracy_score(y_train, model.predict(X_train)))
    print('test  accuracy = %s\n' % accuracy_score(y_test,  model.predict(X_test)))

