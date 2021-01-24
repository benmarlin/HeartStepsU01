import pandas as pd
import numpy as np

import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

def get_pacf(df, names):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    for i in range(0, len(names), 2):
        fig, ax = plt.subplots(1,2, figsize=(12,3))
        name = names[i]
        title = name + ' Partial Autocorrelation'
        plot_pacf(df[name], ax=ax[0], title=title)
        if i+1 < len(names):
            name = names[i+1]
            title = name + ' Partial Autocorrelation'
            plot_pacf(df[name], ax=ax[1], title=title)

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
    plt.figure(figsize=(9,8))
    sn.heatmap(correlations, cmap=cm.seismic, annot=True, vmin=-1, vmax=1)

def perform_linear_regression(df):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df = df.reset_index()

    plt.figure(figsize=(5,4))
    plt.hist(df['Fitbit Step Count']) 
    plt.xlabel('Fitbit Step Count')
    plt.title('Fitbit Step Count histogram')

    df['Fitbit Log Step Count'] = np.log(df['Fitbit Step Count'] + 1e-7)

    y_names = {'Fitbit Step Count': 'step_count',
               'Fitbit Log Step Count': 'log_step_count',
               'Fitbit Minutes Worn': 'minutes_worn'}

    df = df.rename(columns=y_names)
    df = df.rename(columns={'Subject ID': 'subject_id' })
    
    equation  = " ~ Busy + Committed + Rested + Energetic"
    equation += " + Fatigued + Happy + Relaxed + Sad + Stressed + Tense"

    for y_display, y_name in y_names.items():        
        model = y_name + equation
        
        mod  = smf.ols(model, data=df)
        res0 = mod.fit()
        print('%s =\n%s\n\n\n' % (y_display, res0.summary()))

        ex = sm.cov_struct.Exchangeable()
        mod = smf.gee(model, "subject_id", data=df, cov_struct=ex)    
        res1 = mod.fit()
        print('%s =\n%s\n\n\n' % (y_display, res1.summary()))

        mod = smf.mixedlm(model, df, groups="subject_id")    
        res2 = mod.fit()
        print('%s =\n%s\n\n\n' % (y_display, res2.summary()))
        
        df_coef = res0.params.to_frame().rename(columns={0: 'coef'})
        ax = df_coef.plot.barh(figsize=(14, 7))
        ax.axvline(0, color='black', lw=1)
        plt.title(y_display + ' using OLS')
        
        df_coef = res1.params.to_frame().rename(columns={0: 'coef'})
        ax = df_coef.plot.barh(figsize=(14, 7))
        ax.axvline(0, color='black', lw=1)
        plt.title(y_display + ' using GEE Regression')

        df_coef = res2.params.to_frame().rename(columns={0: 'coef'})
        ax = df_coef.plot.barh(figsize=(14, 7))
        ax.axvline(0, color='black', lw=1)
        plt.title(y_display + ' using Mixed Linear Model Regression')

def perform_classification(df):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df = df.reset_index()
    df = df.drop(columns=['Fitbit Minutes Worn', 'Subject ID', 'Date'])      

    #Turn Fitbit Step Count per day into a binary variable steps = 0 and steps > 0
    column_name = 'Fitbit Step Count'
    df[column_name] = df[column_name].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop(columns=column_name).values
    y = df[column_name].values

    # Split the data and targets into training/testing sets
    split_percent = 0.8
    split = int(len(X) * split_percent)    
    X_train = X[:split]
    X_test  = X[split:]
    y_train = y[:split]
    y_test  = y[split:]

    print('Classification for ' + column_name)
    print('\ntrain split   = {}%'.format(int(split_percent*100)))
    print('X_train.shape =', X_train.shape)
    print('y_train.shape =', y_train.shape)
    print('X_test.shape  =', X_test.shape)
    print('y_test.shape  =', y_test.shape)    

    model = LogisticRegression(solver='lbfgs', random_state=0)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('\nmodel =', model)
    print('\ntrain accuracy =', accuracy_score(y_train, model.predict(X_train)))
    print('test  accuracy =', accuracy_score(y_test,  model.predict(X_test)))

