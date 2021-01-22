import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import statsmodels.api as sm
import statsmodels.formula.api as smf

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
    return  df[['Fitbit Step Count', 'Fitbit Minutes Worn']]

def get_correlations(df):
    df = df.replace({True: 1, False: 0})    
    correlations = df.corr()
    plt.figure(figsize=(9,8))
    sn.heatmap(correlations, cmap=cm.seismic, annot=True, vmin=-1, vmax=1)

def perform_linear_regression(df):

    df = df.dropna()
    df = df.reset_index()

    y_names = {'Fitbit Step Count': 'step_count',
               'Fitbit Minutes Worn': 'minutes_worn'}

    df = df.rename(columns=y_names)
    df = df.rename(columns={'Subject ID': 'subject_id' })
    
    equation  = " ~ Busy + Committed + Rested + Energetic"
    equation += " + Fatigued + Happy + Relaxed + Sad + Stressed + Tense"    

    for y_display, y_name in y_names.items():        
        model = y_name + equation

        ind = sm.cov_struct.Exchangeable()
        mod = smf.gee(model, "subject_id", data=df, cov_struct=ind)    
        res1 = mod.fit()
        print('%s =\n%s\n\n\n' % (y_display, res1.summary()))

        mod = smf.mixedlm(model, df, groups="subject_id")    
        res2 = mod.fit()
        print('%s =\n%s\n\n\n' % (y_display, res2.summary()))

        df_coef = res1.params.to_frame().rename(columns={0: 'coef'})
        ax = df_coef.plot.barh(figsize=(14, 7))
        ax.axvline(0, color='black', lw=1)
        plt.title(y_display)

        df_coef = res2.params.to_frame().rename(columns={0: 'coef'})
        ax = df_coef.plot.barh(figsize=(14, 7))
        ax.axvline(0, color='black', lw=1)
        plt.title(y_display)
