import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from . import data_utils


def process_morning_survey(df):
    #Get Mood categories and create new columns
    df['Mood'] = pd.Categorical(df['Mood'])
    df['Mood Code'] = df['Mood'].cat.codes
    categories = dict(enumerate(df['Mood'].cat.categories))

    df_selected = df['Mood']
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
    sn.heatmap(correlations, cmap=cm.Blues, annot=True)




