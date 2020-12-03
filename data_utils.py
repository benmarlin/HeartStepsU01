import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
import zipfile
from IPython.display import display, Markdown, Latex
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

study_prefix = "U01 Data/usc-study.u01."

def get_user_id_from_filename(f):
    #Get user id from from file name
    return(f.split(".")[2])

def get_file_names_from_zip(z, file_type=None, prefix=study_prefix):  
    #Extact file list
    file_list = list(z.filelist)
    if(filter is None):
        filtered = [f.filename for f in file_list if (prefix in f.filename)]
    else:
        filtered = [f.filename for f in file_list if (file_type in f.filename and prefix in f.filename)]
    return(filtered)

def get_df_from_zip(file_type,zip_file):
    
    #Open data zip file
    z     = zipfile.ZipFile(zip_file)
    
    #Get list of files of specified type
    file_list = get_file_names_from_zip(z, file_type=file_type)
    
    #Open file inside zip
    dfs=[]
    for file_name in file_list:  
        sid = get_user_id_from_filename(file_name)
        f = z.open(file_name)
        df  = pd.read_csv(f)
        df["Subject ID"] = sid
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.set_index(["Subject ID","Date"])    
    #Read file with pandas
    return(df)

def fix_df_column_types(df, categorical_fields):
    #Set categorical fields to string type to prevent
    #interpretation as numeric
    for field in categorical_fields:
        df=df.astype({field: 'str'})
    return(df)

def load_data(data_dict, zip_file):
    df = get_df_from_zip(data_dict["file_name"],zip_file)
    df = fix_df_column_types(df,data_dict["categorical_fields"])
    return(df)
    
def get_subject_ids(df):
  sids = list(df.index.levels[0])
  return list(sids)

def get_variables(df): 
  cols = [c for c in list(df.columns) if df.dtypes[c] == np.dtype('float64')]
  return(cols)
  
def plot_summary_histograms(df, dd, cols=3, fields=[]):  
    num_fields = len(dd["categorical_fields"])+len(dd["numerical_fields"])
    rows = int(np.ceil(num_fields/3))
    fig, axes = plt.subplots(rows, cols, figsize=(4*3,rows*3))
    i=0
    for field in list(df.keys()):
        if(field in fields or len(fields)==0):
            if field in dd["categorical_fields"]:
                this_ax = axes[i//cols,i%cols]
                df[field].value_counts().plot(kind="bar",ax=this_ax)
                this_ax.grid(axis='y')
                this_ax.set_title(field)
                i=i+1
            if field in dd["numerical_fields"]: 
                this_ax = axes[i//cols,i%cols]
                df[field].hist(figure=fig,ax=this_ax)
                this_ax.grid(True)
                this_ax.set_title(field)
                i=i+1
    plt.tight_layout()
    plt.show()

def plot_indifivual_time_series(df,variable,subject_id):
  this_df = df.xs(subject_id, level=0, axis=0, drop_level=True)
  this_df[variable].plot(kind='bar', grid=True, figsize=(12,4) )
  plt.title("Subject %s: %s"%(subject_id,variable))
  plt.show()

def show_individual_time_series_visualizer(df):
  sids=get_subject_ids(df)
  vars=get_variables(df)
  interact(plot_indifivual_time_series, df=fixed(df), subject_id=sids,variable = vars);

data_dicts = {}
data_dicts["daily-metrics"]={}
data_dicts["daily-metrics"]["file_name"] = "daily-metrics"
data_dicts["daily-metrics"]["name"] = "Daily Metrics Data"
data_dicts["daily-metrics"]["categorical_fields"] = ["App Used","Fitbit Updated Completely","Fitbit Worn"]
data_dicts["daily-metrics"]["numerical_fields"] = ['App Page Views','Fitbit Minutes Worn','Fitbit Step Count','Fitbit Update Count','Number of Location Records', 'Messages Sent','Messages Received', 'Messages Opened', 'Messages Engaged']

