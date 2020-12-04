import numpy as np
import pandas as pd
import os
import zipfile

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

def get_data_info(dictionary_file,file_filter,long_name):
    dd=pd.read_csv(dictionary_file)
    dd=dd.set_index("ElementName")
    di={}
    di["file_name"]=file_filter
    di["name"]=long_name
    di["dictionary"]=dd
    return(di)


data_dicts = {}
data_dicts["daily-metrics"]={}
data_dicts["daily-metrics"]["file_name"] = "daily-metrics"
data_dicts["daily-metrics"]["name"] = "Daily Metrics Data"
data_dicts["daily-metrics"]["categorical_fields"] = ["App Used","Fitbit Updated Completely","Fitbit Worn"]
data_dicts["daily-metrics"]["numerical_fields"] = ['App Page Views','Fitbit Minutes Worn','Fitbit Step Count','Fitbit Update Count','Number of Location Records', 'Messages Sent','Messages Received', 'Messages Opened', 'Messages Engaged']

