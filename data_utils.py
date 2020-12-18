import numpy as np
import pandas as pd
import os
import zipfile

study_prefix = "U01"

def get_user_id_from_filename(f):
    #Get user id from from file name
    return(f.split(".")[3])

def get_file_names_from_zip(z, file_type=None, prefix=study_prefix):  
    #Extact file list
    file_list = list(z.filelist)
    if(filter is None):
        filtered = [f.filename for f in file_list if (prefix in f.filename) and (".csv" in f.filename)]
    else:
        filtered = [f.filename for f in file_list if (file_type in f.filename and prefix in f.filename)]
    return(filtered)

def get_data_catalog(catalog_file, data_file, data_dir, dict_dir):
  dc=pd.read_csv(catalog_file)
  dc=dc.set_index("Data Product Name")
  dc.data_file=data_dir+data_file #add data zip file field
  dc.data_dir=data_dir #add data zip file field
  dc.dict_dir=dict_dir #add data distionary directory field
  return(dc)

def get_data_dictionary(data_catalog, data_product_name):
    dictionary_file = data_catalog.dict_dir + data_catalog.loc[data_product_name]["Data Dictionary File Name"]
    dd=pd.read_csv(dictionary_file)
    dd=dd.set_index("ElementName")
    dd.data_file_name = data_catalog.loc[data_product_name]["Data File Name"] #add data file name pattern field
    dd.name = data_product_name #add data product name field
    dd.index_fields = data_catalog.loc[data_product_name]["Index Fields"] #add index fields
    dd.description = data_catalog.loc[data_product_name]["Data Product Description"]
    return(dd)

def get_df_from_zip(file_type,zip_file, participants):
    
    #Get participant list from participants data frame
    participant_list = list(participants["Participant ID"])

    #Open data zip file
    z     = zipfile.ZipFile(zip_file)
    
    #Get list of files of specified type
    file_list = get_file_names_from_zip(z, file_type=file_type)
    
    #Open file inside zip
    dfs=[]
    for file_name in file_list:  
        sid = get_user_id_from_filename(file_name)
        if(sid in participant_list):
            f = z.open(file_name)
            df  = pd.read_csv(f)
            df["Subject ID"] = sid
            dfs.append(df)
    df = pd.concat(dfs)
    return(df)

def fix_df_column_types(df, dd):
    #Set Boolean/String fields to string type to prevent
    #interpretation as numeric for now. Leave nans in to
    #indicate missing data.
    for field in list(df.keys()):
        
        if dd.loc[field]["DataType"] in ["Boolean","String"]:
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else str(x))

    return(df)

def get_participant_info(data_catalog):
    file = data_catalog.data_dir + data_catalog.loc["Participant Information"]["Data File Name"]
    df   = pd.read_csv(file)
    return(df)

def get_participants_by_type(data_catalog, participant_type):
    pi = get_participant_info(data_catalog)
    pi = pi[pi["Participant Type"]==participant_type]
    return(pi)

def load_data(data_catalog, data_product):
    participant_df   = get_participants_by_type(data_catalog,"full")
    data_dictionary = get_data_dictionary(data_catalog, data_product)
    df = get_df_from_zip(data_dictionary.data_file_name, data_catalog.data_file, participant_df)
    index = [x.strip() for x in data_dictionary.index_fields.split(";")]
    df = df.set_index(index)    
    df = fix_df_column_types(df,data_dictionary)
    df = df.sort_index(level=0)
    df.name = data_dictionary.name
    return(df)
    
def get_subject_ids(df):
    sids = list(df.index.levels[0])
    return list(sids)

def get_variables(df): 
    numerical_types = [dtype('int64'), np.dtype('float64')]
    cols = [c for c in list(df.columns) if df.dtypes[c] in numerical_types]
    return(cols)