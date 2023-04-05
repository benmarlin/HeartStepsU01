import numpy as np
import pandas as pd
import datetime as dt
import os
import zipfile
from datetime import datetime, timedelta
from urllib.parse import urlparse

study_prefix = "U01"

def get_user_id_from_filename(f):
    #Get user id from from file name
    temp = f.split(".")
    if(len(temp)==6):
      return(temp[3])

    temp = f.split("/")
    if(len(temp)==3):
      return(temp[1])

    raise ValueError("Can not get user id from file name.")

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
    z = zipfile.ZipFile(zip_file)
    
    #Get list of files of specified type
    file_list = get_file_names_from_zip(z, file_type=file_type)
    
    #Open file inside zip
    dfs=[]
    for file_name in file_list:
        sid = get_user_id_from_filename(file_name)
        if(sid in participant_list):
            f = z.open(file_name)
            file_size = z.getinfo(file_name).file_size
            if file_size > 0:
                df  = pd.read_csv(f, low_memory=False)
                df["Subject ID"] = sid
                dfs.append(df)
            else:
                print('warning %s is empty (size = 0)' % file_name)
    df = pd.concat(dfs)
    return(df)

def fix_df_column_types(df, dd):
    #Set Boolean/String fields to string type to prevent
    #interpretation as numeric for now. Leave nans in to
    #indicate missing data.
    for field in list(df.keys()):
        if not (field in dd.index): continue
        dd_type = dd.loc[field]["DataType"]    
        if dd_type in ["Boolean","String","Categorical"]:
            if field == 'url':
                urls = df[field].values
                for index, url in enumerate(urls):
                    parsed = urlparse(url)
                    df[field].values[index] = parsed.path[1:]
            else:
                df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else str(x))   
        elif dd_type in ["Ordinal"]:
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else int(x))
        elif dd_type in ["Time"]:
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else pd.to_timedelta(x))
        elif dd_type in ["Date"]:
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else datetime.strptime(x, "%Y-%m-%d"))   
        elif dd_type in ["DateTime"]:
            #Keep only time for now
            max_length = max([len(str(x).split(':')[-1]) for x in df[field].values]) # length of last item after ':'
            if max_length < 6: # this includes time with AM/PM
                df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else pd.to_timedelta(x[11:]))
            else: # for example: 2020-06-12 23:00:1592002802
                df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else
                                          pd.to_timedelta(pd.to_datetime(x[:16]).strftime("%H:%M:%S")))  
            #print('\n%s nlargest(10) =\n%s' % (field, df[field].value_counts().nlargest(10)))
    return(df)

def get_participant_info(data_catalog):
    file = data_catalog.data_dir + data_catalog.loc["Participant Information"]["Data File Name"]
    df   = pd.read_csv(file)
    return(df)

def get_participants_by_type(data_catalog, participant_type):
    pi = get_participant_info(data_catalog)
    check_type = []
    for type_i in pi["Participant Type"].values:
        if str(type_i).find(participant_type) >= 0:
            check_type.append(True)
        else:
            check_type.append(False)
    pi = pi[check_type]
    return(pi)

def crop_data(participants_df, df, b_display, b_crop_end=True):
    #Crop before the intervention start date
    #Set b_crop_end = True to also crop after the end date (for withdrew status)
    participants_df = participants_df.set_index("Participant ID")
    fields = list(df.keys())
    #Create an observation indicator for an observed value in any
    #of the above fields. Sort to make sure data frame is in date order
    #per participant
    obs_df = 0+((0+~df[fields].isnull()).sum(axis=1)>0)
    obs_df.sort_index(axis=0, inplace=True,level=1)
    #Get the participant ids according to the data frame
    participants = list(obs_df.index.levels[0])
    frames = []
    for p in participants:
        intervention_date = participants_df.loc[p]['Intervention Start Date']
        dates = pd.to_datetime(obs_df[p].index)
        #Check if there is any data for the participant
        if(len(obs_df[p]))>0:
            new_obs_df = obs_df[p].copy()
            if str(intervention_date).lower() != "nan":
                #Check if intervention date is past today's date
                intervention_date = pd.to_datetime(intervention_date)
                new_obs_df = new_obs_df.loc[dates >= intervention_date]
                dates = pd.to_datetime(new_obs_df.index)
                today = pd.to_datetime(dt.date.today())
                if (intervention_date > today) and b_display:
                  print('{:<3} intervention date {} is past today\'s date {}'.format(
                          p, intervention_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
                #Crop before the intervention start date
                dates_df = pd.to_datetime(df.loc[p].index)
                new_df = df.loc[p].copy()
                new_df = new_df.loc[dates_df >= intervention_date]
                if b_crop_end:
                    status = participants_df.loc[p]["Participant Status"]
                    end_date = participants_df.loc[p]['End Date']                    
                    if status == 'withdrew':
                        end_date = pd.to_datetime(end_date)
                        dates_df = pd.to_datetime(new_df.index)
                        new_df = new_df.loc[dates_df <= end_date]
                new_df['Subject ID'] = p
                new_df = new_df.reset_index()
                date_name = 'DATE'
                columns = list(new_df.columns)
                if date_name not in columns:
                    for col_name in columns:
                        if col_name.find('Date') >= 0:
                            date_name = col_name              
                new_df['Date'] = pd.to_datetime(new_df[date_name]).dt.strftime('%Y-%m-%d')
                new_df = new_df.set_index(['Subject ID', 'Date'])
                frames.append(new_df)
            else:
                if b_display:
                    status = participants_df.loc[p]["Participant Status"]
                    if (status != 'withdrew') and (str(status).lower() != 'nan'): 
                        print('{:<3} ({}) missing intervention start date'.format(p, status))
                continue
    if len(frames) > 0:
        df = pd.concat(frames)
        df = df.sort_index(level=0)
    return df

def crop_end_fitbit_per_minute(data_product, participants_df, df, b_display):
    #For Fitbit Data Per Minute, we only crop after the end date (for withdrew status)
    #Fitbit Data Per Minute has 'Subject ID', 'time' as indices and 'date' as column
    participants_df = participants_df.set_index("Participant ID")
    fields = list(df.keys())
    initial_indices = df.index.names
    #Create an observation indicator for an observed value in any
    #of the above fields. Sort to make sure data frame is in date order
    #per participant
    obs_df = 0+((0+~df[fields].isnull()).sum(axis=1)>0)
    obs_df.sort_index(axis=0, inplace=True,level=1)
    #Get the participant ids according to the data frame
    participants = list(obs_df.index.levels[0])
    frames = []
    for p in participants:
        #Check if there is any data for the participant
        if(len(obs_df[p]))>0:
            new_df = df.loc[p].copy()
            date_name = ''
            if 'Date' in new_df:
                date_name = 'Date'
            elif 'date' in new_df:
                date_name = 'date'
            if date_name != '':        
                status = participants_df.loc[p]["Participant Status"]
                if status == 'withdrew':
                    new_df[date_name] = pd.to_datetime(new_df[date_name])
                    end_date = pd.to_datetime(participants_df.loc[p]['End Date'])
                    new_df = new_df.loc[new_df[date_name] <= end_date]
                    new_df[date_name] = new_df[date_name].dt.strftime('%Y-%m-%d')
                    if (b_display):
                        print('%s: cropped after %s for withdrew participant %s' % (
                               data_product, end_date.strftime('%Y-%m-%d'), str(p)))
            new_df['Subject ID'] = p
            new_df = new_df.reset_index()
            new_df = new_df.set_index(initial_indices)            
            frames.append(new_df)
    if len(frames) > 0:
        df = pd.concat(frames)
        df = df.sort_index(level=0)
        if (b_display):
            print('\nchecking data types...\n')
    return df

def load_data(data_catalog, data_product, b_crop=True, b_display=True):
    participant_df  = get_participants_by_type(data_catalog,"full")
    data_dictionary = get_data_dictionary(data_catalog, data_product)
    df = get_df_from_zip(data_dictionary.data_file_name, data_catalog.data_file, participant_df)
    index = [x.strip() for x in data_dictionary.index_fields.split(";")]
    df = df.set_index(index)
    df = df.sort_index(level=0)    
    if (b_crop) and (data_product != 'Fitbit Data Per Minute'):        
        df = crop_data(participant_df, df, b_display, b_crop_end=True)
    elif (b_crop) and (data_product == 'Fitbit Data Per Minute'):
        df = crop_end_fitbit_per_minute(data_product, participant_df, df, b_display)
    df = fix_df_column_types(df,data_dictionary)
    df.name = data_dictionary.name   
    return(df)

def load_baseline(data_catalog, data_product, filename):
    data_dictionary = get_data_dictionary(data_catalog, data_product)
    df = pd.read_csv(filename)    
    index = [x.strip() for x in data_dictionary.index_fields.split(";")]
    df = df.set_index(index)
    df = fix_df_column_types(df,data_dictionary)
    df.sort_index(level=0)
    df.name = data_dictionary.name
    return(df)
        
def get_subject_ids(df, b_isbaseline=False):
    if b_isbaseline:
        sids = df.index.astype(str)
    else:
        sids = list(df.index.levels[0])    
    return list(sids)

def get_variables(df): 
    numerical_types = [np.dtype('int64'), np.dtype('float64')]
    cols = [c for c in list(df.columns) if df.dtypes[c] in numerical_types]
    return(cols)

def get_catalogs(catalog_file):
    df = pd.read_csv(catalog_file)
    df = df["Data Product Name"]
    df = df[df.values != "Participant Information"]
    df = df[df.values != "Baseline Survey"]
    return list(df)

def get_categories(dd, field):
    categories = dd.loc[field]['Notes'].split(' | ')
    return categories

def resample_fitbit_per_minute(participant='105', df=None, filename=None, interval='30Min', b_dropna=True):
    #1. Set df to desired input df, or set filename to load df (df=None)
    #2. Set participant ID, for example: '105'
    #3. Set interval for resampling, for example: '30Min'
    if filename != None:
        print('loading data for participant %s from %s' % (participant, filename))
        df = pd.read_csv(filename, low_memory=False)
    else:
        print('getting data for participant', participant)
        df = df.reset_index()
    df = df.groupby(by='Subject ID').get_group(participant)
    #Temporary fix for data export issue: replace S with 00 in time format
    df['time'] = df['time'].map(lambda x: str(x).replace('S', '00'))
    df['datetime'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('datetime')
    df = df.resample(interval, level=0).first()
    df = df.reset_index().set_index(['Subject ID', 'time'])
    df.sort_index(level=0)
    df.name = 'Fitbit Data Per Minute'
    if b_dropna:
        df = df.dropna()
    return df

def merge_data_frames(dc, data_set_names, short_names):

  dfs={}
  dds={}
  columns={}

  #Get all dataframes and columns
  for name in data_set_names:
    df              = load_data(dc, name, b_crop=True, b_display=True)
    dfs[name]       = df
    columns[name]   = list(df.columns)
    dds[name]       = get_data_dictionary(dc,name)

  #Merge data sets
  for name in data_set_names:

    #Get column names in other frames
    all_cols = []
    for key in columns:
      if(key != name):
        all_cols = all_cols + columns[key]
    all_cols = set(all_cols)

    #Get overlapping column names
    overlap_cols = list(set(columns[name]).intersection(all_cols))

    #Re-map column names
    name_map = {x: short_names[name] + " " + x for x in overlap_cols}
    dfs[name] = dfs[name].rename(name_map,axis=1,errors='raise')
    dds[name] = dds[name].rename(name_map,axis=0,errors='raise')

  #Concatenate frames with re-mapped names
  df = pd.concat([dfs[name] for name in data_set_names],axis=1)
  dd = pd.concat([dds[name] for name in data_set_names],axis=0)
  dd = dd.drop(labels=["Date","Subject ID"])

  return(df,dd)

def relabel_participants(df,id_map=None):
  #Do a basic relabeling of participant IDs
  if id_map is None:
    if isinstance(df.index, pd.MultiIndex):
      ids = list(df.index.levels[0])
    else:
      ids = list(df.index)

    new_ids = np.random.permutation(len(ids))
    id_map_list = list(zip(ids,new_ids ))
    id_map = {x[0]:x[1] for x in id_map_list }
    print("Creating new ID map")
  else:
    print("Using supplied ID map")

  #Drop any participants not in the mapping
  if isinstance(df.index, pd.MultiIndex):
    participants_in_index = list(df.index.levels[0])
  else:
    participants_in_index = list(df.index)

  participants_in_index = [str(x) for x in participants_in_index]
  participants_in_map   = [str(x) for x in id_map.keys()]
  participants_to_drop = list(set(participants_in_index) - set(participants_in_map))

  df = df.drop(labels = participants_to_drop )

  #Re-map the ids
  df = df.rename(id_map)
  return(df,id_map)

def add_date_indicators(df,dd):
  #add collection of date related indicators
  dates = pd.to_datetime(df.index.get_level_values(1))
  
  #Day of week
  df["Day of Week"] = dates.dayofweek
  dd = dd.append(pd.Series({"DataType": "Integer",	"Required": "True", 	"ElementDescription":"Day of the week. 0 is Monday. 6 is Sunday", 	"ValueRange": "{0..6}"},name="Day of Week"))

  #Is weekend day
  df["Is Weekend Day"] = dates.dayofweek >=5 
  dd = dd.append(pd.Series({"DataType": "Boolean",	"Required": "True", 	"ElementDescription":"Is this day a weekend day. True or False.", 	"ValueRange": "{0,1}"},name="Is Weekend Day"))

  #Day of year
  df["Day of Year"] = dates.dayofyear
  dd = dd.append(pd.Series({"DataType": "Integer",	"Required": "True", 	"ElementDescription":"Day of the year. Jan 1 is day 1.", 	"ValueRange": "{1..365}"},name="Day of Year"))

  #Get days in study variable for each participant
  study_days = []
  for id in list(df.index.levels[0]):
    dates      = pd.to_datetime(df.loc[id].index)
    date_diff  = dates-dates[0]
    study_days = study_days + list(date_diff.days)
  df["Study day"] = study_days
  dd = dd.append(pd.Series({"DataType": "Integer",	"Required": "True", 	"ElementDescription":"Days since start of study. First day is day 0.", 	"ValueRange": "{0,...}"},name="Study day"))

  return df,dd

def get_df_from_zip(file_type,zip_file, participants,interval=None,crop=True):
    
    #Get participant list from participants data frame
    participant_list = list(participants["Participant ID"])

    participants["Start Date"] = participants["Intervention Start Date"].apply(pd.to_datetime)
    participants["End Date"] = participants["End Date"].apply(pd.to_datetime)
    participants = participants.set_index("Participant ID")

    #Open data zip file
    z = zipfile.ZipFile(zip_file)
    
    #Get list of files of specified type
    file_list = get_file_names_from_zip(z, file_type=file_type)
    
    #Open file inside zip
    dfs=[]
    for count,file_name in enumerate(file_list):

        sid = get_user_id_from_filename(file_name)
        if(sid not in participant_list):
            print("Processing ID %s (%d/%d)"%(sid,count,len(file_list)))
            print("  ID not in participants list")
        else:
            print("Processing ID %s (%d/%d)"%(sid,count,len(file_list)))

            f = z.open(file_name)
            file_size = z.getinfo(file_name).file_size
            if file_size > 0:
                df  = pd.read_csv(f, low_memory=False)
                df["Participant ID"] = sid

                df['time'] = df['time'].map(lambda x: str(x).replace('S', '00'))
                df['datetime'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
                
                #Require both steps and heart rate to not be nan
                #Set both to nan if either is and consider minute to
                #be invalid in this case
                df['valid_minutes'] = np.logical_and(df["steps"].notna(), df["heart_rate"].notna())
                df.loc[df["valid_minutes"]==False, 'steps'] = np.nan
                df.loc[df["valid_minutes"]==False, 'heart_rate'] = np.nan

                if(interval is not None):
                  df1 = df.set_index('datetime').resample(interval).first()
                  df2 = df.set_index('datetime').resample(interval).sum()
                  df3 = df.set_index('datetime').resample(interval).mean()

                  df1["steps"] = df2["steps"]
                  df1["valid_minutes"] = df2["valid_minutes"]
                  df1["heart_rate"] = df3["heart_rate"]

                  df1=df1.reset_index()
                  df1=df1.drop(columns=["username","fitbit_account"])

                  df = df1

                if(crop):
                  df=df.set_index(["datetime"])
                  start=participants.loc[sid]["Start Date"]
                  end=participants.loc[sid]["End Date"]

                  if(start is pd.NaT): 
                    print("  Participant %s start date is missing"%sid)
                    if(end is pd.NaT):
                      print("  Participant %s end date is missing"%sid)
                    else:
                      df = df[:end]
                  else:
                    if(end is pd.NaT):
                      print("  Participant %s end date is missing"%sid)
                      df = df[start:]
                    else:
                      print(  "  cropping dates to ",start, " ", end)
                      df = df[start:end]                 

                df = df.reset_index()
                df=df.set_index(["Participant ID","datetime"])
                df.loc[df["valid_minutes"]==0, 'steps'] = np.nan
                df.loc[df["valid_minutes"]==0, 'heart_rate'] = np.nan

                print("  has %d rows"%len(df))

                dfs.append(df)

            else:
                print('warning %s is empty (size = 0)' % file_name)

    df = pd.concat(dfs)
    return(df)

def apply_transforms(df,dd,transforms):
  for t in transforms:
    t_type = t["type"]

    #Drop the columns
    if(t_type=="drop"):
      if(t["col"] in df.columns):
        df=df.drop(labels=t["col"],axis=1)
      if(t["col"] in list(dd.index)):
        dd=dd.drop(labels=t["col"],axis=0)

    #Add a missing indicator
    elif(t_type=="miss_ind"):  
      df[t["new_name"]] = df[t["col"]].notna()
      dd = dd.append(pd.Series({"DataType": "Boolean",	"Required": "True", 	"ElementDescription":t["desc"], 	"ValueRange": "{True, False}"},name=t["new_name"]))

    #Rename columns:
    elif(t_type=="rename"):  
      df=df.rename({t["col"]:t["new_name"]},axis=1) 
      dd=dd.rename({t["col"]:t["new_name"]},axis=0) 

    #Merge columns by averaging
    elif(t_type=="avg"):
      df[t["new_name"]] = df[t["cols"]].mean(axis=1)
      dd = dd.append(pd.Series({"DataType": "Float",	"Required": "True", 	"ElementDescription":t["desc"], 	"ValueRange": ""},name=t["new_name"]))
      print("Producing average column")

  return(df,dd)

