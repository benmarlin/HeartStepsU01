import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn
import os
import zipfile
from IPython.display import HTML, display, Markdown, Latex
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import tabulate
from . import data_utils

def highlight_greaterthan(s, threshold, column):
    
    #Custom row shader that alternates a highlight color
    #for normal rows as rows greater than or equal to threshold
    styler={(True,True):'background-color: #eeaaaa',
            (False,True):'background-color: #ffdddd',
            (True,False):'background-color: #aaaaaa',
            (False,False):'background-color: #dddddd'}
    id = np.arange(len(s))
    even = (id%2==0)
    special = list(s[column] >= threshold)

    return pd.DataFrame([ [styler[v]]*len(s.columns) for v in zip(even, special)],columns=s.columns, index=s.index)

def format_date(t, fmt='{:%Y-%m-%d}'):
    #Custom date formatter that can deal with NaT dates
    try:
        return fmt.format(t) # or strftime
    except ValueError:
        return "-"
        
def format_numeric_with_nan(x, fmt="{:.0f}"):
    if(np.isnan(x)):
        return("-")
    else:
        return(fmt.format(x))


def morning_survey_response_report(dc, threshold=3, b_crop=False):

  #Get participants listing
  participants_df = data_utils.get_participant_info(dc)
  participants_df = participants_df.set_index("Participant ID")
  
  #Select morning survey and get data frame
  product = "Morning Survey"
  df=data_utils.load_data(dc, product)

  #Select fields to use when computing response rate
  fields = ["Busy", "Committed", "Rested", "Mood"]

  #Create an observation indicator for an observed value in any
  #of the above fields. Sort to make sure data frame is in date order
  #per participant
  obs_df = 0+((0+~df[fields].isnull()).sum(axis=1)>0)
  obs_df.sort_index(axis=0, inplace=True,level=1) 

  #Get the participant ids according to the data frame
  participants = list(obs_df.index.levels[0])

  #Construct report data frame
  response_data = []
  for p in participants:

    #Get the participant's status
    status = participants_df.loc[p]["Participant Status"]
    comments = "Participant status: %s. "%status

    #Check if there is any data for the participant
    if(len(obs_df[p]))>0:

      #Get dates with data and record first and last dates with data
      dates = pd.to_datetime(obs_df[p].index)     
      last_date = dates[-1]
      first_date = dates[0]

      intervention_date = participants_df.loc[p]['Intervention Start Date']
      end_date = participants_df.loc[p]['End Date']
    
      new_obs_df = obs_df[p].copy()
      if b_crop:
          if str(intervention_date).lower() != "nan":
              intervention_date = pd.to_datetime(intervention_date)             
              new_obs_df = new_obs_df.loc[dates >= intervention_date]
              dates = pd.to_datetime(new_obs_df.index)
              today = pd.to_datetime(dt.date.today())
              if intervention_date > today:
                  print('{:<3} intervention date {} is past today\'s date {}'.format(
                          p, intervention_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
          else:
              print('{:<3} missing intervention date'.format(p))
              continue
          if status == 'withdrew':
              new_obs_df = new_obs_df.loc[dates <= end_date]
              dates = pd.to_datetime(new_obs_df.index)
              last_date = dates[-1]

      #Compute all-time and weekly response rates
      n_days = len(new_obs_df)
      weekly_response_rate = new_obs_df[max(0,n_days-7):-1].mean()
      alltime_response_rate = new_obs_df.mean()

      #Determine whether it has been more than threshold days since last self report
      dates_with_self_report = dates[new_obs_df>0]
      if(len(dates_with_self_report)>0):
        last_date_with_self_report = dates_with_self_report.max()
        days_since_last_self_report = (last_date - last_date_with_self_report).days
        if(days_since_last_self_report>=threshold):
          if(status=="active"):
            comments = comments + "Requires participant follow-up."
          else:
            days_since_last_self_report = np.nan
      else:
        #If no self-report, record last day as NaT (not a time)
        #and days since last self report as total days 
        last_date_with_self_report = pd.NaT
        if(status=="active"):
            days_since_last_self_report = n_days
        else:
            days_since_last_self_report = np.nan
        comments = comments + "No recorded self report."
    else:
      #No data at all for participant
      #Record everything as missing
      last_date = pd.NaT      
      first_date = pd.NaT
      intervention_date = pd.NaT
      last_date_with_self_report = pd.NaT
      days_since_last_self_report = np.nan
      weekly_response_rate = np.nan
      alltime_response_rate = np.nan
      comments = comments + "No data found for this participant."

    if b_crop:
        start_key = "Intervention date"
        start_value = intervention_date
    else:
        start_key = "First date"
        start_value = first_date

    #Create dataframe row
    response_data.append({"Subject ID":p, 
                          start_key:start_value,
                          "Last date":last_date,
                          "All time response rate": alltime_response_rate,
                          "7-day response rate": weekly_response_rate,
                          "Last Morning Survey":last_date_with_self_report,
                          "Days overdue":days_since_last_self_report,
                          "Comments":comments  
                          })

  #Create dataframe from rows and sort by days overdue
  df_mis_report = pd.DataFrame(response_data)
  df_mis_report.sort_values(by="Days overdue", inplace=True, ascending=False)

  #Style the dataframe
  df_mis_report=df_mis_report.style.apply(highlight_greaterthan, threshold=threshold, column="Days overdue", axis=None)\
    .format({'Days overdue': format_numeric_with_nan,\
             'All time response rate': "{:.1%}",\
             '7-day response rate': "{:.1%}",\
             'Last date':format_date,\
             'Last Morning Survey': format_date,\
             'Intervention date':format_date,\
             start_key:format_date})\
    .set_table_attributes('cellpadding=5')\
    .hide_index()\
    .apply(lambda s: ["text-align: right"]*len(s), subset=["Days overdue","All time response rate","7-day response rate"])\
    .apply(lambda s: ["text-align: center"]*len(s), subset=["Subject ID",start_key,"Last date","Last Morning Survey"])\
    .apply(lambda s: ["text-align: left"]*len(s), subset=["Comments"])\

  return(df_mis_report)
