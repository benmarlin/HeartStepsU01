import numpy as np
import pandas as pd
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

def morning_survey_response_report(dc):

  participants_df = data_utils.get_participant_info(dc)
  participants_df=participants_df.set_index("Participant ID")

  product = "Morning Survey"
  df=data_utils.load_data(dc, product)

  fields = ["Busy", "Committed","Rested","Mood"]
  obs_df = 0+((0+~df[fields].isnull()).sum(axis=1)>0)
  obs_df

  response_data = []

  participants = list(obs_df.index.levels[0])
  for p in participants:

    status = participants_df.loc[p]["Participant Status"]

    comments = "Participant status: %s. "%status
    if(len(obs_df[p]))>0:
      dates = pd.to_datetime(obs_df[p].index)
      last_date = dates.max()
      dates_with_self_report = dates[obs_df[p]>0]
      if(len(dates_with_self_report)>0):
        last_date_with_self_report = dates_with_self_report.max()
        days_since_last_self_report = (last_date - last_date_with_self_report).days

        if(days_since_last_self_report>=3):
          if(status=="active"):
            comments = comments + "Requires participant follow-up."
          else:
            days_since_last_self_report = np.nan
      else:
        last_date_with_self_report = pd.NaT
        days_since_last_self_report = np.nan
        comments = comments + "No recorded self report for this participant"
    else:
      last_date = pd.NaT
      last_date_with_self_report = pd.NaT
      days_since_last_self_report = np.nan 
      comments = comments + "No data found for this participant"

    response_data.append({"Subject ID":p, 
                          "Last date":last_date,
                          "Last Morning Survey":last_date_with_self_report,
                          "Days overdue":days_since_last_self_report,
                          "Comments":comments  
                          })
    
  df_mis_report = pd.DataFrame(response_data)
  df_mis_report=df_mis_report.set_index("Subject ID")
  df_mis_report.sort_values(by="Days overdue", inplace=True, ascending=False)
  return(df_mis_report)