import pandas as pd
from datetime import date
from datetime import datetime

def get_date(str_date):  
  str_date = datetime.strptime(str_date, '%m/%d/%Y').strftime('%Y,%m,%d')
  check_date = str_date.split(',')
  check_date = date(int(check_date[0]),int(check_date[1]),int(check_date[2]))
  return check_date

def dayofweek(str_date):
  day_names = ["ISO week days start from 1", "Monday", "Tuesday",
               "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  return day_names[get_date(str_date).isoweekday()]    

def isweekday(str_date):
  b_weekday = True
  day_week = dayofweek(str_date)
  if (day_week == "Saturday") or (day_week == "Sunday"):
    b_weekday = False
  return b_weekday

def weekof(df_d, df_w, d_date):
  #weekof finds the corresponding week for date
  #df_d is the data frame that contains index level "d"
  #df_w is the data frame that contains index level "w" 
  possible_values = df_d.index.unique(level="d")
  if d_date < len(possible_values):
    str_date = possible_values[d_date]
  else:
    print('cannot find corresponding date for index', d_date)
    return -1
  week = get_date(str_date).isocalendar()[1]
  found_index = -1
  possible_values = df_w.index.unique(level="w")
  for i, value in enumerate(possible_values):
    if value == week:
      found_index = i
      break  
  if found_index < 0:
    print('cannot find corresponding index for week of', str_date)
  return found_index

def studyday(str_date):
  # TODO
  print('studyday =', str_date)
  return 1

def studyweek(str_date):
  # TODO
  print('studyweek =', str_date)
  return 1

