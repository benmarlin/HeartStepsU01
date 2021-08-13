import pandas as pd
from datetime import date
from datetime import datetime

def get_date(str_date):
  str_date = datetime.strptime(str_date, '%m/%d/%Y').strftime('%Y,%m,%d')
  check_date = str_date.split(',')
  check_date = date(int(check_date[0]),int(check_date[1]),int(check_date[2]))
  return check_date

def weekof(str_date):  
  return get_date(str_date).isocalendar()[1]

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

def iweekof(df, previous_levels, str_date):
  #iweekof find the corresponding week for str_date, then find the corresponging 
  #index for the week in the data frame df, given previous_levels  
  week = weekof(str_date)
  desired_level = len(previous_levels)
  index_names = df.index.names
  index_values = {}
  previous_level_names = []
  previous_level_value = []
  for i, index_name in enumerate(index_names):
    index_values[index_name] = list(df.index.unique(index_name))
    if i < desired_level:
      previous_level_names.append(index_name)
      previous_level_value.append(previous_levels[i])
  possible_values = index_values[index_names[desired_level]]
  found_index = -1  
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

