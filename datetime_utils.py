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

def studyday(str_date):
  # TODO
  print('studyday =', str_date)
  return 1

def studyweek(str_date):
  # TODO
  print('studyweek =', str_date)
  return 1

