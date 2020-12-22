import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
import zipfile
from IPython.display import HTML, display, Markdown, Latex
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import tabulate
from . import data_utils

def parse_date_time(input_df, df):
    parse_date_array = []
    for value in input_df.values:
        value = str(value)         
        if value != 'nan':
            date_time = value.split()
            date = date_time[0]
            time_array = date_time[1].split(':')
            hh = time_array[0]
            mm = time_array[1]
            # keep only the time for now
            value = hh + ':' + mm
            parse_date_array.append(value)        
    output_df = pd.Series(parse_date_array)    
    return output_df

def parse_url(input_df):
    options = ['decision', 'nightly', 'initialize']
    parse_date_array = []
    for index, value in enumerate(input_df.values):
        value = str(value).lower()
        candidate = 'other'
        for option in options:
            if (value.find(option) >= 0):
                candidate = option
                break
        parse_date_array.append(candidate)        
    output_df = pd.Series(parse_date_array)    
    return output_df

def shorten_xlabels(this_ax, table):
    selected_ticks = np.linspace(start=0, stop=(len(table)-1), endpoint=True, num=10, dtype=int)
    this_ax.set_xticks(selected_ticks)
    x_labels = []
    for tick in selected_ticks:
        shorten_message = table.index[tick][:11]
        if len(table.index[tick]) > 10:
            shorten_message += '...'
        x_labels.append(shorten_message)                    
    this_ax.set_xticklabels(x_labels)
    return this_ax
                    
def plot_summary_histograms(df, dd, cols=3, fields=[]):  
    display(HTML("<H2>Summary Histograms by Variable: %s</H2>"%df.name))
    
    num_fields = len(list(df.keys()))

    # temporary skip list
    skip_list = ['ID', 'Imputed', 'request_data', 'response_data']

    # shorten_list = list of fields to shorten the x-axis labels
    shorten_list = []
    search_list  = ['notif', 'message', 'date', 'url', 'request']
    for field in list(df.keys()):
        field_name = str(field).lower()
        for search in search_list:
            if (field_name.find(search) >= 0):
                shorten_list.append(field)
                break
                
    rows = int(np.ceil(num_fields/3))
    fig, axes = plt.subplots(rows, cols, figsize=(4*3,rows*3))
    i=0
    for field in list(df.keys()):
        if(field in fields or len(fields)==0):

            field_type = dd.loc[field]["DataType"]
            #print('field =', field, '\t\t ->', field_type)

            this_ax = axes[i//cols,i%cols]
            this_ax.set_title(field)

            if field_type in ["Time"]:
                # plot histogram, one bin per half hour of the day
                df_time = df[field] / pd.Timedelta(minutes=60)
                df_time.hist(figure=fig, ax=this_ax, bins=48)
                this_ax.set_xlim(0,24)                
            elif field_type in ["DateTime"]:                                
                df_datetime = parse_date_time(df[field], df)
                table = df_datetime.value_counts()
                if len(table) > 1:
                    table.plot(kind="bar",ax=this_ax)               
                    this_ax.grid(axis='y')
                    this_ax = shorten_xlabels(this_ax, table)
            elif field_type in ["Boolean","String","Date"]:
                if field not in skip_list:
                    if field == 'url':
                        df_url = parse_url(df[field])
                        table = df_url.value_counts()
                    else:
                        table = df[field].value_counts()
                    table.plot(kind="bar",ax=this_ax)
                    this_ax.grid(axis='y')
                    if field in shorten_list:
                        this_ax = shorten_xlabels(this_ax, table)
            else:                 
                if field not in skip_list:   
                    df[field].hist(figure=fig,ax=this_ax)
                    this_ax.grid(True)
                    
            i=i+1
    plt.tight_layout()
    plt.show()

def plot_indifivual_time_series(df,variable,subject_id):
    this_df = df.xs(subject_id, level=0, axis=0, drop_level=True)
    this_df[variable].plot(kind='bar', grid=True, figsize=(12,4) )
    plt.title("Subject %s: %s"%(subject_id,variable))
    plt.show()

def show_individual_time_series_visualizer(df):
    sids=data_utils.get_subject_ids(df)
    vars=data_utils.get_variables(df)
    interact(plot_indifivual_time_series, df=fixed(df), subject_id=sids,variable = vars);
  
  
def show_summary_table(df):
    subjects = data_utils.get_subject_ids(df)
    cols     = list(df.columns)

    subjects_str=", ".join(subjects)
    cols_str = ", ".join(list(df.columns))

    num_subjects = len(subjects)
    num_days=len(df)
    num_cols = len(cols)

    names = ["Number of Participants","Number of Variables","Total Days","Subjects IDs","Variables"]
    values = [num_subjects, num_cols, num_days, subjects_str, cols_str]

    x=HTML(tabulate.tabulate(zip(names, values), tablefmt='html'))
    x.data = x.data.replace("table", "table style='border-spacing: 0px; border-collapse: collapse; padding: 5px'")
    x.data = x.data.replace("td", "td style='border: 1px solid black;padding: 5px '")
    x.data = x.data
    display(HTML("<H2>Summary Table: %s</H2>"%df.name))
    display(x)
    
    
def show_data_dictionary(dd):
    display(HTML("<H2>Data Dictionary: %s</H2>"%dd.name))
    dfStyler = dd[["DataType","ElementDescription"]].style.set_properties(**{'text-align': 'left','border':'1px solid black'})
    dfStyler =dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left'),('border','1px solid black')])])
    dfStyler =dfStyler.set_table_attributes('style="border-collapse:collapse; border:1px solid black"')
    display(dfStyler)

def show_missing_data_by_variable(df):
    display(HTML("<H2>Missing Data Rate by Variable: %s</H2>"%df.name))
    plt.figure(figsize=(10,8))
    df.isnull().mean().plot(kind='barh')
    plt.grid(True)
    plt.xlim(0,1)
    plt.title("Missing Data Rate by Variable")
    plt.xlabel("Missing data rate")
    plt.show()
    
def show_missing_data_by_participant(df):
    display(HTML("<H2>Missing Data Rate by Participant: %s</H2>"%df.name))    
    plt.figure(figsize=(10,20))
    df.isnull().mean(level=0).mean(axis=1).sort_values().plot(kind='barh')
    plt.grid(True)
    plt.xlim(0,1)
    plt.title("Missing Data Rate by Participant")
    plt.xlabel("Missing data rate")
    plt.show()

def show_data_selector(data_dir):
    l = [f for f in os.listdir(data_dir) if ".zip" in f]
    w=widgets.Dropdown(
        options=l,
        value=l[0],
        description='Data Archive:',
        disabled=False,
    )
    display(w)
    return(w)
