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

def plot_summary_histograms(df, dd, cols=3, fields=[]):  
    display(HTML("<H2>Summary Histograms by Variable: %s</H2>"%df.name))
    
    num_fields = len(list(df.keys()))

    # temporary skip list
    skip_list = ['Time Sent', 'Time Received', 'Time Opened', 'Time Completed']
    
    rows = int(np.ceil(num_fields/3))
    fig, axes = plt.subplots(rows, cols, figsize=(4*3,rows*3))
    i=0
    for field in list(df.keys()):
        if(field in fields or len(fields)==0):
            if dd.loc[field]["DataType"] in ["Boolean","String"]:
                this_ax = axes[i//cols,i%cols]
                df[field].value_counts().plot(kind="bar",ax=this_ax)
                this_ax.grid(axis='y')
                this_ax.set_title(field)
                if ((field == 'Notification') or
                    (field == 'Morning Message') or
                    (field == 'Anchor Message')): 
                    table = df[field].value_counts()
                    selected_ticks = np.linspace(start=0, stop=(len(table)-1), num=10, dtype=int)
                    this_ax.set_xticks(selected_ticks)
                    x_labels = []
                    for tick in selected_ticks:
                        shorten_message = table.index[tick][:11] + '...'
                        x_labels.append(shorten_message)                    
                    this_ax.set_xticklabels(x_labels)                    
                i=i+1
            else: 
                this_ax = axes[i//cols,i%cols]
                if field not in skip_list:   
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
