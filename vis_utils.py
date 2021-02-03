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

def shorten_xlabels(this_ax, table, length_limit):
    selected_ticks = np.linspace(start=0, stop=(len(table)-1), endpoint=True, num=10, dtype=int)
    this_ax.set_xticks(selected_ticks)
    x_labels = []
    for tick in selected_ticks:
        shorten_message = table.index[tick][:(length_limit+1)]
        if len(table.index[tick]) > length_limit:
            shorten_message += '...'
        x_labels.append(shorten_message)                    
    this_ax.set_xticklabels(x_labels)
    return this_ax
                    
def plot_summary_histograms(df, dd, cols=3, fields=[]):  
    display(HTML("<H2>Summary Histograms by Variable: %s</H2>"%df.name))
    num_fields = len(list(df.keys()))                
    rows = int(np.ceil(num_fields/3))
    fig, axes = plt.subplots(rows, cols, figsize=(4*3,rows*3))
    i=0
    for field in list(df.keys()):
        if(field in fields or len(fields)==0):
            if field not in list(dd.index): continue
            this_ax = axes[i//cols,i%cols]
            #Shorten title if it is too long
            title = field.replace(':', '\n')
            max_length_limit = 40
            if len(title) > max_length_limit:
                title = title[:max_length_limit] + str('...')
            this_ax.set_title(title)  
            str_values = [str(value).lower() for value in df[field].values]
            check_all_nan = ((len(set(str_values)) == 1) and (str_values[0] == 'nan'))
            if not check_all_nan:
                field_type = dd.loc[field]["DataType"]
                if field_type in ["Time", "DateTime"]:                         
                    #Plot histogram, one bin per half hour of the day
                    df_time = df[field] / pd.Timedelta(minutes=60)
                    df_time.hist(figure=fig, ax=this_ax, bins=48)
                    hours = [str(i) + ':00' for i in range(0,24,5)]
                    this_ax.set_xticklabels(hours)
                    this_ax.set_xlim(0,24)
                elif field_type in ["Date"]:
                    df[field].hist(figure=fig, ax=this_ax)
                    this_ax.grid(True)
                else:
                    if field_type in ["Boolean", "String", "Ordinal", "Categorical"]:
                        #Pandas automatically converts int into float, thus convert back to int for plot
                        if field_type in ["Ordinal"]:
                            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else str(int(x)))
                        table = df[field].value_counts()                        
                        #Plot table if it is not too big
                        if len(table) < 300:
                            if len(table) < 30: #Sort table if it is not too big                              
                                if field_type in ["Categorical"]:
                                    #Insert missing categories for zero count
                                    categories = data_utils.get_categories(dd, field)
                                    if len(categories) > len(table.index):
                                        zero_counts = [x for x in categories if x not in list(table.index)]
                                        for zero_count in zero_counts:
                                            table[zero_count] = 0                                        
                                table = table.sort_index()                                        
                            table.plot(kind="bar", x=field, ax=this_ax)
                            this_ax.grid(axis='y')
                    else:
                        df[field].hist(figure=fig, align='left', ax=this_ax)
                        if field_type not in ["Integer"]:
                            this_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            this_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                        this_ax.grid(True)
                    #Shorten xlabels if they are too long
                    max_length_limit = 15
                    current_max_length = max([len(str(value)) for value in df[field].values])
                    if current_max_length > max_length_limit:
                        this_ax = shorten_xlabels(this_ax, table, max_length_limit)
            i=i+1
    while (i < (rows*cols)):
        this_ax = axes[i//cols,i%cols]
        fig.delaxes(this_ax)
        i=i+1
    plt.tight_layout()
    plt.show()

def plot_individual_time_series(df,variable,subject_id):
    this_df = df.xs(subject_id, level=0, axis=0, drop_level=True)
    ax = this_df[variable].plot(kind='bar',grid=True, figsize=(12,4))
    
    if(len(this_df)>14):
      every_nth = 7
      all_labels = ax.xaxis.get_ticklabels()
      for n, label in enumerate(ax.xaxis.get_ticklabels()):
          if n % every_nth != 0:
              label.set_visible(False)
    
    plt.title("Subject %s: %s (Values)"%(subject_id,variable))
    plt.show()

    obs_df = 0+ ~this_df[variable].isnull()
    ax2 = obs_df.rolling(window = 3,min_periods=1).mean().plot(grid=True, figsize=(12,4))

    ax2.set_xticks(ax.get_xticks())
    ax2.xaxis.set_ticklabels(all_labels)
    if(len(this_df)>14):
      every_nth = 7
      for n, label in enumerate(ax2.xaxis.get_ticklabels()):
          if n % every_nth != 0:
              label.set_visible(False)
    plt.xticks(rotation=90)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(-0.02,1.05)

    plt.title("Subject %s: %s (3-Day Obs. Rate)"%(subject_id,variable))
    plt.show()


def show_individual_time_series_visualizer(df):
    sids=data_utils.get_subject_ids(df)
    variables=data_utils.get_variables(df)
    if (len(variables) > 0):
        interact(plot_individual_time_series, df=fixed(df), subject_id=sids, variable=variables)
    else:
        print('no variable to display')
  
def show_summary_table(df, b_isbaseline=False):
    subjects = data_utils.get_subject_ids(df, b_isbaseline)
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

def show_summary_table(df, b_isbaseline=False):
    subjects = data_utils.get_subject_ids(df, b_isbaseline)
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

def show_scores_table(df, table_name, subject='', xlim=None, cols=3):
    df = df.set_index(subject)
    columns = df.columns
    
    display(HTML("<H2>Histograms of %s</H2>"%table_name))    
    rows = int(np.ceil(len(columns)/3))
    fig, axes = plt.subplots(rows, cols, figsize=(4*3,rows*3))
    for i, field in enumerate(columns):
        this_ax = axes[i//cols,i%cols]
        df[field].hist(figure=fig, align='left', ax=this_ax)
        this_ax.grid(True)
        this_ax.set_title(field)
        if xlim != None:
            this_ax.set_xlim(xlim) 
    while (i+1 < (rows*cols)):
        i=i+1
        this_ax = axes[i//cols,i%cols]
        fig.delaxes(this_ax)
    plt.tight_layout()
    plt.show()
    
def show_data_dictionary(dd):
    display(HTML("<H2>Data Dictionary: %s</H2>"%dd.name))
    dfStyler = dd[["DataType","ElementDescription"]].style.set_properties(**{'text-align': 'left','border':'1px solid black'})
    dfStyler =dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left'),('border','1px solid black')])])
    dfStyler =dfStyler.set_table_attributes('style="border-collapse:collapse; border:1px solid black"')
    display(dfStyler)

def show_missing_data_by_variable(df):
    display(HTML("<H2>Missing Data Rate by Variable: %s</H2>"%df.name))
    if df.shape[1] < 100:
        plt.figure(figsize=(10,8))
    else:
        plt.figure(figsize=(10,30))
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
    l.sort(reverse=True) 
    w=widgets.Dropdown(
        options=l,
        value=l[0],
        description='Data Archive:',
        disabled=False,
    )
    display(w)
    return(w)

def show_catalog_selector(catalog_file):
    catalogs = data_utils.get_catalogs(catalog_file) 
    w=widgets.Dropdown(
        options=catalogs,
        value=catalogs[0],
        description='Catalog:',
        disabled=False,
    )
    display(w)
    return(w)
    
