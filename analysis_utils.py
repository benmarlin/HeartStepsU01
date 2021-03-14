import pandas as pd
import numpy as np
import math
import collections

import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.ticker as mticker

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import pacf, plot_pacf
from scipy.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

from . import data_utils


def process_morning_survey(df):
    #Get Mood categories and create new columns
    df['Mood'] = pd.Categorical(df['Mood'])
    df['Mood Code'] = df['Mood'].cat.codes
    categories = dict(enumerate(df['Mood'].cat.categories))

    for key, value in categories.items():
        df[value] = df['Mood Code'].apply(lambda x: True if x == key else False)

    column_list = ['Busy', 'Committed', 'Rested']    
    for key, value in categories.items():
        column_list.append(value)
    df_selected = df[column_list]

    return df_selected

def process_daily_metrics(df):
    return df[['Fitbit Step Count', 'Fitbit Minutes Worn']]

def process_previous_fitbit_data(df):
    import warnings
    warnings.filterwarnings('ignore')
    df['Previous Count'] = df['Fitbit Step Count'].shift()
    df['Previous Worn']  = df['Fitbit Minutes Worn'].shift()
    df.loc[df.groupby('Subject ID')['Previous Count'].head(1).index, 'Previous Count'] = 0
    df.loc[df.groupby('Subject ID')['Previous Worn'].head(1).index,  'Previous Worn']  = 0    
    return df

def process_fitbit_minute(df):
    df = df.rename(columns={"datetime": "Date"})
    df = df.drop(columns=['fitbit_account', 'username', 'date'])
    return df

def process_activity_logs(df):
    return df[['Activity Type Title', 'Start Time']]

def get_score_mapping(df_score):
    df_score = df_score.rename(columns={'study_id':'Subject ID'})
    score_mapping = {}
    for score_name in list(df_score.columns):
        if score_name != 'Subject ID':
            score_mapping[score_name] = {}
            for i, subject_id in enumerate(df_score['Subject ID'].values):
                score_mapping[score_name][str(subject_id)] = df_score[score_name][i]
    return score_mapping

def combine_with_score(df1, df_score, b_display_mapping=False):
    score_mapping = get_score_mapping(df_score)
    if b_display_mapping:
        for k,v in score_mapping.items():
            print(k, '->', v, '\n')
    df_combined = df1.copy()
    df_combined = df_combined.reset_index()
    for score_name in score_mapping:
        participants = list(score_mapping[score_name].keys())
        df_combined[score_name] = df_combined['Subject ID'].map(lambda x: score_mapping[score_name][str(x)] if str(x) in participants else np.NaN)
    df_combined = df_combined.set_index(['Subject ID','Date'])
    return df_combined

def concatenate_mood_fitbit_minute(df1, df_fitbit, participant):
    #Concatenate df1, df_fitbit using outer join by 'Date'
    #The start date is the first date of df1
    df1 = df1.reset_index()
    df_fitbit = df_fitbit.reset_index()
    df1['Date'] = pd.to_datetime(df1['Date'])
    df_fitbit['Date'] = pd.to_datetime(df_fitbit['Date'])
    start_date = df1['Date'].min()
    print('combined start_date =', start_date)
    df1 = df1[df1['Date'] >= start_date]
    df_fitbit = df_fitbit[df_fitbit['Date'] >= start_date]
    df1 = df1.set_index(['Date'])
    df_fitbit = df_fitbit.set_index(['Date'])
    df_combined = pd.concat([df1, df_fitbit], join='outer', axis=1)  # union on index
    df_combined = df_combined.loc[:,~df_combined.columns.duplicated()].reset_index()
    df_combined['Subject ID'] = participant
    df_combined['Date'] = pd.to_datetime(df_combined['Date'], format='%Y-%m-%d %H:%M:%S')
    df_combined = df_combined.set_index(['Subject ID', 'Date'])    
    return df_combined

def convert_column_to_binary(df, column_name):
    #Set to 0 if value = 0, and 1 if value > 0
    df[column_name] = df[column_name].apply(lambda x: 1 if x > 0 else 0)
    return df

def plot_time_series(df, y_name, subject_names):
    #Plot time series for subjects in subject_names
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    #Check if the subject name exists in the data
    participants = data_utils.get_subject_ids(df)
    for subject_id in subject_names:
        if subject_id not in participants:
            print('cannot find Participant ID =', subject_id)
            subject_names.remove(subject_id)
    if len(subject_names) > 1:
        df = df.loc[subject_names,:]
    subjects = list(df.index)
    #Plot time series, with mean per day
    plt.figure(figsize=(12,2))
    all_steps = collections.defaultdict(list)
    previous = ''
    for value in subjects:
        subject_id = value[0]
        date = value[1]
        step = df.loc[subject_id, date][y_name]        
        if previous != subject_id:
            previous = subject_id
            steps = list(df.loc[subject_id,:][y_name])
            for pos, num in enumerate(steps):
                all_steps[pos].append(num)
            label = subject_id if len(subject_names) < 10 else ''
            plt.plot(steps, label=label)
    xs = list(all_steps.keys())
    ys = [np.mean(steps) for steps in list(all_steps.values())]
    plt.plot(xs, ys, ls=':', lw=2, label='mean', color='black')
    plt.legend(loc=2, fontsize=8)
    plt.ylabel(y_name)
    plt.xlabel('Number of days')
    plt.title(y_name)

def check_stationary(df, names):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    for name in names:
        adf_test = adfuller(df[name])
        adf_stat = adf_test[0]
        p_value  = adf_test[1]
        critical_values = adf_test[4]
        print('Check if %s is stationary?' % name)
        print('ADF statistic = %f' % adf_stat)
        print('p-value = %f' % p_value)
        print('critical values:')
        for key, value in critical_values.items():
            print('\t%s: %.3f' % (key, value))
        # Ho: time series is non-stationary
        if adf_stat < critical_values['5%']: # reject Ho
            print('%s time series is stationary' % name)             
        else: # failed to reject Ho
            print('%s time series is non-stationary' % name)
        print()

def impute(df, methods):
    #Impute missing data using method = IterativeImputer, KNNImputer:2 (using 2 neighbors)...
    #methods is a dictionary of key, value, where key contains the method 
    #and value contains all the columns where the method is applied.
    df = df.replace({True: 1, False: 0})   
    df_group = df.groupby(by='Subject ID')
    participant_ids = list(set(df.index.get_level_values(0)))
    frames = []
    #Impute for each participant
    for participant_id in participant_ids:
        df_i = df_group.get_group(participant_id)        
        df_i = df_i.reset_index()
        df_i = df_i.set_index(['Date'])
        for method, columns in methods.items():
            if method == 'IterativeImputer':
                imputer = IterativeImputer()
            elif method.find('KNNImputer:') >= 0:
                method_array = method.split(':')
                n_neighbors = int(method_array[1])
                imputer = KNNImputer(n_neighbors=n_neighbors, copy=False)
            else:
                #sklearn KNNImputer default uses 5 neighbors
                imputer = KNNImputer(copy=False)                
            X = df_i[columns].copy()
            X = X.reset_index()
            #sklearn imputer returns an array, so store and add back column names
            all_column_names = X.columns
            X_imputed = pd.DataFrame(imputer.fit_transform(X[columns]))           
            if X_imputed.shape[1] != X[columns].shape[1]:
                print('cannot impute participant %s' % participant_id)
                continue           
            X[columns] = X_imputed
            X.columns = all_column_names
            X = X.set_index(['Date'])           
            df_i[columns] = X
        df_i = df_i.reset_index().set_index(['Subject ID', 'Date'])        
        frames.append(df_i)
    df = pd.concat(frames)
    df = df.dropna()
    df = df.sort_index(level='Subject ID')
    return df
        
def get_pacf(df, names):   
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df_group = df.groupby(by='Subject ID')
    participant_ids = list(set(df.index.get_level_values(0)))
    
    #Compute the pacf values for each participant, and aggregate the pacf values
    for name in names:
        pacf_averages = collections.defaultdict(list)
        confidence_intervals = collections.defaultdict(list)
        
        for index, participant_id in enumerate(participant_ids):
            df = df_group.get_group(participant_id)
            current_threshold_lags = math.floor(df.shape[0]//2)-1
            max_lags = 15
            if (current_threshold_lags > max_lags) and (int(df[name].mean()) > 0):       
                pacf_values, confidence_interval = pacf(df[name], nlags=max_lags, alpha=0.05)
                lags = np.arange(max_lags+1)
                #Plot for sanity test
                #plot_pacf(df[name], lags=max_lags)
                #plt.title(participant_id + ' ' + name + ' Partial Autocorrelation')
                for i, lag in enumerate(lags):
                    pacf_averages[lag].append(pacf_values[i])
                    confidence_intervals[lag].append(confidence_interval[i])
                    
        lags = pacf_averages.keys()
        pacf_values = np.array(list(pacf_averages.values()))
        if len(pacf_values) > 0:
            plt.figure()
            pacf_values = pacf_values.mean(axis=1)
            confidence_interval = np.array(list(confidence_intervals.values())).mean(axis=1)            
            plt.fill_between(lags,
                             confidence_interval[:, 0] - pacf_values,
                             confidence_interval[:, 1] - pacf_values, alpha=0.25)
            plt.scatter(lags, pacf_values, color='C0', zorder=2)
            plt.bar(lags, pacf_values, width=0.1, color='black', zorder=1)
            plt.axhline(y=0)
            title = 'Average ' + name + ' Partial Autocorrelation'
            plt.title(title)
        else:
            print('found no pacf values for', name)
            
def get_pearsonr(df, name, lagged_name, max_lag):
    print('correlation between %s and lagged %s:' % (name, lagged_name))
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    for lag in range(1,max_lag):
        series1 = df[name].iloc[lag:]
        series2 = df[lagged_name].iloc[:-lag]
        results = pearsonr(series1, series2)
        corr = results[0]
        p_value = results[1]
        detail = ''
        if p_value < 0.05:
            detail = 'significant'
        print('lag=%d   corr=%.5f   p_value=%s \t%s' % (lag, corr, p_value, detail))
    print()

def compute_VAR(df, names, max_lag):
    df = df.dropna()
    df = df.reset_index()
    df = df.replace({True: 1, False: 0})
    df = df[names]
    model = VAR(df)
    results = model.fit(maxlags=max_lag)
    print(results.summary())

def get_correlations(df):
    df = df.replace({True: 1, False: 0})    
    df_correlations = df.corr()
    figsize = (8,7)
    if df.shape[1] > 12:
        figsize = (11,9)
    plt.figure(figsize=figsize)
    sn.heatmap(df_correlations, cmap=cm.seismic, annot=True, vmin=-1, vmax=1)
    plt.show()
    return df_correlations

def get_correlations_average_within_participant(df, behaviors, activities, b_plot=False):
    df = df.replace({True: 1, False: 0})
    participants = list(set([p for (p, date) in df.index.values]))
    df=df.rename(columns={'Fitbit Step Count' : 'Daily Steps', 'Fitbit Minutes Worn' : 'Minutes of Activity'})
    columns = list(df.columns)
    valid_columns = []
    for col in columns:
        if (col in activities) or (col in behaviors):
            valid_columns.append(col)     
    df = df[valid_columns]   
    df_correlation_averages=pd.DataFrame(np.zeros((len(valid_columns),len(activities))), index=valid_columns, columns=activities)
    correlations_dict = {}
    group = df.groupby(by='Subject ID')
    for participant in participants:
        df_individual = group.get_group(participant)
        df_correlations = df_individual.corr()
        correlations_dict[participant] = df_correlations        
    for activity in activities:
        activity_averages = []
        for k, df_corr in correlations_dict.items():
            activity_averages.append(df_corr[activity])
        activity_averages = np.array(activity_averages)
        averages = np.nanmean(np.array(activity_averages), axis=0)
        df_correlation_averages[activity] = np.nanmean(np.array(activity_averages), axis=0)
    if b_plot:
        plt.figure(figsize=(2,9))
        sn.heatmap(df_correlation_averages, cmap=cm.seismic, annot=True, vmin=-1, vmax=1)
        plt.show()
    return df_correlation_averages

def get_data_yields(df, td, th, behaviors, yields):
    columns = list(df.columns)
    valid_columns = []
    for col in columns:
        if (col in yields) or (col in behaviors):
            valid_columns.append(col)    
    df_yields=pd.DataFrame(np.zeros((len(valid_columns),len(yields))), index=valid_columns, columns=yields)
    for behavior in behaviors:
        if behavior in df:
            df_behavior = df[['Fitbit Step Count', 'Fitbit Minutes Worn', behavior]]            
            table_days, table_participants, _ = get_available_data([td], [th], df_behavior)
            df_yields.loc[behavior,yields[0]] = table_participants.values[0][0]
            df_yields.loc[behavior,yields[1]] = table_days.values[0][0]
    return df_yields.astype(int)

def update_name(old_name):
    return str(old_name).replace(' ', '_').lower()
    
def perform_gee(df, y_name, x_array, groups_name, fixed_effect='',
                family='Gaussian', cov_struct='Exchangeable', x_lim=None):
    #Perform GEE Linear Regression (additional options are fixed_effect, family and cov_struct)
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df = df.reset_index()

    if family == 'Gaussian':
        fam = sm.families.Gaussian()
    elif family == 'Poisson':
        fam = sm.families.Poisson()
    elif family == 'Binomial':
        fam = sm.families.Binomial()
    else:
        print('undefined family = %s, family is set to Gaussian' % family)
        family = 'Gaussian'
        fam = sm.families.Gaussian()

    if cov_struct == 'Independence':
        cov = sm.cov_struct.Independence() 
    elif cov_struct == 'Exchangeable':
        cov = sm.cov_struct.Exchangeable()
    else:
        print('undefined cov_struct = %s, cov_struct is set to Exchangeable' % cov_struct)
        cov_struct = 'Exchangeable'
        cov = sm.cov_struct.Exchangeable()

    df.columns = [update_name(x) for x in df.columns]
    y_name = update_name(y_name)
    groups = update_name(groups_name)

    equation = y_name + " ~ "
    for index, covariate in enumerate(x_array):
        covariate = update_name(covariate)
        if index == 0:
            equation += covariate
        else:
            equation += ' + ' + covariate

    figsize = (10,3)
    if fixed_effect != '':
        fixed_effect = update_name(fixed_effect)
        equation += " + C(" + fixed_effect + ")"
        figsize = (10,14)

    model = smf.gee(equation, data=df, groups=groups, family=fam, cov_struct=cov)    

    results = model.fit()    
    (QIC, QICu) = results.qic(results.scale)
    print('%s =\n%s\n%s\n%s %s QIC = %.4f, QICu = %.4f\n\n\n' % (
           y_name, results.summary(), cov.summary(), family, cov_struct, QIC, QICu))
        
    df_coef = results.params.to_frame().rename(columns={0: 'coef'})
    figsize_gee = figsize
    if df_coef.shape[0] < 3:
        figsize_gee = (10, max(1, df_coef.shape[0]//2))   
    ax = df_coef.plot.barh(figsize=figsize_gee)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    title = y_name + ' using GEE ' + family + ' ' + cov_struct
    plt.title(title)

def perform_linear_regression(df, y_name, b_fixed_effect=False, x_lim=None):
    #Perform three linear regressions: OLS, GEE, Mixed Linear Model
    
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    df = df.reset_index()

    plt.figure(figsize=(5,4))
    plt.hist(df['Fitbit Step Count']) 
    plt.xlabel('Fitbit Step Count')
    plt.title('Fitbit Step Count histogram')

    df['Fitbit Log Step Count'] = np.log(df['Fitbit Step Count'] + 1e-7)

    df.columns = [update_name(x) for x in df.columns]
    y_display = y_name
    y_name = update_name(y_name)
    
    equation  = " ~ busy + committed + rested + energetic"
    equation += " + fatigued + happy + relaxed + sad + stressed + tense"

    figsize = (10,3)
    if b_fixed_effect:
        #Add unconditional fixed effect on Subject ID
        equation += " + C(subject_id)"
        figsize = (10,14)
              
    model = y_name + equation
    
    mod  = smf.ols(model, data=df)
    res0 = mod.fit()
    print('%s =\n%s\n\n\n' % (y_display, res0.summary()))

    fam_display = 'Gaussian'
    cov_display = 'Exchangeable'
    cov = sm.cov_struct.Exchangeable()
    mod = smf.gee(model, "subject_id", data=df, cov_struct=cov)    
    res1 = mod.fit()    
    (QIC, QICu) = res1.qic(res1.scale)
    print('%s =\n%s\n%s\n%s %s QIC = %.4f, QICu = %.4f\n\n\n' % (
           y_name, res1.summary(), cov.summary(), fam_display, cov_display, QIC, QICu))

    mod = smf.mixedlm(model, df, groups=df['subject_id'])    
    res2 = mod.fit()
    print('%s =\n%s\n\n\n' % (y_display, res2.summary()))
    
    df_coef = res0.params.to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    plt.title(y_display + ' using OLS')
    
    df_coef = res1.params.to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    plt.title(y_display + ' using GEE Regression')

    df_coef = res2.params[:len(res2.params)-1].to_frame().rename(columns={0: 'coef'})
    ax = df_coef.plot.barh(figsize=figsize)
    ax.axvline(0, color='black', lw=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    plt.grid(True)
    plt.title(y_display + ' using Mixed Linear Model Regression')

def split_data(df, y_name, b_split_per_participant, split_percent=0.8, b_display=True):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    
    if b_split_per_participant:
        df = df.drop(columns=['Fitbit Minutes Worn'])       
        X = df.drop(columns=y_name)
        y = df[y_name]        
        X_train = X.groupby('Subject ID').apply(lambda x: x.head(round(split_percent * len(x))))
        y_train = y.groupby('Subject ID').apply(lambda x: x.head(round(split_percent * len(x))))
        X_test  = X.groupby('Subject ID').apply(lambda x: x.tail(len(x) - round(split_percent * len(x))))
        y_test  = y.groupby('Subject ID').apply(lambda x: x.tail(len(x) - round(split_percent * len(x))))
        X_train = X_train.values
        y_train = y_train.values
    else:
        df = df.reset_index()
        df = df.drop(columns=['Fitbit Minutes Worn', 'Subject ID', 'Date'])
        X = df.drop(columns=y_name).values
        y = df[y_name].values
        split = int(len(X) * split_percent)
        X_train = X[:split]
        y_train = y[:split]
        X_test  = X[split:]
        y_test  = y[split:]

    if b_display:
        print('Classification for ' + y_name)
        print('\ntrain split   = {}%'.format(int(split_percent*100)))
        print('X_train.shape =', X_train.shape)
        print('y_train.shape =', y_train.shape)
        print('X_test.shape  =', X_test.shape)
        print('y_test.shape  =', y_test.shape)
        
    return (X_train, y_train, X_test, y_test)

def perform_classification(df, y_name, b_split_per_participant):    
    #Split the data and targets into training/testing sets
    split_data_tuple = split_data(df, y_name, b_split_per_participant=True)
    (X_train, y_train, X_test, y_test) = split_data_tuple
    
    model = LogisticRegression(solver='lbfgs', random_state=0)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print('\nmodel =', model)
    print('\ntrain accuracy = %s' % accuracy_score(y_train, model.predict(X_train)))
    print('test  accuracy = %s\n' % accuracy_score(y_test,  model.predict(X_test)))

def apply_threshold(df, threshold):
    #Extract data rows with Fitbit Minutes Worn > threshold
    #Add new column for Fitbit Minutes Worn > threshold in minutes, for example: 'Worn > 60 minutes'
    name = 'Worn > ' + str(threshold) + ' minutes'
    df[name] = df['Fitbit Minutes Worn'].apply(lambda x: 1 if x > threshold else 0)    
    df = df[df[name] == 1]
    return df

def analyze_fitbit_worn_threshold(df, thresholds):
    #Compute and plot Number of Participant vs. Minutes worn per day (steps > 0)
    number_of_participants = []
    steps_per_day = []
    for threshold in thresholds:
        name = 'Worn > ' + str(threshold) + ' minutes'
        df_threshold = df.copy()
        df_threshold = apply_threshold(df_threshold, threshold)
        df_threshold = df_threshold[df_threshold['Fitbit Step Count'] == 0]
        participants = list(set([x[0] for x in df_threshold['Fitbit Step Count'].index.values]))
        n_participants = len(participants)
        number_of_participants.append(n_participants)
        print('%s\t(steps = 0)\tnumber of participants = %d' % (name, n_participants))
        
    plt.figure(figsize=(4,3))
    plt.plot(thresholds, number_of_participants)
    plt.xlabel('Minutes worn per day (steps = 0)')
    plt.ylabel('Number of participants')
    plt.title('Fitbit Minutes Worn (steps = 0)')

def get_xs_and_ys_average(data_dict):    
    #xs = data_dict keys
    #ys_mean = data_dict values average
    #ys_ci_upper = data_dict values upper 95% confidence interval
    #ys_ci_lower = data_dict values lower 95% confidence interval
    xs = []
    ys_mean = []
    ys_ci_upper = []
    ys_ci_lower = []
    for key, y_values in data_dict.items():              
        y_data = [y for y in y_values if str(y).lower() != 'nan']
        if len(y_data) > 0:
            xs.append(key)
            mean_data = np.mean(y_data)
            ys_mean.append(mean_data)
            ys_ci_upper.append(mean_data + 1.96 * np.std(y_data) / np.sqrt(len(y_data)))
            ys_ci_lower.append(mean_data - 1.96 * np.std(y_data) / np.sqrt(len(y_data)))
    return xs, ys_mean, ys_ci_upper, ys_ci_lower

def get_fitbit_step_per_day_of_week(df, participants, threshold, y_max=None):
    #Compute and plot Mean Fitbit Step Count per Day of Week (worn threshold is in minutes)
    group = df.groupby(by='Subject ID')
    days_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(5,4))
    average_steps_all = collections.defaultdict(list)
    for participant_id in participants:
        df_individual = group.get_group(participant_id)
        df_individual = df_individual.reset_index()    
        df_individual['Date'] = pd.to_datetime(df_individual['Date'])
        df_individual['Day of Week'] = df_individual['Date'].dt.day_name()
        day_group = df_individual.groupby(by='Day of Week')
        day_names_individual = [d for d in days_name if d in set(df_individual['Day of Week'].values)]
        for day_name in day_names_individual:            
            df_day = day_group.get_group(day_name).copy()
            df_day = apply_threshold(df_day, threshold)
            mean_individual = df_day['Fitbit Step Count'].mean()
            average_steps_all[day_name].append(mean_individual)
    xs, ys_mean, ys_ci_upper, ys_ci_lower = get_xs_and_ys_average(average_steps_all)   
    plt.plot(xs, ys_mean, lw=2, label='mean', color='blue')
    plt.fill_between(xs, ys_ci_lower, ys_ci_upper, color='b', alpha=.1, label='95% ci')
    label = 'Mean Step Count (worn > ' + str(threshold) + ' minutes)'
    plt.ylabel(label)
    title = 'Mean Fitbit Step Count (worn > ' + str(threshold) + ' minutes)\nper Participant'
    title += ' (number of participants = ' + str(len(participants)) + ')'
    plt.title(title)
    plt.xticks(rotation=45)
    if y_max == None:
        y_max = 30000  
    plt.ylim((0,y_max))
    plt.legend(loc=2, fontsize=8)
        
def get_fitbit_step_per_mood(df, participants, mood, threshold, y_max=None, x_lim=None):
    #Compute and plot Mean Fitbit Step Count per Mood (worn threshold is in minutes)
    threshold = min(threshold, 24*60)
    group = df.groupby(by='Subject ID')
    plt.figure(figsize=(5,4))
    average_steps_all = collections.defaultdict(list)
    data_plot = collections.defaultdict(list)
    for participant_id in participants:
        df_individual = group.get_group(participant_id)
        df_individual = df_individual.reset_index()        
        df_individual = apply_threshold(df_individual, threshold)
        for i, x in enumerate(df_individual[mood]):
            if str(x).lower() != 'nan':
                x = round(x)
                y = df_individual['Fitbit Step Count'].values[i]
                average_steps_all[x].append(y)            
    average_steps_all = collections.OrderedDict(sorted(average_steps_all.items()))
    xs, ys_mean, ys_ci_upper, ys_ci_lower = get_xs_and_ys_average(average_steps_all)
    plt.plot(xs, ys_mean, lw=2, label='mean', color='blue')
    plt.fill_between(xs, ys_ci_lower, ys_ci_upper, color='b', alpha=.1, label='95% ci')
    label = 'Mean Step Count (worn > ' + str(threshold) + ' minutes)'
    plt.ylabel(label)
    plt.xlabel(mood)
    title = 'Mean Fitbit Step Count (worn > ' + str(threshold) + ' minutes)\nper Participant'
    title += ' (number of participants = ' + str(len(participants)) + ')'
    plt.title(title)
    if y_max == None:
        y_max = 30000
    if x_lim == None:
        x_lim = (1,5) 
    plt.ylim((0,y_max))
    plt.xlim(x_lim[0], x_lim[1])
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.legend(loc=2, fontsize=8)

def get_available_data(tD, tH, df): 
    df_days=pd.DataFrame(np.zeros((len(tD),len(tH))),index=["tD=%d"%x for x in tD],
                         columns=["tH=%d"%x for x in tH])
    df_participants=pd.DataFrame(np.zeros((len(tD),len(tH))),index=["tD=%d"%x for x in tD], 
                                 columns=["tH=%d"%x for x in tH])
    participants = list(set([p for (p, date) in df.index.values]))
    available_dict = {}
    for th in tH:
        key = 'worn>'+str(th)
        df_all = df.copy()
        df_all[key] = df_all['Fitbit Minutes Worn'][df_all['Fitbit Minutes Worn'] > 60*th]
        group = df_all.groupby(by='Subject ID')      
        for td in tD:           
            count = 0
            count_days = 0
            count_participants = 0
            valid_participants = []
            for participant in participants:
                df_individual = group.get_group(participant)
                df_individual = df_individual.dropna()
                count = df_individual[key].shape[0]                
                if count > td:
                    count_days += count
                    count_participants += 1
                    valid_participants.append(participant)
            df_days.loc['tD='+str(td), 'tH='+str(th)] = int(count_days)
            df_participants.loc['tD='+str(td), 'tH='+str(th)] = int(count_participants)
            index_names = df_all.index.names
            df_all = df_all.reset_index()
            df_all = df_all[df_all['Subject ID'].isin(valid_participants)]
            df_all = df_all.set_index(index_names)
            available_dict[(td, th)] = df_all.dropna()         
    return df_days.astype(int), df_participants.astype(int), available_dict

