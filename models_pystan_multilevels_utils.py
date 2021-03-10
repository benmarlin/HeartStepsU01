import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pystan
import os.path
from os import path
import pickle
import collections
import timeit

build_time = collections.defaultdict(list)
def save_build_time(title, start_time):
    duration_filename = 'build_time_pystan_multilevel.csv'
    duration = (int)(timeit.default_timer() - start_time)
    print('duration =', duration, 'seconds')
    build_time[title].append(duration)
    pd.DataFrame.from_dict(data=build_time, orient='index').to_csv(duration_filename, header=False)
    
def load_model(model_name, model_code, b_load_existing):    
    pickle_name = model_name.replace(' ','_') + '.pkl'
    pickle_name = os.path.join('models/', pickle_name)                     
    b_exist = path.exists(pickle_name)
    if not b_exist:
        print(pickle_name + ' file does not exist')
        b_load_existing = False            
    if b_load_existing == False:
        print('compiling Stan model and saving to %s...' % pickle_name)
        stan_model = pystan.StanModel(model_code=model_code)
        with open(pickle_name, 'wb') as f:
            pickle.dump(stan_model, f)
    else:
        print('loading %s...' % pickle_name)
        stan_model = pickle.load(open(pickle_name, 'rb'))
    return stan_model

def print_summary(fit, b_display=True):
    summary_dict = fit.summary()
    df_summary = pd.DataFrame(summary_dict['summary'], columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])
    if b_display:
        print(df_summary) 
    return df_summary

def extract_posterior(model_fit, b_display=False):
    samples = model_fit.extract(permuted=True)
    if b_display:
        for k,v in samples.items():
            display = '{:<10}\t-> {} \t-> mean = {}'.format(k,v.shape,v.mean())
            print(display)
    return samples

if __name__ == '__main__':

    np.random.seed(0)
    figsize = (5,4)
    figsize_compare = (8,8)
    plot_alpha = 0.5
    b_load_existing = True
    n_chains = 1
    n_iters  = 1000
    length_per_group = 500
    y_name = 'steps'
    log_y_name = 'log_'+y_name
    subject_id_name = 'subject_id'
    x_name = 'heart_rate'    
    x_min_value = 0
    x_max_value = 150
    print('length_per_group =', length_per_group)
    
    participants = ['105', '69', '47', '80']    
    COLORS = ['blue', 'magenta', 'purple', 'orangered']
    ROWS = 2
    COLUMNS = 2
    subject_indices = range(len(participants))
    assert(len(participants) == len(COLORS))
    
    data_df = pd.read_csv('./data/fitbit_per_minute.csv', low_memory=False)

    data_df = data_df.dropna()    
    groups = data_df.groupby(by='Subject ID')
    group1 = groups.get_group(participants[0])[:length_per_group]
    group2 = groups.get_group(participants[1])[:length_per_group]
    group3 = groups.get_group(participants[2])[:length_per_group]
    group4 = groups.get_group(participants[3])[:length_per_group]

    data_df = pd.concat([group1, group2, group3, group4])
    data_df = data_df.rename(columns={'Subject ID' : 'subject_id'})
    data_df['subject_id'] = data_df['subject_id'].map(lambda x: str(x))    
    participants_dict = { str(x) : i for i,x in enumerate(participants) }
    data_df[subject_id_name] = data_df[subject_id_name].map(lambda x: participants_dict[str(x)])
    log_steps = np.log(data_df[y_name] + 0.1).values
    data_df[log_y_name] = log_steps
    x_values = data_df[x_name].values
    subject_id = data_df[subject_id_name].values
    unique_subject_ids = data_df[subject_id_name].unique()
    subject_id_lookup = dict(zip(unique_subject_ids, range(len(unique_subject_ids))))    
    print('data_df.shape =', data_df.shape)
    print(list(data_df.columns))
    print('\n\n\n')

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    plot_data = data_df[y_name].apply(lambda x: np.log(x+0.1))
    plot_data.hist(bins=25, ax=ax)
    plt.title('histogram of log('+y_name+')')
    plt.ylabel('count')
    plt.xlabel('log('+y_name+')')
    plt.tight_layout()

    #--------------------------------------------------------------
    title = 'pooled'
    start_time_pooled_model = timeit.default_timer()
    
    pooled_model = """
    data {
      int<lower=0> N; 
      vector[N] x;
      vector[N] y;
    }
    parameters {
      vector[2] b;
      real<lower=0> sigma;
    } 
    model {
      y ~ normal(b[1] + b[2] * x, sigma);
    }
    
    """

    pooled_data_dict = {'N': len(log_steps),
                        'x': x_values,
                        'y': log_steps}

    stan_model = load_model(title, pooled_model, b_load_existing)  
    pooled_fit = stan_model.sampling(data=pooled_data_dict, iter=n_iters, chains=n_chains)
    print_summary(pooled_fit)
    save_build_time(title, start_time_pooled_model)
    pooled_sample = extract_posterior(pooled_fit)
    b0, b1 = pooled_sample['b'].T.mean(1)
    print('b0 =', b0)
    print('b1 =', b1)    

    plt.figure(figsize=figsize)
    label = 'log('+y_name+') vs ' + x_name
    plt.scatter(data_df[x_name], np.log(data_df[y_name]+0.1), alpha=plot_alpha, label=label)
    xs = np.linspace(x_min_value, x_max_value)
    plt.plot(xs, b0+b1*xs, ls='--', color='green', label='fitted')
    plt.xlabel(x_name)
    plt.ylabel(log_y_name)
    plt.title('1a. pooled (same for all)')
    plt.legend(loc=2, fontsize=8)
    print('\n\n\n')
        
    #--------------------------------------------------------------
    title = 'unpooled'
    start_time_unpooled_model = timeit.default_timer()
    
    unpooled_model = """
    data {
      int<lower=0> N; 
      int subject_id[N];
      vector[N] x;
      vector[N] y;
    } 
    parameters {
      vector[4] a;
      real b;
      real<lower=0,upper=100> sigma;
    } 
    transformed parameters {
      vector[N] y_hat;
      for (i in 1:N)
        y_hat[i] <- b * x[i] + a[subject_id[i]];
    }
    model {
      y ~ normal(y_hat, sigma);
    }

    """

    unpooled_data_dict = {'N': len(log_steps),
                          'subject_id': subject_id+1,
                          'x': x_values,
                          'y': log_steps}

    stan_model = load_model(title, unpooled_model, b_load_existing)
    unpooled_fit = stan_model.sampling(data=unpooled_data_dict, iter=n_iters, chains=n_chains)
    print_summary(unpooled_fit)
    save_build_time(title, start_time_unpooled_model)
    unpooled_sample = extract_posterior(unpooled_fit)    
    unpooled_estimates = pd.Series(unpooled_fit['a'].mean(0), index=unique_subject_ids)
    
    fig, axes = plt.subplots(ROWS, COLUMNS, figsize=figsize_compare, sharey=True, sharex=True)
    axes = axes.ravel()
    m = unpooled_fit['b'].mean(0)
    for i,p in enumerate(subject_indices):
        y = data_df[log_y_name][data_df[subject_id_name]==p]
        x = data_df[x_name][data_df[subject_id_name]==p]
        label = 'log('+y_name+') vs '+x_name
        axes[i].scatter(x, y, alpha=plot_alpha, label=label)   
        b = unpooled_estimates[p]        
        xs = np.linspace(x_min_value, x_max_value)
        axes[i].plot(xs, m*xs+b, color='blue', label='unpooled (individual)')
        axes[i].plot(xs, b0+b1*xs, ls=':', color='green', label='pooled (same for all)')
        axes[i].set_xlabel(x_name)
        axes[i].set_title('participant '+participants[p])
        if not i%2:
            axes[i].set_ylabel(log_y_name)
    plt.legend(loc=4, fontsize=8)
    plt.suptitle('compare pooled (same for all) and unpooled (individual)')
    print('\n\n\n')
    
    #-----------------------------------------------
    title = 'partial pooled'
    start_time_partial_pooled_model = timeit.default_timer()

    partial_pooled_model = """
    data {
      int<lower=0> N; 
      int subject_id[N];
      vector[N] y;
    } 
    parameters {
      vector[4] a;
      real mu_a;
      real<lower=0,upper=100> sigma_a;
      real<lower=0,upper=100> sigma_y;
    } 
    transformed parameters {
      vector[N] y_hat;
      for (i in 1:N)
        y_hat[i] <- a[subject_id[i]];
    }
    model {
      mu_a ~ normal(0, 1);
      a ~ normal (10 * mu_a, sigma_a);      
      y ~ normal(y_hat, sigma_y);
    }

    """

    partial_pool_data_dict = {'N': len(log_steps),
                              'subject_id': subject_id+1,
                              'y': log_steps}

    stan_model = load_model(title, partial_pooled_model, b_load_existing)
    partial_pool_fit = stan_model.sampling(data=partial_pool_data_dict, iter=n_iters, chains=n_chains)
    print_summary(partial_pool_fit)
    save_build_time(title, start_time_partial_pooled_model)
    partial_sample = extract_posterior(partial_pool_fit)    
    print('\n\n\n')

    #-----------------------------------------------
    title = 'varying intercept'
    start_time_varying_intercept_model = timeit.default_timer()

    varying_intercept_model = """
    data {
      int<lower=0> J; 
      int<lower=0> N; 
      int subject_id[N];
      vector[N] x;
      vector[N] y;
    } 
    parameters {
      vector[J] a;
      real b;
      real mu_a;
      real<lower=0,upper=100> sigma_a;
      real<lower=0,upper=100> sigma_y;
    } 
    transformed parameters {
      vector[N] y_hat;
      for (i in 1:N)
        y_hat[i] <- a[subject_id[i]] + x[i] * b;
    }
    model {
      sigma_a ~ uniform(0, 100);
      a ~ normal (mu_a, sigma_a);
      b ~ normal (0, 1);
      sigma_y ~ uniform(0, 100);      
      y ~ normal(y_hat, sigma_y);
    }
    
    """
    
    n_subject_id = data_df.groupby('subject_id')['subject_id'].count()
    varying_intercept_data_dict = {'N': len(log_steps),
                                  'J': len(n_subject_id),
                                  'subject_id': subject_id+1,
                                  'x': x_values,
                                  'y': log_steps}

    stan_model = load_model(title, varying_intercept_model, b_load_existing)  
    varying_intercept_fit = stan_model.sampling(data=varying_intercept_data_dict, iter=n_iters, chains=n_chains)
    print_summary(varying_intercept_fit)
    save_build_time(title, start_time_varying_intercept_model)
    varying_intercept_samples = extract_posterior(varying_intercept_fit)

    plt.figure(figsize=figsize)
    xs = np.linspace(x_min_value, x_max_value)
    a = varying_intercept_fit['a'].mean(axis=0)
    b = varying_intercept_fit['b'].mean()
    for i,ai in enumerate(a):
        plt.plot(xs, ai + b*xs, ls='-', alpha=plot_alpha, color=COLORS[i], label=participants[i])
    plt.xlabel(x_name)
    plt.ylabel(log_y_name)
    plt.title('varying intercept a')
    plt.legend(loc=2, fontsize=8)

    fig, axes = plt.subplots(ROWS, COLUMNS, figsize=figsize_compare, sharey=True, sharex=True)
    axes = axes.ravel()
    for i,p in enumerate(subject_indices):        
        y = data_df[log_y_name][data_df[subject_id_name]==p]
        x = data_df[x_name][data_df[subject_id_name]==p]
        label =''
        if i == (len(subject_indices)-1):
            label='log steps vs '+x_name
        axes[i].scatter(x, y, alpha=plot_alpha, label=label)        
        xs = np.linspace(x_min_value, x_max_value)
        label =''
        if i == (len(subject_indices)-1):
            label = 'pooled (same for all)'
        axes[i].plot(xs, b0+b1*xs, ls=':', color='green', label=label)
        label =''
        if i == (len(subject_indices)-1):
            label = 'varying intercept'
        axes[i].plot(xs, a[subject_id_lookup[p]]+b*xs, color='orangered', label=label)        
        axes[i].set_xlabel(x_name)
        axes[i].set_title('participant '+participants[p])
        if not i%2:
            axes[i].set_ylabel(log_y_name)
    plt.legend(loc=4, fontsize=8)
    plt.suptitle('compare pooled (same for all) and varying intercept')
    print('\n\n\n')

    #-----------------------------------------------
    title = 'varying slope'
    start_time_varying_slope_model = timeit.default_timer()
  
    varying_slope_model = """
    data {
      int<lower=0> J; 
      int<lower=0> N; 
      int subject_id[N];
      vector[N] x;
      vector[N] y;
    } 
    parameters {
      real a;
      vector[J] b;
      real mu_b;
      real<lower=0,upper=100> sigma_b;
      real<lower=0,upper=100> sigma_y;
    } 
    transformed parameters {
      vector[N] y_hat;
      for (i in 1:N)
        y_hat[i] <- a + x[i] * b[subject_id[i]];
    }
    model {
      sigma_b ~ uniform(0, 100);
      b ~ normal (mu_b, sigma_b);
      a ~ normal (0, 1);
      sigma_y ~ uniform(0, 100);      
      y ~ normal(y_hat, sigma_y);
    }

    """

    varying_slope_data_dict = {'N': len(log_steps),
                              'J': len(n_subject_id),
                              'subject_id': subject_id+1,
                              'x': x_values,
                              'y': log_steps}

    stan_model = load_model(title, varying_slope_model, b_load_existing)  
    varying_slope_fit = stan_model.sampling(data=varying_slope_data_dict, iter=n_iters, chains=n_chains)
    print_summary(varying_slope_fit)
    save_build_time(title, start_time_varying_slope_model)
    varying_slope_samples = extract_posterior(varying_slope_fit)

    plt.figure(figsize=figsize)
    xs = np.linspace(x_min_value, x_max_value)
    a = varying_slope_fit['a'].mean()
    b = varying_slope_fit['b'].mean(axis=0)
    for i, bi in enumerate(b):
        plt.plot(xs, a+bi*xs, ls='-', alpha=plot_alpha, color=COLORS[i], label=participants[i])
    plt.xlabel(x_name)
    plt.ylabel(log_y_name)
    plt.title('varying slope b')
    plt.legend(loc=2, fontsize=8)
    print('\n\n\n')

    #-----------------------------------------------
    title = 'varying intercept slope'
    start_time_varying_intercept_slope_model = timeit.default_timer()
    
    varying_intercept_slope_model = """
    data {
      int<lower=0> N;
      int<lower=0> J;
      vector[N] y;
      vector[N] x;
      int subject_id[N];
    }
    parameters {
      real<lower=0> sigma;
      real<lower=0> sigma_a;
      real<lower=0> sigma_b;
      vector[J] a;
      vector[J] b;
      real mu_a;
      real mu_b;
    }

    model {
      mu_a ~ normal(0, 100);
      mu_b ~ normal(0, 100);
      a ~ normal(mu_a, sigma_a);
      b ~ normal(mu_b, sigma_b);      
      y ~ normal(a[subject_id] + b[subject_id].*x, sigma);
    }
    
    """

    varying_intercept_slope_data_dict = {'N': len(log_steps),
                              'J': len(n_subject_id),
                              'subject_id': subject_id+1,
                              'x': x_values,
                              'y': log_steps}

    stan_model = load_model(title, varying_intercept_slope_model, b_load_existing)  
    varying_intercept_slope_fit = stan_model.sampling(data=varying_intercept_slope_data_dict, iter=n_iters, chains=n_chains)
    print_summary(varying_intercept_slope_fit)
    save_build_time(title, start_time_varying_intercept_slope_model)
    varying_intercept_slope_samples = extract_posterior(varying_intercept_slope_fit)

    plt.figure(figsize=figsize)
    xs = np.linspace(x_min_value, x_max_value)
    a = varying_intercept_slope_samples['a'].mean(axis=0)
    b = varying_intercept_slope_samples['b'].mean(axis=0)    
    i = 0
    for ai,bi in zip(a,b):
        plt.plot(xs, ai+bi*xs, ls='-', alpha=plot_alpha, color=COLORS[i], label=participants[i])
        i += 1
    plt.xlabel(x_name)
    plt.ylabel(log_y_name)
    plt.title('varying intercept a and slope b')
    plt.legend(loc=2, fontsize=8)
    print('\n\n\n')

    print('finished!')
    plt.show()

