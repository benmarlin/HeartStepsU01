import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from textwrap import wrap
import collections
import timeit

import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.infer import MCMC, NUTS, Predictive

fig_size = (7,5)
build_time = collections.defaultdict(list)
def save_build_time(title, start_time):
    duration = (int)(timeit.default_timer() - start_time)
    print('duration =', duration, 'seconds')
    build_time[title].append(duration)

def build_df(filename):
    df = pd.read_csv(filename).dropna()
    df['Subject ID'] = df['Subject ID'].astype(str)
    df = df.set_index(['Subject ID', 'Date'])
    return df

def plot_data_regression_lines(samples, title, df, x_name, y_name, x_lim=None, y_lim=None, b_show=True):
    #Plot data
    data_x = df[x_name].values
    data_y = df[y_name].values
    plt.figure(figsize=fig_size)
    plt.scatter(data_x, data_y, s=10, alpha=.5, marker='o', color='#1f77b4', label='data')    
    #Plot regression lines
    xs = np.linspace(data_x.min(), data_x.max(), 100)
    a_samples  = samples['intercept']
    b1_samples = samples[x_name]
    a_mean     = jnp.mean(a_samples)
    b1_mean    = jnp.mean(b1_samples)
    n_samples  = 1000
    for i in range(n_samples):
        plt.plot(xs, a_samples[i] + b1_samples[i] * xs, color='blue', alpha=0.005)
    plt.plot(xs, a_mean + b1_mean * xs, color='blue', lw=2, label='fitted')
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    if x_lim != None:
        plt.ylim(y_lim)
    if x_lim != None:
        plt.xlim(x_lim)
    plot_title = "\n".join(wrap(title, 80))
    plot_title += ':\ndata and ' + str(n_samples) + ' fitted regression lines'
    plt.title(plot_title)
    plt.gcf().tight_layout()
    plt.legend(loc=4)
    if b_show:
        plt.show()
    else:
        plt.savefig(title + '_regression.png')
    plt.close('all')

def regression_model2(df, y_name, x_names):
    xs = jnp.array(df[x_names].values)
    y_obs = df[y_name].values   
    mu = numpyro.sample('intercept', dist.Normal(0., 1000))
    M = xs.shape[1]  
    for i in range(M):
        b_name = x_names[i]
        bi = numpyro.sample(b_name, dist.Normal(0., 10.))
        bxi = bi * xs[:,i]
        mu = mu +  bxi        
    log_sigma = numpyro.sample('log_sigma', dist.Normal(0., 10.))    
    numpyro.sample(y_name, dist.Normal(mu, jnp.exp(log_sigma)), obs=y_obs)

def fit_simple_regression_model_numpyro(df_data, y_name, x_names, x_lim=None, y_lim=None, y_mean_lim=None, b_show=True):
    mcmcs = []
    for x_name in x_names:
        start_time_simple_regression = timeit.default_timer()
        title = y_name + ' vs ' + x_name + ' (regression model)'
        print('fitting for %s...' % title)

        #Fit model
        rng_key = random.PRNGKey(0)
        kernel = NUTS(regression_model2)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(rng_key, df=df_data, y_name=y_name, x_names=[x_name])
        
        #Display summary
        print('\nsummary for %s =' % title)
        mcmc.print_summary()
        samples = mcmc.get_samples()                
        samples['sigma'] = jnp.exp(samples['log_sigma'])    
        ss = samples['sigma']
        print('sigma mean = %.2f\tstd = %.2f\tmedian = %.2f\tQ5%% = %.2f\tQ95%% = %.2f' % (
              np.mean(ss), np.std(ss), np.median(ss), np.quantile(ss, 0.05, axis=0), np.quantile(ss, 0.95, axis=0)))     
        save_build_time(title, start_time_simple_regression)
        
        #Plot
        plot_data_regression_lines(samples, title, df_data, x_name, y_name, x_lim, y_lim, b_show)
        mcmcs.append(mcmc)
        print('\n\n\n')
    return mcmcs   
    
def fit_regression_model_numpyro(df_data, y_name, x_names, y_mean_lim=None, b_show=True):
    start_time_regression = timeit.default_timer()
    title = y_name + ' vs ' + str(x_names) + ' (regression model)'
    print('fitting for %s...' % title) 

    #Fit model
    rng_key = random.PRNGKey(0)
    kernel = NUTS(regression_model2)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(rng_key, df=df_data, y_name=y_name, x_names=x_names)

    #Display summary
    print('\nsummary for %s =' % title)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    samples['sigma'] = jnp.exp(samples['log_sigma']) 
    ss = samples['sigma']
    print('sigma mean = %.2f\tstd = %.2f\tmedian = %.2f\tQ5%% = %.2f\tQ95%% = %.2f' % (
          np.mean(ss), np.std(ss), np.median(ss), np.quantile(ss, 0.05, axis=0), np.quantile(ss, 0.95, axis=0)))
    save_build_time(title, start_time_regression)
    print('\n\n\n')
    return mcmc

if __name__ == '__main__':
    filename1 = 'df_mood_fitbit_daily.csv'  #Replace with desired csv file
    filename2 = 'df_imputed_105_10Min.csv'  #Replace with desired csv file      
    chosen_df1 = build_df(filename1)
    chosen_df2 = build_df(filename2)
    print('chosen_df1.shape =', chosen_df1.shape)
    print('chosen_df2.shape =', chosen_df2.shape)
    print()
    b_show = False
    n_repeats = 1 #5 1

    for repeat in range(n_repeats):
        #Analysis with Daily Metrics Data
        x_names = ['Committed', 'Busy', 'Rested']
        y_name = 'Fitbit Step Count'
        mcmcs = fit_simple_regression_model_numpyro(chosen_df1 , y_name, x_names, y_lim=(-5000, 35000), x_lim=(0.5, 5.5),
                                                    b_show=b_show)
        mcmc = fit_regression_model_numpyro(chosen_df1 , y_name, x_names, b_show=b_show)
        print('build_time numpyro (repeat=%d) = %s\n\n\n' % (repeat, dict(build_time)))    
        pd.DataFrame.from_dict(data=build_time, orient='index').to_csv('build_time_numpyro.csv', header=False)

        #Analysis with Fitbit Data Per Minute
        participants = ['105']
        y_name = 'steps'
        x_names = ['Committed', 'Busy', 'Rested']
        mcmcs = fit_simple_regression_model_numpyro(chosen_df2, y_name, x_names, b_show=b_show)
        mcmc = fit_regression_model_numpyro(chosen_df2, y_name, x_names, b_show=b_show)
        print('build_time numpyro (repeat=%d) = %s\n\n\n' % (repeat, dict(build_time)))
        pd.DataFrame.from_dict(data=build_time, orient='index').to_csv('build_time_numpyro.csv', header=False)
        
    print('finished!')

