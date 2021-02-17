import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.infer import MCMC, NUTS, Predictive


def plot_data_regression_lines(samples, title, data_x, x_name, data_y, y_name, b_show=True): 
    #Plot data
    plt.figure(figsize=(7,5))
    plt.scatter(data_x, data_y, s=10, alpha=.5, marker='o', color='#1f77b4', label='data')    
    #Plot regression lines
    x_min = data_x.min()
    x_max = data_x.max()
    y_min = data_y.min()
    y_max = data_y.max()
    xs = np.linspace(x_min, x_max, 100)
    a_samples  = samples['a']
    b1_samples = samples['b1']
    a_mean     = jnp.mean(a_samples)
    b1_mean    = jnp.mean(b1_samples)
    n_samples  = 1000
    for i in range(n_samples):
        plt.plot(xs, a_samples[i] + b1_samples[i] * xs, color='blue', alpha=0.005)
    plt.plot(xs, a_mean + b1_mean * xs, color='blue', lw=2, label='fitted')
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.ylim(y_min-5000, y_max+3000)
    plt.xlim(x_min-0.5,  x_max+0.5)
    plt.title(title + ': data and ' + str(n_samples) + ' fitted regression lines')
    plt.gcf().tight_layout()
    plt.legend(loc=4)
    if b_show:
        plt.show()    
    plt.close('all')

def plot_posterior_predictive(model, posterior_samples, x_name, y_name, data_x, data_y,
                              test_data_x, test_data_y, y_mean_lim=None, b_show=True):
    rng_key_ = random.PRNGKey(0)        
    predictive = Predictive(model, posterior_samples)
    predictions_training = predictive(rng_key_, xs=data_x)['obs']
    mean_predictions_training = jnp.mean(predictions_training, axis=0)

    rng_key_ = random.PRNGKey(0)
    predictive = Predictive(model, posterior_samples)
    predictions_new = predictive(rng_key_, xs=test_data_x)['obs']
    mean_predictions_new = jnp.mean(predictions_new, axis=0)

    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.plot(mean_predictions_training, label='predicted training mean', color='blue')
    plt.plot(mean_predictions_new, ls='--', label='predicted test mean', color='magenta')
    plt.title(y_name + ' predicted mean')
    plt.ylabel(y_name)
    plt.xlabel('sample')
    if y_mean_lim == None:
        y_mean_lim = (4000, 6500)
    plt.ylim(y_mean_lim)
    plt.tight_layout()
    plt.legend(loc=2, fontsize=10)

    plt.subplot(122)
    plt.scatter(data_x, data_y, s=10, alpha=.5, marker='o', color='#1f77b4', label='training data')
    plt.scatter(test_data_x, test_data_y, s=10, alpha=.5, marker='o', color='#ff7f0e', label='test data')
    plt.plot(data_x, mean_predictions_training, color='blue', label='predicted training mean')
    plt.plot(test_data_x, mean_predictions_new, ls='--', color='magenta', label='predicted test mean')
    plt.title(y_name + ' predicted mean vs ' + x_name)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.tight_layout()
    plt.legend(loc=2, fontsize=10)
    if b_show:
        plt.show() 
    plt.close('all')

def model(xs=None, y_obs=None):
    mu = numpyro.sample('a', dist.Normal(0., 1000))
    M = xs.shape[1]  
    for i in range(M):
        b_name = 'b' + str(i+1)
        bi = numpyro.sample(b_name, dist.Normal(0., 10.))
        bxi = bi * xs[:,i]
        mu = mu +  bxi        
    log_sigma = numpyro.sample('log_sigma', dist.Normal(0., 10.))    
    numpyro.sample('obs', dist.Normal(mu, jnp.exp(log_sigma)), obs=y_obs)

def split_data(df, y_name, b_split_per_participant, split_percent=0.8, b_display=True):
    df = df.dropna()
    df = df.replace({True: 1, False: 0})
    
    if b_split_per_participant:   
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
        df = df.drop(columns=['Subject ID', 'Date'])
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

def fit_simple_regression_model_numpyro(df_data, y_name, x_names, b_show=True, y_mean_lim=None):
    mcmcs = []
    for x_name in x_names:
        title = y_name + ' vs ' + x_name
        print('fitting for %s...' % title)        

        #Split the data and targets into training/testing sets, per participant
        df_to_split = df_data[[y_name, x_name]]
        split_data_tuple = split_data(df_to_split, y_name, b_split_per_participant=True, b_display=False)
        (X_train, y_train, X_test, y_test) = split_data_tuple
        data_x = jnp.array(X_train)
        data_y = jnp.array(y_train)
        test_data_x = jnp.array(X_test)
        test_data_y = jnp.array(y_test)     
        
        #Fit model using NUTS
        rng_key = random.PRNGKey(0)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
        mcmc.run(rng_key, xs=data_x, y_obs=data_y)

        #Display summary
        print('\nsummary for %s =' % title)
        mcmc.print_summary()
        samples = mcmc.get_samples()                
        samples['sigma'] = jnp.exp(samples['log_sigma'])    
        ss = samples['sigma']
        print('sigma mean = %.2f\tstd = %.2f\tmedian = %.2f\tQ5%% = %.2f\tQ95%% = %.2f' % (
              np.mean(ss), np.std(ss), np.median(ss), np.quantile(ss, 0.05, axis=0), np.quantile(ss, 0.95, axis=0)))     

        #Plot
        plot_data_regression_lines(samples, title, data_x, x_name, data_y, y_name, b_show)
        plot_posterior_predictive(model, samples, x_name, y_name, data_x, data_y, test_data_x, test_data_y, y_mean_lim, b_show)
        mcmcs.append(mcmc)
        print('\n\n\n')
    return mcmcs   
    
def fit_regression_model_numpyro(df_data, y_name, x_names, b_show=True):        
    title = y_name + ' vs ' + str(x_names)
    print('fitting for %s...' % title)

    #Split the data and targets into training/testing sets, per participant
    df_to_split = df_data[[y_name] + x_names]
    split_data_tuple = split_data(df_to_split, y_name, b_split_per_participant=True, b_display=False)
    (X_train, y_train, X_test, y_test) = split_data_tuple
    data_xs = jnp.array(X_train)
    data_y  = jnp.array(y_train)
    test_data_xs = jnp.array(X_test)
    test_data_y  = jnp.array(y_test)      
        
    #Fit model using NUTS
    rng_key = random.PRNGKey(0)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    mcmc.run(rng_key, xs=data_xs, y_obs=data_y)

    #Display summary
    print('\nsummary for %s =' % title)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    samples['sigma'] = jnp.exp(samples['log_sigma']) 
    ss = samples['sigma']
    print('sigma mean = %.2f\tstd = %.2f\tmedian = %.2f\tQ5%% = %.2f\tQ95%% = %.2f' % (
          np.mean(ss), np.std(ss), np.median(ss), np.quantile(ss, 0.05, axis=0), np.quantile(ss, 0.95, axis=0)))
    print('\n\n\n')
    return mcmc

if __name__ == '__main__':
    test_df = pd.read_csv('test_df.csv')    #Replace with desired test_df
    test_df = test_df.set_index(['Subject ID', 'Date'])
    x_names = ['Committed', 'Busy', 'Rested']
    y_name = 'Fitbit Step Count'
    mcmcs = fit_simple_regression_model_numpyro(test_df , y_name, x_names, b_show=False)
    mcmc  = fit_regression_model_numpyro(test_df , y_name, x_names, b_show=True)
    print('finished!')

