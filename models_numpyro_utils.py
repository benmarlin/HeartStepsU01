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
from numpyro.infer import MCMC, NUTS

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

def model(x1=None, x2=None, x3=None, y_obs=None):
    a = numpyro.sample('a', dist.Normal(0., 1000))
    b1x1, b2x2, b3x3 = 0., 0., 0.
    if x1 is not None:
        b1 = numpyro.sample('b1', dist.Normal(0., 10.))
        b1x1 = b1 * x1
    if x2 is not None:
        b2 = numpyro.sample('b2', dist.Normal(0., 10.))
        b2x2 = b2 * x2
    if x3 is not None:
        b3 = numpyro.sample('b3', dist.Normal(0., 10.))
        b3x3 = b3 * x3
    log_sigma = numpyro.sample('log_sigma', dist.Normal(0., 10.))
    mu = a + b1x1 + b2x2 + b3x3
    numpyro.sample('obs', dist.Normal(mu, jnp.exp(log_sigma)), obs=y_obs)

def fit_simple_regression_model_numpyro(df_data, y_name, x_names, b_show=True):
    for x_name in x_names:    
        N = 1000
        data_x    = df_data[x_name].values[:N]
        data_y = df_data[y_name].values[:N]

        #Fit model using NUTS
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
        mcmc.run(rng_key_, x1=data_x, y_obs=data_y)

        #Display summary
        title = y_name + ' vs ' + str(x_names)
        print('summary for %s =' % title)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        samples['sigma'] = jnp.exp(samples['log_sigma'])    
        ss = samples['sigma']
        print('sigma mean = %.2f\tstd = %.2f\tmedian = %.2f\tQ5%% = %.2f\tQ95%% = %.2f' % (
              np.mean(ss), np.std(ss), np.median(ss), np.quantile(ss, 0.05, axis=0), np.quantile(ss, 0.95, axis=0)))

        #Plot
        plot_data_regression_lines(samples, title, data_x, x_name, data_y, y_name, b_show)
        print('\n')

if __name__ == '__main__':
    test_df = pd.read_csv('test_df.csv')    #Replace with desired test_df
    x_names = ['Committed', 'Busy', 'Rested']
    y_name = 'Fitbit Step Count'
    fit_simple_regression_model_numpyro(test_df , y_name, x_names, b_show=False)    
    print('finished!')

