import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from textwrap import wrap
import collections
import timeit
from datetime import datetime
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import ops, random
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi

fig_size = (7,5)
build_time = collections.defaultdict(list)
def save_build_time(title, start_time):
    duration = (int)(timeit.default_timer() - start_time)
    print('duration =', duration, 'seconds')
    build_time[title].append(duration)

def build_df(filename, b_set_index=True, b_drop=True):
    df = pd.read_csv(filename, low_memory=False)
    if b_drop:
        df = df.dropna()
    if b_set_index:
        df['Subject ID'] = df['Subject ID'].astype(str)
        df = df.set_index(['Subject ID', 'Date'])
    return df

def plot_data(df_data, y_name, x_names, y_lim=None, b_show=True):       
    plt.figure(figsize=fig_size)
    plot_title = y_name + ' vs ' + str(x_names)
    ys = df_data[y_name].values
    for x_name in x_names:
        xs = df_data[x_name].values
        plt.scatter(xs, ys, label=x_name, marker='o', alpha=.3)
    plt.legend(loc=2)
    plt.xlabel('x')
    if y_lim != None:
        plt.ylim(y_lim)
    plt.ylabel(y_name)
    plot_title = y_name + ' vs ' + str(x_names) 
    plt.title(plot_title + ' data')
    if b_show:
        plt.show()
    else:
        filename =  plot_title + "_data.png"
        plt.savefig(filename)
    plt.close('all')
        
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
    if y_lim != None:
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

def plot_data_regression_ci(title, samples, df_data, y_name, x_name, y_lim=None, b_show=True):
    confidence = 0.95
    posterior_mu = jnp.expand_dims(samples['a'], -1)
    posterior_mu += jnp.expand_dims(samples['b_'+x_name], -1) * df_data[x_name].values
    xs_to_sort = df_data[x_name].values    
    idx = jnp.argsort(xs_to_sort)
    xs = xs_to_sort[idx]
    y_means = jnp.mean(posterior_mu, axis=0)[idx]
    hpdis = hpdi(posterior_mu, confidence)[:, idx]
    ys = df_data[y_name].values[idx]
    plt.figure(figsize=(fig_size))
    label = y_name + ' vs ' + x_name
    plt.scatter(xs, ys, label=label, marker='o', alpha=.3)
    plt.plot(xs, y_means, 'k', label='mean')
    label = str(int(confidence*100)) + '% CI'
    plt.fill_between(xs, hpdis[0], hpdis[1], alpha=0.25, color="blue", label=label)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if y_lim != None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.legend(loc=2)
    if b_show:
        plt.show()
    else:
        filename =  title + '_' + str(int(confidence*100)) + 'p_ci_regression.png'
        plt.savefig(filename)
    plt.close('all')

def regression_model(df, y_name, x_names):
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
        kernel = NUTS(regression_model)
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
    kernel = NUTS(regression_model)
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

def get_title(df_data, y_name, x_names, b_classify):
    rows  = df_data.shape[0]
    detail = y_name + ' vs ' + str(x_names)
    if b_classify:
        title = detail + ' (logistic regression N=' + str(rows) + ')'            
    else:
        title = detail + ' (regression N=' + str(rows) + ')' 
    return title
    
def model_impute(data, y_name, x_names, b_classify, y_obs=None):
    a = numpyro.sample('a', dist.Normal(0, 10))
    linear_predictor = a
    for x_name in x_names:
        x_values = data[x_name]
        isnan = np.isnan(x_values)
        x_nan_indices = np.nonzero(isnan)[0]
        if len(x_nan_indices) > 0:        
            x_mu     = numpyro.sample(x_name+'_mu',     dist.Normal(0, 10).expand([1]))
            x_sigma  = numpyro.sample(x_name+'_sigma',  dist.Normal(0, 10).expand([1]))
            x_impute = numpyro.sample(x_name+'_impute', dist.Normal(x_mu[x_nan_indices],
                                      x_sigma[x_nan_indices]).mask(False))            
            x_values = ops.index_update(x_values, x_nan_indices, x_impute)
            numpyro.sample(x_name, dist.Normal(x_mu, x_sigma), obs=x_values)               
        b_value = numpyro.sample('b_'+x_name, dist.Normal(0, 10))
        linear_predictor += b_value * x_values    
    if b_classify:     
        numpyro.sample(y_name, dist.Bernoulli(logits=linear_predictor), obs=y_obs)
    else:
        numpyro.sample(y_name, dist.Normal(linear_predictor, 10), obs=y_obs)

def prepare_data(df_data, y_name, x_names, b_standardize, b_show=True):
    data = {}          
    for x_name in x_names:
        if b_standardize:
            x_mean = df_data[x_name].mean()
            x_std  = df_data[x_name].std()
            df_data[x_name] = df_data[x_name].apply(lambda x: (x - x_mean) / x_std)
        data[x_name] = df_data[x_name].values
    #plot_data(df_data, y_name, x_names, y_lim=None, b_show=b_show)
    return df_data, data
            
def fit_with_missing_data(b_classify, df_data, x_names, y_name, y_lim, b_summary, b_show):
    print('df_data =', df_data.shape)        
    y_obs = df_data[y_name].values
    b_standardize = False
    df_data, data = prepare_data(df_data, y_name, x_names, b_standardize, b_show=b_show)
    start_time = timeit.default_timer()
    title = get_title(df_data, y_name, x_names, b_classify)
    print('%s\nfitting %s...' % (datetime.now(), title))
    
    # perform inference
    mcmc = MCMC(NUTS(model_impute), num_warmup=1000, num_samples=1000) 
    mcmc.run(random.PRNGKey(0), data=data, y_name=y_name, x_names=x_names, b_classify=b_classify, y_obs=y_obs)
    if b_summary:
        mcmc.print_summary()
    samples = mcmc.get_samples()
    save_build_time(title, start_time)
    pd.DataFrame.from_dict(data=build_time, orient='index').to_csv('build_time_impute_numpyro.csv', header=False)

    # posterior predictive distribution
    y_pred = Predictive(model_impute, samples)(random.PRNGKey(1), data=data, y_name=y_name, x_names=x_names,
                                               b_classify=b_classify)[y_name]
    if b_classify:
        y_pred = (y_pred.mean(axis=0) >= 0.5).astype(jnp.uint8)
        print('accuracy =', (y_pred == y_obs).sum() / y_obs.shape[0])
        confusion_matrix = pd.crosstab(pd.Series(y_obs, name='actual'), pd.Series(y_pred, name='predict'))
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)
        #print('\n', confusion_matrix)
    else:
        y_pred = y_pred.mean(axis=0).reshape(-1)
        print('prediction shape =', y_pred.shape)
        print('mean absolute error =', round(sm.mean_absolute_error(y_obs, y_pred), 2)) 
        print('mean squared error  =', round(sm.mean_squared_error(y_obs,  y_pred),  2))  
        print('R2 score =', round(sm.r2_score(y_obs, y_pred), 2))
        if (not b_standardize):
            plot_data_regression_ci(title, samples, df_data, y_name, x_names[0], y_lim, b_show)            
    print('\n\n')
    return samples

if __name__ == '__main__':
    #filename1 = 'df_mood_fitbit_daily.csv'  #Replace with desired csv file
    #filename2 = 'df_imputed_105_10Min.csv'  #Replace with desired csv file
    filename3 = 'fitbit_per_minute.csv'      #Replace with desired csv file
    #chosen_df1 = build_df(filename1)
    #chosen_df2 = build_df(filename2)
    chosen_df3 = build_df(filename3, b_set_index=False, b_drop=False)
    #print('chosen_df1.shape =', chosen_df1.shape)
    #print('chosen_df2.shape =', chosen_df2.shape)
    print('chosen_df3.shape =', chosen_df3.shape)
    print()
    b_show = False
    n_repeats = 1 #5 1

    for repeat in range(n_repeats):
        '''
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
        '''

        #Inference with missing data
        b_summary = False
        x_names = ['heart_rate']
        y_name  = 'steps'
        y_lim =(-0.5, 120)
        chosen_df3 = chosen_df3.dropna(subset=[y_name])
        for col in list(chosen_df3.columns):
            print('nan', chosen_df3[col].isna().sum(), '\t->', col)
        print()
        
        lengths = [1000] # tested [1000, 5000, 10000, 50000, 100000...]      
        for length in lengths:
            #Perform regression
            df_data = chosen_df3.copy()[:length]
            b_classify = False
            samples = fit_with_missing_data(b_classify, df_data, x_names, y_name, y_lim, b_summary, b_show)
            
            #Perform logistic regression
            df_data = chosen_df3.copy()[:length]
            b_classify = True
            if b_classify == True: #Convert steps to binary: 1 if steps > 0, else 0
                df_data[y_name] = df_data[y_name].apply(lambda x: 1 if x > 0 else 0)
            samples = fit_with_missing_data(b_classify, df_data, x_names, y_name, y_lim, b_summary, b_show)
      
    print('finished!')

