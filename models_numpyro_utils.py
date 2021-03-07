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

def map_to_daily_mood(df, x_names):
    df[x_names] = df[x_names].ffill().bfill()  
    return df

def plot_data(df_data1, df_data2, y_name, x_names, y_lim=None, b_show=True):       
    plt.figure(figsize=fig_size)
    plot_title = y_name + ' vs ' + str(x_names)
    ys = df_data1[y_name].values
    for x_name in x_names:
        xs = df_data2[x_name].values
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

def plot_data_regression_ci(title, samples, df_data1, df_data2, y_name, x_name, y_lim=None, b_show=True):
    confidence = 0.95
    posterior_mu = jnp.expand_dims(samples['intercept'], -1)
    posterior_mu += jnp.expand_dims(samples['b_'+x_name], -1) * df_data2[x_name].values
    xs_to_sort = df_data2[x_name].values    
    idx = jnp.argsort(xs_to_sort)
    xs = xs_to_sort[idx]
    y_means = jnp.mean(posterior_mu, axis=0)[idx]
    hpdis = hpdi(posterior_mu, confidence)[:, idx]
    ys = df_data1[y_name].values[idx]
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

def plot_data_AR(title, y_test, y_pred, n_forecast, b_show=True):
    plt.figure(figsize=fig_size)
    plt.plot(y_test, label='y test')
    plt.plot(y_pred, label='y pred', ls=':',  lw=3, color='magenta')
    plt.title('forecast = ' + str(n_forecast) + ' for ' + title)
    plt.ylabel('y')
    plt.xlabel('t')
    plt.legend(loc=2)
    if b_show:
        plt.show()
    else:
        plt.savefig(title + '_predict.png')
    plt.close('all')

def model_regression(df, y_name, x_names):
    xs = jnp.array(df[x_names].values)
    y_obs = df[y_name].values   
    mu = numpyro.sample('intercept', dist.Normal(0., 1000))
    M = xs.shape[1]  
    for i in range(M):
        b_name = 'b_'+x_names[i]
        bi = numpyro.sample(b_name, dist.Normal(0., 10.))
        bxi = bi * xs[:,i]
        mu = mu +  bxi        
    log_sigma = numpyro.sample('log_sigma', dist.Normal(0., 10.))    
    numpyro.sample(y_name, dist.Normal(mu, jnp.exp(log_sigma)), obs=y_obs)

def model_impute(data1, y_name, y_index, x_names, mapping, b_classify, y_obs=None):
    bias = numpyro.sample('intercept', dist.Normal(0, 10))
    linear_predictor = bias
    for x_name in x_names:
        x_values = jnp.array([mapping[x_name][i] for i in list(data1[y_index])])      
        b_value  = numpyro.sample('b_'+x_name, dist.Normal(0., 10.))
        linear_predictor += b_value * x_values
    if b_classify:     
        numpyro.sample(y_name, dist.Bernoulli(logits=linear_predictor), obs=y_obs)
    else:
        log_sigma = numpyro.sample('log_sigma', dist.Normal(0., 10.))
        numpyro.sample(y_name, dist.Normal(linear_predictor, jnp.exp(log_sigma)), obs=y_obs)

def model_AR(y_obs, K, y_matrix):
    b  = numpyro.sample('b', dist.Normal(0., 10.).expand([1+K]))
    mus = y_matrix @ b[1:] + b[0]
    b0 = jnp.array([b[0]])
    mus = jnp.concatenate((b0, mus), axis=0)        
    log_sigma = numpyro.sample('log_sigma', dist.Normal(0., 10.)) 
    y_sample = numpyro.sample('obs', dist.Normal(mus, jnp.exp(log_sigma)), obs=y_obs)
        
def fit_simple_regression_model_numpyro(df_data, y_name, x_names, x_lim=None, y_lim=None, y_mean_lim=None, b_show=True):
    mcmcs = []
    for x_name in x_names:
        start_time_simple_regression = timeit.default_timer()
        title = y_name + ' vs ' + x_name + ' (regression model)'
        print('fitting for %s...' % title)

        #Fit model
        rng_key = random.PRNGKey(0)
        kernel = NUTS(model_regression)
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
    kernel = NUTS(model_regression)
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

def get_title(participant, df_data1, df_data2, y_name, x_names, b_classify):
    rows1 = df_data1.shape[0]
    rows2 = df_data2.shape[0]
    detail = participant + ' ' + y_name + ' vs ' + str(x_names)
    if b_classify:
        title = detail + ' (logistic regression N1=' + str(rows1) + ' N2=' + str(rows2) + ')'           
    else:
        title = detail + ' (regression N1=' + str(rows1) + ' N2=' + str(rows2) + ')' 
    return title
    
def prepare_data(df_data1, df_data2, y_name, y_index, x_names, b_standardize, b_show=True):
    data2 = {}          
    for x_name in x_names:
        if b_standardize:
            x_mean = df_data2[x_name].mean()
            x_std  = df_data2[x_name].std()
            df_data2[x_name] = df_data2[x_name].apply(lambda x: (x - x_mean) / x_std)
        data2[x_name] = df_data2[x_name].values
    data1 = {}          
    if b_standardize:
        y_mean = df_data1[y_name].mean()
        y_std  = df_data1[y_name].std()
        df_data1[y_name] = df_data1[y_name].apply(lambda y: (y - y_mean) / y_std)
    data1[y_name]  = df_data1[y_name].values
    data1[y_index] = df_data1[y_index].values
    #plot_data(df_data1, df_data2, y_name, x_names, y_lim=None, b_show=b_show)
    return df_data1, data1, df_data2, data2

def align_start_stop_date(df_mood, df_fitbit):  
    #The start date is the first date of df_mood
    #The stop  date is the last  date of df_fitbit
    df_mood = df_mood.reset_index()
    df_fitbit = df_fitbit.reset_index()
    df_mood['Date'] = pd.to_datetime(df_mood['Date'])
    df_fitbit['Date'] = pd.to_datetime(df_fitbit['Date'])
    start_date = df_mood['Date'].min()
    print('data start_date =', start_date)
    df_mood   = df_mood[df_mood['Date'] >= start_date]
    df_fitbit = df_fitbit[df_fitbit['Date'] >= start_date]
    stop_date = df_fitbit['Date'].max()
    print('data stop_date  =', stop_date)
    df_mood   = df_mood[df_mood['Date'] <= stop_date]
    df_fitbit = df_fitbit[df_fitbit['Date'] <= stop_date]
    return df_mood, df_fitbit

def fit_with_missing_data(participant, length, b_classify, df_data1, df_data2, x_names, y_name, y_lim, b_summary, b_show):
    start_time = timeit.default_timer()
    df_data2, df_data1 = align_start_stop_date(df_data2, df_data1)
    if (df_data2.shape[0] == 0):
        print('not enough data')
        return []
    b_standardize = False    
    y_index = 'y_index'
    df_data1 = df_data1[:length]
    df_data1[y_index] = df_data1.groupby(['Date']).ngroup()    
    df_data1, data1, df_data2, _ = prepare_data(df_data1, df_data2, y_name, y_index, x_names,
                                                b_standardize, b_show=b_show)
    y_obs = df_data1[y_name].values
    df_data2[x_names] = df_data2[x_names].ffill().bfill()
    mapping = df_data2[x_names].to_dict()
  
    print('df_data1.shape  =', df_data1.shape)
    print('df_data2.shape  =', df_data2.shape)
    
    #Fit model
    title = get_title(participant, df_data1, df_data2, y_name, x_names, b_classify)
    print('%s start fitting %s...\n' % (datetime.now(), title))
    mcmc = MCMC(NUTS(model_impute), num_warmup=500, num_samples=1000) 
    mcmc.run(random.PRNGKey(0), data1=data1, y_name=y_name, y_index=y_index,
             x_names=x_names, mapping=mapping, b_classify=b_classify, y_obs=y_obs)
    if b_summary:
        mcmc.print_summary()
    samples = mcmc.get_samples()
    save_build_time(title, start_time)

    #Posterior predictive distribution
    y_pred = Predictive(model_impute, samples)(random.PRNGKey(1), data1=data1, y_name=y_name, y_index=y_index,
                                               x_names=x_names, mapping=mapping, b_classify=b_classify)[y_name]
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
            plot_data_regression_ci(title, samples, df_data1, df_data2, y_name, x_names[0], y_lim, b_show)            
    print('\n\n')
    return samples

def get_AR_predictions(y_obs, y_test, parameters, window):
    history = y_obs[len(y_obs)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = []
    for t in range(len(y_test)):
        n_hist = len(history)
        lags = [history[i] for i in range(n_hist-window, n_hist)]
        y_predict = parameters[0]
        for d in range(window):
            y_predict += parameters[d+1] * lags[window-d-1]
        obs = y_test[t]
        predictions.append(y_predict)
        history.append(obs)
    return predictions
    
def fit_AR_model(df_data, length, n_forecast, window, K, b_show):
    start_AR_mcmc = timeit.default_timer()
    df_data = df_data[:length]
    y_data = df_data[y_name].values
    y_obs, y_test = y_data[1:len(y_data)-n_forecast+1], y_data[len(y_data)-n_forecast:]

    #Fit model
    title = y_name + ' (AR' + str(K) + ' N=' + str(y_obs.shape[0]) + ')'
    print('\n%s, start fitting %s...' % (datetime.now(), title)) 
    y_matrix = jnp.zeros(shape=(len(y_obs)-1, K))
    for p in range(1,K+1):
        values = [y_obs[t-p] if (t >= p) else 0 for t in range(1, len(y_obs))]
        y_matrix = ops.index_update(y_matrix, ops.index[:, p-1], values)          
    rng_key = random.PRNGKey(0)
    kernel = NUTS(model_AR)    
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    mcmc.run(rng_key, y_obs=y_obs, K=K, y_matrix=y_matrix)

    #Display summary
    print('\nsummary for %s =' % title)
    mcmc.print_summary()  
    samples = mcmc.get_samples()
    samples['sigma'] = jnp.exp(samples['log_sigma'])    
    ss = samples['sigma']
    print('sigma mean = %.2f\tstd = %.2f\tmedian = %.2f\tQ5%% = %.2f\tQ95%% = %.2f' % (
          np.mean(ss), np.std(ss), np.median(ss), np.quantile(ss, 0.05, axis=0), np.quantile(ss, 0.95, axis=0)))
    parameter_means = []
    for k,v in samples.items():
        if k.find('b') >= 0:
            for p in range(K+1):
                mean_values = (float)(v[:,p].mean())
                #print('%s[%d] -> %s -> %s\t-> %s' % (k, p, v[:,p].shape, mean_values, v[:,p][:3]))
                parameter_means.append(mean_values)
    y_pred = get_AR_predictions(y_obs, y_test, parameter_means, window)
    print('test mean squared error =', round(sm.mean_squared_error(y_test, y_pred),  2))
    save_build_time(title, start_AR_mcmc)
    
    # plot
    plot_data_AR(title, y_test, y_pred, n_forecast, b_show)
    print('\n')
    return samples

if __name__ == '__main__':
    #filename1 = 'df_mood_fitbit_daily.csv'     #Replace with desired csv file
    #filename2 = 'df_imputed_105_10Min.csv'     #Replace with desired csv file  
    #chosen_df1 = build_df(filename1)
    #chosen_df2 = build_df(filename2)
    #print('chosen_df1.shape =', chosen_df1.shape)
    #print('chosen_df2.shape =', chosen_df2.shape) 
    #print()
    
    b_show = False
    n_repeats = 1 # tested 1, 5
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

        '''
        #Analysis using fit with missing data
        filename = 'build_time_numpyro_impute.csv'
        participants = [105]  # tested [105, 69, 80]
        for participant in participants:
            participant = str(participant)
            filename3a = 'df_' + participant + '_fitbit_per_minute.csv'  #Replace with desired csv file
            filename3b = 'df_' + participant + '_moods.csv'              #Replace with desired csv file
            chosen_df3a = build_df(filename3a, b_set_index=False, b_drop=False)
            chosen_df3b = build_df(filename3b, b_set_index=False, b_drop=False)
            print('participant =', participant)
            print('chosen_df3a.shape =', chosen_df3a.shape)
            print('chosen_df3b.shape =', chosen_df3b.shape)
            print()

            #Inference with missing data
            b_fill = False
            print('b_fill (forward fill before inference) =', b_fill)
            b_summary = True
            x_names = ['Committed', 'Busy', 'Rested']
            y_name  = 'steps'
            y_lim = None
            if b_fill:
                chosen_df3b = map_to_daily_mood(chosen_df3b, x_names)
            chosen_df3a = chosen_df3a.dropna(subset=[y_name])
            for col in list(chosen_df3b.columns):
                print('nan', chosen_df3b[col].isna().sum(), '\t->', col)
            print()
            lengths = [1000] # [100000, 150000, 300000]
            for length in lengths:
                df_data1 = chosen_df3a.copy()   # fitbit
                df_data2 = chosen_df3b.copy()   # mood

                #Perform regression            
                b_classify = False
                samples = fit_with_missing_data(participant, length, b_classify, df_data1, df_data2, x_names, y_name,
                                                y_lim, b_summary, b_show)
                pd.DataFrame.from_dict(data=build_time, orient='index').to_csv(filename, header=False)
                
                #Perform logistic regression
                b_classify = True
                if b_classify == True: #Convert steps to binary: 1 if steps > 0, else 0
                    df_data1[y_name] = df_data1[y_name].apply(lambda x: 1 if x > 0 else 0)
                samples = fit_with_missing_data(participant, length, b_classify, df_data1, df_data2, x_names, y_name,
                                                y_lim, b_summary, b_show)
                pd.DataFrame.from_dict(data=build_time, orient='index').to_csv(filename, header=False)
        '''
        
        #Analysis using AR model
        duration_filename = 'build_time_numpyro_AR.csv'
        #data_filename = 'df_105_fitbit_per_minute.csv'
        data_filename = 'fitbit_per_minute.csv'        
        df_data = build_df(data_filename, b_set_index=False, b_drop=False)
        df_data = df_data.rename(columns={'date' : 'Date'})
        y_name  = 'steps' # 'heart_rate'  # 'steps'
        K = 10
        window = K
        n_forecast = 200
        print('columns =', list(df_data.columns))
        print('na =\n', pd.isna(df_data).sum())
        print('original shape =', df_data.shape)
        print('K =', K)
        print('window   =', window)
        print('forecast =', n_forecast)
        train_lengths = [50000] # tested [5000, 50000, 100000, 1000000...]
        for train_length in train_lengths:
            length = train_length + n_forecast
            df_data = df_data.dropna()
            df_data = df_data[['Date', 'steps', 'heart_rate']]        
            samples = fit_AR_model(df_data, length, n_forecast, window, K, b_show)
            pd.DataFrame.from_dict(data=build_time, orient='index').to_csv(duration_filename, header=False)
                
    print('finished!')

