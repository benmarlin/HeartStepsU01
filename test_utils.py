import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DASH = '----------------------------------------------------------'

def load_df(filename, t_name='t', df_name='df'):
  input_df = pd.read_csv(filename)
  input_df[t_name] = np.arange(input_df.shape[0])
  input_df = input_df.set_index([t_name])
  input_df.name = df_name
  return input_df

def load_data(filename, participant, chosen_indices, rename_indices,
              chosen_columns, rename_columns, size=None, df_name='df',
              b_display=True, b_display_more=False, rescale_list=[], drop_check_list=[],
              scale_in=[0.0, 0.25, 0.5, .75, 1.], scale_out=[1.,2.,3.,4.,5.],
              skip_rows=0):
  load_df = pd.read_csv(filename)
  if b_display_more:
    print('loading...', filename)
    print('all columns    =', list(load_df.columns)) 
  for rescale_name in rescale_list:
    if rescale_name in load_df:
      load_df[rescale_name] = load_df[rescale_name].replace(scale_in, scale_out)  
  load_df = load_df[chosen_columns+chosen_indices]  
  if len(drop_check_list) > 0:
    # Drop first few rows that contain NaN in drop_check_list
    count_drop_row = 0
    b_found_Nan = True
    while (b_found_Nan) and (count_drop_row < load_df.shape[0]):
      b_found_Nan = False
      for col_name in drop_check_list:
        if col_name in load_df:
          nan_indices = np.argwhere(np.isnan(load_df[col_name].values)).flatten()
          if count_drop_row in nan_indices:
            b_found_Nan = True
      if b_found_Nan == False:
        break
      else:
        count_drop_row += 1
    load_df = load_df.tail(load_df.shape[0]-count_drop_row)
  new_subject_id = None
  for i, old_index in enumerate(chosen_indices):
    load_df = load_df.rename(columns={old_index:rename_indices[i]})
    if old_index == 'Subject ID':
      new_subject_id = rename_indices[i]
  for i, old_name in enumerate(chosen_columns):
    load_df = load_df.rename(columns={old_name:rename_columns[i]}) 
  load_df = load_df.iloc[skip_rows:,:]
  load_df = load_df.astype('float')
  if new_subject_id is not None:
    if new_subject_id in load_df:
      load_df[new_subject_id] = [str(subject_id).replace(".0","") for subject_id in load_df[new_subject_id].values]
  load_df = load_df.set_index(rename_indices)
  load_df.name = df_name
  if b_display_more:
    print('chosen_columns =', chosen_columns)
    print('chosen_indices =', chosen_indices)
  if b_display:
    print('\nparticipant =', participant)
    print('{} =\n{}\ndf.name  = {}\ndf.shape = {}'.format(df_name, load_df.head(), load_df.name, load_df.shape))
    print(DASH)

  if size != None:
    load_df = load_df.head(size)
    if b_display:
      print('df.shape =', load_df.shape, '\t shorter version')
  if b_display:
    print('\ncount NaN =')
    print(load_df.isna().sum())
    print('\n% NaN =')
    print((100 * load_df.isna().sum()/max(1,load_df.shape[0])).astype(int))
  return load_df

def standardize(df_raw, df_name='df', b_display=True):
  df_mean = df_raw.mean()
  df_sd = df_raw.std()
  standardized_df = (df_raw-df_mean)/df_sd
  standardized_df.name = df_name
  if b_display:
    print(DASH)
    print('standardized {} =\n{}'.format(df_name, standardized_df.head()))
  df_info = {'df_raw':df_raw, 'df':standardized_df, 'df_mean':df_mean, 'df_sd':df_sd}
  return df_info

def plot_posterior(samples, parameter_names, figsize=(12,7), init_subplot=341, rescale_df_info=None,
                   n_bins=30, vertical_x=0, b_vertical_x=False, y_lim=(0,110), x_lim=(0,6), plot_loc=2, 
                   b_default_lim=False, plot_title=''):
  input_samples = samples.copy()
  if rescale_df_info is not None:
    df_raw = rescale_df_info['df_raw']
    df_mean = rescale_df_info['df_mean']
    df_sd   = rescale_df_info['df_sd']    
    for name in input_samples.keys():
      for k in df_raw.columns:
        if name.find(k) >= 0:
          input_samples[name] = np.array(input_samples[name]) * df_sd[k] + df_mean[k]
  for name in input_samples.keys():
    input_samples[name] = np.array(input_samples[name])
  print('\n\n{} = {}'.format(plot_title, parameter_names))
  plt.figure(figsize=figsize)
  for i, param in enumerate(parameter_names):
    plt.subplot(init_subplot+i)
    plt.hist(input_samples[param], bins=n_bins, label='posterior distrib.')
    if b_vertical_x:
      plt.axvline(x=vertical_x, ls=':', lw=2, color='dimgray')
    plt.axvline(x=input_samples[param].mean(), ls='-', lw=3, color='orangered', label='posterior mean')
    plt.ylabel('count')
    plt.title(param)
    plt.grid(True)
    if not b_default_lim:
      plt.ylim(y_lim)
      plt.xlim(x_lim)
    plt.tight_layout()
    plt.legend(loc=plot_loc, fontsize=8)
  plt.show()
