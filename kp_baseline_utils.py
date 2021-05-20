import pandas as pd
import numpy as np
import sys, getopt
import os
from os import path

def process_kp_baseline_survey(data_dictionary_filename, data_filename, output_folder):

    print('input dd    =', data_dictionary_filename)
    print('input df    =', data_filename)
    print('output dir  =', output_folder)

    #--------------------------------------------------------------------------------
    #Process data dictionary for KPDataDictionaries
    
    dd = pd.read_csv(data_dictionary_filename)

    dd = dd[['Variable Name', 'Type', 'Label', 'Valid Values (if not included in label)']]

    dd = dd.rename(columns={'Variable Name': 'ElementName',
                            'Type': 'DataType',
                            'Label': 'ElementDescription',
                            'Valid Values (if not included in label)':'Notes'})

    dd['ElementName'] = dd['ElementName'].str.lower()
    dd['ElementName'] = dd['ElementName'].replace('studyid','study_id')
    
    dd['DataType'] = dd['DataType'].replace('integer','Integer')
    dd['DataType'] = dd['DataType'].replace('boolean','Boolean')
    for i, x in enumerate(dd["Notes"].values):
        if str(x).lower().find("1,") >= 0:
            dd['DataType'].values[i] = "Categorical"
    dd['DataType'] = dd['DataType'].replace('','Integer')
    dd = dd[dd['DataType'] != 'text']
    dd.insert(2, 'Size', '')
    dd.insert(3, 'Required', 'Required') 
    dd.insert(5, 'ValueRange', '')
    dd.insert(7, 'Aliases', '')
    dd = dd.dropna(subset=['ElementName'])

    #--------------------------------------------------------------------------------
    #Process data for KPData    

    df = pd.read_csv(data_filename)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={'studyid': 'study_id'})      

    column_names = dd['ElementName'].values
    df = df[column_names]

    for name in list(df.columns):
        if str(name).lower().find('trait') != -1:
            df[name] = df[name].fillna(4.0) 
            df[name] = df[name].astype(float)
            df[name] = df[name].replace(1.0,'1 = Disagree strongly')
            df[name] = df[name].replace(2.0,'2 = Disagree moderately')
            df[name] = df[name].replace(3.0,'3 = Disagree a little')
            df[name] = df[name].replace(4.0,'4 = Neither agree nor disagree')
            df[name] = df[name].replace(5.0,'5 = Agree a little')
            df[name] = df[name].replace(6.0,'6 = Agree moderately')
            df[name] = df[name].replace(7.0,'7 = Agree strongly')   
    
    #--------------------------------------------------------------------------------
    #Create new csv files
    
    output_data_dictionary = 'kp-baseline-survey.csv'
    output_data = 'kp-baseline-survey-data.csv'
    output_data_dictionary_path = os.path.join(output_folder, output_data_dictionary)
    output_data_path = os.path.join(output_folder, output_data)
    dd.to_csv(output_data_dictionary_path, index=False)
    df.to_csv(output_data_path, index=False)

    print('dd shape    =', dd.shape)
    print('df shape    =', df.shape)
    print('dd output   =', output_data_dictionary)
    print('df output   =', output_data)

    #--------------------------------------------------------------------------------
    #Create TIPI scores

    df_data = {}

    for name in list(df.columns):
        if str(name).lower().find('trait') != -1:
            df_data[name] = df[name]

    reverse_map = { '1 = Disagree strongly'          : '7 = Agree strongly',
                    '2 = Disagree moderately'        : '6 = Agree moderately',
                    '3 = Disagree a little'          : '5 = Agree a little',
                    '4 = Neither agree nor disagree' : '4 = Neither agree nor disagree',
                    '5 = Agree a little'             : '3 = Disagree a little',
                    '6 = Agree moderately'           : '2 = Disagree moderately',
                    '7 = Agree strongly'             : '1 = Disagree strongly' }
    
    def reverse(df):
        df = df.apply(lambda x: reverse_map[x])
        return df

    df_tipi = {}
    df_tipi['Extraversion']          = [df_data['trait_1'].values,            reverse(df_data['trait_6']).values]
    df_tipi['Agreeableness']         = [reverse(df_data['trait_2']).values,   df_data['trait_7'].values]
    df_tipi['Conscientiousness']     = [df_data['trait_3'].values,            reverse(df_data['trait_8']).values]
    df_tipi['Emotional Stability']   = [reverse(df_data['trait_4']).values,   df_data['trait_9'].values]
    df_tipi['Openess to Experience'] = [df_data['trait_5'].values,            reverse(df_data['trait_10']).values]

    scores = {}
    for k,v in df_tipi.items():        
        item0 = [int(str(x).split('=')[0]) for x in v[0]]
        item1 = [int(str(x).split('=')[0]) for x in v[1]]
        scores[k] = [(x+y)/2 for x, y in zip(item0, item1)]
    scores = pd.DataFrame(scores).set_index(df['study_id'])

    tipi_scores_filename = 'kp-baseline-survey-tipi.csv'
    tipi_scores_path = os.path.join(output_folder, tipi_scores_filename)
    scores.to_csv(tipi_scores_path)
    print('TIPI scores =', tipi_scores_filename)


def main(argv):

    #For example, run the following command:
    #python kp_baseline_utils.py -d "HeartSteps Data Dictionary.csv" -f "HS Survey Data 10.8.2020.csv" -o ""

    instructions = "baseline_utils.py -d <data_dictionary_filename> -f <data_filename> -o <output_folder>"
    try:
        opts, args = getopt.getopt(argv,"d:f:o:",["data_dictionary=","data=","output_folder="])
    except getopt.GetoptError:
        print(instructions)
        sys.exit(2)

    data_dictionary_filename = ''
    data_filename = '' 
    for opt, arg in opts:
        if opt in ["-d", "--data_dictionary"]:
            data_dictionary_filename = arg
        elif opt in ["-f", "--data"]:
            data_filename = arg
        elif opt in ["-o", "--output_folder"]:
            output_folder = arg

    if data_dictionary_filename == '' or data_filename == '':
        print(instructions)
        sys.exit()

    if output_folder != '' and not os.path.exists(output_folder):
        print(output_folder, 'folder does not exist!')
        print(instructions)
        sys.exit()
    
    process_kp_baseline_survey(data_dictionary_filename, data_filename, output_folder)

if __name__ == '__main__':    
     main(sys.argv[1:])
 
