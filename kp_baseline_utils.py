import pandas as pd
import numpy as np
import sys, getopt
import os
from os import path
import collections

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

    replace_names = {'studyid': 'study_id',
                     'dem_1': 'age',
                     'dem_2': 'gender'}
    for k,v in replace_names.items():
        dd['ElementName'] = dd['ElementName'].replace(k,v)
    
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
    df = df.rename(columns=replace_names)

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

    tipi_scores = {}
    for k,v in df_tipi.items():        
        item0 = [int(str(x).split('=')[0]) for x in v[0]]
        item1 = [int(str(x).split('=')[0]) for x in v[1]]
        tipi_scores[k] = [(x+y)/2 for x, y in zip(item0, item1)]
    tipi_scores = pd.DataFrame(tipi_scores).set_index(df['study_id'])

    tipi_scores_filename = 'kp-baseline-survey-tipi.csv'
    tipi_scores_path = os.path.join(output_folder, tipi_scores_filename)
    tipi_scores.to_csv(tipi_scores_path)
    print('TIPI scores =', tipi_scores_filename)

    #--------------------------------------------------------------------------------
    #Create IPAQ scores

    ipaq_data = collections.defaultdict(list)
    for name in list(df.columns):
        if str(name).lower().find('ipaq') != -1:
            if ((str(name).lower().find('_none') == -1) and
                (str(name).lower().find('_dk') == -1) and
                (str(name).lower().find('ipaq_4') == -1)):
                for item in df[name].values:
                    if str(item).lower() == 'nan':
                        ipaq_data[name].append(0)
                    else:
                        item = str(item).replace('-','')
                        item = float(item)
                        ipaq_data[name].append(item)
                ipaq_data[name] = np.array(ipaq_data[name])

    
    MOVA = ipaq_data['ipaq_1'] * (ipaq_data['ipaq_1_hr'] * 60. + ipaq_data['ipaq_1_min'])
    MOMA = ipaq_data['ipaq_2'] * (ipaq_data['ipaq_2_hr'] * 60. + ipaq_data['ipaq_2_min'])
    MOW  = ipaq_data['ipaq_3'] * (ipaq_data['ipaq_3_hr'] * 60. + ipaq_data['ipaq_3_min'])
    MOTO = MOVA + MOMA + MOW
        
    MMVA = MOVA + MOMA
    MMAE = 2. * MOVA + MOMA                                  
    MMET = 8. * MOVA + 4. * MOMA + 3.3 * MOW
         
    ipaq_scores = {'MOVA':MOVA, 'MOMA':MOMA, 'MOW':MOW, 'MOTO':MOTO, 'MMVA':MMVA, 'MMAE':MMAE, 'MMET':MMET}        
    ipaq_scores = pd.DataFrame(ipaq_scores).set_index(df['study_id'])

    ipaq_scores_filename = 'kp-baseline-survey-ipaq.csv'
    ipaq_scores_path = os.path.join(output_folder, ipaq_scores_filename)
    ipaq_scores.to_csv(ipaq_scores_path)
    print('IPAQ scores =', ipaq_scores_filename)


    #--------------------------------------------------------------------------------
    #Create social csv file

    column_names = ['age', 'gender', 'support_1', 'support_2',
                    'support_3', 'support_4', 'support_5', 'support_6',
                    'ipaq_1_hr', 'ipaq_2_hr', 'ipaq_3_hr', 'ipaq_4_hr',
                    'life_1', 'life_2', 'life_3', 'life_4', 'life_5',
                    'self_1', 'self_2', 'self_3', 'self_4', 'self_5']
    df_social = df[column_names].copy()
    chosen_columns = ['support_1', 'support_2', 'support_3', 'support_4', 'support_5', 'support_6']
    df_social['average_support'] = df_social[df_social[chosen_columns] < 99].mean(axis=1)
    df_social['MOTO'] = MOTO  
    chosen_columns = ['life_1', 'life_2', 'life_3', 'life_4', 'life_5']
    df_social['average_stress'] = df_social[chosen_columns].mean(axis=1)
    chosen_columns = ['self_1', 'self_2', 'self_3', 'self_4', 'self_5']
    df_social['average_efficacy'] = df_social[chosen_columns].mean(axis=1)
    df_social = df_social.set_index(df['study_id'])

    social_filename = 'kp-baseline-survey-social.csv'
    social_path = os.path.join(output_folder, social_filename)
    df_social.to_csv(social_path)
    print('social      =', social_path)
    
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
 
