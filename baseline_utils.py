import pandas as pd
import numpy as np

if __name__ == '__main__':

    data_dictionary_filename = 'HeartSteps_DataDictionary_2021-01-06.csv'
    data_filename            = 'HeartSteps-BaselineSurveyData_DATA_2021-01-06_1422.csv'

    #--------------------------------------------------------------------------------
    #Process data dictionary for U01DataDictionaries
    
    dd = pd.read_csv(data_dictionary_filename)

    dd = dd[dd['Form Name'] == 'baseline_survey_2']

    dd['Field Type'] = np.where((dd['Text Validation Type OR Show Slider Number'] == 'number'), 'Integer', dd['Field Type'])
    
    dd = dd[['Variable / Field Name', 'Field Type', 'Field Label', 'Choices, Calculations, OR Slider Labels']]
    dd = dd.rename(columns={'Variable / Field Name': 'ElementName',
                            'Field Type': 'DataType',
                            'Field Label': 'ElementDescription',
                            'Choices, Calculations, OR Slider Labels':'Notes'})
    dd.insert(2, 'Size', '')
    dd.insert(3, 'Required', 'Required') 
    dd.insert(5, 'ValueRange', '')
    dd.insert(7, 'Aliases', '')
    dd['DataType'] = dd['DataType'].replace('radio','Categorical')
    dd['DataType'] = dd['DataType'].replace('dropdown','Categorical')
    dd['DataType'] = dd['DataType'].replace('yesno','Boolean')
    #Remove remaining text that are not number
    dd = dd[dd['DataType'] != 'text']
    dd = dd[dd['DataType'] != 'descriptive']
    dd = dd[dd['DataType'] != 'notes']

    dd['ElementName'] = [str(x).replace('bsl_', '').replace('_v2', '') for x in dd['ElementName']]
    dd['ElementDescription'] = [str(x).replace('[SELECT ALL THAT APPLY]', '(check all that apply)') for x in dd['ElementDescription']]
    dd['ElementDescription'] = [str(x).replace('(Please check all that apply to you)', '(Check all that apply)') for x in dd['ElementDescription']]
    dd['ElementDescription'] = [str(x).replace(' (list which ones below)', '') for x in dd['ElementDescription']]
     
    #Expand checkbox and set to Boolean
    dd_checkbox = dd[dd['DataType'] == 'checkbox']['ElementName']
    for index, item in enumerate(dd['ElementName'].values):
        for element_name in dd_checkbox:
            if item == element_name:
                notes = str(dd['Notes'].values[index]).split('|')                
                for expand, item in enumerate(notes):
                    item = item.split(',')
                    key = int(item[0])
                    value = str(item[1]).strip()                
                    new_name = element_name + ': ' + value
                    new_name = new_name.replace('(specify below)','').strip()
                    new_index = index + expand
                    line = pd.DataFrame({'ElementName': new_name, 'DataType': 'Boolean', 'Required': 'Required',
                                         'ElementDescription': new_name}, index=[new_index])
                    dd = dd.append(line, ignore_index=False)
    
    #--------------------------------------------------------------------------------
    #Process data for U01Data

    df = pd.read_csv(data_filename)
    
    df.columns = [str(x).replace('bsl_', '').replace('_v2', '') for x in df.columns]
    df = df.drop(columns=['baseline_survey_2_complete'])
    
    dd_categorical = dd[dd['DataType'] == 'Categorical']['ElementName']
    dd_binary   = dd[dd['DataType'] == 'Boolean' ]['ElementName']

    for field in list(df.keys()):
        
        pos = field.find('___')
        if pos > 0:
            #Get the checkbox value after '___'
            #shorten_field is the corresponding checkbox name
            checkbox = int(field[pos+3:])
            shorten_field = str(field)[:pos]
            for index, item in enumerate(dd['ElementName'].values):
                if item == shorten_field:
                    notes = str(dd['Notes'].values[index]).split('|')
            current_value = ''
            dict_notes = {}
            for item in notes:
                item = item.split(',')                
                key = int(item[0])
                value = str(item[1]).strip()
                if checkbox == key:
                    current_value = value
                if key == 0:
                    dict_notes[key] = str(key) + ': No'
                else:
                    dict_notes[key] = str(key) + ': Yes'
            new_name = shorten_field + ': ' + current_value
            new_name = new_name.replace('(specify below)','').strip()
            df = df.rename(columns={field: new_name})
            df[new_name] = df[new_name].map(lambda x: x if str(x).lower()=="nan" else dict_notes[int(x)])            

        if field in dd_binary.values:
            dict_notes = {0: '0: Yes', 1: '1: No'}
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else dict_notes[int(x)])

        if field in dd_categorical.values:
            for index, item in enumerate(dd['ElementName'].values):
                if item == field:
                    notes = str(dd['Notes'].values[index]).split('|')
                    break
            current_value = ''
            dict_notes = {}
            for item in notes:
                item = item.split(',')                
                key = int(item[0])
                value = str(item[1]).strip()
                if value.find('=') > 0:
                    dict_notes[key] = value.replace(' =',':')
                else:
                    dict_notes[key] = str(key) + ': ' + value                      
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else dict_notes[int(x)])            

    #--------------------------------------------------------------------------------
    #Create new csv files
    
    #Remove checkbox from data dictionary, after expanding the checkbox data
    dd = dd[dd['DataType'] != 'checkbox']  

    dd.to_csv('baseline-survey.csv', index=False)
    df.to_csv('baseline-survey-data.csv', index=False)

    print('dd.shape =', dd.shape)
    print('df.shape =', df.shape)



    


