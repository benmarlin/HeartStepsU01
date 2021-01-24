import pandas as pd
import numpy as np

if __name__ == '__main__':

    data_dictionary_filename = 'HeartSteps_DataDictionary_2021-01-22.csv'
    data_filename            = 'HeartSteps-BaselineSurveyData_DATA_2021-01-06_1422.csv'

    #--------------------------------------------------------------------------------
    #Process data dictionary for U01DataDictionaries
    
    dd = pd.read_csv(data_dictionary_filename)

    dd = dd[dd['Form Name'] == 'baseline_survey_2']

    map_to_new_name = pd.Series(dd['New Name'].values, index=dd['Variable / Field Name']).to_dict()
    
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

    dd['ElementDescription'] = [str(x).replace('[SELECT ALL THAT APPLY]', '(check all that apply)') for x in dd['ElementDescription']]
    dd['ElementDescription'] = [str(x).replace('(Please check all that apply to you)', '(Check all that apply)') for x in dd['ElementDescription']]
    dd['Notes'] = [str(x).replace(' (list which ones below)', '') for x in dd['Notes']]
     
    #Expand checkbox and set to Boolean
    dd_checkbox = dd[dd['DataType'] == 'checkbox']['ElementName']
    for index, item in enumerate(dd['ElementName'].values):
        for element_name in dd_checkbox:
            if item == element_name:
                notes = str(dd['Notes'].values[index]).split('|')
                notes = [str(x).strip() for x in notes]
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
            dict_notes = {}
            for index, item in enumerate(dd['ElementName'].values):
                if item == shorten_field:
                    notes = str(dd['Notes'].values[index]).split('|')
                    notes = [str(x).strip() for x in notes]
                    current_value = ''
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
                    break
            new_name = shorten_field + ': ' + current_value
            new_name = new_name.replace('(specify below)','').strip()
            df = df.rename(columns={field: new_name})
            df[new_name] = df[new_name].map(lambda x: x if str(x).lower()=="nan" else dict_notes[int(x)])

        if field in dd_binary.values:
            dict_notes = {0: '0: No', 1: '1: Yes'}
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else dict_notes[int(x)])

        if field in dd_categorical.values:
            dict_notes = {}
            for index, item in enumerate(dd['ElementName'].values):
                if item == field:
                    notes = str(dd['Notes'].values[index]).split('|')
                    notes = [str(x).strip() for x in notes]
                    current_value = ''
                    for item in notes:
                        item = item.split(',')                
                        key = int(item[0])
                        value = str(item[1]).strip()
                        pos = value.find('=')
                        if pos > 0:
                            dict_notes[key] = str(key) + ': ' + value[pos+1:].strip() 
                        else:
                            dict_notes[key] = str(key) + ': ' + value.strip()
                    dd['Notes'].values[index] = ' | '.join(list(dict_notes.values()))
                    break                
            df[field] = df[field].map(lambda x: x if str(x).lower()=="nan" else dict_notes[int(x)])

    #--------------------------------------------------------------------------------
    #Create new csv files
    
    #Remove checkbox from data dictionary, after expanding the checkbox data
    dd = dd[dd['DataType'] != 'checkbox']

    #Replace with new name
    def shorten(x):
        return str(x.split(':')[0]) 
    dd['ElementName'] = dd['ElementName'].map(lambda x: x.replace(shorten(x), map_to_new_name[shorten(x)]))
    dd['ElementDescription'] = dd['ElementDescription'].map(lambda x: x.replace(shorten(x), map_to_new_name[shorten(x)])
                                                            if shorten(x) in map_to_new_name else x)    
    df.columns = df.columns.map(lambda x: x.replace(shorten(x), map_to_new_name[shorten(x)])
                                                            if shorten(x) in map_to_new_name else x)
    
    dd.to_csv('baseline-survey.csv', index=False)
    df.to_csv('baseline-survey-data.csv', index=False)

    print('dd.shape =', dd.shape)
    print('df.shape =', df.shape)



    


