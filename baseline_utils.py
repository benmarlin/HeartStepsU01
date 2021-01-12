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
    dd['DataType'] = dd['DataType'].replace('radio','Ordinal')
    dd['DataType'] = dd['DataType'].replace('checkbox','Checkbox')
    dd['DataType'] = dd['DataType'].replace('dropdown','Ordinal')
    dd['DataType'] = dd['DataType'].replace('yesno','Binary')
    #Remove remaining text that are not number
    dd = dd[dd['DataType'] != 'text']
    dd = dd[dd['DataType'] != 'descriptive']
    dd = dd[dd['DataType'] != 'notes']

    dd['ElementName'] = [str(x).replace('bsl_', '').replace('_v2', '') for x in dd['ElementName']]
        
    dd.to_csv('baseline-survey.csv', index=False)

    #--------------------------------------------------------------------------------
    #Process data for U01Data

    df = pd.read_csv(data_filename)
    
    df.columns = [str(x).replace('bsl_', '').replace('_v2', '') for x in df.columns]
    df = df.drop(columns=['baseline_survey_2_complete'])

    for field in list(df.keys()):
        pos = field.find('___')
        if pos > 0:
            #Get the checkbox value after '___'
            checkbox = int(field[pos+3:])
            shorten_field = str(field)[:pos]
            for index, item in enumerate(dd['ElementName'].values):
                if item == shorten_field:
                    notes = str(dd['Notes'].values[index]).split('|')
            current_value = ''
            for item in notes:
                item = item.split(',')                
                key = int(item[0])
                value = str(item[1]).strip()
                if checkbox == key:
                    current_value = value
            new_name = shorten_field + ':' + current_value
            new_name = new_name.replace('(specify below)','').strip()
            df = df.rename(columns={field: new_name})

    df.to_csv('baseline-survey-data.csv', index=False)

    print('dd.shape =', dd.shape)
    print('df.shape =', df.shape)



    


