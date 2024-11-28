import os
import json
import os
import json
import pandas as pd

# Base directory where you want to save the files
base_dir = 'results'
month_names_german = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember"
}

def format_prompt(prompt_fpath,row):
    '''
    row needs age,gender,party,eastwest and year variables 
    '''
    start_date=pd.to_datetime(row.field_start)
    year=start_date.year
    month=month_names_german[start_date.month]
    prompt = open(prompt_fpath, "r").read().replace('\n', '') + '\n'
    artikel = 'Die' if row['gender'] == 'weiblich' else "Der"
    artikel2= 'die' if row['gender'] == 'weiblich' else "der"
    return prompt.format(month=month,artikel=artikel,artikel2=artikel2, age=row['age'], gender=row['gender'], party=row['leaning_party'], eastwest=row['ostwest'], year=year,education_clause=row['education_clause'])


def preprocess_sample_df(sample_fpath,prompt_fpath):
    sample_df = pd.read_csv(sample_fpath)
    sample_df.age= sample_df.age.astype(int)
    # Function to format the string
    sample_df['formatted_prompt'] = sample_df.apply(lambda row: format_prompt(prompt_fpath,row), axis=1)
    return sample_df

def get_experiment_log(row, survey_wave,model_output):
    """
    Generates a dictionary containing logging information for a model's response.
    """
    log_dict= {
        "survey_wave": survey_wave,
        "user_id": row['lfdn'],
    }
    log_dict.update(model_output)
    
    return log_dict

def save_experiment_log(row, log_dict, experiment_num):
    """
    Saves model answers in JSON format within an experiment subdirectory.
    """

    
    experiment_dir = os.path.join(base_dir, experiment_num)
    os.makedirs(experiment_dir, exist_ok=True)
    
    filename = f"{row['lfdn']}.json"
    file_path = os.path.join(experiment_dir, filename)
    
    if os.path.exists(file_path):
        print(f"File already exists, skipping: {file_path}")
        return  # Skip this row if file exists


    # Save the model answer to the file in JSON format
    with open(file_path, 'w') as file:
        json.dump(log_dict, file, ensure_ascii=False, indent=4)
    
    print(f"Saved: {file_path}")

# Example usage
def main():
    # Example row data (replace with actual data)
    row = {'lfdn': '12345', 'formatted_prompt': 'Example prompt'}
    model_name = 'llama_v2'
    survey_wave = 21
    model_output={}
    
    log_dict = get_experiment_log(row, model_name, survey_wave,model_output)

    # Save the model answers
    save_experiment_log(row, log_dict,experiment_num)

def translate():
    from deep_translator import GoogleTranslator
    gt= GoogleTranslator(source='de', target='en')
    # Function to translate text from German to English
    def translate_to_english(text):
        if text :
            return gt.translate(text) 
        else:
            return None

# Translate the text in the DataFrame
links['education_clause_eng'] = links['education_clause'].apply(translate_to_english)
links['2330_combined_eng'] = links['2330_combined'].apply(translate_to_english)




if __name__ == '__main__':
    main()

    
