import os
import re
import torch
from src.data.process_data import process_open_ended, process_wave_data
from src.data.read_data import load_raw_survey_data, read_stata_file
from src.paths import CODING_DIR, GLES_DIR, PROCESSED_DATA_DIR, PROJECT_DIR, PROMPT_DIR, RAW_DATA_DIR, RESULTS_DIR
from src.utils import format_prompt, get_experiment_log,save_experiment_log
from src.HFTextGenerator import HFTextGenerator
from transformers import  BitsAndBytesConfig
import json
from dotenv import load_dotenv

def llama2_remove_tag(output_decoded):
    return re.sub(r".*\[/INST\]\s*", "", output_decoded)

def gemma_remove_tag(output_decoded):
    split_output = output_decoded.split("model\n", 1)
    without_tags = split_output[1] if len(split_output) > 1 else ""
    return without_tags

def mixtral_remove_tag(output_decoded):
    return re.sub(r".*\[/INST\]\s*", "", output_decoded)

def get_experiment_config(fpath):

    with open(fpath) as f:
        config = json.load(f)

    config['prompt_fpath']=os.path.join(PROMPT_DIR,config['prompt_fpath'])  

    model_name = config['model_name']
    if config['device'] == 'cuda':
        if 'bnb_4bit_compute_dtype' in config['quantization_config']:
            dtype_str = config['quantization_config']['bnb_4bit_compute_dtype']
            if dtype_str == 'float16':
                dtype = torch.float16
            elif dtype_str == 'float32':
                dtype = torch.float32
            
            config['quantization_config']['bnb_4bit_compute_dtype'] = dtype
        
        config['quantization_config'] = BitsAndBytesConfig(**config['quantization_config'])
    if 'mixtral' in model_name:
        config['remove_tag_fnc'] = mixtral_remove_tag
    elif 'llama2' in model_name:
        config['remove_tag_fnc'] = llama2_remove_tag
    elif 'gemma' in model_name:
        config['remove_tag_fnc'] = gemma_remove_tag

    return config

def run_experiment(wave_df_processed, model, experiment_dir, generation_config,batch_size=10):
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving logs to {experiment_dir}")
    completed_lfdn_ids = os.listdir(experiment_dir)
    logs_to_save = []
    saved_batch_count = 0
    for index, row in wave_df_processed.iterrows():
        filename = f"{row['lfdn']}.json"
        survey_wave = row.study #wave id
        if filename in completed_lfdn_ids:
            print(f'skipping {row["lfdn"]} as it already exists.')
        else:
            model_output= model.generate_response(row.formatted_prompt, generation_config,gemma_remove_tag)
            log = get_experiment_log(row, survey_wave,model_output)
            logs_to_save.append(log)

        if (index + 1) % batch_size == 0:
            for log in logs_to_save: 
                save_experiment_log(log['user_id'],log, experiment_dir)
            saved_batch_count += 1
            print(f"Saved {saved_batch_count} x 10 logs")
            logs_to_save = []  # Clear logs_to_save
    for log in logs_to_save:
            save_experiment_log(log['user_id'],log, experiment_dir)    

load_dotenv('/dss/dsshome1/0F/ra46lup2/ma-yoztyurk/experiment_berk.env')
# print found environment variables
print(os.environ.get("HF_HOME"))
print(os.environ.get("HUGGINGFACE_HUB_CACHE"))
print(os.environ.get("EXPERIMENT_CONFIG_PATH"))

torch.cuda.empty_cache()

EXPERIMENT_CONFIG_PATH = os.getenv("EXPERIMENT_CONFIG_PATH")

config = get_experiment_config(EXPERIMENT_CONFIG_PATH)
print(f'Using config: {config}')

wave_number = config['wave_number']
prompt_fpath=os.path.join(PROMPT_DIR,config['prompt_fpath'])
device = config['device']
model_name = config['model_name']
generation_config = config['generation_config']
model_config = config
remove_tag_fnc = config['remove_tag_fnc']




wave_df, wave_open_ended_df, df_coding_840s = load_raw_survey_data(wave_number)
wave_open_ended_df = process_open_ended(wave_open_ended_df, df_coding_840s, wave_number)
wave_df_processed = process_wave_data(wave_df, wave_open_ended_df, wave_number)
wave_df_processed['formatted_prompt'] = wave_df_processed.apply(lambda row: format_prompt(prompt_fpath,row), axis=1)
experiment_num = f"{model_name.replace('/','-')}_{wave_number}_{generation_config['temperature']}"
experiment_dir = os.path.join(RESULTS_DIR, str(experiment_num))



model = HFTextGenerator(model_name, device,**model_config)




if __name__ == "__main__":
    run_experiment(wave_df_processed, model, experiment_dir, generation_config)