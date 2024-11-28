import torch
torch.cuda.empty_cache()
from src.mixtral import MixtralInstruct

import os,time
from src.data.process_data import process_open_ended, process_wave_data
from src.data.read_data import load_raw_survey_data, read_stata_file
from src.paths import CODING_DIR, GLES_DIR, PROCESSED_DATA_DIR, PROJECT_DIR, PROMPT_DIR, RAW_DATA_DIR, RESULTS_DIR
from src.utils import format_prompt, get_experiment_log,save_experiment_log
from src.HFTextGenerator import HFTextGenerator
from transformers import  BitsAndBytesConfig
wave_number = 12
prompt_fpath=os.path.join(PROMPT_DIR,'12_gemma_prompt.txt')  
wave_df, wave_open_ended_df, df_coding_840s = load_raw_survey_data(wave_number)
wave_open_ended_df = process_open_ended(wave_open_ended_df, df_coding_840s, wave_number)
wave_df_processed = process_wave_data(wave_df, wave_open_ended_df, wave_number)
wave_df_processed['formatted_prompt'] = wave_df_processed.apply(lambda row: format_prompt(prompt_fpath,row), axis=1)

#experiment_num=17
model_name = 'google/gemma-7b-it'
#experiment_num as model_name + wave_number + start_time
experiment_num = f"{model_name.replace('/','-')}_{wave_number}_{int(time.time())}"
experiment_num = 'google-gemma-7b-it_12_1712704376'
experiment_dir = os.path.join(RESULTS_DIR, str(experiment_num))
print(f"Saving logs to {experiment_dir}")
device = 'cuda'
generation_config = {
        "max_new_tokens": 300,
        "temperature": 1,
        "do_sample": True
    }
model_config = {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    }



mixtral = HFTextGenerator(model_name, device,**model_config)
experiment_dir = os.path.join(RESULTS_DIR, str(experiment_num))
print(f"Saving logs to {experiment_dir}")
os.makedirs(experiment_dir, exist_ok=True)
completed_lfdn_ids = os.listdir(experiment_dir)

def run_experiment(wave_df_processed, mixtral, experiment_dir, completed_lfdn_ids, generation_config):
    logs_to_save = []
    n=10
    saved_batch_count = 0
    for index, row in wave_df_processed.iterrows():
        filename = f"{row['lfdn']}.json"
        survey_wave = row.study #wave id
        if filename in completed_lfdn_ids:
            print(f'skipping {row["lfdn"]} as it already exists.')
        else:
            model_output= mixtral.generate_response(row.formatted_prompt, generation_config)
            log = get_experiment_log(row, survey_wave,model_output)
            logs_to_save.append(log)

        # Save logs every n generations
        if (index + 1) % n == 0:
            for log in logs_to_save: 
                save_experiment_log(log['user_id'],log, experiment_dir)
            saved_batch_count += 1
            print(f"Saved {saved_batch_count} x 10 logs")
            logs_to_save = []  # Clear logs_to_save
    for log in logs_to_save:
            save_experiment_log(log['user_id'],log, experiment_dir)    

if __name__ == "__main__":
    run_experiment(wave_df_processed, mixtral, experiment_dir, completed_lfdn_ids, generation_config)