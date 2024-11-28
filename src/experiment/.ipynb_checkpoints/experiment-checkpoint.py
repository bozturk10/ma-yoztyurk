import argparse
import os

import torch
from dotenv import load_dotenv
import logging
from src.data.process_data import process_open_ended, process_wave_data
from src.data.read_data import load_raw_survey_data
from src.experiment.experiment_utils import get_experiment_config
from src.HFTextGenerator import HFTextGenerator
from src.paths import (
    PROJECT_DIR,
    PROMPT_DIR,
    RESULTS_DIR,
)
from src.utils import format_prompt, get_experiment_log, save_experiment_log

def experiment_setup():
    dotenv_path = os.path.join(PROJECT_DIR, 'experiment_berk.env')
    load_dotenv(dotenv_path)
    print(os.environ.get("HF_HOME"))
    print(os.environ.get("HUGGINGFACE_HUB_CACHE"))

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('config', type=str, help='The path to the experiment configuration file')
    args = parser.parse_args()
    print(f'Using config: {args.config}')
    EXPERIMENT_CONFIG_PATH = args.config

    config = get_experiment_config(EXPERIMENT_CONFIG_PATH)
    print(f'Using config: {config}')

    wave_number = config['wave_number']
    prompt_fpath=os.path.join(PROMPT_DIR,config['prompt_fpath'])
    device = config['device']
    model_name = config['model_name']
    generation_config = config['generation_config']
    quantization_config = config['quantization_config']
    remove_tag_fnc = config['remove_tag_fnc']

    if config.get('experiment_results_folder') is not None:
        experiment_results_folder = config['experiment_results_folder']
    else:
        experiment_results_folder = f"{model_name.replace('/','-')}_{wave_number}_{generation_config['temperature']}"

    experiment_dir = os.path.join(RESULTS_DIR, experiment_results_folder)


    return wave_number, prompt_fpath, device, model_name, generation_config, quantization_config, remove_tag_fnc,experiment_dir



def run_experiment(wave_df_processed, model, experiment_dir, generation_config,remove_tag_fnc,batch_size=10):
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving logs to {experiment_dir}")
    logs_to_save = []
    saved_batch_count = 0
    for index, row in wave_df_processed.iterrows():
        completed_lfdn_ids = os.listdir(experiment_dir)
        filename = f"{row['lfdn']}.json"
        survey_wave = row.study #wave id
        if filename in completed_lfdn_ids:
            print(f'skipping {row["lfdn"]} as it already exists.')

            logger.info(f'skipping {row["lfdn"]} as it already exists.')
        else:
            model_output= model.generate_response(row.formatted_prompt, generation_config,remove_tag_fnc)
            log = get_experiment_log(row, survey_wave,model_output)
            logs_to_save.append(log)

        if (index + 1) % batch_size == 0:
            for log in logs_to_save: 
                save_experiment_log(log['user_id'],log, experiment_dir)
            saved_batch_count += 1
            print(f"Saved {saved_batch_count} x 10 logs")
            logger.info(f"Saved {saved_batch_count} x 10 logs")
            logs_to_save = []  # Clear logs_to_save
    
    for log in logs_to_save:
            save_experiment_log(log['user_id'],log, experiment_dir)    


#setup
wave_number, prompt_fpath, device, model_name, generation_config, quantization_config, remove_tag_fnc,experiment_dir = experiment_setup()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#data prep
wave_df, wave_open_ended_df, df_coding_840s = load_raw_survey_data(wave_number)
wave_open_ended_df = process_open_ended(wave_open_ended_df, df_coding_840s, wave_number)
wave_df_processed = process_wave_data(wave_df, wave_open_ended_df, wave_number)
wave_df_processed['formatted_prompt'] = wave_df_processed.apply(lambda row: format_prompt(prompt_fpath,row), axis=1)



#model init
model = HFTextGenerator(model_name, device,quantization_config)




if __name__ == "__main__":
    run_experiment(wave_df_processed, model, experiment_dir, generation_config,remove_tag_fnc)