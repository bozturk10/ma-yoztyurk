import argparse
import pandas as pd
from paths import TEXT_GEN_DIR
from bert.bert_classifier import BertClassifier
import os

def clasify_experiments(    wave_id,threshold=0.5,device='cpu'):
    wave_path= os.path.join(TEXT_GEN_DIR, wave_id) 
    experiments_to_clf=['Llama2_model_opinion']
    experiments_to_clf= [os.path.join(wave_path, exp) for exp in experiments_to_clf] 
    
    model = BertClassifier(model_name='bert_mixed_coarse_resample20240708_195103',device=device) 

    for folder in experiments_to_clf:
        experiment_data_path = folder
        print(f"Predicting on data from {experiment_data_path}")
        model.classify_experiment_folder(folder_path=experiment_data_path,  threshold=threshold)
        print(f"Predictions saved to {experiment_data_path}")

if __name__ == "__main__":
    clasify_experiments(wave_id="12",device='cpu')
