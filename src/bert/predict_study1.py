import argparse
import pandas as pd
from bert_classifier import BertClassifier
import os
def main():


    model_name = 'bert_mixed_coarse_resample20240708_195103'
    threshold = 0.5
    model_comparison_experiments= [
    'google-gemma-7b-it_12_1712704376_modified',
    'Llama2_all',
    'mistralai-Mixtral-8x7B-Instruct-v0.1_12_1712772173'
    ]

    
    print(f"Loading model from {model_name}")
    wave_12_folders='/dss/dsshome1/0F/ra46lup2/ma-yoztyurk/outputs/text_generations/12/'
    folders_to_clf=model_comparison_experiments #os.listdir(ablation_experiments)
    folders_to_clf= [os.path.join(wave_12_folders, folder) for folder in folders_to_clf] 
    #folders_to_clf=['/dss/dsshome1/0F/ra46lup2/ma-yoztyurk/outputs/text_generations/12/1VAR_age']
    
    model = BertClassifier(model_name=model_name,device='cuda') 

    for folder in folders_to_clf:
        experiment_data_path = folder
        print(f"Predicting on data from {experiment_data_path}")
        model.classify_experiment_folder(folder_path=experiment_data_path,  threshold=threshold)
        print(f"Predictions saved to {experiment_data_path}")

if __name__ == "__main__":
    main()
