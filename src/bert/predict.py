import argparse
import pandas as pd
from bert_classifier import BertClassifier
import os
def main():


    model_name = 'bert_mixed_coarse_resample20240708_195103'
    threshold = 0.5
    ablation_experiments= [
    '1VAR_berufabschluss',
    '1VAR_eastwest',
    '1VAR_gender',
    '1VAR_party',
    '1VAR_schulabschluss',
    'Llama2_all',
    'Llama2_base',
    'without_age',
    'without_berufabschluss',
    'without_eastwest',
    'without_gender',
    'without_party',
    'without_schulabschluss']

    
    print(f"Loading model from {model_name}")
    wave_12_folders='/dss/dsshome1/0F/ra46lup2/ma-yoztyurk/outputs/text_generations/12/'
    folders_to_clf=['Llama2_model_opinion']#ablation_experiments #os.listdir(ablation_experiments)
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
