# Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study  

## Abstract  


## Dataset
Downloading the raw survey dataset is possible for academic research and teaching, after registration at GESIS- Leibniz Institute for the Social Sciences.
Please find the dataset under Downloads > Datasets and then download 
- ZA6838_v6-0-0.dta.zip (under  data/raw)
- ZA6838_openended_alleWellen_v6-0-0.csv.zip (under data/raw/open_ended)
https://search.gesis.org/research_data/ZA6838?doi=10.4232/1.14114



## Repository Overview  

This repository provides:  
- The **codebase** for replicating the experiments presented in the paper.  
- **Datasets** and preprocessing scripts for the German Longitudinal Election Studies (GLES).  
- Documentation on experimental results and findings.  

## Installation  


- pip install requirements.txt
- extract GESIS data zip file into data\raw
- run python src/data/education_preprocess.py create data/coding_values/education_lookup.csv for getting education levels of participants at the date of surveys.
- to replicate the experiments :
    - adjust `experiment_config.json` depending on your device (device,quantization,HF models) 
    -  you can find the config used in the experimentation `experiment_config_paper.json `
    - run `python  experiment.py --config=experiment_config_paper.json`
    - resulting generations will be stored in `outputs/text_generations/<wave_number>/<experiment_name>`
- to train the text classifier:
    - `python src/bert/train.py --class_mode coarse --dataset mixed --model_name bert_classifier --loss_type dbloss`
    - the BERT model will be saved under `models/`
- to run the text classifier on the experiments:
    -  `python src/bert/predict.py --experiment_data_path EXPERIMENT_DATA_PATH  --model_name bert_classifier`
    - then the text classification results will be appended to the outputs under EXPERIMENT_DATA_PATH.
- to replicate analyssis and the figures:
    - run the notebook `notebooks/paper_figures.ipynb`


# References
The used resampling loss implementation is taken from https://github.com/Roche/BalancedLossNLP/blob/main/Reuters/util_loss.py

