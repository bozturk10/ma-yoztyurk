# Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study  

## Abstract  

Large Language Models (LLMs) are increasingly used for investigating public opinions in recent research. This study investigates the *algorithmic fidelity* of LLMs—their ability to replicate the socio-cultural context and nuanced opinions of human participants. Using open-ended survey data from the German Longitudinal Election Studies (GLES), we prompt different LLMs to generate synthetic public opinions reflective of German subpopulations by incorporating demographic features into the persona prompts and comparing the results to the original survey data.  

Our findings reveal that **Llama2** outperforms other LLMs in representing subpopulations, particularly when opinion diversity within groups is lower. For instance, Llama2 demonstrates higher accuracy in reflecting opinions of supporters of left-leaning parties such as *The Greens* and *The Left* but struggles with right-leaning groups like *AfD*. Variations in prompting, including the inclusion or exclusion of specific variables, significantly affect the model's predictions.  

These results highlight the need for improvements in LLM alignment to robustly model diverse public opinions while reducing political biases.  
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

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/algorithmic-fidelity-llms.git
   cd algorithmic-fidelity-llms


- pip install requirements.txt
- extract zip file into outputs\text_generations
- notebooks/paper_figures.ipynb

- TODO reference to loss func repo, gesis dataset, 
# Data Preprocessing

# Running Experiments
# Evaluation

Contributing
We welcome contributions to extend the evaluation framework or explore other datasets. Please submit a pull request or open an issue for discussion.

License
This project is licensed under the MIT License. See LICENSE for details.

Citation
If you use this work in your research, please cite:

plaintext
Copy code
@article{yourcitation,
  title={Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2024}
}
