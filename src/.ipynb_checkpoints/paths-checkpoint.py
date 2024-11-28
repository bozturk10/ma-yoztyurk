import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
PROMPT_DIR = os.path.join(PROJECT_DIR, 'prompts')
GLES_DIR=  os.path.join(RAW_DATA_DIR,"GLES") 
CODING_DIR = os.path.join(PROJECT_DIR,'data','coding_values') 
MODELS_DIR = os.path.join(PROJECT_DIR,'models')
OPEN_ENDED_GLES_DR = os.path.join(GLES_DIR, 'open_ended')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results2')
if __name__ == "__main__":
    print("PROJECT_DIR:",PROJECT_DIR)
    print("RAW_DATA_DIR:",RAW_DATA_DIR)
    print("PROCESSED_DATA_DIR:",PROCESSED_DATA_DIR)
    print("GLES_DIR:",GLES_DIR)
    print("CODING_DIR:",CODING_DIR)
    print("PROMPT_DIR:",PROMPT_DIR)
    print("MODELS_DIR:",MODELS_DIR)

