from experiment.HFTextGenerator import HFTextGenerator
from experiment.experiment_utils import get_experiment_config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

EXPERIMENT_CONFIG_PATH = r"C:\Users\BerkÖztürk\ma-yoztyurk\experiment_config.json"
config = get_experiment_config(EXPERIMENT_CONFIG_PATH)
quantization_config = config["quantization_config"]
generation_config = config["generation_config"]
remove_tag_fnc = config.get("remove_tag_fnc")

model = HFTextGenerator("google/gemma-2-2b-it", "cuda", quantization_config)
model.generate_response("hello who are you?", generation_config, remove_tag_fnc)




