import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import re
from dotenv import load_dotenv

load_dotenv()
print("helloo")
HF_HOME = os.environ.get("HF_HOME")
HUGGINGFACE_HUB_CACHE = os.environ.get("HUGGINGFACE_HUB_CACHE") if os.environ.get("HUGGINGFACE_HUB_CACHE")!=""  else None

class HFTextGenerator:
    def __init__(self, model_name, device, **model_config):
        self.tokenizer, self.model, self.device = self.load_model(model_name, device, **model_config)

    def load_model(self, model_name, device, **model_config):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HUGGINGFACE_HUB_CACHE)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=HUGGINGFACE_HUB_CACHE, **model_config)
        model.name = model_name
        return tokenizer, model, device

    def generate_response(self, prompt, generation_config):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs)
        print(self.tokenizer.decode(output[0], skip_special_tokens=True))


        begin = time.time()
        if self.device =='cpu':
            model_input = self.tokenizer(prompt, return_tensors="pt")
        else:
            model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_num_token = len(model_input['input_ids'][0])


        with torch.no_grad():
            output = self.model.generate(**model_input,**generation_config)[0]
            output_decoded = self.tokenizer.decode(output, skip_special_tokens=True)
        end = time.time()
        runtime = end - begin
        print('output_decoded', re.sub(r".*\[/INST\]\s*", "", output_decoded))
        print('output_decoded2',output_decoded)

        print('prompt finished. runtime', runtime)
        return {'model': self.model.name,
                'runtime': runtime,
                'prompt': prompt,
                'input_num_token': input_num_token,
                'output_num_token': len(output),
                'output': re.sub(r".*\[/INST\]\s*", "", output_decoded)  # output_decoded.replace(prompt, "")  # return the output without the original prompt
               }.update(generation_config)

def main():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    device = 'cuda'
    model_config = {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    }
    generation_config = {
        "max_new_tokens": 50,
        "temperature": 1,
        "do_sample": True
    }
    mixtral = HFTextGenerator(model_name, AutoModelForCausalLM, device, **model_config)
    response = mixtral.generate_response(prompt,generation_config)
    print(response)

if __name__ == '__main__':
    main()
