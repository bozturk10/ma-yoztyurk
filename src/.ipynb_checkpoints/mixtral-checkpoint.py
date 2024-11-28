
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, pipeline
import time
import re

os.environ['HF_HOME'] = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra46lup2/.cache/huggingface'

os.environ['HUGGINGFACE_HUB_CACHE'] = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra46lup2/.cache/huggingface/hub'
print(os.environ['HUGGINGFACE_HUB_CACHE'])
print(os.environ['HF_HOME'])

class MixtralInstruct:
    def __init__(self, model_name,device):
        self.tokenizer, self.model,self.device = self.load_model(model_name)

    def load_model(self, model_name,device='cuda'):
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir = os.environ['HUGGINGFACE_HUB_CACHE'])

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )


        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,cache_dir = os.environ['HUGGINGFACE_HUB_CACHE'])
        elif device == 'cuda':

            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,cache_dir = os.environ['HUGGINGFACE_HUB_CACHE'])
        else:
            raise ValueError('device should be either cpu or cuda')
        model.name = model_name
        return tokenizer, model,device

    def generate_response(self, prompt, max_new_tokens=300, temperature=1, do_sample=True):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=50)
        print(self.tokenizer.decode(output[0], skip_special_tokens=True))


        begin = time.time()
        if self.device =='cpu':
            model_input = self.tokenizer(prompt, return_tensors="pt")
        else:
            model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_num_token = len(model_input['input_ids'][0])


        with torch.no_grad():
            output = self.model.generate(**model_input, 
                                         max_new_tokens=max_new_tokens, 
                                         temperature=temperature, 
                                         do_sample=do_sample)[0]
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
                'temperature': temperature,
                'do_sample': do_sample,
                'max_new_tokens': max_new_tokens,
                'output': re.sub(r".*\[/INST\]\s*", "", output_decoded)  # output_decoded.replace(prompt, "")  # return the output without the original prompt
               }

# Usage example
def main():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    device = 'cuda'
    mixtral = MixtralInstruct(model_name, device)
    prompt = "[INST] how are you doing [/INST]"
    response = mixtral.generate_response(prompt)
    print(response)
    


if __name__ == '__main__':
    main()

