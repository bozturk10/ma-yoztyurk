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

class Llama2Model:
    def __init__(self, model_name,device):
        self.tokenizer, self.model,self.device = self.load_llama_model(model_name)

    def load_llama_model(self, model_name,device='cuda'):
        tokenizer = LlamaTokenizer.from_pretrained(model_name,cache_dir=os.environ['HUGGINGFACE_HUB_CACHE'] )
        if device == 'cpu':
            model = LlamaForCausalLM.from_pretrained(model_name,device_map=device, torch_dtype=torch.float16,cache_dir=os.environ['HUGGINGFACE_HUB_CACHE'] )
        elif device == 'cuda':
            model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map= device, torch_dtype=torch.float16,cache_dir=os.environ['HUGGINGFACE_HUB_CACHE'] )
        else:
            raise ValueError('device should be either cpu or cuda')
        model.name = model_name
        return tokenizer, model,device

    def generate_llama2_response(self, prompt, max_new_tokens=300, temperature=1, do_sample=True):
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

    model_name = 'meta-llama/Llama-2-13b-chat-hf'
    llama2 = Llama2Model(model_name,device='cuda')
    sample="<s>[INST] <<SYS>> Identifizieren Sie das wichtigste Problem, mit dem Deutschland im November 2019 konfrontiert ist. Geben Sie die Antwort in einem prägnanten Satz an, konzentrieren Sie sich nur auf ein einziges Thema ohne weitere Ausführungen oder Auflistung zusätzlicher Probleme. Wiederholen Sie nicht die Informationen die Ihnen gegeben wurden, und geben Sie Ihre Antwort direkt und ohne einleitende Phrasen. Antworten Sie auf Deutsch und ausschließlich auf Deutsch, verwenden Sie keine Englische Sprache. Antworten Sie aus der Sicht eines Befragten mit deutscher Staatsbürgerschaft und den im nachfolgenden spezifizierten Eigenschaften.<</SYS>>Die Befragte ist 58 Jahre alt und weiblich. Sie hat einen Realschulabschluss und hat einen Berufsfachschulabschluss. Sie lebt in Westdeutschland und unterstützt hauptsächlich die CDU/CSU. [/INST]\n"
    response = llama2.generate_llama2_response(sample)
    print(response)

if __name__ == '__main__':
    main()
    