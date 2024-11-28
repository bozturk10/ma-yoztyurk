# Imports
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, pipeline
import time
import re




def load_llama_model(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    model.name=model_name
    return tokenizer, model

def generate_llama2_response(tokenizer,model,prompt,max_new_tokens=300,temperature=1,do_sample=True):
    begin=time.time()

    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_num_token= len(model_input['input_ids'][0])
    with torch.no_grad():
        output= model.generate(**model_input, 
                                                 max_new_tokens=max_new_tokens, 
                                                 temperature=temperature, 
                                                 do_sample=do_sample,
                                                )[0]
        output_decoded = tokenizer.decode(output,skip_special_tokens=True)
    end=time.time()
    runtime= end-begin
    print('output_decoded',re.sub(r".*\[/INST\]\s*", "", output_decoded))

    print('prompt finished. runtime',runtime)
    return {'model':model.name,
            'runtime':runtime,
            'prompt': prompt,
            'input_num_token': input_num_token,
            'output_num_token':len(output),
            'temperature':temperature,
            'do_sample':do_sample,
            'max_new_tokens':max_new_tokens,
            'output': re.sub(r".*\[/INST\]\s*", "", output_decoded) #output_decoded.replace(prompt, "")  # return the output without the original prompt
           }

# Usage example
def main():
    # Load prompts
    prompt_dir = '/path/to/prompts'
    prompts = load_prompts({
        'prompt1': os.path.join(prompt_dir, 'prompt1.txt'),
        # Add more prompts as needed
    })

    # Prepare data
    data = pd.read_csv('path/to/data.csv')
    # Assume data processing here

    # Load model
    model_name = 'meta-llama/Llama-2-13b-chat-hf'
    tokenizer, model = load_llama_model(model_name)
    model.eval()

    # Generate responses
    for index, row in data.iterrows():
        formatted_prompt = format_prompt(prompts['prompt1'], age=row['age'], gender=row['gender'])
        response = generate_response(model, tokenizer, formatted_prompt, max_new_tokens=50, temperature=1, do_sample=True)
        print(response)

if __name__ == '__main__':
    main()
