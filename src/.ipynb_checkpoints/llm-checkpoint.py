from transformers import AutoModelForMaskedLM, AutoTokenizer

class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=100, temperature=0.7)
        response = self.tokenizer.decode(outputs[0])
        return response

# Usage:
bert = LanguageModel('bert-base-german-cased')
response = bert.generate_response('Hallo, wie geht es dir?')
print(response)