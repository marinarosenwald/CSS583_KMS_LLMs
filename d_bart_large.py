from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel
import transformers
import torch
from LLMs.config import BART_LARGE


#Specify the model you want to download from Hugging Face
hf_model_path = BART_LARGE["hf_model_path"]

#Enter your local directory you want to store the model in
save_path = BART_LARGE["save_path"]

#Load the model and store it 
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_path, return_dict=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",)

if torch.backends.mps.is_available():
    model.to("cpu") # fix for using apple M1 chip

#Save the model and the tokenizer in the local directory specified earlier
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print ("here")