from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import transformers
import torch

from LLMs.config import FALCON_7B

#Specify the model you want to download from Hugging Face
hf_model_path = FALCON_7B["hf_model_path"]

#Enter your local directory you want to store the model in
save_path = FALCON_7B["save_path"]

#Load the model and store it 
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
model = AutoModelForCausalLM.from_pretrained(hf_model_path, return_dict=True, trust_remote_code=True, torch_dtype=torch.bfloat16, )

# if torch.backends.mps.is_available():
#     model.to("msp") # fix for using apple M1 chip

#Save the model and the tokenizer in the local directory specified earlier
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
