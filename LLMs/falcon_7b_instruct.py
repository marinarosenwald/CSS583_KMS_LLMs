from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch
from LLMs.config import FALCON_7B

class Falcon7b:
    _self = None

    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self):
        self.usesPrompts = True
        self.model_name = FALCON_7B["model_name"]
        self.model_path = FALCON_7B["hf_model_path"]
        self.save_path = FALCON_7B["save_path"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache() #Empty GPU cache

        #Define the batch size and load the model's dataset
        torch.utils.data.DataLoader(self.save_path, batch_size=1)

        #Load the model from local storage and infer
        self.local_tokenizer = AutoTokenizer.from_pretrained(self.save_path)
        self.local_model = AutoModelForCausalLM.from_pretrained(self.save_path,
                                                           return_dict=True,
                                                           trust_remote_code=True,
                                                        #    device_map="auto",
                                                           torch_dtype=torch.bfloat16)
        if torch.cuda.is_available():
             self.local_model.to("cuda")

        self.pipeline = pipeline("text-generation",
                                 model=self.local_model,
                                 tokenizer=self.local_tokenizer,
                                 torch_dtype=torch.bfloat16)

    def getName(self):
        return self.model_name

    def run(self, input):
        # system = f"""
        # You are tasked with the job of sumarizing text books for 5th grade students.
        # Do as good of a sumarization as you can.
        # """
        # prompt = f"#### System: {system}\n#### User: \n{input}\n\n#### Response from {self.model_name}"
        falcon_response = self.pipeline(input,
                                    max_length=500,
                                    do_sample=True,
                                    top_k=10,
                                    num_return_sequences=1,
                                    eos_token_id=self.local_tokenizer.eos_token_id)

        return falcon_response[0]['generated_text']