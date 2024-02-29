from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel
import torch
from LLMs.config import LED_LARGE_BOOK

class LEDLargeBook:
    _self = None

    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self):
        self.usesPrompts = False
        self.model_name = LED_LARGE_BOOK["model_name"]
        self.model_path = LED_LARGE_BOOK["hf_model_path"]
        self.save_path = LED_LARGE_BOOK["save_path"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache() #Empty GPU cache

        #Define the batch size and load the model's dataset
        torch.utils.data.DataLoader(self.save_path, batch_size=1)

         #Load the model from local storage and infer
        self.local_tokenizer = AutoTokenizer.from_pretrained(self.save_path)
        self.local_model = AutoModelForSeq2SeqLM.from_pretrained(self.save_path,
                                                           return_dict=True,
                                                           trust_remote_code=True,
                                                           device_map="auto",
                                                           torch_dtype=torch.bfloat16)
        if torch.cuda.is_available():
             self.local_model.to("cuda")

        self.pipeline = pipeline("summarization",
                                model=self.local_model,
                                tokenizer=self.local_tokenizer,
                                torch_dtype=torch.bfloat16) 
        
    def getName(self):
        return self.model_name

    def run(self, input):
        """
            To improve summary quality, use encoder_no_repeat_ngram_size=3 when calling the pipeline object. 
            This setting encourages the model to utilize new vocabulary and construct an abstractive summary.
        """
        response = self.pipeline(
            input, 
            max_length=130, 
            min_length=30,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            repetition_penalty=3.5,
            num_beams=4,
            early_stopping=True
        )
        return response