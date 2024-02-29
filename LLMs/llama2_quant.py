from llama_cpp import Llama
from LLMs.config import LLAMA2

class Llamma7b:
    _self = None

    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self):
        self.usesPrompts = True
        self.model_name = LLAMA2["model_name"]
        self.model_path = LLAMA2["hf_model_path"]
        self.save_path = LLAMA2["save_path"]
        self.model = Llama(model_path=self.save_path,
            n_ctx=512, n_batch=32, verbose=False)

    def getName(self):
        return self.model_name

    def run(self, input):
        system = """
        You are tasked with the job of sumarizing text books for 5th grade students.
        Do as good of a sumarization as you can.
        """
        prompt = f"### System:\n{system}\n\n### User:\n{input}\n\n### Response: \n"
        output = self.model(prompt,temperature = 0.7, max_tokens=512,top_k=20, top_p=0.9, repeat_penalty=1.15)
        res = output['choices'][0]['text'].strip()
        return res

