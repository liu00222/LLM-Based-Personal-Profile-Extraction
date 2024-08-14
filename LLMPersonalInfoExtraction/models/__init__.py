from .PaLM2 import PaLM2
from .Vicuna import Vicuna
from .GPT import GPT
from .Gemini import Gemini
from .Llama import Llama
from .Flan import Flan
from .Internlm import Internlm


def create_model(config):
    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        model = PaLM2(config)
    elif provider == 'vicuna':
        model = Vicuna(config)
    elif provider == 'gpt':
        model = GPT(config)
    elif provider == 'gemini':
        model = Gemini(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'flan':
        model = Flan(config)
    elif provider == 'internlm':
        model = Internlm(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model