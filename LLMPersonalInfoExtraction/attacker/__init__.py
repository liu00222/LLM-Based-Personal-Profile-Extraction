from .Attacker import Attacker

def create_attacker(model, adaptive_attack='no', icl_manager=None, prompt_type='direct_question_answering'):
    return Attacker(model, adaptive_attack, icl_manager, prompt_type)

