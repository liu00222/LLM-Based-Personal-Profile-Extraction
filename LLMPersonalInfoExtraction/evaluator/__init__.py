from .Evaluator import Evaluator

def create_evaluator(model_provider, info_cats, metric_1='acc', metric_2='rouge1'):
    """
    metric_1 is used for email address and phone number.
    metric_2 is used for other categories.
    """
    return Evaluator(model_provider, info_cats, metric_1, metric_2)
