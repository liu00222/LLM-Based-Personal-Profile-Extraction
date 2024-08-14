from .NoDefense import NoDefense
from .SymbolReplacementDefense import SymbolReplacementDefense
from .HyperLinkDefense import HyperLinkDefense
from .MaskDefense import MaskDefense
from .PromptInjectionDefense import PromptInjectionDefense

def create_defense(defense):
    """
    Factory method to create a defense
    """
    if defense in ('no', 'image'):
        return NoDefense(defense)
    
    elif 'replace' in defense:
        type = '_'.join(defense.split('_')[1:])
        return SymbolReplacementDefense(defense, type)

    elif defense == 'hyperlink':
        return HyperLinkDefense(defense)
    
    elif defense == 'mask':
        return MaskDefense(defense)
    
    elif 'pi' in defense:
        type = '_'.join(defense.split('_')[1:])
        return PromptInjectionDefense(defense, type)
    
    err_msg = f"{defense} is not a valid defense strategy."
    err_msg = f"{err_msg}\nValid defense strategy is one of ['no', 'replace_at', 'replace_at_dot', 'replace_dot', 'hyperlink', 'mask', 'pi_ci', 'pi_id', 'pi_ci_id']"
    raise ValueError(err_msg)