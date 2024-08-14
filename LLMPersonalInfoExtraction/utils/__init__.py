import PIL.Image

from .process_config import open_config, print_config, open_json
from .process_txt import open_txt
from .parser import HTMLParser, parsed_data_to_string


def get_parser(dataset, include_link=None):
    if dataset in ('synthetic'):
        return HTMLParser(('p', 'h1', 'h2', 'li'), True if include_link is None else include_link)
    else:
        return HTMLParser(('p', 'h1'), False if include_link is None else include_link)

def load_instruction(type, info_cats):
    try:
        raw_lines = open_txt(f'./data/system_prompts/{type}.txt')
        instruction_map = dict()
        for i in range(len(raw_lines)):
            curr_raw_line = raw_lines[i]
            curr_info = curr_raw_line.split(':')[0]
            curr_instruction = curr_raw_line.split(':')[1:][0]
            assert (curr_info in info_cats and curr_info not in instruction_map.keys())
            instruction_map[curr_info] = curr_instruction
        for info_cat in info_cats:
            assert (info_cat in instruction_map)
        return instruction_map

    except:
        raise NotImplementedError(f'ERROR: {type} instruction is not supported')
    
def remove_symbols(t):
    symbols = [',', '.', '!', '?', ';', ':', '(', ')', '/', '[', ']', '*', '#', '^', '%', '&']
    for s in symbols:
        t = t.replace(f'{s} ', ' ')
        t = t.replace(f'{s}', '')
    return t

def load_image(image_path):
    try:
        img = PIL.Image.open(image_path)
    except PIL.UnidentifiedImageError:
        img = None
    return img