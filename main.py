import argparse
import LLMPersonalInfoExtraction as PIE
from LLMPersonalInfoExtraction.utils import open_config
from LLMPersonalInfoExtraction.utils import open_txt, load_image
from LLMPersonalInfoExtraction.utils import load_instruction, parsed_data_to_string
import time
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIE Execution')
    parser.add_argument('--model_config_path', default='./configs/model_configs/palm2_config.json', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--task_config_path', default='./configs/task_configs/synthetic.json', type=str)
    parser.add_argument('--api_key_pos', default=0, type=int)
    parser.add_argument('--defense', default='no', type=str)
    parser.add_argument('--prompt_type', default='direct', type=str)
    parser.add_argument('--gpus', default='', type=str)
    parser.add_argument('--icl_num', default=0, type=int)
    parser.add_argument('--adaptive_attack', default='no', type=str)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--redundant_info_filtering', default='True', type=str)
    args = parser.parse_args()

    task_config = open_config(config_path=args.task_config_path)
    task_manager, icl_manager = PIE.create_task(task_config)

    defense = PIE.create_defense(args.defense)

    model_config = open_config(config_path=args.model_config_path)
    model_config['model_info']['name'] = args.model_name
    if 'palm' in args.model_config_path or 'gemini' in args.model_config_path or 'gpt' in args.model_config_path:
        assert (0 <= args.api_key_pos < len(model_config["api_key_info"]["api_keys"]))
        model_config["api_key_info"]["api_key_use"] = args.api_key_pos
        print(f'API KEY POS = {model_config["api_key_info"]["api_key_use"]}')
    else:
        args.gpus = args.gpus.split(',')
        assert (args.gpus is not [])
        model_config["params"]["gpus"] = args.gpus
        print(f'GPUS = {model_config["params"]["gpus"]}')

    model = PIE.create_model(config=model_config)
    model.print_model_info()
    attacker = PIE.create_attacker(model, adaptive_attack=args.adaptive_attack, icl_manager=icl_manager, prompt_type=args.prompt_type)

    info_cats = open_txt('./data/system_prompts/info_category.txt')
    evaluator = PIE.create_evaluator(model.provider, info_cats)
    instructions = load_instruction(args.prompt_type, info_cats)

    need_adaptive_attack = (args.defense in ('pi_ci', 'pi_id', 'pi_ci_id', 'no'))
    if not need_adaptive_attack:  args.adaptive_attack = 'no'
    email_only = defense.defense not in ['no', 'pi_ci', 'pi_id', 'pi_ci_id', 'image']
    
    if args.redundant_info_filtering == 'True':
        res_save_path = f'./result/{model.provider}_{model.name.split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}'
    else:
        res_save_path = f'./result/{model.provider}_{model.name.split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}_{args.redundant_info_filtering}'
    os.makedirs(res_save_path, exist_ok=True)

    all_raw_responses = dict(zip(info_cats, [[] for _ in range(len(info_cats))]))
    all_labels = dict(zip(info_cats, [[] for _ in range(len(info_cats))]))

    for i in range(len(task_manager)):
        raw_list, curr_label = task_manager[i]
        try:
            raw_list = defense.apply(raw_list, curr_label)
        except ValueError:
            print('Not applicable. Skip')
            continue
        
        if args.verbose > 0:
            print(f'{i+1} / {len(task_manager)}: {curr_label["name"]}')

        if args.defense == 'image':
            if model.provider == 'gemini':
                img = load_image(f'./data/synthetic_images/{curr_label["name"]}.jpg')
                if img is None:
                    print(f'Skip bad image: ./data/synthetic_images/{curr_label["name"]}.jpg\n')
                    continue
            elif model.provider == 'gpt':
                img = f'./data/synthetic_images/{curr_label["name"]}.jpg'
            else:
                raise ValueError
        else:
            img = None

        raw = '\n'.join(raw_list)

        if args.redundant_info_filtering == 'True':
            redundant_info_filter = PIE.get_parser(task_manager.dataset)
            redundant_info_filter.feed(raw)
            processed_data = redundant_info_filter.data
            try:
                processed_data = defense.apply(parsed_data_to_string(task_manager.dataset, processed_data, model.name), curr_label)
            except ValueError:
                print('Not applicable. Skip')
                continue
        else:
            processed_data = raw

        cnt = 0
        for info_cat, instruction in instructions.items():
            if email_only and info_cat != 'email':
                continue
            
            # Sleep for a while to avoid exceeding the rate limit
            if cnt == 1 and model.provider in ('palm2', 'gpt', 'gemini'):
                time.sleep(3)

            try:
                raw_response = attacker.query(
                    instruction, 
                    processed_data,
                    icl_num=args.icl_num,
                    info_cat=info_cat,
                    need_adaptive_attack=need_adaptive_attack,
                    verbose=1 if i == 0 else 0,
                    idx=i,
                    total=len(task_manager),
                    image=img
                )
            except RuntimeError:
                # This can happen for various reasons: API outdated, instable network, rate limit exceeded, etc. 
                # For simplicity we mark the prediciton as empty. 
                raw_response = ""

            all_raw_responses[info_cat].append(raw_response)
            all_labels[info_cat].append(curr_label[info_cat])
            _ = evaluator.update(raw_response, curr_label, info_cat, defense, verbose=args.verbose)
            cnt = (cnt + 1) % 2
        if args.verbose > 0:  print('\n----------------\n')
    
    np.savez(f'{res_save_path}/all_raw_responses.npz', res=all_raw_responses, label=all_labels)
    print(f'\nResults are saved at: {res_save_path}\n')
    print('[END]')