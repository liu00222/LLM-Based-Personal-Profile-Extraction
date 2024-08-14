import os

def run(provider, model_name, dataset, api_key_pos, defense, prompt_type, icl_num, gpus, adaptive_attack_on_pi, redundant_info_filtering):
    model_config_path = f'./configs/model_configs/{provider}_config.json'
    task_config_path = f'./configs/task_configs/{dataset}.json'

    log_dir = f'./log/{provider}_{model_name.split("/")[-1]}'
    os.makedirs(log_dir, exist_ok=True)

    log_file = f'{log_dir}/{dataset}_{defense}_{prompt_type}_{icl_num}_adaptive_{adaptive_attack_on_pi}_filter_{redundant_info_filtering}.txt'

    if provider == 'internlm':
        cmd = f"CUDA_VISIBLE_DEVICES=0, nohup python3 -u main.py \
                --model_config_path {model_config_path} \
                --model_name {model_name} \
                --task_config_path {task_config_path} \
                --icl_num {icl_num} \
                --prompt_type {prompt_type} \
                --api_key_pos {api_key_pos} \
                --defense {defense} \
                --adaptive_attack {adaptive_attack_on_pi} \
                --redundant_info_filtering {redundant_info_filtering} \
                > {log_file} &"
    else:
        cmd = f"nohup python3 -u main.py \
                --model_config_path {model_config_path} \
                --model_name {model_name} \
                --task_config_path {task_config_path} \
                --icl_num {icl_num} \
                --prompt_type {prompt_type} \
                --api_key_pos {api_key_pos} \
                --defense {defense} \
                --gpus {gpus} \
                --adaptive_attack {adaptive_attack_on_pi} \
                --redundant_info_filtering {redundant_info_filtering} \
                > {log_file} &"

    os.system(cmd)
    return log_file


""" 1 Selecting the LLM """
model_info = [
    'palm2',
    'models/text-bison-001'
]
# model_info = [
#     'palm2',
#     'models/chat-bison-001'
# ]
# model_info = [
#     'gemini',
#     'gemini-pro'
# ]
# model_info = [
#     'gpt',
#     'gpt-3.5-turbo'
# ]
# model_info = [
#     'gpt',
#     'gpt-4'
# ]
# model_info = [
#     'vicuna',
#     'lmsys/vicuna-13b-v1.3'
# ]
# model_info = [
#     'vicuna',
#     'lmsys/vicuna-7b-v1.3'
# ]
# model_info = [
#     'llama',
#     'meta-llama/Llama-2-7b-chat-hf'
# ]
# model_info = [
#     'flan',
#     'google/flan-ul2'
# ]
# model_info = [
#     'internlm',
#     'internlm/internlm-chat-7b'
# ]


""" 2 Selecting the dataset """
datasets = [
    'synthetic',
    # 'celebrity',
    # 'physician'
]


""" 3 Selecting the prompt type """
prompt_types = ['direct']
# prompt_types = ['pseudocode', 'contextual', 'persona']


""" 4 Choosing the num of demonstration examples """
icl_nums = [
    0,
    # 1,
    # 2,
    # 3
]


""" 5 Choosing whether to apply redundant information filtering """
redundant_info_filtering = "True"


""" 
6 Choosing the countermeasures 

- no: no countermeasure
- image: text-to-image
- pi_ci: prompt injection with context ignoring
- pi_id: prompt injection with injected data
- pi_ci_id: prompt injection with context ignoring and injected data
- replace_at: symbol replacement to transform @ to AT
- replace_dot: symbol replacement to transform . to DOT
- replace_at_dot: symbol replacement to transform @ to AT and . to DOT
- hyperlink: hyperlink
- mask: name replacement, e.g., albert.einstein@gmail.com to <first_name>.<last_name>@gmail.com
"""
defenses = ['pi_ci_id']
# defenses = ['image']
# defenses = ['pi_ci', 'pi_id', 'pi_ci_id']
# defenses = ['replace_at', 'replace_dot', 'replace_at_dot', 'hyperlink', 'mask']


""" 7 Choosing the adaptive attack against prompt injection """
adaptive_attacks_on_pi = ['no']
# adaptive_attacks_on_pi = ['sandwich', 'instructional', 'paraphrasing', 'retokenization', 'xml', 'delimiters', 'random_seq']


""" Sanity check to avoid unexpected query cost (especially for GPT-4 and GPT-3.5 which could be a disaster to budgets...) """
for dataset in datasets:
    assert (dataset in ['synthetic', 'celebrity', 'physician'])
for defense in defenses:
    assert (defense in ['no', 'replace_at', 'replace_at_dot', 'replace_dot', 'hyperlink', 'mask', 'pi_ci', 'pi_id', 'pi_ci_id', 'image'])
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    assert (adaptive_attack_on_pi in ['no', 'sandwich', 'xml', 'delimiters', 'random_seq', 'instructional', 'paraphrasing', 'retokenization'])
    if adaptive_attack_on_pi != 'no':
        for defense in defenses:
            assert ( 'pi' in defense or 'no' in defense )


""" Configuring other settings """
api_key_pos = 0
gpus = '0'


user_decision = input(f"Total process: {len(adaptive_attacks_on_pi)*len(datasets)*len(defenses)*len(prompt_types)*len(icl_nums)}\nRun? (y/n): ")
if user_decision.lower() != 'y':
    exit()


provider = model_info[0]
model_name = model_info[1]
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    for data in datasets:

        for defense in defenses:

            for prompt_type in prompt_types:
            
                for icl_num in icl_nums:

                    # execute
                    tmp = run(provider, model_name, data, api_key_pos, defense, prompt_type, icl_num, str(gpus), adaptive_attack_on_pi, redundant_info_filtering)