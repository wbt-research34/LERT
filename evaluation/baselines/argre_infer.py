import json
import os
import sys
import logging
import inspect
import argparse
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
print(f"currentdir: {currentdir}, parentdir: {parentdir}, parentparentdir: {parentparentdir}")
from utils.startup import main
from utils.utils import setup_logging
from reward_model import MyLlamaForCausalLM, MyGPT2LMHeadModel, MyMistralForCausalLM, MyOPTForCausalLM
from reward_model import DetoxLlamaForCausalLM, DetoxMistralForCausalLM,DetoxOPTForCausalLM,DetoxGPT2LMHeadModel,DetoxLlamaForCausalLM30,DetoxLlamaForCausalLM13,DetoxLlamaForCausalLMsft, DetoxMistralForCausalLMsft
parser = argparse.ArgumentParser(description='ARGRE')
parser.add_argument('--config_file', type=str)
parser.add_argument('--score_model_path', type=str)
parser.add_argument('--hidden_dir', type=str, default='')
parser.add_argument('--model_name_suffix', type=str, default='')

parser.add_argument('--vector_path', type=str, required=True, help='Path to your detox vectors/atlas')

args = parser.parse_args()
config_filename = args.config_file
config = main(config_filename=config_filename)
model_name = config.get('Model', 'model_name')
logdir = os.path.join(os.environ["PROJECT_ROOT"], f'{model_name}', model_name+'_'+args.model_name_suffix)
os.makedirs(logdir, exist_ok=True)
os.makedirs(os.path.join(logdir, "log"), exist_ok=True)
setup_logging(os.path.join(logdir, "log"), logname=f'run_{model_name}_{str(time.strftime("%Y-%m-%d-%H-%M-%S"))}', mode='w')
logging.info('Starting')

import numpy as np
from utils.model_utils import load_large_model, MODEL_IDENITFIER
from utils.evaluate_model import evaluate_generate_ppl, evaluate_model_regressive

logging.info(f"args: {args}")

model_id = config.get('Model', 'model_name')

model_path = MODEL_IDENITFIER[model_id]
dtype = torch.float32 if model_id == 'gpt2' else torch.float16

def get_model_class(model_id):
    if 'gpt2' in model_id:
        return DetoxGPT2LMHeadModel
    elif 'mistral-sft' in model_id:
        return DetoxMistralForCausalLMsft
    elif 'llama-7b-sft' in model_id:
        return DetoxLlamaForCausalLMsft
    elif 'llama-13b' in model_id:
        return DetoxLlamaForCausalLM13
    elif 'llama-30b' in model_id:
        return DetoxLlamaForCausalLM30
    elif 'llama-7b-hf' in model_id:
        return DetoxLlamaForCausalLM
    elif 'mistral' in model_id:
        return DetoxMistralForCausalLM
    elif 'opt' in model_id:
        return DetoxOPTForCausalLM
    else:
        raise ValueError(f'Invalid model_id: {model_id}')

model = get_model_class(model_id).from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
    )
model.init_detox_resources(
    reward_stats_path=os.path.join(args.vector_path, f"reward_layer.pt"),
    atlas_path=os.path.join(args.vector_path, f"layer_direction_pca_mean_geometric.pt"),
    reward_model_path=args.score_model_path,
)

def get_mean_reward(hidden_states, attention_mask, score_model, batch_size=512):

    score_model.eval()
    device = next(score_model.parameters()).device
    pos_hidden = hidden_states[:, 0]
    neg_hidden = hidden_states[:, 1]
    pos_mask = attention_mask[:, 0]
    neg_mask = attention_mask[:, 1]
    def compute_masked_reward(hidden, mask):

        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_mask = mask.reshape(-1).bool()
        valid_hidden = flat_hidden[flat_mask]
        num_valid = valid_hidden.size(0)

        rewards = []

        for i in tqdm(range(0, num_valid, batch_size), desc="Scoring tokens", disable=(num_valid <= batch_size)):
            batch = valid_hidden[i:i+batch_size].to(device)
            batch_rewards = score_model(batch).squeeze(-1)
            rewards.append(batch_rewards)
        return torch.cat(rewards)

    pos_rewards = compute_masked_reward(pos_hidden, pos_mask)
    neg_rewards = compute_masked_reward(neg_hidden, neg_mask)

    pos_mean = pos_rewards.mean().cpu().item()
    neg_mean = neg_rewards.mean().cpu().item()
    return pos_mean, neg_mean

model.set_forward_mode("reward_guide")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

logging.info("Evaluating [REWARD_GUIDE] Mode...")

ppl, tox = -1, -1
ppl, tox, toxic_generate_list = evaluate_model_regressive(model, tokenizer)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}')

logging.info("Evaluating [BASE] Mode...")
torch.cuda.empty_cache()
model.set_forward_mode("base")
model.eval()
gentox_ppl, toxic_generate_list = evaluate_generate_ppl(toxic_generate_list=toxic_generate_list,
                    model=model, tokenizer=tokenizer)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}, Generate Perplexity: {gentox_ppl}')

save_file = os.path.join(logdir, f'{model_name}_toxic_generate_list.json')

with open(save_file, 'w') as f:
    json.dump(toxic_generate_list, f, indent=4)

with open(os.path.join(logdir, f'{model_name}.json'), 'w') as f:
    json.dump({
        "toxicity": tox*100,
        "perplexity": ppl,
        "generate_perplexity": gentox_ppl,
    }, f, indent=4)

