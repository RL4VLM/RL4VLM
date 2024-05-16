from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import time
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, get_alfworld_prompt
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

# For alfworld
from alf_utils import load_config_file, get_obs_image, ALF_ACTION_LIST, process_action, compute_reward, AlfEnv


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM

from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    model_device = device
    model_path = args.model_path
    cache_dir = args.cache_dir

    print(f"Path of the model is {model_path}")
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)

    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    ## Inputing Prompt here
    assert args.alf_config is not None, "Alfworld environment requires a config file"
    envs = AlfEnv(args.alf_config)
    obs, infos = envs.reset(seed=args.seed)
    admissible_commands = list(infos['admissible_commands'])[0]
    qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    if "alfred" in args.env_name.lower():
        projection_f = partial(lambda x: x)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              (300, 300, 3), spaces.Discrete(14), args.max_new_tokens)

    image_tensor = obs
    _, output_ids, action, action_log_prob, action_tokens_log_prob = actor_critic.act(image_tensor, INPUT_IDS = INPUT_IDS)
    admissible_commands = list(infos['admissible_commands'])[0]

    print("output_ids:{}".format(output_ids))
    print("prompt:{}".format(prompt))
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_gc_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_and_place = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_two_obj_and_place = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_look_at_obj_in_light = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_heat_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_cool_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_clean_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)
        text_table = wandb.Table(columns=["epoch", "obs_text", "text_action"])
    print("prompt:{}".format(prompt))
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    for j in tqdm(range(num_updates)):

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                        rollouts.obs[step], INPUT_IDS = INPUT_IDS)
                admissible_commands = list(infos['admissible_commands'])[0]
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))

            # Observation, reward and next obs
            obs, reward, done, infos = envs.step(action) # for alf this will already process action
            if "alfred" in args.env_name.lower():
                admissible_commands = list(infos['admissible_commands'])[0]
                qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    # record success rate of different types of tasks
                    if "pick_and_place" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_and_place.append(float(infos['won'][0]))
                    elif "pick_two_obj_and_place" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_two_obj_and_place.append(float(infos['won'][0]))
                    elif "look_at_obj_in_light" in infos["extra.gamefile"][0]:
                        episode_succ_rate_look_at_obj_in_light.append(float(infos['won'][0]))
                    elif "pick_heat_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_heat_then_place_in_recep.append(float(infos['won'][0]))
                    elif "pick_cool_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_cool_then_place_in_recep.append(float(infos['won'][0]))
                    elif "pick_clean_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_clean_then_place_in_recep.append(float(infos['won'][0]))
                    # record the final success rate
                    episode_success_rate.append(float(infos['won'][0]))
                    episode_gc_success_rate.append(float(infos['goal_condition_success_rate'][0]))
                    print(len(episode_success_rate))
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
                    obs, infos = envs.reset(seed=args.seed)

                    admissible_commands = list(infos['admissible_commands'])[0]
                    qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
            # bad_masks is a legact implementation in the storage
            bad_masks = torch.zeros(args.num_processes, 1)
            # action_id is also a legacy implementation in the storage, it is never used in the PPO update
            action_id = None
            for i in range(len(admissible_commands)):
                if admissible_commands[i] == action:
                    action_id = i
                    break
            if not action_id:
                action_id = 0
            action_id = torch.tensor(action_id)

            rollouts.insert(obs, output_id, action_id,
                                action_log_prob, value, reward, masks, bad_masks)

        print("prompt:{}".format(prompt))
        print("action_log_prob:{}".format(action_log_prob))
        print("text_action:{}".format(text_action))
        print("action:{}".format(action))
        print("ground truth:{}".format(infos))
        print("success_rate:{}".format(np.mean(episode_success_rate)))

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], INPUT_IDS = INPUT_IDS).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        lr_scheduler.step()

        rollouts.after_update()
        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        dist_entropy, value_loss, action_loss))
            if args.use_wandb:
                wandb_images = [wandb.Image(image.cpu().numpy()) for image in obs]
                text_table.add_data(j, infos['observation_text'][0], text_action)
                wandb.log({"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode_reward.mean": np.mean(episode_rewards),
                        "episode_reward.median": np.median(episode_rewards),
                        "episode_reward.min": np.min(episode_rewards),
                        "episode_reward.max": np.max(episode_rewards),
                        "episode_success_rate.mean": np.mean(episode_success_rate),
                        "episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                        "episode_(goal_condition)_success_rate.mean": np.mean(episode_gc_success_rate),
                        "episode_succ_rate_pick_and_place.mean": np.mean(episode_succ_rate_pick_and_place),
                        "episode_succ_rate_pick_two_obj_and_place.mean": np.mean(episode_succ_rate_pick_two_obj_and_place),
                        "episode_succ_rate_look_at_obj_in_light.mean": np.mean(episode_succ_rate_look_at_obj_in_light),
                        "episode_succ_rate_pick_heat_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_heat_then_place_in_recep),
                        "episode_succ_rate_pick_cool_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_cool_then_place_in_recep),
                        "episode_succ_rate_pick_clean_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_clean_then_place_in_recep),
                        "episode_num": len(episode_success_rate),
                        "distribution_entropy": dist_entropy,
                        "text": text_table,
                        "image": wandb_images,
                        "value.loss": value_loss,
                        "action.loss": action_loss,
                        "action_log_prob": action_log_prob.to('cpu').float().numpy()[0],
                        "reward.max": rollouts.rewards.max().item(),
                        "reward.min": rollouts.rewards.min().item(),
                        "reward.mean": rollouts.rewards.mean().item(),
                        "reward.std": rollouts.rewards.std().item(),
                        "reward.median": rollouts.rewards.median().item(),
                        "return.max": rollouts.returns.max().item(),
                        "return.min": rollouts.returns.min().item(),
                        "return.mean": rollouts.returns.mean().item(),
                        "return.std": rollouts.returns.std().item(),
                        "value.max": rollouts.value_preds.max().item(),
                        "value.min": rollouts.value_preds.min().item(),
                        "value.mean": rollouts.value_preds.mean().item(),
                        "value.std": rollouts.value_preds.std().item(),})

if __name__ == "__main__":
    main()
