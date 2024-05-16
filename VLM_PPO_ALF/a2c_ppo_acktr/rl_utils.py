import torch
import random
from typing import List
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from alfworld.agents.utils.misc import get_templated_task_desc
from alf_utils import AlfEnv

def get_alfworld_prompt(env_name, obs, admissible_actions, action_only = False):
    """
        This function defines the prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    task = get_templated_task_desc(env_name.env.envs[0].traj_data)
    if not action_only:
        refomratted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions)
        qs = f"Your are an expert in the ALFRED Embodied Environment. "
        qs = qs + f"You are also given the following text description of the current scene: {obs}. "
        qs = qs + f"Your task is to " + task + ". "
        qs = qs + f"Your admissible actions of the current situation are: [{refomratted_admissible_actions}]. "
        qs = qs + "Your response should be a valid json file in the following format: \n\{\n"
        qs = qs + "\"thoughts\": \"{first describe what do you see in the image using the text description, then carefully think about which action to complete the task. }\", \n"
        qs = qs + "\"action\": \"{an admissible action}\"\n\}"
    else:
        refomratted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions)
        qs = f"Your are an expert in the ALFRED Embodied Environment. "
        qs = qs + f"You are also given the following text description of the current scene: {obs}. "
        qs = qs + f"Your task is to " + task + ". "
        qs = qs + f"Your admissible actions of the current situation are: [{refomratted_admissible_actions}]. "
        qs = qs + "Your response should be a valid json file in the following format: \n\{\n"
        qs = qs + "\"action\": \"{an admissible action}\"\n\}"
    return qs

def get_prompt(env_name, infos = None):
    """
        This function defines the prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    if env_name == 'gym_cards/NumberLine-v0':
        qs = "You are playing a game called number line. You will see a target number and a current number in the image. "
        qs = qs + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        qs = qs + "\"current number\": \"x\", \n"
        qs = qs + "\"target number\": \"x\", \n"
        qs = qs + "\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n"
        qs = qs + "\"action\": \"-\" or \"+\" \n}"
    elif env_name == 'gym_cards/Blackjack-v0':
        qs = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        qs = qs + "Your response should be a valid json file in the following format: \n {\n "
        qs = qs + "\"thoughts\": \"{first describe your total points and the dealer's total points then think about which action to choose}\", \n"
        qs = qs + "\"action\": \"stand\" or \"hit\" \n}"

    elif env_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert card game player. You are observing two cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 12, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n"
        qs = qs + " \"cards\": [x, y], \n"
        qs = qs + f"\"current formula\": {text_formula}, \n"
        qs = qs + "\"thoughts\": {First check whether the current formula 'z' is complete. "
        qs = qs + "If the current formula 'z' is complete, output '='. "
        qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 12.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"

    elif env_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert 24 points card game player. You are observing these four cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n"
        qs = qs + " \"cards\": [x, y, z, w], \n"
        qs = qs + f"\"current formula\": {text_formula}, \n"
        qs = qs + "\"thoughts\": {First check whether the current formula equals 24. "
        qs = qs + "If the current formula equals 24, output '='. "
        qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"
    return qs

def get_action_only_prompt(env_name, infos = None):
    """
        This function defines the "action only" prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    if env_name == 'gym_cards/NumberLine-v0':
        qs = "You are playing a game called number line. You will see a target number and a current number in the image. "
        qs = qs + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        qs = qs + "\"action\": \"-\" or \"+\" \n}"

    elif env_name == 'gym_cards/Blackjack-v0':
        qs = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        qs = qs + "Your response should be a valid json file in the following format: \n {\n "
        qs = qs + "\"action\": \"stand\" or \"hit\" \n}"

    elif env_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert card game player. You are observing two cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 12, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n\"action\": \"{number}\" or \"{operator}\" \n \}"

    elif env_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert 24 points card game player. You are observing these four cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n\"action\": \"{number}\" or \"{operator}\" \n \}"
    return qs

# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    output_indices = []
    if env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(random.randint(0, len(action_list) - 1))
            continue
        string = string.lower()
        action_index = string.find('"action":')
        string = string[action_index:]
        contained_actions = []
        # For the 'gym_cards/Points24-v0' environment, handle '10' separately
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        # Find all actions that are contained in the string
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(random.randint(0, len(action_list) - 1))
    return torch.Tensor([output_indices]).long().reshape(-1, 1)
