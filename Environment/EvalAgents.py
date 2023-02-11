# function to evaluate naive agents in my reinforcement learning environment

# source packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydrive.files
import glob

# for the time being, load Env from local machine
# exec(open('C:\\Users\\keith\\PycharmProjects\\RL-FinancialPlanning-Env\\Environment\\env.py').read())

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import FileNotDownloadableError
from pathlib import Path

import stable_baselines3
from stable_baselines3 import DDPG

import gym
from gym import utils


def remove_models(p: Path):
    if not p.exists(): return
    if p.is_file(): return p.unlink()
    for child in p.iterdir():
        remove_models(child)
    p.rmdir()


def download_models(gdrive, path, out_dir):
    out_dir = Path(out_dir)
    remove_models(out_dir)
    out_dir.mkdir()

    prev = 'root'
    for folder in path.split('/'):
        query = f"'{prev}' in parents and title='{folder}' and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        assert len(file_list) == 1
        assert file_list[0]['mimeType'] == 'application/vnd.google-apps.folder'
        prev = file_list[0]['id']

    query = f"'{prev}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    for info in file_list:
        remote_file = drive.CreateFile({'id': info['id']})
        local_filename = out_dir / info['title'].replace(' ', '-')
        try:
            remote_file.GetContentFile(local_filename)
        except FileNotDownloadableError as e:
            print(local_filename)
            continue

    return [f['title'] for f in file_list]


gauth = GoogleAuth()
gauth.LocalWebserverAuth()
# gauth.LoadCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)
download_models(drive, "Colab Notebooks/RL/TrainedModels/x0", "trained_agents")

from env import *

env = TrainingEnv()


def naive_agent_action(env, base_equity_weight, rebal_strategy, age_obs, sop_equity_weight, first_step):
    # env: environment to evaluate agent in
    # base_equity_weight: reference equity weight
    # rebal_strategy: when does the agent rebalance to the base_equity_weight
    # age_obs: age of client as it appears in the environment
    # sop_equity_weight: equity weight at start of step
    # first_step: is this the first step in the episode
    #
    # returns: recommended action: [cash_weight, spy_weight]

    age_years = env.min_start_age + age_obs * (env.max_start_age - env.min_start_age)

    # allRebalStrategies = np.array(['NoRebal', 'FullRebal', '5PctFullRebal', '5PctHalfRebal'])
    if base_equity_weight > 1.0:
        base_equity_weight = min(1.0, base_equity_weight - (age_years / 100))
    else:
        base_equity_weight = base_equity_weight

    if first_step:
        action_equity_weight = base_equity_weight
    else:
        if rebal_strategy == 'NoRebal':
            action_equity_weight = sop_equity_weight
        elif rebal_strategy == 'FullRebal':
            action_equity_weight = base_equity_weight
        elif rebal_strategy == '5PctFullRebal':
            if abs(sop_equity_weight - base_equity_weight) > 0.05:
                action_equity_weight = base_equity_weight
            else:
                action_equity_weight = sop_equity_weight
        elif rebal_strategy == '5PctHalfRebal':
            if sop_equity_weight - base_equity_weight > 0.05:
                action_equity_weight = base_equity_weight + 0.025
            elif sop_equity_weight - base_equity_weight < -0.05:
                action_equity_weight = base_equity_weight - 0.025
            else:
                action_equity_weight = sop_equity_weight
    action = np.float32(np.array([1 - action_equity_weight, action_equity_weight]))
    return action


def eval_agents(env, count_episodes=1000, trained_naive_both='both',
                trained_path='trained_agents/'):
    # env: environment to evaluate agent in
    # count_episodes: number of episodes to evaluate agent in
    # trained_naive_both: are we evaluating a trained agent, a naive agent, or both
    # trained_path: path to trained agents

    # determine the number of naive agents
    if trained_naive_both != 'trained':
        all_base_weights = np.arange(0.10, 1.80, 0.10)
        all_rebal_strategies = np.array(['NoRebal', 'FullRebal', '5PctFullRebal', '5PctHalfRebal'])
        count_naive_agents = all_base_weights.shape[0] * all_rebal_strategies.shape[0]
    else:
        count_naive_agents = 0

    # determine the number of trained agents
    if trained_naive_both != 'naive':
        all_trained_agents = glob.glob(trained_path + '*.zip')
        count_trained_agents = len(all_trained_agents)
    else:
        count_trained_agents = 0

    count_total_agents = count_naive_agents + count_trained_agents

    # create a pandas variable that is the same length as the count_total_agent * count_episodes, and 5 columns
    results = pd.DataFrame(columns=['agent_name', 'episode_number', 'reward', 'ending_age', 'ending_portfolio_value'])

    # loop through all the agents
    for agent_number in range(count_total_agents):

        # name the agent
        if agent_number < count_naive_agents:
            base_equity_weight = all_base_weights[agent_number // all_rebal_strategies.shape[0]]
            rebal_strategy = all_rebal_strategies[agent_number % all_rebal_strategies.shape[0]]
            this_agent_name = 'naive_' + str(base_equity_weight) + '_' + rebal_strategy
        else:
            this_agent_name = all_trained_agents[agent_number - count_naive_agents].replace(trained_path, '')
            this_agent = DDPG.load(all_trained_agents[agent_number - count_naive_agents])

        # loop through all the episodes
        for ep_loop in range(count_episodes):

            # reset the environment
            obs = env.reset()
            ep_done = False
            this_ep_reward = 0

            if agent_number < count_naive_agents:
                this_action = naive_agent_action(env, base_equity_weight, rebal_strategy, obs[0], obs[4],
                                                 first_step=True)
            else:
                this_action = this_agent.predict(obs)

            while not ep_done:

                obs, this_step_reward, ep_done, info = env.step(this_action)
                this_ep_reward += this_step_reward
                if agent_number < count_naive_agents:
                    this_action = naive_agent_action(env, base_equity_weight, rebal_strategy, obs[0], obs[4],
                                                     first_step=False)

            # add the results to the pandas variable
            results.loc[len(results.index)] = \
                {'agent_name': this_agent_name, 'episode_number': ep_loop, 'reward': this_ep_reward,
                 'ending_age': obs[0], 'ending_portfolio_value': obs[1]}



    return results


results = eval_agents(env, count_episodes=10, trained_naive_both='both')

print(results)

# # determine the number of episodes that ended with a reward below zero for each agent
# results['reward_below_zero'] = results['reward'] < 0
# results['reward_below_zero'] = results['reward_below_zero'].astype(int)
#
#
# # calculate the average reward for each agent
# results['reward'] = results['reward'].astype(float)
# results['ending_portfolio_value'] = results['ending_portfolio_value'].astype(float)
# results['ending_age'] = results['ending_age'].astype(float)
# summary_results = results.groupby(['agent_name']).mean()
#
#
#
#
# # make a bar chart of the summary_results
# summary_results['reward'].plot(kind='bar')
# plt.show()
