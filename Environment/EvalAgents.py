# function to evaluate naive agents in my reinforcement learning environment
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for the time being, load Env from local machine
from env import TrainingEnv
from stable_baselines3 import DDPG

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.files import FileNotDownloadableError

DOWNLOAD_MODELS: bool = False

def remove_models(p: Path):
    if not p.exists(): return
    if p.is_file(): return p.unlink()
    for child in p.iterdir():
        remove_models(child)
    p.rmdir()


def download_models(gdrive, path, out_dir, owner=None):
    out_dir = Path(out_dir)
    remove_models(out_dir)
    out_dir.mkdir()

    # Shared folders don't seem to have any parents (not even 'root').
    # The owner name is listed in the 'ownerNames' list, but it seems like
    # Google Drive doesn't let you query by that. The following works for now but
    # is probably not safe.
    prev = '' if owner else 'root'
    for folder in path.split('/'):
        parent = f"'{prev}' in parents and " if prev else ''
        query = f"{parent} title='{folder}' and trashed=false"
        file_list = drive.ListFile({'q': query, 'supportsAllDrives': True, 'includeItemsFromAllDrives': True}).GetList()
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
            print(f"Could not download {info['title']} ; message follows.")
            print(e)
            continue
    return [f['title'] for f in file_list]


# https://stackoverflow.com/questions/24419188/automating-pydrive-verification-process
if DOWNLOAD_MODELS:
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("creds-gdrive.txt")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("creds-gdrive.txt")
    drive = GoogleDrive(gauth)
    download_models(drive, "Colab Notebooks/testfolder/embedded", "models")
    #download_models(drive, "x0", "trained_agents", owner='Keith England')

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
    action = np.array([1.0 - action_equity_weight, action_equity_weight], dtype=np.float32)
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

    print(f"Evaluating {count_trained_agents} trained agents and {count_naive_agents} naive agents")
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
            this_agent_name = all_trained_agents[agent_number - count_naive_agents].lstrip(f"{trained_path[:-1]}{os.sep}Model_").rstrip('.zip')
            this_agent = DDPG.load(all_trained_agents[agent_number - count_naive_agents],
                                   custom_objects={'action_space': env.action_space})

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
                this_action, _states = this_agent.predict(obs)

            while not ep_done:
                obs, this_step_reward, ep_done, info = env.step(this_action)
                this_ep_reward += this_step_reward
                if agent_number < count_naive_agents:
                    this_action = naive_agent_action(env, base_equity_weight, rebal_strategy, obs[0], obs[4],
                                                     first_step=False)
                else:
                    this_action, _states = this_agent.predict(obs)

            # add the results to the pandas variable
            results.loc[len(results.index)] = \
                {'agent_name': this_agent_name, 'episode_number': ep_loop, 'reward': this_ep_reward,
                 'ending_age': obs[0], 'ending_portfolio_value': obs[1]}

    return results

if __name__ == '__main__':
    results = eval_agents(TrainingEnv(), count_episodes=10, trained_naive_both='both',
                          trained_path='trained_agents/')

    base = [('mean', 'mean'), ('std', 'std'), ('spread', np.ptp)]
    aggs = {'reward': [*base, ('% ruin', lambda r: (r < 0).mean())],
            'ending_age': base, 'ending_portfolio_value': base}
    individual = results.groupby('agent_name').agg(aggs)

    # TODO: Break down this file into smaller files (plus others)
    # TODO: Research PyCharm so Keith can explore in console after running
    trained = individual[individual.index.str.contains('Counter\d+$')]
    matches = trained.index.str.extract('(.*)_Counter\d+$', expand=False)
    combined = trained.groupby(matches).mean().rename('{}_Average'.format)

    summary = pd.concat([individual, combined])
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', None,
                           'display.max_colwidth', 25):
        print(summary.sort_values(by=('reward','mean'), ascending=False).round(4))

    def get_kind(label):
        if label.startswith('naive_'):
            return 'Naive'
        assert label.startswith('DDPG_')
        if label.endswith('Average'):
            return 'Trained Average'
        return 'Trained'
    summary['kind'] = summary.index.map(get_kind)


    bool_is_naive = summary['kind'] == 'Naive'
    naive = summary[bool_is_naive].nlargest(3, ('reward', 'mean'))

    comparison_group = pd.concat([naive, summary[~bool_is_naive]])
    comparison_group.reset_index(inplace=True)
    ax = sns.barplot(y='agent_name', x=('reward','mean'), hue='kind',
                     data=comparison_group)
    ax.set_title('Agent Performance')
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f')
    plt.savefig('summary.png', bbox_inches='tight')