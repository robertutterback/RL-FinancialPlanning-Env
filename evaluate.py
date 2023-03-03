# function to evaluate naive agents in my reinforcement learning environment
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for the time being, load Env from local machine
from dotenv import load_dotenv
from env import TrainingEnv
from download import download_models
from stable_baselines3 import DDPG

load_dotenv()

# @TODO: Where to actually put/decide this? How often will we need to download new models? Automatic or manual?
DOWNLOAD_MODELS = True
if DOWNLOAD_MODELS:
    download_models("trained_agents", path=os.getenv('GDRIVE_PATH'), root=os.getenv('GDRIVE_ROOT') or 'root')

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


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    base = [('mean', 'mean'), ('std', 'std'), ('spread', np.ptp)]
    aggs = {'reward': [*base, ('% ruin', lambda r: (r < 0).mean())],
            'ending_age': base, 'ending_portfolio_value': base}
    individual = results.groupby('agent_name').agg(aggs)
    trained = individual[individual.index.str.contains('Counter\d+$')]
    matches = trained.index.str.extract('(.*)_Counter\d+$', expand=False)
    combined = trained.groupby(matches).mean().rename('{}_Average'.format)
    summary = pd.concat([individual, combined])
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', None,
                           'display.max_colwidth', 25):
        print(summary.sort_values(by=('reward', 'mean'), ascending=False).round(4))

    def get_kind(label):
        if label.startswith('naive_'):
            return 'Naive'
        assert label.startswith('DDPG_')
        if label.endswith('Average'):
            return 'Trained Average'
        return 'Trained'

    summary['kind'] = summary.index.map(get_kind)
    return summary


def plot_best(summary: pd.DataFrame, topn_naive=3, topn_trained=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    bool_is_naive = summary['kind'] == 'Naive'

    naive = summary[bool_is_naive]
    if topn_naive is not None:
        naive = summary[bool_is_naive].nlargest(topn_naive, ('reward', 'mean'))

    trained = summary[~bool_is_naive]
    if topn_trained is not None:
        trained = summary[~bool_is_naive].nlargest(topn_trained, ('reward', 'mean'))

    comparison_group = pd.concat([naive, trained])
    comparison_group.reset_index(inplace=True)
    sns.barplot(y='agent_name', x=('reward', 'mean'), hue='kind',
                data=comparison_group, ax=ax)
    ax.set_title('Agent Performance')
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f')
    return ax

if __name__ == '__main__':
    results = eval_agents(TrainingEnv(), count_episodes=10, trained_naive_both='both',
                          trained_path='trained_agents/')
    summary = summarize(results)
    ax = plot_best(summary, topn_naive=3)
    ax.get_figure().savefig('summary.png', bbox_inches='tight')