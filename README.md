# RL-FinancialPlanning-Env
Series of financial planning RL environments that progressively add complexity to eventually become a useful tool for financial planning &amp; retail investment management professionals.


## Description of proof-of-concept environment
The initial environment will be hyper simple in order to serve as a proof of concept. Each episode begins at retirement, with a given amount of savings in a tax-exempt retirement (Roth) account. A fixed amount of spending is withdrawn from the account each timestep to mimic period spending required by the investor. At each timestep, the agent has two investment choices (SP500 & cash), whose returns are then sampled from normal random distributions. The episode is over when one of two eents occur: the agent runs out of money or the agent reaches age 100.

The actions of the agent are two continuous options bounded by 0 & 1 that sum to 1 that indicate the percentage of their account invested in cash and the percentage invested in SP500.

The reward that the agent receives is based on when the agent runs out of money. The earlier the agent runs out of money, the lower the reward. The maximum reward is acheived by getting to age 100 without running out of money. 


## How to evaluate an agent in this environment
Once an agent is trained in the environment, we need a way to evaluate its performance to see if it is superior to existing strategies applied by financial advisors. To do this, we'll run many "evaluation" episodes using each strategy, and compare the results to those of the trained RL agent in its evaluation episodes.

#### Existing strategies
Some existing allocation strategies include:
* Constant allocation (typically in increments of 10% equity)
* Reducing equity allocation in accordance to age (for example, equity weight equals 100-age)

Some existing rebalancing strategies include:
* Never rebalancing
* Full rebalance at each timestep
* Partial rebalance at each timestep
* Trigger-based rebalancing

#### Metrics worth measuring
* Percentage of life in financial ruin
* Percentage of episodes ending in financial ruin

#### Sanity checks on trained agents
* Keeping age constant, does the agent invest more in equities when they have less money?
* Keeping money constant, does the agent invest more in equities as they get older?
* Are the actions similar for similar states?


## Enhancements beyond proof-of-concept
Once we have a trained agent that is superior to existing strategies in our proof-of-concept environment, we will add additional considerations to the environment to make it more realistic. Some of those additional considerations are listed below:

Environment
* Add other account types beyond the initial tax exempt retirement account
* Increase the number of investable securities. First broad asset classes, then sectors, then individual securities, then derivatives of individual securities, etc
* Separate security return into price return, dividend, interest, capital gains distributions, etc
* Change the step frequency to daily
* Introduce mortality risk, so agents may die before age 100
* Incorporate the idea of return regimes to replicate the idea of various stages of the market cycle

Actions
* Asset Location - as we add various account types, use asset location to take advantage of tax breaks of certain account types
* Asset Allocation - as we add more investable securities, the action space will have to increase to accommodate these new securities
* Spending amount - in the original environment, the spending amount was fixed. A more realistic approach may be to allow the agent to decide how much they spend in any timestep.
* Retirement decision - the agent should be able to decide when they stop working
* Social Security - one question facing almost every American investor is determining when they should start taking social security benefits. 
* Gifting - many clients appreciate using some of their money to support charitable causes in their lifetime.

Rewards
* As the agent gets authority to change spending amounts in each period, the reward for that spending should be commensurate. Typically, this reward is concave as spending increases.
* Many investors have a goal of leaving some money to their families / favorite causes after the die. The greater the amount left, the greater the reward. Again, this reward should be concave as the bequest increases.
* Regarding gifting, the more gifting, the more reward. Again, this reward should be concave as gifting amount increases.
* whatever
