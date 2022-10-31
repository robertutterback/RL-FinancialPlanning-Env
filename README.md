# RL-FinancialPlanning-Env
Series of financial planning RL environments that progressively add complexity to eventually become a useful tool for financial planning &amp; retail investment management professionals.

## Description of proof-of-concept environment
The initial environment will be hyper simple in order to serve as a proof of concept. It involves an environment that offers two investment choices (SP500 & cash) that sample from normal random returns each timestep. Each episode begins at retirement with a given amount of lifetime savings in a tax-exempt (Roth) account. A fixed amount of spending is withdrawn from the account each timestep. The episode is over when one of two events occur: the agent runs out of money, or the agent reaches age 100. 

The actions of the agent are simple: two continuous actions that sum to 1 that indicate the percentage of their account invested in cash and the percentage invested in SP500.

Generally, the reward that the agent receives is based on when the agent runs out of money. The earlier the agent runs out of money, the lower the reward. The maximum reward is acheived by getting to age 100 without running out of money. 

## How to evaluate an agent in this environment
Once an agent is trained in the environment, we need a way to evaluate its performance to see if it is superior to existing strategies applied by financial advisors. To do this, we'll run many "evaluation" episodes using each strategy, and compare the results to those of the trained RL agent in its evaluation episodes.

### Existing strategies
Some existing allocation strategies include:
* Constant allocation (typically in increments of 10% equity)
* Reducing equity allocation in accordance to age (for example, equity weight equals 100-age)

Some existing rebalancing strategies include:
* Never rebalancing
* Full rebalance at each timestep
* Partial rebalance at each timestep
* Trigger-based rebalancing

### Metrics worth measuring
* Percentage of life in financial ruin
* Percentage of episodes ending with money

### Sanity checks on trained agents
* Keeping age constant, does the agent invest more in equities when they have less money?
* Keeping money constant, does the agent invest more in equities as they get older?
* Are the actions similar for similar states?

## Enhancements beyond proof-of-concept
Once the 

Actions
* Asset Allocation
* Asset Location
* Spending / spending amount
* Retire / keep working
* Start taking social security

Rewards
* Having enough money in retirement
* Concave ongoing spending
* Bequest
* Gifting

Environment
* Various account types
* Mortality risk
* Step frequency
* Return regimes
* Separating total return into price return + income return
* Number of investable securities
* What is randomized during reset

Evaluation 
* Average reward (compared to naive agents)
* Analysis of final step when failing episode
* Action in sampled significant states
* Compare to intuition
* Are they smooth?
* When the agent has accumulated enough wealth to get the highest reward, does the agent take reasonable actions?
