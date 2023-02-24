import numpy as np
from gym import spaces, Env # TODO: switch to `gymnasium` instead

np.random.seed(123)


class TrainingEnv(Env):

    def __init__(self):

        self.count_buyable_securities = 2

        obs_low = np.concatenate((np.array([0, 0, 0]),  # clientAge, portValue, target_spend_dollars
                                 np.zeros(self.count_buyable_securities)),  # security weights
                                 dtype=np.float32)

        obs_high = np.concatenate((np.ones(3),  # clientAge, portValue, target_spend_dollars
                                  np.ones(self.count_buyable_securities)),  # security weights
                                  dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.count_buyable_securities,), dtype=np.float64)

        self.stepsPerYear = 2

        self.min_start_age = 60
        self.max_start_age = 100
        self.age_step_size = 1 / (self.stepsPerYear * (self.max_start_age - self.min_start_age))
        self.start_age = 0

        self.secExpRet = np.array([0, 0.07]) / self.stepsPerYear
        self.secExpCov = np.array([0,
                                   0, 0.04]) / self.stepsPerYear

    def step(self, action: np.ndarray):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (
        #     action,
        #     type(action),
        # )
        if np.sum(action) < 0.0000001:
            #raise ValueError(f"Action sum is too small ({np.sum(action)})")
            action = action + 0.5

        # TODO: is this next line necessary? We make sure above that the maximum will be the sum, and we can also
        #  ensure that the sum is 1.0 (by moving the below check up above this), so dividing by it won't change
        #  anything.
        invest_pct = action / np.maximum(np.sum(action), 0.000000001)

        total = np.sum(invest_pct)
        if abs(total - 1.0) >= 0.0001:
            raise ValueError(f"Invest percentages do not sum to 1 ({total})")

        sop_client_age, sop_port_value, target_spend_dollars = self.state[np.arange(3)]

        post_trade_pre_wd_security_dollars = sop_port_value * invest_pct

        ############################################################################
        # Set up post-withdrawal, pre-investment
        ############################################################################

        post_trade_post_wd_security_dollars = post_trade_pre_wd_security_dollars
        post_trade_post_wd_security_dollars[0] = post_trade_post_wd_security_dollars[0] - target_spend_dollars

        post_wd_port_value = np.sum(post_trade_post_wd_security_dollars)

        if post_wd_port_value < (1 / 1_000_000):
            # this may not be right....come back to it later
            post_trade_post_wd_security_weights = np.ones(self.count_buyable_securities) / self.count_buyable_securities
        else:
            post_trade_post_wd_security_weights = post_trade_post_wd_security_dollars / post_wd_port_value

        ############################################################################
        # Determine if we ran out of money
        # Assign rewards
        ############################################################################

        if post_trade_post_wd_security_dollars[0] < 0:

            done = True

            # reward = nTimesteps client is broke, scaled to -1 for self.min_start_age and 0 for self.max_start_age
            count_broke_timesteps = (1 - sop_client_age) / self.age_step_size
            max_steps = 1 / self.age_step_size
            reward = -(count_broke_timesteps / max_steps)

            eop_port_value = sop_port_value - target_spend_dollars

            eop_client_age = sop_client_age + self.age_step_size

            self.state = np.concatenate((np.array([eop_client_age, eop_port_value, target_spend_dollars]),
                                         post_trade_post_wd_security_weights))

            return np.array(self.state, dtype=np.float32), reward, done, {}

        else:
            reward = float(0.0)
            done = False

        ############################################################################
        # See how investment performed
        ############################################################################
        exp_cov_matrix = np.array([[0, 0],
                                  [0, 0.04]]) / self.stepsPerYear

        # Calculate investment return
        security_return = np.random.multivariate_normal(self.secExpRet, exp_cov_matrix, 1)[0]
        eop_dollars_security = post_trade_post_wd_security_dollars * (1 + security_return)
        eop_port_value = np.sum(eop_dollars_security)

        if eop_port_value < (1 / 1_000_000):
            eop_weight_security = np.ones(self.count_buyable_securities) / self.count_buyable_securities
        else:
            eop_weight_security = eop_dollars_security / eop_port_value

        eop_client_age = sop_client_age + self.age_step_size
        if eop_client_age >= 1.0:
            done = True

        self.state = np.concatenate((np.array([eop_client_age, eop_port_value, target_spend_dollars]),
                                     eop_weight_security))

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        age_reset = self.start_age

        starting_port_value_reset = 1

        target_spend_dollars_reset = 0.02

        sec_pct_reset = np.ones(self.count_buyable_securities) / self.count_buyable_securities

        self.state = np.concatenate((np.array([age_reset, starting_port_value_reset, target_spend_dollars_reset]),
                                     sec_pct_reset))

        return np.array(self.state, dtype=np.float32)
    
