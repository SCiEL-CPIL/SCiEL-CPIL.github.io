"""Non-stationary linear contextual bandit with context-free EMA approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary contextual bandit using a context-free multi-armed bandit strategy.
    - Maintains Exponential Moving Averages (EMA) of rewards for each arm.
    - Uses a fixed aggressive forgetting factor (gamma=0.90) to track non-stationarity.
    - Selects arms using an Upper Confidence Bound (UCB) strategy based on 
      effective sample size.
    - Ignores context entirely to maximize robustness and speed.
    - Includes a deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.dim = dim
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # EMA parameters
        # gamma is the forgetting factor. Higher gamma = slower forgetting.
        # Directive suggests aggressive forgetting, e.g., 0.90.
        self.gamma = 0.90
        self.alpha = 1.0 - self.gamma  # Weight on new observation

        # State for each arm
        # ema_reward: The exponentially weighted moving average of the reward
        self.ema_reward = [0.0] * n_arms
        # effective_count: Tracks the effective number of samples seen
        # This is used to estimate uncertainty for UCB.
        # N_eff = sum(alpha^i) = (1 - alpha^(t+1)) / (1 - alpha)
        # We update it recursively: N_eff_new = alpha * N_eff_old + 1
        self.effective_count = [0.0] * n_arms
        
        # Global step counter for UCB logarithmic term
        self.t = 0

        # UCB exploration constant
        # Tuned to be moderate. Since N_eff is small (around 1/alpha = 10),
        # the sqrt(1/N_eff) term is around 0.3.
        # We want the bonus to be significant in early stages but decay later.
        self.c_ucb = 1.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        # Deadline guard: if close to deadline, use greedy (highest EMA)
        if not self._check_deadline():
            return int(np.argmax(self.ema_reward))

        self.t += 1
        log_t = np.log(self.t + 1)  # +1 to avoid log(0)

        scores = []
        for a in range(self.n):
            # Estimate uncertainty
            # If count is 0, uncertainty is infinite (or very high)
            if self.effective_count[a] < 1e-6:
                uncertainty = 1e6
            else:
                # Variance of EWMA estimator is approx (1 - gamma) / (1 + gamma) * sigma^2 / N_eff?
                # Simpler: Uncertainty ~ 1 / sqrt(N_eff)
                # N_eff is our effective_count.
                uncertainty = 1.0 / np.sqrt(self.effective_count[a])
            
            # UCB Score
            # We add a bonus proportional to the exploration constant and uncertainty
            # The log_t term in standard UCB is for finite-horizon guarantees.
            # For non-stationary, we might just use the uncertainty term directly.
            # Let's use: Score = EMA + c * uncertainty
            # The effective_count grows to ~ 1/alpha = 10.
            # So uncertainty starts at 1/sqrt(1) = 1.0 and decays to 1/sqrt(10) ~ 0.31.
            # This provides a constant exploration pressure, which is good for non-stationary settings.
            
            score = self.ema_reward[a] + self.c_ucb * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        reward = float(reward)
        
        # Update effective count
        # N_eff = alpha * N_eff_old + 1
        self.effective_count[arm] = self.alpha * self.effective_count[arm] + 1.0
        
        # Update EMA reward
        # M_new = alpha * r + (1 - alpha) * M_old
        # Note: This is a standard EWMA. 
        # It converges to the mean of the last ~1/alpha samples.
        self.ema_reward[arm] = self.alpha * reward + (1.0 - self.alpha) * self.ema_reward[arm]
# EVOLVE-BLOCK-END
