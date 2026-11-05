"""Non-stationary linear contextual bandit with shared-model UCB approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Shared linear context model across arms (efficient parameter sharing)
    - Arm-specific bias terms for per-arm reward offsets
    - UCB exploration with decaying bonus
    - Prediction-error-adaptive forgetting for non-stationarity tracking
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + 1  # augmented with intercept
        self.start_time = time.time()
        self.deadline = 140.0

        # Shared model: V_shared (inverse cov), b_shared
        self.V_shared = np.eye(self.d_aug)
        self.b_shared = np.zeros(self.d_aug)
        
        # Arm-specific biases (scalar per arm)
        self.arm_bias = np.zeros(n_arms)
        self.arm_bias_var = np.ones(n_arms)  # variance of bias estimates
        self.arm_counts = np.zeros(n_arms, dtype=np.int64)

        # Forgetting parameters
        self.base_gamma = 0.92  # more aggressive than 0.97 for faster adaptation
        self.gamma_min = 0.80
        self.gamma_max = 0.99
        self.global_prediction_error = 0.0

        # UCB parameters
        self.ucb_alpha = 1.5  # exploration coefficient
        self.ucb_decay = 0.999  # slow decay of exploration bonus
        self.total_rounds = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _augment(self, context):
        ctx = np.asarray(context, dtype=np.float64)
        return np.concatenate([ctx, [1.0]])

    def _adaptive_gamma(self, prediction_error):
        """Adapt forgetting based on global prediction error."""
        self.global_prediction_error = 0.95 * self.global_prediction_error + 0.05 * prediction_error
        pe = self.global_prediction_error
        pe_scale = min(pe, 4.0)
        gamma = self.base_gamma - 0.08 * pe_scale
        return max(self.gamma_min, min(self.gamma_max, gamma))

    def select(self, context):
        ctx_aug = self._augment(context)
        self.total_rounds += 1

        if not self._check_deadline():
            # Greedy fallback
            theta = self.V_shared @ self.b_shared
            scores = theta @ ctx_aug + self.arm_bias
            return int(np.argmax(scores))

        # UCB: mean + alpha * uncertainty
        theta_hat = self.V_shared @ self.b_shared
        mean_scores = theta_hat @ ctx_aug + self.arm_bias
        
        # Uncertainty: sqrt of diagonal of V_shared (same for all arms in context part)
        # Plus arm bias uncertainty
        context_uncertainty = np.sqrt(np.diag(self.V_shared) @ (ctx_aug ** 2))
        bias_uncertainties = np.sqrt(self.arm_bias_var)
        
        # Decay exploration bonus over time
        exploration_decay = self.ucb_decay ** self.total_rounds
        bonus = self.ucb_alpha * exploration_decay * (context_uncertainty + bias_uncertainties)
        
        scores = mean_scores + bonus
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx_aug = self._augment(context)
        reward = float(reward)
        arm = int(arm)

        # Compute prediction error
        theta_hat = self.V_shared @ self.b_shared
        predicted = theta_hat @ ctx_aug + self.arm_bias[arm]
        prediction_error = abs(reward - predicted)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(prediction_error)

        # Update shared model with Sherman-Morrison + forgetting
        V_old = self.V_shared
        Vx = V_old @ ctx_aug
        xVx = ctx_aug @ Vx
        denom = gamma + xVx

        if denom > 1e-12:
            self.V_shared = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V_shared = np.eye(self.d_aug)

        # The reward attributable to the shared model (subtract arm bias)
        reward_for_shared = reward - self.arm_bias[arm]
        self.b_shared = gamma * self.b_shared + reward_for_shared * ctx_aug

        # Update arm bias with its own forgetting
        self.arm_counts[arm] += 1
        # Bias variance update: treat as 1-D Gaussian update
        prior_var = self.arm_bias_var[arm]
        self.arm_bias_var[arm] = 1.0 / (1.0 / prior_var + 1.0)
        self.arm_bias[arm] = self.arm_bias_var[arm] * (prior_var * self.arm_bias[arm] + reward_for_shared)
# EVOLVE-BLOCK-END
