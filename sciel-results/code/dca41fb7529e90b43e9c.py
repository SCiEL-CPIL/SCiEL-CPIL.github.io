import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

        # Hierarchical forgetting
        self.gamma_global = 0.99
        self.gamma_arm = 0.85
        self.base_gamma_global = 0.99
        self.base_gamma_arm = 0.85
        self.min_gamma = 0.70
        self.max_gamma = 0.999

        # Global Linear Model (shared context effect)
        self.A_global = np.eye(dim)
        self.b_global = np.zeros(dim)
        self.inv_A_global = np.eye(dim)

        # Per-Arm Linear Models (arm-specific context effect)
        self.A_arm = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_arm = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A_arm = [np.eye(dim) for _ in range(n_arms)]

        # UCB parameters
        self.fixed_beta = 1.2
        self.exploration_boost = 0.0

        # Change detection
        self.prediction_errors = []
        self.error_window = 15
        self.error_threshold = 2.5
        self.consecutive_errors = 0
        self.consecutive_threshold = 3

        # Best arm tracking
        self.best_arm = 0
        self.ewma_values = np.zeros(n_arms)
        self.ewma_alpha = 0.15
        self.ewma_counts = np.zeros(n_arms)

    def _within_deadline(self):
        return time.time() < self.deadline

    def _sm_update_inverse(self, A_inv, x, gamma):
        v = A_inv @ x
        denom = gamma + x @ v
        if abs(denom) < 1e-10:
            denom = 1e-10
        sm_term = np.outer(v, v) / denom
        M_inv = A_inv - sm_term
        new_inv = M_inv / gamma
        new_inv = (new_inv + new_inv.T) / 2.0
        return new_inv

    def _detect_change(self, actual_reward, predicted_reward):
        error = abs(actual_reward - predicted_reward)
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.error_window:
            self.prediction_errors.pop(0)

        if len(self.prediction_errors) >= 5:
            mean_err = np.mean(self.prediction_errors)
            std_err = np.std(self.prediction_errors)
            if std_err > 1e-6 and error > mean_err + self.error_threshold * std_err:
                self.consecutive_errors += 1
            else:
                self.consecutive_errors = 0
        else:
            self.consecutive_errors = 0

        return self.consecutive_errors >= self.consecutive_threshold

    def _adapt_forgetting(self, change_detected):
        if change_detected:
            self.gamma_global = max(self.min_gamma, self.gamma_global * 0.90)
            self.gamma_arm = max(self.min_gamma, self.gamma_arm * 0.85)
            self.exploration_boost = 0.8
            self.consecutive_errors = 0
        else:
            self.gamma_global = min(self.max_gamma, self.gamma_global + 0.002)
            self.gamma_arm = min(self.max_gamma, self.gamma_arm + 0.003)
            self.exploration_boost *= 0.90
            if self.exploration_boost < 0.01:
                self.exploration_boost = 0.0

    def select(self, context):
        self.t += 1

        if not self._within_deadline():
            return self.best_arm

        context = np.asarray(context)
        scores = np.zeros(self.n)

        theta_global = self.inv_A_global @ self.b_global
        v_global = context @ self.inv_A_global

        for a in range(self.n):
            theta_arm = self.inv_A_arm[a] @ self.b_arm[a]
            mean = (theta_global + theta_arm) @ context
            v_arm = context @ self.inv_A_arm[a]
            var = context @ (v_global + v_arm)
            if var < 1e-10:
                var = 1e-10
            beta_eff = self.fixed_beta + self.exploration_boost
            scores[a] = mean + beta_eff * np.sqrt(var)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        self.t += 1

        if not self._within_deadline():
            return

        context = np.asarray(context)

        theta_global = self.inv_A_global @ self.b_global
        theta_arm = self.inv_A_arm[arm] @ self.b_arm[arm]
        predicted_reward = (theta_global + theta_arm) @ context
        change_detected = self._detect_change(reward, predicted_reward)
        self._adapt_forgetting(change_detected)

        self.inv_A_global = self._sm_update_inverse(self.inv_A_global, context, self.gamma_global)
        self.b_global = self.gamma_global * self.b_global + reward * context

        self.inv_A_arm[arm] = self._sm_update_inverse(self.inv_A_arm[arm], context, self.gamma_arm)
        self.b_arm[arm] = self.gamma_arm * self.b_arm[arm] + reward * context

        self.ewma_values[arm] = (1 - self.ewma_alpha) * self.ewma_values[arm] + self.ewma_alpha * reward
        self.ewma_counts[arm] += 1

        if self.t % 5 == 0:
            valid_arms = [a for a in range(self.n) if self.ewma_counts[a] > 0]
            if valid_arms:
                self.best_arm = max(valid_arms, key=lambda a: self.ewma_values[a])
