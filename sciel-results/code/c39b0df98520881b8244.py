"""Bootstrap portfolio: Thompson Sampling with adaptive decay for non-stationary linear bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Thompson Sampling:
      - Per-arm Bayesian linear model with posterior covariance.
      - Adaptive decay factor driven by prediction residuals (Mahalanobis distance).
      - When residuals are large (regime shift), decay increases to forget stale data faster.
      - When residuals are small, decay decreases to stabilize estimates.
      - Matrix inversion lemma for efficient posterior updates.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        self.n_arms = n_arms

        # Regularization / prior precision
        self.ridge = 1.0

        # Per-arm state: posterior mean (theta) and covariance (Sigma)
        # Prior: theta ~ N(0, I/ridge)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.Sigma = [np.eye(dim) / self.ridge for _ in range(n_arms)]
        self.Sigma_inv = [np.eye(dim) * self.ridge for _ in range(n_arms)]

        # Adaptive decay tracking
        self.base_decay = 0.995
        self.decay = self.base_decay
        self.residual_history = []
        self.residual_mean = 0.0
        self.residual_std = 1.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115  # 115s safety margin out of 120s

        self.t = 0
        self.rng = np.random.default_rng()

    def _check_deadline(self):
        """Return True if we are close to the deadline."""
        if time.time() > self.deadline:
            return True
        return False

    def _compute_adaptive_decay(self, residual):
        """
        Compute adaptive decay factor based on prediction residual.
        Large residual -> regime shift -> lower decay (faster forgetting)
        Small residual -> stable regime -> higher decay (slower forgetting)
        """
        self.residual_history.append(abs(residual))
        if len(self.residual_history) > 50:
            self.residual_history = self.residual_history[-50:]

        if len(self.residual_history) >= 5:
            residuals = np.array(self.residual_history)
            self.residual_mean = np.mean(residuals)
            self.residual_std = max(np.std(residuals), 1e-6)
        else:
            self.residual_mean = 0.0
            self.residual_std = 1.0

        # Normalize residual by typical scale
        if self.residual_std > 1e-6:
            normalized_residual = abs(residual) / (self.residual_mean + self.residual_std + 1e-6)
        else:
            normalized_residual = 0.0

        # Cap normalized residual to prevent extreme values
        normalized_residual = min(normalized_residual, 5.0)

        # Map normalized residual to decay in [0.95, 0.999]
        # High residual -> low decay (more forgetting)
        # Low residual -> high decay (less forgetting)
        decay = 0.999 - 0.049 * (normalized_residual / 5.0)
        self.decay = max(0.95, min(0.999, decay))
        return self.decay

    def _update_posterior(self, arm, x, reward):
        """
        Update posterior for arm using matrix inversion lemma.
        theta_new = theta_old + Sigma_old * x * (r - x^T theta_old) / (1 + x^T Sigma_old x)
        Sigma_new = Sigma_old - Sigma_old * x * x^T * Sigma_old / (1 + x^T Sigma_old x)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        r = float(reward)

        # Compute prediction and residual
        pred = self.theta[arm] @ x
        residual = r - pred

        # Update adaptive decay based on residual
        decay = self._compute_adaptive_decay(residual)

        # Apply decay to covariance (forgetting)
        # Sigma_decayed = Sigma / decay (increases uncertainty)
        # Actually, for exponential forgetting, we scale the precision:
        # Precision_decayed = decay * Precision
        # So Sigma_decayed = Sigma / decay
        Sigma_decayed = self.Sigma[arm] / decay
        Sigma_inv_decayed = self.Sigma_inv[arm] * decay

        # Matrix inversion lemma update
        # k = Sigma_decayed @ x
        k = Sigma_decayed @ x
        denom = 1.0 + x @ k

        # Update mean
        theta_new = self.theta[arm] + (k * residual) / denom
        self.theta[arm] = theta_new

        # Update covariance
        Sigma_new = Sigma_decayed - np.outer(k, k) / denom
        self.Sigma[arm] = Sigma_new

        # Update inverse covariance (for efficiency in select)
        self.Sigma_inv[arm] = np.linalg.inv(Sigma_new)

    def select(self, context):
        x = np.asarray(context, dtype=np.float64).flatten()

        # Check deadline; if close, use deterministic greedy
        if self._check_deadline():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                score = self.theta[a] @ x
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Thompson Sampling: sample from posterior for each arm
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Sample theta ~ N(theta_a, Sigma_a)
            theta_sample = self.rng.multivariate_normal(self.theta[a], self.Sigma[a])
            scores[a] = theta_sample @ x

        best_arm = int(np.argmax(scores))
        self.t += 1
        return best_arm

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64).flatten()
        self._update_posterior(arm, x, reward)
# EVOLVE-BLOCK-END
