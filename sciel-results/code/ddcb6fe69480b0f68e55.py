# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Exponentially-weighted linear bandit with Sherman-Morrison rank-1 updates and innovation-adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A as identity (ridge) for each arm - stronger ridge for stability
        self.A = [np.eye(dim) * 0.2 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Precompute inverses for Sherman-Morrison updates
        self.A_inv = [np.eye(dim) / 0.2 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Per-arm adaptive forgetting factor - start aggressive for fast non-stationary adaptation
        self.gamma = [0.90] * n_arms
        # Exploration parameter
        self.alpha = 1.8
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s
        self.round_count = 0
        # Per-arm visit counts and prediction error tracking
        self.Visits = [0] * n_arms
        self.recent_errors = [[] for _ in range(n_arms)]
        # Innovation history for adaptive gamma (Kalman-inspired)
        self.innovation_sq = [0.0 for _ in range(n_arms)]
        # Huber loss threshold parameter
        self.huber_delta = 1.0

    def _sherman_morrison_update(self, arm, context, reward, gamma):
        """Update A_inv and theta using Sherman-Morrison formula after rank-1 update."""
        x = context
        inv = self.A_inv[arm]
        g = float(gamma)
        x_over_g = x / g
        Ax = inv @ x_over_g
        denom = 1.0 + x_over_g @ Ax
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse
            self.A[arm] = g * self.A[arm] + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.b[arm] = g * self.b[arm] + reward * x
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        # A_new^{-1} = (1/g) * (inv - np.outer(Ax, Ax) / denom)
        new_inv = (inv - np.outer(Ax, Ax) / denom) / g
        self.A_inv[arm] = new_inv
        
        # Update b: b_new = g * b + reward * x
        self.b[arm] = g * self.b[arm] + reward * x
        # Update theta = A_new^{-1} @ b_new
        self.theta[arm] = new_inv @ self.b[arm]
        
        # Also update A for potential fallback use
        self.A[arm] = g * self.A[arm] + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return a deterministic arm based on context hash to avoid exceptions
            return int(np.argmax(context)) % self.n
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.A_inv[a]
            mean = theta @ context
            # UCB bonus with adaptive alpha
            var = context @ inv @ context
            # Scale bonus by sqrt of number of visits to this arm for better calibration
            # Tighter visit-based scaling to reduce over-exploration of well-visited arms
            visit_bonus_scale = 1.0 / (1.0 + 0.15 * self.Visits[a])
            bonus = self.alpha * visit_bonus_scale * np.sqrt(max(var, 0.0))
            scores.append(mean + bonus)
        
        scores = np.array(scores)
        
        # Epsilon-greedy: with small probability, explore randomly
        if self.round_count < 100 and np.random.random() < 0.1:
            return int(np.random.randint(self.n))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Calculate prediction error (innovation) before update
        pred = self.theta[arm] @ context
        innovation = reward - pred
        innov_sq = innovation * innovation
        
        # Update innovation history (exponential moving average)
        self.innovation_sq[arm] = 0.8 * self.innovation_sq[arm] + 0.2 * innov_sq
        
        # Huber-loss based robust reward: clamp large innovations to reduce outlier sensitivity
        abs_innov = abs(innovation)
        if abs_innov > self.huber_delta:
            # Use Huber loss: for |e| > delta, gradient is sign(e) * delta
            # Effective reward adjustment: pred + sign(innovation) * delta
            robust_reward = pred + np.sign(innovation) * self.huber_delta
        else:
            robust_reward = reward
        
        # Adaptive forgetting: 
        # High innovation -> high process noise -> lower gamma (faster forgetting)
        # Use sqrt of EMA of squared innovation to get a robust error scale
        err_scale = np.sqrt(self.innovation_sq[arm])
        
        # Map error scale to target gamma
        # Range [0.80, 0.99]
        # Steeper response for better non-stationary tracking
        target_gamma = 0.99 - 0.19 * np.tanh(err_scale / 0.5)
        
        # Smoothly adapt per-arm gamma
        # Faster adaptation: 0.25 weight for quicker response to drift
        self.gamma[arm] = 0.75 * self.gamma[arm] + 0.25 * target_gamma
        self.gamma[arm] = np.clip(self.gamma[arm], 0.80, 0.99)
        
        # Perform the update with the adapted per-arm gamma using robust reward
        self._sherman_morrison_update(arm, context, robust_reward, self.gamma[arm])
        
        # Track visits
        self.Visits[arm] += 1
        
        # Adjust exploration: reduce alpha over time but keep some exploration
        # Slightly faster decay to exploit more after initial phase
        self.alpha = max(0.6, self.alpha * 0.99990)
# EVOLVE-BLOCK-END
