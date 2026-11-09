"""Non-stationary linear contextual bandit with sliding window and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Sliding window (fixed size) for robust adaptation to changing arm weights.
    2. Thompson Sampling for exploration/exploitation.
    
    For each arm, we maintain a sliding window of the most recent observations.
    We compute the ridge regression estimate from the window and sample from
    the posterior to decide which arm to pull.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # safety margin before 120s limit
        
        # Sliding window size per arm
        self.window_size = 40
        
        # Ridge regularization parameter
        self.lam = 1.0
        
        # Per-arm sliding window data
        # queues[a] stores (context, reward) in chronological order
        self.queues = [collections.deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Cached A and b matrices for fast updates
        # A[a] = sum of x_i x_i^T over window, b[a] = sum of r_i x_i
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Precompute Cholesky factorizations lazily
        self._chol_cache = [None] * n_arms

    def _update_arm_stats(self, arm, context, reward):
        """Update the sliding window statistics for a given arm."""
        q = self.queues[arm]
        
        # If window is full, remove the oldest entry
        if len(q) == self.window_size:
            old_ctx, old_reward = q[0]
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_reward * old_ctx
        
        # Add new entry
        q.append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Invalidate Cholesky cache
        self._chol_cache[arm] = None

    def _get_theta_and_covariance(self, arm):
        """
        Compute the posterior mean (theta) and covariance for an arm.
        Uses ridge regression: theta = A^{-1} b, Cov = A^{-1}
        where A includes the ridge term.
        """
        A = self.A[arm]
        b = self.b[arm]
        
        # Compute theta = A^{-1} b
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
        
        # Compute covariance = A^{-1}
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            cov = np.eye(self.d)
        
        return theta, cov

    def select(self, context):
        """
        Select an arm using Thompson Sampling.
        For each arm, sample a parameter vector from the posterior,
        compute the predicted reward, and select the arm with the highest sample.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check: if we're running out of time, use a simple greedy fallback
        if time.time() > self.deadline:
            # Greedy fallback: pick arm with highest predicted reward (no exploration)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta, _ = self._get_theta_and_covariance(a)
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)
        
        # Thompson Sampling
        samples = []
        for a in range(self.n):
            theta, cov = self._get_theta_and_covariance(a)
            # Sample from N(theta, cov)
            sampled_theta = theta + np.random.multivariate_normal(np.zeros(self.d), cov)
            # Predicted reward for this arm
            pred_reward = sampled_theta @ context
            samples.append(pred_reward)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        """
        Update the policy with a new observation.
        """
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        self._update_arm_stats(arm, context, reward)
# EVOLVE-BLOCK-END
