"""
Recurrent Sliding Window LinUCB for Non-Stationary Linear Bandits.

Augments the context with the previous round's reward and selected arm to create
a recurrent feature that implicitly captures temporal dependencies. Combined with
a sliding window for forgetting, this allows the linear model to adapt to
non-stationary shifts in arm weights by leveraging recent interaction history.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.aug_dim = dim + 2  # Original dim + prev_reward + prev_arm_onehot_scalar
        
        # Sliding window parameters
        self.window_size = 25  # Number of recent samples to keep per arm
        self.reg = 1.0         # Regularization parameter (ridge)
        self.alpha = 1.5       # Exploration parameter for UCB
        
        # Data storage for sliding window (stores augmented contexts)
        self.contexts = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.rewards = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Linear regression state: A = reg*I + sum(x x^T), b = sum(r x)
        self.A = [self.reg * np.eye(self.aug_dim) for _ in range(n_arms)]
        self.b = [np.zeros(self.aug_dim) for _ in range(n_arms)]
        
        # Cached Cholesky factor and weights for fast select
        # A = L @ L.T
        self.L = [np.linalg.cholesky(self.A[i]) for i in range(n_arms)]
        self.theta = [np.linalg.solve(self.L[i].T, np.linalg.solve(self.L[i], self.b[i])) for i in range(n_arms)]
        
        # Recurrent state: previous reward and arm
        self.prev_reward = 0.0
        self.prev_arm = 0
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def _augment_context(self, context):
        """Augment context with previous reward and one-hot arm indicator."""
        ctx = np.asarray(context, dtype=np.float64)
        # Append prev_reward and a scalar encoding of prev_arm
        # Use prev_arm as a continuous feature (0 to n_arms-1)
        aug = np.concatenate([ctx, [self.prev_reward, float(self.prev_arm)]])
        return aug

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        aug_ctx = self._augment_context(context)
        scores = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # Predicted mean
            pred = aug_ctx @ self.theta[a]
            
            # Confidence term: sqrt(x^T A_inv x) = || L^-1 x ||_2
            # Solve L z = x
            z = np.linalg.solve(self.L[a], aug_ctx)
            conf = np.linalg.norm(z)
            
            # UCB score
            scores[a] = pred + self.alpha * conf
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Create augmented context using PREVIOUS state (before this round's update)
        aug_ctx = self._augment_context(context)
        
        # If window is full, remove the oldest sample
        if len(self.contexts[arm]) == self.window_size:
            old_ctx = self.contexts[arm][0]
            old_rew = self.rewards[arm][0]
            
            # Incrementally subtract from A and b
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_rew * old_ctx
            
        # Add new sample
        self.contexts[arm].append(aug_ctx)
        self.rewards[arm].append(reward)
        
        # Incrementally add to A and b
        self.A[arm] += np.outer(aug_ctx, aug_ctx)
        self.b[arm] += reward * aug_ctx
        
        # Update recurrent state for next round
        self.prev_reward = reward
        self.prev_arm = arm
        
        # Recompute Cholesky factor and weights for the updated arm only
        # Ensure numerical stability
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm])
            # Solve L y = b, then L.T theta = y
            y = np.linalg.solve(self.L[arm], self.b[arm])
            self.theta[arm] = np.linalg.solve(self.L[arm].T, y)
        except np.linalg.LinAlgError:
            # If singular or not positive definite, add small regularization
            self.A[arm] += 1e-6 * np.eye(self.aug_dim)
            try:
                self.L[arm] = np.linalg.cholesky(self.A[arm])
                y = np.linalg.solve(self.L[arm], self.b[arm])
                self.theta[arm] = np.linalg.solve(self.L[arm].T, y)
            except np.linalg.LinAlgError:
                # Fallback: add more regularization
                self.A[arm] += 1e-4 * np.eye(self.aug_dim)
                self.L[arm] = np.linalg.cholesky(self.A[arm])
                y = np.linalg.solve(self.L[arm], self.b[arm])
                self.theta[arm] = np.linalg.solve(self.L[arm].T, y)
# EVOLVE-BLOCK-END
