"""
Adaptive Sliding Window LinUCB for Non-Stationary Linear Bandits.
Uses an adaptive window size based on recent prediction errors to balance
reactivity to changes and estimation stability.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        
        # Parameters
        self.base_window = 40
        self.min_window = 15
        self.max_window = 100
        self.reg = 1.0
        self.beta = 1.2  # Exploration constant
        
        # Adaptive window state
        self.window_sizes = [self.base_window] * n_arms
        self.residuals = [deque(maxlen=10) for _ in range(n_arms)]
        
        # Per-arm data structures
        # We store (context, reward) pairs in a deque
        self.histories = [deque() for _ in range(n_arms)]
        
        # Pre-computed matrices (recomputed when window changes)
        self.A_inv = [None] * n_arms
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin

    def _recompute_arm(self, arm):
        """Recompute A, b, and A_inv for a specific arm based on its current window."""
        if len(self.histories[arm]) == 0:
            self.A[arm] = self.reg * np.eye(self.dim)
            self.b[arm] = np.zeros(self.dim)
            self.A_inv[arm] = np.eye(self.dim) / self.reg
            return

        # Build A and b from the window
        A = self.reg * np.eye(self.dim)
        b = np.zeros(self.dim)
        
        for ctx, rew in self.histories[arm]:
            A += np.outer(ctx, ctx)
            b += rew * ctx
            
        self.A[arm] = A
        self.b[arm] = b
        
        # Use Cholesky for better numerical stability
        try:
            L = np.linalg.cholesky(A)
            # Solve L L^T X = I to get A_inv
            self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Add jitter if singular
            try:
                L = np.linalg.cholesky(A + 1e-6 * np.eye(self.dim))
                self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.linalg.inv(A + 1e-4 * np.eye(self.dim))

    def _adapt_window(self, arm, residual):
        """Adapt the window size for the arm based on recent residuals."""
        self.residuals[arm].append(abs(residual))
        
        if len(self.residuals[arm]) < 5:
            return
        
        mean_res = np.mean(self.residuals[arm])
        
        # If residuals are high, the model is likely stale or the environment changed.
        # We want to react faster -> smaller window.
        # If residuals are low, the model is good -> larger window for stability.
        
        current_w = self.window_sizes[arm]
        
        if mean_res > 0.8:
            # High error: shrink window more aggressively
            new_w = max(self.min_window, current_w - 8)
        elif mean_res < 0.05:
            # Low error: grow window
            new_w = min(self.max_window, current_w + 3)
        else:
            new_w = current_w
            
        if new_w != current_w:
            self.window_sizes[arm] = new_w
            # Adjust history length if necessary
            while len(self.histories[arm]) > new_w:
                self.histories[arm].popleft()

    def select(self, context):
        if time.time() - self.start_time > self.deadline:
            return 0

        context = np.asarray(context, dtype=np.float64)
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Ensure A_inv is up to date
            if self.A_inv[a] is None:
                self._recompute_arm(a)
            
            # LinUCB score
            theta = self.A_inv[a] @ self.b[a]
            mu = theta @ context
            variance = context @ self.A_inv[a] @ context
            score = mu + self.beta * np.sqrt(variance)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        if time.time() - self.start_time > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        
        # Calculate residual for adaptation
        if self.A_inv[arm] is not None and len(self.histories[arm]) > 0:
            theta = self.A_inv[arm] @ self.b[arm]
            pred = theta @ context
            residual = reward - pred
        else:
            residual = 0.0
            
        # Adapt window size
        self._adapt_window(arm, residual)
        
        # Add new observation
        self.histories[arm].append((context, reward))
        
        # Trim history if it exceeds the current window size
        if len(self.histories[arm]) > self.window_sizes[arm]:
            self.histories[arm].popleft()
            # Full recompute needed when window shrinks
            self._recompute_arm(arm)
        else:
            # Incremental update: only add the new (context, reward) pair
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context
            
            # Use Sherman-Morrison-Woodbury for rank-1 update of inverse
            # A_new = A_old + x x^T
            # A_new^{-1} = A_old^{-1} - (A_old^{-1} x x^T A_old^{-1}) / (1 + x^T A_old^{-1} x)
            A_inv = self.A_inv[arm]
            Ax = A_inv @ context
            denom = 1.0 + context @ Ax
            self.A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom
        
        self.t += 1
# EVOLVE-BLOCK-END
