# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Sliding-window linear bandit with incremental Gram matrix updates and augmented context (intercept)."""

    def __init__(self, n_arms, dim, window=50, ridge=1e-2, beta0=2.0):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension
        self.window = window
        self.ridge = ridge
        self.beta0 = beta0
        
        # Per-arm storage
        # ctx_buf: (window, D)
        # rw_buf: (window,)
        self.ctx_buf = [np.zeros((window, self.D)) for _ in range(n_arms)]
        self.rw_buf = [np.zeros(window) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.ptr = [0 for _ in range(n_arms)]
        
        # Incremental statistics
        # A: (D, D) sum of x x^T
        # b: (D,) sum of x y
        self.A = [np.zeros((self.D, self.D)) for _ in range(n_arms)]
        self.b = [np.zeros(self.D) for _ in range(n_arms)]
        
        self.t = 0

    def _augment(self, context):
        context = np.asarray(context, dtype=float).ravel()
        x_aug = np.zeros(self.D)
        x_aug[0] = 1.0
        x_aug[1:] = context
        return x_aug

    def select(self, context):
        x_aug = self._augment(context)
        scores = np.zeros(self.n)
        # Decay exploration slightly over time
        beta = self.beta0 * (1.0 / (1.0 + 0.001 * self.t))
        
        for a in range(self.n):
            c = self.count[a]
            if c < 2:
                # High uncertainty initially
                scores[a] = 10.0 + np.random.randn()
                continue
                
            # Solve for theta
            A_reg = self.A[a] + self.ridge * np.eye(self.D)
            try:
                # Compute inverse for both theta and bonus
                A_inv = np.linalg.inv(A_reg)
                theta = A_inv @ self.b[a]
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                A_inv = np.linalg.pinv(A_reg)
                theta = A_inv @ self.b[a]
                
            mu = theta @ x_aug
            
            # Bonus: sqrt(x^T A_inv x)
            v = x_aug @ A_inv @ x_aug
            bonus = beta * np.sqrt(max(v, 0.0))
            
            scores[a] = mu + bonus
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # If window is full, remove the oldest element
        if self.count[arm] == self.window:
            old_x = self.ctx_buf[arm][self.ptr[arm]]
            old_y = self.rw_buf[arm][self.ptr[arm]]
            
            self.A[arm] -= np.outer(old_x, old_x)
            self.b[arm] -= old_x * old_y
        else:
            self.count[arm] += 1
            
        # Add new element
        self.A[arm] += np.outer(x_aug, x_aug)
        self.b[arm] += x_aug * reward
        
        # Update buffers
        self.ctx_buf[arm][self.ptr[arm]] = x_aug
        self.rw_buf[arm][self.ptr[arm]] = reward
        self.ptr[arm] = (self.ptr[arm] + 1) % self.window
        
        self.t += 1
# EVOLVE-BLOCK-END
