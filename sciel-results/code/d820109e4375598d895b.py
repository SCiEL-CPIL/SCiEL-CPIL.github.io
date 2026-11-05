"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with forgetting factor, Cholesky solve, and adaptive exploration."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.eye = np.eye(dim)
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor: closer to 1 means longer memory
        self.beta = 0.99

        # Exploration parameter
        self.alpha = 1.0

        # Track time steps for adaptive alpha
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a safe default (e.g., first arm or based on current best estimate)
            # For simplicity, return 0 or use a quick heuristic
            # Since we must return an arm, we'll just return 0 to be safe and fast
            return 0

        scores = []
        for a in range(self.n):
            # Use Cholesky decomposition for numerical stability and speed
            try:
                L = np.linalg.cholesky(self.A[a] + 1e-8 * self.eye)
                # Solve L * v = context
                v = np.linalg.solve(L, context)
                # Solve L.T * w = v
                w = np.linalg.solve(L.T, v)
                inv_context = w @ context  # context.T @ A^{-1} @ context
            except np.linalg.LinAlgError:
                # Fallback to inverse if Cholesky fails
                inv = np.linalg.inv(self.A[a] + 1e-8 * self.eye)
                inv_context = context @ inv @ context

            # Compute theta = A^{-1} b
            try:
                theta = np.linalg.solve(self.A[a] + 1e-8 * self.eye, self.b[a])
            except np.linalg.LinAlgError:
                inv = np.linalg.inv(self.A[a] + 1e-8 * self.eye)
                theta = inv @ self.b[a]

            mean = theta @ context
            std = np.sqrt(max(0, inv_context))

            # Adaptive alpha: decrease over time to reduce exploration
            # alpha_t = alpha_0 / sqrt(t)
            if self.t > 0:
                alpha_t = self.alpha / np.sqrt(self.t)
            else:
                alpha_t = self.alpha

            scores.append(mean + alpha_t * std)

        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update with forgetting factor
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
