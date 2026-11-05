"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor and maintained inverse."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Maintain A^{-1} directly for O(d^2) updates
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive discount factor with smoother adaptation
        self.gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.995
        self.error_ema = 0.0
        self.error_alpha = 0.1
        self.reward_ema = 0.0
        self.reward_alpha = 0.05
        
        # Exploration parameter with adaptive behavior
        self.C_base = 1.0
        self.C = 1.0
        self.round_count = 0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 115.0  # seconds (safety margin)
        
        # Track per-arm rewards for better exploration
        self.arm_rewards = [0.0 for _ in range(n_arms)]
        self.arm_counts = [0 for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard: if we're getting close to the limit, be conservative
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline:
            # Return best-known arm based on estimated values
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n):
                if self.arm_counts[a] > 0:
                    val = self.b[a] @ context
                    if val > best_val:
                        best_val = val
                        best_arm = a
            return best_arm
        
        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            # Adaptive C: increase exploration when uncertainty is high or arm is under-explored
            uncertainty = np.sqrt(context @ inv @ context)
            C_a = self.C * (1.0 + 0.5 * np.log1p(1.0 / (1.0 + self.arm_counts[a])))
            scores.append(theta @ context + C_a * uncertainty)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        
        # Check deadline before doing expensive work
        if time.time() - self._start_time > self._deadline:
            return
        
        # Track per-arm statistics
        self.arm_rewards[arm] = (1 - 0.1) * self.arm_rewards[arm] + 0.1 * reward
        self.arm_counts[arm] += 1
        self.reward_ema = (1 - self.reward_alpha) * self.reward_ema + self.reward_alpha * reward
        
        # Adaptive gamma: use prediction error to adjust forgetting rate
        if self.round_count > 5:
            inv = self.A_inv[arm]
            theta = inv @ self.b[arm]
            pred = theta @ context
            err = abs(reward - pred)
            self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * err
            
            # Normalize error: high error -> lower gamma (faster forgetting)
            # Use a smooth sigmoid-like mapping
            norm_err = min(err / max(0.1, self.reward_ema + 0.1), 2.0)
            # Map norm_err in [0, 2] to gamma in [gamma_max, gamma_min]
            t = norm_err / 2.0
            self.gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * t
        else:
            self.gamma = 0.97
        
        # Update b with forgetting
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update A^{-1} using Woodbury identity with forgetting
        # A_new = gamma * A_old + x x^T
        # V_new = (gamma * A_old + x x^T)^{-1}
        # V_new = (1/gamma) * V_old - (1/gamma^2) * V_old x x^T V_old / (1/gamma + x^T V_old x)
        V = self.A_inv[arm]
        g = self.gamma
        x = context
        Vx = V @ x
        denom = (1.0 / g) + x @ Vx
        
        # Check for numerical stability
        if denom < 1e-10:
            # Fallback: just scale
            self.A_inv[arm] = (1.0 / g) * V
        else:
            # V_new = (1/g) * V - (1/g^2) * Vx * Vx^T / denom
            self.A_inv[arm] = (1.0 / g) * V - (1.0 / (g * g)) * np.outer(Vx, Vx) / denom
        
        # Symmetrize to avoid numerical drift
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
        
        # Clamp eigenvalues to prevent numerical issues
        # Quick check: if diagonal has negative values, reset that arm
        diag = np.diag(self.A_inv[arm])
        if np.any(diag < -1e-6):
            self.A_inv[arm] = np.eye(self.d)
            self.b[arm] = reward * context
# EVOLVE-BLOCK-END
