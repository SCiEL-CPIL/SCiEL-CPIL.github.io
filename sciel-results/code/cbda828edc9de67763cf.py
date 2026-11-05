# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter linear bandit with adaptive process noise."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Fast timescale: ridge regression state (A_inv, b, theta)
        self.A_inv = [np.eye(dim) / 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Slow timescale: drift tracking per arm
        # Each arm has a drift vector and its covariance
        self.drift = [np.zeros(dim) for _ in range(n_arms)]
        self.drift_P = [np.eye(dim) * 0.01 for _ in range(n_arms)]
        # Process noise base level (tunable)
        self.Q_base = 0.005
        # Observation noise estimate per arm
        self.R_est = [1.0] * n_arms
        # Innovation history for adaptive process noise
        self.innov_history = [[] for _ in range(n_arms)]
        # Exploration parameter
        self.alpha = 1.5
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0
        self.round_count = 0
        # Per-arm visit counts
        self.Visits = [0] * n_arms

    def _kalman_update(self, arm, context, reward):
        """Dual-timescale Kalman update: fast reward estimation + slow drift tracking."""
        x = context
        d = self.d
        
        # --- Fast timescale: standard linear regression update with effective forgetting ---
        # Compute effective process noise Q from innovation
        pred = self.theta[arm] @ x
        innovation = reward - pred
        
        # Store innovation for adaptive Q scaling
        self.innov_history[arm].append(innovation)
        if len(self.innov_history[arm]) > 30:
            self.innov_history[arm].pop(0)
        
        # Adaptive process noise: scale Q by innovation magnitude
        if self.innov_history[arm]:
            mean_innov = np.mean(np.abs(self.innov_history[arm]))
            # Q scales with innovation: high error -> more process noise (faster adaptation)
            Q_scale = 1.0 + 2.0 * np.tanh(mean_innov / 1.0)
        else:
            Q_scale = 1.0
        
        Q = self.Q_base * Q_scale
        
        # Fast update: treat as Bayesian linear regression with process noise
        # A_inv update: (A_inv^{-1} + Q^{-1})^{-1} style via Sherman-Morrison
        # Equivalent to: A_new = A_old + x x^T, with prior covariance inflated by Q
        inv = self.A_inv[arm]
        # Inflate inverse by Q: (Q I + A_old)^{-1}
        # Using matrix inversion lemma: (Q I + A)^{-1} = (1/Q) I - (1/Q^2) A (I + (1/Q) A)^{-1}
        # Simpler: just add Q*I to A before inverting, but we track A_inv
        # Use: A_inv_new = (Q*I + A_old)^{-1}
        # By Woodbury: (Q*I + A)^{-1} = (1/Q) * (I - A * (Q*I + A)^{-1}) ... 
        # Actually simpler: A_new = Q*I + A_old, then A_inv_new = inv(Q*I + A_old)
        # For small Q, A_old dominates. Use perturbation:
        # (A + Q*I)^{-1} ≈ A^{-1} - Q * A^{-1} (I + Q*A^{-1})^{-1}
        # For our purposes, use the standard ridge update with effective gamma
        # gamma_eff = Q / (1 + Q) approximately for scalar case
        # For vector case, use: A_inv_new = (I + Q * A_inv)^{-1} * A_inv / (1 + Q * something)
        
        # Cleaner approach: maintain A directly for the slow timescale
        # Fast timescale uses standard exponential weighting with adaptive gamma
        # Derive effective gamma from Q
        # In 1D: gamma_eff = 1/(1+Q), in multi-D use trace-based
        trace_A_inv = np.trace(inv)
        if trace_A_inv > 0:
            gamma_eff = 1.0 / (1.0 + Q * trace_A_inv / d)
        else:
            gamma_eff = 1.0
        gamma_eff = np.clip(gamma_eff, 0.85, 0.999)
        
        # Standard exponential update with gamma_eff
        x_over_g = x / gamma_eff
        Ax = inv @ x_over_g
        denom = 1.0 + x_over_g @ Ax
        if abs(denom) < 1e-12:
            # Fallback: recompute
            A_new = gamma_eff * (inv + Q * np.eye(d)) + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(A_new)
            self.b[arm] = gamma_eff * self.b[arm] + reward * x
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        new_inv = (inv - np.outer(Ax, Ax) / denom) / gamma_eff
        self.A_inv[arm] = new_inv
        self.b[arm] = gamma_eff * self.b[arm] + reward * x
        self.theta[arm] = new_inv @ self.b[arm]
        
        # --- Slow timescale: drift tracking ---
        # The drift represents slow changes in the true weight vector
        # Update drift using innovation
        # P_prior = drift_P + Q_drift (process noise for drift)
        # K = P_prior @ x / (x^T P_prior x + R)
        # drift_new = drift + K * innovation
        # P_new = (I - K x^T) P_prior
        
        P_prior = self.drift_P[arm] + Q * np.eye(d) * 0.1  # drift moves slower
        S = x @ P_prior @ x + self.R_est[arm]
        if S > 1e-12:
            K = (P_prior @ x) / S
            self.drift[arm] = self.drift[arm] + K * innovation
            self.drift_P[arm] = P_prior - np.outer(K, x) @ P_prior
            # Ensure positive semi-definite
            self.drift_P[arm] = (self.drift_P[arm] + self.drift_P[arm].T) / 2.0
        else:
            self.drift_P[arm] = P_prior
        
        # Update R estimate from recent innovations
        if len(self.innov_history[arm]) >= 5:
            self.R_est[arm] = np.var(self.innov_history[arm][-10:]) + 0.1
            self.R_est[arm] = np.clip(self.R_est[arm], 0.05, 10.0)

    def select(self, context):
        self.round_count += 1
        if time.time() - self.start_time > self.deadline:
            return int(np.argmax(context)) % self.n
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a] + self.drift[a]  # Combine fast + slow estimates
            inv = self.A_inv[a]
            mean = theta @ context
            var = context @ inv @ context
            # Add drift uncertainty to variance
            drift_var = context @ self.drift_P[a] @ context
            total_var = max(var + drift_var, 0.0)
            visit_bonus_scale = 1.0 / (1.0 + 0.1 * self.Visits[a])
            bonus = self.alpha * visit_bonus_scale * np.sqrt(total_var)
            scores.append(mean + bonus)
        
        scores = np.array(scores)
        
        if self.round_count < 100 and np.random.random() < 0.1:
            return int(np.random.randint(self.n))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        self._kalman_update(arm, context, reward)
        self.Visits[arm] += 1
        self.alpha = max(0.8, self.alpha * 0.99995)
# EVOLVE-BLOCK-END
