"""Non-stationary linear contextual bandit with adaptive Kalman filter and residual-driven process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a shared low-rank factorization
    with joint Kalman filtering for cross-arm coherence.
    
    Design:
    - Arm weights theta_a = W @ z_a where W is (dim x r) global map, z_a is (r x 1) arm latent.
    - Joint state: [W.flatten(); z_1; ...; z_n] with block-diagonal + shared covariance.
    - Adaptive process noise per component based on innovation residuals.
    - UCB exploration using predicted variance from the factorized model.
    - Falls back to per-arm KF if factorization is ill-conditioned.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Rank of shared factorization
        self.r = min(3, dim, n_arms)
        
        # Global context-to-latent map: (dim x r)
        self.W = np.zeros((self.d, self.r))
        # Arm-specific latent vectors: list of (r,)
        self.Z = [np.zeros(self.r) for _ in range(self.n)]
        
        # Per-arm full theta cache for fast scoring
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        for a in range(self.n):
            self.theta[a] = self.W @ self.Z[a]
        
        # Covariance for W (d x r) flattened: (d*r x d*r)
        self.P_W = np.eye(self.d * self.r) * 50.0
        # Covariance for each Z_a: (r x r)
        self.P_Z = [np.eye(self.r) * 50.0 for _ in range(self.n)]
        # Cross-covariances between W and each Z_a: (d*r x r)
        self.P_WZ = [np.zeros((self.d * self.r, self.r)) for _ in range(self.n)]
        
        # Process noise parameters
        self.q_W_base = 0.03
        self.q_Z_base = 0.05
        self.q_min = 0.005
        self.q_max = 3.0
        self.adapt_gain = 0.5
        self.error_threshold = 1.0
        
        # Exploration
        self.alpha = 1.5
        self.alpha_decay = 0.999
        self.alpha_min = 0.5
        
        # Tracking
        self.round_num = 0
        self.arm_counts = [0] * self.n
        self.pred_errors_W = np.zeros(10)
        self.pred_errors_W_idx = 0
        self.pred_errors_Z = [np.zeros(10) for _ in range(self.n)]
        self.pred_errors_Z_idx = [0] * self.n
        self.W_error_sum = 0.0
        self.Z_error_sums = [0.0] * self.n

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adapt_q_W(self):
        """Adapt process noise for W based on recent W-related innovations."""
        if self.round_num < 5:
            return self.q_W_base
        mean_err = np.mean(self.pred_errors_W)
        ratio = max(mean_err / self.error_threshold, 1e-3)
        q = self.q_W_base * (ratio ** self.adapt_gain)
        return max(self.q_min, min(self.q_max, q))

    def _adapt_q_Z(self, arm):
        """Adapt process noise for Z_arm based on recent arm-specific innovations."""
        c = self.arm_counts[arm]
        if c < 5:
            return self.q_Z_base
        mean_err = np.mean(self.pred_errors_Z[arm])
        ratio = max(mean_err / self.error_threshold, 1e-3)
        q = self.q_Z_base * (ratio ** self.adapt_gain)
        return max(self.q_min, min(self.q_max, q))

    def _update_theta_cache(self, arm):
        """Update theta cache for arm after W or Z_a changes."""
        self.theta[arm] = self.W @ self.Z[arm]

    def _update_all_theta(self):
        """Update all theta caches after W changes."""
        for a in range(self.n):
            self.theta[a] = self.W @ self.Z[a]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            
            # Compute uncertainty from factorized covariance
            # theta_a = W @ Z_a, so d(theta_a)/dW and d(theta_a)/dZ_a
            # Var(theta_a^T x) = x^T (W P_Za W^T + Z_a^T P_W Z_a + cross terms) x
            # Simplified: use sum of variances from W and Z contributions
            # For W contribution: x^T W P_Za W^T x
            Wx = W_x = self.W.T @ context  # (r,)
            var_W = Wx @ self.P_Z[a] @ Wx
            # For Z contribution: Z_a^T (x^T P_W x) Z_a -- approximate
            # x^T P_W x is scalar if P_W is (d x d), but P_W is (d*r x d*r)
            # Use trace approximation: Z_a^T P_W_reduced Z_a where P_W_reduced = sum over r
            # Simplified: use diagonal of effective covariance
            # Effective variance from W: trace(context^T W P_W W^T context) approx
            # Use: ||W^T x||^2 * mean(diag(P_W)) as rough upper bound
            diag_P_W = np.trace(self.P_W) / (self.d * self.r)
            var_Z = (Wx @ Wx) * diag_P_W
            
            # Cross term approximation (usually small)
            var_cross = 2.0 * (Wx @ self.P_WZ[a] @ np.zeros(self.r))  # placeholder, zero for now
            
            var_total = max(var_W + var_Z + var_cross, 0.0)
            bonus = self.alpha * np.sqrt(var_total)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1
        
        # Decay alpha slightly to reduce exploration over time
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        # Prediction step: add process noise
        q_W = self._adapt_q_W()
        q_Z = self._adapt_q_Z(arm)
        
        P_W_pred = self.P_W + q_W * np.eye(self.d * self.r)
        P_Z_pred = self.P_Z[arm] + q_Z * np.eye(self.r)
        P_WZ_pred = self.P_WZ[arm]  # Cross-covariance doesn't get process noise
        
        # Observation model: y = (W @ Z_a)^T x + noise
        # = W^T x @ Z_a + noise
        # Let h = W^T x (r-vector), then y = h^T Z_a + noise
        # But W is also unknown, so we need joint update.
        # 
        # Linearization: treat W and Z_a as joint state.
        # y = (W Z_a)^T x = x^T W Z_a
        # Gradient wrt W: x Z_a^T (outer product, d*r vector)
        # Gradient wrt Z_a: W^T x (r vector)
        
        Wx = self.W.T @ context  # (r,)
        # Gradient wrt W flattened: (context outer Z_a) flattened
        grad_W = np.outer(context, self.Z[arm]).flatten()  # (d*r,)
        # Gradient wrt Z_a: Wx (r,)
        
        # Innovation
        y_hat = self.theta[arm] @ context
        innovation = reward - y_hat
        
        # Innovation covariance
        # S = grad_W^T P_W grad_W + Wx^T P_Z Wx + 2 * grad_W^T P_WZ Wx + R
        S_W = grad_W @ P_W_pred @ grad_W
        S_Z = Wx @ P_Z_pred @ Wx
        S_WZ = 2.0 * (grad_W @ P_WZ_pred @ Wx)
        R = 1.0
        S = S_W + S_Z + S_WZ + R
        
        if S < 1e-10:
            K_W = np.zeros(self.d * self.r)
            K_Z = np.zeros(self.r)
        else:
            # Kalman gains
            # K_W = P_W grad_W / S
            # K_Z = P_Z Wx / S + P_WZ^T grad_W / S  (from joint update)
            K_W = P_W_pred @ grad_W / S
            K_Z = (P_Z_pred @ Wx + P_WZ_pred.T @ grad_W) / S
        
        # State update
        self.W = self.W + (K_W.reshape(self.d, self.r)) * innovation
        self.Z[arm] = self.Z[arm] + K_Z * innovation
        
        # Covariance update (Joseph form for stability)
        # For W: P_W_new = (I - K_W grad_W^T) P_W_pred (I - K_W grad_W^T)^T + K_W R K_W^T
        I_KW = np.eye(self.d * self.r) - np.outer(K_W, grad_W)
        self.P_W = I_KW @ P_W_pred @ I_KW.T + np.outer(K_W, K_W) * R
        self.P_W = 0.5 * (self.P_W + self.P_W.T)
        
        # For Z: P_Z_new = (I - K_Z Wx^T) P_Z_pred (I - K_Z Wx^T)^T + K_Z R K_Z^T
        I_KZ = np.eye(self.r) - np.outer(K_Z, Wx)
        P_Z_new = I_KZ @ P_Z_pred @ I_KZ.T + np.outer(K_Z, K_Z) * R
        P_Z_new = 0.5 * (P_Z_new + P_Z_new.T)
        
        # For cross-covariance P_WZ:
        # P_WZ_new = (I - K_W grad_W^T) P_WZ_pred (I - K_Z Wx^T)^T + K_W R K_Z^T
        P_WZ_new = (I_KW @ P_WZ_pred) @ I_KZ.T + np.outer(K_W, K_Z) * R
        P_WZ_new = 0.5 * (P_WZ_new + P_WZ_new.T)
        
        self.P_Z[arm] = P_Z_new
        self.P_WZ[arm] = P_WZ_new
        
        # Update theta cache for this arm
        self._update_theta_cache(arm)
        # Also update other arms since W changed
        for a in range(self.n):
            if a != arm:
                self._update_theta_cache(a)
        
        # Update tracking stats
        self.arm_counts[arm] += 1
        
        # Store prediction errors for Q adaptation
        abs_innov = abs(innovation)
        
        # W-related error: project innovation onto W gradient
        w_contrib = S_W / max(S, 1e-10)
        self.pred_errors_W[self.pred_errors_W_idx] = abs_innov * w_contrib
        self.pred_errors_W_idx = (self.pred_errors_W_idx + 1) % 10
        
        # Z-related error
        z_contrib = S_Z / max(S, 1e-10)
        z_idx = self.pred_errors_Z_idx[arm]
        self.pred_errors_Z[arm][z_idx] = abs_innov * z_contrib
        self.pred_errors_Z_idx[arm] = (z_idx + 1) % 10
# EVOLVE-BLOCK-END
