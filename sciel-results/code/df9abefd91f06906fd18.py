# EVOLVE-BLOCK-START
"""Non-stationary linear contextual bandit with shared low-rank factorization and joint Kalman filter."""
import numpy as np
import time

class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy.
    
    Strategy:
    - Shared low-rank factorization: arm weights W_a = W @ v_a, where W is (d x r) and v_a is (r x 1).
    - Joint Kalman filter on flattened [W, v_1, ..., v_n] with adaptive process noise.
    - UCB exploration based on prediction uncertainty.
    
    The key insight is that by sharing the context-to-latent map W across arms, we enforce
    structural coherence, improving sample efficiency especially in the early phase or when
    arms are correlated. The Kalman filter with adaptive process noise (scaled by innovation)
    handles non-stationarity more gracefully than a fixed forgetting factor.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.r = min(4, dim)  # Rank of factorization
        
        # Total state dimension: d*r for W + r*n for V
        self.state_dim_W = self.d * self.r
        self.state_dim_V = self.r * self.n
        self.state_dim = self.state_dim_W + self.state_dim_V
        
        # Initialize state: W and V are small random vectors
        rng = np.random.RandomState(42)
        self.W = rng.randn(self.d, self.r) * 0.1
        self.V = rng.randn(self.r, self.n) * 0.1
        
        # Flatten state: [W.flatten(), V.flatten()]
        self.theta = np.concatenate([self.W.flatten(), self.V.flatten()])
        
        # Initialize covariance: large diagonal for high initial uncertainty
        # Use a scaled identity to maintain numerical stability
        self.P = np.eye(self.state_dim) * 10.0
        
        # Hyperparameters
        self.process_noise_base = 0.01  # Base process noise
        self.process_noise_adaptive = 0.1  # Additional noise scaled by innovation magnitude
        self.alpha_ucb = 1.5  # Exploration bonus coefficient
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin before 150s limit

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def _get_W(self):
        """Reconstruct W from state."""
        return self.theta[:self.state_dim_W].reshape(self.d, self.r)

    def _get_V(self):
        """Reconstruct V from state."""
        return self.theta[self.state_dim_W:].reshape(self.r, self.n)

    def _compute_prediction(self, W, V, arm, context):
        """Compute predicted reward for arm given context."""
        # W is (d, r), V[:, arm] is (r, 1)
        # W @ V[:, arm] gives (d, 1), then dot with context (d,) gives scalar
        w_eff = W @ V[:, arm]
        return float(context @ w_eff)

    def _compute_uncertainty(self, P, W, V, arm, context):
        """
        Compute prediction uncertainty using the Jacobian of the prediction w.r.t. the state.
        
        pred = context^T W V_a
        Let w = W.flatten(), v = V.flatten()
        d pred / d w = outer(context, V[:, arm]) flattened -> (d*r,)
        d pred / d v_a = W^T context -> (r,)
        For v_j (j != a), the derivative is 0.
        
        So the full Jacobian J has:
        - First state_dim_W entries: outer(context, V[:, arm]).flatten()
        - Entries for V[:, arm]: W^T context
        - All other V entries: 0
        """
        J = np.zeros(self.state_dim)
        
        # Derivative w.r.t. W
        dW = np.outer(context, V[:, arm]).flatten()
        J[:self.state_dim_W] = dW
        
        # Derivative w.r.t. V[:, arm]
        dV_arm = W.T @ context  # (r,)
        v_arm_start = self.state_dim_W + arm * self.r
        v_arm_end = v_arm_start + self.r
        J[v_arm_start:v_arm_end] = dV_arm
        
        # Compute J^T P J
        PJ = P @ J
        unc_sq = float(J @ PJ)
        if unc_sq < 0:
            unc_sq = 0.0
        return np.sqrt(unc_sq)

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))
        
        context = np.asarray(context, dtype=np.float64)
        
        W = self._get_W()
        V = self._get_V()
        
        # Compute UCB values for each arm
        ucb_values = np.zeros(self.n)
        for a in range(self.n):
            pred = self._compute_prediction(W, V, a, context)
            unc = self._compute_uncertainty(self.P, W, V, a, context)
            ucb_values[a] = pred + self.alpha_ucb * unc
        
        return int(np.argmax(ucb_values))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        W = self._get_W()
        V = self._get_V()
        
        # Compute prediction
        pred = self._compute_prediction(W, V, arm, context)
        innovation = reward - pred
        
        # Compute Jacobian
        J = np.zeros(self.state_dim)
        dW = np.outer(context, V[:, arm]).flatten()
        J[:self.state_dim_W] = dW
        v_arm_start = self.state_dim_W + arm * self.r
        v_arm_end = v_arm_start + self.r
        dV_arm = W.T @ context
        J[v_arm_start:v_arm_end] = dV_arm
        
        # Kalman Filter Update
        PJ = self.P @ J
        denom = 1.0 + float(J @ PJ)
        if denom < 1e-12:
            denom = 1e-12
        
        K = PJ / denom
        
        # Update state
        self.theta += K * innovation
        
        # Update covariance
        # P_new = (I - K J^T) P
        # For numerical stability, use the Joseph form or direct update
        I = np.eye(self.state_dim)
        self.P = (I - np.outer(K, J)) @ self.P
        
        # Symmetrize
        self.P = (self.P + self.P.T) / 2.0
        
        # Adaptive process noise: scale by innovation magnitude
        # This allows faster adaptation when the model is wrong (large innovation)
        innovation_magnitude = abs(innovation)
        noise_scale = self.process_noise_base + self.process_noise_adaptive * min(innovation_magnitude, 5.0)
        self.P += np.eye(self.state_dim) * noise_scale
        
        # Clip P to prevent numerical issues
        np.clip(self.P, -1e6, 1e6, out=self.P)
# EVOLVE-BLOCK-END
