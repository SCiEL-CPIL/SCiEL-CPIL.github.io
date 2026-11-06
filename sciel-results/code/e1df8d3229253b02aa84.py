import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Global model dimension: context dim + one-hot arm indicators
        self.D = dim + n_arms
        
        # Initialize global theta and P
        # P is inverse covariance (precision) matrix
        # Initialize with large uncertainty
        self.P = np.eye(self.D) * 10.0
        self.theta = np.zeros(self.D)
        
        # Forgetting factor parameters
        self.lam_min = 0.80   # Fast forgetting during high error
        self.lam_max = 0.995  # Slow forgetting during low error
        
        # Error tracking for adaptive forgetting
        self.sq_error = 0.0
        self.sq_error_alpha = 0.3
        self.error_scale = 1.0 # Adaptive scale for normalization
        
        # Exploration
        self.alpha_base = 1.5
        self.total_rounds = 0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

    def _get_lambda(self):
        """
        Compute adaptive forgetting factor based on global squared prediction error.
        """
        # Normalize error using adaptive scale
        # Update scale estimate
        # Simple robust scale: running max or percentile. 
        # Here we use a simple adaptive scale that grows with error.
        if self.sq_error > self.error_scale:
            self.error_scale = self.error_scale * 0.9 + self.sq_error * 0.1
        elif self.sq_error < self.error_scale * 0.1:
            self.error_scale = self.error_scale * 0.9 + self.sq_error * 0.1
            
        normalized = self.sq_error / max(self.error_scale, 1e-6)
        
        # Smooth mapping
        t = 1.0 / (1.0 + np.exp(-10.0 * (normalized - 1.0)))
        lam = self.lam_max - t * (self.lam_max - self.lam_min)
        return np.clip(lam, self.lam_min, self.lam_max)

    def _get_alpha(self):
        """
        Adaptive exploration parameter.
        """
        if self.total_rounds < 30:
            return self.alpha_base + 1.0
        decay = 1.0 / (1.0 + self.total_rounds / 200.0)
        return self.alpha_base * decay + 0.5

    def _make_feature(self, context, arm):
        """
        Construct the global feature vector: [context, one_hot_arm]
        """
        x = np.zeros(self.D)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        alpha = self._get_alpha()
        
        best_score = -np.inf
        best_arm = 0
        
        # Pre-compute context part for efficiency? 
        # x_a = [context, e_a]
        # x_a^T P x_a = context^T P_cc context + 2 * e_a^T P_ca context + e_a^T P_aa e_a
        # Since e_a is one-hot, this simplifies.
        
        # Let P be partitioned:
        # P = [[P_cc, P_ca], [P_ac, P_aa]]
        # P_cc is d x d, P_ca is d x n, P_aa is n x n
        
        P_cc = self.P[:self.d, :self.d]
        P_ca = self.P[:self.d, self.d:]
        P_aa = self.P[self.d:, self.d:]
        
        # theta partitioned:
        theta_c = self.theta[:self.d]
        theta_a = self.theta[self.d:]
        
        # Base score part from context: theta_c^T context
        base_mu = theta_c @ context
        
        # Uncertainty part from context: context^T P_cc context
        Px = P_cc @ context
        var_c = context @ Px
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Mean: base_mu + theta_a[a]
            mu = base_mu + theta_a[a]
            
            # Variance: var_c + 2 * context^T P_ca[:, a] + P_aa[a, a]
            # P_ca[:, a] is the column a of P_ca
            cross_term = 2.0 * (Px @ P_ca[:, a]) # Wait, Px is P_cc x. We need x^T P_ca[:, a]
            # x^T P_ca[:, a] is the same as (P_ca[:, a])^T x
            cross_val = P_ca[:, a] @ context
            var = var_c + 2.0 * cross_val + P_aa[a, a]
            std = np.sqrt(max(0.0, var))
            
            score = mu + alpha * std
            scores[a] = score
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check
        if time.time() > self._deadline:
            return

        self.total_rounds += 1
        
        # Construct feature vector
        x = self._make_feature(context, arm)
        
        # Compute prediction error
        pred = self.theta @ x
        error = reward - pred
        sq_err = error * error
        
        # Update global EMA of squared error
        self.sq_error = (1.0 - self.sq_error_alpha) * self.sq_error + self.sq_error_alpha * sq_err
        
        # Get adaptive forgetting factor
        lam = self._get_lambda()
        
        # RLS Update
        P = self.P
        theta = self.theta
        
        Px = P @ x
        denom = lam + x @ Px
        if denom < 1e-12:
            denom = 1e-12
            
        K = Px / denom
        
        # Update theta
        self.theta = theta + K * error
        
        # Update P: P = (P - K x^T P) / lam
        # Using matrix identity
        self.P = (P - np.outer(K, x) @ P) / lam
        
        # Ensure symmetry and positive definiteness
        P_new = self.P
        P_new = (P_new + P_new.T) * 0.5
        P_new += 1e-6 * np.eye(self.D)
        self.P = P_new
