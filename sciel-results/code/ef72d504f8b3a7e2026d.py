"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using a single global linear model
    with context-augmented features (one-hot arm encoding + context).
    
    Strategy:
    1. Global Model: Instead of maintaining separate linear models per arm, we use a 
       single weight vector theta of dimension (n_arms + dim). The feature vector is 
       constructed by concatenating a one-hot encoding of the arm with the context:
       phi = [one_hot(arm), context].
       The reward is modeled as r = phi^T theta.
       
       This structure captures:
       - Arm-specific offsets (the first n_arms elements of theta).
       - Common context-dependent trends (the last dim elements of theta).
       - It does NOT allow for arm-specific context interactions (which would require
         n_arms * dim parameters, effectively the same as separate models).
       
    2. Exponential Forgetting: We use Recursive Least Squares (RLS) with an exponential
       forgetting factor gamma to handle non-stationarity. This is computationally 
       efficient (O(d^2) per update) and provides smooth adaptation.
       
    3. Thompson Sampling: We sample from the posterior distribution of theta to balance
       exploration and exploitation. The posterior covariance P is maintained recursively.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Exponential forgetting factor
        # 0.9 is a common choice for responsive adaptation
        self.gamma = 0.9
        
        # Regularization
        self.lambda_ = 1.0
        
        # Total feature dimension: n_arms (one-hot) + dim (context)
        self.total_dim = self.n + self.d
        
        # Global parameters:
        # theta: mean estimate of the global weight vector (total_dim,)
        # P: inverse of the effective covariance matrix (total_dim, total_dim)
        
        self.theta = np.zeros(self.total_dim)
        self.P = np.eye(self.total_dim) / self.lambda_
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def _get_features(self, arm, context):
        """
        Construct the augmented feature vector: [one_hot(arm), context]
        """
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        return np.concatenate([one_hot, context])

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy
        if time.time() > self.deadline:
            ctx = np.asarray(context, dtype=float)
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n):
                phi = self._get_features(a, ctx)
                val = self.theta @ phi
                if val > best_val:
                    best_val = val
                    best_arm = a
            return best_arm

        context = np.asarray(context, dtype=float)
        scores = []
        
        # Thompson Sampling: Sample theta_sample ~ N(theta, P)
        try:
            L = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self.P = self.P + 1e-6 * np.eye(self.total_dim)
            try:
                L = np.linalg.cholesky(self.P)
            except np.linalg.LinAlgError:
                # Fallback: use mean only
                L = None
        
        if L is not None:
            z = np.random.randn(self.total_dim)
            theta_sample = self.theta + L @ z
        else:
            theta_sample = self.theta
        
        for a in range(self.n):
            phi = self._get_features(a, context)
            score = theta_sample @ phi
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        phi = self._get_features(arm, context)
        
        # Recursive Least Squares Update with Exponential Forgetting
        P_prev = self.P
        
        # Compute K
        Pphi = P_prev @ phi
        denom = self.gamma + phi @ Pphi
        K = Pphi / denom
        
        # Update P
        # P_new = (1/gamma) * (I - K phi^T) * P_prev
        # P_new = (1/gamma) * P_prev - (1/gamma) * K[:, None] * (phi @ P_prev)
        P_new = (P_prev - np.outer(K, phi @ P_prev)) / self.gamma
        
        # Update theta
        # theta_t = theta_{t-1} + K (y - phi^T theta_{t-1})
        error = reward - phi @ self.theta
        self.theta = self.theta + K * error
        
        # Store updated P
        self.P = P_new
        
        self.t += 1
# EVOLVE-BLOCK-END
