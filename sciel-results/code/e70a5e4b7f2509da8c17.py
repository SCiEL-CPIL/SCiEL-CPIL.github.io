"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Kalman-Filter-Inspired Thompson Sampling for Non-Stationary Linear Bandits.
    
    Combines the structural idea of Thompson Sampling (Bayesian posterior sampling)
    with a principled state-space model where arm weights follow a random walk.
    
    Key mechanisms:
    1. Random Walk Model: theta_t = theta_{t-1} + w_t, w_t ~ N(0, Q)
    2. Adaptive Process Noise: Q is estimated online from innovation sequences
       (prediction errors), replacing heuristic thresholds.
    3. Thompson Sampling: Sample from posterior N(theta_hat, Sigma) to balance
       exploration and exploitation.
    4. Adaptive Forgetting: The effective forgetting rate is driven by the
       estimated process noise covariance, allowing fast adaptation during
       regime shifts and stable estimates during quiet periods.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110  # Safety margin

        # Initialize per-arm posterior parameters
        # A[a] is precision matrix (inverse covariance)
        # b[a] is precision vector
        # Start with a weak prior: A = I, b = 0
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Kalman filter parameters for random walk model
        # Q is the process noise covariance (how much weights can change per step)
        # We estimate Q online using innovation sequences
        self.Q_base = 0.01  # Base process noise level
        self.Q_scale = 1.0  # Adaptive scale factor for Q
        self.Q_min = 0.001  # Minimum process noise
        self.Q_max = 1.0    # Maximum process noise
        
        # Innovation tracking for adaptive Q estimation
        self.innovation_history = []
        self.innovation_window = 20
        self.innovation_threshold = 1.0
        
        # Precompute Cholesky decompositions for efficiency
        self._cache_chol = [None] * n_arms
        self._cache_time = [0.0] * n_arms

    def _get_chol(self, a):
        """Get or compute Cholesky decomposition with caching."""
        current_time = time.time()
        if current_time > self.deadline:
            return np.linalg.cholesky(self.A[a] + 1e-6 * np.eye(self.dim))
        
        # Check if cache is valid
        if self._cache_chol[a] is None or current_time - self._cache_time[a] > 0.001:
            try:
                # Add small regularization for numerical stability
                A_reg = self.A[a] + 1e-6 * np.eye(self.dim)
                self._cache_chol[a] = np.linalg.cholesky(A_reg)
                self._cache_time[a] = current_time
            except np.linalg.LinAlgError:
                # Fallback: add more regularization
                A_reg = self.A[a] + 1e-3 * np.eye(self.dim)
                self._cache_chol[a] = np.linalg.cholesky(A_reg)
                self._cache_time[a] = current_time
        return self._cache_chol[a]

    def _estimate_process_noise(self, arm, context, reward):
        """
        Estimate process noise Q from innovation sequences.
        
        The innovation is the difference between observed reward and predicted reward.
        In a random walk model, large innovations suggest either high observation noise
        or high process noise (regime change). We adapt Q to balance these.
        """
        try:
            # Get current estimate
            chol = self._get_chol(arm)
            L = chol
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            prediction = theta @ context
            
            # Innovation (prediction error)
            innovation = reward - prediction
            innovation_sq = innovation ** 2
            
            # Track innovation history
            self.innovation_history.append(innovation_sq)
            if len(self.innovation_history) > self.innovation_window:
                self.innovation_history.pop(0)
            
            # Estimate Q based on average innovation
            # If innovations are large, increase Q (faster adaptation)
            # If innovations are small, decrease Q (more stable)
            if len(self.innovation_history) >= 5:
                avg_innovation = np.mean(self.innovation_history)
                
                # Map average innovation to Q scale
                # Use a smooth mapping: Q_scale = base * (1 + avg_innovation / threshold)
                # Clip to [Q_min, Q_max]
                target_scale = 1.0 + (avg_innovation / self.innovation_threshold)
                target_scale = np.clip(target_scale, self.Q_min / self.Q_base, self.Q_max / self.Q_base)
                
                # Smoothly adapt Q_scale
                self.Q_scale = 0.9 * self.Q_scale + 0.1 * target_scale
                self.Q_scale = np.clip(self.Q_scale, self.Q_min / self.Q_base, self.Q_max / self.Q_base)
                
        except Exception:
            pass

    def _apply_kalman_update(self, arm, context, reward):
        """
        Apply Kalman-filter-style update with adaptive process noise.
        
        The update combines:
        1. State prediction: theta_pred = theta_old (random walk, no drift)
        2. Measurement update: Standard Bayesian update with new observation
        3. Adaptive forgetting: The effective prior is weakened by Q
        """
        # Estimate process noise
        self._estimate_process_noise(arm, context, reward)
        
        # Current Q
        Q = self.Q_base * self.Q_scale
        
        # In a random walk model, the prior precision for the next step is:
        # A_prior = (A_old + Q^{-1} I)^{-1} ... but we work with precision directly
        # The effective forgetting is: A_new = (A_old^{-1} + Q I)^{-1}
        # But computing inverse is expensive. Instead, we use an approximation:
        # A_new ≈ A_old * (1 - alpha) + alpha * Q^{-1} I, where alpha controls forgetting
        
        # Simpler approach: Scale A by a forgetting factor derived from Q
        # The effective number of "equivalent observations" in the prior is:
        # n_eff = trace(A_old) / dim
        # The process noise adds uncertainty: A_prior = (A_old^{-1} + Q I)^{-1}
        # For large A_old (small uncertainty), A_prior ≈ A_old - Q I (first order)
        # For small A_old (large uncertainty), A_prior ≈ (1/Q) I
        
        # We use a practical approximation:
        # A_new = A_old * beta + (1 - beta) * Q^{-1} * I
        # where beta is chosen to balance the two terms
        
        # Compute beta based on Q and current A
        # If Q is large (high process noise), beta should be small (forget more)
        # If Q is small (low process noise), beta should be large (retain more)
        
        # Trace of A gives a measure of information
        trace_A = np.trace(self.A[arm])
        n_eff = trace_A / self.dim
        
        # beta = n_eff / (n_eff + 1/Q)
        # This gives the relative weight of old information vs. process noise
        beta = n_eff / (n_eff + 1.0 / Q) if Q > 0 else 1.0
        beta = np.clip(beta, 0.5, 0.999)  # Keep beta in a reasonable range
        
        # Apply update
        A_old = self.A[arm]
        b_old = self.b[arm]
        
        # New prior: A_prior = beta * A_old + (1 - beta) * (1/Q) * I
        # But this isn't quite right. Let's use the standard discounted update:
        # A_new = discount * A_old + outer(x, x)
        # b_new = discount * b_old + reward * x
        
        # The discount factor should reflect the process noise:
        # If Q is large, we should discount more aggressively
        # discount = 1 / (1 + Q * trace(A_old))
        discount = 1.0 / (1.0 + Q * trace_A)
        discount = np.clip(discount, 0.7, 0.999)
        
        # Apply discounted update
        self.A[arm] = discount * A_old + np.outer(context, context)
        self.b[arm] = discount * b_old + reward * context
        
        # Invalidate cache
        self._cache_chol[arm] = None

    def select(self, context):
        """Select arm using Thompson Sampling."""
        if time.time() > self.deadline:
            # Emergency fallback: return best arm based on current estimates
            scores = []
            for a in range(self.n_arms):
                try:
                    chol = self._get_chol(a)
                    L = chol
                    theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                    scores.append(theta @ context)
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))
        
        scores = []
        for a in range(self.n_arms):
            try:
                chol = self._get_chol(a)
                L = chol
                # Solve for theta: A @ theta = b
                theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                # Sample from posterior: theta ~ N(theta_hat, (L^{-1})^T @ L^{-1})
                # theta_sample = theta_hat + L^{-T} @ z where z ~ N(0, I)
                z = np.random.randn(self.dim)
                w = np.linalg.solve(L.T, z)
                theta_sample = theta + w
                scores.append(theta_sample @ context)
            except Exception:
                # Fallback to mean estimate
                try:
                    chol = self._get_chol(a)
                    L = chol
                    theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                    scores.append(theta @ context)
                except Exception:
                    scores.append(0.0)
        
        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model with observed reward using Kalman-filter-style update."""
        self._apply_kalman_update(arm, context, reward)
# EVOLVE-BLOCK-END
