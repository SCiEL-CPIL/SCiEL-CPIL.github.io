# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """
    Bayesian Optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    
    Strategy:
    1. Initialize with a set of random points to establish a baseline surrogate.
    2. Fit a Gaussian Process (GP) surrogate model to the observed data.
    3. Use Expected Improvement (EI) acquisition function to select the next point.
    4. Update the model and repeat until budget is exhausted.
    5. Handle the 'step' nature by relying on the GP's ability to model non-smooth 
       transitions through its covariance structure (Matern 5/2).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        self.X_obs = []
        self.y_obs = []

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None
        self.X_obs = []
        self.y_obs = []

        rng = np.random.default_rng(42)

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            self.X_obs.append(x)
            self.y_obs.append(fx)
            return fx

        # Helper: Gaussian Process kernel (Matern 5/2)
        def matern_52_kernel(X1, X2, l, sigma_f, sigma_n):
            """
            Matern 5/2 kernel:
            K(x, x') = sigma_f^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
            where r = ||x - x'|| / l
            """
            # Compute pairwise distances
            # X1 shape: (N1, d), X2 shape: (N2, d)
            X1_exp = X1[:, np.newaxis, :]
            X2_exp = X2[np.newaxis, :, :]
            diff = X1_exp - X2_exp
            r = np.sqrt(np.sum(diff**2, axis=-1))
            
            r = np.maximum(r, 1e-10) # Avoid division by zero issues if needed, though sqrt handles 0
            
            sqrt5 = np.sqrt(5.0)
            term = (1.0 + sqrt5 * r + (5.0 / 3.0) * r**2) * np.exp(-sqrt5 * r)
            
            K = sigma_f**2 * term
            # Add noise on diagonal for X1 == X2
            if X1.shape[0] == X2.shape[0] and np.array_equal(X1, X2):
                K += sigma_n**2 * np.eye(X1.shape[0])
            
            return K

        # Helper: Fit GP parameters via simple grid search or fixed values
        # For robustness and speed, we use fixed reasonable hyperparameters 
        # and rely on the EI acquisition.
        # Length scale l: controls correlation length. 
        # Signal variance sigma_f^2: scale of function.
        # Noise variance sigma_n^2: observation noise.
        
        def predict_gp(X_new):
            """
            Predict mean and variance at X_new given observed X_obs, y_obs.
            """
            if len(self.X_obs) == 0:
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0]) * 100.0
            
            X_train = np.array(self.X_obs)
            y_train = np.array(self.y_obs)
            
            # Hyperparameters
            l = 1.0  # Length scale
            sigma_f = 10.0 # Signal std
            sigma_n = 0.01 # Noise std (small, as func is deterministic)
            
            # Compute K(X_train, X_train)
            K = matern_52_kernel(X_train, X_train, l, sigma_f, sigma_n)
            
            # Compute K(X_train, X_new)
            K_star = matern_52_kernel(X_train, X_new, l, sigma_f, 0.0)
            
            # Solve for alpha = K^{-1} y
            try:
                # Add small jitter for numerical stability
                K_reg = K + 1e-6 * np.eye(K.shape[0])
                alpha = np.linalg.solve(K_reg, y_train)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                alpha = np.linalg.pinv(K_reg) @ y_train
                
            # Mean prediction
            mu = K_star.T @ alpha
            
            # Variance prediction
            K_new = matern_52_kernel(X_new, X_new, l, sigma_f, 0.0)
            var = np.diag(K_new) - np.sum(K_star**2 * alpha[:, np.newaxis], axis=0)
            var = np.maximum(var, 1e-10) # Ensure positive variance
            
            return mu, var

        def expected_improvement(mu, var, best_f, xi=0.01):
            """
            Expected Improvement acquisition function.
            """
            imp = best_f - mu - xi
            # Avoid division by zero
            std = np.sqrt(var)
            std = np.maximum(std, 1e-10)
            
            z = imp / std
            ei = imp * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Phase 1: Initial Exploration
        # Sample 20 points randomly to initialize the GP
        n_init = 20
        if budget < n_init + 10:
             n_init = max(5, budget // 2)
             
        for _ in range(n_init):
            if self._call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            wrapped(x)
            
        if self._call_count >= budget or time.time() > deadline:
            return self._best_x, self._best_f

        # Phase 2: Bayesian Optimization Loop
        # We will evaluate one point at a time, updating the model each time.
        # To speed up, we might batch a few evaluations if the model is cheap, 
        # but for 5D and 500 budget, sequential is fine and more accurate.
        
        # Pre-allocate some memory if needed, but lists are fine for 500 items.
        
        while self._call_count < budget:
            if time.time() > deadline:
                break
                
            # 1. Generate candidate points
            # Sample 1000 random candidates from the box
            n_candidates = 1000
            X_cand = rng.uniform(-5.0, 5.0, (n_candidates, dim))
            
            # 2. Predict mean and variance
            mu, var = predict_gp(X_cand)
            
            # 3. Compute EI
            ei = expected_improvement(mu, var, self._best_f)
            
            # 4. Select best candidate
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            # 5. Evaluate
            wrapped(x_next)
            
            # Optional: Early stop if variance is very low everywhere and no improvement
            # But GP handles this via EI -> 0.
            
        return self._best_x, self._best_f

# EVOLVE-BLOCK-END
