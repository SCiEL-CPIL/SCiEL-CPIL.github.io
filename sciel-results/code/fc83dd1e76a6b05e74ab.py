# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Bayesian Optimization optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    Uses a Gaussian Process surrogate model with Expected Improvement acquisition
    to select next points, maximizing information gain per evaluation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        # Surrogate model data
        X = []
        Y = []
        
        # Kernel parameters
        # For BBOB, scales are often around 1-10. We use a fixed reasonable kernel.
        # Matern 5/2 is robust.
        def kernel(x1, x2, l=1.5, sigma_f=1.0, noise=1e-6):
            """
            Matern 5/2 kernel + noise.
            x1: (n, d) or (d,)
            x2: (m, d) or (d,)
            """
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
            
            # Compute pairwise squared distances
            # Using broadcasting: (n, 1, d) - (1, m, d) -> (n, m, d)
            diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            # Avoid division by zero for identical points
            sq_dist[sq_dist == 0] = 1e-12
            
            r = np.sqrt(sq_dist)
            # Matern 5/2: (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
            sqrt5_r = np.sqrt(5.0) * r
            K = (1.0 + sqrt5_r + (5.0/3.0) * r**2) * np.exp(-sqrt5_r)
            
            # Scale by length scale
            # Note: The standard Matern kernel argument is r/l. 
            # My r is actual distance. So I should have computed r/l inside.
            # Let's fix: The formula above assumes r is already scaled? 
            # Standard: K(x,y) = sigma_f^2 * (1 + sqrt(5)*d/l + 5/3*(d/l)^2) * exp(-sqrt(5)*d/l)
            # My sq_dist is d^2. r is d.
            # So I need to divide r by l.
            # Let's re-implement kernel properly to be safe.
            pass
            
        def kernel_matern52(x1, x2, l=1.5, sigma_f=1.0, noise=1e-6):
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
            
            diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            # d / l
            r = np.sqrt(sq_dist) / l
            
            # Matern 5/2
            K = (1.0 + np.sqrt(5.0) * r + (5.0/3.0) * r**2) * np.exp(-np.sqrt(5.0) * r)
            
            # Add noise on diagonal
            if x1.shape[0] == x2.shape[0] and np.allclose(x1, x2):
                K += noise * np.eye(x1.shape[0])
                
            return sigma_f**2 * K

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            
            # Add to surrogate
            X.append(np.array(x, dtype=float))
            Y.append(fx)
            
            return fx

        # Initialize with some diverse points to give the GP a starting frame
        # Center and a few random points
        rng = np.random.default_rng(42)
        
        init_points = []
        init_points.append(np.zeros(dim))
        # Add random points
        for _ in range(4):
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        
        # Evaluate initial points
        for p in init_points:
            if self._call_count >= budget:
                break
            if time.time() > deadline:
                break
            wrapped(p)
            
        # Bayesian Optimization Loop
        # We will use Expected Improvement (EI) with a small exploration bonus.
        # To select the next point, we sample candidates from a grid or random set
        # and evaluate the acquisition function.
        
        # Parameters for GP
        l = 1.5
        sigma_f = 1.0
        noise = 1e-6
        
        # Pre-compute alpha for GP updates? 
        # Since N is small (< 500), we can solve linear system each time.
        # K_inv * y
        
        while self._call_count < budget:
            if time.time() > deadline:
                break
            
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Fit GP
            K = kernel_matern52(X_arr, X_arr, l=l, sigma_f=sigma_f, noise=noise)
            try:
                K_inv = np.linalg.inv(K)
                alpha = K_inv @ Y_arr
            except np.linalg.LinAlgError:
                # Fallback if singular (should be rare with noise)
                K_inv = np.linalg.pinv(K)
                alpha = K_inv @ Y_arr
            
            # Generate candidates
            # Use a mix of random points and local perturbations around best
            # to balance exploration and exploitation.
            
            n_candidates = 50
            candidates = []
            
            # 50% random
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-4.9, 4.9, dim))
            
            # 50% local around best
            if self._best_x is not None:
                for _ in range(n_candidates // 2):
                    # Gaussian perturbation
                    # Scale based on current best? Or fixed?
                    # Fixed small scale for exploitation
                    pert = rng.normal(0, 0.5, dim)
                    x_cand = self._best_x + pert
                    x_cand = np.clip(x_cand, -4.9, 4.9)
                    candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Evaluate Acquisition Function (Expected Improvement)
            # EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z)
            # where Z = (mu(x) - f_best - xi) / sigma(x)
            
            f_best = self._best_f
            xi = 0.01 * (abs(f_best) + 1e-6) # Small exploration
            
            K_s = kernel_matern52(X_arr, candidates, l=l, sigma_f=sigma_f, noise=0.0)
            # Note: noise is 0 for prediction covariance
            
            mu = K_s @ alpha
            # Variance: sigma_f^2 - K_s @ K_inv @ K_s.T
            # Efficient calculation:
            # v = K_inv @ K_s.T
            # var = sigma_f^2 - np.sum(K_s * v, axis=1)
            
            v = K_inv @ K_s.T
            var = sigma_f**2 - np.sum(K_s * v, axis=1)
            var = np.maximum(var, 1e-10) # Numerical stability
            sigma = np.sqrt(var)
            
            # Calculate EI
            # Handle sigma == 0
            Z = (mu - f_best - xi) / sigma
            # Phi: CDF of standard normal
            # phi: PDF of standard normal
            # Using math.erf for Phi
            # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
            # phi(z) = 1/sqrt(2*pi) * exp(-z^2/2)
            
            Phi = 0.5 * (1.0 + np.vectorize(math.erf)(Z / np.sqrt(2.0)))
            phi = np.exp(-0.5 * Z**2) / np.sqrt(2.0 * np.pi)
            
            EI = (mu - f_best - xi) * Phi + sigma * phi
            
            # Select best candidate
            idx_best_cand = np.argmax(EI)
            x_next = candidates[idx_best_cand]
            
            # Evaluate
            wrapped(x_next)
            
            # Optional: If we get stuck (very low EI), force a random jump?
            # The mix above handles some of this.
            
        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f

# EVOLVE-BLOCK-END
