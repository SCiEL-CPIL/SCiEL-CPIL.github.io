import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget
        used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Data collection for surrogate
        X_data = []
        F_data = []

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal used, best_f, best_x
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            used += 1
            if np.isfinite(f_val):
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clip.copy()
                X_data.append(x_clip.copy())
                F_data.append(f_val)
            return f_val

        # --- Surrogate Model: Sparse Gaussian Process ---
        class SparseGP:
            def __init__(self, dim, n_inducing=20):
                self.dim = dim
                self.n_inducing = n_inducing
                self.K = 1.0  # Length scale
                self.var = 1.0 # Signal variance
                self.nu = 2.0  # Smoothness
                self.Z = None
                self.alpha = None
                self.mean = None
                self.var_pred = None

            def _kern(self, A, B):
                # Matern 5/2 kernel
                sq_dist = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                sq_dist = np.maximum(sq_dist, 0)
                r = np.sqrt(sq_dist) / self.K
                return self.var * (1 + np.sqrt(5) * r + 5/3 * r**2) * np.exp(-np.sqrt(5) * r)

            def fit(self, X, Y):
                # Select inducing points via farthest point sampling or k-means-like
                n = X.shape[0]
                if n <= self.n_inducing:
                    self.Z = X.copy()
                else:
                    # Simple greedy selection: pick points far from existing set
                    Z = [X[0]]
                    while len(Z) < self.n_inducing:
                        Z_arr = np.array(Z)
                        dists = np.min(np.sum((X[:, None, :] - Z_arr[None, :, :])**2, axis=2), axis=1)
                        next_idx = np.argmax(dists)
                        Z.append(X[next_idx])
                    self.Z = np.array(Z)

                # Compute K_XX (Z,Z), K_XZ (X,Z)
                K_XX = self._kern(self.Z, self.Z)
                K_XZ = self._kern(X, self.Z)
                
                # Regularization
                K_XX += 1e-6 * np.eye(len(self.Z))
                
                # Solve for alpha
                try:
                    L = np.linalg.cholesky(K_XX)
                    v = np.linalg.solve(L.T, np.linalg.solve(L, K_XZ.T))
                    # v is n_inducing x n
                    # alpha = K_XX^{-1} Y
                    # We need K_XZ.T @ alpha? No.
                    # y = K_XZ @ alpha => alpha = K_ZZ^{-1} K_ZX^T y
                    # K_ZX = K_XZ.T
                    # alpha = K_ZZ^{-1} K_XZ.T @ Y
                    alpha_tmp = np.linalg.solve(L.T, np.linalg.solve(L, K_XZ.T @ Y))
                    self.alpha = alpha_tmp
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    self.alpha = np.linalg.pinv(K_XX) @ (K_XZ.T @ Y)

            def predict(self, X_new):
                K_new_Z = self._kern(X_new, self.Z)
                mean = K_new_Z @ self.alpha
                # Variance
                K_ZZ = self._kern(self.Z, self.Z) + 1e-6 * np.eye(len(self.Z))
                try:
                    K_inv = np.linalg.inv(K_ZZ)
                except:
                    K_inv = np.linalg.pinv(K_ZZ)
                
                var = self.var - np.sum((K_new_Z @ K_inv) * K_new_Z, axis=1)
                var = np.maximum(var, 1e-10)
                return mean, var

        # --- Optimization Loop ---
        
        # Phase 1: Initial Sampling (Latin Hypercube + Center)
        # Use 20% of budget for initial global exploration
        init_budget = int(total_budget * 0.2)
        if init_budget < 10:
            init_budget = 10
        if init_budget > 50:
            init_budget = 50

        # Center
        evaluate(np.zeros(dim))
        
        # LHS points
        rng = np.random.default_rng(12345)
        for _ in range(init_budget - 1):
            if used >= total_budget or check_deadline():
                break
            # Sample from LHS-like distribution
            x = rng.uniform(-4, 4, dim)
            # Add some local exploration around 0
            if rng.random() < 0.2:
                x = rng.normal(0, 1.5, dim)
            evaluate(x)

        # Phase 2: Surrogate-Guided Search
        # Fit GP, predict EI, select next point, evaluate, update GP
        gp = SparseGP(dim, n_inducing=min(30, max(10, len(X_data))))
        
        # We will refit the GP every few steps to keep it fresh and handle drift
        refit_interval = 5
        
        current_gp_fit = 0
        
        while used < total_budget and not check_deadline():
            # Refit GP periodically or if data has grown significantly
            if current_gp_fit == 0 or len(X_data) - current_gp_fit >= refit_interval:
                if len(X_data) >= 5:
                    X_arr = np.array(X_data)
                    F_arr = np.array(F_data)
                    # Normalize Y for numerical stability in GP? 
                    # For EI, we need raw values, but GP internals might benefit from normalization.
                    # Let's stick to raw for simplicity in EI calculation, but ensure scale is reasonable.
                    gp.fit(X_arr, F_arr)
                    current_gp_fit = len(X_data)
                else:
                    # Not enough data for GP, use random
                    x_new = rng.uniform(-4, 4, dim)
                    evaluate(x_new)
                    continue

            # Generate candidate points
            # 1. Random candidates
            # 2. Local perturbations of best
            # 3. GP-based EI maximization
            
            candidates = []
            
            # Random
            for _ in range(5):
                candidates.append(rng.uniform(-4, 4, dim))
            
            # Local around best
            for _ in range(5):
                x_try = best_x + rng.normal(0, 0.5, dim)
                candidates.append(np.clip(x_try, -5, 5))
            
            # GP-based EI
            # Sample a grid of points to find max EI
            # Since dim=5, a grid is expensive. Use random sampling for EI maximization.
            # Sample 200 points, compute EI, pick best.
            ei_candidates = rng.uniform(-4, 4, (200, dim))
            
            # Predict mean and var for these candidates
            try:
                mean, var = gp.predict(ei_candidates)
                # Expected Improvement
                # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
                # Z = (f_best - mu) / sigma
                sigma = np.sqrt(var)
                Z = (best_f - mean) / (sigma + 1e-8)
                # Standard Normal CDF and PDF
                # Use scipy.stats.norm if available, else approximate
                from scipy.stats import norm
                EI = (best_f - mean) * norm.cdf(Z) + sigma * norm.pdf(Z)
                best_ei_idx = np.argmax(EI)
                x_gp = ei_candidates[best_ei_idx]
                candidates.append(x_gp)
            except Exception:
                pass

            # Evaluate the best candidate from the set? 
            # Or evaluate a few top ones? 
            # To save budget, let's pick the one with highest EI from the GP set, 
            # but also consider the local/random ones if they are diverse.
            
            # Simple strategy: Pick the GP candidate if EI is significant, otherwise random/local
            # Let's just evaluate the GP candidate first, then maybe one random.
            # Actually, let's evaluate the top 2 from the combined pool based on a heuristic.
            # For now, let's just pick the GP candidate. It's the "model-driven" part.
            
            x_next = x_gp
            
            # If GP prediction is bad (high uncertainty and high mean), maybe random is better?
            # EI handles this.
            
            evaluate(x_next)

            # If we have budget left, maybe do a second evaluation per loop?
            # No, keep it simple. 1 eval per loop iteration.
            # The loop will run until budget is exhausted.

        return best_x, best_f
