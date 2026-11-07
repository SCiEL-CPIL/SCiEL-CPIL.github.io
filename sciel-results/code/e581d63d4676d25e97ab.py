"""Bayesian Optimization with Gaussian Process surrogate for BBOB functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform

class Optimizer:
    """
    Bayesian Optimizer using a Gaussian Process surrogate.
    
    Strategy:
    - Initialize with a small space-filling sample (Latin Hypercube).
    - Iteratively select the next point to evaluate using Expected Improvement (EI).
    - Use a Matern-5/2 kernel for flexibility.
    - Handle the box constraint [-5, 5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        # 1. Initialization: Latin Hypercube Sampling for diverse start points
        n_init = 15
        rng = np.random.default_rng(42)
        
        # Latin Hypercube
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            X_init[:, i] = rng.permutation(n_init) / n_init * 10.0 - 5.0
            
        X = X_init.copy()
        Y = np.array([func(x) for x in X])
        calls = n_init
        
        best_idx = np.argmin(Y)
        y_best = Y[best_idx]
        x_best = X[best_idx].copy()

        # GP Parameters
        # Matern 5/2 kernel parameters
        # L: length scale (initially 1.0)
        # sigma_f: signal variance (initially 1.0)
        # sigma_n: noise variance (small, e.g., 1e-4)
        
        def matern_52_kernel(X1, X2, L=1.0):
            """
            Compute the Matern 5/2 kernel matrix between X1 and X2.
            X1: (n1, d), X2: (n2, d)
            """
            # Euclidean distances
            D = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2))
            # Matern 5/2 formula
            sqrt3_D = np.sqrt(3) * D
            K = (1 + sqrt3_D + sqrt3_D**2 / 3) * np.exp(-sqrt3_D)
            return K

        def gp_predict(X_new, X_train, Y_train, L, sigma_f, sigma_n):
            """
            Predict mean and variance of GP at X_new given training data.
            """
            K = matern_52_kernel(X_train, X_train, L) + sigma_n * np.eye(len(X_train))
            K_s = matern_52_kernel(X_new, X_train, L)
            
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # Regularization if singular
                K_inv = np.linalg.pinv(K)
                
            mu = K_s @ K_inv @ Y_train
            v = K_s @ K_inv @ K_s.T
            var = sigma_f**2 - np.diag(v)
            var = np.maximum(var, 1e-10) # Ensure positive variance
            return mu, var

        def expected_improvement(mu, var, y_best, xi=0.01):
            """
            Calculate Expected Improvement.
            """
            imp = y_best - mu - xi
            sigma = np.sqrt(var)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            # For minimization, EI is defined with (y_best - mu)
            return ei

        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate candidate points for acquisition
            # We use a random sampling approach for the acquisition function
            # to avoid expensive grid search in 5D, or a coarse grid.
            # Given 5D, a random sampling of 500 candidates per step is reasonable.
            n_cands = 500
            X_cands = rng.uniform(-5, 5, (n_cands, dim))
            
            # Predict on candidates
            mu, var = gp_predict(X_cands, X, Y, L=1.0, sigma_f=1.0, sigma_n=1e-4)
            
            # Calculate EI
            ei = expected_improvement(mu, var, y_best, xi=0.01)
            
            # Select best candidate
            best_cand_idx = np.argmax(ei)
            x_next = X_cands[best_cand_idx]
            
            # Evaluate
            y_next = func(x_next)
            calls += 1
            
            # Update data
            X = np.vstack([X, x_next])
            Y = np.append(Y, y_next)
            
            if y_next < y_best:
                y_best = y_next
                x_best = x_next.copy()
                
            # Optional: Simple hyperparameter adaptation
            # For robustness and speed, we keep L and sigma_f fixed or adapt them slowly.
            # A fixed L=1.0 is often sufficient for these normalized BBOB functions.
            # If we had more time, we could maximize log-marginal likelihood.

        return x_best, y_best

# EVOLVE-BLOCK-END
