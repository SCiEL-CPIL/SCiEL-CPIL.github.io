"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Gaussian Process Surrogate Optimizer for f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def check_time():
            return (time.time() - start_time) < time_limit

        # Global best tracking
        x_global_best = None
        f_global_best = np.inf
        calls = 0

        # Helper to evaluate and update global best
        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Data storage for GP
        X = []
        Y = []

        def add_to_gp(x, f):
            X.append(x)
            Y.append(f)

        # Phase 1: Initial random sampling to establish a baseline
        # We need enough points for the GP to have some structure, but not too many
        # that we waste budget without exploitation.
        n_init = 20
        for _ in range(n_init):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            add_to_gp(x, f)

        # GP Parameters
        # Kernel: Matern 5/2 is good for smooth functions with some roughness.
        # Or RBF (Squared Exponential) for very smooth. f22 is multimodal but smooth.
        # Let's use a combination or just Matern 5/2 with automatic lengthscale scaling.
        # For simplicity and speed in 5D, we'll use a simple isotropic RBF kernel
        # or Matern. Matern 5/2 is often better for BBOB.
        
        # We will implement a simple GP update loop.
        # To keep it fast, we might need to limit the number of points in the GP
        # or use a sparse approximation if n gets large. With 500 calls, n=500 is manageable
        # for 5D if we do it carefully. Inverting a 500x500 matrix is O(n^3) = 125M ops,
        # which is fine in NumPy if done a few times, but we do it every step.
        # 500 steps * 125M ops = 62.5B ops. This might be too slow in Python/NumPy loop.
        # We should probably limit the GP size to the best 100 points or so,
        # or use a faster update.
        
        # Strategy: Keep all points but if n > 100, subsample or use a sparse kernel?
        # Actually, for 5D, 500 points is small. Let's try keeping all points first.
        # If it's too slow, we can subsample.
        
        # Hyperparameters for GP
        # We will fix the hyperparameters to avoid optimization overhead.
        # Lengthscale: 1.0 (box is 10, so 1.0 is reasonable local scale)
        # Noise: 1e-6 (to ensure invertibility)
        # Variance: 1.0
        
        # Let's adapt the lengthscale slightly based on dimension?
        # Standard lengthscale for box [-5, 5] is often around 1-2.
        
        def gp_predict(X_train, Y_train, X_test, lengthscale=1.0, noise=1e-6, var=1.0):
            """
            Simple GP prediction with RBF kernel.
            X_train: (N, D)
            Y_train: (N,)
            X_test: (M, D)
            Returns: mean (M,), std (M,)
            """
            N = X_train.shape[0]
            D = X_train.shape[1]
            
            # Kernel matrix K (N, N)
            # RBF: exp(-0.5 * ||x_i - x_j||^2 / l^2)
            # Compute squared distances
            # (N, 1, D) - (1, N, D) -> (N, N, D)
            diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            K = var * np.exp(-0.5 * sq_dist / (lengthscale ** 2))
            
            # Add noise
            K[np.diag_indices_from(K)] += noise
            
            # Kernel vector k (N, M)
            diff_test = X_train[:, np.newaxis, :] - X_test[np.newaxis, :, :]
            sq_dist_test = np.sum(diff_test ** 2, axis=2)
            k = var * np.exp(-0.5 * sq_dist_test / (lengthscale ** 2))
            
            # Solve K alpha = Y
            try:
                alpha = np.linalg.solve(K, Y_train)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                alpha = np.linalg.pinv(K) @ Y_train
            
            # Mean: k^T alpha
            mean = k.T @ alpha
            
            # Variance: var - k^T K^-1 k
            # We already have alpha = K^-1 Y, but we need K^-1 k for each test point
            # This is expensive. 
            # Alternative: solve K v = k for each column of k?
            # Or use Cholesky.
            
            # Let's use Cholesky for stability and speed.
            try:
                L = np.linalg.cholesky(K)
                # alpha = L^-T L^-1 Y
                # Solve L v = Y
                v = np.linalg.solve(L, Y_train)
                # Solve L^T alpha = v
                alpha = np.linalg.solve(L.T, v)
                
                mean = k.T @ alpha
                
                # For variance:
                # v_k = L^-1 k (N, M)
                vk = np.linalg.solve(L, k)
                var_pred = var - np.sum(vk ** 2, axis=0)
                # Clip to 0 for numerical errors
                var_pred = np.maximum(var_pred, 0.0)
                std = np.sqrt(var_pred)
                
            except np.linalg.LinAlgError:
                # Fallback: Use identity or simple heuristic
                mean = np.mean(Y_train) * np.ones(X_test.shape[0])
                std = np.std(Y_train) * np.ones(X_test.shape[0])
                
            return mean, std

        # Phase 2: GP Optimization Loop
        # We will use Expected Improvement (EI) to select the next point.
        # EI(x) = (mu(x) - f_best - xi) * Phi(z) + sigma(x) * phi(z)
        # where z = (mu(x) - f_best - xi) / sigma(x)
        
        # To avoid evaluating EI on the entire domain (which is expensive in high D),
        # we will sample a set of candidate points and pick the one with highest EI.
        # Candidate generation:
        # 1. Random points
        # 2. Points around current best
        # 3. Points around other promising regions (local maxima of GP mean?)
        
        # For 5D, we can sample ~100-500 candidates per step.
        # 500 steps * 500 candidates * GP predict (N=500, M=500) ->
        # GP predict cost is O(N^2 M + N^3).
        # If N=500, N^3 is 125M. M=500, N^2 M is 125M.
        # Total ~250M ops per step. 500 steps * 250M = 125B ops.
        # This is likely too slow for 300s in Python.
        
        # Optimization:
        # 1. Limit GP size to best K points (e.g., K=50).
        # 2. Use fewer candidates (e.g., 100).
        # 3. Use a faster kernel or approximation.
        
        # Let's limit the GP to the best 100 points based on f value.
        # And use 200 candidates per step.
        
        max_gp_size = 100
        n_candidates = 200
        
        while calls < budget and check_time():
            # Update GP data if needed
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # If we have more points than max_gp_size, keep the best ones
            if len(Y) > max_gp_size:
                # Sort by f value (lower is better)
                idx = np.argsort(Y_arr)
                idx = idx[:max_gp_size]
                X_arr = X_arr[idx]
                Y_arr = Y_arr[idx]
            
            f_best = np.min(Y_arr)
            
            # Generate candidates
            # Mix of random and local perturbations around best
            cands = []
            
            # 1. Random
            n_rand = n_candidates // 2
            cands.append(rng.uniform(-5, 5, (n_rand, dim)))
            
            # 2. Around global best
            n_local = n_candidates // 4
            if x_global_best is not None:
                local_sigma = 0.5
                cands.append(rng.normal(x_global_best, local_sigma, (n_local, dim)))
            
            # 3. Around other good points (top 5)
            n_other = n_candidates // 4
            if len(Y_arr) > 5:
                top_idx = np.argsort(Y_arr)[:5]
                for i in top_idx:
                    n_per_point = max(1, n_other // 5)
                    local_sigma = 0.5
                    cands.append(rng.normal(X_arr[i], local_sigma, (n_per_point, dim)))
            
            # Concatenate and clip
            cands = np.vstack(cands)
            cands = np.clip(cands, -5, 5)
            
            # Predict mean and std for candidates
            mean, std = gp_predict(X_arr, Y_arr, cands)
            
            # Calculate EI
            # xi is exploration parameter, small value
            xi = 0.01
            imp = f_best - mean
            z = imp / std
            # Handle std=0
            mask = std > 1e-10
            ei = np.zeros_like(mean)
            ei[mask] = imp[mask] * norm.cdf(z[mask]) + std[mask] * norm.pdf(z[mask])
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_next = cands[best_idx]
            
            # Evaluate
            f_next = evaluate(x_next)
            add_to_gp(x_next, f_next)
            
            # Optional: If we are very close to the limit, stop
            if calls >= budget - 5:
                break

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
