"""Optimizer for BBOB f6 (Attractive sector) using surrogate-based search with CMA-ES polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Surrogate-based optimizer for f6 with CMA-ES local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        max_calls = budget

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to build surrogate training data
        # Use Sobol-like or Latin Hypercube for better space-filling
        n_init = 50
        init_points = []
        
        # Include origin
        init_points.append(np.zeros(dim))
        
        # Use a simple space-filling approach: jittered grid or Latin Hypercube
        # For 5D, let's use a simple quasi-random sequence
        # We'll use a stratified sampling approach
        n_per_dim = max(2, int(round((n_init - 1) ** (1.0 / dim))))
        
        # Create a grid and then subsample
        grid = np.meshgrid(*[np.linspace(-4.5, 4.5, n_per_dim)] * dim)
        grid = np.stack([g.ravel() for g in grid], axis=1)
        
        # If grid is too large, subsample; if too small, add random points
        if len(grid) >= n_init - 1:
            # Subsample randomly
            indices = rng.choice(len(grid), n_init - 1, replace=False)
            init_points.extend(grid[indices].tolist())
        else:
            init_points.extend(grid.tolist())
            # Add random points to fill up to n_init
            while len(init_points) < n_init:
                init_points.append(rng.uniform(-4.5, 4.5, dim))
        
        init_points = np.array(init_points[:n_init])
        
        # Evaluate initial points
        X_train = []
        y_train = []
        for i in range(len(init_points)):
            if time_left() < 5.0 or calls >= max_calls:
                break
            f = eval_func(init_points[i])
            X_train.append(init_points[i])
            y_train.append(f)
        
        if calls >= max_calls or time_left() < 10.0:
            return best_x, best_f

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Phase 2: Surrogate-based optimization
        # Use RBF (Radial Basis Function) surrogate with Expected Improvement
        
        def rbf_kernel(X1, X2, length_scale=1.0):
            """Compute RBF kernel matrix."""
            # X1: (n1, d), X2: (n2, d)
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-sq_dists / (2 * length_scale**2))
        
        def train_rbf(X, y, length_scale=1.0, noise=1e-6):
            """Train RBF surrogate. Returns kernel matrix and weights."""
            K = rbf_kernel(X, X, length_scale)
            K += noise * np.eye(len(X))
            try:
                weights = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(K, y, rcond=None)[0]
            return K, weights
        
        def predict_rbf(X_new, X_train, weights, y_mean, y_std, length_scale=1.0):
            """Predict mean and variance using RBF."""
            K_new = rbf_kernel(X_new, X_train, length_scale)
            mean = K_new @ weights + y_mean
            # Approximate variance
            K_train = rbf_kernel(X_train, X_train, length_scale)
            K_train += 1e-6 * np.eye(len(X_train))
            try:
                alpha = np.linalg.solve(K_train, y_train - y_mean)
                var = 1.0 - K_new @ K_train @ alpha
                var = np.maximum(var, 0.0)
            except:
                var = np.ones(len(X_new)) * y_std**2
            return mean, np.sqrt(var)
        
        def expected_improvement(mean, std, best_f, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            if std < 1e-10:
                return np.zeros_like(mean)
            z = (best_f - mean - xi) / std
            ei = (best_f - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei
        
        # Train initial surrogate
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) if len(y_train) > 1 else 1.0
        length_scale = 1.5  # Initial length scale
        
        K_train, weights = train_rbf(X_train, y_train, length_scale)
        
        # Surrogate optimization loop
        n_surrogate = 0
        max_surrogate = max_calls - calls - 30  # Reserve 30 calls for CMA-ES polish
        stagnant = 0
        best_f_prev = best_f
        
        while n_surrogate < max_surrogate and time_left() > 10.0:
            # Generate candidate points: mix of random and local around best
            n_candidates = 100
            candidates = np.zeros((n_candidates, dim))
            
            # 50% random uniform
            n_random = n_candidates // 2
            candidates[:n_random] = rng.uniform(-4.5, 4.5, (n_random, dim))
            
            # 30% local around current best
            n_local = n_candidates // 3
            local_radius = max(0.5, 2.0 * (1 - n_surrogate / max(1, max_surrogate)))
            candidates[n_random:n_random + n_local] = best_x + rng.standard_normal((n_local, dim)) * local_radius
            candidates[n_random:n_random + n_local] = np.clip(candidates[n_random:n_random + n_local], -4.5, 4.5)
            
            # 20% around second best (if available)
            if len(y_train) >= 2:
                second_best_idx = np.argsort(y_train)[1]
                second_best_x = X_train[second_best_idx]
                n_second = n_candidates - n_random - n_local
                candidates[n_random + n_local:] = second_best_x + rng.standard_normal((n_second, dim)) * local_radius * 1.5
                candidates[n_random + n_local:] = np.clip(candidates[n_random + n_local:], -4.5, 4.5)
            
            # Predict mean and variance for candidates
            mean_pred, std_pred = predict_rbf(candidates, X_train, weights, y_mean, y_std, length_scale)
            
            # Compute EI
            ei = expected_improvement(mean_pred, std_pred, best_f, xi=0.01)
            
            # Select top candidates
            top_k = 5
            top_indices = np.argsort(ei)[-top_k:]
            
            # Evaluate the best candidate
            best_candidate_idx = top_indices[-1]
            x_new = candidates[best_candidate_idx]
            f_new = eval_func(x_new)
            n_surrogate += 1
            
            # Update training data
            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, f_new)
            
            # Retrain surrogate periodically
            if n_surrogate % 10 == 0:
                y_mean = np.mean(y_train)
                y_std = np.std(y_train) if len(y_train) > 1 else 1.0
                K_train, weights = train_rbf(X_train, y_train, length_scale)
            
            # Check for stagnation
            if best_f < best_f_prev - 1e-8:
                stagnant = 0
                best_f_prev = best_f
            else:
                stagnant += 1
            
            # If stagnant for too long, try a random restart
            if stagnant > 15:
                x_restart = rng.uniform(-4.0, 4.0, dim)
                f_restart = eval_func(x_restart)
                n_surrogate += 1
                X_train = np.vstack([X_train, x_restart])
                y_train = np.append(y_train, f_restart)
                stagnant = 0
                best_f_prev = best_f
        
        # Phase 3: CMA-ES local polish on the best point found
        remaining_budget = max_calls - calls
        if remaining_budget > 20 and time_left() > 10.0:
            def cmaes(x0, sigma0, max_evals):
                nonlocal calls, best_f, best_x
                
                if max_evals < 20:
                    return
                
                n = dim
                lambda_pop = 4 + int(3 * np.log(n))
                mu = lambda_pop // 2
                weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights_cma /= np.sum(weights_cma)
                mueff = 1.0 / np.sum(weights_cma ** 2)
                
                cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                cs = (mueff + 2) / (n + mueff + 5)
                c1 = 2 / ((n + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
                
                x_mean = x0.copy()
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = sigma0
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                
                def compute_sqrt_C(C):
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    return eigvecs @ np.diag(np.sqrt(eigvals))
                
                sqrt_C = compute_sqrt_C(C)
                
                n_evals = 0
                stagnant_generations = 0
                best_in_cma = best_f
                best_x_in_cma = best_x.copy()
                
                while n_evals < max_evals and time_left() > 5.0:
                    z = rng.standard_normal((lambda_pop, n))
                    offsprings = x_mean + sigma * (z @ sqrt_C.T)
                    
                    fitness = np.zeros(lambda_pop)
                    for i in range(lambda_pop):
                        if n_evals >= max_evals or time_left() < 5.0:
                            break
                        fitness[i] = eval_func(offsprings[i])
                        n_evals += 1
                    
                    indices = np.argsort(fitness)
                    x_sorted = offsprings[indices]
                    f_sorted = fitness[indices]
                    
                    if f_sorted[0] < best_f:
                        best_f = f_sorted[0]
                        best_x = x_sorted[0].copy()
                    
                    if f_sorted[0] < best_in_cma:
                        best_in_cma = f_sorted[0]
                        best_x_in_cma = x_sorted[0].copy()
                        stagnant_generations = 0
                    else:
                        stagnant_generations += 1
                    
                    x_mean_new = x_mean + sigma * sum(weights_cma[i] * x_sorted[i] for i in range(mu))
                    
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * sqrt_C.T @ (x_mean_new - x_mean)
                    if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_evals + 1) / lambda_pop)) < (c1 + 1) * chi_n + 0.3:
                        pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                    
                    diff = x_sorted[:mu] - x_mean
                    C = (1 + c1 * cmu - c1) * C + c1 * np.outer(pc, pc) + cmu * (diff * weights_cma[:, None]).T @ diff
                    
                    C = (C + C.T) / 2
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C += 1e-10 * np.eye(n)
                    
                    sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 1e10)
                    
                    x_mean = x_mean_new
                    sqrt_C = compute_sqrt_C(C)
                    
                    if stagnant_generations > 15:
                        break
                    if sigma < 1e-12:
                        break
                
                return best_x_in_cma, best_in_cma
            
            cma_budget = remaining_budget
            if cma_budget > 30:
                x_start = best_x.copy()
                sigma_start = 0.5
                cmaes(x_start, sigma_start, cma_budget)
        
        return best_x, best_f
