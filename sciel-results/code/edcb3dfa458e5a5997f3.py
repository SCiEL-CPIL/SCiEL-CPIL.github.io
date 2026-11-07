import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 300.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        # Store all evaluated points for surrogate model
        X_hist = []
        Y_hist = []

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64).flatten(), -5, 5)
            if x.shape != (dim,):
                return np.inf
            try:
                fx = func(x)
                calls_made += 1
                if np.isfinite(fx):
                    X_hist.append(x.copy())
                    Y_hist.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # 1. Initialization: Sample a few points to get a good starting mean
        n_init = 10
        init_points = rng.uniform(-5, 5, (n_init, dim))
        init_fvals = np.array([safe_eval(p) for p in init_points])

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Find the best initial point
        best_init_idx = np.argmin(init_fvals)
        mean = init_points[best_init_idx].copy()

        # 2. CMA-ES Phase
        # Allocate ~70-80% of budget to CMA-ES
        cma_budget_target = int(budget * 0.75)

        lam = 12  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial state
        sigma = 1.5  # Start with a moderate sigma
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Cache eigen decomposition for speed (dim is small)
        def sample_from_cov(mean, sigma, C, rng):
            # C is symmetric positive definite
            # Use eigh for stability
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-12, None)
            sqrt_val = np.sqrt(val)
            z = rng.standard_normal(dim)
            return mean + sigma * (vec @ (z * sqrt_val))

        while calls_made < cma_budget_target and time.time() < deadline:
            # Generate population
            pop = []
            fvals = []

            for _ in range(lam):
                if calls_made >= cma_budget_target or time.time() > deadline:
                    break
                x_new = sample_from_cov(mean, sigma, C, rng)
                x_new = np.clip(x_new, -5, 5)
                fx = safe_eval(x_new)
                pop.append(x_new)
                fvals.append(fx)

            if len(pop) < lam:
                # If we ran out of budget or time mid-generation, break
                break

            fvals = np.array(fvals)
            order = np.argsort(fvals)
            sorted_pop = np.array([pop[i] for i in order])
            sorted_fvals = fvals[order]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

            # Update evolution paths
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-12, None)
            inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

            diff = (new_mean - mean) / sigma
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            delta_sigma = (norm_ps / chi_n - 1) * d_sigma
            sigma *= np.exp(c_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C
            y = (sorted_pop[:mu] - mean) / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            decay = 1 - c_1 - c_mu
            rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
            C = decay * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            mean = new_mean

            # Stop if convergence is reached or sigma is tiny
            if sigma < 1e-8:
                break

        # 3. Surrogate-based Refinement Phase
        # Use the remaining budget for a local search around the best point found,
        # guided by a Gaussian Process surrogate.
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 5:
            try:
                # Prepare data for GP
                X = np.array(X_hist)
                Y = np.array(Y_hist)
                
                # Normalize Y for stability
                y_mean = np.mean(Y)
                y_std = np.std(Y) + 1e-8
                Y_norm = (Y - y_mean) / y_std

                # GP Kernel: Matern 5/2
                def kernel(X1, X2, l, sf):
                    # X1: (n1, dim), X2: (n2, dim)
                    dist2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                    dist2 = np.maximum(dist2, 0)
                    dist = np.sqrt(dist2)
                    return sf**2 * (1 + np.sqrt(5) * dist / l) * np.exp(-np.sqrt(5) * dist / l)

                # Hyperparameters (tuned for low-dim BBOB)
                l = 1.0  # Length scale
                sf = 1.0  # Signal variance
                noise = 1e-6  # Noise variance

                # Precompute K for historical data
                K = kernel(X, X, l, sf) + noise * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                    L_inv = np.linalg.inv(L)
                    alpha = L_inv.T @ (L_inv @ Y_norm)
                except np.linalg.LinAlgError:
                    # Fallback to least squares if Cholesky fails
                    K_reg = K + 1e-6 * np.eye(len(X))
                    try:
                        L = np.linalg.cholesky(K_reg)
                        L_inv = np.linalg.inv(L)
                        alpha = L_inv.T @ (L_inv @ Y_norm)
                    except np.linalg.LinAlgError:
                        alpha = np.linalg.lstsq(K_reg, Y_norm, rcond=None)[0]
                    L = None

                def gp_predict(X_new):
                    # X_new: (n, dim)
                    if len(X) == 0:
                        return np.zeros(len(X_new)), np.zeros(len(X_new))
                    
                    K_s = kernel(X_new, X, l, sf)  # (n, N)
                    if L is not None:
                        v = L_inv @ K_s.T  # (N, n)
                        mu = (K_s @ alpha)
                        var = sf**2 - np.sum(v**2, axis=0)
                    else:
                        # Fallback
                        mu = K_s @ alpha
                        var = sf**2  # Approximation
                    
                    var = np.maximum(var, 1e-8)
                    std = np.sqrt(var)
                    # Denormalize
                    mu = mu * y_std + y_mean
                    std = std * y_std
                    return mu, std

                def expected_improvement(X_new):
                    mu, std = gp_predict(X_new)
                    # Best value so far
                    f_best = best_f
                    # EI
                    z = (f_best - mu) / (std + 1e-8)
                    ei = (f_best - mu) * norm.cdf(z) - std * norm.pdf(z)
                    return ei

                # Local search around best_x using GP-EI
                # We will sample candidates in a neighborhood of best_x and pick the one with highest EI
                # Then evaluate it, update GP, repeat.
                
                # Define a neighborhood
                # Start with a moderate radius, shrink as we get closer
                radius = 0.5
                n_candidates = 50
                n_local_steps = min(remaining_budget, 100) # Max steps
                
                current_best_local = best_x.copy()
                
                for step in range(n_local_steps):
                    if calls_made >= budget or time.time() > deadline - 1:
                        break
                    
                    # Generate candidates around current_best_local
                    # Mix of random and best-so-far
                    candidates = []
                    
                    # 1. Random points in a box around current best
                    for _ in range(n_candidates // 2):
                        x_cand = current_best_local + radius * rng.standard_normal(dim)
                        x_cand = np.clip(x_cand, -5, 5)
                        candidates.append(x_cand)
                        
                    # 2. Points around the global best
                    for _ in range(n_candidates // 2):
                        x_cand = best_x + radius * rng.standard_normal(dim)
                        x_cand = np.clip(x_cand, -5, 5)
                        candidates.append(x_cand)
                        
                    candidates = np.array(candidates)
                    
                    # Compute EI for each candidate
                    ei_scores = expected_improvement(candidates)
                    
                    # Select candidate with highest EI
                    best_idx = np.argmax(ei_scores)
                    x_next = candidates[best_idx]
                    
                    # Evaluate
                    f_next = safe_eval(x_next)
                    
                    if np.isfinite(f_next) and f_next < best_f:
                        # Update current best local
                        current_best_local = x_next.copy()
                        # Shrink radius
                        radius *= 0.8
                    else:
                        # No improvement, shrink radius slightly
                        radius *= 0.95
                        
                    # Update GP data
                    # (Already updated in safe_eval)
                    # Re-compute GP internals? 
                    # For efficiency, we can just update X and Y and recompute L and alpha
                    # This is O(N^3) which might be slow if N is large.
                    # With 500 calls, N=500, 500^3 = 125M ops. Might be too slow.
                    # Let's limit the GP update frequency or use a simpler update.
                    # Or, just recompute every few steps.
                    
                    # Actually, recomputing Cholesky for N=500 in Python/NumPy is fast enough (~ms).
                    # Let's do it every step.
                    X = np.array(X_hist)
                    Y = np.array(Y_hist)
                    y_mean = np.mean(Y)
                    y_std = np.std(Y) + 1e-8
                    Y_norm = (Y - y_mean) / y_std
                    K = kernel(X, X, l, sf) + noise * np.eye(len(X))
                    try:
                        L = np.linalg.cholesky(K)
                        L_inv = np.linalg.inv(L)
                        alpha = L_inv.T @ (L_inv @ Y_norm)
                    except np.linalg.LinAlgError:
                        K_reg = K + 1e-6 * np.eye(len(X))
                        try:
                            L = np.linalg.cholesky(K_reg)
                            L_inv = np.linalg.inv(L)
                            alpha = L_inv.T @ (L_inv @ Y_norm)
                        except np.linalg.LinAlgError:
                            alpha = np.linalg.lstsq(K_reg, Y_norm, rcond=None)[0]
                            L = None

            except Exception:
                pass
        
        return best_x, best_f
