"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    for the BBOB f22 (Gallagher 21 peaks) problem.
    
    Strategy:
    1. Stratified sampling to identify promising basins.
    2. Run CMA-ES from the best found point.
    3. If budget remains, perform a second CMA-ES restart with a perturbed mean 
       to escape potential local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        def cma_es(x0, sigma0, max_evals):
            """
            Simple CMA-ES implementation.
            """
            if not time_ok():
                return
            
            n = dim
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            
            # Weights
            weights = np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            # Initial state
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Update rates
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Damping for pc
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Cholesky decomposition helper
            def cholesky(mat):
                try:
                    return np.linalg.cholesky(mat)
                except np.linalg.LinAlgError:
                    # Regularize if not positive definite
                    return np.linalg.cholesky(mat + 1e-10 * np.eye(n))
            
            B = np.eye(n)
            D = np.ones(n)
            B_inv = np.eye(n)
            
            evals_used = 0
            
            while evals_used < max_evals and time_ok() and calls < budget:
                # Generate offspring
                Y = rng.standard_normal((lambda_, n))
                X = mean + sigma * (Y @ B.T) * D
                F = np.array([eval_and_track(x) for x in X])
                evals_used += lambda_
                
                if not time_ok() or calls >= budget:
                    break
                
                # Sort
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum(weights[:, None] * X[:mu], axis=0)
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * (np.sum(weights[:, None] * X[:mu], axis=0) - old_mean)))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                # Clip sigma to prevent explosion
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                # Update covariance matrix
                v = (np.sum(weights[:, None] * X[:mu], axis=0) - old_mean)
                C = (1 + c_1 + c_mu - c_1 * np.trace(C)) * C + \
                    c_1 * (pc @ pc.T - (1 - c_c) * (2 - c_c) * C) + \
                    c_mu / mueff * np.sum(weights[:, None, None] * (X[:mu] - old_mean)[:, None, :] @ (X[:mu] - old_mean)[:, None, :], axis=0)
                
                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * v / sigma
                
                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, 1e10)
                D = np.sqrt(eigvals)
                B = eigvecs
                B_inv = np.linalg.inv(B)
                
            return mean

        # Phase 1: Stratified Exploration
        # We need to find a good starting point for CMA-ES.
        # f22 has 21 peaks. We sample broadly.
        n_explore = min(100, budget // 5)
        initial_points = []
        
        # Stratified sampling: 2 bins per dim -> 32 strata
        n_strata = 2 ** dim
        points_per_stratum = max(1, n_explore // n_strata)
        
        for s in range(n_strata):
            bins = []
            temp = s
            for d in range(dim):
                bins.append(temp % 2)
                temp //= 2
            bins = np.array(bins)
            for _ in range(points_per_stratum):
                pt = np.zeros(dim)
                for d in range(dim):
                    if bins[d] == 0:
                        pt[d] = rng.uniform(-5.0, 0.0)
                    else:
                        pt[d] = rng.uniform(0.0, 5.0)
                initial_points.append(pt)
        
        # Add center and some structured points
        initial_points.append(np.zeros(dim))
        
        for x in initial_points:
            if not time_ok():
                break
            eval_and_track(x)
            
        if not time_ok():
            return best_x, best_f

        # Phase 2: CMA-ES Refinement
        # Start CMA-ES from the best point found so far.
        # Initial sigma should be large enough to explore the local basin.
        # A typical basin width in BBOB is around 0.5-1.0.
        sigma0 = 0.5
        max_evals_cma1 = min(300, budget - calls - 50) # Leave some budget for potential second run
        
        if max_evals_cma1 > 10:
            cma_es(best_x.copy(), sigma0, max_evals_cma1)
            
        if not time_ok():
            return best_x, best_f

        # Phase 3: Second CMA-ES Restart
        # If we have budget left, try to escape the current local minimum
        # by restarting CMA-ES with a slightly perturbed mean and larger sigma.
        if calls < budget - 20:
            # Perturb the mean
            perturbation = rng.normal(0, 0.5, size=dim)
            x0_restart = np.clip(best_x + perturbation, -5.0, 5.0)
            sigma_restart = 0.8
            max_evals_cma2 = budget - calls - 10
            
            if max_evals_cma2 > 10:
                cma_es(x0_restart, sigma_restart, max_evals_cma2)

        return best_x, best_f
# EVOLVE-BLOCK-END
