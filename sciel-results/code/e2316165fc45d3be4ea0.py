import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Leave 20s margin for 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f_val = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Broad Exploration with Sobol Sequence
        n_explore = min(100, budget)
        start_points = []
        if n_explore > 0:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            n_sobol = 128
            sobol_points = sampler.random(n_sobol)
            sobol_points = sobol_points * 10.0 - 5.0
            
            sobol_values = []
            for i in range(n_explore):
                if evals_used >= budget or time.time() > deadline:
                    break
                x = sobol_points[i]
                f_val = try_func(x)
                sobol_values.append((f_val, x))
            
            if sobol_values:
                sobol_values.sort(key=lambda item: item[0])
                start_points = [item[1] for item in sobol_values[:5]]
            else:
                start_points = [best_x]

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES Optimization
        # Implementing a compact CMA-ES
        
        def run_cma_es(start_point, max_evals):
            nonlocal evals_used, best_f, best_x
            
            # CMA-ES Parameters
            n = dim
            lam = 10  # Population size
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            w = np.concatenate([w, np.zeros(lam - mu)])
            
            mueff = 1.0 / np.sum(w**2)
            
            # Strategy parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - mueff / (n + 2) + c_c) / ((n + 1.3) ** 2 + mueff + c_c)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Initialization
            x_mean = start_point.copy()
            sigma = 1.0  # Initial step size
            B = np.eye(n)  # Rotation matrix
            D = np.ones(n)  # Eigenvalues
            C = np.eye(n)  # Covariance matrix
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Precompute
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Main loop
            while evals_used < budget and time.time() < deadline:
                if max_evals <= 0:
                    break
                
                # Generate population
                ps = np.zeros((lam, n))
                xs = np.zeros((lam, n))
                fs = np.zeros(lam)
                
                for i in range(lam):
                    # Check budget
                    if evals_used >= budget or time.time() > deadline:
                        return
                    
                    # Generate random vector
                    z = np.random.standard_normal(n)
                    ps[i] = z
                    xs[i] = x_mean + sigma * (B @ (D * z))
                    xs[i] = np.clip(xs[i], -5.0, 5.0)
                    
                    # Evaluate
                    fs[i] = try_func(xs[i])
                    max_evals -= 1
                
                # Sort by fitness
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                ps = ps[order]
                
                # Update mean
                y_w = np.sum(w[:mu, None] * ps[:mu], axis=0)
                x_mean = x_mean + sigma * (B @ (D * y_w))
                x_mean = np.clip(x_mean, -5.0, 5.0)
                
                # Update step size
                y_sigma = B.T @ (D * y_w)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                y_c = B.T @ y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c
                
                delta_heaviside = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (max_evals - evals_used + 1) / (n + 1))) < (1 + 2 / (n + 1)) * chi_n else 0.0
                
                C = (1 + c_1 * delta_heaviside - c_1 - c_mu) * C + \
                    c_1 * np.outer(p_c, p_c) + \
                    c_mu * np.sum(w[:mu, None, None] * np.outer(ps[:mu], ps[:mu]), axis=0)
                
                # Update B and D (Eigenvalue decomposition)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                
                # Termination criteria
                if sigma < 1e-12 or (fs[0] == fs[1]):
                    break
                if np.max(np.abs(C - np.eye(n))) < 1e-12:
                    break

        # Try CMA-ES from the best Sobol point
        if start_points:
            remaining_budget = budget - evals_used
            run_cma_es(start_points[0], remaining_budget)
        
        # If we have budget left and didn't converge well, try a second start point
        if len(start_points) > 1 and evals_used < budget and time.time() < deadline:
            remaining_budget = budget - evals_used
            run_cma_es(start_points[1], remaining_budget)

        return best_x, best_f
