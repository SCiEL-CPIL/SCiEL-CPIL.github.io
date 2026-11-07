"""Optimized BBOB f12 (Bent Cigar) solver with CMA-ES + L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES global search followed by L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Wrap func to count calls and handle exceptions
        call_count = [0]

        def safe_func(x):
            try:
                x = np.asarray(x, dtype=float)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                x = np.clip(x, -5.0, 5.0)
                val = func(x)
                call_count[0] += 1
                if not np.isfinite(val):
                    return 1e10
                return float(val)
            except Exception:
                call_count[0] += 1
                return 1e10

        # --- Phase 1: CMA-ES for global exploration ---
        # Use scipy's CMA-ES via minimize with method='CMA' if available,
        # otherwise implement a simple CMA-ES manually.
        
        # We'll implement CMA-ES manually for control over budget and early stopping.
        
        rng = np.random.default_rng(42)
        
        # Initial point: try center of box
        x = np.zeros(dim)
        f_best = safe_func(x)
        best_x = x.copy()
        
        # CMA-ES parameters
        lambda_pop = 4 * dim  # population size
        sigma = 2.0  # initial step size
        c1 = (4 + np.log(dim)) / (dim**1.6 + 4 + 2 * np.log(dim))
        cmu = c1 * 2 / (c1 + dim)
        cs = (dim + 1.3) / (dim + 3)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        cc = (4 + dim / cs) / (4 + cs + dim)
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Eigen decomposition for sqrt
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_eigvals = np.sqrt(np.maximum(eigvals, 1e-10))
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        best_x = x.copy()
        best_f = f_best
        
        remaining = budget - call_count[0]
        
        while call_count[0] < budget:
            if time.time() > deadline - 10:
                break
            
            # Population size for this generation
            pop_size = min(lambda_pop, remaining)
            if pop_size < 2:
                break
            
            # Generate population
            archive = []
            for i in range(pop_size):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Update sqrt of C via eigen decomposition (every few generations for speed)
                if i == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 1e-10))
                    D = eigvecs @ np.diag(sqrt_eigvals)
                    D_inv = eigvecs @ np.diag(1.0 / np.maximum(sqrt_eigvals, 1e-10))
                
                z = rng.standard_normal(dim)
                y = x + sigma * D @ z
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                archive.append((fy, y))
            
            if not archive:
                break
            
            # Sort by fitness
            archive.sort(key=lambda t: t[0])
            
            # Update best
            if archive[0][0] < best_f:
                best_f = archive[0][0]
                best_x = archive[0][1].copy()
            
            # Compute mean of top half
            mu = pop_size // 2
            w = np.ones(mu)
            w_sum = w.sum()
            w /= w_sum
            mu_eff = 1.0 / np.sum(w**2)
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += w[i] * archive[i][1]
            new_mean = new_mean  # already weighted
            
            # Update paths
            d_mean = (new_mean - x) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * D_inv @ d_mean
            chi_n = np.sqrt(dim) * (1 - 1/(2*dim**2) + 1/(12*dim**4))
            hsig = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1-cs)**2) / chi_n) < (1.4 + 2/(dim+1)) * (1 - cs) else 0
            
            p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * d_mean
            
            # Update covariance
            delta_h = (1 - hsig) * cc * (1 - cc)
            C = (1 + c1 + cmu - c1) * C + c1 * (np.outer(p_c, p_c) - delta_h * C)
            for i in range(mu):
                y_i = D_inv @ (archive[i][1] - x) / sigma
                C += cmu * w[i] * (np.outer(y_i, y_i) - C)
            C = 0.5 * (C + C.T)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            x = new_mean
            
            remaining = budget - call_count[0]
            if remaining <= 0:
                break
        
        # --- Phase 2: L-BFGS-B local refinement ---
        remaining = budget - call_count[0]
        if remaining > 5:
            # Use L-BFGS-B from best_x
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': min(remaining // 2, 200), 'maxfun': remaining}
                )
                if result.fun < best_f and np.isfinite(result.fun):
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass
        
        # If we still have budget, do a few more random restarts + L-BFGS-B
        remaining = budget - call_count[0]
        if remaining > 20:
            n_restarts = min(remaining // 15, 3)
            for r in range(n_restarts):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                x0 = rng.uniform(-5, 5, dim)
                f0 = safe_func(x0)
                try:
                    result = minimize(
                        safe_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': min((budget - call_count[0]) // 2, 100),
                                  'maxfun': budget - call_count[0]}
                    )
                    if result.fun < best_f and np.isfinite(result.fun):
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
