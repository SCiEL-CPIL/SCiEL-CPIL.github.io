"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling followed by Nelder-Mead refinement on top HV contributors."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Phase 1: Sobol sequence sampling
        n_sobol = 128
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_sobol)
        X_sobol = lo + sobol_samples * span
        X_sobol = np.clip(X_sobol, lo, hi)

        # Evaluate Sobol samples
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        # Combine into a single archive
        X_all = X_sobol.copy()
        F_all = F_sobol.copy()

        # Phase 2: Identify top hypervolume contributors and refine with Nelder-Mead
        # Reference point for HV calculation
        ref_point = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points dominated by ref (minimization)."""
            # Filter points strictly better than ref in every objective
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            max_f2 = -np.inf
            for i in range(len(pts)):
                f1, f2 = pts[i]
                if f2 < max_f2:
                    hv += (ref[0] - f1) * (max_f2 - f2) if max_f2 > f2 else 0.0
                else:
                    max_f2 = f2
                    hv += (ref[0] - f1) * (ref[1] - f2)
            return hv

        def hv_contribution(idx, points, ref):
            """Compute the hypervolume contribution of point idx to the set."""
            full_hv = hypervolume_2d(points, ref)
            remaining = np.delete(points, idx, axis=0)
            if len(remaining) == 0:
                return full_hv
            reduced_hv = hypervolume_2d(remaining, ref)
            return full_hv - reduced_hv

        # Find top 10 hypervolume contributors
        n_top = 10
        contributions = np.array([hv_contribution(i, F_all, ref_point) for i in range(len(F_all))])
        top_indices = np.argsort(contributions)[::-1][:n_top]
        
        # Ensure we have enough budget for Nelder-Mead
        remaining_budget = budget - used
        if remaining_budget > 0:
            # For each top contributor, run Nelder-Mead on a weighted sum of objectives
            # We use a few different weightings to explore the Pareto front
            weights = [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([2.0, 1.0]),
                np.array([1.0, 2.0]),
                np.array([3.0, 1.0]),
                np.array([1.0, 3.0]),
                np.array([5.0, 1.0]),
                np.array([1.0, 5.0]),
                np.array([2.0, 2.0]),
            ]
            
            evals_per_nm = max(1, remaining_budget // n_top)
            
            for i, idx in enumerate(top_indices):
                if used >= budget:
                    break
                x0 = F_all[idx]  # Use the objective values as a starting point? No, use the decision variable
                x0 = X_all[idx].copy()
                w = weights[i % len(weights)]
                
                def objective(x):
                    nonlocal used
                    if used >= budget:
                        return 0.0
                    fx = problem(x)
                    used += 1
                    return np.dot(fx, w)
                
                # Nelder-Mead options
                options = {
                    'maxiter': evals_per_nm,
                    'maxfev': evals_per_nm,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'disp': False
                }
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                    if result.fun is not None and used < budget:
                        x_best = np.clip(result.x, lo, hi)
                        f_best = problem(x_best)
                        used += 1
                        X_all = np.vstack([X_all, x_best])
                        F_all = np.vstack([F_all, f_best])
                except Exception:
                    pass

        return X_all
# EVOLVE-BLOCK-END
