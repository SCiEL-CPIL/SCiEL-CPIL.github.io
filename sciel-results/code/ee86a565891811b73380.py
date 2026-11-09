"""Rewritten optimizer for BBOB f20 (Schwefel) with hybrid global-local search."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Hybrid optimizer: Sobol global scan + Differential Evolution + L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                fx = np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sequence for global coverage
        # Generate points using a simple quasi-random approach
        # We'll use a stratified sampling approach for better coverage
        
        n_global = min(80, budget // 5)  # Use ~80 points for global scan
        
        # Generate stratified samples in [-5, 5]^dim
        # Divide each dimension into n_global^(1/dim) bins
        n_per_dim = max(2, int(round(n_global ** (1.0 / dim))))
        
        # Create grid of indices
        from itertools import product
        grid_indices = list(product(range(n_per_dim), repeat=dim))
        
        # Shuffle and take first n_global
        rng = np.random.default_rng(42)
        rng.shuffle(grid_indices)
        
        global_points = []
        for idx in grid_indices[:n_global]:
            point = np.zeros(dim)
            for d in range(dim):
                # Sample uniformly within the bin
                lo = -5.0 + 10.0 * idx[d] / n_per_dim
                hi = -5.0 + 10.0 * (idx[d] + 1) / n_per_dim
                point[d] = rng.uniform(lo, hi)
            global_points.append(point)
        
        # Evaluate global points
        for x in global_points:
            if time.time() > deadline or call_count >= budget:
                break
            safe_call(x)
        
        # Phase 2: Differential Evolution
        if time.time() < deadline and call_count < budget:
            # Use remaining budget for DE
            remaining = budget - call_count
            de_maxiter = max(1, remaining // max(dim * 10, 50))
            de_popsize = max(10, min(50, remaining // max(dim, 5)))
            
            # Set bounds
            bounds = [(-5.0, 5.0)] * dim
            
            # Use best point found so far to help DE
            # Create initial population around best_x
            popsize = de_popsize
            initial_pop = [best_x.copy() + rng.normal(0, 1.0, dim) for _ in range(popsize)]
            # Clip to bounds
            initial_pop = [np.clip(p, -5.0, 5.0) for p in initial_pop]
            
            try:
                de_result = differential_evolution(
                    lambda x: safe_call(x),
                    bounds=bounds,
                    maxiter=de_maxiter,
                    popsize=popsize,
                    tol=1e-10,
                    init=initial_pop,
                    seed=42,
                    polish=False  # We'll do our own local refinement
                )
                if de_result.success or de_result.fun < best_f:
                    if np.isfinite(de_result.fun):
                        if de_result.fun < best_f:
                            best_f = de_result.fun
                            best_x = de_result.x.copy()
            except Exception:
                pass
        
        # Phase 3: Local refinement with L-BFGS-B
        if time.time() < deadline and call_count < budget:
            # Try multiple local optimizations from best point and nearby points
            local_starts = [best_x.copy()]
            # Add a few perturbations
            for _ in range(3):
                perturbed = best_x + rng.normal(0, 0.5, dim)
                perturbed = np.clip(perturbed, -5.0, 5.0)
                local_starts.append(perturbed)
            
            for start_x in local_starts:
                if time.time() > deadline or call_count >= budget:
                    break
                try:
                    result = minimize(
                        lambda x: safe_call(x),
                        start_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.fun < best_f and np.isfinite(result.fun):
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        # Phase 4: Final Nelder-Mead polish if time/budget allows
        if time.time() < deadline and call_count < budget:
            try:
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_f and np.isfinite(result.fun):
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
