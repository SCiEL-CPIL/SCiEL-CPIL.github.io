"""Combined optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses diverse initialization with Rosenbrock-valley seeds, a (1+1)-ES /
CMA-like local search phase, and a final high-precision polish pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
from math import exp, sqrt, log
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _init_points(self, rng, n):
        """Generate diverse initial points including Rosenbrock-valley seeds."""
        pts = []
        # Center
        pts.append(np.zeros(self.dim))
        
        # Rosenbrock valley seeds at various base levels
        # The Rosenbrock function has its minimum at (1, 1, ..., 1)
        # With shift/rotation, the valley is shifted, but we can try seeds near 1
        for base in [1.0, 0.5, 0.0, -0.5, -1.0]:
            if len(pts) >= n:
                break
            # Add some noise to the valley seed
            noise = 0.3 * rng.standard_normal(self.dim)
            pts.append(np.clip(np.full(self.dim, base) + noise, -5, 5))
        
        # Random uniform points
        while len(pts) < n:
            r = rng.random()
            if r < 0.4:
                pts.append(rng.uniform(-5, 5, self.dim))
            elif r < 0.7:
                pts.append(rng.uniform(-3, 3, self.dim))
            else:
                pts.append(rng.integers(-3, 4, self.dim).astype(float))
        
        return pts[:n]

    def _nelder_mead(self, x0, func, budget, deadline, rng, min_improve=1e-10):
        """Nelder-Mead simplex local refinement with tight tolerance."""
        n = self.dim
        # Initialize simplex
        simplex = [np.clip(x0, -5, 5).copy()]
        f_vals = [float(func(np.clip(x0, -5, 5)))]
        
        # Add perturbed vertices with smaller step for tighter convergence
        step = 0.1
        for i in range(n):
            v = simplex[0].copy()
            v[i] += step
            v = np.clip(v, -5, 5)
            simplex.append(v)
            f_vals.append(float(func(v)))
        
        # Sort by function value
        idx = np.argsort(f_vals)
        simplex = [simplex[i] for i in idx]
        f_vals = [f_vals[i] for i in idx]
        
        calls = 1 + n
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 300
        
        for it in range(max_iter):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = float(func(x_r))
            calls += 1
            
            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = float(func(x_e))
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = float(func(x_c))
                calls += 1
                
                if f_c < f_vals[-1]:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = float(func(simplex[i]))
                        calls += 1
                        if calls >= budget or time.time() >= deadline:
                            break
                    if calls >= budget or time.time() >= deadline:
                        break
            
            # Check convergence with tight tolerance
            if it > 10 and abs(f_vals[0] - f_vals[-1]) < min_improve:
                break
            
            # Re-sort
            idx = np.argsort(f_vals)
            simplex = [simplex[i] for i in idx]
            f_vals = [f_vals[i] for i in idx]
        
        # Return best point
        best_idx = np.argmin(f_vals)
        return simplex[best_idx], f_vals[best_idx], calls

    def _es_phase(self, x0, func, budget, deadline, rng):
        """(1+1)-ES phase with adaptive step size."""
        x = np.clip(x0, -5, 5).copy()
        f_best = float(func(x))
        calls = 1
        step = 1.0
        success = 0
        total = 0
        min_improve = 1e-12
        
        while calls < budget and time.time() < deadline:
            # Generate candidate
            delta = step * rng.standard_normal(self.dim)
            x_cand = np.clip(x + delta, -5, 5)
            f_cand = float(func(x_cand))
            calls += 1
            
            total += 1
            if f_cand < f_best - min_improve:
                x = x_cand
                f_best = f_cand
                success += 1
                # Increase step if successful
                if success > 0 and total > 10:
                    rate = success / total
                    if rate > 0.35:
                        step *= 1.1
                    elif rate < 0.15:
                        step *= 0.8
                success = max(0, success - 1)  # Decay success counter
            else:
                step *= 0.95
            
            # Reset success tracking periodically
            if total % 20 == 0:
                success = 0
                total = 0
            
            # Early stop if step is too small
            if step < 1e-8:
                break
        
        return x, f_best, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Phase 1: Diverse initial sampling
        n_init = 15
        init_pts = self._init_points(rng, n_init)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        for x in init_pts:
            if time.time() >= deadline:
                break
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: ES phase from best point
        es_budget = min(100, self.budget - calls - 50)
        if es_budget > 10 and best_x is not None:
            try:
                es_x, es_f, es_calls = self._es_phase(
                    best_x, func, es_budget, deadline - 10.0, rng
                )
                calls += es_calls
                if es_f < best_f:
                    best_f = es_f
                    best_x = es_x.copy()
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish from best point
        nm_budget = min(150, self.budget - calls - 10)
        if nm_budget > 10 and best_x is not None:
            try:
                nm_x, nm_f, nm_calls = self._nelder_mead(
                    best_x, func, nm_budget, deadline - 5.0, rng
                )
                calls += nm_calls
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x.copy()
            except Exception:
                pass
        
        # Phase 4: Final high-precision polish
        polish_budget = min(50, self.budget - calls)
        if polish_budget > 5 and best_x is not None and time.time() < deadline - 2.0:
            try:
                # Tighter NM with smaller initial step
                polish_x, polish_f, polish_calls = self._nelder_mead(
                    best_x, func, polish_budget, deadline, rng, min_improve=1e-14
                )
                calls += polish_calls
                if polish_f < best_f:
                    best_f = polish_f
                    best_x = polish_x.copy()
            except Exception:
                pass
        
        return best_x
# EVOLVE-BLOCK-END
