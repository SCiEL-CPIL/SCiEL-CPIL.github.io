"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


class Optimizer:
    """Hybrid optimizer: L-BFGS-B from multiple starts + adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Evaluate a diverse set of initial points directly (fast, improves early AOCC)
        # Structured points: origin, axis-aligned, mid-box, corners
        initial_points = [np.zeros(self.dim)]
        
        # Axis-aligned points: ±2, ±4 along each axis
        for i in range(self.dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(self.dim)
                p[i] = val
                initial_points.append(p)
        
        # Mid-box points
        for val in [-2.5, 0.0, 2.5]:
            initial_points.append(np.full(self.dim, val))
        
        # Random points
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, self.dim))
        
        # Evaluate all initial points directly (cheap, gives good early progress)
        for p in initial_points:
            if not time_ok() or call_count >= self.budget - 100:
                break
            fx = tracked_func(p)
        
        # Phase 2: L-BFGS-B from the top 8 best initial points
        # Sort initial points by their function values
        evaluated = []
        for p in initial_points:
            if call_count >= self.budget - 100:
                break
            # We already evaluated these; use cached best from tracked_func
            # Instead, re-evaluate only the top candidates via L-BFGS-B
            pass
        
        # Select starting points for L-BFGS-B: best 8 from initial evaluation
        # We need to track which points gave which values
        # Re-do: evaluate and store
        point_vals = []
        for p in initial_points[:20]:  # Limit to first 20 to save calls
            if not time_ok() or call_count >= self.budget - 150:
                break
            fx = tracked_func(p)
            point_vals.append((fx, p))
        
        point_vals.sort(key=lambda x: x[0])
        
        # L-BFGS-B from top 8 starting points
        maxfun_per_start = max(20, (self.budget - call_count - 50) // 8)
        for i, (fv, x0) in enumerate(point_vals[:8]):
            if not time_ok() or call_count >= self.budget - 30:
                break
            
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxls': 30, 'maxfun': maxfun_per_start}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            if not time_ok() or call_count >= self.budget - 30:
                break

        # Phase 3: Adaptive (1+1)-ES with restarts
        if time_ok() and call_count < self.budget - 10:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.3
            stuck = 0
            best_sigma = sigma
            last_improvement = 0
            restarts = 0
            max_restarts = 5

            remaining_budget = self.budget - call_count - 5
            
            while call_count < self.budget - 5 and time_ok():
                if stuck > 30 + 10 * restarts:
                    # Restart: jump to a new random point near the best
                    if restarts >= max_restarts:
                        # Final restart: try a few random global jumps
                        for _ in range(min(5, self.budget - call_count - 5)):
                            if not time_ok() or call_count >= self.budget - 5:
                                break
                            y = np.clip(best_x + 2.0 * rng.standard_normal(self.dim), -5, 5)
                            fy = tracked_func(y)
                            if fy < fx:
                                x, fx = y, fy
                                sigma = 0.5
                                stuck = 0
                                restarts = 0
                        break
                    
                    restarts += 1
                    # Try a moderately perturbed version of the best
                    x = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.5
                    stuck = 0
                    continue
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    best_sigma = sigma
                    stuck = 0
                    last_improvement = call_count
                else:
                    sigma *= 0.95
                    stuck += 1

                # Periodically check if we should restart based on no improvement
                if stuck > 20 and call_count - last_improvement > 40:
                    stuck = 30 + 10 * restarts  # Force restart condition

        # Phase 4: Final high-precision polish with tight bounds
        if time_ok() and call_count < self.budget - 5:
            try:
                bounds = []
                for i in range(self.dim):
                    lo = max(-5.0, best_x[i] - 0.5)
                    hi = min(5.0, best_x[i] + 0.5)
                    bounds.append((lo, hi))

                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxls': 50, 'maxfun': min(100, self.budget - call_count - 3),
                             'gtol': 1e-10}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
