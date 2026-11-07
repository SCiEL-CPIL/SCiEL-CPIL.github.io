"""Multi-start L-BFGS-B with Sobol sampling and adaptive local search for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start L-BFGS-B with Sobol sampling for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()

        rng = np.random.default_rng(42)

        # Phase 1: Generate Sobol points for diverse initial sampling
        # We use a smaller initial set to leave more budget for L-BFGS-B restarts
        n_sobol = 32
        sobol_engine = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sobol_engine.random(n_sobol)
        sobol_samples = sobol_samples * 10.0 - 5.0

        # Evaluate initial Sobol points
        start_points = []
        for pt in sobol_samples:
            if eval_count >= budget - 100:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
            start_points.append(pt.copy())

        if not start_points:
            f_val = wrapped_func(np.zeros(dim))
            update_best(np.zeros(dim), f_val)
            start_points = [np.zeros(dim)]

        # Phase 2: Multi-start L-BFGS-B
        # Strategy:
        # 1. Sort initial points by fitness.
        # 2. Run L-BFGS-B from the best few points.
        # 3. If no improvement, generate new random points or perturbations and repeat.
        
        # Sort by fitness
        start_points.sort(key=lambda p: wrapped_func(p) if p not in [pt for pt in start_points if False] else best_f) 
        # Note: re-evaluating is bad. We should have stored values.
        # Let's redo Phase 1 to store values properly.
        
        # Restart Phase 1 logic to store values
        start_points = []
        start_values = []
        for pt in sobol_samples:
            if eval_count >= budget - 100:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
            start_points.append(pt.copy())
            start_values.append(f_val)
            
        # Sort indices by value
        sorted_indices = np.argsort(start_values)
        
        # We will maintain a list of "promising" start points.
        # Initially, the top Sobol points.
        # We also add random points and perturbations dynamically.
        
        # To avoid re-evaluating, we track which points have been used as starts for L-BFGS-B.
        # But L-BFGS-B itself consumes budget.
        
        # Strategy:
        # 1. Take the best Sobol point. Run L-BFGS-B.
        # 2. Take the next best Sobol point (if not too close to best). Run L-BFGS-B.
        # 3. If budget remains, generate new random points, evaluate, and if good, run L-BFGS-B.
        
        used_starts = set()
        
        # Helper to check if a point is "close" to any used start
        def is_close(x, threshold=0.5):
            for us in used_starts:
                if np.linalg.norm(x - us) < threshold:
                    return True
            return False

        # Process initial Sobol points
        for idx in sorted_indices:
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            sp = start_points[idx]
            if is_close(sp):
                continue
            
            used_starts.add(sp.copy())
            
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Adaptive exploration
        # While budget remains, pick a new point (random or perturbation) and refine.
        # We bias towards exploration if we haven't improved recently, or exploitation if we have.
        
        last_improve_count = eval_count
        no_improve_window = 0
        
        while eval_count < budget - 20 and time.time() < deadline:
            # Decide next start point
            if no_improve_window > 10:
                # Explore: Random point in box
                sp = rng.uniform(-5.0, 5.0, size=dim)
                no_improve_window = 0
            else:
                # Exploit: Perturb current best or pick a random point near best
                # Mix of small and medium perturbations
                scale = rng.choice([0.5, 1.0, 2.0])
                sp = best_x + rng.normal(0, scale, size=dim)
                sp = np.clip(sp, -5.0, 5.0)
            
            if is_close(sp, threshold=0.2):
                # If too close, try again or just skip to save time? 
                # Better to just run a very short local search or skip.
                # Let's just skip to avoid redundant L-BFGS-B runs.
                continue

            used_starts.add(sp.copy())
            
            # Evaluate the start point first to see if it's promising?
            # L-BFGS-B will do this anyway. But if the start point is bad, L-BFGS-B might waste budget.
            # However, L-BFGS-B is cheap if it converges quickly or fails quickly.
            
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 10, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    no_improve_window = 0
                else:
                    no_improve_window += 1
            except Exception:
                no_improve_window += 1

        # Phase 4: Final polish with Nelder-Mead or small step random search
        # L-BFGS-B can get stuck in flat regions. A small random walk can help.
        step_size = 0.1
        for _ in range(20):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            
            x = best_x + rng.normal(0, step_size, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            else:
                step_size *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
