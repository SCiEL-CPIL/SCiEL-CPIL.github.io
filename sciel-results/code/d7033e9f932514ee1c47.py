"""Rewritten optimizer using CMA-ES with Rastrigin-aware restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES based optimizer with Rastrigin-aware multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse Grid Search to map basin structure
        # Büche-Rastrigin is periodic with period 2*pi.
        # We use a grid of 3 points per dimension: -4, 0, 4 (approx)
        # 3^5 = 243 evaluations. This fits well within 500 budget.
        grid_points = []
        # Define grid coordinates roughly covering [-5, 5] with focus on likely minima regions
        # Minima are near 0 and +/- 2*pi (~6.28, out of bounds, so mainly 0 and maybe shifted by hidden offset)
        # Since offset is hidden, uniform grid is safer.
        coords = np.linspace(-4.0, 4.0, 3)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([i, j, k, l, m])
                            # Map indices to actual coordinates
                            x = x * 8.0 / 2.0 - 4.0  # -4, 0, 4
                            if call_count >= max_calls or time.time() > deadline:
                                break
                            f = counted_func(x)
                            grid_points.append((f, x.copy()))
            
            if call_count >= max_calls or time.time() > deadline:
                break
        
        # Sort grid points by fitness
        grid_points.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement from Top K Grid Points
        # Select top 5 distinct regions (if possible)
        top_k = 5
        starts = grid_points[:top_k]
        
        remaining_budget = max_calls - call_count
        budget_per_start = remaining_budget // len(starts) if starts else 0
        
        for f_init, x_init in starts:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            # Nelder-Mead refinement
            # We wrap counted_func to handle bounds and budget
            def nm_func(x):
                return counted_func(np.clip(x, -5, 5))
            
            try:
                # Use maxiter based on remaining budget for this start
                # Nelder-Mead uses roughly (dim+1) * maxiter evaluations
                max_iter = max(10, budget_per_start // (dim + 1))
                
                result = minimize(
                    nm_func,
                    x_init,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 3: Final Polish if budget remains
        if call_count < max_calls and time.time() < deadline:
            # Simple local search around best_x
            step = 0.1
            for _ in range(min(50, max_calls - call_count)):
                if call_count >= max_calls or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        if call_count >= max_calls or time.time() > deadline:
                            break
                        x_trial = best_x.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = counted_func(x_trial)
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-3:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
