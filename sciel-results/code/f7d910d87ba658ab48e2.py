"""Combined optimizer: multi-start SLSQP with diversity-aware seeding and adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SLSQP with diversity-aware seeding and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            call_count[0] += 1
            return func(np.asarray(x, dtype=np.float64))

        def time_left():
            return (deadline - time.time()) > 5.0

        def budget_left():
            return (budget - call_count[0]) > 5

        def can_continue():
            return time_left() and budget_left()

        # Phase 1: Diversified random sampling to find good starting regions
        # Use structured + random points for better initial coverage
        n_initial = min(20, max(10, budget // 25))
        
        # Start with structured points (origin, axis-aligned)
        structured_points = [np.zeros(dim)]
        for i in range(dim):
            for val in [1.0, 2.0, -1.0, -2.0]:
                pt = np.zeros(dim)
                pt[i] = val
                structured_points.append(pt)
        
        # Add random points
        for i in range(n_initial):
            if not can_continue():
                break
            
            if i < len(structured_points):
                x0 = structured_points[i]
            else:
                x0 = rng.uniform(-5.0, 5.0, dim)
            
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Collect diverse seeds from initial exploration
        seeds = []
        for i in range(n_initial):
            if not can_continue():
                break
            if i < len(structured_points):
                x0 = structured_points[i]
            else:
                x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)
            seeds.append((x0.copy(), f0))
        
        # Sort by function value and keep top diverse seeds
        seeds.sort(key=lambda s: s[1])
        max_seeds = min(5, len(seeds))
        diverse_seeds = [seeds[0]]
        for s in seeds[1:]:
            if len(diverse_seeds) >= max_seeds:
                break
            is_diverse = True
            for ts in diverse_seeds:
                if np.linalg.norm(s[0] - ts[0]) < 2.5:
                    is_diverse = False
                    break
            if is_diverse:
                diverse_seeds.append(s)

        # Phase 2: Primary SLSQP from the best point
        remaining = budget - call_count[0] - 20
        if remaining > 10 and can_continue():
            max_iter = max(10, min(50, remaining // (dim + 1)))
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Multi-start local refinement with SLSQP from diverse seeds
        n_local_starts = min(len(diverse_seeds), max(3, (budget - call_count[0] - 20) // 40))
        
        for j in range(n_local_starts):
            if not can_continue():
                break
            
            # Use diverse seeds as starting points
            if j < len(diverse_seeds):
                x0 = diverse_seeds[j][0].copy()
            else:
                # Create perturbations around the best point with decreasing scale
                scale = 1.5 / (j + 1)
                x0 = best_x + rng.normal(0, scale, dim)
                x0 = np.clip(x0, -5.0, 5.0)
            
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = budget - call_count[0] - 15
            if remaining > 5 and can_continue():
                max_iter = max(5, min(30, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    # Fallback to Nelder-Mead if SLSQP fails
                    remaining = budget - call_count[0] - 10
                    if remaining > 5 and can_continue():
                        try:
                            res = minimize(
                                wrapped_func,
                                x0,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(50, remaining // 3),
                                    'xatol': 1e-8,
                                    'fatol': 1e-8,
                                    'adaptive': True,
                                    'disp': False
                                }
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                        except Exception:
                            pass

        # Phase 4: Multi-scale refinement around the best point
        if best_x is not None and can_continue():
            scales = [1.0, 0.5, 0.25, 0.1]
            for scale in scales:
                if not can_continue():
                    break
                
                # Generate perturbation around best point
                perturbation = rng.normal(0, scale, dim)
                start_point = np.clip(best_x + perturbation, -5.0, 5.0)
                
                remaining = budget - call_count[0] - 15
                if remaining > 5 and can_continue():
                    max_iter = max(5, min(25, remaining // (dim + 1)))
                    try:
                        res = minimize(
                            wrapped_func,
                            start_point,
                            method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 5: Global exploration with random restarts
        n_global = min(4, max(2, (budget - call_count[0] - 30) // 50))
        for j in range(n_global):
            if not can_continue():
                break
            
            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = budget - call_count[0] - 15
            if remaining > 10 and can_continue():
                max_iter = max(10, min(40, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 6: Final high-precision SLSQP polish
        if best_x is not None and can_continue():
            remaining = budget - call_count[0] - 5
            if remaining > 5:
                max_iter_final = max(10, min(100, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter_final, 'ftol': 1e-14, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    # Fallback to tight Nelder-Mead
                    remaining = budget - call_count[0] - 3
                    if remaining > 3 and can_continue():
                        try:
                            res = minimize(
                                wrapped_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(50, remaining // 2),
                                    'xatol': 1e-10,
                                    'fatol': 1e-10,
                                    'adaptive': True,
                                    'disp': False
                                }
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                        except Exception:
                            pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
