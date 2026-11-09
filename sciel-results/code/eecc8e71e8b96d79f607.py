"""Optimizer for BBOB-style f13 (Sharp ridge) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with Random Restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        n = self.dim
        total_budget = self.budget
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def nelder_mead(x_start, max_evals):
            nonlocal calls_made, best_x, best_f, deadline
            
            # Initialize simplex
            # Vertices: x_start, x_start + delta * e_i
            delta = 0.5 
            vertices = [x_start.copy()]
            for i in range(n):
                v = x_start.copy()
                v[i] += delta
                vertices.append(v)
            
            # Evaluate initial simplex
            f_values = np.array([safe_eval(v) for v in vertices])
            
            # If we ran out of budget, return
            if calls_made >= total_budget or time.time() > deadline:
                return

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            while calls_made < total_budget and time.time() < deadline:
                # Sort vertices by function value
                order = np.argsort(f_values)
                vertices = [vertices[i] for i in order]
                f_values = f_values[order]
                
                # Check for convergence (simple: if range is small)
                if f_values[-1] - f_values[0] < 1e-10:
                    break

                x_centroid = np.mean(vertices[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - vertices[-1])
                f_ref = safe_eval(x_ref)
                
                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = safe_eval(x_exp)
                    if f_exp < f_ref:
                        vertices[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        vertices[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    vertices[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho * (vertices[-1] - x_centroid)
                    f_con = safe_eval(x_con)
                    if f_con < f_values[-1]:
                        vertices[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            vertices[i] = vertices[0] + sigma * (vertices[i] - vertices[0])
                            f_values[i] = safe_eval(vertices[i])
                            if calls_made >= total_budget or time.time() > deadline:
                                return

        # Strategy: Multiple restarts
        # We allocate budget dynamically. 
        # Start with a few random points to find a good basin, then refine.
        
        # Phase 1: Random sampling to find good starting points
        n_initial = min(20, total_budget // 10)
        if n_initial > 0:
            for _ in range(n_initial):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-5, 5, n)
                safe_eval(x_rand)
        
        # Phase 2: Local optimization from best points found
        # We will run Nelder-Mead on the best current point.
        # Then restart from a perturbation of the best point if time/budget allows.
        
        # Estimate remaining budget for NM
        # We'll run NM as long as we have budget and time.
        # To handle the "Sharp Ridge", we might need multiple attempts.
        
        # Attempt 1: From best_x found in Phase 1
        if best_x is not None:
            x_start = best_x.copy()
            # Give NM a generous budget, but check limits inside
            nelder_mead(x_start, total_budget)
        
        # If we still have budget, do a few more restarts from perturbations
        # This helps if the first NM got stuck.
        max_restarts = 3
        for i in range(max_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            # Perturb best_x
            if best_x is not None:
                # Scale perturbation based on remaining budget
                scale = 1.0 / (i + 2)
                x_start = best_x + scale * rng.standard_normal(n)
            else:
                x_start = rng.uniform(-5, 5, n)
            
            x_start = np.clip(x_start, -5, 5)
            nelder_mead(x_start, total_budget)
            
            # If no improvement in the last restart, we might be done or need a bigger jump
            # But for simplicity, we just try a few.

        return best_x, best_f
# EVOLVE-BLOCK-END
