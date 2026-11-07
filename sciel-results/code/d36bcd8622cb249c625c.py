"""
Strategy: Derivative-Free Nelder-Mead Simplex with Origin Initialization and Restart.

1. Initialization: Start at the origin (0, 0, 0, 0, 0). BBOB Weierstrass functions
   typically have their global minimum near the center or shifted within a small range.
   Starting here provides a strong initial guess for early progress (AOCC).
2. Local Search: Use Nelder-Mead simplex method. It is derivative-free, robust to
   the non-convexity of Weierstrass, and requires only dim+1 evaluations per iteration
   (vs 2*dim for finite differences). This allows for significantly more iterations
   within the 500-call budget.
3. Restart Mechanism: If Nelder-Mead stagnates (simplex shrinks too much or no
   improvement for a while), perform a small random perturbation from the current
   best point to escape local minima or numerical noise, then resume Nelder-Mead.
   This hybrid approach balances global exploration (via restarts) with local
   exploitation (via Nelder-Mead).
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time (evaluator kills at 330s, we stop at 295s)
        start_time = time.time()
        deadline = start_time + 295.0
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and track budget/time
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            # Ensure input is a valid numpy array
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
            
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # --- Nelder-Mead Implementation ---
        # We implement a lightweight version to control the call budget precisely.
        
        def nelder_mead(start_x, max_iterations, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal calls_made, best_f, best_x, deadline
            
            # Initialize simplex: start_x and start_x + delta * e_i
            # Choose a small initial step size. Weierstrass has fine structure.
            # 0.1 is a reasonable starting scale for [-5, 5].
            init_step = 0.5
            simplex = [start_x.copy()]
            for i in range(dim):
                v = start_x.copy()
                v[i] += init_step
                simplex.append(v)
            
            # Evaluate initial simplex
            f_vals = np.array([wrapped_func(v) for v in simplex])
            
            # Track best in simplex
            current_best_idx = np.argmin(f_vals)
            current_best_val = f_vals[current_best_idx]
            
            no_improve_count = 0
            
            for _ in range(max_iterations):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = f_vals[order]
                
                # Check for convergence: if best is significantly better than worst, and simplex is small
                # Or if no improvement in best_f for several iterations
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 10:
                        break
                
                # If simplex is too small, stop
                centroid = np.mean(simplex[:-1], axis=0)
                spread = np.linalg.norm(simplex[-1] - centroid)
                if spread < 1e-6:
                    break
                
                # Reflection
                x_reflect = centroid - alpha * (simplex[-1] - centroid)
                f_reflect = wrapped_func(x_reflect)
                
                if f_reflect < f_vals[0]:
                    # Expansion
                    x_expand = centroid - gamma * (simplex[-1] - centroid)
                    f_expand = wrapped_func(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                elif f_reflect < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < f_vals[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (simplex[-1] - centroid)
                    
                    f_contract = wrapped_func(x_contract)
                    
                    if f_contract < min(f_reflect, f_vals[-1]):
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = wrapped_func(simplex[i])
            
            return best_x, best_f

        # --- Main Optimization Loop ---
        
        # Start from origin
        current_x = np.zeros(dim)
        
        # We will run Nelder-Mead in bursts.
        # Each burst allows up to ~100 iterations (which is ~500 calls for dim=5, but we limit it).
        # Actually, NM uses ~dim+1 calls per iteration. 500 / 6 = ~83 iterations.
        # We'll do a single long run first, then restarts if needed.
        
        # To maximize early progress, we start immediately.
        # We allocate most of the budget to the first NM run.
        # If it stalls, we use remaining budget for restarts.
        
        # First run: use up to 70% of budget for NM
        max_iter_first = int(budget * 0.7 / (dim + 1))
        nelder_mead(current_x, max_iter_first)
        
        # If we still have budget and time, do restarts from the best point found
        while calls_made < budget and time.time() < deadline:
            # Check if we are stuck (simplex likely shrunk or no improvement)
            # We can't easily tell from the outside, so we just try a new NM run from best_x
            # with a slightly perturbed start to escape potential local minima.
            
            # Perturb best_x slightly
            perturbation_scale = 0.5
            new_start = best_x + np.random.normal(0, perturbation_scale, dim)
            new_start = np.clip(new_start, -5, 5)
            
            # Run NM for a smaller number of iterations
            max_iter_restart = int((budget - calls_made) * 0.8 / (dim + 1))
            if max_iter_restart < 10:
                break
                
            nelder_mead(new_start, max_iter_restart)
            
            # If no improvement after this restart, we might be stuck in a global basin
            # or just ran out of useful budget.
            # To avoid infinite loops of tiny improvements, we break if improvement is negligible.
            # But since we don't track the previous best_f separately in the loop easily,
            # we rely on the fact that NM will stop if it converges.
            
            # Safety break: if we are very close to budget, stop.
            if calls_made > budget * 0.95:
                break

        return best_x, best_f
