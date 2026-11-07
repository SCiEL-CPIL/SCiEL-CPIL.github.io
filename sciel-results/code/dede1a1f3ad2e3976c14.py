"""Derivative-Free Coordinate Descent with multi-resolution grid for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # Safety margin for 300s limit
        d = self.dim
        budget = self.budget

        best_x = np.zeros(d)
        best_f = np.inf
        evals_used = 0

        def time_left():
            return timeout - (time.time() - start_time)

        def evaluate(x):
            nonlocal best_x, best_f, evals_used
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid scan to identify promising basins
        # For Lunacek, the global minimum is at -4 in each dim (shifted by hidden rotation)
        # We do a coarse scan with 5 points per dimension
        coarse_vals = np.linspace(-5, 5, 5)  # [-5, -2.5, 0, 2.5, 5]
        
        # Instead of full grid (5^5=3125), we use coordinate descent on coarse grid
        current = np.zeros(d)
        for _ in range(3):  # 3 passes
            improved = False
            for i in range(d):
                best_val = None
                best_f_coord = np.inf
                for v in coarse_vals:
                    x_trial = current.copy()
                    x_trial[i] = v
                    f_trial = evaluate(x_trial)
                    if f_trial < best_f_coord:
                        best_f_coord = f_trial
                        best_val = v
                if best_val is not None and best_val != current[i]:
                    current[i] = best_val
                    improved = True
            if not improved:
                break
        
        # Phase 2: Multi-resolution refinement
        # Start with moderate resolution and refine
        resolutions = [9, 17, 33]  # points per dimension in refinement window
        
        for res in resolutions:
            if evals_used >= budget or time_left() < 5.0:
                break
            
            # For each dimension, refine around current best
            for i in range(d):
                if evals_used >= budget or time_left() < 2.0:
                    break
                
                # Determine search window around current value
                center = current[i]
                # Window size: start wide, narrow down
                if res == 9:
                    window = 2.5
                elif res == 17:
                    window = 1.2
                else:
                    window = 0.5
                
                # Ensure window stays within bounds
                lo = max(-5, center - window)
                hi = min(5, center + window)
                
                # Generate resolution points in this window
                vals = np.linspace(lo, hi, res)
                
                best_val = None
                best_f_coord = np.inf
                for v in vals:
                    x_trial = current.copy()
                    x_trial[i] = v
                    f_trial = evaluate(x_trial)
                    if f_trial < best_f_coord:
                        best_f_coord = f_trial
                        best_val = v
                
                if best_val is not None:
                    current[i] = best_val
        
        # Phase 3: Fine coordinate descent with adaptive step
        # Use Nelder-Mead-like refinement per coordinate
        step = 0.3
        min_step = 1e-4
        max_iter = 200
        
        for it in range(max_iter):
            if evals_used >= budget or time_left() < 2.0:
                break
            
            improved = False
            for i in range(d):
                if evals_used >= budget or time_left() < 1.0:
                    break
                
                # Try positive and negative steps
                x_plus = current.copy()
                x_plus[i] = min(5, current[i] + step)
                f_plus = evaluate(x_plus)
                
                x_minus = current.copy()
                x_minus[i] = max(-5, current[i] - step)
                f_minus = evaluate(x_minus)
                
                if f_plus < best_f - 1e-10:
                    current[i] = x_plus[i]
                    improved = True
                    step *= 1.2  # Expand step
                elif f_minus < best_f - 1e-10:
                    current[i] = x_minus[i]
                    improved = True
                    step *= 1.2  # Expand step
                else:
                    # No improvement, shrink step for this dimension
                    step = max(min_step, step * 0.5)
            
            if not improved and step < min_step:
                break
        
        # Phase 4: Final polish with very small steps
        if evals_used < budget and time_left() > 1.0:
            step = 0.05
            for it in range(100):
                if evals_used >= budget or time_left() < 0.5:
                    break
                improved = False
                for i in range(d):
                    if evals_used >= budget or time_left() < 0.5:
                        break
                    x_trial = current.copy()
                    # Random direction
                    direction = np.random.choice([-1, 1])
                    x_trial[i] += direction * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = evaluate(x_trial)
                    if f_trial < best_f - 1e-10:
                        current[i] = x_trial[i]
                        improved = True
                        step *= 1.1
                    else:
                        step *= 0.8
                        if step < 1e-5:
                            break
                if not improved and step < 1e-5:
                    break
            current = best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
