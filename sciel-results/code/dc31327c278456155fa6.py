"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Coordinate Descent with adaptive step size.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Coordinate Descent optimizer with adaptive step size and random restarts.
    Designed for Lunacek bi-Rastrigin (f24) which has separable local structures.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = func(x)
                if not np.isfinite(val):
                    return 1e10
                return float(val)
            except:
                return 1e10

        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start initialization to find promising basins
        # Lunacek has a global min and a local min. We sample diverse points.
        start_points = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            # Specific heuristic: Lunacek local min is often around 2.5*sqrt(dim) offset from global?
            # Actually, the shift is hidden. Just use randoms.
            rng.uniform(-5, 5, dim)
        ]

        for sp in start_points:
            if call_count >= budget - 10 or time.time() > deadline:
                break
            sp = np.clip(sp, -5.0, 5.0)
            fx = safe_func(sp)
            if fx < best_f:
                best_f = fx
                best_x = sp.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # Phase 2: Coordinate Descent with Adaptive Step Size
        # We use a global step size that decays when no improvement is found.
        # We also perform random perturbations to escape local minima.
        
        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.0
        min_step = 1e-6
        no_improve_count = 0
        max_no_improve = 10 # Decay step size after this many cycles without improvement
        perturbation_freq = 20 # Every N cycles, try a random perturbation

        while call_count < budget - 5 and time.time() < deadline:
            improved_in_cycle = False
            
            # Cyclic coordinate descent
            for i in range(dim):
                if call_count >= budget - 5 or time.time() > deadline:
                    break
                
                # Try positive step
                x_pos = current_x.copy()
                x_pos[i] += step_size
                x_pos = np.clip(x_pos, -5.0, 5.0)
                f_pos = safe_func(x_pos)
                
                if f_pos < current_f:
                    current_x = x_pos
                    current_f = f_pos
                    improved_in_cycle = True
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                    # Optional: increase step size slightly on success? 
                    # Usually safer to keep it or increase very slightly.
                    # step_size = min(step_size * 1.1, 5.0) 
                    continue
                
                # Try negative step
                x_neg = current_x.copy()
                x_neg[i] -= step_size
                x_neg = np.clip(x_neg, -5.0, 5.0)
                f_neg = safe_func(x_neg)
                
                if f_neg < current_f:
                    current_x = x_neg
                    current_f = f_neg
                    improved_in_cycle = True
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                    continue

            # End of one full cycle through dimensions
            if not improved_in_cycle:
                no_improve_count += 1
                # Decay step size
                step_size *= 0.7
                
                # If step size is very small, we might be stuck in a local minimum
                # or converged. Try a random perturbation.
                if no_improve_count >= max_no_improve or step_size < min_step:
                    # Random perturbation
                    perturbation_magnitude = max(step_size * 2, 0.5)
                    x_pert = current_x + rng.normal(0, perturbation_magnitude, dim)
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    f_pert = safe_func(x_pert)
                    
                    if f_pert < current_f:
                        current_x = x_pert
                        current_f = f_pert
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                        
                    # Reset counters and step size for a fresh attempt from perturbed point
                    step_size = max(0.5, min(1.0, perturbation_magnitude))
                    no_improve_count = 0
                    max_no_improve = 10
                    perturbation_freq += 10 # Increase perturbation frequency over time? Or keep constant.
                    
                    # If we are still stuck and step size is tiny, break
                    if step_size < min_step and no_improve_count > 5:
                        break
            else:
                no_improve_count = 0

            # Safety check: if step size is extremely small, we are likely converged
            if step_size < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
