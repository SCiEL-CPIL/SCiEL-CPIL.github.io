"""Model-and-Optimize Optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a multi-start Adaptive Simulated Annealing (ASA) strategy
    combined with local refinement.
    1. Multi-start ASA for global exploration and basin identification.
    2. Intensive Powell's method on the best candidates.
    3. Final polish with Nelder-Mead.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 20s safety margin
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        # --- Phase 1: Multi-start Adaptive Simulated Annealing (ASA) ---
        # ASA uses a deterministic direction combined with temperature-controlled random perturbations.
        # This is effective for rotated landscapes as it doesn't rely on gradient estimates or covariance matrices.
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Biased points
        for val in [1.0, -1.0, 2.0, -2.0]:
            starts.append(np.ones(dim) * val)
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))
            
        # We will run ASA from each start point, but limit the total calls.
        # Allocate ~60% of budget to exploration (ASA), 40% to refinement.
        exploration_budget = int(budget * 0.6)
        
        # ASA parameters
        # T0 is initial temperature, T_end is final temperature
        # We scale T based on the problem scale (box size 10)
        T0 = 5.0
        T_end = 0.01
        # Number of steps per ASA run
        steps_per_asa = 20 
        # Total ASA calls allowed
        max_asa_calls = exploration_budget
        
        candidates = []
        
        for start_x in starts:
            if calls_made + steps_per_asa > budget:
                break
            if time.time() > deadline:
                break
                
            # Run ASA from start_x
            x = start_x.copy()
            f_curr = call_func(x)
            x_best = x.copy()
            f_best = f_curr
            
            # Initial direction
            d = rng.normal(size=dim)
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                d /= d_norm
            else:
                d = np.ones(dim) / np.sqrt(dim)
                
            for i in range(steps_per_asa):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Annealing schedule: geometric cooling
                # T_i = T0 * (T_end / T0) ** (i / steps_per_asa)
                T = T0 * (T_end / T0) ** (i / steps_per_asa)
                
                # Step size proportional to T and dimension
                step_size = T * np.sqrt(dim)
                
                # Propose new point: x_new = x + d * step_size + random_noise
                # The random noise is also scaled by T
                noise = rng.normal(size=dim) * T
                x_new = x + d * step_size + noise
                
                # Clip to box [-5, 5]
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = call_func(x_new)
                
                # Acceptance criterion: Metropolis
                if f_new < f_curr:
                    # Always accept improvement
                    x = x_new
                    f_curr = f_new
                    if f_new < f_best:
                        f_best = f_new
                        x_best = x_new.copy()
                    # Keep direction if improved (or randomize slightly)
                    # ASA often keeps direction if step is accepted, but we can also randomize
                    # Let's randomize direction with some probability to escape local minima
                    if rng.random() < 0.5:
                        d = rng.normal(size=dim)
                        d_norm = np.linalg.norm(d)
                        if d_norm > 1e-10:
                            d /= d_norm
                        else:
                            d = np.ones(dim) / np.sqrt(dim)
                else:
                    # Deterioration: accept with probability exp(-(f_new - f_curr) / T)
                    prob = np.exp(-(f_new - f_curr) / T)
                    if rng.random() < prob:
                        x = x_new
                        f_curr = f_new
                        # Randomize direction more often when accepting bad moves
                        d = rng.normal(size=dim)
                        d_norm = np.linalg.norm(d)
                        if d_norm > 1e-10:
                            d /= d_norm
                        else:
                            d = np.ones(dim) / np.sqrt(dim)
                    # Else: reject, x stays the same, direction stays the same (or could be perturbed)
                
            # Add the best point found in this ASA run to candidates
            candidates.append(x_best)
            
        # Deduplicate candidates roughly
        unique_candidates = []
        for c in candidates:
            if not any(np.linalg.norm(c - uc) < 0.5 for uc in unique_candidates):
                unique_candidates.append(c)
        
        # Ensure global_best_x is in candidates if it was updated
        if global_best_x is not None:
            if not any(np.linalg.norm(global_best_x - uc) < 0.5 for uc in unique_candidates):
                unique_candidates.append(global_best_x.copy())

        # Select top K candidates for intensive refinement
        # Sort unique_candidates by their current function value? 
        # We don't have a direct list of (x, f) for all unique_candidates, only global_best.
        # However, the ASA runs found local minima. The global_best is the best of all.
        # We can just pick the global_best and a few diverse points.
        
        top_candidates = []
        if global_best_x is not None:
            top_candidates.append(global_best_x)
            
        # Add 2 other diverse points from unique_candidates
        for c in unique_candidates:
            if global_best_x is not None and np.linalg.norm(c - global_best_x) > 1.0:
                top_candidates.append(c)
                if len(top_candidates) >= 3:
                    break
        
        # If we didn't find enough diverse points, add random ones
        while len(top_candidates) < 3:
            top_candidates.append(rng.uniform(-5, 5, dim))

        # --- Phase 2: Intensive Refinement ---
        # Allocate remaining budget to these candidates
        # Reserve ~20% of original budget for final polish
        polish_budget = int(budget * 0.2)
        remaining_budget = budget - calls_made - polish_budget
        if remaining_budget < 0:
            remaining_budget = max(0, budget - calls_made)
            
        calls_per_refine = max(10, remaining_budget // len(top_candidates))
        
        from scipy.optimize import minimize
        
        for b_start in top_candidates:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # If we are running low on budget, skip
            if calls_made + calls_per_refine > budget - polish_budget:
                # Use whatever is left
                calls_per_refine = max(5, budget - polish_budget - calls_made)
                if calls_per_refine <= 0:
                    break

            try:
                # Use Powell's method for intensive refinement
                res = minimize(
                    call_func, 
                    b_start, 
                    method='Powell', 
                    options={'maxfev': calls_per_refine, 'maxiter': 100}
                )
            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        # Use Nelder-Mead for high precision on the global best
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            if polish_budget > 5:
                try:
                    res = minimize(
                        call_func, 
                        global_best_x, 
                        method='Nelder-Mead', 
                        options={
                            'xatol': 1e-10, 
                            'fatol': 1e-12, 
                            'maxfev': polish_budget, 
                            'maxiter': 200
                        }
                    )
                    if res.fun < global_best_f:
                        global_best_f = res.fun
                        global_best_x = res.x.copy()
                except Exception:
                    pass

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
