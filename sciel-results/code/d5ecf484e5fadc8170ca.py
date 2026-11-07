"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive Coordinate Descent with Gradient Polishing.
    
    Strategy:
    1. Global exploration via random perturbations with adaptive per-dimension step sizes.
    2. Local exploitation via periodic finite-difference gradient descent with line search.
    3. Restart mechanism to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget
        
        # Trackers
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def evaluate(x):
            nonlocal calls_made
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            return x_clipped, f_val

        # 1. Initialization: Sample a few points to find a good start
        init_samples = 10
        init_x = None
        init_f = np.inf
        for _ in range(init_samples):
            x_try = rng.uniform(-5, 5, d)
            x_try, f_try = evaluate(x_try)
            if f_try < init_f:
                init_f = f_try
                init_x = x_try
        
        if init_x is None:
            init_x = np.zeros(d)
            init_f = func(init_x)
            calls_made += 1
            
        best_x = init_x.copy()
        best_f = init_f
        x = init_x.copy()
        fx = init_f
        
        # Adaptive step sizes
        sigma = np.full(d, 1.0)
        
        # Success history for step size adaptation
        window_size = 20
        successes = np.zeros((d, window_size), dtype=int)
        succ_idx = 0
        
        # Polish function: Gradient Descent with Line Search
        def polish(x_cur, f_cur):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget - 5: # Need budget for at least one grad eval
                return x_cur, f_cur
                
            x_work = x_cur.copy()
            f_work = f_cur
            
            # Adaptive epsilon for gradient
            # Scale epsilon with sigma to ensure we sample within the "noise" level
            eps_base = 1e-3
            max_steps = 15
            
            for _ in range(max_steps):
                if calls_made >= budget - 1:
                    break
                    
                # Estimate gradient
                grad = np.zeros(d)
                for i in range(d):
                    if calls_made >= budget:
                        break
                    eps = max(eps_base, sigma[i] * 1e-2) # Ensure eps is not too small relative to sigma
                    xp = x_work.copy()
                    xm = x_work.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    _, fp = evaluate(xp)
                    _, fm = evaluate(xm)
                    grad[i] = (fp - fm) / (2 * eps)
                
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break
                    
                # Direction is negative gradient
                direction = -grad / gnorm
                
                # Line Search (Backtracking)
                step_size = 0.5
                improved = False
                for _ in range(10):
                    x_try = x_work + step_size * direction
                    x_try, f_try = evaluate(x_try)
                    if f_try < f_work - 1e-8: # Sufficient decrease
                        x_work = x_try
                        f_work = f_try
                        improved = True
                        break
                    step_size *= 0.5
                    
                if not improved:
                    break
                    
            if f_work < best_f:
                best_x = x_work.copy()
                best_f = f_work
                
            return x_work, f_work

        # Main Loop
        polish_interval = 20
        calls_since_polish = 0
        stall_counter = 0
        
        while calls_made < budget:
            # Generate candidate
            y = x + sigma * rng.standard_normal(d)
            y, fy = evaluate(y)
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                stall_counter = 0
            else:
                stall_counter += 1
            
            # Update success history
            for i in range(d):
                if fy < fx:
                    successes[i, succ_idx] = 1
                else:
                    successes[i, succ_idx] = 0
            
            succ_idx = (succ_idx + 1) % window_size
            
            # Accept or Reject
            if fy < fx:
                x = y
                fx = fy
            else:
                # Keep x, but maybe move halfway? No, standard rejection.
                # However, to keep moving, we might accept if it's not worse by much?
                # For simplicity, reject.
                pass
                
            # Update Step Sizes
            for i in range(d):
                rate = np.mean(successes[i])
                if rate > 0.5:
                    sigma[i] *= 1.05
                elif rate < 0.5:
                    sigma[i] *= 0.95
                sigma[i] = np.clip(sigma[i], 1e-4, 5.0)
                
            calls_since_polish += 1
            
            # Trigger Polish
            if calls_since_polish >= polish_interval or (stall_counter > 10 and calls_since_polish > 5):
                calls_since_polish = 0
                stall_counter = 0
                x, fx = polish(x, fx)
                
                # Check for stagnation after polish
                # If sigma is very small, we might be stuck
                if np.mean(sigma) < 1e-3:
                    # Restart from best with perturbation
                    if calls_made < budget - 10:
                        x = best_x + rng.normal(0, 0.5, d)
                        x, fx = evaluate(x)
                        sigma = np.full(d, 1.0)
                        successes = np.zeros((d, window_size), dtype=int)
                        succ_idx = 0

        # Final Polish
        if calls_made < budget - 5:
            x, fx = polish(best_x, best_f)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
