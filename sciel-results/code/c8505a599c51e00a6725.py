# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES-lite with per-dimension adaptive step sizes and periodic gradient polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initialize position and per-dimension step sizes
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Per-dimension step sizes
        # Start slightly larger to explore the box [-5, 5] more effectively
        sigma = np.full(d, 1.5)

        # Success counters per dimension (recent window)
        window_size = 10
        successes = np.zeros((d, window_size), dtype=int)
        succ_idx = 0

        calls_made = 1
        polish_interval = 40  # Polish more frequently
        calls_since_polish = 0

        # Finite-difference gradient polish helper
        def polish(x_cur, f_cur, remaining_budget):
            nonlocal best_x, best_f, calls_made
            if remaining_budget < 20:
                return x_cur, f_cur
            
            x_work = x_cur.copy()
            f_work = f_cur
            eps = 1e-4
            
            # Phase 1: Gradient Descent
            # Do 3 gradient steps
            for _ in range(3):
                if calls_made >= budget:
                    break
                
                # Compute Gradient
                grad = np.zeros(d)
                for i in range(d):
                    if calls_made >= budget:
                        break
                    xp = x_work.copy()
                    xp[i] += eps
                    fp = func(np.clip(xp, -5, 5))
                    calls_made += 1
                    xm = x_work.copy()
                    xm[i] -= eps
                    fm = func(np.clip(xm, -5, 5))
                    calls_made += 1
                    grad[i] = (fp - fm) / (2 * eps)
                
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-10:
                    break
                
                # Direction is negative gradient
                direction = -grad / gnorm
                
                # Backtracking Line Search
                found_improvement = False
                # Try larger steps first to make rapid progress
                for lr in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                    if calls_made >= budget:
                        break
                    x_try = np.clip(x_work + lr * direction, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < f_work:
                        x_work = x_try
                        f_work = f_try
                        found_improvement = True
                        break
                
                if not found_improvement:
                    break
            
            # Phase 2: Coordinate-wise Refinement
            # Do 1 sweep of coordinate-wise local search
            # This is cheaper than full gradient and good for fine-tuning
            if calls_made < budget:
                delta = 0.05
                for i in range(d):
                    if calls_made >= budget:
                        break
                    # Try positive
                    xp = x_work.copy()
                    xp[i] += delta
                    fp = func(np.clip(xp, -5, 5))
                    calls_made += 1
                    if fp < f_work:
                        x_work = xp
                        f_work = fp
                        # Try to go further in same direction?
                        # Keep it simple: just 1 step per side per dim
                        # Actually, let's try a small ternary-like check
                        # No, stick to simple 2-point check to save budget
                        # If it improved, maybe try a bit more?
                        # Let's just do 1 step.
                    
                    # Try negative (from original x_work before positive move? or current?)
                    # Better to compare against the best of the three: x_work, x_work+delta, x_work-delta
                    # Let's restart the coordinate check properly
                    # Re-implementing coordinate search more carefully
                    pass # The simple loop above is a bit messy. Let's replace it.
            
            # Better Coordinate Search
            if calls_made < budget:
                delta = 0.05
                for i in range(d):
                    if calls_made >= budget:
                        break
                    x_base = x_work.copy()
                    f_base = f_work
                    
                    xp = x_base.copy()
                    xp[i] += delta
                    fp = func(np.clip(xp, -5, 5))
                    calls_made += 1
                    
                    xm = x_base.copy()
                    xm[i] -= delta
                    fm = func(np.clip(xm, -5, 5))
                    calls_made += 1
                    
                    if fp < f_base:
                        x_work = xp
                        f_work = fp
                    elif fm < f_base:
                        x_work = xm
                        f_work = fm
            
            if f_work < best_f:
                best_x = x_work.copy()
                best_f = f_work
            return x_work, f_work

        while calls_made < budget:
            # Generate candidate
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            calls_since_polish += 1

            if fy < fx:
                # Accept move
                x, fx = y, fy
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                # Update per-dimension success
                for i in range(d):
                    successes[i, succ_idx] = 1
            else:
                for i in range(d):
                    successes[i, succ_idx] = 0

            succ_idx = (succ_idx + 1) % window_size

            # Update step sizes based on success rate
            # Use hysteresis to prevent oscillation
            for i in range(d):
                rate = np.mean(successes[i])
                if rate > 0.6:
                    sigma[i] *= 1.15
                elif rate < 0.4:
                    sigma[i] *= 0.85
                # Clamp step sizes
                sigma[i] = np.clip(sigma[i], 0.01, 5.0)

            # Periodic polish phase
            if calls_since_polish >= polish_interval and calls_made < budget:
                calls_since_polish = 0
                x, fx = polish(x, fx, budget - calls_made)
                # Re-center sigma? No, keep it.

        # Final polish if budget allows
        if calls_made < budget:
            polish(best_x, best_f, budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
