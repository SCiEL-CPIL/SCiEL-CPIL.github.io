"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-structure exploit for BBOB f5 (Linear slope).
    
    Strategy:
    1. Sample a small set of points to estimate the linear gradient.
    2. Use the gradient to jump directly toward the optimal vertex.
    3. Refine with a few local steps if needed.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Track best point and value
        best_x = None
        best_f = np.inf
        
        def _eval(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Estimate gradient via central differences ---
        # Start from a reasonable initial point (center of box)
        x0 = np.zeros(d)
        f0 = _eval(x0)
        
        # Evaluate at x0 +/- delta along each axis to estimate gradient
        delta = 0.5
        grad = np.zeros(d)
        calls = 1  # we've used 1 call
        
        for i in range(d):
            if time.time() > deadline:
                break
            xp = x0.copy()
            xp[i] += delta
            xp = np.clip(xp, -5, 5)
            fp = _eval(xp)
            calls += 1
            
            xm = x0.copy()
            xm[i] -= delta
            xm = np.clip(xm, -5, 5)
            fm = _eval(xm)
            calls += 1
            
            # Central difference
            grad[i] = (fp - fm) / (2 * delta)
        
        # --- Phase 2: Jump toward the optimal vertex using the gradient ---
        # For a linear function f(x) = c + g^T x, the minimum on [-5,5]^d
        # is at the vertex where each coordinate is at the bound opposite to the sign of g[i].
        # But the function may have a shift/rotation, so the effective gradient in the
        # original coordinates is what we estimated.
        
        # Compute the step to move each coordinate to its optimal bound
        x_jump = x0.copy()
        for i in range(d):
            if grad[i] > 0:
                x_jump[i] = -5.0  # move in negative direction
            elif grad[i] < 0:
                x_jump[i] = 5.0   # move in positive direction
            else:
                x_jump[i] = x0[i]  # no preference
        
        f_jump = _eval(x_jump)
        calls += 1
        
        # --- Phase 3: Refine around the jumped point ---
        # If the jump didn't improve much, try a local search
        # Use a few coordinate descent steps to fine-tune
        
        x = x_jump.copy()
        fx = f_jump
        
        # Try small perturbations in each direction
        for i in range(d):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Try moving coordinate i slightly in each direction
            for direction in [-1, 1]:
                if time.time() > deadline or calls >= budget:
                    break
                x_trial = x.copy()
                x_trial[i] += direction * 0.1
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = _eval(x_trial)
                calls += 1
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
        
        # --- Phase 4: If we have budget left, do a quick local refinement ---
        # Use a simple random local search
        rng = np.random.default_rng(42)
        sigma = 0.5
        while calls < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = _eval(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                if sigma < 0.01:
                    break
        
        return x, best_f
# EVOLVE-BLOCK-END
