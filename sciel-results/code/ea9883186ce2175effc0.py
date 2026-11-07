"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Exploits the affine structure of BBOB f5 (Linear slope).
    1. Identifies the exact affine model f(x) = w^T x + b using d+1 evaluations.
    2. Computes the exact constrained minimum x* (interior or boundary).
    3. Jumps to x* in ~6 calls.
    4. Uses remaining budget for high-resolution local search to maximize AOCC.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        def timed_out():
            return time.time() > deadline

        d = self.dim
        calls = 0
        best_fx = float('inf')
        best_x = np.zeros(d)
        
        # 1. Affine Model Identification
        # We assume f(x) = w^T x + b.
        # We need d+1 points to determine w (d components) and b (1 component).
        # Using an orthogonal simplex centered at 0 with radius r.
        # Points: p0 = 0, p_i = r * e_i for i=1..d.
        # This design is orthogonal, making the linear system for w and b trivial.
        # f(p0) = b
        # f(p_i) = w_i * r + b  => w_i = (f(p_i) - f(p0)) / r
        
        # Choose r to be safe within bounds [-5, 5]. r=1.0 is safe.
        r = 1.0
        
        # Evaluate p0 (origin)
        p0 = np.zeros(d)
        f0 = func(p0)
        calls += 1
        if f0 < best_fx:
            best_fx = f0
            best_x = p0.copy()
            
        if timed_out():
            return best_x, best_fx

        w = np.zeros(d)
        for i in range(d):
            pi = np.zeros(d)
            pi[i] = r
            fi = func(pi)
            calls += 1
            if fi < best_fx:
                best_fx = fi
                best_x = pi.copy()
                
            w[i] = (fi - f0) / r
            
            if timed_out():
                break

        # 2. Compute Exact Constrained Minimum
        # The unconstrained minimum of a linear function is at infinity.
        # The constrained minimum on [-5, 5]^d is at the vertex determined by the sign of w.
        # If w_i > 0, we want x_i as small as possible (-5).
        # If w_i < 0, we want x_i as large as possible (5).
        # If w_i == 0, any x_i is optimal (flat direction).
        
        x_opt = np.zeros(d)
        for i in range(d):
            if w[i] > 1e-10:
                x_opt[i] = -5.0
            elif w[i] < -1e-10:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0 # Arbitrary choice for flat direction

        f_opt = func(x_opt)
        calls += 1
        if f_opt < best_fx:
            best_fx = f_opt
            best_x = x_opt.copy()

        # 3. Local Refinement
        # Although the affine model is exact for a truly linear function,
        # the task mentions "BBOB-style" which might imply slight non-linearities
        # or numerical noise in the evaluation.
        # More importantly, AOCC rewards early progress and sustained low error.
        # If we are at the global minimum, error is 0 (or 1e-8).
        # If we are close, we want to stay close.
        # 
        # If the function is exactly linear, we are done.
        # However, if there is a hidden quadratic term or the "Linear slope" is 
        # actually a piecewise linear or has a small quadratic component, 
        # a local search can improve.
        # 
        # Given the budget is 500 and we used ~6, we have ~494 calls left.
        # We will perform a simple (1+1) Evolution Strategy starting from x_opt.
        # This ensures we don't wander away if x_opt is correct, but can 
        # adjust if the linear assumption was slightly off.
        
        # Initialize (1+1)-ES parameters
        # Start with a small sigma to stay close to the found optimum.
        # If the function is strictly linear, sigma doesn't matter much, 
        # but we want to avoid large jumps that might increase error if 
        # our model was slightly wrong.
        sigma = 0.1
        x_curr = best_x.copy()
        f_curr = best_fx
        
        # Use a fixed seed for reproducibility in case of ties or specific runs?
        # The task says "hidden seeded shift", implying the function is deterministic per instance.
        # The optimizer's internal randomness should probably be deterministic or just random.
        # Using np.random is fine.
        
        while calls < self.budget and not timed_out():
            # Generate candidate
            direction = np.random.randn(d)
            y = np.clip(x_curr + sigma * direction, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < f_curr:
                x_curr = y
                f_curr = fy
                if f_curr < best_fx:
                    best_fx = f_curr
                    best_x = y.copy()
                # Increase sigma slightly to explore more if we are improving
                # But for a linear function, we are already at the corner.
                # If we are at the corner and move inwards, f increases.
                # So we will rarely accept.
                # If we accept, it means we were not at the true optimum.
                sigma *= 1.1
            else:
                # Decrease sigma to refine
                sigma *= 0.9
                if sigma < 1e-8:
                    # Converged to numerical precision
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
