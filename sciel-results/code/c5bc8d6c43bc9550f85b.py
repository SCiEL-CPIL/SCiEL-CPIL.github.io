import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer for BBOB f21 (Gallagher 101 peaks).
    Combines LHS for global coverage, (1+1)-ES with adaptive sigma for local refinement,
    and a final polish phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin for 330s limit

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            # Clip to domain
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial diverse sampling using Latin Hypercube Sampling (LHS)
        # We want to spread out the initial evaluations to cover the landscape.
        # f21 is multi-modal with 101 peaks, so broad coverage is important.
        n_initial = min(30, self.budget // 10)  # 30 points for 500 budget
        
        def generate_lhs(n, dim, rng):
            points = np.zeros((n, dim))
            for j in range(dim):
                # Random permutation of the n intervals for each dimension
                perm = rng.permutation(n)
                # Add random offset within the interval
                points[:, j] = perm / n + rng.random(n) / n
            # Map from [0, 1] to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        initial_points = generate_lhs(n_initial, self.dim, rng)
        
        # Evaluate initial points
        initial_vals = []
        for p in initial_points:
            if calls_made >= self.budget or time_left() < 5:
                break
            v = evaluate(p)
            initial_vals.append((v, p.copy()))
            
        # Sort by value
        initial_vals.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement on the best candidates using (1+1)-ES
        # We will try to refine the top few candidates.
        # (1+1)-ES is robust and simple, works well for smooth local minima.
        max_candidates_to_refine = 5
        # Allocate budget for local search
        # Reserve some budget for final polish
        budget_for_refinement = self.budget - calls_made - 20
        
        for i, (val, x_start) in enumerate(initial_vals[:max_candidates_to_refine]):
            if calls_made >= self.budget or time_left() < 10:
                break
            
            # If we don't have enough budget for a reasonable run, skip
            if budget_for_refinement < 20:
                break
                
            x = x_start.copy()
            fx = val
            sigma = 1.0  # Start with a moderate step size
            no_improve = 0
            max_no_improve = 50  # Stop if no improvement for 50 steps
            
            # Run (1+1)-ES from this candidate
            while calls_made < self.budget - 20 and time_left() > 5 and no_improve < max_no_improve:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    # Increase step size on success
                    sigma *= 1.2
                    no_improve = 0
                else:
                    # Decrease step size on failure
                    sigma *= 0.85
                    no_improve += 1
                    
                # If sigma becomes too small, we've likely converged
                if sigma < 1e-4:
                    break

        # Phase 3: Final polish with smaller steps around the global best
        if calls_made < self.budget and time_left() > 2 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with a small step size for polishing
            no_improve = 0
            max_no_improve = 30
            
            while calls_made < self.budget and time_left() > 1 and no_improve < max_no_improve:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

        return best_x, best_f
