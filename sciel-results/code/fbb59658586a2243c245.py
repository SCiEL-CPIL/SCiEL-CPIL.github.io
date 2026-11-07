"""Multi-start adaptive (1+1)-ES with restarts and deadline guard for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with restarts for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Use system entropy for better exploration diversity across instances/runs
        rng = np.random.default_rng()
        deadline = time.time() + 280.0  # safety margin below the 330s kill

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Latin Hypercube Sampling (LHS) for broad initial coverage
        # LHS ensures that each dimension is covered uniformly, which is good for multi-modal functions
        n_initial = 30
        if n_initial > budget:
            n_initial = budget // 2
            
        # Generate LHS points
        # For each dimension, divide the range into n_initial intervals and pick one random point in each
        lhs_points = []
        for _ in range(n_initial):
            pass # Placeholder, we generate all at once below

        # Efficient LHS generation
        # perm[i] is a permutation of [0, n_initial)
        # point[i, j] is in the interval [perm[i, j]/n_initial, (perm[i, j]+1)/n_initial]
        
        if n_initial > 0:
            perms = [rng.permutation(n_initial) for _ in range(dim)]
            for i in range(n_initial):
                point = np.zeros(dim)
                for d in range(dim):
                    # Uniform in [0, 1] within the bin
                    frac = (perms[d][i] + rng.random()) / n_initial
                    point[d] = -5.0 + 10.0 * frac
                lhs_points.append(point)
        
        # Also add center and some random points
        lhs_points.append(np.zeros(dim))
        
        for s in lhs_points:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: (1+1)-ES with Restart Strategy
        # We run the ES until no improvement for a certain number of steps, then restart
        # from a new random point or the best point with a larger step size.
        
        max_restarts = 10
        restart_threshold = 50 # If no improvement for 50 steps, restart
        current_sigma = 1.5
        
        for restart_iter in range(max_restarts):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            # If this is not the first restart, we might want to start from a new random point
            # or perturb the best point. For Weierstrass, starting fresh is often better to escape local minima.
            if restart_iter > 0:
                # Pick a new random start point
                start_x = rng.uniform(-4.5, 4.5, dim)
                fx = safe_func(start_x)
                if fx < best_f:
                    best_f = fx
                    best_x = start_x.copy()
                
                # Reset sigma for the new local search
                current_sigma = rng.uniform(0.5, 2.0)
            else:
                # First iteration, use the best found in Phase 1
                current_sigma = 1.5

            no_improve_count = 0
            
            while n_calls[0] < budget and time.time() < deadline:
                # Generate candidate
                candidate = best_x + current_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    no_improve_count = 0
                    # On success, slightly decrease sigma (fine-tuning)
                    current_sigma *= 0.9
                else:
                    no_improve_count += 1
                    # On failure, slightly increase sigma (exploration)
                    current_sigma *= 1.1

                # Keep sigma in bounds
                current_sigma = max(current_sigma, 1e-4)
                current_sigma = min(current_sigma, 5.0)

                # Check for restart condition
                if no_improve_count > restart_threshold:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
