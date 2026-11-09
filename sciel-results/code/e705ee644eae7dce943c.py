"""Adaptive Differential Evolution optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Differential Evolution (DE) for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Population size: typically 5-10 * dim. For dim=5, 25-50 is good.
        # Budget 500. Let's use a population size that allows ~10-20 generations.
        # lambda = 20. 500/20 = 25 generations.
        pop_size = 20
        
        # Initialize population uniformly in [-5, 5]^5
        pop = rng.uniform(-5.0, 5.0, (pop_size, dim))
        f_vals = np.array([call_func(pop[i]) for i in range(pop_size)])
        
        # Initial best
        idx_best = np.argmin(f_vals)
        best_x = pop[idx_best].copy()
        best_f = f_vals[idx_best]

        # DE parameters
        # F: Mutation factor (0.5 is standard, adapt if needed)
        F = 0.5
        # CR: Crossover rate (0.5 is standard)
        CR = 0.5
        
        # Stagnation counter
        stagnation_count = 0
        # Threshold for restart
        restart_threshold = 10 # generations without improvement
        
        # Adaptive step size for restarts
        restart_scale = 1.0

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(f_vals)
            pop = pop[order]
            f_vals = f_vals[order]
            
            # Check for improvement
            if f_vals[0] < best_f - 1e-12:
                best_f = f_vals[0]
                best_x = pop[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # If stagnated, perform a restart
            if stagnation_count >= restart_threshold:
                # Re-initialize a large portion of the population around the best point
                # with a reduced scale to exploit local region, or random if scale is too small
                if restart_scale > 0.1:
                    restart_scale *= 0.5
                else:
                    # If scale is very small, try a global random reset for some individuals
                    # to escape local minima
                    restart_scale = 1.0
                
                # Keep the top half, re-initialize the bottom half
                keep_count = pop_size // 2
                for i in range(keep_count, pop_size):
                    if restart_scale < 0.1:
                        pop[i] = rng.uniform(-5.0, 5.0, dim)
                    else:
                        pop[i] = best_x + rng.normal(0, restart_scale, dim)
                        pop[i] = np.clip(pop[i], -5.0, 5.0)
                    f_vals[i] = call_func(pop[i])
                
                # Re-sort after restart
                order = np.argsort(f_vals)
                pop = pop[order]
                f_vals = f_vals[order]
                stagnation_count = 0
                
                if calls >= budget:
                    break

            # Generate new population
            new_pop = np.zeros_like(pop)
            new_f_vals = np.zeros(pop_size)
            
            for i in range(pop_size):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Mutation: DE/rand/1
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                # Random bit for each dimension
                cross_mask = rng.random(dim) < CR
                # Ensure at least one dimension is taken from mutant
                if not np.any(cross_mask):
                    cross_mask[rng.integers(dim)] = True
                
                u = np.where(cross_mask, v, pop[i])
                u = np.clip(u, -5.0, 5.0)
                
                f_u = call_func(u)
                
                # Selection: Greedy
                if f_u <= f_vals[i]:
                    new_pop[i] = u
                    new_f_vals[i] = f_u
                else:
                    new_pop[i] = pop[i]
                    new_f_vals[i] = f_vals[i]

            pop = new_pop
            f_vals = new_f_vals
            
            # Update global best
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < best_f:
                best_f = f_vals[idx_best]
                best_x = pop[idx_best].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
