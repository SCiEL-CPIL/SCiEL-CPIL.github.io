"""Multi-start Differential Evolution for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """DE/rand/1 with population of 50, enhanced diversity injection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 50

        # Initialize population
        pop = np.zeros((pop_size, dim))
        # Structured starts: mix of origin, random, and boundary
        pop[0] = 0.0
        pop[1] = rng.uniform(-1, 1, dim)
        pop[2] = np.full(dim, 5.0)
        pop[3] = np.full(dim, -5.0)
        for i in range(4, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fvals = np.array([func(p) for p in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_used = pop_size

        # DE parameters
        F = 0.8
        CR = 0.9
        
        # Stagnation tracking
        calls_since_improvement = 0
        stagnation_threshold = 20 # Inject diversity if no improvement in 20 calls

        while calls_used < budget:
            # Sort population by fitness to identify worst for replacement
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # If we are stagnating, replace the worst individual with a new random point
            if calls_since_improvement >= stagnation_threshold:
                if calls_used < budget:
                    # Replace the worst (last in sorted array)
                    worst_idx = pop_size - 1
                    pop[worst_idx] = rng.uniform(-5, 5, dim)
                    fvals[worst_idx] = func(pop[worst_idx])
                    calls_used += 1
                    if fvals[worst_idx] < best_f:
                        best_f = fvals[worst_idx]
                        best_x = pop[worst_idx].copy()
                        calls_since_improvement = 0
                    else:
                        # Reset counter partially to avoid immediate re-injection loop, 
                        # but keep pressure high
                        calls_since_improvement = 0 
                        # Actually, if it didn't improve, we should maybe keep counting? 
                        # No, the act of injection itself is a change. 
                        # Let's reset to 0 to give the new point a chance.
                else:
                    break

            # Generate offspring for each member
            for i in range(pop_size):
                if calls_used >= budget:
                    break

                # Efficient selection of 3 distinct indices
                # Use rng.choice on a permutation or just simple rejection with pre-allocated list
                # For pop_size=50, rejection is fast enough, but let's use a faster method
                indices = np.arange(pop_size)
                # Remove i
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                candidates = indices[mask]
                # Pick 3 distinct
                chosen = rng.choice(candidates, size=3, replace=False)
                a, b, c = chosen[0], chosen[1], chosen[2]

                # Adaptive F: slightly reduce F if we are close to convergence? 
                # Or keep it high for exploration. 
                # Let's try oscillating F slightly or keeping it fixed but improving diversity.
                # Standard DE works well with F=0.8, CR=0.9.
                
                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                # Vectorized crossover for speed
                mask_cr = rng.random(dim) < CR
                mask_cr[j_rand] = True
                trial[:] = np.where(mask_cr, v, pop[i])

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        calls_since_improvement = 0
                    else:
                        calls_since_improvement += 1
                else:
                    calls_since_improvement += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
