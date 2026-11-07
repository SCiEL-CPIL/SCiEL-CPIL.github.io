# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution (DE/rand/1) for multi-objective optimization (DTLZ4)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population size
        pop_size = 50
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 4)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DE/rand/1/bin parameters
        F_factor = 0.5
        CR = 0.5

        while used < self.budget:
            # Select three distinct random parents
            j1 = rng.integers(pop_size)
            j2 = rng.integers(pop_size)
            j3 = rng.integers(pop_size)
            # Ensure distinctness (simple retry logic)
            while j1 == j2 or j1 == j3 or j2 == j3:
                j1 = rng.integers(pop_size)
                j2 = rng.integers(pop_size)
                j3 = rng.integers(pop_size)

            # DE mutation: v = x_j1 + F * (x_j2 - x_j3)
            v = X[j1] + F_factor * (X[j2] - X[j3])
            
            # Clip to bounds
            v = np.clip(v, lo, hi)

            # Crossover
            # We need a victim 'i' for the trial. 
            # Let's pick the victim using tournament selection (size 2) to target weaker individuals.
            a = rng.integers(pop_size)
            b = rng.integers(pop_size)
            while a == b:
                b = rng.integers(pop_size)
            
            # Determine victim i
            # If F[a] dominates F[b], i=b (worse)
            # If F[b] dominates F[a], i=a (worse)
            # Else random
            dom_a_over_b = np.all(F[a] <= F[b]) and np.any(F[a] < F[b])
            dom_b_over_a = np.all(F[b] <= F[a]) and np.any(F[b] < F[a])
            
            if dom_a_over_b:
                i = b
            elif dom_b_over_a:
                i = a
            else:
                i = a if rng.random() < 0.5 else b

            # Now generate trial based on X[i] (base vector)
            # Wait, DE/rand/1 uses x_j1 as the base vector? 
            # Standard DE/rand/1: v = x_j1 + F*(x_j2 - x_j3). 
            # The trial is compared against x_i. 
            # Usually, i is the index of the base vector? 
            # In DE/rand/1, the base vector is one of the random ones. 
            # Let's stick to the previous structure where i was the base vector? 
            # No, in the previous code:
            # i = rng.integers(pop_size)
            # j1, j2, j3 random.
            # v = X[j1] + ...
            # trial = X[i].copy() ...
            # This is DE/rand/1/bin where i is the target, and j1, j2, j3 are parents.
            # This is valid.
            
            trial = X[i].copy()
            rand_idx = rng.integers(self.dim)
            for k in range(self.dim):
                if rng.random() < CR or k == rand_idx:
                    trial[k] = v[k]
            
            # Evaluate trial
            f_trial = problem(trial)
            used += 1

            # Selection
            dom_trial_over_i = np.all(f_trial <= F[i]) and np.any(f_trial < F[i])
            dom_i_over_trial = np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)
            
            if dom_trial_over_i:
                X[i] = trial
                F[i] = f_trial
            elif not dom_i_over_trial:
                # Non-dominated: Accept with probability 0.5 to maintain diversity
                if rng.random() < 0.5:
                    X[i] = trial
                    F[i] = f_trial
            # else: X[i] dominates trial, discard

        return X
# EVOLVE-BLOCK-END
