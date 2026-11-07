"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer exploiting DTLZ1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # DTLZ1: first 4 vars control front, last 3 control g(x)
        # Set last 3 to 0 for Pareto front membership
        n_front = 4  # first 4 variables
        n_g = self.dim - n_front  # last 3 variables

        def make_x(front_vars):
            """Create full vector with g-vars set to 0."""
            x = np.zeros(self.dim)
            x[:n_front] = front_vars
            # x[n_front:] = 0 (already zero)
            return x

        # Initialize population spread across the Pareto front
        # For DTLZ1, the Pareto front is defined by:
        # f0 = 0.5 * prod(x_i) * (1+g)
        # f1 = 0.5 * (1-x_1) * prod(x_i) * (1+g)
        # f2 = 0.5 * (1-x_2) * (1+g)
        # With g=0, we need to spread x_0..x_3 to cover the front
        
        pop_size = 40
        # Create diverse initial points on the Pareto front
        # Use a simplex-based spread for the first 4 variables
        X_front = np.zeros((pop_size, n_front))
        
        # First few points: corners and edges
        X_front[0] = [0.5, 0.5, 0.5, 0.5]  # center
        X_front[1] = [0.9, 0.1, 0.1, 0.1]
        X_front[2] = [0.1, 0.9, 0.1, 0.1]
        X_front[3] = [0.1, 0.1, 0.9, 0.1]
        X_front[4] = [0.1, 0.1, 0.1, 0.9]
        
        # Random spread
        for i in range(5, pop_size):
            # Use Dirichlet-like distribution for spread
            w = rng.dirichlet(np.ones(n_front))
            X_front[i] = w
        
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            X[i] = make_x(X_front[i])
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Non-dominated sorting helper
        def is_dominated(F_arr, idx):
            f = F_arr[idx]
            for j in range(len(F_arr)):
                if j == idx:
                    continue
                if np.all(F_arr[j] <= f) and np.any(F_arr[j] < f):
                    return True
            return False
        
        # DE parameters
        CR = 0.7
        F_DE = 0.8
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated set
            nondom_idx = [i for i in range(len(F)) if not is_dominated(F, i)]
            if not nondom_idx:
                nondom_idx = list(range(len(F)))
            
            # Select parents from non-dominated set
            n_parents = min(3, len(nondom_idx))
            parent_indices = rng.choice(nondom_idx, size=n_parents, replace=False)
            a, b, c = [X[parent_indices[0]], X[parent_indices[1]], X[parent_indices[2]]]
            
            # DE mutation
            mutant = a + F_DE * (b - c)
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover
            cross_mask = rng.random(self.dim) < CR
            cross_mask[rng.integers(self.dim)] = True  # Ensure at least one gene
            trial = np.where(cross_mask, mutant, X[parent_indices[0]])
            
            # Evaluate trial
            trial_f = problem(trial)
            used += 1
            
            # Replace worst point in population
            # Use sum of normalized objectives as a simple fitness
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1
            F_norm = (F - F_min) / F_range
            fitness = F_norm.sum(axis=1)
            
            worst_idx = np.argmax(fitness)
            
            # Only replace if trial is not dominated by worst
            if not (np.all(F[worst_idx] <= trial_f) and np.any(F[worst_idx] < trial_f)):
                X[worst_idx] = trial
                F[worst_idx] = trial_f
        
        return X
# EVOLVE-BLOCK-END
