"""Hypervolume-based Multi-Objective Evolutionary Algorithm for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II with tuned parameters for small budgets.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Smaller population allows more generations with fixed budget
        pop_size = 50
        remaining = self.budget
        n_gen = max(1, remaining // pop_size)
        
        # Initialize population
        X = lo + rng.random((pop_size, self.dim)) * (hi - lo)
        F = np.array([problem(x) for x in X])
        remaining -= pop_size
        
        def dominates(a, b):
            """a dominates b if a <= b in all objectives and a < b in at least one."""
            return np.all(a <= b) and np.any(a < b)
        
        def get_non_dominated(F):
            """Get indices of non-dominated points."""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if dominates(F[i], F[j]):
                        nd[j] = False
            return np.where(nd)[0]
        
        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of points."""
            n = len(indices)
            if n == 0:
                return np.array([])
            dist = np.zeros(n)
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                sorted_local = np.argsort(F_sub[:, k])
                for pos in range(n):
                    idx = sorted_local[pos]
                    if pos == 0 or pos == n - 1:
                        dist[idx] = np.inf
                    else:
                        prev_idx = sorted_local[pos - 1]
                        next_idx = sorted_local[pos + 1]
                        dist[idx] += (F_sub[next_idx, k] - F_sub[prev_idx, k])
            return dist

        def select_survivors(F, pop_size):
            """Standard NSGA-II selection: Non-dominated sorting + Crowding distance."""
            n = len(F)
            if n <= pop_size:
                return np.arange(n)
            
            # Non-dominated sorting
            remaining_indices = np.arange(n)
            fronts = []
            
            while len(remaining_indices) > 0:
                # Get non-dominated from remaining
                n = len(remaining_indices)
                nd_mask = np.ones(n, dtype=bool)
                for i in range(n):
                    if not nd_mask[i]:
                        continue
                    for j in range(n):
                        if i == j or not nd_mask[j]:
                            continue
                        if dominates(F[remaining_indices[i]], F[remaining_indices[j]]):
                            nd_mask[j] = False
                
                front_indices = remaining_indices[nd_mask]
                fronts.append(front_indices)
                
                # Remove these from remaining
                remaining_indices = remaining_indices[~nd_mask]
            
            # Select from fronts
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to pick from this front
                    needed = pop_size - len(selected)
                    dists = crowding_distance(F, front)
                    # Sort by distance descending
                    order = np.argsort(-dists)
                    selected.extend(front[order[:needed]])
                    break
            
            return np.array(selected)
        
        # Evolutionary loop
        for gen in range(n_gen):
            if remaining <= 0:
                break
            
            # Generate offspring
            n_offspring = min(pop_size, remaining)
            X_off = np.zeros((n_offspring, self.dim))
            
            for i in range(n_offspring):
                # Binary tournament selection for two parents from current population
                # Use NSGA-II ranking (front index) and crowding for tournament
                # Simplified: Random selection from non-dominated set if possible, else random
                nd_idx = get_non_dominated(F)
                if len(nd_idx) > 0:
                    p1_idx = nd_idx[rng.integers(len(nd_idx))]
                    p2_idx = nd_idx[rng.integers(len(nd_idx))]
                else:
                    p1_idx = rng.integers(pop_size)
                    p2_idx = rng.integers(pop_size)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # SBX crossover
                if rng.random() < 0.9:
                    eta = 15.0
                    x_off = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            x_off[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            x_off[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                else:
                    x_off = x1.copy()
                
                # Polynomial mutation with higher probability
                eta_m = 20.0
                for d in range(self.dim):
                    if rng.random() < 0.2: # 20% mutation rate per gene
                        u = rng.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        x_off[d] += delta * (hi[d] - lo[d])
                
                x_off = np.clip(x_off, lo, hi)
                X_off[i] = x_off
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in X_off])
            remaining -= n_offspring
            
            # Combine parent and offspring populations
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Select next generation
            sel_idx = select_survivors(F_comb, pop_size)
            X = X_comb[sel_idx]
            F = F_comb[sel_idx]
        
        # Return the final non-dominated set
        nd_idx = get_non_dominated(F)
        return [X[i] for i in nd_idx]
# EVOLVE-BLOCK-END
