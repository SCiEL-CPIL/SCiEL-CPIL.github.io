"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        N = 100
        if self.budget < N + N: # Need at least init + 1 generation
             N = self.budget // 2
        if N < 10:
             N = 10

        # Generate weight vectors
        # Simple grid or random? Random with normalization is fine for small N.
        # Or systematic. For n_obj=3, we can use a systematic approach or just random.
        # Let's use random normalized weights.
        weights = rng.random((N, n_obj))
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Find neighbors
        # Distance based
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        T = max(5, N // 5) # Neighborhood size
        neighbors = np.zeros((N, T), dtype=int)
        for i in range(N):
            sorted_indices = np.argsort(dists[i])
            # Exclude self
            sorted_indices = sorted_indices[sorted_indices != i]
            neighbors[i] = sorted_indices[:T]

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Ideal point
        ideal_point = F.min(axis=0)

        # Tchebycheff scalarization
        def tchebycheff(f, w, z):
            return np.max(w * np.abs(f - z))

        # Main loop
        while used < self.budget:
            # Determine how many subproblems we can update in this generation
            # Each update costs 1 evaluation.
            # We can update up to (budget - used) subproblems.
            # To maintain diversity, we should cycle through subproblems.
            
            # Shuffle subproblem indices to avoid bias
            idx_order = rng.permutation(N)
            
            # We will update as many as we can
            # But we must be careful: if we update one, the population changes.
            # Standard MOEA/D updates one subproblem at a time.
            
            for i in idx_order:
                if used >= self.budget:
                    break
                
                # Select parents from neighbors
                # Pick 2 random neighbors
                n_idx = neighbors[i]
                r1_idx = n_idx[rng.integers(T)]
                r2_idx = n_idx[rng.integers(T)]
                
                x_r1 = X[r1_idx]
                x_r2 = X[r2_idx]
                x_r0 = X[i] # Current solution
                
                # DE Mutation: x_new = x_r1 + F * (x_r2 - x_r1)
                F_factor = 0.5
                x_mut = x_r1 + F_factor * (x_r2 - x_r1)
                
                # Crossover
                CR = 0.5
                x_child = np.zeros_like(x_r0)
                cross_point = rng.random(dim) < CR
                x_child[cross_point] = x_mut[cross_point]
                x_child[~cross_point] = x_r0[~cross_point]
                
                # Clip
                x_child = np.clip(x_child, lo, hi)
                
                # Evaluate
                f_child = problem(x_child)
                used += 1
                
                # Update ideal point
                ideal_point = np.minimum(ideal_point, f_child)
                
                # Check if child is better than current solution for subproblem i
                # And potentially for neighbors
                # Standard MOEA/D: update if g(child, w_i) < g(current, w_i)
                g_child = tchebycheff(f_child, weights[i], ideal_point)
                g_current = tchebycheff(F[i], weights[i], ideal_point)
                
                if g_child < g_current:
                    # Replace current
                    X[i] = x_child
                    F[i] = f_child
                    
                    # Optionally update neighbors if child is better for them
                    # This helps information flow
                    for j in neighbors[i]:
                        if j == i:
                            continue
                        g_child_j = tchebycheff(f_child, weights[j], ideal_point)
                        g_current_j = tchebycheff(F[j], weights[j], ideal_point)
                        if g_child_j < g_current_j:
                            X[j] = x_child
                            F[j] = f_child

        # Extract Pareto front
        # Simple non-dominated sort on F
        front_indices = []
        for i in range(N):
            dominated = False
            for j in range(N):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                front_indices.append(i)
                
        if not front_indices:
            # Fallback: return all
            front_indices = list(range(N))
            
        return X[front_indices]
# EVOLVE-BLOCK-END
