"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOEA/D (Decomposition-based) for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for MOEA/D
        # With budget 200, we can afford 4 generations of size 50 or 2 generations of size 100.
        # Decomposition benefits from more generations to propagate information.
        pop_size = 50
        
        # Generate weight vectors
        # For 2 objectives, we can use a linear grid
        weights = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            weights[i, 0] = i / (pop_size - 1)
            weights[i, 1] = 1 - weights[i, 0]
        
        # Define neighborhood size
        T = 10 # Top 10 nearest weight vectors
        
        # Calculate neighborhood for each sub-problem
        # Distance between weight vectors
        dists = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(pop_size):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighbors = []
        for i in range(pop_size):
            # Sort indices by distance to i
            sorted_idx = np.argsort(dists[i])
            # Exclude self
            sorted_idx = [j for j in sorted_idx if j != i]
            neighbors.append(sorted_idx[:T])

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n_vars))
        F = np.array([problem(x) for x in X])
        remaining_budget = budget - pop_size

        # Ideal point (lower bound of objectives)
        # Initialize with first solution or infinity
        ideal = np.min(F, axis=0)
        
        # Tchebycheff scalarization
        def tchebycheff(F, w, ideal):
            # Handle division by zero if ideal is 0? ZDT4 f1, f2 >= 0.
            # If ideal is 0, we use weighted sum or just distance.
            # Standard Tchebycheff: max(w_i * |f_i - z_i^*|)
            # If z_i^* is 0, it's fine.
            diffs = np.abs(F - ideal)
            # Avoid division by zero in normalization if used, but Tchebycheff doesn't require normalization if weights are scaled.
            # However, objective scales differ. f1 ~ [0, 1], f2 ~ [0, 400].
            # We should normalize objectives by their range or use weighted sum with normalized objectives?
            # Tchebycheff is scale sensitive. 
            # Let's use Weighted Sum on normalized objectives for robustness in this short run.
            # Or normalize F by (max - min) of current population? 
            # Better: Normalize by known bounds of ZDT4?
            # f1 in [0, 1], f2 in [0, 400].
            # Let's normalize F dynamically.
            return diffs

        def scalar_fitness(F, w, ideal, F_min, F_max):
            # Normalize objectives
            F_norm = (F - F_min) / (F_max - F_min + 1e-8)
            # Tchebycheff
            diffs = np.abs(F_norm - 0) # Ideal is 0 in normalized space? 
            # Actually, ideal in original space is min. In normalized space, ideal is 0.
            # So g(x) = max(w_i * f_norm_i)
            return np.max(w * F_norm)

        # Update loop
        while remaining_budget >= pop_size:
            new_X = np.zeros_like(X)
            new_F = np.zeros_like(F)
            
            # Current min/max for normalization
            F_min = np.min(F, axis=0)
            F_max = np.max(F, axis=0)
            
            for i in range(pop_size):
                if remaining_budget < 1:
                    break
                
                # Select parents from neighborhood
                # Pick 2 parents from neighbors[i] + [i]
                cand_indices = [i] + neighbors[i]
                # Shuffle and pick 2 distinct
                p1_idx = cand_indices[rng.integers(0, len(cand_indices))]
                p2_idx = cand_indices[rng.integers(0, len(cand_indices))]
                while p2_idx == p1_idx:
                    p2_idx = cand_indices[rng.integers(0, len(cand_indices))]
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # DE Mutation
                # x_new = x1 + F * (x1 - x2) ? No, standard DE is x_r1 + F*(x_r2 - x_r3)
                # Let's use x_r1 + F*(x_r2 - x_r3)
                # We need 3 parents. Let's pick 3 from neighborhood.
                if len(cand_indices) >= 3:
                    p3_idx = cand_indices[rng.integers(0, len(cand_indices))]
                    while p3_idx == p1_idx or p3_idx == p2_idx:
                        p3_idx = cand_indices[rng.integers(0, len(cand_indices))]
                    x3 = X[p3_idx]
                else:
                    x3 = lo + (hi - lo) * rng.random(n_vars)
                
                F_factor = 0.8 # Mutation factor
                x_new = x1 + F_factor * (x2 - x3)
                
                # Crossover
                # Binary crossover
                crossover_rate = 0.5
                mask = rng.random(n_vars) < crossover_rate
                # Ensure at least one bit is from x1
                if not np.any(mask):
                    idx = rng.integers(0, n_vars)
                    mask[idx] = True
                    
                x_child = np.where(mask, x1, x_new)
                
                # Mutation (Gaussian or Polynomial)
                # Add small Gaussian noise
                x_child = x_child + 0.1 * (hi - lo) * rng.standard_normal(n_vars)
                x_child = np.clip(x_child, lo, hi)
                
                # Evaluate
                f_child = problem(x_child)
                remaining_budget -= 1
                
                new_X[i] = x_child
                new_F[i] = f_child
            
            # Selection: Update population
            # For each i, compare new_F[i] with F[i] and neighbors' F
            # If new solution is better than current for sub-problem i, replace.
            # Also, if new solution is better than neighbor's current solution, replace neighbor.
            # This is the "Update Neighborhood" step.
            
            # First, compute scalar fitness for all current and new solutions
            # We need to decide which individuals to keep.
            # Standard MOEA/D: 
            # For each i, if g(new_x_i, w_i) < g(x_i, w_i), x_i = new_x_i
            # Then, for each neighbor j of i, if g(x_i, w_j) < g(x_j, w_j), x_j = x_i
            
            # This requires evaluating scalar fitness multiple times.
            # To simplify and speed up, let's just do:
            # 1. Compute scalar fitness for new population using current ideal.
            # 2. Compute scalar fitness for old population.
            # 3. For each i, if new is better than old for weight w_i, keep new.
            # 4. Then do the neighbor update pass.
            
            # Update ideal point
            ideal = np.min(np.vstack([F, new_F]), axis=0)
            
            F_min = np.min(np.vstack([F, new_F]), axis=0)
            F_max = np.max(np.vstack([F, new_F]), axis=0)
            
            # Scalar fitness function
            def get_scalar(F_vec, w):
                F_norm = (F_vec - F_min) / (F_max - F_min + 1e-8)
                return np.max(w * F_norm)
            
            # Evaluate old population
            old_scales = np.array([get_scalar(F[i], weights[i]) for i in range(pop_size)])
            # Evaluate new population
            new_scales = np.array([get_scalar(new_F[i], weights[i]) for i in range(pop_size)])
            
            # Update X, F
            for i in range(pop_size):
                if new_scales[i] < old_scales[i]:
                    X[i] = new_X[i]
                    F[i] = new_F[i]
            
            # Neighbor update
            for i in range(pop_size):
                # Check if X[i] is better for neighbor j
                # We need to compare g(X[i], w_j) vs g(X[j], w_j)
                # To avoid O(N^2) scalar calcs, we can just do it. N=50 is small.
                for j in neighbors[i]:
                    # Compute g(X[i], w_j)
                    F_i_norm = (F[i] - F_min) / (F_max - F_min + 1e-8)
                    g_i_j = np.max(weights[j] * F_i_norm)
                    
                    # Compute g(X[j], w_j)
                    F_j_norm = (F[j] - F_min) / (F_max - F_min + 1e-8)
                    g_j_j = np.max(weights[j] * F_j_norm)
                    
                    if g_i_j < g_j_j:
                        X[j] = X[i]
                        F[j] = F[i]

        # Final step: Extract Pareto front from population
        # We want to return the non-dominated points to maximize hypervolume.
        # The recorded points are all evaluations. The return value is ignored but must be valid.
        # The score is based on *all recorded vectors*.
        # So we just need to make sure we evaluated good points.
        # The return value can be the current population.
        
        return X
# EVOLVE-BLOCK-END
