import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # WFG9 specific heuristic:
        # The first 3 variables (x1, x2, x3) primarily define the shape of the Pareto front.
        # The remaining variables (x4...x12) are often bias/positioning variables.
        # A common strategy is to fix bias variables to their midpoints to ensure 
        # we are on or very close to the Pareto front, and search the shape variables.
        
        # 1. Initialize with a grid on the first 3 variables
        # Budget is 200. Let's use ~100 for grid, ~100 for refinement.
        # Grid: 10x10x1 = 100 points? Or 5x5x4 = 100?
        # Let's try a 10x10 grid on x0, x1 and fix x2 to midpoint initially, 
        # or 5x5x4 grid.
        # Actually, WFG9 front is 2D for 3 objectives? No, WFG9 is 3-objective, so front is 2D surface.
        # Variables 1 and 2 usually control the two degrees of freedom of the front.
        # Variable 3 might also be a shape variable.
        
        # Let's use a grid of 10x10 on the first two variables, and 2 values for the 3rd?
        # 10 * 10 * 2 = 200. This uses the whole budget for grid.
        # Better: 8 * 8 * 3 = 192. Leaves 8 for refinement.
        # Or: 10 * 10 * 1 = 100. Leaves 100 for refinement.
        
        # Let's go with a moderate grid to ensure coverage, then refine.
        n_grid_0 = 10
        n_grid_1 = 10
        n_grid_2 = 2
        
        grid_size = n_grid_0 * n_grid_1 * n_grid_2
        
        # Create base vector with midpoints for bias variables (index 3 to dim-1)
        base_x = (lo + hi) / 2.0
        
        X = np.zeros((grid_size, dim))
        F = np.zeros((grid_size, n_obj))
        
        idx = 0
        g0 = np.linspace(lo[0], hi[0], n_grid_0)
        g1 = np.linspace(lo[1], hi[1], n_grid_1)
        g2 = np.linspace(lo[2], hi[2], n_grid_2)
        
        for x0 in g0:
            for x1 in g1:
                for x2 in g2:
                    if idx >= grid_size:
                        break
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    X[idx] = x
                    F[idx] = problem(x)
                    idx += 1
                    if time.time() > deadline:
                        return X
        
        used = grid_size
        
        # 2. Refinement Phase
        # Use the remaining budget to perform local search.
        # We will maintain a population of size pop_size (e.g., 50) from the best grid points.
        # But simpler: just keep the grid points and replace the worst ones with mutated children.
        
        pop_size = min(50, grid_size)
        
        # Sort initial grid points by a simple metric to keep the best ones?
        # Or just keep them all if grid_size <= pop_size.
        # If grid_size > pop_size, we need to select.
        # Let's keep the non-dominated points first.
        
        def get_non_dominated_indices(F_pop):
            n = len(F_pop)
            if n == 0: return np.array([], dtype=int)
            # Vectorized dominance check for small n
            # F_pop shape (n, n_obj)
            # i dominates j if F[i] <= F[j] and F[i] != F[j]
            # We want indices i such that no j dominates i.
            # le[i,j] is True if F[i] <= F[j]
            le = np.all(F_pop[:, np.newaxis, :] <= F_pop[np.newaxis, :, :], axis=2)
            lt = np.any(F_pop[:, np.newaxis, :] < F_pop[np.newaxis, :, :], axis=2)
            dom_matrix = le & lt
            np.fill_diagonal(dom_matrix, False)
            # i is dominated if any j dominates i -> row i has any True? No.
            # dom_matrix[i,j] is True if i dominates j.
            # So j is dominated if any i dominates j -> column j has any True.
            dominated_mask = np.any(dom_matrix, axis=0)
            return np.where(~dominated_mask)[0]

        # Initial population for refinement: take the non-dominated points from the grid
        nd_idx = get_non_dominated_indices(F)
        if len(nd_idx) > pop_size:
            # If too many non-dominated, pick a subset randomly or by crowding
            # For simplicity, random subset
            keep_idx = np.random.choice(nd_idx, pop_size, replace=False)
        else:
            keep_idx = nd_idx.copy()
            # Fill remaining slots with random points from the rest if needed to maintain pop_size?
            # Or just work with smaller pop_size.
            if len(keep_idx) < pop_size:
                remaining_idx = np.setdiff1d(np.arange(grid_size), keep_idx)
                if len(remaining_idx) > 0:
                    extra = np.random.choice(remaining_idx, min(pop_size - len(keep_idx), len(remaining_idx)), replace=False)
                    keep_idx = np.concatenate([keep_idx, extra])
        
        X_pop = X[keep_idx]
        F_pop = F[keep_idx]
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run
        
        # Mutation parameters
        eta = 20.0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Select parent from non-dominated set in current population
            nd_idx_pop = get_non_dominated_indices(F_pop)
            if len(nd_idx_pop) == 0:
                p_idx = rng.integers(len(F_pop))
            else:
                p_idx = nd_idx_pop[rng.integers(len(nd_idx_pop))]
            
            x_parent = X_pop[p_idx]
            
            # Polynomial mutation on all variables, but with smaller step for bias variables?
            # Or just standard polynomial mutation.
            x_child = x_parent.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    r = rng.random()
                    if r <= 0.5:
                        delta = (2.0 * r) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))
                    x_child[i] = x_parent[i] + delta * (hi[i] - lo[i])
                    x_child[i] = np.clip(x_child[i], lo[i], hi[i])
            
            f_child = problem(x_child)
            used += 1
            
            # Replacement Strategy
            # Add child to temp population
            X_temp = np.vstack([X_pop, x_child.reshape(1, -1)])
            F_temp = np.vstack([F_pop, f_child.reshape(1, -1)])
            
            # Find non-dominated in temp
            nd_temp = get_non_dominated_indices(F_temp)
            
            # If number of non-dominated is <= pop_size, keep all non-dominated
            # and fill rest with best dominated?
            # Simpler: Keep top pop_size based on non-dominance and then sum of objectives.
            
            # Calculate rank (0 for non-dominated, 1 for dominated)
            ranks = np.ones(len(F_temp), dtype=int)
            ranks[nd_temp] = 0
            
            # Sort by rank, then by sum of objectives
            sums = F_temp.sum(axis=1)
            sort_idx = np.lexsort((sums, ranks))
            
            # Keep the first pop_size
            keep_temp = sort_idx[:pop_size]
            
            X_pop = X_temp[keep_temp]
            F_pop = F_temp[keep_temp]
            
            # Small sleep to avoid CPU spin? No, numpy is fast enough.
            
        return X_pop
