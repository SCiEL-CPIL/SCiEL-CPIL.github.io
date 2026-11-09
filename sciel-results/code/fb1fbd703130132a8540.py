import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # State
        X = []
        F = []
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            fx = np.array(problem(x))
            X.append(x)
            F.append(fx)
            eval_count += 1
            return fx
            
        def check_time():
            return time.time() < deadline - 5.0

        # Helper: Check if a point is non-dominated by the current set F
        def is_non_dominated(f_new, F_current):
            for f_old in F_current:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        # Helper: Local Search (Adaptive Step Size Perturbation)
        # Exploiting local smoothness of WFG4 with adaptive step size
        def local_search(x, f, max_steps=8):
            nonlocal eval_count
            current_x = x.copy()
            current_f = f
            step_size = 0.15 * (hi - lo)
            for step in range(max_steps):
                if not check_time() or eval_count >= budget - 10: # Reserve budget
                    break
                # Generate a neighbor with adaptive step size
                neighbor = current_x + np.random.randn(dim) * step_size
                neighbor = np.clip(neighbor, lo, hi)
                
                f_neighbor = eval_point(neighbor)
                if f_neighbor is None:
                    break
                
                # Accept if non-dominated by the current point
                if is_non_dominated(f_neighbor, [current_f]):
                    current_x = neighbor
                    current_f = f_neighbor
                    step_size *= 0.8  # Shrink step on improvement
                else:
                    step_size *= 0.7  # Shrink step on rejection
            return current_x, current_f

        # 1. Grid-based decomposition with LHS per cell for maximal spatial coverage
        # Partition the 12-dimensional space into a grid of hyper-rectangular cells
        # Use a 2x2x2x2x2x2 = 64 cell grid (2^6) to balance coverage and computation
        # Each cell gets one LHS sample, ensuring uniform coverage of the Pareto front manifold
        
        # Define grid resolution per dimension (total cells = product of resolutions)
        # Use 2 cells in first 6 dims, 1 in remaining 6 to keep total at 64 cells
        grid_res = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        n_cells = int(np.prod(grid_res))  # 64 cells
        
        # Generate one sample per cell using LHS within each cell
        X_init = np.zeros((n_cells, dim))
        for i in range(dim):
            res_i = grid_res[i]
            cell_width = (hi[i] - lo[i]) / res_i
            for cell_idx in range(n_cells):
                # Determine which cell this sample belongs to in dimension i
                # Map linear index to multi-dimensional cell index
                cell_coord = cell_idx
                for j in range(i, -1, -1):
                    cell_coord = cell_coord // grid_res[j]
                # Sample within the cell using uniform random
                X_init[cell_idx, i] = lo[i] + cell_coord * cell_width + cell_width * np.random.rand()
        
        # Evaluate and refine each cell's representative point
        for i in range(n_cells):
            if not check_time() or eval_count >= budget - 20:
                break
            fx = eval_point(X_init[i])
            if fx is None:
                break
            # Perform local search to refine within the cell
            X_init[i], fx = local_search(X_init[i], fx, max_steps=4)
            
        # Helper: Get non-dominated indices
        def get_non_dominated_indices(F_list):
            nd_indices = []
            for i, f_i in enumerate(F_list):
                dominated = False
                for j, f_j in enumerate(F_list):
                    if i == j: continue
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        # 2. Expansion Phase: Refine non-dominated points with additional LHS samples
        if eval_count < budget - 10:
            nd_indices = get_non_dominated_indices(F)
            if len(nd_indices) > 0:
                # Generate offspring from non-dominated points with finer LHS
                offspring_to_eval = []
                for idx in nd_indices[:10]:  # Limit to top 10 to control budget
                    parent_x = X[idx]
                    # Generate 2 children with moderate perturbation
                    for _ in range(2):
                        if eval_count >= budget - 10:
                            break
                        child_x = parent_x + np.random.randn(dim) * 0.08 * (hi - lo)
                        child_x = np.clip(child_x, lo, hi)
                        offspring_to_eval.append(child_x)
                        
                # Evaluate and local search offspring
                for ox in offspring_to_eval:
                    if not check_time() or eval_count >= budget - 5:
                        break
                    of = eval_point(ox)
                    if of is None:
                        break
                    # Local search with moderate steps
                    ox, of = local_search(ox, of, max_steps=3)

        # 3. Final Polish Phase: Very fine-tuning of best points
        if eval_count < budget - 5 and check_time():
            nd_indices = get_non_dominated_indices(F)
            if len(nd_indices) > 0:
                # Pick top 5 non-dominated points for final polish
                top_points = nd_indices[:min(5, len(nd_indices))]
                for idx in top_points:
                    if eval_count >= budget - 2 or not check_time():
                        break
                    # Very small step local search
                    x = X[idx].copy()
                    f = F[idx]
                    for _ in range(3):
                        if eval_count >= budget - 1 or not check_time():
                            break
                        neighbor = x + np.random.randn(dim) * 0.02 * (hi - lo)
                        neighbor = np.clip(neighbor, lo, hi)
                        fn = eval_point(neighbor)
                        if fn is None:
                            break
                        if is_non_dominated(fn, [f]):
                            x = neighbor
                            f = fn

        return X
