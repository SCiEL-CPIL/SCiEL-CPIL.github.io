# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives).
    Strategy:
    1. Coarse grid-based decomposition to systematically explore the domain.
    2. Identify promising regions based on non-domination in the coarse grid.
    3. Local refinement using Nelder-Mead on promising points.
    4. Deadline guard to ensure we return before 330s wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        evaluated = 0
        max_evals = self.budget

        # Store all evaluated points and their objective values
        all_X = []
        all_F = []

        def check_deadline():
            return (time.time() - start_time) > deadline or evaluated >= max_evals

        def evaluate(x):
            nonlocal evaluated
            if evaluated >= max_evals:
                return None
            fx = problem(x)
            evaluated += 1
            all_X.append(x)
            all_F.append(fx)
            return fx

        # Phase 1: Coarse grid exploration
        # Use a 4x4x4x4 grid = 256 points, but we only have 200 budget
        # So use a 3x3x3x3 grid = 81 points for coarse exploration
        # Then use remaining budget for local refinement
        grid_size = 3
        total_grid = grid_size ** dim  # 81 for dim=4
        
        # Reserve some budget for refinement
        coarse_budget = min(total_grid, max_evals // 2)
        
        # Generate grid points
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1) if grid_size > 1 else (lo[0] + hi[0]) / 2,
                            lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1) if grid_size > 1 else (lo[1] + hi[1]) / 2,
                            lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1) if grid_size > 1 else (lo[2] + hi[2]) / 2,
                            lo[3] + (hi[3] - lo[3]) * l / (grid_size - 1) if grid_size > 1 else (lo[3] + hi[3]) / 2,
                        ])
                        grid_points.append(x)
        
        # Shuffle grid points for diversity
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = [grid_points[i] for i in perm]
        
        # Evaluate coarse grid
        for x in grid_points:
            if check_deadline():
                break
            evaluate(x)
        
        # Phase 2: Identify promising points (non-dominated in coarse grid)
        F_array = np.array(all_F)
        X_array = np.array(all_X)
        
        # Find non-dominated points
        non_dominated_idx = []
        for i in range(len(F_array)):
            dominated = False
            for j in range(len(F_array)):
                if i == j:
                    continue
                if np.all(F_array[j] <= F_array[i]) and np.any(F_array[j] < F_array[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated_idx.append(i)
        
        # Phase 3: Local refinement using Nelder-Mead on promising points
        # For each non-dominated point, try to improve it locally
        from scipy.optimize import minimize
        
        # Select top promising points for refinement
        # Use a weighted sum approach to rank them
        if len(F_array) > 0:
            # Normalize objectives
            F_min = F_array.min(axis=0)
            F_max = F_array.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            F_norm = (F_array - F_min) / F_range
            weighted_scores = F_norm.sum(axis=1)
            
            # Sort by weighted score (lower is better)
            sorted_idx = np.argsort(weighted_scores)
            
            # Take top candidates for refinement
            n_refine = min(20, len(sorted_idx))
            candidates = sorted_idx[:n_refine]
        
        else:
            candidates = []
        
        # Local refinement
        for idx in candidates:
            if check_deadline():
                break
            
            x0 = X_array[idx].copy()
            
            # Try Nelder-Mead with a few different weight vectors
            # Weight vectors for 3 objectives
            weight_vectors = [
                np.array([0.8, 0.1, 0.1]),
                np.array([0.1, 0.8, 0.1]),
                np.array([0.1, 0.1, 0.8]),
                np.array([0.5, 0.25, 0.25]),
                np.array([0.25, 0.5, 0.25]),
                np.array([0.25, 0.25, 0.5]),
            ]
            
            for w in weight_vectors:
                if check_deadline():
                    break
                
                def objective(x):
                    fx = problem(x)
                    return np.dot(w, fx)
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    
                    # Evaluate the result
                    if evaluated < max_evals:
                        fx = problem(result.x)
                        evaluated += 1
                        all_X.append(result.x)
                        all_F.append(fx)
                        
                        # Update x0 for next weight vector if improved
                        if np.dot(w, fx) < np.dot(w, F_array[idx]):
                            x0 = result.x.copy()
                            F_array[idx] = fx
                            X_array[idx] = result.x
                except Exception:
                    pass
        
        # Phase 4: Additional random sampling if budget remains
        while not check_deadline():
            x = lo + (hi - lo) * np.random.default_rng(0).random(dim)
            evaluate(x)
        
        return X_array
# EVOLVE-BLOCK-END
