# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives).
    Strategy:
    1. Coarse grid-based decomposition (3x3x3x3 = 81 points).
    2. Identify non-dominated points.
    3. Local refinement using Nelder-Mead on diverse non-dominated points with diverse weight vectors.
    4. Deadline and budget guards.
    5. Final random sampling to fill remaining budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        max_evals = self.budget
        eval_count = 0

        all_X = []
        all_F = []

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                return np.ones(n_obj) * 10.0
            x = np.asarray(x)
            fx = problem(x)
            eval_count += 1
            all_X.append(x.copy())
            all_F.append(np.asarray(fx))
            return fx

        def check_limits():
            return (time.time() - start_time) > deadline or eval_count >= max_evals

        # Phase 1: Coarse grid exploration
        # 3x3x3x3 grid = 81 points
        grid_size = 3
        grid_points = []
        
        # Generate grid points
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1),
                            lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1),
                            lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1),
                            lo[3] + (hi[3] - lo[3]) * l / (grid_size - 1),
                        ])
                        grid_points.append(x)
        
        # Shuffle grid points for diversity
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = [grid_points[i] for i in perm]
        
        for x in grid_points:
            if check_limits():
                break
            safe_problem(x)
        
        if not all_F:
            return np.array([])

        F_array = np.array(all_F)
        X_array = np.array(all_X)
        
        # Phase 2: Identify non-dominated points
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
        
        # Phase 3: Local refinement
        remaining_budget = max_evals - eval_count
        
        if len(non_dominated_idx) > 0 and remaining_budget > 10:
            # Select top candidates based on sum of normalized objectives
            F_min = F_array.min(axis=0)
            F_max = F_array.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            F_norm = (F_array - F_min) / F_range
            weighted_scores = F_norm.sum(axis=1)
            
            # Sort non-dominated indices by their score
            nd_scores = [(weighted_scores[i], i) for i in non_dominated_idx]
            nd_scores.sort()
            
            # Determine how many candidates to refine
            # Each Nelder-Mead run with maxfev=15 uses ~15-20 evals.
            # Let's aim for more candidates with fewer evals each to cover more of the front.
            evals_per_candidate = 12
            max_candidates = max(1, min(8, remaining_budget // evals_per_candidate))
            
            candidates = [idx for score, idx in nd_scores[:max_candidates]]
            
            # Diverse weight vectors for scalarization to cover the Pareto front
            # More weight vectors for better coverage
            weight_vectors = [
                np.array([0.9, 0.05, 0.05]),
                np.array([0.05, 0.9, 0.05]),
                np.array([0.05, 0.05, 0.9]),
                np.array([0.5, 0.25, 0.25]),
                np.array([0.25, 0.5, 0.25]),
                np.array([0.25, 0.25, 0.5]),
                np.array([0.4, 0.3, 0.3]),
                np.array([0.3, 0.4, 0.3]),
                np.array([0.3, 0.3, 0.4]),
                np.array([1.0/3, 1.0/3, 1.0/3]),
                np.array([0.7, 0.15, 0.15]),
                np.array([0.15, 0.7, 0.15]),
                np.array([0.15, 0.15, 0.7]),
                np.array([0.6, 0.2, 0.2]),
                np.array([0.2, 0.6, 0.2]),
                np.array([0.2, 0.2, 0.6]),
            ]
            
            for i, idx in enumerate(candidates):
                if check_limits():
                    break
                
                x0 = X_array[idx].copy()
                w = weight_vectors[i % len(weight_vectors)]
                
                def bounded_objective(x, w=w):
                    x_clipped = np.clip(x, lo, hi)
                    fx = safe_problem(x_clipped)
                    return np.dot(w, fx)
                
                try:
                    result = minimize(
                        bounded_objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 8, 'xatol': 1e-4, 'fatol': 1e-6, 'maxfev': evals_per_candidate}
                    )
                except Exception:
                    pass
        
        # Phase 4: Fill remaining budget with random sampling
        # Use a fixed seed for reproducibility within the run
        rng2 = np.random.default_rng(0)
        while not check_limits():
            x = lo + (hi - lo) * rng2.random(dim)
            safe_problem(x)
        
        return np.array(all_X)
# EVOLVE-BLOCK-END
