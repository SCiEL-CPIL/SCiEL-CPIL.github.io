# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Grid-Based Decomposition with Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 280  # 50s safety margin for 330s limit

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        # Helper to add a point if valid
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                # Clip to bounds just in case, though problem should do it
                x_clipped = np.clip(x, lo, hi)
                # Check if already added (simple check for exact match or close)
                # For efficiency in small lists, linear scan is fine
                for i in range(len(X_list)):
                    if np.allclose(X_list[i], x_clipped, atol=1e-8):
                        return
                X_list.append(x_clipped)
                F_list.append(f)
        
        # Phase 1: Grid-Based Coarse Search
        # With dim=4, a grid of size 3 per dimension gives 3^4 = 81 points.
        # Budget is 200. This leaves 119 for refinement.
        
        grid_size = 3
        n_grid = grid_size ** self.dim
        
        if n_grid > self.budget // 2:
             # Fallback to smaller grid if budget is tight
             grid_size = 2
             n_grid = grid_size ** self.dim

        # Generate grid points
        grid_axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(self.dim)]
        grid_points = np.array(np.meshgrid(*grid_axes)).T.reshape(-1, self.dim)
        
        # Evaluate grid points
        for x in grid_points:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
        
        if not F_list:
            # Fallback if grid failed completely
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0 is not None: add_point(x0, f0)
            return X_list if X_list else [x0]

        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Identify Promising Regions for Local Refinement
        # We want to refine points that are non-dominated or close to the Pareto front.
        
        # Normalize objectives to [0, 1] based on current min/max for scalarization
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        F_norm = (F - f_min) / f_range
        
        # Sum of normalized objectives (lower is better)
        sums = np.sum(F_norm, axis=1)
        
        # Select top candidates for refinement.
        # We have ~100-120 evals left. Nelder-Mead in 4D might take ~20-50 evals per run.
        # Let's try to run 4-6 local optimizations.
        n_refine = 5
        
        # Fast non-dominated sort for selection
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i+1, n):
                if is_dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_dominated[j] = True
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        non_dom_indices = np.where(~is_dominated)[0]
        
        if len(non_dom_indices) == 0:
            # Fallback to best sum
            non_dom_indices = np.argsort(sums)[:n_refine]
        else:
            # If too many non-dominated, sample or pick best by sum among them
            if len(non_dom_indices) > n_refine * 2:
                 # Pick best n_refine * 2 by sum, then sample n_refine
                 sums_non_dom = sums[non_dom_indices]
                 best_non_dom = non_dom_indices[np.argsort(sums_non_dom)]
                 candidates = best_non_dom[:n_refine * 2]
                 # Random selection to ensure diversity in start points
                 rng = np.random.default_rng(42)
                 non_dom_indices = rng.choice(candidates, size=n_refine, replace=False)
            else:
                 # Use all non-dominated if few
                 non_dom_indices = non_dom_indices[:n_refine]
        
        # Phase 3: Local Refinement using Nelder-Mead
        # Use a diverse set of weight vectors to cover the Pareto front
        # Generate weight vectors: vertices, edge midpoints, and center
        weight_vectors = []
        
        # Vertices
        weight_vectors.append([1.0, 0.0, 0.0])
        weight_vectors.append([0.0, 1.0, 0.0])
        weight_vectors.append([0.0, 0.0, 1.0])
        
        # Edge midpoints
        weight_vectors.append([0.5, 0.5, 0.0])
        weight_vectors.append([0.5, 0.0, 0.5])
        weight_vectors.append([0.0, 0.5, 0.5])
        
        # Center
        weight_vectors.append([1.0/3, 1.0/3, 1.0/3])
        
        # Add random weights if we have more refinement slots
        while len(weight_vectors) < n_refine:
            w = np.random.rand(self.n_obj)
            w = w / np.sum(w)
            weight_vectors.append(w.tolist())
        
        weight_vectors = np.array(weight_vectors[:n_refine])
        
        for i, idx in enumerate(non_dom_indices):
            if self.eval_count >= self.budget or self._check_time():
                break
            
            x_start = X[idx].copy()
            w = weight_vectors[i % len(weight_vectors)]
            
            def objective(x, w=w):
                f = self._evaluate(x, problem)
                if f is None:
                    return 1e6
                return np.dot(w, f)
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'maxfev': 30, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                # Evaluate the final result to get the vector
                if res.x is not None:
                    f_final = self._evaluate(res.x, problem)
                    if f_final is not None:
                        add_point(res.x, f_final)
                        
            except Exception:
                pass
        
        return X_list
# EVOLVE-BLOCK-END
