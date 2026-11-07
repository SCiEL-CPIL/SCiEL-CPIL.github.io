"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using multi-start Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._ref = np.array([3000.0, 0.06])
        self._eval_count = 0
        self._deadline = None
        self._problem = None
        self._lo = None
        self._hi = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        self._deadline = start_time + 320.0  # Safety margin before 330s kill
        self._problem = problem
        self._lo = np.array([b[0] for b in bounds])
        self._hi = np.array([b[1] for b in bounds])
        self._eval_count = 0

        dim = self.dim
        lo = self._lo
        hi = self._hi
        budget = self.budget

        # 1. Generate diverse initial points
        # Use a stratified sampling approach for better coverage
        n_starts = min(50, budget // 10)  # Use up to 50 starts, ensuring enough budget for optimization
        if n_starts < 5:
            n_starts = 5
            
        # Ensure we don't exceed budget with initial points
        n_starts = min(n_starts, budget // 4)
        
        rng = np.random.default_rng(42)
        
        # Create a grid of starting points
        # For 4D, we can create a 2x2x2x2 grid = 16 points, or similar
        # Let's use a combination of random and grid points
        starts = []
        
        # Add center
        starts.append((lo + hi) / 2)
        
        # Add corners
        for i in range(2**dim):
            corner = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            # Inset slightly to avoid boundary issues if needed, but bounds are handled by clip
            starts.append(corner)
            
        # Add random points to fill up to n_starts
        while len(starts) < n_starts:
            pt = lo + (hi - lo) * rng.random(dim)
            starts.append(pt)
            
        starts = np.array(starts[:n_starts])

        # Evaluate initial points
        all_points = []
        for x in starts:
            if self._eval_count >= budget or time.time() > self._deadline:
                break
            f = self._evaluate(x)
            all_points.append(f)
            
        # 2. Local Search Phase
        # Allocate remaining budget to local search
        # Each Nelder-Mead run consumes some evaluations.
        # We will run local search on the best candidates from the initial set.
        
        # Sort initial points by sum of objectives (simple heuristic for "goodness")
        scores = np.sum(all_points, axis=1)
        order = np.argsort(scores)
        
        # Select top candidates for local refinement
        # We want to spread the search, so maybe don't just take the top 1, but top few?
        # Or take points that are non-dominated among the initial set?
        # Let's take the non-dominated set from the initial evaluations.
        front_indices = self._non_dominated_indices(np.array(all_points))
        
        # If there are too many, limit to ensure we have budget
        max_local_searches = min(len(front_indices), (budget - self._eval_count) // 20)
        if max_local_searches == 0:
            max_local_searches = 1
            
        # Prioritize local searches on points that are "spread out"
        # For simplicity, just take the first few from the sorted list of non-dominated
        # Or better, take a subset of the non-dominated front
        selected_indices = front_indices[:max_local_searches]
        
        for idx in selected_indices:
            if self._eval_count >= budget or time.time() > self._deadline:
                break
                
            x0 = starts[idx]
            f0 = all_points[idx]
            
            # Run Nelder-Mead
            # Objective function for scipy
            def obj(x):
                if self._eval_count >= budget or time.time() > self._deadline:
                    # Return a very large value to stop optimization
                    return [1e10, 1e10]
                return self._evaluate(x)
                
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                # The result is recorded by _evaluate calls inside obj
            except Exception:
                pass

        # Return value is ignored, but we return something just in case
        return None

    def _evaluate(self, x):
        """Evaluate a point, tracking budget and time."""
        if self._eval_count >= self.budget:
            raise RuntimeError("Budget exceeded")
        if time.time() > self._deadline:
            raise TimeoutError("Time limit exceeded")
            
        # Clip to bounds
        x = np.clip(x, self._lo, self._hi)
        
        # Call problem
        f = self._problem(x)
        
        self._eval_count += 1
        return f

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (N, n_obj)."""
        n = len(F)
        is_dom = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dom[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if F[j] <= F[i] and F[j] != F[i]
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom[i] = True
                    break
        return np.where(~is_dom)[0]


# EVOLVE-BLOCK-END
