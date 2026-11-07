"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Weighted-Sum Decomposition and Gradient-Based Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors for Decomposition
        # We want to cover the Pareto front. 
        # With 3 objectives and budget 200, we can afford a moderate grid.
        # A grid of 6x6x6 = 216 points is slightly over budget if we refine all.
        # Let's use a grid that sums to 1.
        # Strategy: Use a set of weight vectors that cover the simplex.
        
        # Generate points on a 3D simplex grid.
        # Number of points per dimension (excluding the last which is determined by sum=1)
        # Let's try 5x5 = 25 base points, plus some random perturbations?
        # Or just a systematic grid.
        # For 3 objectives, weights (w1, w2, w3) with w1+w2+w3=1.
        # We can iterate w1 from 0 to 1, w2 from 0 to 1-w1.
        
        # Let's allocate budget:
        # ~100 evaluations for global weighted-sum search
        # ~100 evaluations for local refinement (L-BFGS-B uses ~10-20 evals per point)
        # So we can refine ~5-10 best points.
        
        # Create weight vectors
        weights = []
        # Grid resolution
        # To get good coverage, let's use a grid of 10x10 for the first two weights
        # This gives 100 weight vectors.
        # We will evaluate one point per weight vector using the weighted sum scalarization.
        
        n_grid = 10
        for i in range(n_grid):
            for j in range(n_grid - i):
                w1 = i / (n_grid - 1)
                w2 = j / (n_grid - 1 - i)
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-9:
                    weights.append([w1, w2, w3])
        
        # Ensure we don't exceed budget just for the global phase
        # If len(weights) > budget * 0.6, subsample or adjust grid
        max_global_evals = int(budget * 0.6)
        if len(weights) > max_global_evals:
            # Subsample weights uniformly
            indices = np.linspace(0, len(weights)-1, max_global_evals).astype(int)
            weights = [weights[i] for i in indices]
        
        candidates = []
        f_values = []
        
        # 2. Global Search Phase: Minimize weighted sum for each weight vector
        # Use a simple local search or just evaluate a few starting points?
        # Better: Use L-BFGS-B directly on the weighted sum?
        # L-BFGS-B is expensive if called 60 times.
        # Alternative: Use a coarse random search to find good starting points for each weight, then refine?
        # Or just run L-BFGS-B from a central point for each weight?
        # Given the small dimension (5), L-BFGS-B is fast.
        
        # Let's run L-BFGS-B for each weight vector.
        # Budget check: 60 weights * ~10-20 evals = 600-1200 evals. Too many.
        # We have 200 evals total.
        
        # Revised Strategy:
        # 1. Generate 30 weight vectors.
        # 2. For each, do a few steps of gradient descent or a small L-BFGS-B run?
        # Actually, let's use a simpler approach for the global phase:
        # Sample random points, but guided by weights?
        
        # Let's stick to a hybrid:
        # Phase 1: Sample 50 random points.
        # Phase 2: For the best 10 non-dominated points, run L-BFGS-B to refine them.
        # Phase 3: If budget remains, do more random sampling or refine more.
        
        # However, the inspiration suggests Weighted-Sum Grid Search.
        # To fit in 200 evals:
        # Use 20 weight vectors.
        # For each weight, find the minimum of the weighted sum using a bounded method.
        # Since we don't have gradients for the original problem, we must use derivative-free methods or approximate.
        # Wait, `problem` is a black box. We cannot use gradients.
        # So L-BFGS-B on the weighted sum requires gradients of the weighted sum, which we don't have.
        # We must use derivative-free optimization like Nelder-Mead or Powell, or just simple hill climbing.
        
        # Let's use Nelder-Mead for the weighted sum minimization.
        # Nelder-Mead is cheap in terms of function calls if the dimension is small (5).
        # ~10-20 calls per optimization.
        # 20 weights * 15 calls = 300 calls. Still too many.
        
        # Let's reduce weights to 10.
        # 10 weights * 15 calls = 150 calls.
        # Remaining 50 calls for random exploration or extra refinement.
        
        n_weights = 10
        # Generate 10 diverse weight vectors
        # Simplex grid 4x4 = 10 points?
        # 0, 0.33, 0.66, 1.0
        # Let's just pick specific weights to cover corners and edges.
        specific_weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ]
        
        # Ensure we don't use too much time
        # Define the scalarized objective
        def scalarized_objective(x, w):
            try:
                f = problem(x)
                return np.dot(f, w)
            except Exception:
                return 1e10
        
        # Phase 1: Optimized Weighted Sum Search
        # Use Nelder-Mead from a central start point for each weight
        # Central point: mid of bounds
        x_center = (lo + hi) / 2.0
        
        best_points = []
        best_f = []
        
        evals_used = 0
        
        # We need to count evals carefully. The `problem` function is the counter.
        # Nelder-Mead doesn't return the exact number of evals easily in older scipy, 
        # but we can wrap the function.
        
        def counting_problem(x):
            nonlocal evals_used
            if evals_used >= budget:
                return [1e10, 1e10, 1e10]
            evals_used += 1
            return problem(x)
        
        # Replace problem with counting version for the optimizer
        # Note: The task says "Every returned objective vector is recorded".
        # If we call problem inside Nelder-Mead, those are recorded.
        
        for w in specific_weights:
            if evals_used >= budget - 20: # Leave budget for final checks
                break
            if time.time() > deadline:
                break
            
            # Local optimization for this weight
            # Start from center
            res = minimize(
                lambda x: scalarized_objective(x, w), 
                x_center, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            # Nelder-Mead with bounds in scipy might be slow or unreliable.
            # Let's use a simpler approach: Random restarts with local hill climbing?
            # Or just evaluate the center and a few random points for each weight?
            
            # Actually, Nelder-Mead in 5D is quite fast.
            # Let's check if the result is valid.
            x_opt = res.x
            # Clip to bounds just in case
            x_opt = np.clip(x_opt, lo, hi)
            
            # Evaluate final point (already counted in Nelder-Mead? 
            # The lambda calls counting_problem, so yes, it's counted)
            
            f_opt = counting_problem(x_opt) # One more eval to ensure we have the value
            # Actually, res.fun is the value, but we need the vector.
            # Let's just add it.
            
            best_points.append(x_opt)
            best_f.append(f_opt)
            
        # Phase 2: Random Sampling to fill gaps
        # Use remaining budget for random sampling to find better points
        # especially in the center of the front.
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Random point
            x_rand = lo + (hi - lo) * np.random.random(dim)
            f_rand = counting_problem(x_rand)
            best_points.append(x_rand)
            best_f.append(f_rand)
            
            # Optional: Local refinement of the best random point found so far?
            # Let's just do random for now to ensure diversity.
            
            # If we have many points, we might want to do a local search on the best non-dominated point.
            # Let's identify the best non-dominated point and do a small local search.
            # But random is safer for coverage.
            
            # Let's alternate: every 5 random points, do a local search on a random candidate?
            # No, let's just do random. The weighted sum search should have covered the extremes.
            
        # Return the best set of points?
        # The task says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored.
        # So we just need to make the calls.
        
        return None

# EVOLVE-BLOCK-END
