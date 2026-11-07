"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid surrogate-guided exploration with multi-start local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')

        # Data storage for surrogate
        X_data = []
        Y_data = []

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise ValueError("Time limit exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            X_data.append(x_clipped.copy())
            Y_data.append(val)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Initial Exploration
        # Evaluate center and a set of random points to get a sense of the landscape
        initial_points = [np.zeros(dim)]
        for _ in range(15):
            initial_points.append(rng.uniform(-5, 5, size=dim))

        for pt in initial_points:
            if call_count >= max_calls or time.time() > deadline:
                break
            try:
                counted_func(pt)
            except ValueError:
                break

        if best_x is None:
            return np.zeros(dim), float('inf')

        # Phase 2: Surrogate-Guided Search using simple linear interpolation/nearest neighbor
        # Instead of RBF which can be unstable, we use a simple approach:
        # 1. Fit a simple surrogate (e.g., linear regression or just use nearest neighbors)
        # 2. Actually, let's use a different approach: 
        #    We'll use the data to guide a multi-start local search.
        #    We'll generate candidates near the best points found so far, 
        #    and also some random candidates.
        
        # Let's do a few rounds of:
        # 1. Generate candidates: some near best, some random
        # 2. Evaluate them
        # 3. Run local refinement on the best few
        
        num_rounds = 5
        candidates_per_round = 8

        for round_idx in range(num_rounds):
            if call_count >= max_calls or time.time() > deadline:
                break

            candidates = []
            
            # Add points near the current best
            if best_x is not None:
                for _ in range(3):
                    x_new = best_x + rng.normal(0, 0.5, size=dim)
                    candidates.append(np.clip(x_new, -5, 5))
            
            # Add random points
            for _ in range(candidates_per_round - 3):
                candidates.append(rng.uniform(-5, 5, size=dim))
            
            # Evaluate candidates
            for x_cand in candidates:
                if call_count >= max_calls or time.time() > deadline:
                    break
                try:
                    counted_func(x_cand)
                except ValueError:
                    break
            
            # Run local refinement on the best point found in this round
            if best_x is not None and call_count < max_calls and time.time() < deadline:
                try:
                    res = minimize(
                        counted_func,
                        best_x,
                        method='Powell',
                        options={'maxiter': 20, 'xtol': 1e-5, 'ftol': 1e-5}
                    )
                except Exception:
                    pass

        # Phase 3: Final Multi-Start Local Refinement
        # Use the best point and a few other good points to run local optimization
        # Sort by function value
        if len(X_data) > 0:
            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)
            sort_idx = np.argsort(Y_arr)
            
            # Take top 5 points
            top_k = min(5, len(sort_idx))
            top_points = X_arr[sort_idx[:top_k]]
            
            for x0 in top_points:
                if call_count >= max_calls or time.time() > deadline:
                    break
                try:
                    res = minimize(
                        counted_func,
                        x0,
                        method='Powell',
                        options={'maxiter': 30, 'xtol': 1e-6, 'ftol': 1e-6}
                    )
                except Exception:
                    break

        # Ensure best_x is valid
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count < max_calls:
                try:
                    counted_func(best_x)
                except:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
