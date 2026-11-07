import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        dim = self.dim
        max_calls = self.budget
        call_count = 0
        best_x = None
        best_f = float('inf')
        
        # Data storage
        X_data = []
        Y_data = []

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            X_data.append(x_clipped.copy())
            Y_data.append(val)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Coarse Exploration using Stratified Sampling
        # We use a Halton-like sequence or stratified random to cover the space efficiently.
        # In 5D, 200 points is a good balance between coverage and leaving budget for refinement.
        n_initial = 200
        
        # Generate stratified points
        # Divide each dimension into 3 intervals: [-5, -1.66], [-1.66, 1.66], [1.66, 5]
        # This creates a 3^5 = 243 grid. We can sample from this or use random stratification.
        # Let's use random stratification to avoid grid bias.
        points = []
        for i in range(n_initial):
            # For each dimension, pick a random sub-interval and a random point within it
            pt = np.zeros(dim)
            for d in range(dim):
                # 3 strata per dimension
                stratum = np.random.randint(0, 3)
                lower = -5 + stratum * (10.0 / 3.0)
                upper = lower + 10.0 / 3.0
                pt[d] = np.random.uniform(lower, upper)
            points.append(pt)
        
        # Ensure we also check the center and corners
        points.append(np.zeros(dim))
        points.append(np.full(dim, 5.0))
        points.append(np.full(dim, -5.0))
        
        # Shuffle to avoid systematic bias in evaluation order
        np.random.shuffle(points)
        
        for pt in points:
            if call_count >= max_calls or time.time() > deadline:
                break
            try:
                counted_func(pt)
            except Exception:
                break

        if best_x is None:
            return np.zeros(dim), float('inf')

        # Phase 2: Select Top K Candidates for Local Refinement
        # Sort by function value
        indices = np.argsort(Y_data)
        top_k_indices = indices[:5]  # Pick top 5
        top_k_points = [X_data[i] for i in top_k_indices]
        
        # Phase 3: Local Refinement using Nelder-Mead
        # Nelder-Mead is robust and fast. We limit iterations to conserve budget.
        # Budget remaining: 500 - 203 = ~297 calls.
        # 5 candidates * 50 calls each = 250 calls. Leaves ~47 for final polish.
        
        for pt in top_k_points:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            try:
                res = minimize(
                    counted_func,
                    pt,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 4: Final Polish on the Best Point
        if best_x is not None and call_count < max_calls and time.time() < deadline:
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
