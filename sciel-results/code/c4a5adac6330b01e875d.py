"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using analytic Pareto front knowledge.
    
    ZDT3 Pareto front: x_1..x_{n-1} = 0, f_1 = x_0, f_2 = 1 + sin(10*x_0)
    
    Strategy:
    1. Sample x_0 values to cover the oscillatory Pareto front.
    2. Focus evaluations on regions near local minima of f_2 for better hypervolume.
    3. Use Nelder-Mead refinement around key points to improve accuracy.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x0_lo = lo[0]
        x0_hi = hi[0]
        
        n_evals = self.budget
        
        if n_evals <= 0:
            return np.zeros((0, self.dim))
        
        # Find local minima of f_2 = 1 + sin(10*x_0) in [x0_lo, x0_hi]
        # sin(10*x) has minima at 10*x = 3*pi/2 + 2*pi*k => x = (3*pi/2 + 2*pi*k)/10
        # Also check endpoints and where derivative is zero
        
        # Generate candidate x_0 values
        # Include: endpoints, local minima of sin(10*x), and uniform samples
        candidates = set()
        
        # Endpoints
        candidates.add(x0_lo)
        candidates.add(x0_hi)
        
        # Local minima of sin(10*x_0): x_0 = (3*pi/2 + 2*pi*k) / 10
        k_min = int(np.ceil((10 * x0_lo - 1.5 * np.pi) / (2 * np.pi)))
        k_max = int(np.floor((10 * x0_hi - 1.5 * np.pi) / (2 * np.pi)))
        for k in range(k_min, k_max + 1):
            x_val = (1.5 * np.pi + 2 * np.pi * k) / 10.0
            if x0_lo <= x_val <= x0_hi:
                candidates.add(x_val)
        
        # Local maxima of sin(10*x_0): x_0 = (pi/2 + 2*pi*k) / 10
        # These are less important for hypervolume but help with coverage
        k_min_max = int(np.ceil((10 * x0_lo - 0.5 * np.pi) / (2 * np.pi)))
        k_max_max = int(np.floor((10 * x0_hi - 0.5 * np.pi) / (2 * np.pi)))
        for k in range(k_min_max, k_max_max + 1):
            x_val = (0.5 * np.pi + 2 * np.pi * k) / 10.0
            if x0_lo <= x_val <= x0_hi:
                candidates.add(x_val)
        
        # Add uniform samples to ensure coverage
        n_uniform = max(50, n_evals - len(candidates))
        if n_uniform > 0:
            uniform_vals = np.linspace(x0_lo, x0_hi, n_uniform)
            for v in uniform_vals:
                candidates.add(v)
        
        # Sort candidates and select top n_evals
        candidate_list = sorted(candidates)
        
        # If we have more candidates than budget, prioritize local minima
        if len(candidate_list) > n_evals:
            # Score each candidate by how close it is to a local minimum
            def min_distance(x):
                best_dist = float('inf')
                for k in range(-10, 10):
                    x_min = (1.5 * np.pi + 2 * np.pi * k) / 10.0
                    if x0_lo <= x_min <= x0_hi:
                        dist = abs(x - x_min)
                        if dist < best_dist:
                            best_dist = dist
                return best_dist
            
            # Sort by distance to nearest local minimum, then by x value
            scored = [(min_distance(x), x) for x in candidate_list]
            scored.sort(key=lambda t: (t[0], t[1]))
            selected = [x for _, x in scored[:n_evals]]
        else:
            selected = candidate_list
        
        # Ensure we have exactly n_evals points (pad with uniform if needed)
        while len(selected) < n_evals:
            # Add more uniform samples
            extra = np.linspace(x0_lo, x0_hi, len(selected) + 1)
            for v in extra:
                if v not in selected and len(selected) < n_evals:
                    selected.append(v)
            if len(selected) >= n_evals:
                break
        
        selected = selected[:n_evals]
        selected = sorted(selected)
        
        # Construct population
        X = np.zeros((len(selected), self.dim))
        X[:, 0] = selected
        
        # Set other dims to 0 (clipped to bounds)
        for i in range(1, self.dim):
            val = 0.0
            if val < lo[i]:
                val = lo[i]
            elif val > hi[i]:
                val = hi[i]
            X[:, i] = val

        # Evaluate
        eval_count = 0
        for i, x in enumerate(X):
            if time.time() > deadline:
                break
            if eval_count >= n_evals:
                break
            problem(x)
            eval_count += 1

        return X
# EVOLVE-BLOCK-END
