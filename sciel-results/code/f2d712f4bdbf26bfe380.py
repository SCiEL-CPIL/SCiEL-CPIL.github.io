import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 280s safety margin out of 300s limit

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        # Phase 1: Initial sampling
        # Use 10% of budget for initialization to leave more for exploitation
        n_init = min(50, self.budget // 10)
        rng = np.random.default_rng(42)
        
        init_points = []
        init_vals = []
        
        for _ in range(n_init):
            if time.time() > deadline or self.calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_func(x)
            init_points.append(x)
            init_vals.append(f)
        
        if not init_points:
            x = np.zeros(self.dim)
            safe_func(x)
            return self.best_x, self.best_f

        # Find the best initial point
        best_idx = np.argmin(init_vals)
        x0 = init_points[best_idx].copy()

        # Phase 2: Nelder-Mead Simplex Search
        # This is more efficient for smooth valleys than coordinate descent
        # and handles the rotated landscape well without needing to guess step sizes.
        
        # Initialize simplex
        step = 0.5
        simplex = [x0.copy()]
        for i in range(self.dim):
            x = x0.copy()
            x[i] += step
            simplex.append(x)
        
        # Evaluate initial simplex
        values = []
        for x in simplex:
            if self.calls >= self.budget or time.time() > deadline:
                break
            values.append(safe_func(x))
        
        # If we didn't fill the simplex, fallback to coordinate descent
        if len(values) < self.dim + 1:
            # Simple coordinate descent fallback
            current_x = self.best_x.copy()
            current_f = self.best_f
            steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
            for step in steps:
                if self.calls >= self.budget or time.time() > deadline:
                    break
                for d in range(self.dim):
                    for direction in [1, -1]:
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        new_x = current_x.copy()
                        new_x[d] += direction * step
                        new_x = np.clip(new_x, -5, 5)
                        f_new = safe_func(new_x)
                        if f_new < current_f:
                            current_x = new_x
                            current_f = f_new
            return self.best_x, self.best_f

        # Main Nelder-Mead Loop
        while self.calls < self.budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_func(x_r)
            
            if f_r < values[0]:
                # Expansion
                x_e = centroid + 2 * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    values[-1] = f_r
            elif f_r < values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                values[-1] = f_r
            else:
                # Contraction
                x_c = centroid + 0.5 * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_func(x_c)
                if f_c < values[-1]:
                    simplex[-1] = x_c
                    values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        values[i] = safe_func(simplex[i])
            
            # Check for convergence (optional, but we run until budget is exhausted)
            # We can break if the simplex is very small
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-6:
                break

        return self.best_x, self.best_f
