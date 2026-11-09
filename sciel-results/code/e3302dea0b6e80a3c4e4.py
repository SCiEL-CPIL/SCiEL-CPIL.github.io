import numpy as np
import time

class Optimizer:
    """Deterministic coordinate-wise local search with geometric step decay for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls = 0

        def try_point(x):
            nonlocal best_x, best_f, calls
            if calls >= self.budget:
                return
            if time.time() > deadline:
                return
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Coarse grid search to find a good region
        # Use a 3^5 grid over [-4, 4] (125 points) - within budget
        grid_range = np.linspace(-4, 4, 3)
        for x0 in grid_range:
            for x1 in grid_range:
                for x2 in grid_range:
                    for x3 in grid_range:
                        for x4 in grid_range:
                            x = np.array([x0, x1, x2, x3, x4])
                            try_point(x)
                            if calls >= self.budget:
                                break
                        if calls >= self.budget:
                            break
                    if calls >= self.budget:
                        break
                if calls >= self.budget:
                    break
            if calls >= self.budget:
                break

        if best_x is None:
            # Fallback
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Coordinate-wise local refinement with geometric step decay
        # For each coordinate, try steps in decreasing magnitude
        current_x = best_x.copy()
        current_f = best_f

        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        
        # Multiple passes with decreasing step sizes
        for step in step_sizes:
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            improved = True
            while improved and calls < self.budget:
                if time.time() > deadline:
                    break
                improved = False
                for d in range(self.dim):
                    if calls >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_plus = current_x.copy()
                    x_plus[d] = min(5.0, x_plus[d] + step)
                    if x_plus[d] != current_x[d]:
                        fp = func(x_plus)
                        calls += 1
                        if fp < current_f:
                            current_x = x_plus
                            current_f = fp
                            if fp < best_f:
                                best_f = fp
                                best_x = x_plus.copy()
                            improved = True
                    
                    # Try negative step
                    x_minus = current_x.copy()
                    x_minus[d] = max(-5.0, x_minus[d] - step)
                    if x_minus[d] != current_x[d]:
                        fm = func(x_minus)
                        calls += 1
                        if fm < current_f:
                            current_x = x_minus
                            current_f = fm
                            if fm < best_f:
                                best_f = fm
                                best_x = x_minus.copy()
                            improved = True

        return best_x, best_f
