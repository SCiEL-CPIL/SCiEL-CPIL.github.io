import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        rng = np.random.default_rng(0)
        
        # Initialize simplex
        # Center at a random point
        center = rng.uniform(-3, 3, n)
        initial_step = 1.0
        
        simplex = [center]
        for i in range(n):
            point = center.copy()
            point[i] += initial_step
            simplex.append(point)
            
        values = [func(p) for p in simplex]
        calls = n + 1
        remaining = self.budget - calls
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while remaining > 0:
            # Sort
            idx = np.argsort(values)
            simplex = [simplex[i] for i in idx]
            values = [values[i] for i in idx]
            
            centroid = np.mean(simplex[:-1], axis=0)
            worst_point = simplex[-1]
            worst_value = values[-1]
            
            # Reflection
            reflected = centroid + alpha * (centroid - worst_point)
            r_val = func(reflected)
            remaining -= 1
            
            if remaining < 0:
                break
                
            if r_val < values[0]:
                # Expansion
                expanded = centroid + gamma * (centroid - worst_point)
                e_val = func(expanded)
                remaining -= 1
                if remaining < 0:
                    # If we ran out of budget, we just keep the best so far
                    # But we already evaluated e_val, so we can use it if it's better
                    if e_val < r_val:
                        simplex[-1] = expanded
                        values[-1] = e_val
                    else:
                        simplex[-1] = reflected
                        values[-1] = r_val
                    break
                if e_val < r_val:
                    simplex[-1] = expanded
                    values[-1] = e_val
                else:
                    simplex[-1] = reflected
                    values[-1] = r_val
            elif r_val < values[-2]:
                simplex[-1] = reflected
                values[-1] = r_val
            else:
                if r_val < worst_value:
                    # Outside contraction
                    contracted = centroid + rho * (reflected - centroid)
                    c_val = func(contracted)
                    remaining -= 1
                    if remaining < 0:
                        if c_val < r_val:
                            simplex[-1] = contracted
                            values[-1] = c_val
                        else:
                            simplex[-1] = reflected
                            values[-1] = r_val
                        break
                    if c_val < r_val:
                        simplex[-1] = contracted
                        values[-1] = c_val
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if remaining <= 0: break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            values[i] = func(simplex[i])
                            remaining -= 1
                else:
                    # Inside contraction
                    contracted = centroid - rho * (centroid - worst_point)
                    c_val = func(contracted)
                    remaining -= 1
                    if remaining < 0:
                        if c_val < worst_value:
                            simplex[-1] = contracted
                            values[-1] = c_val
                        else:
                            simplex[-1] = reflected
                            values[-1] = r_val
                        break
                    if c_val < worst_value:
                        simplex[-1] = contracted
                        values[-1] = c_val
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if remaining <= 0: break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            values[i] = func(simplex[i])
                            remaining -= 1

        best_idx = np.argmin(values)
        return simplex[best_idx], values[best_idx]
