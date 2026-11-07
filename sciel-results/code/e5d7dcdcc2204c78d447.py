import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Initialize simplex at the origin
        x0 = np.zeros(dim)
        
        # Initial step size
        step_size = 1.0 
        
        # Create simplex: x0 and x0 + step_size * e_i
        simplex = [x0.copy()]
        for i in range(dim):
            point = x0.copy()
            point[i] += step_size
            simplex.append(point)
            
        # Evaluate initial simplex
        values = [func(p) for p in simplex]
        
        # Track best
        best_idx = np.argmin(values)
        best_val = values[best_idx]
        best_point = simplex[best_idx].copy()
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        calls_used = dim + 1
        max_calls = budget
        
        while calls_used < max_calls:
            # Sort simplex by function values
            sorted_pairs = sorted(zip(simplex, values), key=lambda x: x[1])
            simplex = [p for p, v in sorted_pairs]
            values = [v for p, v in sorted_pairs]
            
            # Update best
            if values[0] < best_val:
                best_val = values[0]
                best_point = simplex[0].copy()
                
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_point = simplex[-1]
            worst_val = values[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_point)
            fr = func(xr)
            calls_used += 1
            
            if fr < values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst_point)
                fe = func(xe)
                calls_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                # Simple reflection
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                if fr < worst_val:
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    calls_used += 1
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        # We need to be careful not to exceed budget
                        # Shrink requires dim calls
                        if calls_used + dim <= max_calls:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                values[i] = func(simplex[i])
                                calls_used += 1
                        else:
                            # Not enough budget for full shrink, just stop or do partial?
                            # Better to break to avoid exception
                            break
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst_point)
                    fc = func(xc)
                    calls_used += 1
                    
                    if fc < worst_val:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        if calls_used + dim <= max_calls:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                values[i] = func(simplex[i])
                                calls_used += 1
                        else:
                            break

        return best_point, best_val
