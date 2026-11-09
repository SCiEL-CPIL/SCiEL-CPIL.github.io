import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 290.0 # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(123)
        
        best_x = None
        best_f = float('inf')
        
        # Use CMA-ES if available? No, stick to the design family (NM).
        # The directive says: "KEEP this design and investigate improvements within its family."
        
        num_starts = 5
        
        for start_idx in range(num_starts):
            if time.time() > deadline:
                break
            if budget <= 0:
                break
            
            # Initialize simplex
            center = rng.uniform(-5, 5, dim)
            
            simplex = [center.copy()]
            f_values = [func(center)]
            
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = center.copy()
            budget -= 1
            
            delta = 0.5
            for i in range(dim):
                p = center.copy()
                p[i] += delta
                p = np.clip(p, -5, 5)
                f_p = func(p)
                simplex.append(p)
                f_values.append(f_p)
                budget -= 1
                
                if f_p < best_f:
                    best_f = f_p
                    best_x = p.copy()
            
            # Nelder-Mead iterations
            while budget > 0 and time.time() < deadline:
                # Sort simplex
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Convergence
                if abs(f_values[0] - f_values[-1]) < 1e-12 * (1 + abs(f_values[0])):
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                
                if budget < 1: break
                f_r = func(x_r)
                budget -= 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    
                    if budget < 1:
                         # Can't expand, use reflection
                         simplex[-1] = x_r
                         f_values[-1] = f_r
                         continue
                         
                    f_e = func(x_e)
                    budget -= 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                        
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Simple Reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if budget < 1:
                         # Can't contract, shrink? or just break?
                         # If we can't even contract, we are stuck.
                         break
                         
                    f_c = func(x_c)
                    budget -= 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                        
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        # Need dim evals
                        if budget < dim:
                            break
                            
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            budget -= 1
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                            if budget <= 0:
                                break
                        
                        if budget <= 0:
                            break
                            
        return best_f
