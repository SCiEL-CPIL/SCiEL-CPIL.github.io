import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # 1. Evaluate center
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        best_x = x_center
        best_f = f_center
        calls_used = 1
        
        # 2. Estimate gradient signs using only positive steps
        # For linear f(x) = a^T x + b:
        # If f(e_i) < f(0), a_i < 0, so we want x_i = 5
        # If f(e_i) > f(0), a_i > 0, so we want x_i = -5
        signs = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
                
            x_probe = x_center.copy()
            x_probe[i] += 1.0
            x_probe = np.clip(x_probe, -5, 5)
            
            f_probe = func(x_probe)
            calls_used += 1
            
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()
                
            if f_probe < f_center:
                signs[i] = 1.0  # Move to +5
            else:
                signs[i] = -1.0 # Move to -5
                
        # 3. Construct vertex
        x_vertex = np.where(signs > 0, 5.0, -5.0)
        f_vertex = func(x_vertex)
        calls_used += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        return best_x, best_f

    def _nelder_mead(self, func, x, fx, budget, deadline):
        # Simple Nelder-Mead implementation
        dim = self.dim
        # Initial simplex
        x0 = x
        f0 = fx
        
        # Create simplex
        simplex = [x0.copy()]
        f_simplex = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            # Small step in dimension i
            step = 0.1 if abs(x0[i]) < 4.9 else 0.01
            xi[i] += step
            # Clip to bounds
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            f_simplex.append(fi)
            
            if fi < f0:
                f0 = fi
                x0 = xi.copy()
                
        # Track best
        best_x = x0
        best_f = f0
        
        # Nelder-Mead iterations
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        calls = dim + 1 # Initial simplex evals
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex
            idx = np.argsort(f_simplex)
            simplex = [simplex[i] for i in idx]
            f_simplex = [f_simplex[i] for i in idx]
            
            # Check convergence
            if abs(f_simplex[0] - f_simplex[-1]) < 1e-12:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_simplex[0] <= f_r < f_simplex[-2]:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            elif f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_simplex[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            
            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0].copy()

        return best_x, best_f
