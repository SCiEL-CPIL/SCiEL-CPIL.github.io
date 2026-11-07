import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0
        n = self.dim
        budget = self.budget
        
        # Initial point at origin
        x0 = np.zeros(n)
        f0 = func(x0)
        calls = 1
        
        # Initialize Simplex
        # Step size 1.0 is reasonable for box [-5,5]
        delta = 1.0
        simplex = [x0]
        fvals = [f0]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += delta
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            
        best_x = x0.copy()
        best_f = f0
        
        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main Loop
        while calls < budget and time.time() < deadline:
            # Sort by fitness
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            # Convergence Check
            # If the spread is very small, we are done
            if fvals[-1] - fvals[0] < 1e-10:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst)
            fr = func(xr)
            calls += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                fe = func(xe)
                calls += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < worst_f:
                    # Inside Contraction
                    xc = centroid + rho * (worst - centroid)
                    fc = func(xc)
                    calls += 1
                    
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                else:
                    # Outside Contraction
                    xo = centroid + rho * (centroid - worst)
                    fo = func(xo)
                    calls += 1
                    
                    if fo < worst_f:
                        simplex[-1] = xo
                        fvals[-1] = fo
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        
            if calls >= budget:
                break

        return best_x, best_f
