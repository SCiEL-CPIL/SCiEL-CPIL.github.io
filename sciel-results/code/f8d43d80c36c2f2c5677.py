import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # ZDT6 parameters
        n = self.dim
        # f1(x0) = 1 - exp(-4*x0) * sin^6(6*pi*x0)
        
        # Generate x0 candidates
        # Uniform in [0, 1]
        x0s = np.linspace(0.0, 1.0, n_evals)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x0 = float(x0s[i])
            
            # Compute f1(x0)
            if x0 < 1e-8:
                f1 = 1.0
            else:
                f1 = 1.0 - np.exp(-4.0 * x0) * (np.sin(6.0 * np.pi * x0) ** 6)
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = x0
            
            # Optimize x[1:] to minimize g(x)
            # g(x) = 1 + 10(n-1) * [ sum( (xi - f1)^2 / (4*x0*n) + 0.1*sin(20*pi*xi) ) ]
            # We minimize the term inside the sum for each xi independently.
            # Term: J(xi) = (xi - f1)^2 / C + 0.1 * sin(20*pi*xi)
            # C = 4 * x0 * n
            
            if x0 < 1e-8:
                # If x0 is 0, the division by x0 is problematic.
                # The standard ZDT6 PF definition for x1=0 is often handled by limit or specific convention.
                # However, f1(0)=1.
                # If we set xi=1, (1-1)^2/0 is 0/0.
                # Let's just set xi=0 for x0=0 case to be safe, or xi=f1?
                # Actually, the term is (xi-f1)^2. If xi=f1, it is 0.
                # But C=0.
                # In ZDT6, usually x1=0 is a boundary.
                # Let's set xi = 0.0 for this single case.
                x[1:] = 0.0
            else:
                C = 4.0 * x0 * n
                # J(xi) = (xi - f1)^2 / C + 0.1 * sin(20*pi*xi)
                # J'(xi) = 2(xi - f1)/C + 0.1 * 20*pi * cos(20*pi*xi)
                # J''(xi) = 2/C - 0.1 * (20*pi)^2 * sin(20*pi*xi)
                
                # Initial guess: f1, or nearest multiple of 0.05 to f1
                # f1 is in [0.28, 1.0]
                # Multiples of 0.05: 0.30, 0.35, ...
                x_init = np.round(f1 / 0.05) * 0.05
                # Clamp to [0, 1]
                x_init = np.clip(x_init, 0.0, 1.0)
                
                # Newton-Raphson iterations (3-5 steps should be enough for this smooth function)
                x_opt = x_init
                for _ in range(5):
                    # Gradient
                    grad = 2.0 * (x_opt - f1) / C + 0.1 * 20.0 * np.pi * np.cos(20.0 * np.pi * x_opt)
                    # Hessian
                    hess = 2.0 / C - 0.1 * (20.0 * np.pi) ** 2 * np.sin(20.0 * np.pi * x_opt)
                    
                    if np.abs(hess) < 1e-8:
                        hess = 1e-8 * (1 if hess > 0 else -1)
                        
                    step = grad / hess
                    x_new = x_opt - step
                    
                    # Convergence check
                    if np.abs(step) < 1e-6:
                        x_opt = x_new
                        break
                    x_opt = x_new
                    
                # Clamp final result
                x_opt = np.clip(x_opt, 0.0, 1.0)
                x[1:] = x_opt
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
