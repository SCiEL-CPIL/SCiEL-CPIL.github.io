import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin for 330s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        if HAS_CMA:
            try:
                x0 = [0.0] * self.dim
                # Standard initial step size for BBOB functions in [-5, 5]
                sigma0 = 2.0 
                # Larger population for better covariance estimation early on
                popsize = 12
                
                # Use 90% of budget for CMA-ES
                cma_budget = int(self.budget * 0.9)
                maxiter = (cma_budget - 1) // popsize
                
                options = {
                    'verbose': -9,
                    'seed': 0,
                    'bounds': [-5.0] * self.dim + [5.0] * self.dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    # Relax stopping criteria to ensure we use the allocated budget
                    'stopitersize': 1e-12,
                    'stopconditionnumber': 1e14,
                    'stopfgap': 1e-15,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                # Evaluate initial point
                fx = func(x0)
                eval_count += 1
                best_f = fx
                best_x = np.array(x0)
                
                while not es.stop() and eval_count < cma_budget:
                    if time.time() > deadline - 5:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= cma_budget:
                            break
                        if time.time() > deadline - 5:
                            break
                        fx = func(x)
                        eval_count += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass

        # Local Polish: (1+1)-ES with small step size
        # Use remaining budget
        remaining = self.budget - eval_count
        if remaining < 0:
            remaining = 0
            
        if remaining > 0 and time.time() < deadline - 5:
            rng = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            
            # Start with a small sigma for polishing
            sigma = 0.1
            no_improve = 0
            limit = 10 # Restart local search if stuck
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    # Try a slightly larger step or reset
                    sigma = 0.5
                    no_improve = 0
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1
                eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.8
                    no_improve += 1
                    # If we are very close to the optimum, sigma will become very small.
                    if sigma < 1e-5:
                        break
                        
        return best_x, best_f
