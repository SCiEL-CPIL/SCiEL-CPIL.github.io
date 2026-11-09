import numpy as np
import time
import math

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        rng = np.random.default_rng(42)
        
        def try_eval(x):
            nonlocal best_f, best_x, eval_count
            if eval_count >= self.budget or time.time() > deadline - 2:
                return None
            try:
                fx = func(x)
            except Exception:
                return None
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling
        n_init = min(30, self.budget // 5)
        for i in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            try_eval(x)
        
        # Phase 2: CMA-ES for global convergence
        if HAS_CMA:
            try:
                x0 = best_x.tolist()
                sigma0 = 0.5
                popsize = 4 + int(3 * math.log(self.dim))
                if popsize < 4:
                    popsize = 4
                
                remaining_budget = self.budget - eval_count
                cma_budget = int(remaining_budget * 0.6)
                maxiter = max(1, (cma_budget - 1) // popsize)
                
                options = {
                    'verbose': -9,
                    'seed': 1,
                    'bounds': [-5.0] * self.dim + [5.0] * self.dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    'stopitersize': 1e-10,
                    'stopconditionnumber': 1e12,
                    'stopfgap': 1e-12,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget:
                    if time.time() > deadline - 15:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        fx = try_eval(x)
                        if fx is None:
                            break
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass

        # Phase 3: Nelder-Mead for refinement
        if HAS_SCIPY_OPT and eval_count < self.budget and time.time() < deadline - 15:
            try:
                def wrapper(x):
                    nonlocal eval_count, best_f, best_x
                    if eval_count >= self.budget or time.time() > deadline - 5:
                        return float('inf')
                    
                    x = np.clip(x, -5, 5)
                    try:
                        fx = func(x)
                    except Exception:
                        return float('inf')
                    eval_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                nm_budget = self.budget - eval_count
                maxiter_nm = max(10, nm_budget // (self.dim + 1))
                
                res = minimize(wrapper, best_x, method='Nelder-Mead', 
                               options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if hasattr(res, 'fun') and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
                    
            except Exception:
                pass

        # Phase 4: Local Polish: (1+1)-ES with adaptive step size
        remaining = self.budget - eval_count
        if remaining < 0:
            remaining = 0
            
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(123)
            x = best_x.copy()
            fx = best_f
            
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = 0.1
                    no_improve = 0
                    
                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if sigma < 1e-6:
                        break
                        
        return best_x, best_f
