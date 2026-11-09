import numpy as np
import time
import math

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
        deadline = t0 + 290  # Safety margin for 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Latin Hypercube Sampling for diverse initial points
        n_init = min(25, budget // 8)
        rng = np.random.default_rng(123)
        lxs = rng.uniform(-5, 5, size=(n_init, dim))
        
        init_results = []
        for x in lxs:
            if eval_count >= budget:
                break
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            init_results.append((fx, x.copy()))
        
        # Sort initial points by fitness
        init_results.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start Nelder-Mead
        n_starts = min(5, len(init_results))
        
        if HAS_SCIPY_OPT and n_starts > 0:
            for start_idx in range(n_starts):
                if eval_count >= budget or time.time() > deadline - 20:
                    break
                
                x0 = init_results[start_idx][1].copy()
                
                def nm_wrapper(x):
                    nonlocal eval_count, best_f, best_x
                    if eval_count >= budget:
                        return best_f
                    x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    fx = func(x)
                    eval_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx
                
                try:
                    remaining_after_init = budget - eval_count
                    maxiter = max(20, min(100, remaining_after_init // (dim + 2)))
                    
                    res = minimize(
                        nm_wrapper,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxiter,
                            'xatol': 1e-5,
                            'fatol': 1e-5,
                            'adaptive': True
                        }
                    )
                    
                    if hasattr(res, 'fun') and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass
                
                # Second pass: tighter NM from current best
                if eval_count < budget - 15 and time.time() < deadline - 10:
                    try:
                        remaining2 = budget - eval_count
                        maxiter2 = max(10, min(60, remaining2 // (dim + 2)))
                        
                        def nm_wrapper2(x):
                            nonlocal eval_count, best_f, best_x
                            if eval_count >= budget:
                                return best_f
                            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                            fx = func(x)
                            eval_count += 1
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            return fx
                        
                        res2 = minimize(
                            nm_wrapper2,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': maxiter2,
                                'xatol': 1e-7,
                                'fatol': 1e-7,
                                'adaptive': True
                            }
                        )
                        
                        if hasattr(res2, 'fun') and res2.fun < best_f:
                            best_f = res2.fun
                            best_x = res2.x.copy()
                    except Exception:
                        pass
        
        # Phase 3: Local Polish with (1+1)-ES
        remaining = budget - eval_count
        if remaining > 0 and time.time() < deadline - 10:
            rng_local = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 10:
                if no_improve > limit:
                    sigma = max(0.01, sigma * 2.0)
                    no_improve = 0
                
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if sigma < 1e-6:
                        break

        return best_x, best_f


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
