import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy:
        # 1. Diverse Global Initialization using Latin Hypercube Sampling (LHS) for better coverage than uniform random.
        # 2. Aggressive Local Refinement using L-BFGS-B and Nelder-Mead from the top diverse basins.
        # 3. Robust Global Search using Differential Evolution if significant budget remains.
        # 4. Final Polish on the global best.

        # Phase 1: Diverse Initialization
        # LHS ensures that each dimension is sampled evenly across the range, which is often more effective
        # for finding diverse basins in 5D than pure random sampling, especially for multimodal functions.
        
        num_lhs = 40
        lhs_points = []
        for i in range(num_lhs):
            # For each dimension, generate a unique bin index
            # This creates a stratified sample
            u = np.sort(rng.random(d))
            # Map to bins [0, 1/d), [1/d, 2/d), etc.
            x = (np.arange(d) + u) / d * 10.0 - 5.0
            lhs_points.append(x)
        
        # Add a few pure random points to break any potential LHS symmetry issues
        num_rand = 10
        for i in range(num_rand):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            lhs_points.append(x)
            
        candidates = []
        for x in lhs_points:
            if calls_made >= self.budget or time.time() > deadline:
                break
            f = call_func(x)
            candidates.append((f, x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement
        # Run local optimization from the top candidates.
        max_local_runs = 12
        
        for i in range(max_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Select start point
            if i < len(candidates):
                _, x_start = candidates[i]
            else:
                x_start = rng.uniform(-5.0, 5.0, d)
                call_func(x_start)
            
            # Dynamic budget allocation
            remaining = self.budget - calls_made
            runs_left = max_local_runs - i
            if runs_left <= 0:
                runs_left = 1
            
            # Target ~20-30 calls per L-BFGS-B run, but scale with remaining budget
            # We want to ensure we don't exhaust budget on one run
            target_per_run = 25
            maxfev_lbfgs = int(min(remaining * 0.6, max(15, target_per_run * (remaining // (runs_left * 1.5)))))
            maxfev_lbfgs = max(15, min(60, maxfev_lbfgs))
            
            # Run L-BFGS-B
            res_x = x_start
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev_lbfgs,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                res_x = res.x
            except Exception:
                pass
            
            # Polish with Nelder-Mead
            if calls_made < self.budget and time.time() < deadline:
                remaining_iter = self.budget - calls_made
                nm_maxfev = max(10, min(25, remaining_iter // max(1, runs_left)))
                
                try:
                    res_nm = minimize(
                        call_func,
                        res_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-10,
                            'fatol': 1e-10
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Differential Evolution if significant budget remains
        # DE is robust and can explore the space more thoroughly than random sampling.
        if calls_made < self.budget - 50 and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            # Allocate up to 80 calls to DE, but not more than half the remaining budget
            # DE is expensive (popsize * dim per iteration), so be careful
            de_maxiter = min(80, remaining // 2)
            if de_maxiter > 10:
                try:
                    def de_wrapper(x):
                        return call_func(x)
                    
                    res_de = differential_evolution(
                        de_wrapper,
                        bounds,
                        maxiter=de_maxiter,
                        popsize=12, # Slightly smaller pop to fit more iterations in budget
                        tol=1e-10,
                        seed=42,
                        polish=False,  # We do our own polish
                        init='latinhypercube'
                    )
                except Exception:
                    pass
        
        # Phase 4: Final Polish
        # If we have budget left, do one more intensive run from the global best.
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 10:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': min(30, remaining),
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
                
                if calls_made < self.budget and time.time() < deadline:
                    remaining = self.budget - calls_made
                    if remaining > 5:
                        try:
                            minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': min(15, remaining),
                                    'xatol': 1e-12,
                                    'fatol': 1e-12
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
