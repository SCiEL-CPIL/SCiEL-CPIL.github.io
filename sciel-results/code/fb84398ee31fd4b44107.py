# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-scale coordinate descent + L-BFGS-B refinement for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.perf_counter()
        deadline = t_start + 285.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.perf_counter()

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            update_best(x, f)
            return f

        # Phase 1: Multi-scale coordinate descent from multiple starting points
        # This exploits the "different powers" structure where coordinates may
        # have different scales of importance
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        
        # Halton sequence for quasi-random coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, 15):
            pt = np.array([halton(i, b) for b in range(2, dim + 2)])
            pt = pt * 10.0 - 5.0
            starts.append(pt)
        
        # Some random points
        for _ in range(10):
            starts.append(rng.uniform(-5.0, 5.0, dim))
        
        # Axis-aligned points
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        
        rng.shuffle(starts)
        
        # Multi-scale coordinate descent
        # Start with large strides and geometrically shrink
        stride = 4.0
        shrink_factor = 0.5
        
        # For each start, do coordinate descent with shrinking strides
        # We limit total calls to stay within budget
        cd_calls_limit = int(budget * 0.4)  # Use 40% of budget for CD phase
        cd_calls_used = 0
        
        cd_results = []  # Store (f, x) for each CD run
        
        for x0 in starts:
            if cd_calls_used >= cd_calls_limit or time_left() < 10.0:
                break
            
            x = x0.copy()
            f = wrapped_func(x)
            cd_calls_used += 1
            update_best(x, f)
            
            # Coordinate descent with shrinking strides
            for stride_iter in range(8):
                current_stride = stride * (shrink_factor ** stride_iter)
                if current_stride < 1e-6:
                    break
                
                improved = False
                for d in range(dim):
                    if cd_calls_used >= cd_calls_limit or time_left() < 10.0:
                        break
                    
                    # Try moving in positive direction
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x[d] + current_stride, -5.0, 5.0)
                    f_trial = wrapped_func(x_trial)
                    cd_calls_used += 1
                    
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        update_best(x, f)
                    elif cd_calls_used < cd_calls_limit and time_left() > 10.0:
                        # Try moving in negative direction
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x[d] - current_stride, -5.0, 5.0)
                        f_trial = wrapped_func(x_trial)
                        cd_calls_used += 1
                        
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            update_best(x, f)
                
                if not improved:
                    break
            
            cd_results.append((f, x.copy()))
        
        # Phase 2: L-BFGS-B refinement from the best CD results
        # Sort by function value and take top candidates
        cd_results.sort(key=lambda item: item[0])
        
        # Take top 5 candidates for L-BFGS-B
        top_candidates = cd_results[:5]
        
        bounds = [(-5.0, 5.0)] * dim
        lbfgs_calls_limit = budget - cd_calls_used - 50  # Reserve some budget
        
        for f_cand, x_cand in top_candidates:
            if lbfgs_calls_limit <= 0 or time_left() < 15.0:
                break
            
            x_cand = np.clip(x_cand, -5.0, 5.0)
            
            try:
                res = minimize(
                    wrapped_func,
                    x_cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 80,
                        'maxls': 15,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': max(10, lbfgs_calls_limit // 2)
                    }
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                
                lbfgs_calls_limit -= 1  # Approximate; actual calls tracked by wrapped_func
            except Exception:
                pass
        
        # Phase 3: If we still have budget, do a few more random restarts with L-BFGS-B
        remaining_budget = budget - calls_made
        if remaining_budget > 50 and time_left() > 20.0:
            n_extra = min(3, remaining_budget // 50)
            for _ in range(n_extra):
                if time_left() < 10.0 or calls_made >= budget:
                    break
                
                x0 = rng.uniform(-5.0, 5.0, dim)
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 40,
                            'maxls': 10,
                            'ftol': 1e-10,
                            'gtol': 1e-8,
                            'maxfun': 30
                        }
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass
        
        # Final refinement: one more L-BFGS-B from the best point found
        if calls_made < budget - 10 and time_left() > 5.0:
            try:
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 30,
                        'maxls': 10,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxfun': 20
                    }
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
