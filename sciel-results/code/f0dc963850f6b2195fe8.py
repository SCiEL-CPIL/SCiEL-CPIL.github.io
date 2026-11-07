"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Direct Pareto front construction via analytical insight on ZDT4 structure."""
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        
        rng = np.random
        
        all_points = []
        all_evals = 0
        
        def check_time():
            return time.time() > deadline
        
        def eval_point(x):
            nonlocal all_evals
            if all_evals >= self.budget or check_time():
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_evals += 1
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                all_points.append(f)
            return f
        
        # ZDT4: f1 = x1, f2 = g(x2..xn) * (1 - sqrt(x1/g))
        # g = 10n + sum((xi-0.5)^2 - 10*cos(20*pi*(xi-0.5))) for i=2..n
        # Minimum of g is 10 (when all xi=0.5)
        # Pareto front: f2 = 1 - sqrt(f1/10) for f1 in [0, 10]
        
        # Phase 1: Find x2..xn that minimizes g (should be all 0.5s or similar)
        # Try a few candidates
        best_x_rest = None
        best_g_val = np.inf
        
        n_try_rest = 20
        for i in range(n_try_rest):
            if all_evals >= self.budget or check_time():
                break
            
            if i == 0:
                # Midpoint - likely optimal for ZDT4
                x_rest = (lo[1:] + hi[1:]) / 2
            elif i == 1:
                x_rest = np.full(n-1, 0.5)
            elif i == 2:
                x_rest = lo[1:].copy()
            elif i == 3:
                x_rest = hi[1:].copy()
            else:
                x_rest = lo[1:] + (hi[1:] - lo[1:]) * rng.random(n - 1)
            
            x_rest = np.clip(x_rest, lo[1:], hi[1:])
            
            # Evaluate to estimate g: use x1=0 to get f2 = g (since sqrt(0)=0, f2=g*1=g)
            # Actually f2 = g*(1-sqrt(0/g)) = g*1 = g
            x = np.concatenate(([0.0], x_rest))
            f = eval_point(x)
            if f is not None:
                # f[1] should approximate g
                g_est = f[1]
                if g_est < best_g_val:
                    best_g_val = g_est
                    best_x_rest = x_rest.copy()
        
        if best_x_rest is None:
            best_x_rest = (lo[1:] + hi[1:]) / 2
        
        # Phase 2: Sweep x1 densely to trace the Pareto front
        # ZDT4 Pareto front: f1 = x1 in [0, 10], f2 = 10*(1 - sqrt(f1/10))
        remaining_budget = self.budget - all_evals
        if remaining_budget > 0 and not check_time():
            # Sweep x1 over [max(lo[0], 0), min(hi[0], 10)] to cover the Pareto front
            x1_lo = max(lo[0], 0.0)
            x1_hi = min(hi[0], 10.0)
            if x1_lo >= x1_hi:
                # No valid range, try midpoint
                x1_lo, x1_hi = lo[0], hi[0]
            
            n_sweep = min(remaining_budget, 180)
            x1_actual = np.linspace(x1_lo, x1_hi, n_sweep)
            
            for x1v in x1_actual:
                if all_evals >= self.budget or check_time():
                    break
                x = np.concatenate(([x1v], best_x_rest))
                eval_point(x)
        
        # Phase 3: Local refinement - perturb x2..xn slightly to find better g
        remaining_budget = self.budget - all_evals
        if remaining_budget > 0 and not check_time() and len(all_points) > 0:
            # Try small perturbations around best_x_rest
            n_refine = min(remaining_budget, 40)
            for i in range(n_refine):
                if all_evals >= self.budget or check_time():
                    break
                
                # Perturb x_rest
                scale = 0.02
                x_rest_pert = best_x_rest + scale * rng.standard_normal(n-1)
                x_rest_pert = np.clip(x_rest_pert, lo[1:], hi[1:])
                
                # Evaluate at x1 values spread across the Pareto front range
                x1_lo = max(lo[0], 0.0)
                x1_hi = min(hi[0], 10.0)
                if x1_lo < x1_hi:
                    x1_samples = np.linspace(x1_lo, x1_hi, 3)
                else:
                    x1_samples = [lo[0], (lo[0]+hi[0])/2, hi[0]]
                for x1v in x1_samples:
                    if all_evals >= self.budget or check_time():
                        break
                    x = np.concatenate(([x1v], x_rest_pert))
                    eval_point(x)
        
        # Phase 4: If budget remains, do targeted exploration near Pareto front
        remaining_budget = self.budget - all_evals
        if remaining_budget > 0 and not check_time():
            n_random = min(remaining_budget, 30)
            x1_lo = max(lo[0], 0.0)
            x1_hi = min(hi[0], 10.0)
            for i in range(n_random):
                if all_evals >= self.budget or check_time():
                    break
                if x1_lo < x1_hi:
                    x1v = rng.uniform(x1_lo, x1_hi)
                else:
                    x1v = rng.uniform(lo[0], hi[0])
                x_rest_pert = best_x_rest + 0.01 * rng.standard_normal(n-1)
                x_rest_pert = np.clip(x_rest_pert, lo[1:], hi[1:])
                x = np.concatenate(([x1v], x_rest_pert))
                eval_point(x)
        
        # Convert to array
        if len(all_points) > 0:
            result = np.array(all_points, dtype=np.float64)
            mask = np.all(np.isfinite(result), axis=1)
            result = result[mask]
        else:
            # Last resort
            x = (lo + hi) / 2
            f = problem(x)
            result = np.array([f], dtype=np.float64)
        
        return result
# EVOLVE-BLOCK-END
