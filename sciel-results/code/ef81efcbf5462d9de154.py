import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using stratified global search followed by CMA-ES local refinement
    and coordinate descent polish.
    1. Stratified sampling (Sobol-like) for uniform global coverage (~150 evals)
    2. CMA-ES for robust local optimization on rotated/elongated basins (~200 evals)
    3. Coordinate descent polish for final refinement (~remaining evals)
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                f_val = float(func(x))
            except:
                f_val = np.inf
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Stratified Global Search ---
        # Use stratified sampling for better coverage than pure grid
        # Divide each dimension into 4 strata, sample one point per stratum combination
        # 4^5 = 1024 combinations is too many, so use a subset
        # Instead, use a quasi-random (Sobol-like) sequence for uniform coverage
        
        # Generate stratified points: for each dimension, divide [-5,5] into 4 intervals
        # and pick a random point in each. Then combine.
        # To keep it manageable, use 2^5 = 32 base points with jitter, then expand
        
        # Better approach: use a 3^5 grid but with jitter, or use a Sobol sequence
        # Let's use a simple stratified approach: 3 levels per dim = 3^5 = 243 is standard
        # But we can do better with a smaller set of well-distributed points
        
        # Use a 4-point per dimension stratified sample: 4^5 = 1024 is too many
        # Use 3^5 = 243 grid points as in original, but we can reduce to ~150
        # Let's use a 3^5 grid but only evaluate a subset if time is tight
        
        # Actually, let's use a smarter approach: Halton sequence for quasi-random sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate ~150 quasi-random points in [0,1]^5, then map to [-5,5]^5
        n_global = 150
        points = np.zeros((n_global, dim))
        for i in range(n_global):
            for j in range(dim):
                # Use different primes for each dimension
                bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                points[i, j] = halton(i + 1, bases[j])
        
        # Map from [0,1] to [-5,5]
        points = points * 10.0 - 5.0
        
        # Evaluate all global points
        for x in points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)
        
        # If we ran out, return best
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # --- Phase 2: CMA-ES Local Refinement ---
        # Use scipy's minimize with Nelder-Mead or a custom CMA-ES-like approach
        # Since we can't use scipy.optimize.cma directly (not available), 
        # let's implement a simple CMA-ES or use Nelder-Mead
        
        # Actually, scipy.optimize.minimize with method='Nelder-Mead' is available
        # But it can be slow. Let's implement a simple (1, lambda)-ES with covariance adaptation
        
        # Simpler: use multiple restarts of Nelder-Mead from top few global points
        # Or implement a basic CMA-ES
        
        # Let's use Nelder-Mead from the best point, with a time limit
        if best_x is not None and calls < budget:
            x0 = best_x.copy()
            f0 = best_f
            
            # Use Nelder-Mead with maxiter based on remaining budget
            remaining = budget - calls
            max_iter = max(1, remaining // 2)  # Each NM iteration uses dim+1 evals
            
            # Wrap func for scipy
            def scipy_func(x):
                if calls >= budget or time.time() > deadline:
                    return np.inf
                return evaluate(x)
            
            try:
                result = minimize(scipy_func, x0, method='Nelder-Mead',
                                 options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except:
                pass
        
        # If we ran out, return best
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # --- Phase 3: Coordinate Descent Polish ---
        # Fine-tune each coordinate individually
        if best_x is not None and calls < budget:
            x = best_x.copy()
            f = best_f
            step = 0.5
            
            # Adaptive coordinate descent
            for coord in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                # Try positive and negative steps
                for direction in [1, -1]:
                    for _ in range(10):  # Up to 10 steps per direction per coord
                        if calls >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[coord] += direction * step
                        f_trial = evaluate(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            best_f = f
                            best_x = x.copy()
                        else:
                            step *= 0.5
                            if step < 1e-10:
                                break
            
            # Final refinement with smaller steps
            step = 0.01
            for coord in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                for direction in [1, -1]:
                    for _ in range(5):
                        if calls >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[coord] += direction * step
                        f_trial = evaluate(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            best_f = f
                            best_x = x.copy()
                        else:
                            step *= 0.5
                            if step < 1e-12:
                                break
        
        return best_x, best_f
