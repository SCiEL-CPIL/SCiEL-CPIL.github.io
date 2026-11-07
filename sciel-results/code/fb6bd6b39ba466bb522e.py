import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                eval_count += 1
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                eval_count += 1
                return np.inf

        # Phase 1: Sparse Global Sampling using Halton Sequence
        # Katsuura is highly multimodal. We need good coverage.
        # 3^5 = 243 points is a lot for global phase if we want to save budget for local search.
        # Let's use 50 Halton points for initial spread.
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][d % 10]
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i-1, d] = r
            return points

        n_initial = 60
        halton_pts = halton_sequence(n_initial, dim)
        # Map from [0,1]^d to [-5,5]^d
        halton_pts = halton_pts * 10.0 - 5.0
        
        for pt in halton_pts:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)

        # Phase 2: CMA-ES for Global Exploration
        # CMA-ES is robust for black-box optimization.
        # We use a relatively large initial sigma to explore.
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                # Use scipy's CMA-ES implementation if available, or minimize with method='CMA' (if in scipy version)
                # scipy.optimize.minimize doesn't have CMA built-in in older versions, but we can use pygmo or similar?
                # No, standard library + numpy + scipy.
                # scipy.optimize.minimize supports 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-...', 'dual-annealing', 'shgo', 'differential_evolution'.
                # 'differential_evolution' is a good global optimizer.
                # Let's try Differential Evolution first for global search, as it's population-based and robust.
                
                # However, DE can be expensive. 
                # Let's stick to a multi-start local search strategy which is often more effective for Katsuura with limited budget.
                
                # Strategy:
                # 1. Initial Halton (done)
                # 2. Select top K candidates from Halton
                # 3. Run Nelder-Mead from each top candidate, or run a single CMA-ES-like population.
                
                # Since we don't have native CMA in scipy.optimize.minimize easily (it's in scipy.optimize._minimize? No, it's often in separate packages or newer scipy versions have it? 
                # Actually, scipy.optimize.minimize does NOT have CMA-ES as a method string in standard releases usually, unless very new. 
                # Let's check: methods include 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-...', 'dual-annealing', 'shgo', 'differential_evolution'.
                
                # 'differential_evolution' is available.
                # Let's use Differential Evolution for the remaining budget, but limit it.
                # Or, better: Multi-start Nelder-Mead.
                
                # Let's do Multi-start Nelder-Mead.
                # We need to store history of points to pick top K.
                
                # Refactor counting_func to store history?
                # Or just re-evaluate? No, we have limited budget.
                # Let's modify counting_func to keep a list of (f, x).
                
                pass # Placeholder, we need to restructure
                
            except Exception:
                pass

        # Let's restart the logic with a cleaner structure that tracks history.
        
        # Reset state
        best_x = None
        best_f = np.inf
        eval_count = 0
        history = [] # List of (f, x)

        def counting_func_v2(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                eval_count += 1
                history.append((val, x_clipped.copy()))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                eval_count += 1
                return np.inf

        # Phase 1: Halton Sampling
        n_initial = 80
        halton_pts = halton_sequence(n_initial, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        
        for pt in halton_pts:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func_v2(pt)

        # Phase 2: Multi-Start Nelder-Mead
        # Pick top 5 candidates from history
        if len(history) > 0:
            # Sort by fitness
            history_sorted = sorted(history, key=lambda item: item[0])
            top_k = min(5, len(history_sorted))
            starts = [item[1] for item in history_sorted[:top_k]]
            
            remaining = budget - eval_count
            # Allocate budget per start
            # If 5 starts, 100 evals each? 
            # NM is fast.
            budget_per_start = max(10, remaining // top_k)
            
            for start_x in starts:
                if eval_count >= budget or time.time() > deadline:
                    break
                
                try:
                    res_nm = scipy.optimize.minimize(
                        counting_func_v2,
                        start_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': budget_per_start,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Final Polish on the global best
        remaining = budget - eval_count
        if remaining > 5 and best_x is not None:
            try:
                res_polish = scipy.optimize.minimize(
                    counting_func_v2,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
        
        return best_x, best_f
