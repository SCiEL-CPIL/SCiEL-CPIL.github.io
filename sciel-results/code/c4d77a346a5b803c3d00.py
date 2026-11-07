import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        bounds = [(-5.0, 5.0)] * self.dim
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        d = self.dim

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

        # Strategy: Adaptive global sampling + priority-based local refinement with restarts
        
        # Phase 1: Global Sampling - more samples for better basin coverage
        num_global = 50
        candidates = []
        for i in range(num_global):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            candidates.append((f, x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement
        # We will perform local optimization from the top K candidates.
        # K depends on remaining budget.
        # Each local run gets a maxfev limit to ensure we try multiple starts.
        
        # Estimate cost of a local run. L-BFGS-B in 5D usually converges in < 50 evals.
        # NM can be similar or slightly more.
        # Let's set a dynamic maxfev for local runs.
        
        # We'll try to fit as many local runs as possible.
        # Let's say we want at least 5 local runs.
        # If we have 270 calls left, 270/5 = 54 calls per run.
        
        # Let's iterate:
        # Pick the best candidate not yet refined (or just the top K).
        # To avoid redundancy, we can check distance to already refined points,
        # but simple top-K is often sufficient for BBOB.
        
        num_local_runs = 10  # We will try up to 10 local runs
        # But we stop if budget runs out.
        
        # To make it more robust, we'll interleave:
        # Refine top 5, then if budget remains, refine next 5, etc.
        # Actually, just a loop over the sorted candidates is fine.
        
        # We need to be careful not to re-evaluate the same point too much.
        # L-BFGS-B will evaluate the start point again. That's 1 call.
        
        for i in range(num_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Pick a start point
            if i < len(candidates):
                _, x_start = candidates[i]
            else:
                # If we run out of candidates, generate new random ones
                x_start = rng.uniform(-5.0, 5.0, d)
                f = call_func(x_start)
            
            # Determine maxfev for this local run
            remaining = self.budget - calls_made
            runs_left = num_local_runs - i
            if runs_left <= 0:
                runs_left = 1
            maxfev = max(10, min(50, remaining // runs_left))
            
            # Run L-BFGS-B first (good for smooth)
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass
            
            # If we still have budget and time, polish with Nelder-Mead
            if calls_made < self.budget and time.time() < deadline:
                nm_maxfev = max(5, min(20, (self.budget - calls_made) // max(1, runs_left)))
                try:
                    res_nm = minimize(
                        call_func,
                        res.x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final Polish
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 20:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': min(50, remaining),
                            'ftol': 1e-15,
                            'gtol': 1e-10
                        }
                    )
                except Exception:
                    pass
                
                # One more NM polish if budget allows
                if calls_made < self.budget and time.time() < deadline:
                    remaining = self.budget - calls_made
                    if remaining > 5:
                        try:
                            minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': min(20, remaining),
                                    'xatol': 1e-10,
                                    'fatol': 1e-10
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            # Ensure at least one call is made if something went wrong
            call_func(best_x)

        return best_x, best_f
