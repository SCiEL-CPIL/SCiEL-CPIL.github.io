import numpy as np
import time
import scipy.optimize
import scipy.stats

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.func = None

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500
        self.func = func

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        # Phase 1: Global exploration with Sobol sequence
        # Use 40 points to get a good initial guess quickly
        n_global = 40
        # Sobol sequence requires power of 2 for perfect uniformity, but works for any N
        # We use a fixed seed for reproducibility in the sequence generation, 
        # but the function itself is random.
        sobol = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Generate points in [0, 1]^d
        sobol_points = sobol.random(n_global)
        # Scale to [-5, 5]^d
        sobol_points = sobol_points * (2 * self.bound) - self.bound
        
        candidates = []
        for i in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = sobol_points[i]
            fx = wrapped_func(p)
            candidates.append((fx, p.copy()))
            
        if not candidates:
            # Fallback if something went wrong
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)
            return self.best_x, self.best_f

        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local refinement on the best candidates
        # We will try to refine the top 3 candidates if budget allows
        # Budget allocation: 
        # - Top 1: ~200 calls
        # - Top 2: ~100 calls
        # - Top 3: ~50 calls
        # Remaining: Fine polish
        
        # Define a helper to run local optimization with a specific budget cap
        def run_local_opt(x_start, max_calls):
            # We need a wrapper that stops after max_calls additional calls
            # The global wrapped_func already handles the total budget and best tracking.
            # We just need to ensure we don't exceed the local cap.
            local_start_count = self.call_count
            local_max = local_start_count + max_calls
            
            # SciPy minimize can be tricky with strict call limits.
            # We use a callback or just let it run and catch exceptions/budget errors.
            # Better: Use a custom wrapper that raises an exception if local budget is hit.
            
            def local_wrapped_func(x):
                if self.call_count >= local_max:
                    raise RuntimeError("Local budget exceeded")
                if time.time() > self.deadline:
                    raise TimeoutError()
                return wrapped_func(x)
            
            try:
                res = scipy.optimize.minimize(
                    local_wrapped_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_calls, # Hard limit on iterations, but calls > iterations
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except (RuntimeError, TimeoutError):
                # Budget exceeded, which is fine, we just stop
                pass
            except Exception:
                pass

        # Try top 1
        if self.call_count < self.budget - 20:
            best_cand = candidates[0]
            run_local_opt(best_cand[1], 200)
            
        # Try top 2 if it's sufficiently different
        if len(candidates) > 1 and self.call_count < self.budget - 10:
            # Check if the 2nd best is far from the 1st best
            # If they are close, refining the 2nd might not help much
            if np.linalg.norm(candidates[0][1] - candidates[1][1]) > 0.5:
                run_local_opt(candidates[1][1], 100)
            else:
                # If close, just give the remaining budget to the best one again?
                # Or skip to top 3 if available and distinct
                if len(candidates) > 2 and np.linalg.norm(candidates[0][1] - candidates[2][1]) > 0.5:
                     run_local_opt(candidates[2][1], 100)

        # Try top 3 if distinct and budget allows
        if len(candidates) > 2 and self.call_count < self.budget - 5:
            # Ensure it's distinct from the ones already tried (approximate check)
            # This is a heuristic, but good enough
            if np.linalg.norm(candidates[0][1] - candidates[2][1]) > 0.5 and \
               np.linalg.norm(candidates[1][1] - candidates[2][1]) > 0.5:
                run_local_opt(candidates[2][1], 50)

        # Phase 3: Fine polish on the current best
        if self.best_x is not None and self.call_count < self.budget - 2:
            # A very small step local search
            x = self.best_x.copy()
            f_best = self.best_f
            step = 0.01
            # Simple coordinate descent polish
            for _ in range(10): # Max 10 iterations of coordinate descent
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                improved = False
                for i in range(self.dim):
                    if self.call_count >= self.budget or time.time() > self.deadline:
                        break
                    for sign in [1, -1]:
                        if self.call_count >= self.budget or time.time() > self.deadline:
                            break
                        x_try = x.copy()
                        x_try[i] += sign * step
                        x_try = np.clip(x_try, -self.bound, self.bound)
                        f_try = wrapped_func(x_try)
                        if f_try < f_best - 1e-12:
                            x = x_try
                            f_best = f_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break

        return self.best_x, self.best_f
