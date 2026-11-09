import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start L-BFGS-B + Valley-Following Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def valley_follow(x0, max_iters=20):
            """Budget-aware coordinate descent exploiting Rosenbrock valley: x[i+1] ~ x[i]^2."""
            x = np.clip(x0.copy(), -5, 5)
            f = wrapped_func(x)
            
            for it in range(max_iters):
                if self.calls_made >= self.budget - 5:
                    break
                if time.time() > self.deadline:
                    break
                
                improved = False
                
                # Valley-following: set x[i+1] = x[i]^2 for each i
                for i in range(self.dim - 1):
                    if self.calls_made >= self.budget - 5:
                        break
                    target_next = x[i] ** 2
                    if abs(x[i+1] - target_next) > 0.01:
                        x_vf = x.copy()
                        x_vf[i+1] = target_next
                        x_vf = np.clip(x_vf, -5, 5)
                        f_vf = wrapped_func(x_vf)
                        if f_vf < f:
                            x = x_vf
                            f = f_vf
                            improved = True
                
                # Simple coordinate descent
                for i in range(self.dim):
                    if self.calls_made >= self.budget - 5:
                        break
                    for direction in [1.0, -1.0]:
                        if self.calls_made >= self.budget - 5:
                            break
                        step = 0.1 * direction
                        x_new = x.copy()
                        x_new[i] += step
                        x_new = np.clip(x_new, -5, 5)
                        f_new = wrapped_func(x_new)
                        if f_new < f:
                            x = x_new
                            f = f_new
                            improved = True
                            break
                
                if not improved:
                    break
            
            return x, f

        rng = np.random.default_rng(42)
        
        starts = [np.ones(self.dim)]
        for scale in [0.5, 1.0, 2.0]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))
        for _ in range(5):
            starts.append(rng.uniform(-3, 5, self.dim))
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 15:
                continue
            
            # Phase 1: L-BFGS-B for fast initial convergence
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-8
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                break
            except Exception:
                pass
            
            # Phase 2: Valley-following refinement
            if self.best_x is not None and self.calls_made < self.budget - 10:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=15)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    break
                except Exception:
                    pass
        
        # Final polish
        if self.best_x is not None and self.calls_made < self.budget - 5 and time.time() < self.deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 150,
                        'ftol': 1e-14,
                        'gtol': 1e-12
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                pass
            except Exception:
                pass
            
            if self.best_x is not None and self.calls_made < self.budget - 5:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=10)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    pass
                except Exception:
                    pass

        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))
        
        return self.best_x, self.best_f
