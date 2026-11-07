import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        
        def wrapped_func(x):
            self.call_count += 1
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > self.deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -self.bound, self.bound)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = np.inf
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Sobol Sequence Sampling for Global Exploration
        # Use scipy.spatial.Sobol for quasi-random sampling
        try:
            from scipy.spatial import Sobol
            sobol = Sobol(d=self.dim, scramble=True)
            # Generate 64 points for initial exploration
            n_sobol = 64
            sobol_points = sobol.random(n_sobol)
            # Map to [-5, 5]^dim
            sobol_points = sobol_points * 10.0 - 5.0
            
            for p in sobol_points:
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                wrapped_func(p)
        except Exception:
            # Fallback to random sampling if Sobol fails
            for _ in range(64):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                p = np.random.uniform(-5, 5, size=self.dim)
                wrapped_func(p)
        
        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Multi-start Nelder-Mead Local Refinement
        # Find top candidates from the sampling phase
        # We'll use the best point and a few nearby points for multi-start
        
        # First, do a quick Nelder-Mead on the best point
        try:
            res = scipy.optimize.minimize(
                wrapped_func,
                self.best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = res.x
        except (TimeoutError, RuntimeError):
            pass
        except Exception:
            pass
        
        # If we have budget left, try a second start from a different region
        if self.call_count < self.budget - 50 and time.time() < self.deadline:
            # Generate a few additional starting points around the best
            n_additional = 3
            for i in range(n_additional):
                if self.call_count >= self.budget - 20 or time.time() > self.deadline:
                    break
                # Perturb the best point
                perturbation = np.random.normal(0, 1.0, size=self.dim)
                x_start = self.best_x + perturbation
                x_start = np.clip(x_start, -self.bound, self.bound)
                
                try:
                    res2 = scipy.optimize.minimize(
                        wrapped_func,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res2.fun < self.best_f:
                        self.best_f = res2.fun
                        self.best_x = res2.x
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    break

        return self.best_x, self.best_f
