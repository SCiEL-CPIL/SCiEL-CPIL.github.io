import numpy as np
import time
import scipy.optimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_fx = np.inf
        evaluated_points = [] # List of (fx, x)

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            evaluated_points.append((fx, x.copy()))
            return fx

        bounds = [(-5, 5)] * self.dim

        # Phase 1: Global Exploration
        n_global = 30
        candidates = []
        
        # Structured points
        candidates.append(np.zeros(self.dim))
        candidates.append(np.ones(self.dim))
        candidates.append(np.full(self.dim, 0.5))
        candidates.append(np.full(self.dim, -0.5))
        
        while len(candidates) < n_global:
            candidates.append(rng.uniform(-4, 4, self.dim))
            
        for x0 in candidates:
            if call_count >= self.budget - 100: # Reserve budget for local
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            tracked_func(best_x)

        # Sort evaluated points to get top K
        # evaluated_points contains (fx, x)
        evaluated_points.sort(key=lambda p: p[0])
        
        # Select top K points for local optimization
        K = 5
        top_points = [x for fx, x in evaluated_points[:K]]
        
        # Phase 2: Local Optimization with Powell
        # Allocate remaining budget to these K points.
        remaining_budget = self.budget - call_count
        # We want to give each point a chance. 
        # Powell can be expensive. Let's give each a max of 50-80 evals.
        # If we have 470 remaining, and K=5, that's ~90 each.
        
        for x_start in top_points:
            if call_count >= self.budget - 10:
                break
            if time.time() > deadline - 10:
                break
            
            # Use Powell for robustness on Rosenbrock
            # Nelder-Mead is faster but Powell is more stable for the valley.
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    x_start,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxfev': 80, # Limit per restart
                        'xtol': 1e-12,
                        'ftol': 1e-12
                    }
                )
            except TimeoutError:
                break
            except Exception:
                pass

        # Phase 3: Final Polish
        # If we have budget left, do a tight Nelder-Mead or Powell on the best point
        if call_count < self.budget - 5 and time.time() < deadline - 5:
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-10,
                        'fatol': 1e-10
                    }
                )
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_fx
