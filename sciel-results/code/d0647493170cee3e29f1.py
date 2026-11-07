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
        deadline = start_time + 270.0
        rng = np.random.default_rng(12345)
        
        call_count = 0
        best_x = None
        best_fx = np.inf
        history = []  # (fx, x) sorted by fx

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                x = x[:self.dim]
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            history.append((fx, x.copy()))
            return fx

        dim = self.dim
        bounds = [(-5, 5)] * dim

        # Phase 1: Global Exploration using Sobol-like or random + structured points
        # We want to cover the space well. Let's use a mix of random and a few structured points.
        # Since dim=5, we can afford a reasonable number of initial points.
        n_global = 100
        points = []
        
        # Random points in [-4.5, 4.5]
        for _ in range(n_global):
            points.append(rng.uniform(-4.5, 4.5, dim))
        
        # Add center and a few other key points
        points.append(np.zeros(dim))
        
        # Evaluate all points
        for x0 in points:
            if call_count >= self.budget - 100:
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            tracked_func(best_x)

        # Sort history to get best points
        history.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization from top diverse points
        # We will run a local optimizer from the best point, and if budget allows, from a few others.
        # Use Nelder-Mead as it's derivative-free and robust.
        
        # Helper to run local optimization
        def run_local_optimizer(x0, maxfev):
            nonlocal best_x, best_fx
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': maxfev,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except TimeoutError:
                pass
            except Exception:
                pass

        # Run from the best point
        if time.time() < deadline and call_count < self.budget - 50:
            remaining = self.budget - call_count - 20
            run_local_optimizer(best_x, remaining)

        # If budget remains, try from the second best point if it's sufficiently different
        if time.time() < deadline and call_count < self.budget - 30 and len(history) > 1:
            # Find a diverse point
            diverse_x = None
            for fx, x in history[1:]:
                if np.linalg.norm(x - best_x) > 1.0:
                    diverse_x = x
                    break
            
            if diverse_x is not None:
                remaining = self.budget - call_count - 10
                if remaining > 10:
                    run_local_optimizer(diverse_x, remaining)

        # Phase 3: Final polish with L-BFGS-B if smoothness is expected (Rosenbrock is smooth)
        if time.time() < deadline and call_count < self.budget - 10:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 500,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-15,
                            'eps': 1e-8
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        return best_x, best_fx
