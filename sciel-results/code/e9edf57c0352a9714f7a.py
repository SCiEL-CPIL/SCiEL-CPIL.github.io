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
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_fx = np.inf
        dim = self.dim

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                x = x[:dim]
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # We need to find the basin of attraction for the rotated Rosenbrock.
        # Since the minimizer is in [-4, 4]^5, we sample densely there.
        # 5D space is large. Random sampling is inefficient.
        # Let's use a combination of:
        # 1. Center
        # 2. Grid points (coarse)
        # 3. Random points
        
        n_explore = 80
        points = []
        
        # Center
        points.append(np.zeros(dim))
        
        # Coarse grid: 3 points per dimension -> 3^5 = 243 points. 
        # This might be too many if we want to leave budget for local.
        # 500 budget. 243 grid + 1 center = 244. Leaves 256 for local.
        # Let's try 2 points per dim? 2^5 = 32. Too few.
        # Let's try a mix. 
        # Actually, for Rosenbrock, the valley is narrow. 
        # If we miss the valley entry, we fail.
        # The valley roughly follows x_i = x_{i-1}^2.
        # Let's sample along this manifold? No, it's rotated.
        
        # Let's stick to random + structured.
        # 60 random points in [-4.5, 4.5]
        for _ in range(60):
            points.append(rng.uniform(-4.5, 4.5, dim))
            
        # Add corners of [-4, 4]^5
        for i in range(2**dim):
            corner = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            corner = corner * 8 - 4
            points.append(corner)
            
        # Add mid-points of axes?
        for i in range(dim):
            for val in [-4, 4]:
                p = np.zeros(dim)
                p[i] = val
                points.append(p)

        # Evaluate all exploration points
        for x0 in points:
            if call_count >= self.budget - 100: # Reserve 100 for local
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                tracked_func(best_x)
            except Exception:
                return best_x, best_fx

        # Phase 2: Local Optimization
        # We will run a sequence of local optimizers from the best point found.
        # 1. Nelder-Mead (robust, derivative-free)
        # 2. Powell (derivative-free, handles narrow valleys well)
        # 3. L-BFGS-B (gradient-based, fast convergence if close)
        
        # We will interleave or chain them based on remaining budget.
        # Strategy: Run NM first to get close. Then Powell to polish. Then L-BFGS-B.
        
        # Run Nelder-Mead
        if time.time() < deadline and call_count < self.budget - 50:
            remaining = self.budget - call_count - 20
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'maxfev': remaining,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except TimeoutError:
                pass
            except Exception:
                pass

        # Run Powell
        if time.time() < deadline and call_count < self.budget - 20:
            remaining = self.budget - call_count - 10
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='Powell',
                        options={
                            'maxfev': remaining,
                            'xtol': 1e-12,
                            'ftol': 1e-12
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        # Run L-BFGS-B
        if time.time() < deadline and call_count < self.budget - 5:
            remaining = self.budget - call_count - 2
            if remaining > 5:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 1000,
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
        
        # If we have a lot of budget left (shouldn't happen with maxfev limits, 
        # but just in case), do a final random restart + local if stuck?
        # No, better to just return.

        return best_x, best_fx
