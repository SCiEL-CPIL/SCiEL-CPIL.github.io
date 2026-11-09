"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer with Global Initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 280.0  # Safety margin for 300s limit (evaluator kills at 330s)
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            # Ensure x is a numpy array of correct shape and within bounds
            # The task says func clips, but we do it here too for safety in our internal logic
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            self.calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            
            if val < self.best_fx:
                self.best_fx = val
                self.best_x = x.copy()
                
            return val

        self.best_x = np.zeros(self.dim)
        self.best_fx = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Global Search
        # Bent Cigar is convex but has a narrow valley. We need to find the center of the valley.
        # Spend ~10% of budget on initialization to find a good starting point.
        # 500 * 0.1 = 50 calls.
        num_initial = 20 
        candidates = []
        
        # Always try the center first
        x0_center = np.zeros(self.dim)
        f0_center = eval_func(x0_center)
        candidates.append((x0_center, f0_center))
        
        for _ in range(num_initial - 1):
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            x_init = rng.uniform(-4.5, 4.5, self.dim)
            f_init = eval_func(x_init)
            candidates.append((x_init, f_init))
            
        if not candidates:
            return self.best_x, self.best_fx
            
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        
        # Phase 2: Powell's Method
        # Powell is excellent for functions with narrow valleys (like Bent Cigar) because
        # it aligns its search directions with the valley.
        
        def powell_func(x):
            return eval_func(x)
            
        try:
            res = minimize(
                powell_func, 
                x0, 
                method='Powell',
                bounds=[(-5, 5)] * self.dim,
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-12,
                    'disp': False
                }
            )
            
            # Note: res.fun might not be updated if we stopped early, but eval_func updates self.best
            # However, res.x is the last point evaluated.
            if res.fun < self.best_fx:
                self.best_fx = res.fun
                self.best_x = res.x.copy()
                
        except StopIteration:
            pass
        except Exception:
            pass
            
        # Phase 3: Refine with Nelder-Mead if budget remains
        # Powell can get stuck in local minima if the valley is very narrow and the initial simplex
        # doesn't capture the direction well. NM with a small delta can polish.
        
        if self.calls < self.budget and time.time() - self.start_time < self.time_limit - 5:
             # Try a quick Nelder-Mead refinement from the Powell result
             x_nm = self.best_x.copy()
             
             # Initialize simplex around x_nm with small delta
             delta = 0.1
             simplex = [x_nm.copy()]
             simplex_f = [self.best_fx]
             
             for i in range(self.dim):
                 if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                     break
                 x_i = x_nm.copy()
                 x_i[i] += delta
                 x_i = np.clip(x_i, -5.0, 5.0)
                 f_i = eval_func(x_i)
                 simplex.append(x_i)
                 simplex_f.append(f_i)
                 
             # Simple NM loop for refinement
             rho = 1.0
             chi = 2.0
             gamma = 0.5
             sigma = 0.5
             
             for _ in range(50): # Small cap for refinement
                 if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                     break
                 
                 indices = np.argsort(simplex_f)
                 simplex = [simplex[i] for i in indices]
                 simplex_f = [simplex_f[i] for i in indices]
                 
                 if simplex_f[-1] - simplex_f[0] < 1e-10:
                     break
                 
                 centroid = np.mean(simplex[:-1], axis=0)
                 
                 xr = centroid + rho * (centroid - simplex[-1])
                 xr = np.clip(xr, -5.0, 5.0)
                 fr = eval_func(xr)
                 
                 if fr < simplex_f[0]:
                     if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                         break
                     xe = centroid + chi * (xr - centroid)
                     xe = np.clip(xe, -5.0, 5.0)
                     fe = eval_func(xe)
                     if fe < fr:
                         simplex[-1] = xe
                         simplex_f[-1] = fe
                     else:
                         simplex[-1] = xr
                         simplex_f[-1] = fr
                 elif fr < simplex_f[-2]:
                     simplex[-1] = xr
                     simplex_f[-1] = fr
                 else:
                     if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                         break
                     xc = centroid + gamma * (simplex[-1] - centroid)
                     xc = np.clip(xc, -5.0, 5.0)
                     fc = eval_func(xc)
                     if fc < simplex_f[-1]:
                         simplex[-1] = xc
                         simplex_f[-1] = fc
                     else:
                         # Shrink
                         for i in range(1, self.dim + 1):
                             if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                                 break
                             simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                             simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                             simplex_f[i] = eval_func(simplex[i])
                             
                         indices = np.argsort(simplex_f)
                         simplex = [simplex[i] for i in indices]
                         simplex_f = [simplex_f[i] for i in indices]
                         
                         if simplex_f[0] < self.best_fx:
                             self.best_fx = simplex_f[0]
                             self.best_x = simplex[0].copy()

        return self.best_x, self.best_fx
# EVOLVE-BLOCK-END
