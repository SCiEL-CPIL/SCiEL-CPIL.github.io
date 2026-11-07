import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Nelder-Mead simplex optimizer for BBOB-style Rosenbrock (rotated).
    
    Strategy: Derivative-free optimization avoids the 10x overhead of finite-difference
    gradients. Nelder-Mead adapts the simplex shape to navigate the narrow Rosenbrock
    valley without explicit gradient computation. Multi-start handles the hidden
    rotation and shift by sampling multiple initial regions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # State for best solution
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to call func and track budget strictly
        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                # Soft stop: return best known value without calling func
                return best_f if best_f != np.inf else 1e100
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Generate diverse starting points for multi-start
        rng = np.random.default_rng(42)
        
        # Key starting points:
        # 1. Origin (0,0,0,0,0) - often good for shifted/rotated problems
        # 2. (1,1,1,1,1) - standard Rosenbrock minimum location
        # 3. (-1,-1,-1,-1,-1) - opposite corner
        # 4. Random points in [-4, 4]^5
        start_points = [
            np.zeros(dim),
            np.ones(dim),
            -np.ones(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.array([2, 2, 2, 2, 2], dtype=float),
            np.array([-2, -2, -2, -2, -2], dtype=float),
        ]
        
        # Allocate budget across starts
        # Nelder-Mead is efficient: ~100-200 calls per convergence
        # With 500 budget, we can afford 2-3 full runs or more partial ones
        
        # Strategy: Use 40% budget for first start, 30% for second, rest for others
        # This prioritizes deep exploration of promising starts
        
        # Sort starts by heuristic priority (origin and (1,1,..) first)
        # But we'll just iterate and allocate dynamically
        
        remaining_budget = budget
        
        for idx, x0 in enumerate(start_points):
            if time.time() > deadline - 5:
                break
            if calls_made >= budget:
                break
            
            # Determine budget for this start
            # First start gets more budget, subsequent ones less
            if idx == 0:
                alloc = min(remaining_budget, int(budget * 0.4))
            elif idx == 1:
                alloc = min(remaining_budget, int(budget * 0.3))
            else:
                alloc = min(remaining_budget, int(budget * 0.15))
            
            if alloc < 20:  # Need at least ~20 calls for Nelder-Mead to be useful
                break
            
            # Save current best to restore if this start doesn't improve
            prev_best_x = best_x.copy() if best_x is not None else None
            prev_best_f = best_f
            
            # Run Nelder-Mead with allocated budget
            # maxiter controls iterations, but we rely on calls_made cap
            # Nelder-Mead uses ~dim+1 = 6 vertices, each move = 1-2 function calls
            # So ~alloc/2 iterations is reasonable
            
            try:
                # Use a reasonable maxiter that won't exceed our budget
                # Nelder-Mead: each iteration does 1 reflection (1 call), 
                # possibly 1 expansion (1 call), 1 contraction (1 call), 1 shrink (dim calls)
                # Worst case ~dim+3 calls per iteration
                # Let's be conservative: maxiter = alloc / 10
                max_iter = max(10, alloc // 10)
                
                res = minimize(
                    tracked_func,
                    x0,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'initial_simplex': None  # Use default
                    }
                )
                
                # After Nelder-Mead, if we have budget left and found improvement,
                # do a local refinement with a tighter tolerance
                if calls_made < budget and best_f < prev_best_f:
                    # Use remaining budget for refinement from best found
                    ref_budget = budget - calls_made
                    if ref_budget > 30:
                        try:
                            res2 = minimize(
                                tracked_func,
                                best_x if best_x is not None else x0,
                                method='Nelder-Mead',
                                bounds=[(-5, 5)] * dim,
                                options={
                                    'maxiter': max(10, ref_budget // 8),
                                    'xatol': 1e-8,
                                    'fatol': 1e-10,
                                    'adaptive': True
                                }
                            )
                        except Exception:
                            pass
                            
            except Exception:
                pass
            
            # If this start didn't improve, continue to next
            # If it did improve, we've already updated best_x/best_f
            
            remaining_budget = budget - calls_made
            if remaining_budget < 20:
                break
        
        # Final check: if we have some budget left, try one more random restart
        if calls_made < budget and time.time() < deadline - 10:
            for _ in range(3):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                x_rand = rng.uniform(-4, 4, dim)
                # Quick Nelder-Mead with small budget
                try:
                    minimize(
                        tracked_func,
                        x_rand,
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 15,
                            'xatol': 1e-4,
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass
        
        # Return best found
        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0
        
        return best_x, best_f
