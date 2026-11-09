import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        calls = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if time.time() > deadline:
                if calls[0] >= self.budget:
                    return float('inf')
            
            if calls[0] >= self.budget:
                return float('inf')
            
            calls[0] += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Latin Hypercube Sampling
        # 20 points
        n_lhs = 20
        if self.budget > n_lhs:
            # Generate LHS
            # For each dimension, divide [-5, 5] into n_lhs bins
            # Pick one random point in each bin
            lhs_points = []
            for i in range(n_lhs):
                point = []
                for d in range(self.dim):
                    # Bin [i/n_lhs, (i+1)/n_lhs] mapped to [-5, 5]
                    low = -5 + (10 * i / n_lhs)
                    high = -5 + (10 * (i + 1) / n_lhs)
                    point.append(np.random.uniform(low, high))
                lhs_points.append(point)
            
            lhs_points = np.array(lhs_points)
            
            # Evaluate all
            for x in lhs_points:
                if calls[0] >= self.budget or time.time() > deadline:
                    break
                wrapped_func(x)
        
        # If we used all budget in LHS, return best
        if calls[0] >= self.budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Powell's Method
        # Start from best_x
        # x0 is best_x
        # Bounds are [-5, 5]
        
        # Powell options
        # maxiter: max number of iterations. 
        # Each iteration involves several function calls.
        # We have ~480 calls left.
        # Powell in 5D might use ~5-10 calls per iteration?
        # So ~50-100 iterations.
        
        try:
            res = minimize(wrapped_func, best_x, method='Powell', 
                           bounds=[(-5, 5)] * self.dim,
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12})
            
            # Update best if Powell found something better (though wrapped_func tracks it)
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception as e:
            # If minimize crashes, we keep the best found so far
            pass
        
        # Phase 3: Adaptive refinement based on estimated gradient magnitude
        remaining_budget = self.budget - calls[0]
        if remaining_budget > 20 and time.time() < deadline:
            # Estimate local gradient magnitude via finite differences
            # This tells us if we're close to optimum or still far
            try:
                eps = 1e-6
                grad_est = np.zeros(self.dim)
                for d in range(self.dim):
                    x_plus = best_x.copy()
                    x_minus = best_x.copy()
                    x_plus[d] += eps
                    x_minus[d] -= eps
                    # Clip to bounds
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)
                    
                    if calls[0] < self.budget and time.time() < deadline:
                        fp = wrapped_func(x_plus)
                    else:
                        fp = best_f + eps * 100  # penalty
                    if calls[0] < self.budget and time.time() < deadline:
                        fm = wrapped_func(x_minus)
                    else:
                        fm = best_f + eps * 100  # penalty
                    
                    grad_est[d] = (fp - fm) / (2 * eps)
                
                grad_norm = np.linalg.norm(grad_est)
                
                # For sphere: grad_norm ≈ 2 * ||Q|| * ||x - x_opt||
                # Large grad_norm → still far from optimum → use larger simplex for NM
                # Small grad_norm → very close → use tiny simplex for precision
                if grad_norm > 10:
                    nm_init_simplex = 0.5
                    nm_maxiter = 50
                elif grad_norm > 1:
                    nm_init_simplex = 0.1
                    nm_maxiter = 80
                elif grad_norm > 0.1:
                    nm_init_simplex = 0.01
                    nm_maxiter = 100
                else:
                    nm_init_simplex = 0.001
                    nm_maxiter = 150
                
                nm_res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                 bounds=[(-5, 5)] * self.dim,
                                 options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-14,
                                         'initial_simplex': nm_init_simplex})
                
                if nm_res.fun < best_f:
                    best_f = nm_res.fun
                    best_x = nm_res.x.copy()
                    
            except Exception as e:
                pass
        
        # Phase 4: Final precision polish with very tight tolerance
        remaining_budget = self.budget - calls[0]
        if remaining_budget > 10 and time.time() < deadline:
            try:
                # Use a very small perturbation for final precision
                final_x = best_x.copy()
                # Try a few random tiny perturbations to escape any local flat spots
                for _ in range(min(5, remaining_budget // 2)):
                    if calls[0] >= self.budget or time.time() > deadline:
                        break
                    perturbation = np.random.normal(0, 1e-4, self.dim)
                    candidate_x = np.clip(final_x + perturbation, -5, 5)
                    val = wrapped_func(candidate_x)
                    if val < best_f:
                        best_f = val
                        final_x = candidate_x.copy()
                
                # One more NM pass with extremely tight settings
                if calls[0] < self.budget - 5 and time.time() < deadline:
                    nm_res2 = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                     bounds=[(-5, 5)] * self.dim,
                                     options={'maxiter': 30, 'xatol': 1e-12, 'fatol': 1e-16,
                                             'initial_simplex': 1e-4})
                    if nm_res2.fun < best_f:
                        best_f = nm_res2.fun
                        best_x = nm_res2.x.copy()
                        
            except Exception as e:
                pass

        return best_x, best_f
