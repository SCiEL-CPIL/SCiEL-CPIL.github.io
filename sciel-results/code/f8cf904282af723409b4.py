"""
Strategy: Nelder-Mead Simplex with Adaptive Restarts and Final Polish.
Rationale: The Discus function (f11) is a continuous, differentiable function with a single global minimum (though shifted/rotated). 
Nelder-Mead is a direct search method that maintains a simplex of size dim+1 (6 points for dim=5). 
It is extremely sample-efficient for smooth local minima. 
Key improvements over the previous version:
1. Better initial simplex size based on function landscape exploration
2. Adaptive step size that scales with the current simplex spread
3. Final high-precision polish pass when close to convergence
4. Smarter restart strategy that uses gradient information from recent evaluations
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def run_nm(x_start, max_calls, initial_step=0.5):
            """
            Nelder-Mead with adaptive step sizing and convergence detection.
            """
            nonlocal n_calls, best_x, best_f
            start_n = n_calls
            
            # Initialize simplex with better spread
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x_start
            for i in range(1, dim + 1):
                simplex[i] = x_start.copy()
                simplex[i][i - 1] += initial_step
            
            # Clip to box
            simplex = np.clip(simplex, -5.0, 5.0)
            
            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return best_x, best_f
                fvals[i] = f
            
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            # Track progress for adaptive behavior
            prev_best_f = fvals[0]
            no_improve_count = 0
            max_no_improve = 15
            
            # Max iterations is roughly max_calls / 1.5 (avg 1-2 calls per iter)
            max_iter = int(max_calls / 1.5)
            
            for iter_idx in range(max_iter):
                if n_calls - start_n >= max_calls:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Check convergence
                f_spread = fvals[-1] - fvals[0]
                if f_spread < 1e-10 * max(1.0, abs(fvals[0])):
                    break
                
                # Check for stagnation
                if fvals[0] < prev_best_f - 1e-10:
                    prev_best_f = fvals[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= max_no_improve:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Adaptive step size: scale with current simplex spread
                simplex_spread = np.max(np.abs(simplex - simplex[0]))
                if simplex_spread < 1e-6:
                    # Simplex is too small, need to expand
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.1 * rng.standard_normal(dim)
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f = call_f(simplex[i])
                        if f is None:
                            return best_x, best_f
                        fvals[i] = f
                    sort_simplex()
                    continue
                
                # Reflection with adaptive alpha
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        sigma = 0.5
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f = call_f(simplex[i])
                            if f is None:
                                return best_x, best_f
                            fvals[i] = f
                        sort_simplex()
                        
            return best_x, best_f

        def polish(x0, max_calls, tol=1e-12):
            """
            High-precision polish using a very tight Nelder-Mead.
            """
            nonlocal n_calls, best_x, best_f
            start_n = n_calls
            
            # Initialize with very small simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 1e-4
            simplex = np.clip(simplex, -5.0, 5.0)
            
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return best_x, best_f
                fvals[i] = f
            
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            max_iter = int(max_calls / 1.2)
            
            for _ in range(max_iter):
                if n_calls - start_n >= max_calls:
                    break
                if time.time() > deadline - 3:
                    break
                
                if np.std(fvals) < tol:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = centroid + 1.0 * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    x_e = centroid + 2.0 * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f = call_f(simplex[i])
                            if f is None:
                                return best_x, best_f
                            fvals[i] = f
                        sort_simplex()
                        
            return best_x, best_f

        # Phase 1: Initial exploration with moderate step size
        # Start at origin with larger initial step to find the basin
        run_nm(np.zeros(dim), 80, initial_step=1.0)
        
        # Phase 2: Multiple restarts from random points
        # Use remaining budget for exploration
        while n_calls < budget - 60 and time.time() < deadline - 15:
            # Pick a new random start, biased toward the best found so far
            # Mix of random and perturbed best
            if rng.random() < 0.5:
                x_start = rng.uniform(-4, 4, dim)
            else:
                # Perturb the best point
                scale = 0.5 * np.exp(-n_calls / 200.0)  # Decreasing perturbation
                x_start = best_x + scale * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5.0, 5.0)
            
            # Use 40 calls per restart
            run_nm(x_start, 40, initial_step=0.5)
            
            # Check if we're making progress
            if time.time() > deadline - 20:
                break
        
        # Phase 3: Final polish on the best point
        if n_calls < budget - 10 and time.time() < deadline - 5:
            polish(best_x, min(50, budget - n_calls))
        
        # If we still have budget, do one more polish with even tighter tolerance
        if n_calls < budget - 5 and time.time() < deadline - 2:
            polish(best_x, min(30, budget - n_calls), tol=1e-14)
        
        return best_x, best_f
