"""
Optimizer using multi-start Nelder-Mead with adaptive refinement for BBOB f20 (Schwefel).
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, step, calls_ref, budget, deadline, n):
        """Run Nelder-Mead from starting point x0 with given step size.
        Returns (best_x, best_fx, calls_used)."""
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0
            simplex[i + 1, i] += step
        
        fvals = np.zeros(n + 1)
        calls = 0
        
        for i in range(n + 1):
            if calls_ref[0] + calls >= budget or time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls += 1
        
        if calls == 0:
            return x0.copy(), float('inf'), 0
        
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_fx = fvals[best_idx]
        
        max_iter = budget - calls_ref[0] - calls
        stagnant = 0
        
        while calls < max_iter:
            if time.time() > deadline:
                break
            
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = simplex[0].copy()
                stagnant = 0
            else:
                stagnant += 1
            
            if stagnant > 50:
                break
            
            worst_idx = n
            worst_x = simplex[worst_idx].copy()
            worst_f = fvals[worst_idx]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            x_ref = centroid + alpha * (centroid - worst_x)
            x_ref = np.clip(x_ref, -5, 5)
            
            if calls >= max_iter or time.time() > deadline:
                break
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                
                if calls >= max_iter or time.time() > deadline:
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    break
                
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    simplex[worst_idx] = x_exp
                    fvals[worst_idx] = f_exp
                else:
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    
            elif f_ref < fvals[n - 1]:
                simplex[worst_idx] = x_ref
                fvals[worst_idx] = f_ref
                
            else:
                x_con = centroid + rho * (worst_x - centroid)
                x_con = np.clip(x_con, -5, 5)
                
                if calls >= max_iter or time.time() > deadline:
                    break
                
                f_con = func(x_con)
                calls += 1
                
                if f_con < worst_f:
                    simplex[worst_idx] = x_con
                    fvals[worst_idx] = f_con
                else:
                    shrink_done = True
                    for i in range(1, n + 1):
                        if calls >= max_iter or time.time() > deadline:
                            shrink_done = False
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if not shrink_done:
                        break
        
        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 285s safety margin for 330s limit
        
        n = self.dim
        budget = self.budget
        calls_ref = [0]
        
        def eval_func(x):
            calls_ref[0] += 1
            return func(x)
        
        # Phase 1: Broad exploration with diverse starting points
        # Use deterministic diverse starts to cover the search space
        np.random.seed(42)
        start_points = []
        
        # Origin
        start_points.append(np.zeros(n))
        
        # Centered random points with varying scales
        scales = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5]
        for scale in scales:
            x = np.random.uniform(-scale, scale, n)
            start_points.append(x)
        
        # Axis-aligned points
        for i in range(n):
            x = np.zeros(n)
            x[i] = 3.0
            start_points.append(x)
        
        # Limit to budget constraints: each NM needs at least (n+1) calls
        # Budget 500, n=5, so max ~75 NM runs theoretically, but we'll be more conservative
        # Allocate ~30% of budget to exploration, 70% to refinement
        explore_budget = int(budget * 0.35)
        refine_budget = budget - explore_budget
        
        # Run NM from each start point, allocating budget proportionally
        # Give more budget to earlier starts (they're more diverse)
        total_starts = len(start_points)
        per_start_budget = max(20, explore_budget // total_starts)
        
        best_x = np.zeros(n)
        best_fx = float('inf')
        
        for idx, x0 in enumerate(start_points):
            if calls_ref[0] >= explore_budget or time.time() > deadline:
                break
            
            # Adaptive step size: larger for early starts, smaller for later
            base_step = 2.0 * (1.0 - 0.5 * idx / total_starts)
            base_step = max(0.5, base_step)
            
            remaining = explore_budget - calls_ref[0]
            if remaining < (n + 1) + 5:
                break
            
            bx, bf, used = self._nelder_mead(
                eval_func, x0, base_step, calls_ref, 
                explore_budget, deadline, n
            )
            
            if bf < best_fx:
                best_fx = bf
                best_x = bx.copy()
        
        # Phase 2: Refine the best point found with smaller steps
        if calls_ref[0] < budget and time.time() < deadline:
            # Try multiple refinement scales
            refine_steps = [1.0, 0.5, 0.25, 0.1, 0.05]
            
            for step in refine_steps:
                if calls_ref[0] >= budget or time.time() > deadline:
                    break
                
                remaining = budget - calls_ref[0]
                if remaining < (n + 1) + 5:
                    break
                
                # Start refinement from current best
                bx, bf, used = self._nelder_mead(
                    eval_func, best_x, step, calls_ref,
                    budget, deadline, n
                )
                
                if bf < best_fx:
                    best_fx = bf
                    best_x = bx.copy()
        
        # Phase 3: Final coordinate-wise polish
        if calls_ref[0] < budget and time.time() < deadline:
            for _ in range(3):  # 3 polish passes
                if calls_ref[0] >= budget or time.time() > deadline:
                    break
                
                for d in range(n):
                    if calls_ref[0] >= budget or time.time() > deadline:
                        break
                    
                    # Try small perturbations in each direction
                    for delta in [0.1, -0.1, 0.05, -0.05]:
                        if calls_ref[0] >= budget or time.time() > deadline:
                            break
                        
                        x_trial = best_x.copy()
                        x_trial[d] += delta
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        if np.allclose(x_trial, best_x):
                            continue
                        
                        f_trial = eval_func(x_trial)
                        
                        if f_trial < best_fx:
                            best_fx = f_trial
                            best_x = x_trial.copy()
        
        return best_x, best_fx
