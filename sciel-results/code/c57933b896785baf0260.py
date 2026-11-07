"""
Strategy: Active Model Fitting with Exact Linear Regression
Instead of estimating gradients via finite differences (which can be noisy or sensitive to step size), 
we treat the function as a black box that we suspect is affine (linear + offset).
1. We evaluate the function at $d+1$ points forming a simplex (origin + basis vectors).
2. We solve for the coefficients of the affine model $f(x) \approx c + w^T x$ using exact linear algebra.
3. We compute the analytical optimum of this model within the box [-5, 5]^d.
4. We evaluate this predicted optimum.
5. If the model is exact (residuals are near zero), we have found the global minimum.
6. If the model is not exact (residuals are significant), we fall back to a robust global search 
   (e.g., a coarse grid search followed by local refinement) to ensure we don't miss the true minimum 
   due to non-linearity or implementation details.

This approach is structurally different from "gradient estimation + local search" because it 
explicitly validates the linear assumption and switches strategies if the assumption fails.
"""

import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0
        
        # Helper to update best
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 1: Build the affine model
        # We need d+1 points to determine d+1 coefficients (offset + d weights).
        # Points: 0, e1, e2, ..., ed
        points = [np.zeros(dim)]
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 1.0
            points.append(p)
            
        values = []
        for p in points:
            if time.time() > deadline or calls >= budget:
                break
            fx = func(p)
            values.append(fx)
            calls += 1
            update_best(p, fx)
        
        # If we couldn't get enough points to build the model, fallback to random search
        if len(values) < dim + 1:
            # Fallback: Random search
            while calls < budget and time.time() < deadline:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                update_best(x, fx)
            return best_x, best_fx

        # Solve for affine model: f(x) = c + w^T x
        # We have:
        # f(0) = c
        # f(e_i) = c + w_i
        # So c = values[0]
        # w_i = values[i+1] - values[0]
        
        c = values[0]
        w = np.array(values[1:]) - c
        
        # The model is f(x) = c + sum(w_i * x_i)
        
        # Phase 2: Find the optimum of the affine model
        # Minimize c + w^T x subject to -5 <= x_i <= 5
        # x_i^* = -5 if w_i > 0, 5 if w_i < 0, 0 if w_i == 0
        x_model_opt = np.zeros(dim)
        for i in range(dim):
            if w[i] > 1e-12:
                x_model_opt[i] = -5.0
            elif w[i] < -1e-12:
                x_model_opt[i] = 5.0
            else:
                x_model_opt[i] = 0.0
        
        # Evaluate the model's predicted optimum
        if time.time() > deadline or calls >= budget:
            return best_x, best_fx
            
        f_model_opt = func(x_model_opt)
        calls += 1
        update_best(x_model_opt, f_model_opt)
        
        # Phase 3: Validate the model
        # Check if the model fits the observed data well.
        # For a linear function, the residual should be 0 (within floating point precision).
        # Let's check a few random points to see if the model holds.
        # If it holds, we are done (global min found).
        # If it doesn't, we need to search more broadly.
        
        residuals = []
        num_checks = min(10, budget - calls)
        for _ in range(num_checks):
            if time.time() > deadline or calls >= budget:
                break
            x_check = rng.uniform(-5, 5, dim)
            f_check = func(x_check)
            calls += 1
            update_best(x_check, f_check)
            
            # Predicted value
            f_pred = c + np.dot(w, x_check)
            residuals.append(abs(f_check - f_pred))
        
        max_residual = max(residuals) if residuals else 0.0
        
        # If the model is accurate, we likely have the global minimum.
        # If not, we need a fallback search.
        # Threshold for "linear enough"
        # Note: f_opt can be in [-100, 100]. A residual of 1e-5 is tiny.
        # If the function is truly linear, residual ~ 1e-15.
        # If it's not linear, residual could be large.
        
        if max_residual < 1e-5:
            # Model is valid. The analytical optimum of a linear function on a box is a vertex.
            # We have evaluated that vertex. It is the global minimum.
            # We can use remaining budget to just return or do trivial checks.
            # To be safe, we'll just stop.
            pass
        else:
            # Model is invalid. The function is not purely linear in the observed coordinates.
            # This could happen if the "rotation" mentioned in the prompt is applied to the input 
            # in a way that makes the function non-linear in the *original* coordinates? 
            # Wait, f(x) = f5(R(x-shift)). If f5 is linear, f is linear. 
            # Why would the residual be non-zero?
            # 1. Floating point errors? Unlikely to be > 1e-5 for such small dims.
            # 2. The function is NOT linear. Maybe "Linear Slope" in this specific implementation 
            #    has a different definition, e.g., f(x) = sum(x_i) + noise? Or clipping?
            #    The prompt says "func(x) clips a finite length-5 vector to the box". 
            #    Clipping is linear within the box.
            # 3. Maybe the "hidden seeded shift, rotation and offset" implies that the *function* 
            #    itself is transformed, but if it's linear, it stays linear.
            #    
            #    However, if the model is wrong, we must search.
            #    Let's perform a global search using the remaining budget.
            #    Strategy: Coarse Grid Search + Local Refinement.
            
            remaining_budget = budget - calls
            if remaining_budget > 0:
                # Coarse grid search in 5D is hard. 
                # Let's use a more efficient global method: 
                # Since we suspect it might be close to linear but not quite, 
                # or maybe it's a different function entirely, let's use 
                # a combination of random sampling and local hill climbing.
                
                # Actually, if the model is wrong, our best bet is to use 
                # a robust global optimizer like a small population of 
                # (1+1)-ES agents or a simple grid search on a reduced resolution.
                
                # Given 5D, a grid of 3 points per dim is 3^5 = 243 points.
                # We have up to 500 - (5+1+1+10) = 483 calls left.
                # We can afford a grid search!
                
                # Let's do a grid search with 3 points per dimension: -5, 0, 5.
                # This covers the corners and center.
                
                grid_points = []
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                for m in range(3):
                                    grid_points.append(np.array([
                                        [-5, 0, 5][i],
                                        [-5, 0, 5][j],
                                        [-5, 0, 5][k],
                                        [-5, 0, 5][l],
                                        [-5, 0, 5][m]
                                    ]))
                
                # Shuffle grid points to ensure early progress
                rng.shuffle(grid_points)
                
                for x in grid_points:
                    if time.time() > deadline or calls >= budget:
                        break
                    fx = func(x)
                    calls += 1
                    update_best(x, fx)
                    
                # If we still have budget, refine the best point found
                if calls < budget and time.time() < deadline:
                    # Local refinement from best_x
                    x = best_x.copy()
                    fx = best_fx
                    sigma = 1.0
                    while calls < budget and time.time() < deadline:
                        y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)
                            sigma *= 1.05
                        else:
                            sigma *= 0.9
                            if sigma < 1e-4:
                                break

        return best_x, best_fx
