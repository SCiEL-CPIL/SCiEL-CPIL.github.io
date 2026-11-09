"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for Linear Slope function.
    Strategy:
    1. Estimate gradient using center and axis points.
    2. Predict the optimal vertex based on gradient signs.
    3. Evaluate predicted vertex.
    4. Refine with (1+1)-ES if necessary.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        rng = np.random.default_rng(42)
        
        # 1. Evaluate center
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        calls_made = 1
        best_x = x0.copy()
        best_f = f0
        
        # 2. Evaluate axis points to estimate gradient
        # Step size 0.5 is well within [-5, 5]
        step = 0.5
        grad_est = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            xi = np.zeros(self.dim)
            xi[i] = step
            fi = func(xi)
            calls_made += 1
            
            # Update best
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
            
            # Estimate gradient component
            grad_est[i] = (fi - f0) / step
        
        # 3. Predict optimal vertex
        # If grad[i] > 0, function increases with x[i], so min is at -5
        # If grad[i] < 0, function decreases with x[i], so min is at +5
        x_pred = np.zeros(self.dim)
        for i in range(self.dim):
            if grad_est[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
        
        # Evaluate predicted vertex
        if calls_made < self.budget and time.time() < deadline:
            f_pred = func(x_pred)
            calls_made += 1
            if f_pred < best_f:
                best_f = f_pred
                best_x = x_pred.copy()
        
        # 4. Refine with (1+1)-ES
        # Start from best_x found so far
        x = best_x.copy()
        fx = best_f
        
        # Initial sigma: small, since we likely found the global min already
        sigma = 0.5 
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                best_x = x.copy()
                best_f = fx
                sigma *= 1.2  # Increase step size on success
            else:
                sigma *= 0.9  # Decrease step size on failure
                
            if sigma < 1e-6:
                # If sigma is too small, we've converged or stuck
                # We can break to save time, but budget is large
                # However, if we break, we stop calling func.
                # The problem says "repeating the last n_i if fewer calls are made".
                # So breaking early is fine and saves time.
                break
                
        return x, fx
# EVOLVE-BLOCK-END
