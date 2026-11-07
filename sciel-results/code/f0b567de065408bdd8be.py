import numpy as np
import time

class Optimizer:
    """Simulated Annealing optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Initialize at origin (common heuristic for BBOB, though shifted)
        # Or random? Origin is safe.
        x_curr = np.zeros(dim)
        
        # Evaluate first point
        f_curr = func(x_curr)
        if f_curr < best_f:
            best_f = f_curr
            best_x = x_curr.copy()
            
        calls = 1
        
        # SA Parameters
        # Temperature schedule: Exponential cooling
        # T_start needs to be high enough to accept worse solutions easily.
        # Rastrigin values can be large. Let's estimate typical delta.
        # f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Local minima at even integers? No, Rastrigin minima are at integers.
        # Spacing is 1.0 in standard Rastrigin.
        # In BBOB, it's scaled. Let's assume spacing ~2.
        # Delta f between minima and ridge peaks can be ~10-20.
        # T_start = 50 seems reasonable.
        
        T_start = 100.0
        T_end = 1e-6
        alpha = 0.95  # Cooling rate
        
        # Move parameters
        # Initial step size
        sigma_start = 5.0
        sigma_end = 0.01
        # Jump probability to escape local minima
        p_jump = 0.1
        
        for i in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Update Temperature
            T = T_start * (T_end / T_start) ** (i / max(1, budget - 1))
            
            # Update Step Size (Annealing the noise)
            sigma = sigma_start * (sigma_end / sigma_start) ** (i / max(1, budget - 1))
            
            # Generate New Point
            if np.random.rand() < p_jump:
                # Large jump: uniform in box or large Gaussian
                # Uniform jump is good for global exploration
                x_new = np.random.uniform(-5, 5, dim)
            else:
                # Local refinement
                x_new = x_curr + np.random.normal(0, sigma, dim)
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            f_new = func(x_new)
            calls += 1
            
            # Acceptance Criterion
            if f_new < f_curr:
                x_curr = x_new
                f_curr = f_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            else:
                # Metropolis criterion
                # Avoid overflow in exp
                delta = f_new - f_curr
                if delta < 0:
                    accept_prob = 1.0
                else:
                    if T > 0:
                        accept_prob = np.exp(-delta / T)
                    else:
                        accept_prob = 0.0
                
                if np.random.rand() < accept_prob:
                    x_curr = x_new
                    f_curr = f_new
        
        return best_x, best_f
