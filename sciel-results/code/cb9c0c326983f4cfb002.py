import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = 0

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 270  # Safety margin
        
        def evaluate(x):
            x = np.clip(x, -5, 5)
            f = func(x)
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f
        
        # Multi-start (1+1)-ES with momentum
        n_starts = 12
        budget_per_start = max(10, self.budget // n_starts)
        
        for start_idx in range(n_starts):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            # Initialize with diverse starting points
            if start_idx == 0:
                x = np.random.uniform(-5, 5, self.dim)
            else:
                # Mix of random restarts and perturbations of best found
                if np.random.rand() < 0.5:
                    x = np.random.uniform(-5, 5, self.dim)
                else:
                    x = self.best_x + np.random.normal(0, 2.0, self.dim)
                    x = np.clip(x, -5, 5)
            
            f_x = evaluate(x)
            
            # (1+1)-ES with momentum
            sigma = 1.5
            momentum = np.zeros(self.dim)
            momentum_decay = 0.7
            best_in_run = f_x
            best_x_in_run = x.copy()
            stagnation_count = 0
            max_stagnation = 20
            
            for step in range(budget_per_start):
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                
                # Generate perturbation
                z = np.random.randn(self.dim)
                
                # Apply momentum
                momentum = momentum_decay * momentum + (1 - momentum_decay) * z
                step_size = sigma * np.linalg.norm(momentum)
                if step_size > 0:
                    direction = momentum / step_size
                else:
                    direction = np.random.randn(self.dim)
                    direction /= np.linalg.norm(direction)
                
                x_new = x + sigma * direction * np.sqrt(self.dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                
                if f_new < f_x:
                    # Success: increase momentum in this direction, keep x
                    momentum = momentum_decay * momentum + (1 - momentum_decay) * direction * np.sqrt(self.dim)
                    x = x_new
                    f_x = f_new
                    if f_x < best_in_run:
                        best_in_run = f_x
                        best_x_in_run = x.copy()
                    stagnation_count = 0
                    # Slightly increase sigma on success
                    sigma *= 1.05
                else:
                    # Failure: decrease sigma
                    sigma *= 0.85
                    stagnation_count += 1
                
                # Adaptive sigma bounds
                sigma = np.clip(sigma, 0.01, 5.0)
                
                # Restart this sub-run if stagnated
                if stagnation_count >= max_stagnation:
                    break
            
            # After each sub-run, if we found a better point, update global best
            # The global best is already tracked in evaluate()
        
        return self.best_x, self.best_f
