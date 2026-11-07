import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        # Wrap func to track calls and handle safety
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                return 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        rng = np.random.RandomState(42)
        
        # Strategy: Multi-Start (1+1)-ES with Momentum
        # The Lunacek function is bi-modal. We need to explore both basins.
        # We will perform a series of "restarts". Each restart starts from a random point,
        # performs a local search with momentum to converge to a local minimum,
        # and keeps the global best.
        
        # Budget allocation:
        # 1. Initial random sample to get a rough idea (10 calls)
        # 2. Main loop: Multi-start local searches until budget or time runs out.
        
        # 1. Initial Exploration
        # Evaluate center and a few random points
        initial_points = [np.zeros(self.dim)]
        for _ in range(9):
            initial_points.append(rng.uniform(-5, 5, self.dim))
        
        for x in initial_points:
            if self.calls >= self.budget or not self._check_time():
                break
            safe_func(x)
            
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = safe_func(self.best_x)

        # 2. Multi-Start Local Search
        # Each restart:
        #   a. Pick a start point (random or perturbed best)
        #   b. Run a local (1+1)-ES with momentum for a fixed number of steps or until convergence
        #   c. Update global best
        
        # Parameters for local search
        max_steps_per_restart = 50
        initial_step_size = 1.0
        momentum = 0.5
        
        # Number of restarts depends on budget
        # We have ~490 calls left.
        # Let's do restarts until budget is exhausted.
        
        while self.calls < self.budget and self._check_time():
            # Choose start point
            # 80% chance random, 20% chance near current best (to polish)
            if rng.rand() < 0.8:
                x_curr = rng.uniform(-5, 5, self.dim)
            else:
                x_curr = self.best_x + rng.normal(0, 0.5, self.dim)
                x_curr = np.clip(x_curr, -5, 5)
            
            f_curr = safe_func(x_curr)
            
            # Local Search with Momentum
            v = np.zeros(self.dim) # Velocity/Momentum
            step_size = initial_step_size
            
            for step in range(max_steps_per_restart):
                if self.calls >= self.budget or not self._check_time():
                    break
                    
                # Generate proposal
                z = rng.standard_normal(self.dim)
                x_prop = x_curr + step_size * (v + z)
                x_prop = np.clip(x_prop, -5, 5)
                
                f_prop = safe_func(x_prop)
                
                if f_prop < f_curr:
                    # Accept
                    v = momentum * v + (1 - momentum) * (x_prop - x_curr) / step_size
                    x_curr = x_prop
                    f_curr = f_prop
                    step_size *= 1.05 # Slight expansion on success
                else:
                    # Reject
                    v = momentum * v
                    step_size *= 0.9 # Shrink on failure
                    
                    # Early stopping if step size is too small
                    if step_size < 1e-4:
                        break

            # Update global best
            if f_curr < self.best_f:
                self.best_f = f_curr
                self.best_x = x_curr.copy()

        # Final Polish: Very fine local search around global best
        if self.best_x is not None and self.calls < self.budget and self._check_time():
            x_curr = self.best_x.copy()
            f_curr = self.best_f
            v = np.zeros(self.dim)
            step_size = 0.1
            max_polish_steps = self.budget - self.calls
            
            for step in range(max_polish_steps):
                if self.calls >= self.budget or not self._check_time():
                    break
                    
                z = rng.standard_normal(self.dim)
                x_prop = x_curr + step_size * (v + z)
                x_prop = np.clip(x_prop, -5, 5)
                
                f_prop = safe_func(x_prop)
                
                if f_prop < f_curr:
                    v = 0.5 * v + 0.5 * (x_prop - x_curr) / step_size
                    x_curr = x_prop
                    f_curr = f_prop
                    step_size *= 1.02
                else:
                    v = 0.5 * v
                    step_size *= 0.95
                    
                if step_size < 1e-6:
                    break
            
            if f_curr < self.best_f:
                self.best_f = f_curr
                self.best_x = x_curr.copy()

        return self.best_x, self.best_f
