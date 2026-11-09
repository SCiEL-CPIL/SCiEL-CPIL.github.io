import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # Safety margin

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Fast Initialization
        # Reduced from 100 to 20 to start local search earlier
        n_init = 20
        init_points = []
        init_vals = []
        
        for _ in range(n_init):
            if time.time() > deadline or self.calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_func(x)
            init_points.append(x)
            init_vals.append(f)
            
        if not init_points:
            x = np.zeros(self.dim)
            safe_func(x)
            return self.best_x, self.best_f

        best_idx = np.argmin(init_vals)
        current_x = init_points[best_idx].copy()
        current_f = self.best_f

        # Phase 2: Adaptive Local Search with Restarts
        # Strategy: 
        # 1. Perform local search around current_x.
        # 2. If stuck (no improvement for a while), restart from a new random point.
        # 3. Repeat until budget/time runs out.
        
        # Parameters for local search
        initial_step = 1.0
        min_step = 1e-5
        step_decay = 0.5
        stagnation_limit = 10 # Cycles without improvement before declaring stuck
        restarts_allowed = 10 # Max number of restarts to prevent infinite loops
        
        restarts_used = 0
        
        while self.calls < self.budget and time.time() < deadline:
            step = initial_step
            stagnant_cycles = 0
            
            # Local Search Loop
            while step > min_step and self.calls < self.budget and time.time() < deadline:
                improved = False
                stagnant_cycles += 1
                
                if stagnant_cycles > stagnation_limit:
                    break # Stuck, break out to restart
                    
                # Try all dimensions
                for d in range(self.dim):
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    new_x = current_x.copy()
                    new_x[d] += step
                    new_x = np.clip(new_x, -5, 5)
                    f_new = safe_func(new_x)
                    
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
                        improved = True
                        stagnant_cycles = 0 # Reset stagnation counter on improvement
                        
                    # Try negative direction
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                        
                    new_x = current_x.copy()
                    new_x[d] -= step
                    new_x = np.clip(new_x, -5, 5)
                    f_new = safe_func(new_x)
                    
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
                        improved = True
                        stagnant_cycles = 0
                        
                if not improved:
                    step *= step_decay
                else:
                    # If we improved, we could keep step or increase slightly.
                    # Keeping step is safer to ensure we don't jump over minima.
                    # However, if we are moving fast, we might want to keep momentum.
                    # Let's keep step constant if we improved, to allow for larger moves.
                    pass 

            # If we exited the local search loop due to stagnation or min step
            # And we still have budget, we should restart
            if self.calls >= self.budget or time.time() > deadline:
                break
                
            if restarts_used >= restarts_allowed:
                # If we've used all restarts, continue local search with tiny steps?
                # Or just break. Given the landscape, further micro-optimization might not be worth it.
                # Let's try to do a final polish with very small steps if we have budget.
                final_step = 1e-4
                for _ in range(5): # A few more cycles
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(self.dim):
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            if self.calls >= self.budget or time.time() > deadline:
                                break
                            new_x = current_x.copy()
                            new_x[d] += direction * final_step
                            new_x = np.clip(new_x, -5, 5)
                            f_new = safe_func(new_x)
                            if f_new < current_f:
                                current_x = new_x
                                current_f = f_new
                                improved = True
                    if not improved:
                        break
                break

            # Perform Restart
            restarts_used += 1
            # New starting point: Random uniform
            # This helps escape local minima in rotated landscapes
            new_x = rng.uniform(-5, 5, self.dim)
            f_new = safe_func(new_x)
            
            # If the new point is better, start local search from there
            # Otherwise, we might want to go back to the previous best?
            # For BBOB, it's often better to explore new regions.
            # However, if the new point is much worse, we are wasting time.
            # Let's adopt a strategy: If new point is not significantly worse, use it.
            # Actually, simpler: Just use the new point. The local search will minimize it.
            # If it converges to a worse local min, we'll restart again.
            
            current_x = new_x.copy()
            current_f = f_new

        return self.best_x, self.best_f
