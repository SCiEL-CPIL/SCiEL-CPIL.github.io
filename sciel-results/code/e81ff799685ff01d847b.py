import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280  # Safety margin

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start Adaptive Coordinate Descent
        # We allocate a portion of the budget to each "trial"
        # Each trial consists of a random start followed by a local search
        # This helps escape local minima in multimodal functions like Rastrigin

        # Parameters
        min_trial_budget = 20  # Minimum calls per trial to allow some local search
        max_trials = self.budget // min_trial_budget
        # We will run as many trials as possible, but cap it to avoid too short trials
        # If budget is 500, min_trial 20 -> 25 trials.
        # Let's use a slightly larger trial budget to ensure convergence.
        # 500 / 10 = 50 calls per trial. 10 trials.
        calls_per_trial = 50
        n_trials = self.budget // calls_per_trial
        
        if n_trials < 1:
            n_trials = 1
            calls_per_trial = self.budget

        # Ensure we don't exceed budget
        # Total calls will be approx n_trials * calls_per_trial
        
        for t in range(n_trials):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break

            # 1. Random Start
            # Sample from a slightly larger range? No, box is [-5, 5].
            # Uniform is fine.
            x = rng.uniform(-5, 5, self.dim)
            try:
                f = safe_func(x)
            except (ValueError, TimeoutError):
                break

            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()

            # 2. Local Search: Adaptive Coordinate Descent
            # We use the remaining budget for this trial
            trial_budget = calls_per_trial - 1  # 1 call already used for start
            
            current_x = x.copy()
            current_f = f
            
            # Adaptive step size
            step = 1.0
            min_step = 1e-7
            
            # We will iterate until budget is exhausted or step is too small
            # To ensure we use the budget effectively, we loop while calls remain
            # and step is not too small.
            
            # We need to track calls specifically for this trial to stop at the right time
            # But safe_func tracks global calls.
            # We can just check self.calls against a limit.
            
            trial_start_calls = self.calls
            max_calls_this_trial = trial_start_calls + trial_budget
            
            while self.calls < max_calls_this_trial:
                if time.time() > self.deadline:
                    break
                if self.calls >= self.budget:
                    break
                
                # Pick a dimension to explore
                # Cyclic order is often good for coordinate descent
                d = (self.calls - trial_start_calls) % self.dim
                
                improved = False
                
                # Try positive direction
                new_x = current_x.copy()
                new_x[d] += step
                new_x = np.clip(new_x, -5, 5)
                
                # If we are already at the boundary and moving out, skip or handle carefully
                # np.clip handles the boundary, but if new_x[d] == current_x[d], no move
                if not np.allclose(new_x, current_x):
                    try:
                        f_new = safe_func(new_x)
                    except (ValueError, TimeoutError):
                        break
                    
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
                        improved = True
                        # Shrink step slightly to refine
                        step = max(step * 0.5, min_step)
                        # If we improved, we might want to continue in this direction?
                        # Standard CD tries next dimension.
                        # But for Rastrigin, staying in the valley is good.
                        # Let's just shrink and move to next dim.
                    
                # If not improved, try negative direction
                if not improved and self.calls < max_calls_this_trial:
                    new_x = current_x.copy()
                    new_x[d] -= step
                    new_x = np.clip(new_x, -5, 5)
                    
                    if not np.allclose(new_x, current_x):
                        try:
                            f_new = safe_func(new_x)
                        except (ValueError, TimeoutError):
                            break
                        
                        if f_new < current_f:
                            current_x = new_x
                            current_f = f_new
                            improved = True
                            step = max(step * 0.5, min_step)
                
                if not improved:
                    # No improvement in either direction for this dimension
                    # Shrink step to try smaller moves
                    step = max(step * 0.5, min_step)
                    
                    if step <= min_step:
                        # Converged to a local minimum
                        break

        return self.best_x, self.best_f
