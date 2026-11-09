"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with step size adaptation for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin for 300s limit

        call_count = [0]
        max_calls = budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                return float('inf')
            call_count[0] += 1
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        # Initial point: Random within the box
        # Schwefel functions often have global minima far from origin, 
        # but random start is safer against hidden shifts.
        x = rng.uniform(-5.0, 5.0, dim)
        f_x = counted_func(x)
        
        # Initial step size: roughly 1/10th of the box width
        sigma = 1.0 
        best_x = x.copy()
        best_f = f_x
        
        # History for step size adaptation
        # We track successes in a window of size 'window'
        window = 10
        successes = [0] * window
        success_idx = 0
        total_successes_in_window = 0

        # Main loop: (1+1)-ES
        # We use all budget for this method as it provides continuous updates.
        while call_count[0] < max_calls and time.time() < deadline:
            # Generate candidate
            # Add noise
            noise = rng.standard_normal(dim) * sigma
            x_trial = x + noise
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = counted_func(x_trial)
            
            if f_trial < f_x:
                # Success
                x = x_trial
                f_x = f_trial
                best_x = x.copy()
                best_f = f_trial
                
                # Update success history
                successes[success_idx] = 1
                total_successes_in_window += 1
            else:
                # Failure
                successes[success_idx] = 0
                # If we failed, we don't update x, so x remains the best so far
                # Note: In (1+1), if we fail, we stay at current best.
                # So best_x/best_f are already correct.
                
                successes[success_idx] = 0
                # We need to decrement total_successes if the slot we are overwriting was 1?
                # Let's re-implement the circular buffer logic more carefully.
            
            # Advance circular buffer pointer
            success_idx = (success_idx + 1) % window
            
            # Calculate success rate in current window
            # We need to know what was in the slot we just overwrote to update total_successes_in_window
            # Actually, simpler: just sum the window each time? Window is small (10), so sum is cheap.
            success_rate = np.sum(successes) / window
            
            # Adapt sigma
            # Standard 1/5 rule:
            # If success rate > 1/5, increase sigma
            # If success rate < 1/5, decrease sigma
            if success_rate > 0.2:
                sigma *= 1.05
            else:
                sigma /= 1.05
            
            # Clamp sigma to reasonable bounds
            # Min sigma: very small for precision
            # Max sigma: large enough to escape local minima
            sigma = np.clip(sigma, 1e-5, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
