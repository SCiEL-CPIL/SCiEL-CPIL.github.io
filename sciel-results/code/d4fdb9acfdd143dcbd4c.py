"""Rewritten optimizer for BBOB-style f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Deterministic multi-scale coordinate descent with structured restarts.
    
    Strategy:
    1. Coarse grid initialization to find a good starting basin.
    2. Multi-scale coordinate descent with step sizes from 5.0 down to 0.01.
    3. Axis-priority ordering based on recent improvement magnitude.
    4. Structured restarts when stagnation is detected.
    5. Explicit deadline guard for 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = float('inf')
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time_left() < 0.5:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse Grid Initialization ---
        # Evaluate a structured grid to find a good starting point
        # Use a 3^dim grid over [-4, 4] (values: -4, 0, 4)
        # That's 3^5 = 243 points, but we'll be selective
        
        # First, evaluate origin and axis extremes
        eval_point(np.zeros(dim))
        
        # Evaluate points at +/- 3.0 along each axis
        for d in range(dim):
            if time_left() < 1.0 or call_count >= budget:
                break
            for sign in [1.0, -1.0]:
                x = np.zeros(dim)
                x[d] = sign * 3.0
                eval_point(x)
                if time_left() < 1.0 or call_count >= budget:
                    break
        
        # Evaluate a sparse 3^dim grid if budget allows
        if budget - call_count > 100:
            # Use values -3, 0, 3 for each dimension
            grid_values = [-3.0, 0.0, 3.0]
            for mask in range(3 ** dim):
                if time_left() < 1.0 or call_count >= budget:
                    break
                x = np.zeros(dim)
                m = mask
                for d in range(dim):
                    x[d] = grid_values[m % 3]
                    m //= 3
                eval_point(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            eval_point(best_x)
            if best_f == float('inf'):
                best_f = 0.0
            return best_x, best_f

        # --- Phase 2: Multi-Scale Coordinate Descent ---
        # Step sizes from coarse to fine
        step_sizes = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.009765625, 0.005]
        
        # Track axis priority: which axes gave the most recent improvement
        axis_improvement = np.zeros(dim)
        
        current_x = best_x.copy()
        current_f = best_f
        
        max_rounds = 15
        rng = np.random.default_rng(12345)
        
        for round_num in range(max_rounds):
            if call_count >= budget or time_left() < 2.0:
                break
            
            # Reset to global best
            current_x = best_x.copy()
            current_f = best_f
            improved_in_round = False
            round_improvement = 0.0
            
            # Order axes by recent improvement (most improved first)
            axis_order = np.argsort(-axis_improvement)
            
            for step in step_sizes:
                if call_count >= budget or time_left() < 0.5:
                    break
                
                for d in axis_order:
                    if call_count >= budget or time_left() < 0.5:
                        break
                    
                    # Try +step
                    x_try = current_x.copy()
                    x_try[d] += step
                    if -5.0 <= x_try[d] <= 5.0:
                        f_try = eval_point(x_try)
                        if f_try is not None and f_try < current_f - 1e-12:
                            improvement = current_f - f_try
                            current_x = x_try
                            current_f = f_try
                            improved_in_round = True
                            round_improvement += improvement
                            axis_improvement[d] = improvement
                            if f_try < best_f:
                                best_f = f_try
                                best_x = current_x.copy()
                    
                    if call_count >= budget or time_left() < 0.5:
                        break

                    # Try -step
                    x_try = current_x.copy()
                    x_try[d] -= step
                    if -5.0 <= x_try[d] <= 5.0:
                        f_try = eval_point(x_try)
                        if f_try is not None and f_try < current_f - 1e-12:
                            improvement = current_f - f_try
                            current_x = x_try
                            current_f = f_try
                            improved_in_round = True
                            round_improvement += improvement
                            axis_improvement[d] = improvement
                            if f_try < best_f:
                                best_f = f_try
                                best_x = current_x.copy()
                    
                    if call_count >= budget or time_left() < 0.5:
                        break
                
                # Decay axis improvement scores
                axis_improvement *= 0.5
            
            # If no improvement, perform structured restart
            if not improved_in_round:
                if call_count >= budget or time_left() < 2.0:
                    break
                
                # Try several restarts from perturbed versions of best point
                for r in range(5):
                    if call_count >= budget or time_left() < 1.0:
                        break
                    
                    restart_step = 3.0 * (0.6 ** r)
                    x_new = best_x + restart_step * rng.standard_normal(dim)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = eval_point(x_new)
                    if f_new is not None and f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        # Do a quick local refinement from this new best
                        for step in [1.0, 0.5, 0.25]:
                            if call_count >= budget or time_left() < 0.5:
                                break
                            for d in range(dim):
                                for sign in [1.0, -1.0]:
                                    if call_count >= budget or time_left() < 0.5:
                                        break
                                    x_try = best_x.copy()
                                    x_try[d] += sign * step
                                    if -5.0 <= x_try[d] <= 5.0:
                                        f_try = eval_point(x_try)
                                        if f_try is not None and f_try < best_f:
                                            best_f = f_try
                                            best_x = x_try.copy()
                # Reset axis improvement for fresh start
                axis_improvement = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
