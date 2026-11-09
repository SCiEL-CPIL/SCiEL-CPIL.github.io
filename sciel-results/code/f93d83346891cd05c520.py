import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Phase 1: Initial Sampling - diverse random points
        n_init = 30
        for _ in range(n_init):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)

        # Phase 2: Coordinate descent local search
        def coordinate_descent(start_x, max_iter=30):
            nonlocal best_x, best_y, calls_made
            if best_x is None:
                return
            x = start_x.copy()
            current_f = call_func(x)
            if current_f is None:
                return
            
            step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625]
            for step in step_sizes:
                if not time_ok() or calls_made >= budget:
                    break
                improved = True
                while improved and time_ok() and calls_made < budget:
                    improved = False
                    for i in range(d):
                        if not time_ok() or calls_made >= budget:
                            break
                        # Try positive and negative directions
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = call_func(x_trial)
                            if f_trial is None:
                                return
                            if f_trial < current_f:
                                x = x_trial
                                current_f = f_trial
                                improved = True
                                if current_f < best_y:
                                    best_y = current_f
                                    best_x = x.copy()

        # Phase 3: Main loop with multiple restarts
        # Keep track of best points found
        best_points = [(best_y, best_x.copy())] if best_x is not None else []
        
        while time_ok() and calls_made < budget:
            remaining = budget - calls_made
            if remaining < 15:
                break
            
            # Choose starting point: mix of exploitation and exploration
            r = rng.random()
            if r < 0.3 and best_x is not None:
                # Exploit: perturb current best
                noise_scale = 0.5 * np.exp(-calls_made / 100.0)  # decreasing noise
                start_x = best_x + rng.normal(0, noise_scale, d)
            elif r < 0.6 and len(best_points) > 1:
                # Semi-exploit: perturb a good past point
                idx = rng.integers(0, len(best_points))
                start_x = best_points[idx][1] + rng.normal(0, 0.3, d)
            else:
                # Explore: random point
                start_x = rng.uniform(-5, 5, d)
            
            start_x = np.clip(start_x, -5, 5)
            
            # Run coordinate descent
            coordinate_descent(start_x)
            
            # Update best points list
            if best_x is not None and (len(best_points) == 0 or best_y < best_points[0][0]):
                best_points.insert(0, (best_y, best_x.copy()))
            elif best_x is not None:
                # Add if sufficiently different
                is_new = True
                for _, px in best_points:
                    if np.linalg.norm(best_x - px) < 0.1:
                        is_new = False
                        break
                if is_new:
                    best_points.append((best_y, best_x.copy()))
            
            # Keep only top 10 best points
            if len(best_points) > 10:
                best_points.sort(key=lambda p: p[0])
                best_points = best_points[:10]

        # Final polish: very fine coordinate descent from best point
        if best_x is not None and time_ok() and calls_made < budget:
            fine_steps = [0.01, 0.005, 0.002, 0.001]
            x = best_x.copy()
            current_f = call_func(x)
            if current_f is not None:
                for step in fine_steps:
                    if not time_ok() or calls_made >= budget:
                        break
                    improved = True
                    while improved and time_ok() and calls_made < budget:
                        improved = False
                        for i in range(d):
                            if not time_ok() or calls_made >= budget:
                                break
                            for direction in [1, -1]:
                                x_trial = x.copy()
                                x_trial[i] += direction * step
                                x_trial = np.clip(x_trial, -5, 5)
                                f_trial = call_func(x_trial)
                                if f_trial is None:
                                    return best_x, best_y
                                if f_trial < current_f:
                                    x = x_trial
                                    current_f = f_trial
                                    improved = True
                                    if current_f < best_y:
                                        best_y = current_f
                                        best_x = x.copy()

        return best_x, best_y
