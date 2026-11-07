"""Two-phase radial sweep + aggressive (1+1)-ES for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Radial sweep followed by aggressive (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense radial sweep from origin to identify the anisotropic axis
        # The Discus function has extreme anisotropy: one narrow "needle" axis and flat others
        # We sweep along many directions to find the needle direction
        
        # Generate a dense set of directions on the unit sphere in 5D
        # Use Fibonacci sphere for uniform coverage
        directions = []
        
        # Add coordinate axes
        for i in range(dim):
            for sign in [1, -1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        # Add pairwise combinations
        for i in range(dim):
            for j in range(i+1, dim):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        d = np.zeros(dim)
                        d[i] = si
                        d[j] = sj
                        norm = np.linalg.norm(d)
                        if norm > 0:
                            directions.append(d / norm)
        
        # Add random directions for more coverage
        for _ in range(30):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 0:
                directions.append(d / norm)
        
        # Remove duplicates and normalize
        unique_dirs = []
        for d in directions:
            d = d / (np.linalg.norm(d) + 1e-10)
            # Check if close to any existing direction
            is_dup = False
            for ud in unique_dirs:
                if np.dot(d, ud) > 0.95:
                    is_dup = True
                    break
            if not is_dup:
                unique_dirs.append(d)
        
        # Sweep along each direction at multiple radii
        radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        
        sweep_results = []
        for direction in unique_dirs:
            if call_count >= budget or time.time() > deadline - 80:
                break
            for r in radii:
                if call_count >= budget or time.time() > deadline - 80:
                    break
                point = r * direction
                point = np.clip(point, -5.0, 5.0)
                f = try_eval(point)
                if f is None:
                    break
                sweep_results.append((point, f))
        
        # Sort by function value
        sweep_results.sort(key=lambda x: x[1])
        
        # Phase 2: Aggressive (1+1)-ES with high mutation rate starting from best sweep point
        if sweep_results:
            start_point = sweep_results[0][0].copy()
        else:
            start_point = np.zeros(dim)
        
        # Aggressive (1+1)-ES parameters
        sigma = 1.0  # Start with larger step to explore
        x = start_point.copy()
        fx = try_eval(x)
        
        if fx is not None:
            success_count = 0
            eval_count = 0
            no_improve_count = 0
            
            while call_count < budget and time.time() < deadline - 20:
                # Generate mutant with high mutation rate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                
                eval_count += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15  # Increase step on success
                    success_count += 1
                    no_improve_count = 0
                else:
                    sigma *= 0.9  # Decrease step on failure
                    no_improve_count += 1
                
                # Adaptive sigma adjustment based on success rate over last 20 evals
                if eval_count % 20 == 0 and eval_count > 0:
                    rate = success_count / 20.0
                    if rate > 0.6:
                        sigma *= 1.1
                    elif rate < 0.3:
                        sigma *= 0.9
                    success_count = 0
                
                # If stuck for a while, try a larger jump
                if no_improve_count > 30:
                    sigma *= 2.0
                    no_improve_count = 0
        
        # Phase 3: Fine-tuning with very small steps around the best point
        if best_x is not None and call_count < budget:
            fine_sigma = 0.05
            x_fine = best_x.copy()
            fx_fine = best_f
            
            for _ in range(min(50, budget - call_count)):
                if time.time() > deadline - 10:
                    break
                if call_count >= budget:
                    break
                y_fine = np.clip(x_fine + fine_sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy_fine = try_eval(y_fine)
                if fy_fine is None:
                    break
                if fy_fine < fx_fine:
                    x_fine, fx_fine = y_fine, fy_fine
                    fine_sigma *= 1.1
                else:
                    fine_sigma *= 0.95
        
        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
