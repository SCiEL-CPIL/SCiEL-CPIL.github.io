import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return 1e10
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx
        
        # Phase 1: Deterministic Global Sampling
        # This ensures early progress by covering the domain systematically.
        # 1. Center
        # 2. 32 Vertices of [-5, 5]^5
        # 3. 10 Axis midpoints (±2.5 on each axis)
        
        points_to_eval = []
        
        # Center
        points_to_eval.append(np.zeros(n))
        
        # Vertices
        for i in range(2**n):
            vertex = []
            for j in range(n):
                if (i >> j) & 1:
                    vertex.append(5.0)
                else:
                    vertex.append(-5.0)
            points_to_eval.append(np.array(vertex))
            
        # Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)
        
        # Evaluate these points
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 10:
                break
            wrapped_func(p)
            
        # If we didn't find any points (shouldn't happen), fallback to random
        if best_x is None:
            rng = np.random.default_rng(42)
            best_x = rng.uniform(-5, 5, n)
            best_f = wrapped_func(best_x)

        # Phase 2: Fixed-Step Local Hill Climb
        # Deterministic refinement from the best point found in Phase 1.
        # We try steps in each dimension (positive and negative) with decreasing magnitudes.
        # This avoids the overhead and potential instability of Nelder-Mead on highly multimodal landscapes.
        
        if calls < budget and time.time() < deadline - 5:
            current_x = best_x.copy()
            current_f = best_f
            
            # Define a sequence of step sizes to try
            # Start relatively large to escape local minima, then shrink for fine-tuning
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]
            
            for step in step_sizes:
                if calls >= budget or time.time() > deadline - 5:
                    break
                
                improved = True
                while improved and calls < budget and time.time() < deadline - 5:
                    improved = False
                    for i in range(n):
                        if calls >= budget or time.time() > deadline - 5:
                            break
                        for sign in [-1, 1]:
                            if calls >= budget or time.time() > deadline - 5:
                                break
                            new_x = current_x.copy()
                            new_x[i] += sign * step
                            # Clip to box [-5, 5]
                            new_x = np.clip(new_x, -5.0, 5.0)
                            
                            # Only evaluate if it's different from current (avoid redundant calls if at boundary)
                            if np.allclose(new_x, current_x):
                                continue
                                
                            new_f = wrapped_func(new_x)
                            
                            if new_f < current_f - 1e-10:
                                current_x = new_x
                                current_f = new_f
                                improved = True
                                # Break out of sign loop if improvement found, restart dimension loop
                                # Actually, standard hill climb tries all directions from new best
                                # So we don't break the for i loop immediately, we just update current_x
                                # But we should probably restart the dimension loop to ensure we don't miss a better direction
                                # due to the order of signs? 
                                # Standard approach: update current_x, and continue checking other dimensions from the new best.
                                # This is what we are doing.
                                
        return best_x, best_f
