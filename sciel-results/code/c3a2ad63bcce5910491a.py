import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search / Initialization
        # Use 10% of budget for random sampling to find a good starting basin
        n_global = max(10, self.budget // 10)
        
        x_best = None
        f_best = np.inf
        
        # Random sampling
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()
            self.budget -= 1 # We need to track budget manually or use a counter
            # Actually, better to use a counter variable to not modify self.budget if we want to reuse, but here it's single call.
            # Let's use a local counter.
        
        # Wait, the prompt says "Every call is recorded". I must ensure I don't exceed 500 calls.
        # I will use a local `calls` counter.
        
        # Re-implementing with local counter
        calls = 0
        max_calls = self.budget
        
        # Phase 1
        n_global = max(10, max_calls // 10)
        for i in range(n_global):
            if calls >= max_calls: break
            if time.time() > deadline: break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
        if x_best is None:
            x_best = np.zeros(self.dim)
            f_best = func(x_best)
            calls += 1

        # Phase 2: Nelder-Mead
        # Initialize simplex
        # Base point x_best
        # Step size
        delta = 0.5 
        # If x_best is close to boundary, reduce delta?
        # Simple: just use delta.
        
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x_best
        f_values = np.zeros(self.dim + 1)
        f_values[0] = f_best
        
        for i in range(1, self.dim + 1):
            simplex[i] = x_best.copy()
            simplex[i][i-1] += delta
            # Clip to box? NM can go out of bounds. 
            # The func clips inputs, so we can let NM explore outside, 
            # but the "effective" point is clipped. 
            # However, if we let NM think it's at x_out but func sees x_clip, 
            # the geometry is distorted.
            # Better to keep simplex within bounds or handle clipping in the NM logic?
            # Standard practice: Let NM explore, but if it goes out, the function value is for the clipped point.
            # This can cause issues if the clipped point is on the boundary.
            # For f14, the minimum is in [-4,4], so if we start in [-5,5] and use small steps, 
            # we are likely safe.
            simplex[i][i-1] = np.clip(simplex[i][i-1], -5, 5)
            f_values[i] = func(simplex[i])
            calls += 1
            if calls >= max_calls:
                break
            if time.time() > deadline:
                break

        # NM Loop
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        while calls < max_calls:
            if time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            x_worst = simplex[-1]
            x_best_nm = simplex[0]
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < f_worst, else inside
                if f_r < f_values[-1]:
                    x_c = x_centroid + rho * (x_r - x_centroid)
                else:
                    x_c = x_centroid - rho * (x_r - x_centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = x_best_nm + sigma * (simplex[i] - x_best_nm)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if calls >= max_calls:
                            break
                    if calls >= max_calls:
                        break

            # Update global best
            current_best_f = np.min(f_values)
            if current_best_f < f_best:
                f_best = current_best_f
                x_best = simplex[np.argmin(f_values)].copy()
            
            # Convergence check
            if np.max(f_values) - np.min(f_values) < 1e-10:
                break

        return x_best, f_best
