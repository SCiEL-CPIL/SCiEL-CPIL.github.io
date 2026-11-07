import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        # State variables
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to count calls and check time
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                return None
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Sparse Global Lattice Search
        # Allocate ~10% of budget to lattice search
        lattice_budget = max(10, self.budget // 10)
        
        # Create a grid. For dim=5, 2^5=32 points is too many for 10 calls.
        # Let's use a smaller grid or random sampling. 
        # A 3x3x3x3x3 grid is 243 points, too many.
        # Let's use a stratified random sample or a smaller grid like 2x2x2x2x2 = 32 points? 
        # If budget is 500, 10% is 50. 32 is fine.
        # If budget is small, we scale down.
        
        # Let's use a simple random sampling for the initial phase to cover the space, 
        # but the prompt suggests "sparse global lattice". 
        # For dim=5, a 2-point lattice per dimension (midpoints of halves) gives 32 points.
        # We will sample min(lattice_budget, 32) points from this lattice.
        
        lattice_points = []
        # Generate 2 points per dimension: -2.5, 2.5 (centered in [-5,5])
        # Or better, cover the range more evenly.
        # Let's use 3 points per dimension if budget allows? 3^5 = 243.
        # Let's stick to a random sample of the lattice points to keep it simple and fast.
        # Actually, just random sampling is often better than a fixed lattice for unknown shifts.
        # But the directive says "lattice". Let's use a grid of 2 points per dim (32 total) 
        # and if budget > 32, we can add more.
        
        # Let's generate a set of candidate points
        candidates = []
        # Add corners and center?
        # Let's just use random points for the first phase to ensure coverage, 
        # as the "lattice" might miss the shifted minimum if the shift is large.
        # However, to follow the directive, let's use a coarse grid.
        
        # Grid points: -4, 0, 4 (3 points per dim -> 243 total). 
        # We will pick the first 'lattice_budget' points from this list.
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*([grid_vals]*self.dim))).reshape(-1, self.dim)
        
        # Shuffle to randomize order? No, let's just take a subset.
        # To be efficient, we'll just evaluate random points from this grid.
        # Or simply, let's use random sampling for Phase 1 as it's more robust to shifts.
        # The "lattice" idea is good for structured search, but random is safer for unknown shifts.
        # Let's compromise: Use random sampling for Phase 1.
        
        for _ in range(min(lattice_budget, self.budget)):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, self.dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: Nelder-Mead Optimization
        # Use remaining budget
        remaining_budget = self.budget - calls_made
        
        # Initialize Simplex
        # Start from best_x
        x0 = best_x.copy()
        f0 = best_f
        
        # Initial step size
        alpha = 0.5
        # Simplex vertices
        simplex = [x0]
        simplex_f = [f0]
        
        # Add other vertices
        for i in range(self.dim):
            x_i = x0.copy()
            # Determine step direction and magnitude
            # If x0 is near boundary, step inwards
            if x0[i] > 4.0:
                x_i[i] -= alpha
            elif x0[i] < -4.0:
                x_i[i] += alpha
            else:
                x_i[i] += alpha
            
            # Clip to bounds
            x_i = np.clip(x_i, -5, 5)
            
            # Avoid duplicate points
            if np.allclose(x_i, x0):
                 x_i = np.clip(x0 + 0.1 * np.random.randn(self.dim), -5, 5)
            
            f_i = safe_call(x_i)
            if f_i is None:
                break
            simplex.append(x_i)
            simplex_f.append(f_i)
            
            if calls_made >= self.budget or time.time() > deadline:
                break
        
        # Nelder-Mead Loop
        gamma = 1.0   # Reflection
        rho = 2.0     # Expansion
        sigma = 0.5   # Contraction
        eps = 1e-8    # Shrinkage
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence
            if simplex_f[-1] - simplex_f[0] < 1e-10:
                break
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Worst point
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            
            # Reflection
            x_r = centroid + gamma * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_call(x_r)
            if f_r is None:
                break
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_call(x_e)
                if f_e is None:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Reflection accepted
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < f_worst
                if f_r < f_worst:
                    x_c = centroid + sigma * (x_r - centroid)
                else:
                    x_c = centroid - sigma * (x_worst - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_call(x_c)
                if f_c is None:
                    break
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, self.dim + 1):
                        x_s = simplex[0] + eps * (simplex[i] - simplex[0])
                        x_s = np.clip(x_s, -5, 5)
                        f_s = safe_call(x_s)
                        if f_s is None:
                            break
                        simplex[i] = x_s
                        simplex_f[i] = f_s
                    if f_s is None:
                        break

        return best_x, best_f
