"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Rastrigin-aware coordinate-wise ladder search exploiting separability."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Sample a few points to get a starting point in a reasonable basin.
        # Since we don't know the shift/rotation, we sample broadly but sparingly.
        # The ladder search will do the heavy lifting.
        n_init = 20
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                # Sample in [-4, 4] to stay away from boundaries if possible
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Coordinate-wise Ladder Search
        # Exploit the fact that Rastrigin f(x) = sum_i (10(x_i^2 - cos(2*pi*x_i)) + 10)
        # Even with rotation/shift, the function is a composition of 1D periodic functions.
        # Optimizing one coordinate at a time while holding others fixed is highly effective.
        
        # We will iterate through dimensions, and for each, perform a 1D search.
        # To handle the "ladder" nature (finding the global minimum of the 1D slice),
        # we use a dense grid scan followed by local refinement.
        
        # The period of Rastrigin is 2*pi. Local minima are at multiples of pi (in the unshifted case).
        # With shift s and rotation R, the minima are shifted but the periodicity along the rotated axes remains.
        # However, since we are searching in the original coordinate space, the "periodicity" is mixed.
        # But the separability means that for fixed x_j (j != i), f(x) is a 1D function of x_i.
        # This 1D function is a sum of a quadratic term (from the rotation mixing) and periodic terms.
        # Actually, if there is rotation, f(x) is NOT separable in the original coordinates.
        # f(x) = Rastrigin(R x + s).
        # Rastrigin(y) = sum_i 10(y_i^2 - cos(2 pi y_i)) + 10 dim.
        # y_i = sum_j R_ij x_j + s_i.
        # So f(x) = sum_i 10( (sum_j R_ij x_j + s_i)^2 - cos(2 pi (sum_j R_ij x_j + s_i)) ) + ...
        # This is NOT separable in x. The term (sum_j R_ij x_j)^2 couples all x_j.
        
        # HOWEVER, the "ladder search" finding from the prompt suggests it works well.
        # Why? Because the quadratic part is a positive definite quadratic form (if R is rotation, R R^T = I, so the quadratic part is sum x_i^2 = ||x||^2).
        # The periodic part is the main challenge.
        # Let's re-read the insight: "Coordinate-wise ladder search exploits Rastrigin's separable structure... allowing each dimension to be optimized independently".
        # This implies that for the specific instances or due to the nature of the error landscape, coordinate descent works.
        # Or perhaps the "separable structure" refers to the fact that the local minima are arranged in a grid, and coordinate-wise search can jump between them.
        
        # Let's implement a robust coordinate-wise search.
        # For each dimension d:
        #   1. Scan a range of x_d values around the current best_x[d].
        #   2. Find the best x_d.
        #   3. Repeat until convergence or budget/time runs out.
        
        # Range to scan: The basin of attraction for Rastrigin is roughly [-pi, pi] around a minimum.
        # In the original coordinates, due to rotation, the "width" of the basin in each coordinate might be larger or smaller.
        # Let's scan a window of size 2*pi * sqrt(dim) to be safe? No, that's too big.
        # Let's scan a window of size 4.0 (approx 2 * 2*pi/3).
        # Actually, a full period scan is 2*pi ~ 6.28.
        # Let's try scanning [-4, 4] relative to current center? That's 8 units.
        # With 500 calls, we can afford a modest grid.
        
        # Strategy:
        # Outer loop: cycles through dimensions.
        # Inner loop: 1D search for the active dimension.
        
        # 1D Search Method:
        # - Grid scan with step 0.2 in a window of +/- 3.0 around current x_d.
        # - This is 30 points per dimension per cycle.
        # - 5 dimensions * 30 points = 150 calls per full cycle.
        # - We have ~480 calls left. So we can do ~3 full cycles.
        # - After each cycle, the best point should be improved.
        
        # Let's refine the 1D search to be adaptive.
        # Start with a wider window, then narrow it.
        
        def one_dim_search(x_base, dim_idx, window, n_points):
            """Search for best value of x_base[dim_idx] within x_base[dim_idx] +/- window."""
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return x_base
            
            x_best = x_base.copy()
            f_best = np.inf
            
            # Generate grid points
            # We want to cover [x_base[dim_idx] - window, x_base[dim_idx] + window]
            # Clip to [-5, 5]
            low = max(-5.0, x_base[dim_idx] - window)
            high = min(5.0, x_base[dim_idx] + window)
            
            if high <= low:
                return x_base
                
            x_grid = np.linspace(low, high, n_points)
            
            for x_val in x_grid:
                if calls_made >= budget or time.time() > deadline:
                    break
                x_trial = x_base.copy()
                x_trial[dim_idx] = x_val
                f_val = counted_func(x_trial)
                if f_val < f_best:
                    f_best = f_val
                    x_best = x_trial.copy()
            
            return x_best

        # Main Loop
        # We will perform multiple passes.
        # Pass 1: Wide window to find the correct basin.
        # Pass 2: Narrower window to refine.
        # Pass 3: Even narrower.
        
        passes = [
            (4.0, 20),  # Window 4.0, 20 points (step ~0.4)
            (2.0, 20),  # Window 2.0, 20 points (step ~0.2)
            (1.0, 20),  # Window 1.0, 20 points (step ~0.1)
            (0.5, 20),  # Window 0.5, 20 points (step ~0.05)
            (0.2, 20),  # Window 0.2, 20 points (step ~0.02)
        ]
        
        for window, n_points in passes:
            for _ in range(2): # 2 cycles per pass
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 5:
                        break
                    
                    # Update best_x using the current best from 1D search
                    # Note: counted_func updates global best_x/best_f.
                    # But one_dim_search returns the local best for that dimension.
                    # We should use the global best_x as the base for the next dimension?
                    # Or the local best?
                    # Using global best_x is safer because it might have been updated by a previous dimension in this cycle.
                    # However, one_dim_search modifies a copy. It doesn't update best_x directly unless it finds a new global best.
                    # Let's make one_dim_search return the new x, and we update a working_x.
                    
                    # Actually, let's just use the global best_x as the starting point for each dimension search.
                    # This ensures we are always refining the best known point.
                    x_new = one_dim_search(best_x, d, window, n_points)
                    
                    # one_dim_search has already called counted_func, so best_x is updated if x_new is better.
                    # No need to do anything else, next dimension will use the updated best_x.
        
        # If we still have budget, do some random restarts in the vicinity of the best point
        # to escape any local minima that the coordinate search might have gotten stuck in.
        # But coordinate search on Rastrigin is usually good at hopping between local minima
        # if the window is large enough.
        
        # Final check: if we have time, do a few random perturbations
        if calls_made < budget and time.time() < deadline - 10:
            for _ in range(5):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                x_perturb = best_x + rng.normal(0, 0.5, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                counted_func(x_perturb)
                
                # Quick local refinement from this new point
                if calls_made < budget and time.time() < deadline - 5:
                    # One cycle of narrow coordinate search
                    for d in range(dim):
                        if calls_made >= budget or time.time() > deadline - 5:
                            break
                        one_dim_search(best_x, d, 0.5, 10)

        return best_x, best_f
# EVOLVE-BLOCK-END
