"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based global search + (1+1)-ES local refinement for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def check_deadline():
            return calls[0] >= budget or time.time() > deadline

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has periodicity 2. Minima are at integers.
        # We evaluate a grid of points at integer and half-integer coordinates.
        # This ensures we sample the "peaks" of the Rastrigin landscape systematically.
        
        # Generate candidate points:
        # 1. All integer points in [-5, 5]^5 is too many (11^5 = 161051).
        # 2. We restrict to the region where the minimum likely lies or use a coarser grid.
        # The problem states minimizer lies in [-4, 4]^5.
        # Let's sample integers in [-4, 4] and half-integers in [-4.5, 4.5].
        
        # Actually, a smarter approach:
        # The function is f(x) = 10*A + sum(x_i^2 - 10*cos(2*pi*x_i)).
        # The global minimum is at x=0 (if no shift). With shift, it's at integer coords.
        # We can sample a sparse lattice first.
        
        # Let's create a list of candidate points.
        # We will evaluate a subset of the grid defined by coordinates in {-5, -4.5, ..., 4.5, 5}
        # But 11^5 is too big.
        
        # Alternative: Use a stratified sampling approach or a specific lattice.
        # Given the periodicity, sampling at integers is key.
        # Let's sample all integer points in [-4, 4]^5? 9^5 = 59049. Too many for 500 budget.
        
        # We need to be smarter.
        # The "lattice" idea should be a low-discrepancy sequence snapped to a grid, 
        # or a specific structured subset.
        
        # Let's try a "Halton-like" or "Sobol" but snapped to a finer grid, 
        # OR a specific deterministic sequence that covers the space efficiently.
        
        # Actually, the prompt suggests: "systematically evaluating points at integer coordinates 
        # and half-integer offsets within the box to rapidly identify the global basin"
        
        # Since we have 500 calls, we can't do a full grid.
        # We can do a "randomized lattice" or a "quasi-random lattice".
        
        # Let's use a Sobol sequence but snap to a grid of step 0.5.
        # This is similar to the current program but the directive asks for a "lattice-based deterministic grid search".
        # A true deterministic grid search in 5D with 500 points is impossible to cover the whole space uniformly 
        # if we stick to a fine grid.
        
        # However, we can prioritize the center.
        # Let's generate points in a specific order:
        # 1. The origin (0,0,0,0,0)
        # 2. Points at +/- 1, +/- 2, etc. in each dimension.
        # 3. Then expand outwards.
        
        # This is a "Breadth-First Search" on the lattice of integers/half-integers.
        
        # Generate all points in [-4, 4]^5 with step 1 (integers).
        # Sort them by distance from origin.
        # Evaluate the closest ones first.
        
        # 9^5 = 59049 points. We only have ~200-300 calls for exploration.
        # We will take the first 200 points from this sorted list.
        
        # Generate integer grid in [-4, 4]
        coords = np.arange(-4, 5, 1)  # -4 to 4
        grid = np.meshgrid(*[coords]*dim, indexing='ij')
        grid_points = np.stack(grid, axis=-1).reshape(-1, dim)
        
        # Calculate distance from origin
        dists = np.sum(grid_points**2, axis=1)
        # Sort by distance
        sorted_indices = np.argsort(dists)
        
        # Take the first 200 closest integer points
        n_explore = 200
        explore_points = grid_points[sorted_indices[:n_explore]]
        
        # Also add some half-integer points? 
        # Maybe mix them. Let's stick to integers first as they are the exact minima locations.
        # If the shift is non-integer, we might miss. But the problem says "hidden seeded shift".
        # Usually in BBOB, shifts can be anything.
        # If we only sample integers, we might miss the basin if the shift is 0.5.
        
        # Let's use a Sobol sequence snapped to 0.5 grid as a fallback or mix.
        # But the directive says "lattice-based deterministic grid search".
        
        # Let's try a different lattice: 
        # Sample points where each coordinate is an integer or half-integer.
        # And prioritize those close to the center.
        
        # Let's generate a larger set of candidates: integers and half-integers in [-4, 4].
        # Coords: -4, -3.5, ..., 3.5, 4. (17 values per dim)
        # 17^5 = 1,419,857. Too many to generate and sort fully in Python quickly?
        # 1.4M points is manageable for numpy sorting.
        
        coords_half = np.arange(-4, 4.5, 0.5)
        grid_half = np.meshgrid(*[coords_half]*dim, indexing='ij')
        grid_half_points = np.stack(grid_half, axis=-1).reshape(-1, dim)
        
        # Filter to box [-5, 5] (already in [-4, 4])
        
        # Distance from origin
        dists_half = np.sum(grid_half_points**2, axis=1)
        sorted_indices_half = np.argsort(dists_half)
        
        # We will evaluate the first 250 points from this half-integer lattice.
        # This ensures we cover the center densely and expand outwards.
        n_explore = 250
        explore_points = grid_half_points[sorted_indices_half[:n_explore]]
        
        best_x = None
        best_f = np.inf
        
        # Evaluate exploration points
        for i in range(len(explore_points)):
            if check_deadline():
                break
            x = explore_points[i]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback to origin if something went wrong
            best_x = np.zeros(dim)
            best_f = call_func(best_x)
            
        # Phase 2: Local refinement with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        no_improve = 0
        max_no_improve = 20
        
        while not check_deadline():
            # Generate trial
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = call_func(y)
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.05
                no_improve = 0
            else:
                sigma *= 0.95
                no_improve += 1
            
            # Restart logic
            if no_improve >= max_no_improve:
                no_improve = 0
                sigma = 0.3
                # Jump to a random nearby point or back to best
                # Let's try a small jump from best
                offset = 0.5 * rng.standard_normal(dim)
                x = np.clip(best_x + offset, -5, 5)
                fx = call_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Fine-tuning
        if not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            while not check_deadline():
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.02
                else:
                    sigma *= 0.9
                if sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
