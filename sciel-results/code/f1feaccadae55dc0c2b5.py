# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic Multi-Start Coordinate Descent with Golden Section Line Search.
    
    Strategy:
    1. Initialize with a set of deterministic grid points to cover the search space.
    2. For each promising starting point, perform coordinate-wise line search.
       This leverages the "Sharp Ridge" structure where the function is flat in some
       directions and sharp in others. By optimizing one coordinate at a time,
       we can efficiently traverse the ridge without getting stuck in local minima
       caused by stochastic noise or poor step size adaptation.
    3. Use Golden Section Search for 1D optimization along each axis to ensure
       robust convergence without requiring gradients.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(self.dim)

        def timed_call(x):
            nonlocal best_f, best_x, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic Initialization
        # Use a stratified grid to ensure coverage.
        # 3 points per dimension -> 3^5 = 243 points.
        # This is a small fraction of the budget (500) but provides good coverage.
        
        # We will use a slightly finer grid for the first few dimensions if budget allows, 
        # but 3^5 is a good start. Let's try 4^5 = 1024? No, too many.
        # Let's stick to 3^5 = 243 points.
        
        # Actually, let's use a mix:
        # 1. Center point: 1 call
        # 2. Grid points: 3^5 = 243 calls
        # Total: 244 calls.
        # Remaining: 500 - 244 = 256 calls for local search.
        
        # Generate grid points
        grid_points = []
        # Use 3 points per dim: -5, 0, 5
        # But the minimum is in [-4, 4]^5. So -4, 0, 4 is better.
        # Let's use -4, 0, 4.
        
        # Actually, to be safe, let's use -5, 0, 5 to cover the whole box.
        # The function clips to [-5, 5].
        
        # Let's use a slightly more dense grid around the center.
        # But for now, simple 3-point grid.
        
        # We will evaluate all grid points.
        # To save time, we can stop early if we find a very low value, 
        # but we need to be careful not to miss the global min.
        # Given the budget, evaluating all 243 is fast.
        
        # Generate all combinations
        # Use np.meshgrid
        # But 5D meshgrid is heavy. Let's just iterate.
        
        # Alternative: Use random sampling for initialization?
        # The directive says "deterministic multi-start local search using a grid".
        # So let's stick to grid.
        
        # 3^5 = 243 points.
        # Let's evaluate them.
        
        # To make it more efficient, we can prioritize points closer to the center.
        # But for now, just evaluate all.
        
        # Let's create the list of grid points
        grid_vals = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        # We can use itertools.product but it's in stdlib.
        # Or just nested loops.
        
        # Let's use a recursive generator or just hardcode for dim=5?
        # No, dim is given.
        
        # Let's use np.indices or similar?
        # Simpler:
        # points = np.array(list(itertools.product(grid_vals, repeat=self.dim)))
        
        # But we need to import itertools.
        import itertools
        
        # Generate points
        # Sort them by distance from center to prioritize center?
        # No, just evaluate all.
        
        # Actually, 243 calls is fine.
        
        # Let's evaluate all grid points
        # We'll do this in a batch-like manner (loop)
        
        # Initialize list of starting points for local search
        # We will keep the best few grid points as starting points for local search.
        # Or just use the best grid point.
        # Let's use the top 3 grid points as multi-start points.
        
        grid_scores = []
        
        for coords in itertools.product(grid_vals, repeat=self.dim):
            if calls_made >= self.budget or time.time() > deadline - 2:
                break
            x = np.array(coords)
            fx = timed_call(x)
            if fx is not None:
                grid_scores.append((fx, x))
        
        if not grid_scores:
            # Fallback
            x = np.zeros(self.dim)
            timed_call(x)
            return best_x, best_f
            
        # Sort grid scores
        grid_scores.sort(key=lambda item: item[0])
        
        # Phase 2: Coordinate Descent with Golden Section
        # Start from the best grid point
        # And maybe the 2nd and 3rd best if budget allows.
        
        # We have roughly 500 - 243 = 257 calls left.
        # Each coordinate descent step requires a 1D optimization.
        # Golden section takes ~20-30 function evaluations to converge to high precision.
        # 5 dimensions * 30 = 150 calls per full sweep.
        # So we can do 1-2 full sweeps.
        
        # Let's implement a single coordinate descent sweep starting from the best point.
        
        x = grid_scores[0][1].copy()
        fx = grid_scores[0][0]
        
        # Golden Section Search function
        def golden_search(func, x, dim, lo, hi, calls_made_ref):
            # Minimizes func along dimension 'dim' of x, with x[dim] in [lo, hi]
            # Returns (new_x, new_f, calls_used)
            
            nonlocal calls_made
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2  # ~0.618
            
            # Initial points
            x1 = lo + (1 - gr) * (hi - lo)
            x2 = lo + gr * (hi - lo)
            
            f1 = None
            f2 = None
            
            # We need to evaluate f1 and f2
            # But we must respect the budget and deadline.
            
            x_test1 = x.copy()
            x_test1[dim] = x1
            if calls_made < self.budget and time.time() <= deadline - 2:
                f1 = func(x_test1)
                calls_made += 1
                if f1 < best_f:
                    best_f = f1
                    best_x = x_test1.copy()
            else:
                return x, fx, 0 # No calls made
            
            x_test2 = x.copy()
            x_test2[dim] = x2
            if calls_made < self.budget and time.time() <= deadline - 2:
                f2 = func(x_test2)
                calls_made += 1
                if f2 < best_f:
                    best_f = f2
                    best_x = x_test2.copy()
            else:
                return x, fx, 1 # One call made
            
            calls_used = 2
            
            # Iterate
            max_iter = 20
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                if f1 < f2:
                    # Minimum is in [lo, x2]
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = lo + (1 - gr) * (hi - lo)
                    x_test1 = x.copy()
                    x_test1[dim] = x1
                    if calls_made < self.budget and time.time() <= deadline - 2:
                        f1 = func(x_test1)
                        calls_made += 1
                        calls_used += 1
                        if f1 < best_f:
                            best_f = f1
                            best_x = x_test1.copy()
                    else:
                        break
                else:
                    # Minimum is in [x1, hi]
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    x_test2 = x.copy()
                    x_test2[dim] = x2
                    if calls_made < self.budget and time.time() <= deadline - 2:
                        f2 = func(x_test2)
                        calls_made += 1
                        calls_used += 1
                        if f2 < best_f:
                            best_f = f2
                            best_x = x_test2.copy()
                    else:
                        break
            
            # Determine the best point found in this 1D search
            # We need to track the best x and f from the evaluations
            # This is tricky because we only return the best global best.
            # Let's return the best x along this dimension.
            
            # Actually, we should track local best.
            # Let's refactor golden_search to return local best.
            
            # For now, let's just assume the global best is updated correctly.
            # We need to update x to the best point found in this dimension.
            
            # Let's re-implement golden_search to track local best properly.
            
            return x, fx, calls_used # Placeholder, needs fix
        
        # The above golden_search is flawed because it doesn't return the best x found.
        # Let's rewrite it properly.
        
        def golden_search_1d(func, x, dim, lo, hi):
            nonlocal calls_made, best_f, best_x
            
            gr = (np.sqrt(5) - 1) / 2
            x1 = lo + (1 - gr) * (hi - lo)
            x2 = lo + gr * (hi - lo)
            
            f1 = None
            f2 = None
            best_local_x = x.copy()
            best_local_f = np.inf
            
            x_test1 = x.copy()
            x_test1[dim] = x1
            if calls_made < self.budget and time.time() <= deadline - 2:
                f1 = func(x_test1)
                calls_made += 1
                if f1 < best_local_f:
                    best_local_f = f1
                    best_local_x = x_test1.copy()
                if f1 < best_f:
                    best_f = f1
                    best_x = x_test1.copy()
            else:
                return x, fx # No improvement or no calls made
            
            x_test2 = x.copy()
            x_test2[dim] = x2
            if calls_made < self.budget and time.time() <= deadline - 2:
                f2 = func(x_test2)
                calls_made += 1
                if f2 < best_local_f:
                    best_local_f = f2
                    best_local_x = x_test2.copy()
                if f2 < best_f:
                    best_f = f2
                    best_x = x_test2.copy()
            else:
                return best_local_x, best_local_f
            
            max_iter = 30
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = lo + (1 - gr) * (hi - lo)
                    x_test1 = x.copy()
                    x_test1[dim] = x1
                    if calls_made < self.budget and time.time() <= deadline - 2:
                        f1 = func(x_test1)
                        calls_made += 1
                        if f1 < best_local_f:
                            best_local_f = f1
                            best_local_x = x_test1.copy()
                        if f1 < best_f:
                            best_f = f1
                            best_x = x_test1.copy()
                    else:
                        break
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    x_test2 = x.copy()
                    x_test2[dim] = x2
                    if calls_made < self.budget and time.time() <= deadline - 2:
                        f2 = func(x_test2)
                        calls_made += 1
                        if f2 < best_local_f:
                            best_local_f = f2
                            best_local_x = x_test2.copy()
                        if f2 < best_f:
                            best_f = f2
                            best_x = x_test2.copy()
                    else:
                        break
            
            return best_local_x, best_local_f
        
        # Run coordinate descent
        # Start from the best grid point
        x_current = grid_scores[0][1].copy()
        f_current = grid_scores[0][0]
        
        # We will do a few passes of coordinate descent
        # Each pass optimizes each dimension once.
        
        # How many passes can we afford?
        # Each pass: 5 dims * ~30 calls = 150 calls.
        # We have ~250 calls left. So 1-2 passes.
        
        for pass_num in range(2):
            if calls_made >= self.budget or time.time() > deadline - 2:
                break
            
            for dim in range(self.dim):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                
                # Define the search range for this dimension
                # Use the current x[dim] +/- some range
                # Or use the full box [-5, 5]
                # Using full box is safer but slower.
                # Let's use a range around the current value, e.g., +/- 2.0
                # But if we are far from the optimum, this might miss it.
                # Since we started from a grid point, let's use the full box [-5, 5]
                # for the first pass, and a smaller range for subsequent passes.
                
                if pass_num == 0:
                    lo = -5.0
                    hi = 5.0
                else:
                    # Shrink the range
                    center = x_current[dim]
                    lo = max(-5.0, center - 1.0)
                    hi = min(5.0, center + 1.0)
                
                # Perform golden section search
                x_new, f_new = golden_search_1d(func, x_current, dim, lo, hi)
                
                # Update x_current if improved
                if f_new < f_current:
                    x_current = x_new
                    f_current = f_new
        
        # If we have budget left, we can do another round of local search from the best point
        # Or just return the best found.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
