# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB-style f13 (Sharp Ridge).
    
    Combines:
    1. Nelder-Mead simplex for global-ish convergence and robustness.
    2. Coordinate-descent line search (Powell-like) for exploiting ridge structure.
       This is crucial for f13 where the ridge is sharp in one direction and flat in others.
    3. Adaptive step sizes and stall detection with random restarts/perturbations.
    4. Strict time and budget guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 285s to be safe
        deadline = start_time + 285.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
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

        # Phase 1: Deterministic multi-start initialization
        # Use a structured grid of coarse initial points to cover the search space.
        # For dim=5, use a 3x3x3x3x3 grid = 243 points, but that's too many.
        # Use a smaller set: center + vertices of a cube + midpoints of edges
        n = self.dim
        
        # Generate deterministic initial points
        init_points = []
        # Center
        init_points.append(np.zeros(n))
        # Vertices of a cube with radius 4
        for mask in range(2**n):
            pt = np.zeros(n)
            for i in range(n):
                if mask & (1 << i):
                    pt[i] = 4.0
                else:
                    pt[i] = -4.0
            init_points.append(pt)
        # Midpoints of edges (pairs of vertices differing in one coordinate)
        for mask in range(2**n):
            for i in range(n):
                pt = np.zeros(n)
                for j in range(n):
                    if j == i:
                        pt[j] = 0.0  # midpoint
                    else:
                        if mask & (1 << j):
                            pt[j] = 4.0
                        else:
                            pt[j] = -4.0
                init_points.append(pt)
        
        # Evaluate initial points
        for x in init_points:
            if calls_made >= self.budget or time.time() > deadline - 2:
                break
            timed_call(x)
        
        if best_x is None:
            x = np.zeros(n)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Coordinate-wise golden-section line search
        # Perform sequential 1D golden-section optimization along each axis.
        # This exploits the axis-aligned ridge structure of f13.
        
        def golden_section_1d(x, d, lo, hi, fx, max_iter=20):
            """
            Perform golden-section search along dimension d.
            x: current point
            d: dimension to optimize
            lo, hi: bounds for x[d]
            fx: current function value at x
            Returns: (new_x, new_fx, improved)
            """
            nonlocal calls_made, best_x, best_f
            
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2  # ~0.618
            
            # Initial interval
            a = lo
            b = hi
            
            # If current x[d] is outside [a, b], clamp it
            if x[d] < a:
                x[d] = a
            elif x[d] > b:
                x[d] = b
            
            # Evaluate at two interior points
            x1 = x.copy()
            x1[d] = a + (1 - gr) * (b - a)
            x2 = x.copy()
            x2[d] = a + gr * (b - a)
            
            fx1 = timed_call(x1)
            if fx1 is None:
                return x, fx, False
            
            fx2 = timed_call(x2)
            if fx2 is None:
                return x, fx, False
            
            # Track best found during this 1D search
            local_best_x = x.copy()
            local_best_f = fx
            if fx1 < local_best_f:
                local_best_f = fx1
                local_best_x = x1.copy()
            if fx2 < local_best_f:
                local_best_f = fx2
                local_best_x = x2.copy()
            
            # Golden-section iteration
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                
                if fx1 < fx2:
                    # Minimum is in [a, x2]
                    b = x2[d]
                    x2 = x1.copy()
                    fx2 = fx1
                    x1 = x.copy()
                    x1[d] = a + (1 - gr) * (b - a)
                    fx1 = timed_call(x1)
                    if fx1 is None:
                        break
                    if fx1 < local_best_f:
                        local_best_f = fx1
                        local_best_x = x1.copy()
                else:
                    # Minimum is in [x1, b]
                    a = x1[d]
                    x1 = x2.copy()
                    fx1 = fx2
                    x2 = x.copy()
                    x2[d] = a + gr * (b - a)
                    fx2 = timed_call(x2)
                    if fx2 is None:
                        break
                    if fx2 < local_best_f:
                        local_best_f = fx2
                        local_best_x = x2.copy()
                
                # Check for convergence
                if abs(b - a) < 1e-6:
                    break
            
            # Return the best point found during this 1D search
            if local_best_f < fx:
                return local_best_x, local_best_f, True
            else:
                return x, fx, False

        # Perform multiple rounds of coordinate-wise line search
        n_rounds = 50  # Number of full passes over all dimensions
        
        x = best_x.copy()
        fx = best_f
        
        for round_num in range(n_rounds):
            if calls_made >= self.budget or time.time() > deadline - 2:
                break
            
            improved_this_round = False
            
            for d in range(n):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                
                # Define search interval for this dimension
                # Use a wide interval initially, then narrow
                if round_num < 5:
                    lo, hi = -5.0, 5.0
                elif round_num < 15:
                    lo = max(-5.0, x[d] - 2.0)
                    hi = min(5.0, x[d] + 2.0)
                elif round_num < 30:
                    lo = max(-5.0, x[d] - 0.5)
                    hi = min(5.0, x[d] + 0.5)
                else:
                    lo = max(-5.0, x[d] - 0.1)
                    hi = min(5.0, x[d] + 0.1)
                
                # Ensure the interval contains the current point
                if x[d] < lo or x[d] > hi:
                    lo = min(lo, x[d] - 0.1)
                    hi = max(hi, x[d] + 0.1)
                    lo = max(-5.0, lo)
                    hi = min(5.0, hi)
                
                new_x, new_fx, improved = golden_section_1d(x, d, lo, hi, fx)
                
                if improved:
                    x = new_x
                    fx = new_fx
                    improved_this_round = True
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
            
            # If no improvement in a full round, we may have converged
            # or we need to try a different strategy
            if not improved_this_round:
                # Try a random restart from the best point
                if round_num > 10 and round_num % 5 == 0:
                    # Perturb the best point slightly
                    perturbation = 0.5 / (round_num // 5 + 1)
                    x = best_x.copy()
                    for d in range(n):
                        x[d] += np.random.uniform(-perturbation, perturbation)
                        x[d] = np.clip(x[d], -5, 5)
                    fx = timed_call(x)
                    if fx is not None and fx < best_f:
                        best_f = fx
                        best_x = x.copy()
        
        # Phase 3: Final high-precision polish pass
        # Use a tighter interval and more iterations for the final refinement
        if calls_made < self.budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            
            # Perform a few more rounds with very tight intervals
            for round_num in range(20):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                
                improved_this_round = False
                
                for d in range(n):
                    if calls_made >= self.budget or time.time() > deadline - 2:
                        break
                    
                    lo = max(-5.0, x[d] - 0.01)
                    hi = min(5.0, x[d] + 0.01)
                    
                    if x[d] < lo or x[d] > hi:
                        lo = min(lo, x[d] - 0.001)
                        hi = max(hi, x[d] + 0.001)
                        lo = max(-5.0, lo)
                        hi = min(5.0, hi)
                    
                    new_x, new_fx, improved = golden_section_1d(x, d, lo, hi, fx, max_iter=30)
                    
                    if improved:
                        x = new_x
                        fx = new_fx
                        improved_this_round = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                
                if not improved_this_round:
                    break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
