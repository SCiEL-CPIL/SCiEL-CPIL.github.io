import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses differential evolution to find a feasible starting point, then polishes with L-BFGS-B.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # Variables: x0, y0, s0, x1, y1, s1, ..., x9, y9, s9
    # Total 30 variables.
    # Bounds:
    # x, y in [0, 1]
    # s in [0.01, 1.0] (side must be > 0)
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 1.0)) # s

    def objective(params):
        # Check deadline
        if time.time() > deadline:
            return float('inf')
            
        x = params[0::3]
        y = params[1::3]
        s = params[2::3]
        
        # Boundary penalty
        # Square i is inside if s_i/2 <= x_i <= 1 - s_i/2
        # Penalty is distance outside
        pen_boundary = 0.0
        for i in range(n):
            pen_boundary += max(0.0, s[i]/2.0 - x[i])
            pen_boundary += max(0.0, x[i] - (1.0 - s_i/2.0))
            pen_boundary += max(0.0, s[i]/2.0 - y[i])
            pen_boundary += max(0.0, y[i] - (1.0 - s_i/2.0))
        
        # Overlap penalty
        # Penalty is area of intersection
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Intersection width
                w = max(0.0, (s[i] + s[j]) / 2.0 - abs(x[i] - x[j]))
                # Intersection height
                h = max(0.0, (s[i] + s[j]) / 2.0 - abs(y[i] - y[j]))
                pen_overlap += w * h
                
        # Total penalty
        # We want to minimize -sum(s) + lambda * penalty
        # If penalty is 0, we maximize sum(s).
        # If penalty > 0, we discourage it.
        # A large lambda ensures feasibility is prioritized initially.
        
        return -np.sum(s) + 100.0 * (pen_boundary + pen_overlap)

    # Initial guess: Grid
    k = 3
    side = 1.0 / k
    x0 = np.zeros(n)
    y0 = np.zeros(n)
    s0 = np.zeros(n)
    for i in range(n):
        col = i % k
        row = i // k
        x0[i] = (col + 0.5) * side
        y0[i] = (row + 0.5) * side
        s0[i] = side
        
    x0_params = np.zeros(3 * n)
    x0_params[0::3] = x0
    x0_params[1::3] = y0
    x0_params[2::3] = s0

    best_params = x0_params.copy()
    best_val = objective(best_params)

    try:
        # Run Differential Evolution with reduced iterations to ensure speed
        # We want a feasible starting point for polishing, not necessarily the global optimum itself.
        # popsize=20, maxiter=200 is much faster.
        
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-6,
            polish=False
        )
        
        if result.success or result.fun < best_val:
            best_params = result.x
            best_val = result.fun

        # Polish step: Use L-BFGS-B to refine the solution.
        # We use a higher penalty weight for overlap to ensure strict feasibility during polishing.
        
        def objective_polish(params):
            if time.time() > deadline:
                return float('inf')
            x = params[0::3]
            y = params[1::3]
            s = params[2::3]
            
            pen_boundary = 0.0
            for i in range(n):
                pen_boundary += max(0.0, s[i]/2.0 - x[i])**2
                pen_boundary += max(0.0, x[i] - (1.0 - s_i/2.0))**2
                pen_boundary += max(0.0, s[i]/2.0 - y[i])**2
                pen_boundary += max(0.0, y[i] - (1.0 - s_i/2.0))**2
            
            pen_overlap = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    w = max(0.0, (s[i] + s[j]) / 2.0 - abs(x[i] - x[j]))
                    h = max(0.0, (s[i] + s[j]) / 2.0 - abs(y[i] - y[j]))
                    pen_overlap += w * h
                    
            # High penalty for overlap to force feasibility
            return -np.sum(s) + 1000.0 * (pen_boundary + pen_overlap)

        if time.time() < deadline - 30:
            res_polish = minimize(
                objective_polish, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res_polish.fun < best_val:
                best_params = res_polish.x
                best_val = res_polish.fun

        x = best_params[0::3]
        y = best_params[1::3]
        s = best_params[2::3]
        
        # Final verification and cleanup
        # If the solution is still invalid, we might need to shrink squares or move them.
        # For now, we clip sides to a minimum and check bounds.
        
        centers = np.stack([x, y], axis=1)
        sides = s
        
        # Ensure sides are > 0 and finite
        sides = np.maximum(sides, 1e-6)
        
        # Simple repair: if any square is out of bounds, shrink it
        for i in range(n):
            if centers[i, 0] - sides[i]/2 < 0 or centers[i, 0] + sides[i]/2 > 1 or \
               centers[i, 1] - sides[i]/2 < 0 or centers[i, 1] + sides[i]/2 > 1:
                # Shrink to fit
                max_s_x = 2 * min(centers[i, 0], 1 - centers[i, 0])
                max_s_y = 2 * min(centers[i, 1], 1 - centers[i, 1])
                sides[i] = min(sides[i], max_s_x, max_s_y, 1.0)
                sides[i] = max(sides[i], 1e-6)
                
        return centers, sides

    except Exception:
        # Fallback to grid
        k = 3
        side = 1.0 / k
        x = np.zeros(n)
        y = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            col = i % k
            row = i // k
            x[i] = (col + 0.5) * side
            y[i] = (row + 0.5) * side
            s[i] = side
        centers = np.stack([x, y], axis=1)
        return centers, s
