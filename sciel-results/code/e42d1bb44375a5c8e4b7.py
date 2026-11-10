import numpy as np
import time
from scipy.optimize import minimize
import itertools

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 axis-aligned squares in the unit square.
    Strategy: Multi-start Local Search with strict feasibility projection.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    timeout = 590  # Safety margin
    
    # State representation: x = [cx1, cy1, s1, cx2, cy2, s2, ...]
    # cx, cy in [0, 1], s in [0, 1]
    
    def unpack(x):
        centers = np.array([x[i*3], x[i*3+1]] for i in range(n))
        sides = np.array([x[i*3+2] for i in range(n)])
        return centers, sides

    def pack(centers, sides):
        x = np.zeros(n * 3)
        for i in range(n):
            x[i*3] = centers[i, 0]
            x[i*3+1] = centers[i, 1]
            x[i*3+2] = sides[i]
        return x

    def is_valid(centers, sides):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Check boundaries
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        
        # Check overlap
        # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def objective(x):
        # We want to maximize sum(sides), so minimize -sum(sides)
        # If invalid, return a large penalty
        centers, sides = unpack(x)
        if not is_valid(centers, sides):
            return 1e9
        return -np.sum(sides)

    def project_to_feasible(x):
        """
        Attempts to project an invalid configuration back into the feasible set.
        Strategy: 
        1. Clip sides to [0, 1].
        2. Clip centers to keep squares inside [0,1].
        3. If overlaps remain, iteratively shrink the largest overlapping squares 
           or move them apart. For simplicity and robustness in local search,
           we will just shrink sides slightly until valid, as shrinking is always feasible.
        """
        centers, sides = unpack(x)
        s = sides.copy()
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        
        # 1. Boundary constraints
        for i in range(n):
            s[i] = min(s[i], 1.0)
            s[i] = max(s[i], 0.0)
            # Adjust center to keep inside
            if cx[i] - s[i]/2 < 0:
                cx[i] = s[i]/2
            if cx[i] + s[i]/2 > 1:
                cx[i] = 1 - s[i]/2
            if cy[i] - s[i]/2 < 0:
                cy[i] = s[i]/2
            if cy[i] + s[i]/2 > 1:
                cy[i] = 1 - s[i]/2
        
        # 2. Overlap constraints
        # Iteratively shrink squares that are overlapping
        max_iter = 50
        for _ in range(max_iter):
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    t = (s[i] + s[j]) / 2.0 - tol
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    if dx < t and dy < t:
                        # Overlap detected
                        valid = False
                        # Shrink both squares slightly to resolve
                        # We shrink the larger one more to preserve total area/sum? 
                        # Actually, to maximize sum, we should shrink minimally.
                        # But for projection, just ensure validity.
                        shrink_factor = 0.95
                        s[i] *= shrink_factor
                        s[j] *= shrink_factor
                        # Re-clip boundaries after shrinking
                        if cx[i] - s[i]/2 < 0: cx[i] = s[i]/2
                        if cx[i] + s[i]/2 > 1: cx[i] = 1 - s[i]/2
                        if cy[i] - s[i]/2 < 0: cy[i] = s[i]/2
                        if cy[i] + s[i]/2 > 1: cy[i] = 1 - s[i]/2
                        
                        if cx[j] - s[j]/2 < 0: cx[j] = s[j]/2
                        if cx[j] + s[j]/2 > 1: cx[j] = 1 - s[j]/2
                        if cy[j] - s[j]/2 < 0: cy[j] = s[j]/2
                        if cy[j] + s[j]/2 > 1: cy[j] = 1 - s[j]/2
            
            if valid:
                break
        
        if not valid:
            # Fallback: return a known valid small grid
            k = int(np.ceil(np.sqrt(n)))
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            s = np.full(n, side)
            
        return pack(np.stack([cx, cy], axis=1), s)

    def generate_initial_configurations():
        """
        Generate diverse starting points.
        """
        configs = []
        
        # 1. Grid-based (3x3 + 1)
        # 3x3 grid of size 1/3, plus one small square in the center? 
        # Or 2x5 grid?
        # Let's try a few grid patterns
        patterns = [
            (3, 3), # 9 squares, 1 extra
            (2, 5), # 10 squares
            (1, 10), # 10 squares in a row
        ]
        
        for kx, ky in patterns:
            if kx * ky >= n:
                side = 1.0 / max(kx, ky)
                # Place first n squares in grid
                idx = np.arange(n)
                cols = idx % kx
                rows = idx // kx
                cx = (cols + 0.5) * side
                cy = (rows + 0.5) * side
                s = np.full(n, side)
                # Add some random jitter
                for _ in range(5):
                    jitter = np.random.uniform(-0.05, 0.05, n)
                    cx_j = cx + jitter
                    cy_j = cy + np.random.uniform(-0.05, 0.05, n)
                    s_j = s * np.random.uniform(0.8, 1.0, n)
                    x = pack(np.stack([cx_j, cy_j], axis=1), s_j)
                    x = project_to_feasible(x)
                    configs.append(x)
        
        # 2. Random valid configurations via rejection sampling
        # Try to place squares randomly, shrinking if they collide
        for _ in range(20):
            cx = np.random.uniform(0, 1, n)
            cy = np.random.uniform(0, 1, n)
            s = np.random.uniform(0.1, 0.5, n)
            x = pack(np.stack([cx, cy], axis=1), s)
            x = project_to_feasible(x)
            configs.append(x)
            
        # 3. "Corner" strategy: Put big squares in corners
        # 4 corners + 1 center + 5 in between?
        # Just generate a random permutation of sizes and place them greedily
        for _ in range(20):
            s = np.sort(np.random.uniform(0.1, 0.4, n))[::-1] # Largest first
            cx = np.zeros(n)
            cy = np.zeros(n)
            # Greedy placement
            placed = []
            for i in range(n):
                # Try to place square i in a random spot
                placed_x = False
                for _ in range(100):
                    cx_try = np.random.uniform(s[i]/2, 1 - s[i]/2)
                    cy_try = np.random.uniform(s[i]/2, 1 - s[i]/2)
                    # Check collision with placed
                    collides = False
                    for j in range(len(placed)):
                        pj = placed[j]
                        if abs(cx_try - pj[0]) < (s[i] + s[pj[1]])/2 - tol:
                            if abs(cy_try - pj[2]) < (s[i] + s[pj[1]])/2 - tol:
                                collides = True
                                break
                    if not collides:
                        cx[i] = cx_try
                        cy[i] = cy_try
                        placed.append((cx_try, i, cy_try))
                        placed_x = True
                        break
                if not placed_x:
                    # Shrink and try again or give up on this config
                    s[i] *= 0.8
                    # Simple fallback: put it in a corner
                    cx[i] = s[i]/2
                    cy[i] = s[i]/2
                    placed.append((cx[i], i, cy[i]))
            x = pack(np.stack([cx, cy], axis=1), s)
            x = project_to_feasible(x)
            configs.append(x)
            
        return configs

    # Main Optimization Loop
    best_score = -np.inf
    best_x = None
    initial_configs = generate_initial_configs() if False else generate_initial_configurations()
    
    # We will run Nelder-Mead on each initial config
    # Nelder-Mead is good for small n (30 vars) and non-smooth (due to projection/validity)
    
    num_starts = len(initial_configs)
    # If we have time, we can generate more random starts
    
    for i in range(num_starts):
        if time.time() - start_time > timeout:
            break
            
        x0 = initial_configs[i]
        
        # Run Nelder-Mead
        # Use a tight tolerance and reasonable maxiter
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            
            if res.fun < 1e8: # Valid solution found
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_x = res.x
                    # Print progress (optional, but good for debugging)
                    # print(f"Start {i}: Score {score}")
                    
        except Exception as e:
            pass
            
        # If we have lots of time left, do more random starts
        if time.time() - start_time < timeout - 10:
             # Generate 5 more random starts
             for _ in range(5):
                 if time.time() - start_time > timeout:
                     break
                 cx = np.random.uniform(0, 1, n)
                 cy = np.random.uniform(0, 1, n)
                 s = np.random.uniform(0.1, 0.6, n)
                 x_new = pack(np.stack([cx, cy], axis=1), s)
                 x_new = project_to_feasible(x_new)
                 
                 try:
                     res = minimize(objective, x_new, method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
                     if res.fun < 1e8:
                         score = -res.fun
                         if score > best_score:
                             best_score = score
                             best_x = res.x
                 except:
                     pass

    # Final Polish: Run a few more iterations of Nelder-Mead on the best solution found
    # with tighter tolerances
    if best_x is not None:
        try:
             res_final = minimize(objective, best_x, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
             if res_final.fun < 1e8:
                 best_x = res_final.x
                 best_score = -res_final.fun
        except:
             pass

    # Fallback if no valid solution found
    if best_x is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    centers, sides = unpack(best_x)
    
    # Final validation check
    if not is_valid(centers, sides):
        # Project one last time
        best_x = project_to_feasible(best_x)
        centers, sides = unpack(best_x)
        if not is_valid(centers, sides):
             # Fallback
             k = int(np.ceil(np.sqrt(n)))
             side = 1.0 / k
             idx = np.arange(n)
             cols = idx % k
             rows = idx // k
             cx = (cols + 0.5) * side
             cy = (rows + 0.5) * side
             centers = np.stack([cx, cy], axis=1).astype(np.float64)
             sides = np.full(n, side, dtype=np.float64)
             return centers, sides

    return centers, sides
