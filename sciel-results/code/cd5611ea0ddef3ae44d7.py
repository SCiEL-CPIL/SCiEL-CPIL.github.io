import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # Variables: cx, cy, s for each square
    # x = [cx0, cy0, s0, cx1, cy1, s1, ...]
    
    def unpack(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def penalty(x):
        centers, sides = unpack(x)
        cx, cy = centers[:, 0], centers[:, 1]
        s = sides
        
        # Boundary penalties
        # Squares must be inside [0, 1]
        # cx - s/2 >= 0 => cx >= s/2
        # cx + s/2 <= 1 => cx <= 1 - s/2
        # If violated, penalize quadratically
        b_pen = 0.0
        b_pen += np.sum(np.maximum(0, s/2 - cx)**2)
        b_pen += np.sum(np.maximum(0, s/2 - cy)**2)
        b_pen += np.sum(np.maximum(0, (cx - (1 - s/2)))**2)
        b_pen += np.sum(np.maximum(0, (cy - (1 - s/2)))**2)
        
        # Overlap penalties
        o_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s[i] + s[j]) / 2
                req_y = (s[i] + s[j]) / 2
                # If dx < req_x AND dy < req_y, they overlap
                # Penalty is the product of the violations
                vx = max(0, req_x - dx)
                vy = max(0, req_y - dy)
                o_pen += vx * vy
                
        # Small penalty for very small sides to avoid degenerate solutions
        s_pen = np.sum(np.maximum(0, 0.01 - s)**2)
        
        return b_pen + o_pen + s_pen

    def objective(x):
        centers, sides = unpack(x)
        return -np.sum(sides) + 100.0 * penalty(x)

    # Initial guesses
    inits = []
    
    # 1. 3x3 grid + center
    # 9 squares of 1/3, 1 square of 0.1 in center
    # Adjust to avoid overlap: shrink 3x3 to 0.31
    s_grid = 0.31
    s_cen = 0.1
    centers = []
    for r in range(3):
        for c in range(3):
            centers.append([(c+0.5)*s_grid, (r+0.5)*s_grid])
    centers.append([0.5, 0.5])
    sides = [s_grid]*9 + [s_cen]
    x0 = np.array(centers).flatten()
    x0 = np.concatenate([x0, sides])
    inits.append(x0)
    
    # 2. 1 large + 9 small
    # Large 0.4 at (0.2, 0.2)
    # 9 small in remaining space
    # This is hard to initialize perfectly. Let's use a rough guess.
    # Large square at bottom-left
    # 4 squares of 0.2 to its right?
    # 4 squares of 0.2 above it?
    # 1 square in top-right?
    # Let's just use random for the rest.
    
    # Random inits
    for _ in range(5):
        s = np.random.uniform(0.1, 0.4, n)
        cx = np.random.uniform(s/2, 1-s/2, n)
        cy = np.random.uniform(s/2, 1-s/2, n)
        x0 = np.concatenate([np.array([cx, cy]).T.flatten(), s])
        inits.append(x0)
        
    best_x = None
    best_score = -np.inf
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        # Use Powell for derivative-free optimization
        # Maxiter is important
        res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success:
            x_opt = res.x
            centers, sides = unpack(x_opt)
            # Check validity
            if penalty(x_opt) < 1e-5:
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_x = x_opt
                    
    if best_x is None:
        # Fallback to baseline
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        return centers, sides
        
    centers, sides = unpack(best_x)
    
    # Final polish: ensure constraints are strictly satisfied
    # This is a simple projection or clipping
    # If a square is out of bounds, move it in
    # If overlap, shrink it?
    # For now, return as is if penalty is low.
    
    return centers, sides
