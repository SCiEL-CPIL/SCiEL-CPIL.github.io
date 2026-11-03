import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Helper function to unpack parameters
    # params: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    def unpack(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return x, y, r

    # Objective function to minimize: -sum(r) + penalty
    def objective(params):
        x, y, r = unpack(params)
        
        # Penalty for boundary
        # x must be in [r, 1-r]
        # If x < r, penalty is (r-x)^2. If x > 1-r, penalty is (x-(1-r))^2
        pen_x_low = np.maximum(0, r - x)**2
        pen_x_high = np.maximum(0, x - (1 - r))**2
        pen_y_low = np.maximum(0, r - y)**2
        pen_y_high = np.maximum(0, y - (1 - r))**2
        boundary_pen = np.sum(pen_x_low + pen_x_high + pen_y_low + pen_y_high)
        
        # Penalty for overlap
        # dist(i,j) >= r_i + r_j
        # Violation = max(0, r_i + r_j - dist(i,j))^2
        overlap_pen = 0.0
        # Vectorized calculation for speed
        # diff_x = x[:, None] - x[None, :]
        # dist = sqrt(diff_x**2 + diff_y**2)
        # min_dist = r[:, None] + r[None, :]
        # viol = np.maximum(0, min_dist - dist)
        # overlap_pen = np.sum(viol**2)
        
        # For N=26, N^2 is 676, which is small. Vectorization is fast.
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        viol = np.maximum(0, min_dist - dist)
        # Exclude diagonal
        np.fill_diagonal(viol, 0)
        overlap_pen = np.sum(viol**2)
        
        total_pen = boundary_pen + overlap_pen
        sum_r = np.sum(r)
        
        return -sum_r + 100.0 * total_pen

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append([
                    (i + 0.5) / side,
                    (j + 0.5) / side,
                    0.5 / side * 0.9 # Slightly smaller to avoid initial overlap penalty spikes
                ])
        if len(cells) < N:
            # Pad if necessary, though side*side >= N
            pass
        guesses.append(np.array(cells[:N]).flatten())
        
        # 2. Random
        np.random.seed(42)
        for _ in range(5):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.1 + 0.05 # Small initial radii
            params = np.column_stack((x, y, r)).flatten()
            guesses.append(params)
            
        # 3. Hexagonal-ish
        # Just jitter the grid
        grid = guesses[0].copy()
        x, y, r = unpack(grid)
        x += np.random.normal(0, 0.01, N)
        y += np.random.normal(0, 0.01, N)
        # Clip to valid range roughly
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
        guesses.append(np.column_stack((x, y, r)).flatten())
        
        return guesses

    best_result = None
    best_score = -np.inf
    
    guesses = get_initial_guesses()
    
    for i, guess in enumerate(guesses):
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B for speed and efficiency with gradients
            res = minimize(objective, guess, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Verify validity and calculate raw score
                x, y, r = unpack(res.x)
                
                # Check constraints strictly
                valid = True
                # Boundary
                if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
                    valid = False
                if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
                    valid = False
                
                # Overlap
                if valid:
                    diff_x = x[:, None] - x[None, :]
                    diff_y = y[:, None] - y[None, :]
                    dist = np.sqrt(diff_x**2 + diff_y**2)
                    min_dist = r[:, None] + r[None, :]
                    np.fill_diagonal(min_dist, 0)
                    if np.any(dist < min_dist - 1e-9):
                        valid = False
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = res.x
                else:
                    # If invalid, try to repair by shrinking radii? 
                    # Or just skip. The penalty should have prevented this.
                    # If we get here, the optimizer failed to satisfy constraints.
                    pass
                    
        except Exception:
            continue
            
    # If no valid result found, return a safe trivial solution
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format output
    x, y, r = unpack(best_result)
    # Final safety clip
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # One last check and repair if clipping caused overlap (unlikely if r was small enough)
    # Re-check overlap
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    dist = np.sqrt(diff_x**2 + diff_y**2)
    min_dist = r[:, None] + r[None, :]
    np.fill_diagonal(min_dist, 0)
    
    # If any overlap remains, shrink radii of the overlapping pairs
    # This is a simple repair heuristic
    for _ in range(10):
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        if np.max(viol) < 1e-9:
            break
        # Reduce radii of circles involved in max violation
        idx_i, idx_j = np.unravel_index(np.argmax(viol), viol.shape)
        # Reduce both radii slightly
        reduction = viol[idx_i, idx_j] / 2 + 1e-6
        r[idx_i] = max(r[idx_i] - reduction, 1e-6)
        r[idx_j] = max(r[idx_j] - reduction, 1e-6)
        # Update positions to center in available space? No, just shrink.
        # Recalculate dists
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        dist = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
        
    return np.column_stack((x, y, r))
