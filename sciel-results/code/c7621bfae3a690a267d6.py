"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(circles):
    """Compute the sum of radii for a valid configuration."""
    return np.sum(circles[:, 2])

def _is_valid(circles, tol=1e-7):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _fix_validity(circles):
    """Fix minor validity issues by shrinking radii and adjusting positions."""
    circles = circles.copy()
    for _ in range(100):
        valid = True
        # Check boundaries
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Shrink radius to fit in boundary if needed
            max_r = min(x, y, 1 - x, 1 - y)
            if r > max_r:
                circles[i, 2] = max_r
                valid = False
        
        # Check overlaps
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    # Shrink the larger circle or both
                    overlap = min_dist - dist
                    if ri >= rj:
                        circles[i, 2] = max(0, ri - overlap / 2)
                    else:
                        circles[j, 2] = max(0, rj - overlap / 2)
                    valid = False
        
        if valid:
            break
    
    return circles

def _maximize_radii(circles):
    """Try to maximize radii from a valid configuration by local optimization."""
    circles = circles.copy()
    n = len(circles)
    
    # For each circle, try to increase its radius while maintaining validity
    for i in range(n):
        x, y, r = circles[i]
        # Maximum possible radius given position
        max_r = min(x, y, 1 - x, 1 - y)
        
        # Check constraints with other circles
        for j in range(n):
            if i == j:
                continue
            xj, yj, rj = circles[j]
            dist = np.sqrt((x - xj)**2 + (y - yj)**2)
            # r + rj <= dist => r <= dist - rj
            max_r = min(max_r, dist - rj)
        
        circles[i, 2] = max(0, max_r)
    
    return circles

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s safety margin
    
    # Initial grid-based configuration
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_circles = np.array(cells[:N])
    best_score = _compute_score(best_circles)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 20
    
    for start in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial configuration
        if start == 0:
            # Use grid as initial
            circles = best_circles.copy()
        else:
            # Random initialization with some structure
            # Place circles in a loose grid with random perturbations
            side = int(np.ceil(np.sqrt(N)))
            circles = []
            for k in range(N):
                i = k // side
                j = k % side
                x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                r = 0.5 / side - 0.02
                circles.append((x, y, r))
            circles = np.array(circles)
        
        # Define objective: minimize negative sum of radii subject to constraints
        def objective(params):
            if time.time() > deadline - 2:
                return 1e10
            pos = params[:2 * N]
            x = pos[0::2]
            y = pos[1::2]
            r = params[2 * N:]
            
            # Penalty for being outside unit square
            penalty = 0
            for i in range(N):
                if x[i] - r[i] < 0:
                    penalty += (x[i] - r[i])**2
                if x[i] + r[i] > 1:
                    penalty += (x[i] + r[i] - 1)**2
                if y[i] - r[i] < 0:
                    penalty += (y[i] - r[i])**2
                if y[i] + r[i] > 1:
                    penalty += (y[i] + r[i] - 1)**2
            
            # Penalty for overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            return -np.sum(r) + 100 * penalty
        
        # Initial parameters
        x0 = np.zeros(3 * N)
        x0[0::2] = circles[:, 0]
        x0[1::2] = circles[:, 1]
        x0[2 * N:] = circles[:, 2]
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
            bounds.append((0.01, 0.99))  # y
            bounds.append((0.001, 0.5))  # r
        
        # Optimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.success or result.fun < 0:
                pos = result.x[:2 * N]
                r = result.x[2 * N:]
                new_circles = np.column_stack((pos[0::2], pos[1::2], r))
                
                # Fix validity
                new_circles = _fix_validity(new_circles)
                
                if _is_valid(new_circles):
                    new_score = _compute_score(new_circles)
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = new_circles
        except Exception:
            pass
    
    # Final post-processing: maximize radii
    best_circles = _maximize_radii(best_circles)
    best_circles = _fix_validity(best_circles)
    
    if _is_valid(best_circles):
        return best_circles
    
    # Fallback to original grid
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
