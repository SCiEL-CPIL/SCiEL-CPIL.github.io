"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _is_valid(circles, tol=1e-7):
    """Check if all circles are inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    # Boundary check
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Overlap check
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _objective(circles):
    """Sum of radii, penalized if invalid."""
    if not _is_valid(circles):
        return -1e6
    return np.sum(circles[:, 2])

def _hex_packing_init():
    """Generate a good initial hexagonal packing for 26 circles."""
    # Start with a hexagonal grid arrangement
    # We'll try different row/column configurations
    
    best_circles = None
    best_score = -1e6
    
    # Try grid-like arrangements with hex offset
    # For 26 circles, try various rectangular arrangements
    for rows in range(3, 8):
        for cols in range(3, 9):
            if rows * cols < N:
                continue
            
            # Calculate spacing
            # Hex packing: horizontal spacing = 2r, vertical spacing = sqrt(3)*r
            # Fit in unit square: cols * 2r <= 1, rows * sqrt(3)*r + r <= 1
            # r <= 1/(2*cols) and r <= 1/(rows*sqrt(3) + 1)
            
            r_from_cols = 0.5 / cols
            r_from_rows = 1.0 / (rows * np.sqrt(3) + 1)
            r = min(r_from_cols, r_from_rows) * 0.95  # small margin
            
            circles = []
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= N:
                        break
                    # Hex offset: odd rows shifted by r
                    x_offset = r if i % 2 == 1 else 0
                    x = (j + 0.5) * 2 * r + x_offset
                    y = (i + 0.5) * np.sqrt(3) * r + 0.5 * r
                    
                    # Keep in bounds
                    if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                        circles.append([x, y, r])
            
            if len(circles) >= N:
                circles = np.array(circles[:N])
                score = _objective(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
    
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles

def _local_optimization(circles, max_iters=5000, tol=1e-10):
    """Local optimization using gradient-free approach (Nelder-Mead style)."""
    circles = circles.copy()
    n = len(circles)
    
    # We'll use a simple iterative improvement: push overlapping circles apart
    # and pull boundary circles inward
    
    for iteration in range(max_iters):
        if _remaining_time() < 5.0:
            break
            
        improved = False
        
        for i in range(n):
            xi, yi, ri = circles[i]
            
            # Check boundary constraints
            # If circle touches or exceeds boundary, move it inward
            if xi - ri < 0:
                xi = ri + 1e-8
                improved = True
            if xi + ri > 1:
                xi = 1 - ri - 1e-8
                improved = True
            if yi - ri < 0:
                yi = ri + 1e-8
                improved = True
            if yi + ri > 1:
                yi = 1 - ri - 1e-8
                improved = True
            
            # Check overlaps with other circles
            for j in range(n):
                if i == j:
                    continue
                
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = ri + rj
                
                if dist < min_dist - tol:
                    # Overlap: push circle i away from j
                    if dist > 1e-12:
                        push = (min_dist - dist) * 0.5
                        xi += (dx / dist) * push
                        yi += (dy / dist) * push
                    else:
                        # Nearly coincident, push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        xi += push * np.cos(angle)
                        yi += push * np.sin(angle)
                    improved = True
            
            circles[i, 0] = xi
            circles[i, 1] = yi
        
        if not improved:
            break
    
    return circles

def _increase_radii(circles, max_attempts=200):
    """Try to increase radii while maintaining validity."""
    circles = circles.copy()
    n = len(circles)
    
    for attempt in range(max_attempts):
        if _remaining_time() < 5.0:
            break
        
        # Try to increase a random circle's radius
        i = np.random.randint(n)
        old_r = circles[i, 2]
        new_r = old_r * 1.001 + 1e-6
        
        if new_r > 0.5:  # Can't be larger than 0.5 in unit square
            continue
        
        circles[i, 2] = new_r
        
        # Check if still valid; if not, revert
        if not _is_valid(circles, tol=1e-8):
            circles[i, 2] = old_r
        else:
            # Valid, keep the increase and do a quick local fix
            circles = _local_optimization(circles, max_iters=100)
            if not _is_valid(circles, tol=1e-8):
                circles[i, 2] = old_r
                # Revert positions too
                # This is a simplification; in practice we'd need to save state
    
    return circles

def construct():
    # Generate initial packing
    circles = _hex_packing_init()
    
    # Local optimization to fix any issues
    circles = _local_optimization(circles)
    
    # Try to increase radii
    best_circles = circles.copy()
    best_score = _objective(circles)
    
    # Multiple restarts with different initializations
    while _remaining_time() > 10.0:
        # Perturb the current best and try to improve
        perturbed = best_circles.copy()
        # Small random perturbations to positions
        perturbed[:, 0] += np.random.uniform(-0.01, 0.01, N)
        perturbed[:, 1] += np.random.uniform(-0.01, 0.01, N)
        # Clamp to valid range
        for i in range(N):
            perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2], 1 - perturbed[i, 2])
            perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2], 1 - perturbed[i, 2])
        
        # Optimize the perturbed version
        optimized = _local_optimization(perturbed)
        score = _objective(optimized)
        
        if score > best_score:
            best_score = score
            best_circles = optimized.copy()
            
            # Try to increase radii from this better position
            improved = _increase_radii(best_circles, max_attempts=50)
            improved_score = _objective(improved)
            if improved_score > best_score:
                best_score = improved_score
                best_circles = improved.copy()
    
    # Final local optimization
    best_circles = _local_optimization(best_circles, max_iters=2000)
    
    # Ensure validity
    if not _is_valid(best_circles):
        # Fallback to a known valid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
