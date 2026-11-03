"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def _check_validity(circles):
    """Check if circles are valid (disjoint interiors, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0 or x < -r or x > 1 + r or y < -r or y > 1 + r:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist = ri + rj
            if dist_sq < min_dist * min_dist - 1e-9:
                return False
    return True

def _repair(circles, n):
    """
    Repair circles to ensure validity:
    - Clamp positions so circles stay inside [0,1]^2
    - Shrink radii to resolve overlaps
    """
    circles = circles.copy()
    # Clamp to unit square
    for i in range(n):
        x, y, r = circles[i]
        # Ensure circle is inside [0,1]^2
        x = min(max(x, r), 1 - r)
        y = min(max(y, r), 1 - r)
        circles[i] = [x, y, r]
    
    # Iteratively shrink radii to resolve overlaps
    for _ in range(100):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj:
                    valid = False
                    # Shrink the larger circle (or both) to fix overlap
                    overlap = ri + rj - dist
                    # Shrink both proportionally
                    shrink_i = overlap / 2
                    shrink_j = overlap / 2
                    circles[i][2] = max(0, ri - shrink_i)
                    circles[j][2] = max(0, rj - shrink_j)
        if valid:
            break
    
    # Final clamp after shrinking
    for i in range(n):
        x, y, r = circles[i]
        x = min(max(x, r), 1 - r)
        y = min(max(y, r), 1 - r)
        circles[i] = [x, y, r]
    
    return circles

def _objective_and_grad(x, n):
    """
    Objective: maximize sum of radii, penalize overlaps and boundary violations.
    x = [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    # Reshape
    circles = x.reshape(n, 3)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Objective: negative sum of radii (to minimize)
    obj = -np.sum(rs)
    
    # Penalty for overlaps (softplus-like smooth penalty)
    penalty = 0.0
    grad_penalty = np.zeros_like(x)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist = rs[i] + rs[j]
            # Penalty when dist < min_dist
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                violation = min_dist - dist
                # Smooth penalty: 0.5 * violation^2
                penalty += 0.5 * violation * violation
                # Gradient w.r.t. positions and radii
                # d(violation)/dxi = -dx/dist, etc.
                inv_dist = 1.0 / dist
                grad_xi = -dx * inv_dist * violation
                grad_yi = -dy * inv_dist * violation
                grad_xj = dx * inv_dist * violation
                grad_yj = dy * inv_dist * violation
                grad_ri = violation
                grad_rj = violation
                
                idx_i = i * 3
                idx_j = j * 3
                grad_penalty[idx_i] += grad_xi
                grad_penalty[idx_i + 1] += grad_yi
                grad_penalty[idx_i + 2] += grad_ri
                grad_penalty[idx_j] += grad_xj
                grad_penalty[idx_j + 1] += grad_yj
                grad_penalty[idx_j + 2] += grad_rj
    
    # Penalty for boundary violations
    for i in range(n):
        # Left boundary: x - r < 0
        viol_l = max(0, -(xs[i] - rs[i]))
        if viol_l > 0:
            penalty += 0.5 * viol_l * viol_l
            grad_penalty[i*3] += -viol_l
            grad_penalty[i*3+2] += viol_l
        # Right boundary: x + r > 1
        viol_r = max(0, xs[i] + rs[i] - 1)
        if viol_r > 0:
            penalty += 0.5 * viol_r * viol_r
            grad_penalty[i*3] += viol_r
            grad_penalty[i*3+2] += viol_r
        # Bottom boundary: y - r < 0
        viol_b = max(0, -(ys[i] - rs[i]))
        if viol_b > 0:
            penalty += 0.5 * viol_b * viol_b
            grad_penalty[i*3+1] += -viol_b
            grad_penalty[i*3+2] += viol_b
        # Top boundary: y + r > 1
        viol_t = max(0, ys[i] + rs[i] - 1)
        if viol_t > 0:
            penalty += 0.5 * viol_t * viol_t
            grad_penalty[i*3+1] += viol_t
            grad_penalty[i*3+2] += viol_t
    
    total_obj = obj + penalty
    grad = np.zeros_like(x)
    # Gradient of obj w.r.t. radii
    for i in range(n):
        grad[i*3+2] = -1.0
    grad += grad_penalty
    
    return total_obj, grad

def _grid_initial(n, side):
    """Generate grid-based initial positions."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([
                (i + 0.5) / side,
                (j + 0.5) / side,
                0.45 / side
            ])
    # Shuffle and take first n
    np.random.shuffle(cells)
    return cells[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = N_CIRCLES
    best_circles = None
    best_score = -np.inf
    
    # Multi-start with different random seeds and grid sizes
    attempts = 0
    max_attempts = 200
    
    for seed in range(max_attempts):
        if time.time() > deadline - 5.0:
            break
        
        attempts += 1
        
        # Choose a grid size for initialization
        side_options = [4, 5, 6, 7, 8]
        side = side_options[seed % len(side_options)]
        
        # Generate initial configuration
        np.random.seed(seed)
        cells = _grid_initial(n, side)
        
        # Add some random perturbation
        for c in cells:
            c[0] += np.random.uniform(-0.02, 0.02)
            c[1] += np.random.uniform(-0.02, 0.02)
            c[2] += np.random.uniform(-0.01, 0.01)
        
        x0 = np.array(cells).flatten()
        
        # Bounds: positions in [0,1], radii in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, 0.5))  # radius
        
        # Optimize
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.status == 0:
                circles = res.x.reshape(n, 3)
                # Repair to ensure validity
                circles = _repair(circles, n)
                
                if _check_validity(circles):
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles.copy()
        except Exception:
            continue
    
    # If no valid solution found, fall back to a simple grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
        # Ensure validity
        best_circles = _repair(best_circles, n)
    
    return best_circles
# EVOLVE-BLOCK-END
