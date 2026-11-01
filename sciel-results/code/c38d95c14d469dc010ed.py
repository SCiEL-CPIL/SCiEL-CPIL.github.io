"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _perimeter_constraint(w, h):
    """Return perimeter for width w, height h."""
    return 2 * (w + h)

def _scale_to_perimeter(circles, target_perim=4.0):
    """
    Given circles in some rectangle, scale them so the bounding rectangle has perimeter exactly target_perim.
    Returns scaled circles and the scale factor.
    """
    if len(circles) == 0:
        return circles, 1.0
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Bounding box of circle centers
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w <= 0 or h <= 0:
        return circles, 1.0
    
    current_perim = 2 * (w + h)
    
    if current_perim <= 0:
        return circles, 1.0
    
    # We want to scale so that new_perim = target_perim
    # If we scale by factor s, new_perim = s * current_perim
    # So s = target_perim / current_perim
    s = target_perim / current_perim
    
    # Scale circles around origin (0,0)
    # But first, let's center the bounding box at origin for cleaner scaling
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    scaled = circles.copy()
    scaled[:, 0] = (circles[:, 0] - cx) * s + cx
    scaled[:, 1] = (circles[:, 1] - cy) * s + cy
    scaled[:, 2] = circles[:, 2] * s
    
    return scaled, s

def _is_valid(circles, tol=1e-9):
    """Check if circles are disjoint and within a rectangle of perimeter <= 4."""
    if len(circles) == 0:
        return True
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii are positive
    if np.any(rs <= 0):
        return False
    
    # Check pairwise disjointness
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    # Check perimeter constraint
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w < 0 or h < 0:
        return False
    
    perim = 2 * (w + h)
    if perim > 4.0 + tol:
        return False
    
    return True

def _objective(x, N_circles):
    """
    Objective: maximize sum of radii, subject to constraints.
    x = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    We use a penalty method: minimize negative sum of radii + penalties for violations.
    """
    x = x.reshape(-1, 3)
    xs = x[:, 0]
    ys = x[:, 1]
    rs = x[:, 2]
    
    n = len(x)
    
    # Negative sum of radii (we want to maximize sum)
    obj = -np.sum(rs)
    
    # Penalty for overlap
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                # Quadratic penalty for overlap
                overlap = min_dist - dist
                penalty += 100.0 * overlap * overlap
    
    # Penalty for perimeter
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    perim = 2 * (w + h)
    if perim > 4.0:
        penalty += 100.0 * (perim - 4.0) ** 2
    
    # Penalty for negative radii
    if np.any(rs < 0):
        penalty += 1000.0 * np.sum(np.clip(-rs, 0, None) ** 2)
    
    return obj + penalty

def _gradients(x, N_circles):
    """Compute gradients numerically or analytically."""
    # Use numerical gradient for simplicity
    eps = 1e-7
    n = len(x)
    grad = np.zeros_like(x)
    f0 = _objective(x, N_circles)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        fp = _objective(xp, N_circles)
        grad[i] = (fp - f0) / eps
    return grad

def _initial_grid(N_circles):
    """Create an initial grid of circles."""
    # Use a 5x5 grid for 21 circles
    side = int(np.ceil(np.sqrt(N_circles)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N_circles:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
    return np.array(cells[:N_circles])

def _local_search(circles, max_iters=500, tol=1e-10):
    """
    Perform local search using gradient descent with adaptive step size.
    """
    x = circles.flatten().copy()
    n = len(circles)
    
    best_x = x.copy()
    best_obj = _objective(x, n)
    
    step_size = 0.01
    for it in range(max_iters):
        grad = _gradients(x, n)
        
        # Adaptive step size
        x_new = x - step_size * grad
        obj_new = _objective(x_new, n)
        
        if obj_new < best_obj:
            best_obj = obj_new
            best_x = x_new.copy()
            x = x_new
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.1)
        else:
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-8:
                break
    
    return best_x.reshape(-1, 3)

def _expand_and_relax(circles, deadline):
    """
    Expansion and relaxation strategy:
    1. Try to increase radii slightly
    2. Relax positions to maintain disjointness
    3. Scale to perimeter 4
    """
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    # Try small expansions
    for expansion in [1.01, 1.02, 1.03, 1.05]:
        if time.time() > deadline - 10:
            break
        
        expanded = circles.copy()
        expanded[:, 2] *= expansion
        
        # Local search to fix overlaps
        relaxed = _local_search(expanded, max_iters=200)
        
        # Scale to perimeter 4
        scaled, _ = _scale_to_perimeter(relaxed, 4.0)
        
        if _is_valid(scaled, tol=1e-6):
            score = np.sum(scaled[:, 2])
            if score > best_score:
                best_score = score
                best_circles = scaled.copy()
    
    return best_circles

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Initial grid
    circles = _initial_grid(N)
    
    # Scale to perimeter 4
    circles, _ = _scale_to_perimeter(circles, 4.0)
    
    # Initial local search
    circles = _local_search(circles, max_iters=300)
    circles, _ = _scale_to_perimeter(circles, 4.0)
    
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    # Expansion and relaxation loop
    while time.time() < deadline - 5:
        circles = _expand_and_relax(circles, deadline)
        
        if np.sum(circles[:, 2]) > best_score:
            best_score = np.sum(circles[:, 2])
            best_circles = circles.copy()
        else:
            # No improvement, break
            break
    
    # Final polish: high-precision local search
    if time.time() < deadline - 2:
        polished = _local_search(best_circles, max_iters=1000, tol=1e-12)
        polished, _ = _scale_to_perimeter(polished, 4.0)
        
        if _is_valid(polished, tol=1e-8):
            if np.sum(polished[:, 2]) > best_score:
                best_circles = polished
    
    return best_circles
# EVOLVE-BLOCK-END
