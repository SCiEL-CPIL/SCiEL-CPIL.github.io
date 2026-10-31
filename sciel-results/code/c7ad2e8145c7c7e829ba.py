"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 110  # seconds, safety margin before 120s limit

# EVOLVE-BLOCK-START

def _compute_score(circles):
    """
    circles: (n, 3) array with [cx, cy, r]
    Returns (value, valid).
    value = sum of radii if valid, else 0.
    Validity: all circles inside rectangle of perimeter 4, disjoint interiors.
    We find the minimal bounding rectangle (axis-aligned) and check perimeter <= 4.
    Also check pairwise distances >= sum of radii (with small tolerance).
    """
    n = len(circles)
    if n == 0:
        return 0.0, False
    
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Check radii positive
    if np.any(r <= 0):
        return 0.0, False
    
    # Bounding box
    xmin = np.min(cx - r)
    xmax = np.max(cx + r)
    ymin = np.min(cy - r)
    ymax = np.max(cy + r)
    
    width = xmax - xmin
    height = ymax - ymin
    
    if width <= 0 or height <= 0:
        return 0.0, False
    
    perimeter = 2 * (width + height)
    if perimeter > 4.0 + 1e-9:
        return 0.0, False
    
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    
    return np.sum(r), True


def _scale_to_perimeter(circles, target_perim=4.0):
    """Scale the configuration so the bounding rectangle has exactly target_perim."""
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    xmin = np.min(cx - r)
    xmax = np.max(cx + r)
    ymin = np.min(cy - r)
    ymax = np.max(cy + r)
    
    width = xmax - xmin
    height = ymax - ymin
    
    if width <= 0 or height <= 0:
        return circles
    
    current_perim = 2 * (width + height)
    if current_perim <= 0:
        return circles
    
    scale = target_perim / current_perim
    
    # Scale around the center of the bounding box
    cx_center = (xmin + xmax) / 2
    cy_center = (ymin + ymax) / 2
    
    new_cx = cx_center + (cx - cx_center) * scale
    new_cy = cy_center + (cy - cy_center) * scale
    new_r = r * scale
    
    return np.column_stack([new_cx, new_cy, new_r])


def _generate_initial_grid(n_circles):
    """Generate an initial grid-like configuration."""
    # Try different grid layouts
    best_val = -1
    best_circles = None
    
    # Try square-ish grids
    for cols in range(3, 8):
        rows = int(np.ceil(n_circles / cols))
        if rows * cols < n_circles:
            rows += 1
        
        # Create grid positions
        circles = []
        for i in range(n_circles):
            col = i % cols
            row = i // cols
            # Center of cell
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            # Radius: half the smaller cell dimension minus small gap
            r = 0.5 / max(cols, rows) - 0.001
            circles.append([x, y, r])
        
        circles = np.array(circles)
        # Scale to perimeter 4
        circles = _scale_to_perimeter(circles)
        val, valid = _compute_score(circles)
        if valid and val > best_val:
            best_val = val
            best_circles = circles
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n_circles)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n_circles])
        best_circles = _scale_to_perimeter(best_circles)
    
    return best_circles


def _local_search(circles, max_iter=5000, tol=1e-8):
    """
    Local search to maximize sum of radii while maintaining constraints.
    Uses gradient-based optimization with penalty for constraint violations.
    """
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(x):
        """Negative sum of radii + penalty for constraint violations."""
        circles = x.reshape(n, 3)
        cx = circles[:, 0]
        cy = circles[:, 1]
        r = circles[:, 2]
        
        # Penalty for negative radii
        penalty = 0
        if np.any(r < 1e-6):
            penalty += 100 * np.sum(np.maximum(0, 1e-6 - r)) ** 2
        
        # Penalty for overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += 100 * (min_dist - dist) ** 2
        
        # Compute bounding box and perimeter penalty
        xmin = np.min(cx - r)
        xmax = np.max(cx + r)
        ymin = np.min(cy - r)
        ymax = np.max(cy + r)
        width = xmax - xmin
        height = ymax - ymin
        perim = 2 * (width + height)
        
        if perim > 4.0:
            penalty += 100 * (perim - 4.0) ** 2
        
        # Objective: maximize sum of radii, so minimize negative sum
        return -np.sum(r) + penalty
    
    # Use L-BFGS-B
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    
    new_circles = result.x.reshape(n, 3)
    # Ensure radii are positive
    new_circles[:, 2] = np.maximum(new_circles[:, 2], 1e-6)
    
    return new_circles


def _expand_radii(circles, max_rounds=20):
    """
    Try to expand radii iteratively by adjusting positions.
    """
    n = len(circles)
    best_circles = circles.copy()
    best_val, _ = _compute_score(best_circles)
    
    for round_idx in range(max_rounds):
        # Try to increase all radii by a small factor
        factor = 1.0 + 0.01
        new_circles = best_circles.copy()
        new_circles[:, 2] *= factor
        
        # Now try to fix overlaps by moving circles
        # Use local search to find a valid configuration
        fixed = _local_search(new_circles, max_iter=2000)
        
        val, valid = _compute_score(fixed)
        if valid and val > best_val:
            best_circles = fixed
            best_val = val
        else:
            # Try smaller factor
            factor = 1.0 + 0.005
            new_circles = best_circles.copy()
            new_circles[:, 2] *= factor
            fixed = _local_search(new_circles, max_iter=1000)
            val, valid = _compute_score(fixed)
            if valid and val > best_val:
                best_circles = fixed
                best_val = val
            else:
                break
    
    return best_circles


def construct():
    start_time = time.time()
    
    # Generate initial configuration
    circles = _generate_initial_grid(N)
    
    # Run local search
    circles = _local_search(circles, max_iter=5000)
    
    # Scale to perimeter 4
    circles = _scale_to_perimeter(circles)
    
    # Expand radii iteratively
    circles = _expand_radii(circles)
    
    # Final local search
    circles = _local_search(circles, max_iter=3000)
    
    # Scale to perimeter 4
    circles = _scale_to_perimeter(circles)
    
    # Check validity
    val, valid = _compute_score(circles)
    if not valid:
        # Fallback to initial grid
        circles = _generate_initial_grid(N)
        circles = _local_search(circles, max_iter=2000)
        circles = _scale_to_perimeter(circles)
    
    return circles

# EVOLVE-BLOCK-END
