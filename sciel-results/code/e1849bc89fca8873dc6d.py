"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _validate(circles):
    """Check if circles are valid (disjoint, inside perimeter 4)."""
    n = len(circles)
    if n != N:
        return False
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
        
    # Perimeter check
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return False
        
    # Disjoint check
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True

def _score(circles):
    if not _validate(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _objective(params):
    """
    Minimize negative sum of radii + penalties.
    params: [x1, y1, r1, ...]
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    penalty = 0.0
    
    # Radii positive
    penalty += 1000 * np.sum(np.maximum(0, -rs)) ** 2
    
    # Perimeter constraint
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    if perimeter > 4:
        penalty += 500 * (perimeter - 4) ** 2
        
    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = rs[i] + rs[j] - dist
            if violation > 0:
                penalty += 500 * violation ** 2
                
    return -np.sum(rs) + penalty

def _hex_seed(aspect_ratio):
    """
    Generate a hexagonal seed for 21 circles.
    We use a 5-row hex pattern: 4, 5, 4, 5, 3 circles.
    """
    # Rows of circles
    rows = [4, 5, 4, 5, 3]
    circles = []
    
    # Normalize hex geometry
    # In a hex grid, vertical distance is sqrt(3)/2 * side
    # Horizontal distance is side
    # We want to fit this into a box of aspect_ratio W/H
    # Let's define the extent in hex units
    # Width in units: max row count + 0.5 (for offsets)
    # Height in units: 4 * sqrt(3)/2
    
    # We will place them in a unit square first, then scale
    # Hex side length s = 1
    # Row 0 (4 circles): x = 0, 1, 2, 3 ; y = 0
    # Row 1 (5 circles): x = -0.5, 0.5, 1.5, 2.5, 3.5 ; y = sqrt(3)/2
    # Row 2 (4 circles): x = 0, 1, 2, 3 ; y = sqrt(3)
    # Row 3 (5 circles): x = -0.5, 0.5, 1.5, 2.5, 3.5 ; y = 3*sqrt(3)/2
    # Row 4 (3 circles): x = 0, 1, 2 ; y = 2*sqrt(3)
    
    h = np.sqrt(3) / 2
    y_coords = [0, h, 2*h, 3*h, 4*h]
    
    centers = []
    for i, count in enumerate(rows):
        y = y_coords[i]
        # Center the row horizontally
        # Max width is 4 units (from -0.5 to 3.5 is 4 units)
        # Row with 4: 0 to 3 (width 3)
        # Row with 5: -0.5 to 3.5 (width 4)
        # Row with 3: 0 to 2 (width 2)
        # Let's center all rows around x=1.5 (center of width 4)
        center_x = 1.5
        start_x = center_x - (count - 1) / 2.0
        for k in range(count):
            x = start_x + k
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Calculate bounding box of centers
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    w_centers = max_x - min_x
    h_centers = max_y - min_y
    
    # We want the final rectangle (including radii) to have aspect_ratio
    # Approximation: The radii will be roughly uniform initially.
    # Let's set initial radius r0 such that they touch.
    # In hex grid, distance between centers is 1. So r0 = 0.5.
    r0 = 0.5
    
    # The total width will be w_centers + 2*r0
    # The total height will be h_centers + 2*r0
    total_w = w_centers + 2*r0
    total_h = h_centers + 2*r0
    
    # We want total_w / total_h = aspect_ratio
    # But we can scale the whole thing.
    # Let's just place them in a box of width 1, height 1/aspect_ratio?
    # No, the perimeter is fixed.
    # Let's scale the centers to fit a box of width W, height H where 2(W+H)=4.
    # W = 2 * aspect_ratio / (1 + aspect_ratio)
    # H = 2 / (1 + aspect_ratio)
    
    W = 2 * aspect_ratio / (1 + aspect_ratio)
    H = 2 / (1 + aspect_ratio)
    
    # Scale centers to fit in [r0, W-r0] x [r0, H-r0]?
    # This is tricky because r0 is part of the variable.
    # Simple approach: Scale centers to fit in a box slightly smaller than W,H
    # and set r0 to be the min distance / 2.
    
    # Normalize centers to [0,1] x [0,1]
    centers_norm = (centers - [min_x, min_y]) / [w_centers, h_centers]
    
    # Scale to fit in [0.1, 0.9] to leave room for radii
    scale_x = 0.8
    scale_y = 0.8
    centers_scaled = centers_norm * [scale_x, scale_y] + [0.1, 0.1]
    
    # Now, we need to map this to the actual W, H.
    # But the aspect ratio of the seed might not match the target.
    # Let's just use the centers as is, and let the optimizer adjust.
    # We will initialize radii to be small enough to not overlap.
    
    # Calculate min distance between centers in the scaled space
    min_dist = np.inf
    for i in range(len(centers_scaled)):
        for j in range(i+1, len(centers_scaled)):
            d = np.linalg.norm(centers_scaled[i] - centers_scaled[j])
            if d < min_dist:
                min_dist = d
                
    r_init = min_dist * 0.45 # Leave some gap
    
    # Construct params
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers_scaled[i, 0]
        params[3*i+1] = centers_scaled[i, 1]
        params[3*i+2] = r_init
        
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_circles = None
    best_score = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        init_params = _hex_seed(ar)
        
        # Optimization
        try:
            res = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() > deadline:
                break
                
            circles = res.x.reshape(N, 3)
            # Ensure radii are positive
            circles[:, 2] = np.maximum(circles[:, 2], 1e-4)
            
            score = _score(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                
        except Exception:
            continue
            
    # If no valid solution found, fallback to a simple grid
    if best_circles is None:
        # Fallback: 5x5 grid, take 21
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
# EVOLVE-BLOCK-END
