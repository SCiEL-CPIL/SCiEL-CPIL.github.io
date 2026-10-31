"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _evaluate_solution(x):
    """
    x: array of shape (3*N,) with [cx, cy, r] for each circle.
    We need to find a rectangle with perimeter 4 (i.e., 2*(w+h)=4 => w+h=2)
    that contains all circles, and circles must be disjoint.
    
    Strategy: 
    - For a given set of circles (positions and radii), we need to find if there exists
      a rectangle of width w, height h with w+h=2 that contains all circles.
    - The minimal bounding box of the circles has width W and height H.
    - We need W <= w and H <= h for some w+h=2.
    - This is possible iff W + H <= 2.
    - So the constraint is: (max(cx) - min(cx) + 2*max(r) ... wait, no.
    
    Actually, the bounding rectangle must contain all circles.
    The minimal axis-aligned bounding rectangle that contains all circles has:
      left = min(cx_i - r_i), right = max(cx_i + r_i)
      bottom = min(cy_i - r_i), top = max(cy_i + r_i)
    Width = right - left, Height = top - bottom.
    We need Width + Height <= 2.
    
    Also, circles must be disjoint: distance between centers >= r_i + r_j.
    
    Objective: maximize sum of radii.
    """
    x = x.reshape(N, 3)
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Constraint: radii must be positive
    if np.any(r <= 0):
        return -1e10
    
    # Bounding box
    left = np.min(cx - r)
    right = np.max(cx + r)
    bottom = np.min(cy - r)
    top = np.max(cy + r)
    width = right - left
    height = top - bottom
    
    # Perimeter constraint: width + height <= 2
    perimeter_violation = max(0, width + height - 2.0)
    
    # Disjointness constraints
    min_dist_violation = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                min_dist_violation += (min_dist - dist)
    
    total_violation = perimeter_violation + min_dist_violation
    
    if total_violation > 1e-6:
        return -1e10 + total_violation  # Infeasible
    
    return np.sum(r)


def construct():
    # Use a grid-based initialization with some randomness, then optimize
    np.random.seed(42)
    
    # Initial guess: place circles in a grid pattern
    # We want to maximize sum of radii, so let's start with a reasonable configuration
    # and use optimization to improve it.
    
    # Start with a hexagonal-like packing or grid
    # Let's try a 5x5 grid (25 cells) and pick 21
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9])
    
    # Shuffle and take first N
    np.random.shuffle(positions)
    positions = positions[:N]
    
    x0 = np.array(positions).flatten()
    
    # Try multiple random starts
    best_x = None
    best_val = -1e10
    
    for attempt in range(5):
        # Random perturbation
        if attempt == 0:
            x = x0.copy()
        else:
            x = x0 + np.random.randn(N * 3) * 0.1
            # Ensure radii are positive
            x[2::3] = np.abs(x[2::3]) + 0.01
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(
                _evaluate_solution,
                x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            val = _evaluate_solution(result.x)
            if val > best_val:
                best_val = val
                best_x = result.x
        except:
            pass
    
    if best_x is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract circles
    circles = best_x.reshape(N, 3)
    
    # Verify and adjust if needed
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Shift to center in [0, w] x [0, h] where w+h=2
    left = np.min(cx - r)
    right = np.max(cx + r)
    bottom = np.min(cy - r)
    top = np.max(cy + r)
    width = right - left
    height = top - bottom
    
    if width + height > 2.0:
        # Scale down
        scale = 2.0 / (width + height) * 0.999
        cx = (cx - (left + right) / 2) * scale + 1.0
        cy = (cy - (bottom + top) / 2) * scale + 1.0
        r = r * scale
        circles = np.column_stack([cx, cy, r])
    
    # Ensure all radii are positive
    r = np.maximum(r, 1e-6)
    circles = np.column_stack([cx, cy, r])
    
    return circles


# EVOLVE-BLOCK-END
