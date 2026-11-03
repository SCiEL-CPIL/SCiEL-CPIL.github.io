"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4 to maximize sum of radii.
    
    Strategy:
    1. Use a structured initial layout (grid-like with some perturbation).
    2. Optimize positions and radii using differential evolution.
    3. The rectangle dimensions are (w, h) with 2*(w+h) = 4, so w+h = 2.
    4. Each circle i has center (x_i, y_i) and radius r_i.
    5. Constraints:
       - 0 <= x_i - r_i and x_i + r_i <= w
       - 0 <= y_i - r_i and y_i + r_i <= h
       - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j
    6. Maximize sum(r_i).
    
    To handle constraints in differential_evolution, we use a penalty method:
    - Penalty for circles going outside the rectangle
    - Penalty for overlapping circles
    """
    # Rectangle dimensions: w + h = 2, w, h > 0
    # We'll optimize w as well (h = 2 - w)
    # Variables: w, then for each circle: x_i, y_i, r_i
    # Total variables: 1 + 21*3 = 64
    
    def objective(params):
        w = params[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10  # Invalid dimensions
        
        circles = params[1:].reshape(N, 3)
        # circles[:, 0] = x, circles[:, 1] = y, circles[:, 2] = r
        
        # Penalty for going out of bounds
        penalty = 0.0
        for i in range(N):
            x, y, r = circles[i]
            if x - r < 0:
                penalty += (x - r) ** 2
            if x + r > w:
                penalty += (x + r - w) ** 2
            if y - r < 0:
                penalty += (y - r) ** 2
            if y + r > h:
                penalty += (y + r - h) ** 2
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                min_dist = ri + rj
                if dist_sq < min_dist ** 2:
                    penalty += (min_dist - np.sqrt(dist_sq)) ** 2
        
        sum_radii = np.sum(circles[:, 2])
        return penalty - sum_radii  # Minimize this (negative sum of radii + penalties)
    
    # Bounds for each variable
    # w: [0.5, 1.5] (since h = 2-w, both should be reasonable)
    bounds = [(0.5, 1.5)]
    # For each circle: x in [0, 2], y in [0, 2], r in [0.01, 0.5]
    for _ in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    w0, h0 = 1.0, 1.0
    x0 = [w0]
    for i in range(N):
        row = i // side
        col = i % side
        cx = (col + 0.5) * w0 / side
        cy = (row + 0.5) * h0 / side
        r0 = 0.5 * w0 / side - 0.01
        x0.append(cx)
        x0.append(cy)
        x0.append(r0)
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        x0=x0,
        seed=42,
        maxiter=200,
        popsize=15,
        tol=1e-6,
        polish=True,
        workers=1
    )
    
    best_params = result.x
    w = best_params[0]
    h = 2.0 - w
    circles = best_params[1:].reshape(N, 3)
    
    # Project circles back into valid region (clip)
    for i in range(N):
        x, y, r = circles[i]
        # Clip radius to fit in box
        r = min(r, w / 2.0, h / 2.0)
        r = max(r, 0.001)
        circles[i, 2] = r
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        circles[i, 0] = x
        circles[i, 1] = y
    
    # Final overlap resolution: iterative push-apart
    for iteration in range(50):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx ** 2 + dy ** 2)
                min_dist = ri + rj
                if dist < min_dist - 1e-6:
                    if dist < 1e-10:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = min_dist - dist
                    # Push apart equally
                    push_x = dx / dist * overlap * 0.5
                    push_y = dy / dist * overlap * 0.5
                    circles[i, 0] -= push_x
                    circles[i, 1] -= push_y
                    circles[j, 0] += push_x
                    circles[j, 1] += push_y
                    moved = True
        # Re-clip to bounds
        for i in range(N):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, w - r)
            circles[i, 1] = np.clip(circles[i, 1], r, h - r)
        if not moved:
            break
    
    return circles


# EVOLVE-BLOCK-END
