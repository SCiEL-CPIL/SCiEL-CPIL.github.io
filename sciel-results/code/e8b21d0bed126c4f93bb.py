"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21
DEADLINE = 145.0  # safety margin before 150s kill


def _bounding_box(circles):
    """Return (W, H) bounding box of circle centers expanded by radii."""
    if len(circles) == 0:
        return 0.0, 0.0
    x_min = np.min(circles[:, 0] - circles[:, 2])
    x_max = np.max(circles[:, 0] + circles[:, 2])
    y_min = np.min(circles[:, 1] - circles[:, 2])
    y_max = np.max(circles[:, 1] + circles[:, 2])
    return x_max - x_min, y_max - y_min


def _voronoi_inradii(circles, W, H):
    """For each circle, compute the distance to the nearest wall or other circle.
    Returns array of inradii (max possible radius at that center).
    """
    n = len(circles)
    inradii = np.empty(n)
    for i in range(n):
        xi, yi, ri = circles[i]
        # distance to walls
        d_wall = min(xi, W - xi, yi, H - yi)
        # distance to other circles (center-to-center minus their radii)
        for j in range(n):
            if i == j:
                continue
            dx = xi - circles[j, 0]
            dy = yi - circles[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            d_avail = d - circles[j, 2]
            if d_avail < d_wall:
                d_wall = d_avail
        inradii[i] = d_wall
    return inradii


def _objective_and_grad(circles, W, H):
    """Compute sum of radii and gradient w.r.t. positions and radii.
    We maximize sum(r_i) subject to:
      - circles disjoint: dist(i,j) >= r_i + r_j
      - inside box: r_i <= min(x_i, W-x_i, y_i, H-y_i)
    
    We use a penalty-free approach: set r_i = inradius_i, then optimize positions
    to maximize sum(inradii).
    """
    n = len(circles)
    # Compute inradii
    inradii = _voronoi_inradii(circles, W, H)
    # Clip to non-negative
    inradii = np.maximum(inradii, 0.0)
    obj = np.sum(inradii)
    
    # Gradient w.r.t. positions: for each circle, the inradius is determined by
    # the closest constraint. We compute gradient of inradius w.r.t. position.
    grad_pos = np.zeros((n, 2))
    for i in range(n):
        xi, yi, ri = circles[i]
        # Find the active constraint (closest)
        best_d = min(xi, W - xi, yi, H - yi)
        best_dx, best_dy = 0.0, 0.0
        # Check walls
        if xi <= best_d:
            best_dx, best_dy = 1.0, 0.0
        elif W - xi <= best_d:
            best_dx, best_dy = -1.0, 0.0
        elif yi <= best_d:
            best_dx, best_dy = 0.0, 1.0
        elif H - yi <= best_d:
            best_dx, best_dy = 0.0, -1.0
        # Check other circles
        for j in range(n):
            if i == j:
                continue
            dx = xi - circles[j, 0]
            dy = yi - circles[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                continue
            d_avail = d - circles[j, 2]
            if d_avail < best_d:
                best_d = d_avail
                best_dx = dx / d
                best_dy = dy / d
        grad_pos[i, 0] = best_dx
        grad_pos[i, 1] = best_dy
    
    return obj, grad_pos


def _scale_to_perimeter_4(circles):
    """Scale circles so bounding box has perimeter exactly 4."""
    W, H = _bounding_box(circles)
    perim = 2 * (W + H)
    if perim < 1e-12:
        return circles
    scale = 4.0 / perim
    new_circles = circles.copy()
    new_circles[:, 0] *= scale
    new_circles[:, 1] *= scale
    new_circles[:, 2] *= scale
    return new_circles


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    # Initial placement: grid-like
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.3 / side])
    circles = np.array(cells[:N])
    
    # Optimize positions to maximize sum of inradii in a unit box,
    # then scale to perimeter 4.
    W = 1.0
    H = 1.0
    
    # Use gradient ascent with adaptive step size
    lr = 0.01
    best_obj = -np.inf
    best_circles = circles.copy()
    
    # First, compute initial objective
    obj, _ = _objective_and_grad(circles, W, H)
    best_obj = obj
    best_circles = circles.copy()
    
    steps = 0
    while time.time() < deadline:
        obj, grad = _objective_and_grad(circles, W, H)
        
        # Update positions
        circles[:, 0] += lr * grad[:, 0]
        circles[:, 1] += lr * grad[:, 1]
        
        # Keep circles inside the box (with small margin)
        margin = 0.001
        circles[:, 0] = np.clip(circles[:, 0], margin, W - margin)
        circles[:, 1] = np.clip(circles[:, 1], margin, H - margin)
        
        # Update radii to be the inradii (clipped to non-negative)
        inradii = _voronoi_inradii(circles, W, H)
        circles[:, 2] = np.maximum(inradii, 0.0)
        
        steps += 1
        
        # Adaptive learning rate
        if steps % 50 == 0:
            obj_new, _ = _objective_and_grad(circles, W, H)
            if obj_new > best_obj:
                best_obj = obj_new
                best_circles = circles.copy()
                lr = min(lr * 1.1, 0.05)
            else:
                lr *= 0.8
                # Revert to best
                circles = best_circles.copy()
        
        # Periodically check time
        if steps % 100 == 0 and time.time() > deadline - 5.0:
            break
    
    # Final: set radii to inradii
    inradii = _voronoi_inradii(best_circles, W, H)
    best_circles[:, 2] = np.maximum(inradii, 0.0)
    
    # Scale to perimeter 4
    result = _scale_to_perimeter_4(best_circles)
    
    # Verify and return
    return result
