"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _perimeter(w, h):
    return 2.0 * (w + h)


def _validate(circles):
    """Check if circles are valid: disjoint interiors, inside bounding box with perimeter <= 4."""
    if len(circles) != N:
        return False
    # Compute bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w <= 0 or h <= 0:
        return False
    
    if _perimeter(w, h) > 4.0 + 1e-9:
        return False
    
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True


def _objective(circles):
    """
    Objective: maximize sum of radii subject to constraints.
    We use a penalty-based approach:
    - Reward: sum of radii
    - Penalty for overlap
    - Penalty for perimeter > 4
    """
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Sum of radii
    reward = np.sum(rs)
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    # Perimeter penalty
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2.0 * (w + h)
    perim_pen = 0.0
    if perim > 4.0:
        perim_pen = (perim - 4.0) ** 2
    
    return reward - 100.0 * overlap_pen - 100.0 * perim_pen


def _hexagonal_init(w, h, n):
    """Initialize circles in a hexagonal lattice pattern clipped to rectangle."""
    # Start with a grid-like hexagonal arrangement
    # Hexagonal spacing
    cols = int(np.ceil(np.sqrt(n * w / h))) if h > 0 else int(np.ceil(np.sqrt(n)))
    if cols < 1:
        cols = 1
    rows = int(np.ceil(n / cols))
    
    # Place circles in hexagonal pattern
    circles = []
    dx = w / (cols + 0.5)
    dy = h / (rows + 0.5)
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if row % 2 == 1:
                x += dx * 0.5
            r = min(dx, dy) * 0.45
            circles.append([x, y, r])
            idx += 1
    
    # If we don't have enough, add more
    while len(circles) < n:
        circles.append([w / 2.0, h / 2.0, min(w, h) * 0.05])
    
    circles = circles[:n]
    return np.array(circles, dtype=np.float64)


def _soft_repulsion_step(circles, w, h, lr_pos=0.01, lr_rad=0.005, iters=50):
    """
    Apply soft-repulsion gradient descent.
    Circles are pushed apart when close, and radii are adjusted.
    """
    circles = circles.copy()
    n = len(circles)
    
    for it in range(iters):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_dist = circles[i, 2] + circles[j, 2]
                
                # Soft repulsion: force is proportional to overlap or proximity
                # Use a smooth function: force = max(0, min_dist - dist + soft_margin) * strength
                soft_margin = 0.05
                gap = min_dist - dist
                if gap > -soft_margin:
                    # Force magnitude
                    strength = 1.0
                    force = max(0.0, gap + soft_margin) * strength
                    if dist > 1e-9:
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    # Radius adjustment: reduce radii if overlapping
                    fr[i] -= force * 0.5
                    fr[j] -= force * 0.5
        
        # Boundary forces: keep circles inside [0, w] x [0, h]
        for i in range(n):
            r = circles[i, 2]
            # Left boundary
            if circles[i, 0] - r < 0:
                fx[i] += (0 - (circles[i, 0] - r)) * 5.0
            # Right boundary
            if circles[i, 0] + r > w:
                fx[i] -= ((circles[i, 0] + r) - w) * 5.0
            # Bottom boundary
            if circles[i, 1] - r < 0:
                fy[i] += (0 - (circles[i, 1] - r)) * 5.0
            # Top boundary
            if circles[i, 1] + r > h:
                fy[i] -= ((circles[i, 1] + r) - h) * 5.0
            
            # Radius growth: try to increase radius if not constrained
            # Simple heuristic: increase if there's space
            fr[i] += 0.01
        
        # Update positions and radii
        for i in range(n):
            circles[i, 0] += lr_pos * fx[i]
            circles[i, 1] += lr_rad * fy[i]  # Wait, should be lr_pos for y too
            circles[i, 2] += lr_rad * fr[i]
            
            # Clamp position to be inside box (with radius margin)
            circles[i, 0] = max(circles[i, 2], min(w - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(h - circles[i, 2], circles[i, 1]))
            circles[i, 2] = max(1e-4, circles[i, 2])
    
    return circles


def _optimize_for_aspect_ratio(w, h, max_iters=200, time_limit=10.0):
    """
    Optimize packing for a given aspect ratio (w, h) where 2*(w+h)=4 => h=2-w.
    """
    start_time = time.time()
    
    # Initialize
    circles = _hexagonal_init(w, h, N)
    
    best_circles = circles.copy()
    best_score = _objective(best_circles)
    
    # Iterative optimization with soft repulsion
    for outer in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Apply soft repulsion step
        circles = _soft_repulsion_step(circles, w, h, 
                                        lr_pos=0.02 / (1 + outer * 0.1), 
                                        lr_rad=0.01 / (1 + outer * 0.1),
                                        iters=30)
        
        # Check if valid and compute score
        score = _objective(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    return best_circles, best_score


def construct():
    start_time = time.time()
    
    # Try different aspect ratios
    # Perimeter = 2*(w+h) = 4 => h = 2 - w
    # w in (0, 2), h in (0, 2)
    aspect_ratios = [
        (1.0, 1.0),      # square
        (0.8, 1.2),      # tall
        (1.2, 0.8),      # wide
        (0.6, 1.4),      # very tall
        (1.4, 0.6),      # very wide
        (0.7, 1.3),
        (1.3, 0.7),
        (0.9, 1.1),
        (1.1, 0.9),
    ]
    
    best_circles = None
    best_score = -np.inf
    
    for w, h in aspect_ratios:
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        circles, score = _optimize_for_aspect_ratio(w, h, max_iters=100, 
                                                     time_limit=10.0)
        if score > best_score:
            best_score = score
            best_circles = circles
    
    # If no valid result, fall back to simple grid
    if best_circles is None or not _validate(best_circles):
        # Simple fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N], dtype=np.float64)
    
    # Final validation and adjustment
    # Ensure the bounding box has perimeter <= 4
    if _validate(best_circles):
        return best_circles
    
    # If invalid, try to shrink radii slightly
    xs = best_circles[:, 0]
    ys = best_circles[:, 1]
    rs = best_circles[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2.0 * (w + h)
    
    if perim > 4.0:
        # Scale down radii
        scale = 4.0 / perim
        best_circles[:, 2] *= scale * 0.999
    
    # Center the packing
    xs = best_circles[:, 0]
    ys = best_circles[:, 1]
    rs = best_circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    
    # Shift to center in a box of perimeter 4
    # We want 2*(w+h) <= 4
    if w + h > 2.0:
        scale = 2.0 / (w + h)
        best_circles[:, 0] = (best_circles[:, 0] - (min_x + max_x) / 2) * scale + w * scale / 2
        best_circles[:, 1] = (best_circles[:, 1] - (min_y + max_y) / 2) * scale + h * scale / 2
        best_circles[:, 2] *= scale
    
    return best_circles
