"""Rewritten program: force-directed circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

def _compute_objective(x, w, h):
    """
    x: array of shape (3N,) = [r_0, r_1, ..., r_{N-1}, cx_0, cy_0, cx_1, cy_1, ...]
    w, h: rectangle dimensions (w + h = 2 for perimeter 4)
    
    Objective: minimize negative sum of radii + penalties
    """
    radii = x[:N]
    centers = x[N:].reshape(N, 2)
    
    # Penalty for negative radii
    neg_r_penalty = np.sum(np.clip(-radii, 0, None) ** 2) * 100.0
    
    # Penalty for circles going outside rectangle
    # Circle i must satisfy: cx_i >= r_i, cx_i + r_i <= w, cy_i >= r_i, cy_i + r_i <= h
    left_viol = np.clip(radii - centers[:, 0], 0, None)
    right_viol = np.clip(radii - (w - centers[:, 0]), 0, None)
    bottom_viol = np.clip(radii - centers[:, 1], 0, None)
    top_viol = np.clip(radii - (h - centers[:, 1]), 0, None)
    
    boundary_penalty = np.sum(left_viol ** 2 + right_viol ** 2 + bottom_viol ** 2 + top_viol ** 2) * 50.0
    
    # Penalty for overlapping circles
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    overlap_penalty *= 100.0
    
    total_penalty = neg_r_penalty + boundary_penalty + overlap_penalty
    sum_radii = np.sum(radii)
    
    return -sum_radii + total_penalty

def _construct_initial(w, h):
    """Create initial layout with circles in a grid pattern."""
    # Use a 5x5 grid, take first 21
    cols = 5
    rows = 5
    positions = []
    for i in range(cols):
        for j in range(rows):
            cx = (i + 0.5) * w / cols
            cy = (j + 0.5) * h / rows
            positions.append((cx, cy))
    
    positions = positions[:N]
    
    # Initial radius: minimum distance to boundary / 2, and to neighbors / 2
    radii = []
    for i, (cx, cy) in enumerate(positions):
        r_boundary = min(cx, w - cx, cy, h - cy) * 0.45
        r_neighbors = r_boundary
        for j, (cx2, cy2) in enumerate(positions):
            if i != j:
                d = np.sqrt((cx - cx2) ** 2 + (cy - cy2) ** 2)
                r_neighbors = min(r_neighbors, d * 0.45)
        radii.append(max(0.01, r_neighbors))
    
    x = np.zeros(3 * N)
    x[:N] = radii
    for i in range(N):
        x[N + 2 * i] = positions[i][0]
        x[N + 2 * i + 1] = positions[i][1]
    
    return x

def _refine(x, w, h, time_limit):
    """Use SLSQP to refine the packing."""
    start_time = time.time()
    
    # Bounds: radii >= 0, centers within [0, w] x [0, h]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # radius
    for i in range(N):
        bounds.append((0.0, w))  # cx
        bounds.append((0.0, h))  # cy
    
    result = minimize(
        _compute_objective,
        x,
        args=(w, h),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-10}
    )
    
    return result.x

def construct():
    start_time = time.time()
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(w + h) = 4, so w + h = 2
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 2.0]
    
    # Also try some non-integer ratios
    extra_ratios = [0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.3, 1.6, 1.8]
    all_ratios = aspect_ratios + extra_ratios
    
    # Shuffle to diversify
    np.random.seed(42)
    np.random.shuffle(all_ratios)
    
    for aspect in all_ratios:
        if time.time() - start_time > DEADLINE:
            break
        
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() - start_time > DEADLINE:
                break
            
            # Create initial layout
            x0 = _construct_initial(w, h)
            
            # Add some random perturbation for diversity
            if trial > 0:
                x0[N:] += np.random.uniform(-0.05, 0.05, (N, 2))
                x0[N:] = np.clip(x0[N:], 0.01, [w - 0.01, h - 0.01])
                x0[:N] *= np.random.uniform(0.8, 1.2, N)
            
            # Refine
            x_opt = _refine(x0, w, h, DEADLINE)
            
            # Extract results
            radii = x_opt[:N]
            centers = x_opt[N:].reshape(N, 2)
            
            # Check validity
            valid = True
            if np.any(radii < -1e-9):
                valid = False
            if np.any(centers[:, 0] < -1e-9) or np.any(centers[:, 0] > w + 1e-9):
                valid = False
            if np.any(centers[:, 1] < -1e-9) or np.any(centers[:, 1] > h + 1e-9):
                valid = False
            
            # Check boundary violations
            for i in range(N):
                if centers[i, 0] - radii[i] < -1e-9:
                    valid = False
                if centers[i, 0] + radii[i] - w < 1e-9:
                    valid = False
                if centers[i, 1] - radii[i] < -1e-9:
                    valid = False
                if centers[i, 1] + radii[i] - h < 1e-9:
                    valid = False
            
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
            
            if not valid:
                # Try to repair by scaling down
                # Find max violation
                max_viol = 0
                for i in range(N):
                    max_viol = max(max_viol, max(0, radii[i] - centers[i, 0]))
                    max_viol = max(max_viol, max(0, radii[i] - (w - centers[i, 0])))
                    max_viol = max(max_viol, max(0, radii[i] - centers[i, 1]))
                    max_viol = max(max_viol, max(0, radii[i] - (h - centers[i, 1])))
                
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                        max_viol = max(max_viol, max(0, radii[i] + radii[j] - d))
                
                if max_viol > 0:
                    scale = 1.0 / (1.0 + max_viol * 2)
                    radii *= scale
                    # Re-check
                    valid = True
                    for i in range(N):
                        if centers[i, 0] - radii[i] < -1e-9:
                            valid = False
                        if centers[i, 0] + radii[i] - w < 1e-9:
                            valid = False
                        if centers[i, 1] - radii[i] < -1e-9:
                            valid = False
                        if centers[i, 1] + radii[i] - h < 1e-9:
                            valid = False
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                            if d < radii[i] + radii[j] - 1e-9:
                                valid = False
            
            if not valid:
                continue
            
            value = np.sum(radii)
            
            if value > best_value:
                best_value = value
                best_result = (radii.copy(), centers.copy(), w, h)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    # If we didn't find a valid solution, use a fallback
    if best_result is None:
        # Fallback: simple grid in square
        w = h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append((r, cx, cy))
        return np.array(cells[:N])
    
    radii, centers, w, h = best_result
    
    # Construct output: (cx, cy, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = centers[i, 0]
        result[i, 1] = centers[i, 1]
        result[i, 2] = radii[i]
    
    return result
