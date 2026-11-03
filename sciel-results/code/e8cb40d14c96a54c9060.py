import numpy as np
import time

N = 21
DEADLINE = time.time() + 140  # 10s safety margin

def _hexagonal_init(w, h, n):
    """Initialize n circles in a hexagonal pattern within w x h rectangle."""
    # Try to fit hexagonal packing
    # For hexagonal packing, rows alternate
    # Estimate rows and cols
    best = None
    best_score = -1
    
    for rows in range(1, n + 1):
        cols = int(np.ceil(n / rows))
        # Hexagonal spacing
        dx = w / (cols + 0.5)  # horizontal spacing
        dy = dx * np.sqrt(3) / 2  # vertical spacing for hex
        
        # Check if it fits
        total_h = (rows - 1) * dy + 2 * dx  # approximate
        if total_h > h * 1.5:
            continue
        
        # Generate positions
        positions = []
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= n:
                    break
                x = (c + 0.5 + (r % 0.5)) * dx
                y = (r + 0.5) * dy + dx
                positions.append((x, y))
        
        if len(positions) >= n:
            positions = positions[:n]
            # Check if all fit
            valid = all(0 < x < w and 0 < y < h for x, y in positions)
            if valid:
                # Score: prefer more uniform spacing
                dists = []
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        d = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                        dists.append(d)
                if dists:
                    score = np.min(dists) - np.var(dists) * 0.1
                    if score > best_score:
                        best_score = score
                        best = np.array(positions)
    
    if best is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n)))
        best = np.array([[(i + 0.5) * w / side, (j + 0.5) * h / side] 
                        for i in range(side) for j in range(side)][:n])
    
    return best

def _compute_radii(positions, w, h):
    """Compute max radius for each circle given positions and rectangle."""
    n = len(positions)
    radii = np.zeros(n)
    
    for i in range(n):
        # Distance to walls
        r_wall = min(positions[i][0], w - positions[i][0], 
                     positions[i][1], h - positions[i][1])
        r_i = r_wall
        
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                       (positions[i][1] - positions[j][1])**2)
            r_i = min(r_i, d / 2)
        
        radii[i] = max(0, r_i - 1e-10)
    
    return radii

def _objective_and_grad(positions, w, h):
    """Compute objective (sum of radii) and gradient."""
    n = len(positions)
    radii = _compute_radii(positions, w, h)
    obj = np.sum(radii)
    
    # Numerical gradient (approximate)
    grad = np.zeros((n, 2))
    eps = 1e-5
    for i in range(n):
        for d in range(2):
            pos_plus = positions.copy()
            pos_plus[i, d] += eps
            r_plus = _compute_radii(pos_plus, w, h)
            
            pos_minus = positions.copy()
            pos_minus[i, d] -= eps
            r_minus = _compute_radii(pos_minus, w, h)
            
            grad[i, d] = (np.sum(r_plus) - np.sum(r_minus)) / (2 * eps)
    
    return obj, radii, grad

def _optimize(positions, w, h, max_iter=5000, lr_init=0.01):
    """Gradient descent optimization."""
    n = len(positions)
    positions = positions.copy()
    best_obj = -np.inf
    best_positions = positions.copy()
    best_w, best_h = w, h
    
    lr = lr_init
    momentum = np.zeros((n, 2))
    beta1, beta2 = 0.9, 0.999
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        obj, radii, grad = _objective_and_grad(positions, w, h)
        
        # Add variance penalty to encourage equal radii
        var_penalty = 0.1 * np.var(radii)
        obj_penalized = obj - var_penalty
        
        # Gradient of variance penalty
        grad_var = np.zeros((n, 2))
        eps = 1e-5
        for i in range(n):
            for d in range(2):
                pos_p = positions.copy()
                pos_p[i, d] += eps
                r_p = _compute_radii(pos_p, w, h)
                pos_m = positions.copy()
                pos_m[i, d] -= eps
                r_m = _compute_radii(pos_m, w, h)
                grad_var[i, d] = 0.1 * (np.var(r_p) - np.var(r_m)) / (2 * eps)
        
        total_grad = grad - grad_var
        
        # Adam update
        momentum = beta1 * momentum + (1 - beta1) * total_grad
        positions = positions + lr * momentum
        
        # Project back into rectangle
        radii_est = _compute_radii(positions, w, h)
        for i in range(n):
            positions[i, 0] = np.clip(positions[i, 0], radii_est[i], w - radii_est[i])
            positions[i, 1] = np.clip(positions[i, 1], radii_est[i], h - radii_est[i])
        
        if obj > best_obj:
            best_obj = obj
            best_positions = positions.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_positions, best_obj

def _try_aspect_ratio(w, h, max_iter=3000):
    """Try optimization for given aspect ratio."""
    if time.time() > DEADLINE:
        return None, -np.inf
    
    positions = _hexagonal_init(w, h, N)
    if positions is None:
        return None, -np.inf
    
    positions, obj = _optimize(positions, w, h, max_iter=max_iter)
    
    if time.time() > DEADLINE:
        return None, -np.inf
    
    # Validate
    radii = _compute_radii(positions, w, h)
    valid = True
    for i in range(N):
        if positions[i, 0] - radii[i] < -1e-6 or positions[i, 0] + radii[i] > w + 1e-6:
            valid = False
            break
        if positions[i, 1] - radii[i] < -1e-6 or positions[i, 1] + radii[i] > h + 1e-6:
            valid = False
            break
    
    if not valid:
        return None, -np.inf
    
    return positions, obj

def construct():
    best_positions = None
    best_obj = -np.inf
    best_w, best_h = 1.0, 1.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > DEADLINE - 5:
            break
        
        # w + h = 2 (perimeter = 4)
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        positions, obj = _try_aspect_ratio(w, h, max_iter=2000)
        
        if positions is not None and obj > best_obj:
            best_obj = obj
            best_positions = positions
            best_w, best_h = w, h
    
    if best_positions is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        best_positions = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                                  for i in range(side) for j in range(side)][:N])
        best_w, best_h = 1.0, 1.0
        best_obj = np.sum(_compute_radii(best_positions, best_w, best_h))
    
    # Final polish: high-precision optimization
    if time.time() < DEADLINE - 10:
        positions, obj = _optimize(best_positions, best_w, best_h, max_iter=5000, lr_init=0.005)
        if obj > best_obj:
            best_positions = positions
            best_obj = obj
    
    # Build result
    radii = _compute_radii(best_positions, best_w, best_h)
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_positions[i, 0], best_positions[i, 1], radii[i]]
    
    return result
