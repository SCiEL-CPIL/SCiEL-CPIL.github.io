"""Rewritten program: circle packing in perimeter-4 rectangle via gradient-based optimization."""
import numpy as np
import time

N = 21
DEADLINE = 150.0  # seconds
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4.
    Maximize sum of radii.
    
    Returns array of shape (21, 3): [x_center, y_center, radius]
    """
    start = time.time()
    deadline = start + 145.0  # 5s safety margin
    
    # Perimeter constraint: 2*(W + H) <= 4 => W + H <= 2
    # We'll optimize over positions, radii, and aspect ratio (W, H with W+H=2)
    
    # Strategy: 
    # 1. Initialize with a grid-like packing
    # 2. Use gradient ascent on sum of radii with collision constraints
    # 3. The objective is to maximize sum(r_i) subject to:
    #    - Circles don't overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #    - Circles inside rectangle: r_i <= x_i <= W - r_i, r_i <= y_i <= H - r_i
    #    - W + H = 2 (or <= 2)
    
    # We'll use a smooth penalty method for the constraints and gradient ascent on the objective.
    
    # Initialize: place circles in a grid
    # Try different aspect ratios and find good initial positions
    best_result = None
    best_score = -np.inf
    
    # Try several aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W/H = ar, W + H = 2
        # H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        # Initialize circles in a grid
        # Estimate how many fit in each dimension
        # For 21 circles, try arrangements like 3x7, 4x6, 5x5, etc.
        arrangements = []
        for rows in range(1, 8):
            cols = int(np.ceil(21.0 / rows))
            if rows * cols >= 21:
                arrangements.append((rows, cols))
        
        for (rows, cols) in arrangements[:4]:  # Limit to a few
            if time.time() > deadline:
                break
            
            # Initialize positions
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                # Center of each cell
                xs[i] = (col + 0.5) * W / cols
                ys[i] = (row + 0.5) * H / rows
                # Initial radius: half the smaller cell dimension
                cell_w = W / cols
                cell_h = H / rows
                rs[i] = 0.4 * min(cell_w, cell_h)
            
            # Optimize using gradient ascent with penalty for violations
            # Objective: maximize sum(rs)
            # Constraints: no overlap, inside rectangle
            
            # We'll do iterative optimization
            lr = 0.01
            penalty_weight = 10.0
            n_iter = 2000
            
            for it in range(n_iter):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dx = np.zeros(N)
                dy = np.zeros(N)
                dr = np.zeros(N)
                
                # Gradient of sum(rs) is 1 for each r_i
                dr[:] = 1.0
                
                # Penalty for overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx_ij = xs[i] - xs[j]
                        dy_ij = ys[i] - ys[j]
                        dist = np.sqrt(dx_ij**2 + dy_ij**2 + 1e-12)
                        min_dist = rs[i] + rs[j]
                        
                        if dist < min_dist:
                            # Overlap: push apart
                            # Gradient of penalty: penalty_weight * (min_dist - dist) * direction
                            # The constraint violation is max(0, min_dist - dist)
                            # d/dx_i of (min_dist - dist)^2 = -2*(min_dist - dist) * (-dx_ij/dist)
                            # = 2*(min_dist - dist) * dx_ij/dist
                            grad_factor = 2.0 * penalty_weight * (min_dist - dist) / dist
                            dx[i] += grad_factor * dx_ij
                            dx[j] -= grad_factor * dx_ij
                            dy[i] += grad_factor * dy_ij
                            dy[j] -= grad_factor * dy_ij
                            # For radii: increasing r_i increases min_dist, which increases penalty
                            # d/dr_i of (min_dist - dist)^2 = 2*(min_dist - dist) * 1
                            dr[i] -= 2.0 * penalty_weight * (min_dist - dist)
                            dr[j] -= 2.0 * penalty_weight * (min_dist - dist)
                
                # Penalty for being outside rectangle
                for i in range(N):
                    # Left wall
                    if xs[i] < rs[i]:
                        viol = rs[i] - xs[i]
                        dx[i] += 2.0 * penalty_weight * viol
                        dr[i] -= 2.0 * penalty_weight * viol
                    # Right wall
                    if xs[i] > W - rs[i]:
                        viol = xs[i] - (W - rs[i])
                        dx[i] -= 2.0 * penalty_weight * viol
                        dr[i] -= 2.0 * penalty_weight * viol
                    # Bottom wall
                    if ys[i] < rs[i]:
                        viol = rs[i] - ys[i]
                        dy[i] += 2.0 * penalty_weight * viol
                        dr[i] -= 2.0 * penalty_weight * viol
                    # Top wall
                    if ys[i] > H - rs[i]:
                        viol = ys[i] - (H - rs[i])
                        dy[i] -= 2.0 * penalty_weight * viol
                        dr[i] -= 2.0 * penalty_weight * viol
                
                # Update
                xs += lr * dx
                ys += lr * dy
                rs += lr * dr
                
                # Clamp radii to positive
                rs = np.maximum(rs, 0.01)
                
                # Clamp positions to be inside (approximately)
                xs = np.clip(xs, rs, W - rs)
                ys = np.clip(ys, rs, H - rs)
            
            # Check validity
            valid = True
            for i in range(N):
                if rs[i] < 0.01 or xs[i] < rs[i] - 1e-6 or xs[i] > W - rs[i] + 1e-6:
                    valid = False
                    break
                if ys[i] < rs[i] - 1e-6 or ys[i] > H - rs[i] + 1e-6:
                    valid = False
                    break
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([xs, ys, rs]).copy()
            
            # Also try without strict validity (allow small violations)
            score = np.sum(rs)
            if score > best_score:
                # Check if violations are small
                max_viol = 0
                for i in range(N):
                    max_viol = max(max_viol, abs(rs[i] - xs[i]), abs(W - rs[i] - xs[i]),
                                   abs(rs[i] - ys[i]), abs(H - rs[i] - ys[i]))
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        viol = rs[i] + rs[j] - dist
                        if viol > 0:
                            max_viol = max(max_viol, viol)
                
                if max_viol < 0.01:
                    best_score = score
                    best_result = np.column_stack([xs, ys, rs]).copy()
    
    # Final polish: try to improve the best result
    if best_result is not None:
        xs = best_result[:, 0].copy()
        ys = best_result[:, 1].copy()
        rs = best_result[:, 2].copy()
        
        # Determine W and H from the bounding box
        W = 2.0 * np.max(xs + rs) / (np.max(xs + rs) + np.max(ys + rs)) * (np.max(xs + rs) + np.max(ys + rs))
        # Actually, let's just use the current bounding box
        W = np.max(xs + rs)
        H = np.max(ys + rs)
        
        # Fine-tune with smaller learning rate
        lr = 0.001
        penalty_weight = 50.0
        n_iter = 1000
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            
            dx = np.zeros(N)
            dy = np.zeros(N)
            dr = np.zeros(N)
            dr[:] = 1.0
            
            for i in range(N):
                for j in range(i+1, N):
                    dx_ij = xs[i] - xs[j]
                    dy_ij = ys[i] - ys[j]
                    dist = np.sqrt(dx_ij**2 + dy_ij**2 + 1e-12)
                    min_dist = rs[i] + rs[j]
                    
                    if dist < min_dist:
                        grad_factor = 2.0 * penalty_weight * (min_dist - dist) / dist
                        dx[i] += grad_factor * dx_ij
                        dx[j] -= grad_factor * dx_ij
                        dy[i] += grad_factor * dy_ij
                        dy[j] -= grad_factor * dy_ij
                        dr[i] -= 2.0 * penalty_weight * (min_dist - dist)
                        dr[j] -= 2.0 * penalty_weight * (min_dist - dist)
            
            for i in range(N):
                if xs[i] < rs[i]:
                    viol = rs[i] - xs[i]
                    dx[i] += 2.0 * penalty_weight * viol
                    dr[i] -= 2.0 * penalty_weight * viol
                if xs[i] > W - rs[i]:
                    viol = xs[i] - (W - rs[i])
                    dx[i] -= 2.0 * penalty_weight * viol
                    dr[i] -= 2.0 * penalty_weight * viol
                if ys[i] < rs[i]:
                    viol = rs[i] - ys[i]
                    dy[i] += 2.0 * penalty_weight * viol
                    dr[i] -= 2.0 * penalty_weight * viol
                if ys[i] > H - rs[i]:
                    viol = ys[i] - (H - rs[i])
                    dy[i] -= 2.0 * penalty_weight * viol
                    dr[i] -= 2.0 * penalty_weight * viol
            
            xs += lr * dx
            ys += lr * dy
            rs += lr * dr
            rs = np.maximum(rs, 0.01)
            xs = np.clip(xs, rs, W - rs)
            ys = np.clip(ys, rs, H - rs)
        
        # Check if improved
        score = np.sum(rs)
        # Verify validity
        valid = True
        W_final = np.max(xs + rs)
        H_final = np.max(ys + rs)
        if 2*(W_final + H_final) > 4.0 + 1e-6:
            valid = False
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid and score > best_score:
            best_result = np.column_stack([xs, ys, rs]).copy()
    
    # If we didn't find a valid result, fall back to a simple grid
    if best_result is None:
        # Simple 3x7 grid in a 1x1 square (perimeter 4)
        W = 1.0
        H = 1.0
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        for i in range(N):
            row = i // 7
            col = i % 7
            xs[i] = (col + 0.5) * W / 7
            ys[i] = (row + 0.5) * H / 3
            rs[i] = 0.45 * min(W/7, H/3)
        best_result = np.column_stack([xs, ys, rs])
    
    # Final validation: ensure perimeter <= 4
    W = np.max(best_result[:, 0] + best_result[:, 2])
    H = np.max(best_result[:, 1] + best_result[:, 2])
    if 2*(W + H) > 4.0:
        # Scale down
        scale = 2.0 / (W + H)
        best_result[:, 0] *= scale
        best_result[:, 1] *= scale
        best_result[:, 2] *= scale
    
    return best_result
