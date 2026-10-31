"""
Circle packing in perimeter-4 rectangle using global optimization with smooth penalties.
Strategy:
1. Generate diverse initial configurations (hexagonal, grid, random).
2. Optimize positions and radii using L-BFGS-B with a smooth penalty function
   that penalizes overlaps and boundary violations.
3. Repair any remaining violations via local projection.
4. Polish the solution with tight tolerances.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def is_valid(circles, W, H, tol=1e-6):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if r < -tol:
                return False
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True

    def penalty_function(params, W, H, mu=100.0):
        """
        Objective function: minimize -sum(r) + mu * (sum of penalties).
        Penalties are smooth approximations of violation magnitudes.
        """
        params = np.array(params)
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Objective: maximize sum of radii
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        # We want x >= r, x <= W-r, y >= r, y <= H-r
        # Violation is max(0, r - x), etc.
        # Smooth max: log(1 + exp(k*v)) / k approximates max(0, v) for large k
        # But for stability, we can use simple quadratic penalty on violation
        # Let's use soft-plus for smoothness
        
        k = 10.0
        def soft_max(v):
            return np.log1p(np.exp(k * v)) / k
        
        # Boundary violations
        pen_left = soft_max(r - x)
        pen_right = soft_max(r - (W - x))
        pen_bottom = soft_max(r - y)
        pen_top = soft_max(r - (H - y))
        
        boundary_pen = np.sum(pen_left) + np.sum(pen_right) + np.sum(pen_bottom) + np.sum(pen_top)
        
        # Overlap violations
        # For each pair (i, j), we want dist(i, j) >= r_i + r_j
        # Violation is (r_i + r_j) - dist(i, j)
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                violation = (r[i] + r[j]) - dist
                overlap_pen += soft_max(violation)
        
        # Also penalize negative radii
        neg_r_pen = np.sum(soft_max(-r))
        
        total_pen = boundary_pen + overlap_pen + neg_r_pen
        
        return obj + mu * total_pen

    def generate_hexagonal_layout(W, H):
        """Generate a hexagonal packing layout."""
        points = []
        # Estimate number of rows and cols
        # Hexagonal packing density is ~0.9069
        # Area of rect = W*H. Area of circles ~ N * pi * r^2
        # Rough estimate for r: r ~ sqrt(W*H / (N * pi * 1.1))
        r_est = np.sqrt(W * H / (N * np.pi * 1.2))
        
        # Hex grid spacing
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        
        row = 0
        y = r_est
        while y <= H - r_est and len(points) < N:
            x = r_est + (row % 2) * dx / 2
            while x <= W - r_est and len(points) < N:
                points.append([x, y, r_est])
                x += dx
            y += dy
            row += 1
        
        # If we didn't fill N, pad with small circles in gaps or just resize
        while len(points) < N:
            # Find a random spot or just add to end (will be fixed by optimizer)
            px = random.uniform(r_est, W - r_est)
            py = random.uniform(r_est, H - r_est)
            points.append([px, py, r_est * 0.5])
            
        return np.array(points[:N])

    def generate_grid_layout(W, H):
        """Generate a rectangular grid layout."""
        points = []
        # Try to fit N in a grid
        cols = int(np.sqrt(N * W / H))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        dx = W / cols
        dy = H / rows
        
        r_est = min(dx, dy) / 2 * 0.95
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N: break
                x = (j + 0.5) * dx
                y = (i + 0.5) * dy
                points.append([x, y, r_est])
                count += 1
        
        while len(points) < N:
            px = random.uniform(0, W)
            py = random.uniform(0, H)
            points.append([px, py, r_est * 0.5])
            
        return np.array(points[:N])

    def repair_and_polish(circles, W, H, time_limit):
        """
        Repair invalid packings by shrinking/moving circles, then polish.
        """
        circles = np.array(circles, dtype=float)
        n = len(circles)
        
        # 1. Repair phase: iteratively fix violations
        for iter in range(50):
            if time.time() > time_limit: break
            
            # Check validity
            valid = True
            for i in range(n):
                x, y, r = circles[i]
                if x < r or x > W - r or y < r or y > H - r or r < 0:
                    valid = False
                    # Shrink and move to center-ish
                    r_new = max(0.01, r * 0.9)
                    x_new = np.clip(x, r_new, W - r_new)
                    y_new = np.clip(y, r_new, H - r_new)
                    circles[i] = [x_new, y_new, r_new]
            
            if not valid:
                continue
                
            # Check overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i][2] + circles[j][2]
                    if dist < min_dist:
                        valid = False
                        # Shrink the larger circle or move them apart
                        # Simple heuristic: shrink both slightly
                        circles[i][2] *= 0.95
                        circles[j][2] *= 0.95
                        break
                if not valid:
                    break
            
            if valid:
                break
        
        if not valid:
            return None
            
        # 2. Polish phase: local optimization with tight constraints
        # Use L-BFGS-B with tight bounds and no penalty (or high penalty)
        # We maximize sum(r) subject to constraints.
        # L-BFGS-B doesn't handle non-linear inequality constraints directly well,
        # so we stick to penalty method but with very high mu and tight tolerances.
        
        def polish_obj(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            # Hard bounds check (return large penalty if violated)
            for i in range(N):
                if x[i] < r[i] - 1e-8 or x[i] > W - r[i] + 1e-8:
                    return 1e6
                if y[i] < r[i] - 1e-8 or y[i] > H - r[i] + 1e-8:
                    return 1e6
                if r[i] < 0:
                    return 1e6
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-8:
                        return 1e6
            
            return -np.sum(r)
        
        # Bounds for L-BFGS-B
        # x: [0, W], y: [0, H], r: [0, min(W,H)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, W)) # x
        for i in range(N):
            bounds.append((0, H)) # y
        for i in range(N):
            bounds.append((0, min(W, H) / 2)) # r
            
        x0 = circles.flatten()
        
        try:
            res = minimize(polish_obj, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                new_circles = res.x.reshape(3, N).T
                # Final validation
                if is_valid(new_circles, W, H, tol=1e-7):
                    circles = new_circles
        except Exception:
            pass
            
        return circles

    # Main logic
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        W = W_ratio
        H = 2.0 - W
        if W <= 0.2 or H <= 0.2:
            continue
            
        # Generate multiple initial layouts
        layouts = []
        
        # Hexagonal
        try:
            hex_layout = generate_hexagonal_layout(W, H)
            layouts.append(hex_layout)
        except Exception:
            pass
            
        # Grid
        try:
            grid_layout = generate_grid_layout(W, H)
            layouts.append(grid_layout)
        except Exception:
            pass
            
        # Random
        for _ in range(2):
            x = np.random.uniform(0, W, N)
            y = np.random.uniform(0, H, N)
            r = np.full(N, min(W, H) / (2 * np.sqrt(N)) * 0.8)
            rand_layout = np.column_stack((x, y, r))
            layouts.append(rand_layout)
            
        for layout in layouts:
            if time.time() > deadline - 5:
                break
                
            # Flatten for optimizer
            x0 = layout.flatten()
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0, W))
            for i in range(N):
                bounds.append((0, H))
            for i in range(N):
                bounds.append((0, min(W, H) / 2))
                
            try:
                # Optimization
                res = minimize(penalty_function, x0, args=(W, H, 50.0), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
                
                if res.success or res.fun < 0: # If it improved
                    opt_circles = res.x.reshape(3, N).T
                    
                    # Repair and Polish
                    polished = repair_and_polish(opt_circles, W, H, time_limit=deadline - 2)
                    
                    if polished is not None:
                        if is_valid(polished, W, H, tol=1e-6):
                            total_r = np.sum(polished[:, 2])
                            if total_r > best_sum_r:
                                best_sum_r = total_r
                                best_config = polished
                                best_W = W
                                best_H = H
            except Exception:
                continue

    # Fallback if no valid solution found
    if best_config is None:
        W = 1.0
        H = 1.0
        # Simple grid fallback
        circles = []
        cols = 5
        rows = 5
        dx = W / cols
        dy = H / rows
        r = min(dx, dy) / 2 * 0.9
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= N:
                    break
                x = (j + 0.5) * dx
                y = (i + 0.5) * dy
                circles.append([x, y, r])
        best_config = np.array(circles[:N])
        best_W = W
        best_H = H

    return best_config

# EVOLVE-BLOCK-END
