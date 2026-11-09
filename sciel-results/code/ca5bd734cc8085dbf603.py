"""Circle packing in perimeter-4 rectangle: Combined Grid, Force-Relaxation, and SLSQP."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_result = None
    best_score = -1.0

    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        if len(positions) != N:
            return
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            return
        if np.any(radii < -1e-9):
            return
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()

    def compute_grid_positions(rows, cols, w, h):
        """Generate grid positions for N circles."""
        if rows * cols < N:
            return None
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            if row == rows - 1:
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            positions[i] = [x, y]
        return positions

    def force_relaxation(positions, radii, w, h, steps=100, lr_pos=0.05, lr_rad=0.005):
        """
        Simple force-directed relaxation to maximize sum of radii.
        1. Try to grow radii.
        2. Push apart overlapping circles.
        3. Push circles inside boundaries.
        """
        pos = positions.copy()
        rad = radii.copy()
        
        # Ensure initial validity (shrink if needed)
        for _ in range(10):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.9
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.9
                        rad[j] *= 0.9
                        valid = False
            if valid:
                break

        for step in range(steps):
            # 1. Attempt to grow radii
            rad += lr_rad
            
            # 2. Resolve Boundary Violations
            for i in range(N):
                # Left
                if pos[i, 0] - rad[i] < 0:
                    pos[i, 0] = rad[i]
                # Right
                if pos[i, 0] + rad[i] > w:
                    pos[i, 0] = w - rad[i]
                # Bottom
                if pos[i, 1] - rad[i] < 0:
                    pos[i, 1] = rad[i]
                # Top
                if pos[i, 1] + rad[i] > h:
                    pos[i, 1] = h - rad[i]
                
                # If still invalid (e.g., w < 2*rad), shrink
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95

            # 3. Resolve Overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]
                    
                    if dist < min_dist:
                        if dist < 1e-6:
                            # Random nudge
                            dx = np.random.uniform(-1, 1)
                            dy = np.random.uniform(-1, 1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Overlap amount
                        overlap = min_dist - dist
                        # Push apart along normal
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Move each half of the overlap
                        push = overlap * 0.5 * lr_pos * 10 # Scale force
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push
                        
                        # If still overlapping significantly, shrink radii
                        if overlap > min_dist * 0.1:
                            rad[i] *= 0.98
                            rad[j] *= 0.98

        # Final cleanup: ensure strict validity
        for _ in range(5):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        valid = False
            if valid:
                break

        return pos, rad

    def run_slsqp(w_init, x0, y0, r0):
        """Polish with SLSQP."""
        h = 2.0 - w_init
        if h <= 0.1 or w_init <= 0.1:
            return

        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            h = 2.0 - w
            if h <= 0.1 or w <= 0.1:
                return 1e10
            
            penalty = 0.0
            # Boundary
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    penalty += 100 * (xs[i] - rs[i])**2
                if xs[i] + rs[i] > w:
                    penalty += 100 * (xs[i] + rs[i] - w)**2
                if ys[i] - rs[i] < 0:
                    penalty += 100 * (ys[i] - rs[i])**2
                if ys[i] + rs[i] > h:
                    penalty += 100 * (ys[i] + rs[i] - h)**2
            
            # Overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 100 * (min_d - dist)**2
            
            return -np.sum(rs) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((0.01, 1.0))
        bounds.append((0.5, 1.5))

        try:
            res = minimize(objective, np.concatenate(([x0, y0, r0, w_init])), method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            return res
        except Exception:
            return None

    # Grid configurations to try
    grid_configs = [
        (3, 7),  # 21 circles
        (4, 6),  # 24 cells, use 21
        (5, 5),  # 25 cells, use 21
        (2, 11), # 22 cells, use 21
        (7, 3),
        (6, 4),
        (5, 5),
        (1, 21),
        (21, 1),
        (3, 8),
        (4, 7),
        (2, 12),
    ]

    # Aspect ratios to try
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 1.33, 0.75, 1.25]

    # 1. Generate random initial configuration
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # 1. Generate random initial configuration
        # Aspect ratio w/h
        # w + h = 2 (we can fix this equality in the optimizer or just penalize)
        # Let's sample w uniformly from [0.5, 1.5] so h = 2 - w is in [0.5, 1.5]
        w = random.uniform(0.5, 1.5)
        h = 2.0 - w
        
        # Initialize circles
        # x, y, r
        x = np.random.uniform(0.1, w - 0.1, N)
        y = np.random.uniform(0.1, h - 0.1, N)
        r = np.random.uniform(0.05, 0.2, N)
        
        # Ensure initial radii don't violate boundaries too badly
        r = np.minimum(r, np.minimum(x, w - x))
        r = np.minimum(r, np.minimum(y, h - y))
        r = np.maximum(r, 0.01)
        
        pos = np.column_stack((x, y))
        rad = r
        
        # Force relaxation
        pos, rad = force_relaxation(pos, rad, w, h, steps=50)
        
        # Validate
        validate_and_update(pos, rad, w, h)
        
        # SLSQP polish
        res = run_slsqp(w, pos[:, 0], pos[:, 1], rad)
        if res is not None:
            # Extract best result
            res_x = res.x[0:3*N:3]
            res_y = res.x[1:3*N:3]
            res_r = res.x[2:3*N:3]
            res_w = res.x[3*N]
            res_h = 2.0 - res_w
            if res_w > 0.1 and res_h > 0.1:
                validate_and_update(np.column_stack((res_x, res_y)), res_r, res_w, res_h)

    # 2. Try grid configurations with various aspect ratios
    for w_ratio in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * w_ratio / (1.0 + w_ratio)
        h = 2.0 - w
        
        for rows, cols in grid_configs:
            if rows * cols < N:
                continue
            
            pos = compute_grid_positions(rows, cols, w, h)
            if pos is None:
                continue
            
            # Initial radii: approximate fit
            # dx = w / cols, dy = h / rows
            # r = min(dx, dy) / 2
            r = np.full(N, min(w / cols, h / rows) / 2.0)
            
            # Force relaxation
            pos, rad = force_relaxation(pos, r, w, h, steps=100)
            
            # Validate
            validate_and_update(pos, rad, w, h)
            
            # SLSQP polish
            res = run_slsqp(w, pos[:, 0], pos[:, 1], rad)
            if res is not None:
                # Extract best result
                res_x = res.x[0:3*N:3]
                res_y = res.x[1:3*N:3]
                res_r = res.x[2:3*N:3]
                res_w = res.x[3*N]
                res_h = 2.0 - res_w
                if res_w > 0.1 and res_h > 0.1:
                    validate_and_update(np.column_stack((res_x, res_y)), res_r, res_w, res_h)

    # If no valid solution found, return a safe fallback (e.g. seed-like grid)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
