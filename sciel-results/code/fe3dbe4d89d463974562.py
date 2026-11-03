"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            if stagger and j % 2 == 1:
                x = (i + 0.5) * w / nc + 0.3 * w / nc
            else:
                x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            positions.append([x, y])
    # Fill remaining spots
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.5
        y = h / 2 + (idx - N / 2) * h / N * 0.5
        positions.append([x, y])
    return np.array(positions[:N])

def _local_optimize(w, h, x0, deadline):
    """Local optimization using L-BFGS-B with penalty method."""
    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        rad = params[2 * N:]
        
        # Compute penalties vectorized
        # Boundary penalties
        pen_x = np.maximum(0, rad - pos[:, 0]) + np.maximum(0, pos[:, 0] - (w - rad))
        pen_y = np.maximum(0, rad - pos[:, 1]) + np.maximum(0, pos[:, 1] - (h - rad))
        boundary_pen = np.sum(pen_x) + np.sum(pen_y)
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)
        
        return -np.sum(rad) + 100.0 * (boundary_pen + overlap_pen)
    
    # Bounds: positions in [0, w] x [0, h], radii in [0, min(w,h)/2]
    max_r = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
    for i in range(N):
        bounds.append((1e-4, max_r))  # radius
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6,
                                  'maxls': 50, 'maxcor': 10})
        return result
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try multiple aspect ratios
    aspect_ratios = np.linspace(0.5, 2.0, 20)
    aspect_ratios = np.unique(np.concatenate([aspect_ratios, [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667]]))
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        # w + h = 2, w/h = ar => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try different grid layouts with staggering
        layouts = [
            (5, 4, True),   # 5x4 staggered
            (4, 5, True),   # 4x5 staggered
            (3, 7, False),  # 3x7
            (7, 3, False),  # 7x3
            (2, 11, False), # 2x11
            (11, 2, False), # 11x2
            (5, 4, False),  # 5x4 non-staggered
            (4, 5, False),  # 4x5 non-staggered
        ]
        
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 5.0:
                break
            
            positions = _grid_init(w, h, nc, nr, stagger)
            r0 = min(w, h) / max(nc, nr) * 0.35
            radii = np.full(N, r0)
            
            x0 = np.concatenate([positions.flatten(), radii])
            
            result = _local_optimize(w, h, x0, deadline)
            if result is not None and result.fun < 1e6:
                pos = result.x[:2 * N].reshape(N, 2)
                rad = result.x[2 * N:]
                
                circles = np.column_stack([pos, rad])
                if _validate(circles, w, h):
                    value = _total_radius(circles)
                    if value > best_value:
                        best_value = value
                        best_result = circles
    
    # If no valid result found, return a safe default
    if best_result is None:
        # Fallback: simple grid in a square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
