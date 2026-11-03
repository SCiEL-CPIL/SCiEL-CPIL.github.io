"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # seconds, with 5s safety margin

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

_START_TIME = time.time()

def _validity_check(circles, tol=1e-8):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _objective(params):
    """
    params: array of shape (3*N,) = [x1, y1, r1, x2, y2, r2, ...]
    Returns negative sum of radii + penalty for constraint violations.
    """
    circles = params.reshape(N, 3)
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    
    sum_r = np.sum(rs)
    
    # Penalty for boundary violations
    penalty = 0.0
    # x - r >= 0 => violation if x < r
    penalty += np.sum(np.maximum(0, rs - xs)) ** 2
    # x + r <= 1 => violation if x + r > 1
    penalty += np.sum(np.maximum(0, xs + rs - 1)) ** 2
    # y - r >= 0
    penalty += np.sum(np.maximum(0, rs - ys)) ** 2
    # y + r <= 1
    penalty += np.sum(np.maximum(0, ys + rs - 1)) ** 2
    
    # Penalty for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Penalize negative radii
    penalty += np.sum(np.maximum(0, -rs)) ** 2
    
    return -sum_r + 10.0 * penalty

def _initial_hex_layout():
    """Create a hexagonal-inspired initial layout."""
    positions = []
    # Try to place circles in a hexagonal pattern
    # For 26 circles, use a pattern like 5-6-5-6-4 or similar
    rows = [5, 6, 5, 6, 4]
    y_coords = []
    total_rows = len(rows)
    for i, count in enumerate(rows):
        y = (i + 0.5) / total_rows
        y_coords.append(y)
    
    idx = 0
    for i, count in enumerate(rows):
        y = y_coords[i]
        x_offset = 0.5 / (count + 1) if i % 2 == 0 else 1.0 / (count + 1)
        for j in range(count):
            x = x_offset + j * (1.0 - 2*x_offset) / max(1, count - 1) if count > 1 else 0.5
            if count == 1:
                x = 0.5
            positions.append((x, y))
            idx += 1
            if idx >= N:
                break
        if idx >= N:
            break
    
    # Pad if needed
    while len(positions) < N:
        positions.append((0.5, 0.5))
    
    positions = positions[:N]
    
    # Compute initial radius
    min_dist = 1.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
            min_dist = min(min_dist, d)
    for x, y in positions:
        min_dist = min(min_dist, x, 1-x, y, 1-y)
    
    base_r = min_dist / 2 * 0.9
    
    circles = np.zeros((N, 3))
    for i, (x, y) in enumerate(positions):
        circles[i] = [x, y, base_r]
    
    return circles

def _optimize(params0, max_iter=500):
    """Optimize from initial params."""
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0.001, 0.5))  # r
    
    result = minimize(
        _objective,
        params0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 20}
    )
    return result.x.reshape(N, 3)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_circles = None
    best_score = -np.inf
    
    # Try hexagonal layout first
    start = _initial_hex_layout()
    optimized = _optimize(start.flatten(), max_iter=300)
    if _validity_check(optimized):
        score = np.sum(optimized[:, 2])
        best_score = score
        best_circles = optimized.copy()
    
    # Try grid layout with perturbations
    for trial in range(20):
        if _time_left() < 10:
            break
        
        # Random perturbation of hex layout
        start = _initial_hex_layout()
        for i in range(N):
            start[i, 0] += np.random.uniform(-0.01, 0.01)
            start[i, 1] += np.random.uniform(-0.01, 0.01)
            start[i, 0] = np.clip(start[i, 0], 0.01, 0.99)
            start[i, 1] = np.clip(start[i, 1], 0.01, 0.99)
        
        optimized = _optimize(start.flatten(), max_iter=200)
        if _validity_check(optimized):
            score = np.sum(optimized[:, 2])
            if score > best_score:
                best_score = score
                best_circles = optimized.copy()
    
    # Try fully random initializations
    for trial in range(30):
        if _time_left() < 5:
            break
        
        circles = np.zeros((N, 3))
        for i in range(N):
            circles[i, 0] = np.random.uniform(0.05, 0.95)
            circles[i, 1] = np.random.uniform(0.05, 0.95)
            circles[i, 2] = np.random.uniform(0.01, 0.06)
        
        optimized = _optimize(circles.flatten(), max_iter=150)
        if _validity_check(optimized):
            score = np.sum(optimized[:, 2])
            if score > best_score:
                best_score = score
                best_circles = optimized.copy()
    
    # Fallback
    if best_circles is None or not _validity_check(best_circles):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
