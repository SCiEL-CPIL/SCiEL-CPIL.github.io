"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np


def _is_valid(x, y, r, tol=1e-9):
    """Check validity: circles inside [0,1]^2 and disjoint interiors."""
    n = len(x)
    for i in range(n):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-10:
                return False
    return True


def _score(x, y, r):
    """Sum of radii."""
    return float(np.sum(r))


def _make_grid_initial(n):
    """Create a grid initial placement."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                positions.append([x, y, r])
    return np.array(positions[:n])


def _make_hex_initial(n):
    """Create a hexagonal packing initial placement."""
    positions = []
    row_height = 1.0 / (n + 2)
    col_width = row_height * 2 / np.sqrt(3)
    y = row_height * 0.5
    row = 0
    while len(positions) < n:
        x_offset = 0.0
        if row % 2 == 1:
            x_offset = col_width * 0.5
        x = x_offset + col_width * 0.5
        col = 0
        while len(positions) < n and x <= 1.0:
            r = row_height * 0.5 - 1e-4
            positions.append([x, y, r])
            x += col_width
            col += 1
        y += row_height
        row += 1
    return np.array(positions[:n])


def _make_random_initial(n, seed):
    """Create a random initial placement."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.01 + rng.random() * 0.05
        positions.append([x, y, r])
    return np.array(positions[:n])


def _objective_and_grad(vars_):
    """
    Objective: negative sum of radii + penalty for violations.
    vars_ layout: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(n):
        # Circle must be inside [0,1]^2
        if x[i] < r[i]:
            penalty += 1000 * (r[i] - x[i]) ** 2
        if x[i] > 1 - r[i]:
            penalty += 1000 * (x[i] - (1 - r[i])) ** 2
        if y[i] < r[i]:
            penalty += 1000 * (r[i] - y[i]) ** 2
        if y[i] > 1 - r[i]:
            penalty += 1000 * (y[i] - (1 - r[i])) ** 2
    
    # Penalty for circle-circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += 1000 * overlap ** 2
    
    obj = -np.sum(r) + penalty
    
    # Gradient
    grad = np.zeros(3 * n)
    
    # Gradient of -sum(r)
    for i in range(n):
        grad[2 * i + 2] = -1.0
    
    # Gradient of boundary penalties
    for i in range(n):
        if x[i] < r[i]:
            grad[i * 3 + 0] += 2000 * (r[i] - x[i]) * (-1)
            grad[i * 3 + 2] += 2000 * (r[i] - x[i]) * (1)
        if x[i] > 1 - r[i]:
            grad[i * 3 + 0] += 2000 * (x[i] - (1 - r[i])) * (1)
            grad[i * 3 + 2] += 2000 * (x[i] - (1 - r[i])) * (-1)
        if y[i] < r[i]:
            grad[i * 3 + 1] += 2000 * (r[i] - y[i]) * (-1)
            grad[i * 3 + 2] += 2000 * (r[i] - y[i]) * (1)
        if y[i] > 1 - r[i]:
            grad[i * 3 + 1] += 2000 * (y[i] - (1 - r[i])) * (1)
            grad[i * 3 + 2] += 2000 * (y[i] - (1 - r[i])) * (-1)
    
    # Gradient of overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap = min_dist - dist
                # d(overlap^2)/d(x_i) = 2*overlap * d(overlap)/d(x_i)
                # d(overlap)/d(x_i) = -dx/dist (since overlap = min_dist - dist, d(dist)/d(x_i) = dx/dist)
                # Actually: d(dist)/d(x_i) = dx/dist, so d(overlap)/d(x_i) = -dx/dist
                # d(overlap)/d(y_i) = -dy/dist
                # d(overlap)/d(r_i) = 1
                # d(overlap)/d(r_j) = 1
                # d(overlap)/d(x_j) = dx/dist (since d(dist)/d(x_j) = -dx/dist)
                # d(overlap)/d(y_j) = dy/dist
                
                factor = 2000 * overlap
                grad[i * 3 + 0] += factor * (-dx / dist)
                grad[i * 3 + 1] += factor * (-dy / dist)
                grad[i * 3 + 2] += factor * 1
                grad[j * 3 + 0] += factor * (dx / dist)
                grad[j * 3 + 1] += factor * (dy / dist)
                grad[j * 3 + 2] += factor * 1
    
    return obj, grad


def _scipy_optimize(positions, maxiter=500, time_limit=30):
    """
    Use scipy.optimize.minimize with L-BFGS-B to optimize the packing.
    Returns optimized positions.
    """
    n = len(positions)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[i * 3 + 0] = positions[i][0]
        x0[i * 3 + 1] = positions[i][1]
        x0[i * 3 + 2] = positions[i][2]
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    start_time = time.time()
    
    result = minimize(
        _objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'gtol': 1e-8}
    )
    
    # Extract result
    opt_vars = result.x
    opt_positions = np.zeros((n, 3))
    for i in range(n):
        opt_positions[i][0] = opt_vars[i * 3 + 0]
        opt_positions[i][1] = opt_vars[i * 3 + 1]
        opt_positions[i][2] = opt_vars[i * 3 + 2]
    
    return opt_positions


def _relax_and_grow(positions, max_iters=200, tol=1e-8):
    """
    Iteratively relax overlaps and grow radii using vectorized operations.
    """
    n = len(positions)
    pos = positions.copy()
    
    for it in range(max_iters):
        # Check for overlaps and resolve
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = pos[i][2] + pos[j][2]
                
                if dist < min_dist - 1e-12:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    pos[i][0] += dx * overlap * 0.5
                    pos[i][1] += dy * overlap * 0.5
                    pos[j][0] -= dx * overlap * 0.5
                    pos[j][1] -= dy * overlap * 0.5
                    moved = True
        
        # Clamp to unit square
        for i in range(n):
            pos[i][0] = np.clip(pos[i][0], pos[i][2], 1 - pos[i][2])
            pos[i][1] = np.clip(pos[i][1], pos[i][2], 1 - pos[i][2])
        
        # Try to grow radii
        max_growth = 0
        for i in range(n):
            max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = (dist - pos[j][2])
                max_r = min(max_r, available)
            max_r = max(max_r, 0)
            if max_r > pos[i][2] + 1e-10:
                growth = (max_r - pos[i][2]) * 0.1
                pos[i][2] += growth
                max_growth = max(max_growth, growth)
        
        # Clamp radii to be valid
        for i in range(n):
            pos[i][2] = min(pos[i][2], pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        
        if max_growth < tol and not moved:
            break
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_make_grid_initial(N))
    inits.append(_make_hex_initial(N))
    
    # Add random inits
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        inits.append(_make_random_initial(N, seed))
    
    best_result = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        # First: quick relax to get a valid starting point
        relaxed = _relax_and_grow(init, max_iters=100)
        
        # Check validity
        if _is_valid(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2]):
            score = _score(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2])
            if score > best_score:
                best_score = score
                best_result = relaxed
        
        # Then: use scipy optimization
        if time.time() > deadline - 20:
            break
        
        opt_result = _scipy_optimize(relaxed, maxiter=300)
        
        # Verify validity
        if _is_valid(opt_result[:, 0], opt_result[:, 1], opt_result[:, 2]):
            opt_score = _score(opt_result[:, 0], opt_result[:, 1], opt_result[:, 2])
            if opt_score > best_score:
                best_score = opt_score
                best_result = opt_result
        else:
            # If scipy result is invalid, try to repair it
            repaired = _relax_and_grow(opt_result, max_iters=100)
            if _is_valid(repaired[:, 0], repaired[:, 1], repaired[:, 2]):
                rep_score = _score(repaired[:, 0], repaired[:, 1], repaired[:, 2])
                if rep_score > best_score:
                    best_score = rep_score
                    best_result = repaired
    
    # Final polish: try to grow radii more aggressively
    if best_result is not None and time.time() < deadline - 5:
        polished = _relax_and_grow(best_result, max_iters=100, tol=1e-10)
        polished_score = _score(polished[:, 0], polished[:, 1], polished[:, 2])
        if _is_valid(polished[:, 0], polished[:, 1], polished[:, 2]) and polished_score > best_score:
            best_result = polished
            best_score = polished_score
    
    if best_result is None:
        # Fallback to grid
        best_result = _make_grid_initial(N)
    
    return best_result
# EVOLVE-BLOCK-END
