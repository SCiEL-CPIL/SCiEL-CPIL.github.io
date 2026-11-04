"""
Strategy: Vectorized Smooth-penalty L-BFGS-B for 21-circle packing.
Replaced slow loops with NumPy vectorization and removed heavy SLSQP polish.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 135.0  # safety margin before 150 s kill

def _check_validity(circles, w, h, tol=1e-6):
    n = len(circles)
    # Boundary check
    if np.any(circles[:, 2] < -tol): return False
    if np.any(circles[:, 0] - circles[:, 2] < -tol): return False
    if np.any(circles[:, 0] + circles[:, 2] > w + tol): return False
    if np.any(circles[:, 1] - circles[:, 2] < -tol): return False
    if np.any(circles[:, 1] + circles[:, 2] > h + tol): return False
    # Pairwise check
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True

def _separate_circles(circles, w, h, max_iter=100, tol=1e-8):
    circles = circles.copy()
    n = len(circles)
    for _ in range(max_iter):
        moved = False
        # Boundary projection
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
        # Pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = circles[i, 2] + circles[j, 2]
                if dist < min_d:
                    if dist < 1e-10:
                        ux, uy = 1.0, 0.0
                    else:
                        ux, uy = dx / dist, dy / dist
                    ov = (min_d - dist) / 2.0 + 1e-9
                    circles[i, 0] += ov * ux
                    circles[i, 1] += ov * uy
                    circles[j, 0] -= ov * ux
                    circles[j, 1] -= ov * uy
                    moved = True
        # Re-project boundaries
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
        if not moved:
            break
    if _check_validity(circles, w, h, tol=1e-6):
        return circles, True
    # Scale down radii if still invalid
    for scale in [0.99, 0.98, 0.95, 0.9]:
        c2 = circles.copy()
        c2[:, 2] *= scale
        if _check_validity(c2, w, h, tol=1e-6):
            return c2, True
    return circles, False

def _smooth_penalty_grad(circles, w, h, beta):
    """
    Vectorized smooth penalty and gradient.
    """
    n = len(circles)
    x = circles[:, 0]; y = circles[:, 1]; r = circles[:, 2]
    
    # Create upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[dist < 1e-12] = 1e-12
    
    overlap = r[i_idx] + r[j_idx] - dist
    be = beta * overlap
    
    # Soft max value: (1/beta) * log(1 + exp(be))
    # Stable log-sum-exp: max(0, be) + log(1 + exp(-abs(be)))
    val_pairs = (np.maximum(0, be) + np.log1p(np.exp(-np.abs(be)))) / beta
    penalty_val = np.sum(val_pairs)
    
    # Gradient of soft max w.r.t overlap: sigmoid(be)
    sigmoid_be = 1.0 / (1.0 + np.exp(-be))
    
    ux = dx / dist
    uy = dy / dist
    
    gx = np.zeros(n); gy = np.zeros(n); gr = np.zeros(n)
    
    # d(overlap)/dx_i = -ux, d(overlap)/dx_j = ux
    np.add.at(gx, i_idx, -sigmoid_be * ux)
    np.add.at(gx, j_idx, sigmoid_be * ux)
    np.add.at(gy, i_idx, -sigmoid_be * uy)
    np.add.at(gy, j_idx, sigmoid_be * uy)
    np.add.at(gr, i_idx, sigmoid_be)
    np.add.at(gr, j_idx, sigmoid_be)
    
    # Boundary penalties
    # x - r, w - x - r, y - r, h - y - r
    # We want penalty for negative values.
    # Let z be the margin. Penalty is soft_max(0, -z).
    # z1 = x - r, z2 = w - x - r, z3 = y - r, z4 = h - y - r
    
    # z1 = x - r
    z1 = x - r
    be1 = -beta * z1
    val1 = (np.maximum(0, be1) + np.log1p(np.exp(-np.abs(be1)))) / beta
    penalty_val += np.sum(val1)
    sig1 = 1.0 / (1.0 + np.exp(-be1))
    # d(pen)/dx = sig1 * d(-z1)/dx = sig1 * 1
    # d(pen)/dr = sig1 * d(-z1)/dr = sig1 * 1
    gx += sig1
    gr += sig1
    
    # z2 = w - x - r
    z2 = w - x - r
    be2 = -beta * z2
    val2 = (np.maximum(0, be2) + np.log1p(np.exp(-np.abs(be2)))) / beta
    penalty_val += np.sum(val2)
    sig2 = 1.0 / (1.0 + np.exp(-be2))
    # d(pen)/dx = sig2 * d(-z2)/dx = sig2 * (-1)
    # d(pen)/dr = sig2 * d(-z2)/dr = sig2 * (-1)
    gx -= sig2
    gr -= sig2
    
    # z3 = y - r
    z3 = y - r
    be3 = -beta * z3
    val3 = (np.maximum(0, be3) + np.log1p(np.exp(-np.abs(be3)))) / beta
    penalty_val += np.sum(val3)
    sig3 = 1.0 / (1.0 + np.exp(-be3))
    # d(pen)/dy = sig3 * 1
    # d(pen)/dr = sig3 * 1
    gy += sig3
    gr += sig3
    
    # z4 = h - y - r
    z4 = h - y - r
    be4 = -beta * z4
    val4 = (np.maximum(0, be4) + np.log1p(np.exp(-np.abs(be4)))) / beta
    penalty_val += np.sum(val4)
    sig4 = 1.0 / (1.0 + np.exp(-be4))
    # d(pen)/dy = sig4 * (-1)
    # d(pen)/dr = sig4 * (-1)
    gy -= sig4
    gr -= sig4
    
    grad = np.concatenate([gx, gy, gr])
    return penalty_val, grad

def _objective_and_grad(x_flat, w, h, mu, beta):
    n = len(x_flat) // 3
    circles = x_flat.reshape(n, 3)
    base = -np.sum(circles[:, 2])
    pen, pen_grad = _smooth_penalty_grad(circles, w, h, beta)
    val = base + mu * pen
    grad = np.zeros_like(x_flat)
    grad[:n] = pen_grad[:n] * mu
    grad[n:2*n] = pen_grad[n:2*n] * mu
    grad[2*n:] = -1.0 + pen_grad[2*n:] * mu
    return val, grad

def _local_optimize(circles, w, h, time_limit):
    n = len(circles)
    x0 = circles.flatten()
    bounds = []
    for i in range(n):
        bounds.append((1e-8, w - 1e-8))
        bounds.append((1e-8, h - 1e-8))
        bounds.append((1e-8, 1.0))
    
    mu = 50.0
    beta = 50.0
    start = time.time()
    
    def obj(x):
        return _objective_and_grad(x, w, h, mu, beta)
    
    for it in range(20): # Limit iterations to ensure speed
        if time.time() - start > time_limit:
            break
        res = minimize(obj, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8})
        x0 = res.x
        c = x0.reshape(n, 3)
        pen_val, _ = _smooth_penalty_grad(c, w, h, beta)
        if pen_val < 1e-8:
            break
        if pen_val > 1e-3:
            mu *= 1.5
        beta = max(20.0, beta * 0.95)
        
    return x0.reshape(n, 3)

def _grid_init(w, h, n):
    side = int(np.ceil(np.sqrt(n)))
    circles = np.zeros((n, 3))
    for i in range(n):
        row, col = i // side, i % side
        circles[i, 0] = (col + 0.5) * w / side
        circles[i, 1] = (row + 0.5) * h / side
        # Small radius to start
        circles[i, 2] = 0.5 * min(w, h) / side * 0.3
    return circles

def _random_init(w, h, n, seed):
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = rng.uniform(0.05, w - 0.05)
        circles[i, 1] = rng.uniform(0.05, h - 0.05)
        circles[i, 2] = rng.uniform(0.02, 0.05)
    return circles

def _best_known():
    w, h = 1.0, 1.0
    circles = np.zeros((N, 3))
    side = 5
    r = 0.09
    for i in range(N):
        row, col = i // side, i % side
        circles[i, 0] = (col + 0.5) / side
        circles[i, 1] = (row + 0.5) / side
        circles[i, 2] = r
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_value = -np.inf
    best_result = None
    
    # Try a few aspect ratios
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Try grid init
        circles = _grid_init(w, h, N)
        circles_sep, valid = _separate_circles(circles, w, h, max_iter=50)
        if valid:
            time_left = deadline - 5.0 - time.time()
            if time_left > 1.0:
                circles_opt = _local_optimize(circles_sep, w, h, time_limit=min(time_left, 5.0))
                circles_final, valid_final = _separate_circles(circles_opt, w, h, max_iter=50)
                if valid_final:
                    value = np.sum(circles_final[:, 2])
                    if value > best_value:
                        best_value = value
                        best_result = circles_final.copy()
        
        # Try random init if time permits
        if time.time() < deadline - 5.0:
            circles_rand = _random_init(w, h, N, seed=42)
            circles_sep_r, valid_r = _separate_circles(circles_rand, w, h, max_iter=50)
            if valid_r:
                time_left = deadline - 5.0 - time.time()
                if time_left > 1.0:
                    circles_opt_r = _local_optimize(circles_sep_r, w, h, time_limit=min(time_left, 4.0))
                    circles_final_r, valid_final_r = _separate_circles(circles_opt_r, w, h, max_iter=50)
                    if valid_final_r:
                        value = np.sum(circles_final_r[:, 2])
                        if value > best_value:
                            best_value = value
                            best_result = circles_final_r.copy()

    if best_result is None:
        best_result = _best_known()
    
    return best_result
