"""Circle packing in perimeter-4 rectangle: force-directed + SLSQP + multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _valid_pack(cx, cy, r, W, H):
    """Check if circles fit in rectangle [0,W]x[0,H] with disjoint interiors."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True

def _force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.002):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 10
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 10
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        cx += dt * fx
        cy += dt * fy
        # Clamp to rectangle
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy

def _slsqp_objective(x, W, H):
    """Objective for SLSQP: minimize -sum(r) with penalty for violations."""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    pen = 0.0
    # Wall violations
    for i in range(N):
        if cx[i] - r[i] < 0:
            pen += (cx[i] - r[i]) ** 2
        if cy[i] - r[i] < 0:
            pen += (cy[i] - r[i]) ** 2
        if cx[i] + r[i] > W:
            pen += (cx[i] + r[i] - W) ** 2
        if cy[i] + r[i] > H:
            pen += (cy[i] + r[i] - H) ** 2
    # Pairwise overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j]:
                pen += (r[i] + r[j] - d) ** 2
    return -np.sum(r) + 10 * pen

def _slsqp_gradient(x, W, H):
    """Gradient for SLSQP objective."""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    grad = np.zeros(3 * N)
    
    # Gradient of -sum(r)
    grad[2*N:3*N] = -1.0
    
    # Wall violation gradients
    for i in range(N):
        if cx[i] - r[i] < 0:
            grad[i] += 20 * (cx[i] - r[i])
            grad[2*N + i] += -20 * (cx[i] - r[i])
        if cy[i] - r[i] < 0:
            grad[N + i] += 20 * (cy[i] - r[i])
            grad[2*N + i] += -20 * (cy[i] - r[i])
        if cx[i] + r[i] > W:
            grad[i] += 20 * (cx[i] + r[i] - W)
            grad[2*N + i] += 20 * (cx[i] + r[i] - W)
        if cy[i] + r[i] > H:
            grad[N + i] += 20 * (cy[i] + r[i] - H)
            grad[2*N + i] += 20 * (cy[i] + r[i] - H)
    
    # Pairwise overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] and d > 1e-12:
                overlap = r[i] + r[j] - d
                # Gradient of (r_i + r_j - d)^2
                # d/dx_i: 2*(r_i+r_j-d)*(-dx/d)
                # d/dx_j: 2*(r_i+r_j-d)*(dx/d)
                # d/dy_i: 2*(r_i+r_j-d)*(-dy/d)
                # d/dy_j: 2*(r_i+r_j-d)*(dy/d)
                # d/dr_i: 2*(r_i+r_j-d)
                # d/dr_j: 2*(r_i+r_j-d)
                factor = 20 * overlap
                grad[i] += factor * (-dx / d)
                grad[N + i] += factor * (-dy / d)
                grad[j] += factor * (dx / d)
                grad[N + j] += factor * (dy / d)
                grad[2*N + i] += factor
                grad[2*N + j] += factor
    
    return grad

def _optimize_for_aspect(W, H, grid_n, deadline):
    """Optimize circle positions and radii for a fixed aspect ratio W:H."""
    # Initial grid layout
    cols = grid_n
    rows = int(np.ceil(N / cols))
    W_cell = W / cols
    H_cell = H / rows
    
    cx = np.array([(i + 0.5) * W_cell for i in range(cols) for _ in range(rows)][:N])
    cy = np.array([(j + 0.5) * H_cell for _ in range(cols) for j in range(rows)][:N])
    r = np.full(N, min(W_cell, H_cell) / 2 * 0.9)
    
    # Force-directed to fix initial overlaps
    cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=300)
    
    # Use SLSQP to maximize sum of radii
    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for i in range(N):
        bounds.append((0, W))
    for i in range(N):
        bounds.append((0, H))
    for i in range(N):
        bounds.append((1e-4, min(W, H) / 2))
    
    try:
        res = minimize(_slsqp_objective, x0, method='SLSQP', jac=lambda x: _slsqp_gradient(x, W, H),
                       bounds=bounds, args=(W, H),
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.fun < _slsqp_objective(x0, W, H):
            cx_opt = res.x[:N]
            cy_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            # Validate
            if _valid_pack(cx_opt, cy_opt, r_opt, W, H):
                return cx_opt, cy_opt, r_opt, float(np.sum(r_opt))
    except:
        pass
    
    # If SLSQP failed or invalid, return force-directed result if valid
    if _valid_pack(cx, cy, r, W, H):
        return cx, cy, r, float(np.sum(r))
    
    return None, None, None, -float('inf')

def _try_aspect_ratio(W, H, deadline):
    """Try multiple grid sizes for a given aspect ratio."""
    best_val = -float('inf')
    best_result = None
    for grid_n in range(3, 8):
        if time.time() > deadline:
            break
        cx, cy, r, val = _optimize_for_aspect(W, H, grid_n, deadline)
        if val > best_val:
            best_val = val
            best_result = (cx, cy, r, W, H)
    return best_result, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_val = -float('inf')
    best_result = None
    
    # Try various aspect ratios (W + H = 2, so H = 2 - W)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        result, val = _try_aspect_ratio(W, H, deadline)
        if val > best_val:
            best_val = val
            best_result = result
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    cx, cy, r, W, H = best_result
    return np.column_stack([cx, cy, r])
