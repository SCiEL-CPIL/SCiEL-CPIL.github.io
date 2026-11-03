"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _hexagonal_init(n_circles, W, H):
    """Generate hexagonal packing initial positions."""
    # Try to fit in a hex pattern
    # For 21 circles, try 5x5 grid with some removed, or 4-5-4-4 pattern
    # Let's use a grid-based hex offset
    positions = []
    # Try different grid arrangements
    for cols in range(3, 8):
        rows = int(np.ceil(n_circles / cols))
        if rows * cols >= n_circles:
            # Check if this fits
            r = min(W / cols, H / rows) * 0.95
            if r > 0:
                for i in range(rows):
                    for j in range(cols):
                        if len(positions) >= n_circles:
                            break
                        x = (j + 0.5) / cols * W + (0.5 / cols * W) * (i % 2) * 0.5
                        y = (i + 0.5) / rows * H
                        # Keep within bounds
                        x = min(max(x, r), W - r)
                        y = min(max(y, r), H - r)
                        positions.append([x, y, r * 0.9])
                    if len(positions) >= n_circles:
                        break
            if len(positions) >= n_circles:
                break
    if len(positions) < n_circles:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n_circles)))
        r = min(W, H) / side * 0.45
        for i in range(side):
            for j in range(side):
                if len(positions) >= n_circles:
                    break
                x = (i + 0.5) / side * W
                y = (j + 0.5) / side * H
                positions.append([x, y, r])
    return np.array(positions[:n_circles])


def _objective_and_grad(params, W, H, n):
    """
    Objective: minimize negative sum of radii subject to constraints.
    params: [x1,y1,r1, x2,y2,r2, ..., xn,yn,rn]
    Constraints: circles don't overlap, stay in bounds.
    """
    # Unpack
    xy_r = params.reshape(n, 3)
    x = xy_r[:, 0]
    y = xy_r[:, 1]
    r = xy_r[:, 2]
    
    # Objective: maximize sum of radii -> minimize negative
    obj = -np.sum(r)
    
    # Compute gradient
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # dr term for objective
    
    # Penalty for overlaps: for each pair, if dist < ri + rj, push apart
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                # Overlap penalty
                overlap = min_dist - dist
                if dist > 1e-10:
                    fx = dx / dist
                    fy = dy / dist
                else:
                    fx = 1.0
                    fy = 0.0
                # Gradient contribution
                grad[3*i] += 2 * overlap * fx
                grad[3*i+1] += 2 * overlap * fy
                grad[3*i+2] += 2 * overlap
                grad[3*j] -= 2 * overlap * fx
                grad[3*j+1] -= 2 * overlap * fy
                grad[3*j+2] += 2 * overlap
                penalty += overlap ** 2
            # Boundary constraints
            # x[i] >= r[i]
            if x[i] < r[i]:
                grad[3*i] += 2 * (r[i] - x[i])
                grad[3*i+2] -= 2 * (r[i] - x[i])
                penalty += (r[i] - x[i]) ** 2
            # x[i] <= W - r[i]
            if x[i] > W - r[i]:
                grad[3*i] -= 2 * (x[i] - (W - r[i]))
                grad[3*i+2] -= 2 * (x[i] - (W - r[i]))
                penalty += (x[i] - (W - r[i])) ** 2
            # y[i] >= r[i]
            if y[i] < r[i]:
                grad[3*i+1] += 2 * (r[i] - y[i])
                grad[3*i+2] -= 2 * (r[i] - y[i])
                penalty += (r[i] - y[i]) ** 2
            # y[i] <= H - r[i]
            if y[i] > H - r[i]:
                grad[3*i+1] -= 2 * (y[i] - (H - r[i]))
                grad[3*i+2] -= 2 * (y[i] - (H - r[i]))
                penalty += (y[i] - (H - r[i])) ** 2
    
    obj += penalty
    return obj, grad


def _is_valid(xy_r, W, H, tol=1e-6):
    """Check if configuration is valid."""
    n = len(xy_r)
    x = xy_r[:, 0]
    y = xy_r[:, 1]
    r = xy_r[:, 2]
    if np.any(r < 0):
        return False
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
            return False
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Perimeter 4: 2*(W+H) = 4, so W+H = 2
    # Try different aspect ratios
    best_result = None
    best_sum_r = -1
    
    # Aspect ratios to try: W/H
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        W = 2.0 * ar / (1 + ar)
        H = 2.0 / (1 + ar)
        
        # Initial layout
        init = _hexagonal_init(N, W, H)
        
        # Bounds for optimizer
        n = N
        lb = np.zeros(3 * n)
        ub = np.zeros(3 * n)
        for i in range(n):
            lb[3*i] = 0.0
            ub[3*i] = W
            lb[3*i+1] = 0.0
            ub[3*i+1] = H
            lb[3*i+2] = 0.001
            ub[3*i+2] = min(W, H) / 2
        
        # Run optimization
        try:
            result = minimize(
                _objective_and_grad,
                init.flatten(),
                args=(W, H, n),
                method='L-BFGS-B',
                jac=True,
                bounds=list(zip(lb, ub)),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        params = result.x.reshape(n, 3)
        
        # Verify validity
        if _is_valid(params, W, H, tol=1e-5):
            sum_r = np.sum(params[:, 2])
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_result = params.copy()
        
        # Try a polish pass with tighter settings if time allows
        if time.time() < deadline - 20 and best_result is not None:
            try:
                result2 = minimize(
                    _objective_and_grad,
                    params.flatten(),
                    args=(W, H, n),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=list(zip(lb, ub)),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if time.time() > deadline:
                    break
                params2 = result2.x.reshape(n, 3)
                if _is_valid(params2, W, H, tol=1e-5):
                    sum_r2 = np.sum(params2[:, 2])
                    if sum_r2 > best_sum_r:
                        best_sum_r = sum_r2
                        best_result = params2.copy()
            except Exception:
                pass
    
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
