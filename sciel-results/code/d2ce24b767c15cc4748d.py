"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _objective_and_constraints(x, W, H):
    """
    x: array of shape (3N,) -> [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    W, H: rectangle dimensions (perimeter 4 -> W + H = 2)
    Returns: negative sum of radii (to minimize), and constraint residuals.
    """
    n = len(x) // 3
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Objective: maximize sum of radii -> minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Constraints:
    # 1. No overlap: dist(i,j) >= r_i + r_j for i != j
    # 2. Boundary: r_i <= x_i, r_i <= y_i, r_i <= W - x_i, r_i <= H - y_i
    
    # We'll use a soft penalty approach for the optimization, but for checking validity
    # we need hard constraints.
    
    return obj

def _is_valid(x, W, H, tol=1e-8):
    """Check if configuration is valid (no overlaps, within bounds)."""
    n = len(x) // 3
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Boundary check
    if np.any(radii < -tol):
        return False
    if np.any(centers[0] - radii < -tol) or np.any(centers[1] - radii < -tol):
        return False
    if np.any(W - centers[0] - radii < -tol) or np.any(H - centers[1] - radii < -tol):
        return False
    
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True

def _sum_radii(x):
    """Sum of radii from configuration x."""
    n = len(x) // 3
    return np.sum(x[2*n:])

def _solve_for_aspect(W, H, max_iters=1000, time_limit=None):
    """
    Solve for given rectangle dimensions W, H with W + H = 2.
    Uses gradient-based optimization with penalty for violations.
    """
    n = N
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(n)))
    cols = side
    rows = int(np.ceil(n / cols))
    
    # Better initial guess: distribute in grid
    xs = np.linspace(W * 0.15, W * 0.85, cols)
    ys = np.linspace(H * 0.15, H * 0.85, rows)
    
    x0 = np.zeros(3 * n)
    idx = 0
    for i in range(n):
        c = i % cols
        r = i // cols
        x0[3*idx] = xs[c]
        x0[3*idx + 1] = ys[r]
        x0[3*idx + 2] = min(W, H) / max(cols, rows) * 0.4  # initial radius guess
        idx += 1
    
    def penalized_objective(x, penalty=10.0):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        # Soft penalty for violations
        penalty_val = 0.0
        
        # Boundary violations
        for i in range(n):
            pen = max(0, radii[i] - centers[i, 0]) + max(0, radii[i] - centers[i, 1]) + \
                  max(0, radii[i] - (W - centers[i, 0])) + max(0, radii[i] - (H - centers[i, 1]))
            penalty_val += pen * pen
        
        # Overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    penalty_val += overlap * overlap
        
        obj = -np.sum(radii) + penalty * penalty_val
        return obj
    
    def penalized_grad(x, penalty=10.0):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        grad = np.zeros(3*n)
        
        # Gradient of -sum(radii)
        grad[2*n:] = -1.0
        
        # Gradient of penalty terms
        # Boundary
        for i in range(n):
            # r_i - x_i
            v = radii[i] - centers[i, 0]
            if v > 0:
                grad[3*i] += -2 * penalty * v
                grad[3*i+2] += 2 * penalty * v
            # r_i - y_i
            v = radii[i] - centers[i, 1]
            if v > 0:
                grad[3*i+1] += -2 * penalty * v
                grad[3*i+2] += 2 * penalty * v
            # r_i - (W - x_i)
            v = radii[i] - (W - centers[i, 0])
            if v > 0:
                grad[3*i] += 2 * penalty * v
                grad[3*i+2] += -2 * penalty * v
            # r_i - (H - y_i)
            v = radii[i] - (H - centers[i, 1])
            if v > 0:
                grad[3*i+1] += 2 * penalty * v
                grad[3*i+2] += -2 * penalty * v
        
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    if dist > 1e-10:
                        gx_i = -dx / dist
                        gy_i = -dy / dist
                        gx_j = dx / dist
                        gy_j = dy / dist
                    else:
                        gx_i = gy_i = gx_j = gy_j = 0
                    grad[3*i] += 2 * penalty * overlap * gx_i
                    grad[3*i+1] += 2 * penalty * overlap * gy_i
                    grad[3*i+2] += 2 * penalty * overlap
                    grad[3*j] += 2 * penalty * overlap * gx_j
                    grad[3*j+1] += 2 * penalty * overlap * gy_j
                    grad[3*j+2] += 2 * penalty * overlap
        
        return grad
    
    # Progressive penalty
    best_x = x0.copy()
    best_val = -np.inf
    best_penalty = 0.0
    
    start_time = time.time()
    
    for pen_level in [1.0, 5.0, 20.0, 100.0, 500.0]:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        res = minimize(penalized_objective, best_x, jac=penalized_grad, 
                      args=(pen_level,), method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.x is not None:
            # Check if this solution is valid
            if _is_valid(res.x, W, H):
                val = _sum_radii(res.x)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
            else:
                # Even if not perfectly valid, keep it as starting point for next penalty level
                best_x = res.x.copy()
    
    return best_x, best_val

def _polish(x, W, H, time_limit=10.0):
    """Local refinement using gradient-based optimization with tighter tolerance."""
    n = N
    start_time = time.time()
    
    def penalized_objective(x, penalty=1000.0):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        penalty_val = 0.0
        for i in range(n):
            pen = max(0, radii[i] - centers[i, 0]) + max(0, radii[i] - centers[i, 1]) + \
                  max(0, radii[i] - (W - centers[i, 0])) + max(0, radii[i] - (H - centers[i, 1]))
            penalty_val += pen * pen
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    penalty_val += overlap * overlap
        
        return -np.sum(radii) + penalty * penalty_val
    
    def penalized_grad(x, penalty=1000.0):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        grad = np.zeros(3*n)
        grad[2*n:] = -1.0
        
        for i in range(n):
            v = radii[i] - centers[i, 0]
            if v > 0:
                grad[3*i] += -2 * penalty * v
                grad[3*i+2] += 2 * penalty * v
            v = radii[i] - centers[i, 1]
            if v > 0:
                grad[3*i+1] += -2 * penalty * v
                grad[3*i+2] += 2 * penalty * v
            v = radii[i] - (W - centers[i, 0])
            if v > 0:
                grad[3*i] += 2 * penalty * v
                grad[3*i+2] += -2 * penalty * v
            v = radii[i] - (H - centers[i, 1])
            if v > 0:
                grad[3*i+1] += 2 * penalty * v
                grad[3*i+2] += -2 * penalty * v
        
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    if dist > 1e-10:
                        gx_i = -dx / dist
                        gy_i = -dy / dist
                        gx_j = dx / dist
                        gy_j = dy / dist
                    else:
                        gx_i = gy_i = gx_j = gy_j = 0
                    grad[3*i] += 2 * penalty * overlap * gx_i
                    grad[3*i+1] += 2 * penalty * overlap * gy_i
                    grad[3*i+2] += 2 * penalty * overlap
                    grad[3*j] += 2 * penalty * overlap * gx_j
                    grad[3*j+1] += 2 * penalty * overlap * gy_j
                    grad[3*j+2] += 2 * penalty * overlap
        
        return grad
    
    current = x.copy()
    best = x.copy()
    best_val = _sum_radii(x)
    
    for _ in range(5):
        if time.time() - start_time > time_limit:
            break
        res = minimize(penalized_objective, current, jac=penalized_grad,
                      args=(1000.0,), method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None:
            if _is_valid(res.x, W, H):
                val = _sum_radii(res.x)
                if val > best_val:
                    best_val = val
                    best = res.x.copy()
            current = res.x.copy()
    
    return best, best_val

def construct():
    start_time = time.time()
    
    best_x = None
    best_val = -np.inf
    best_W = 1.0
    best_H = 1.0
    
    # Try various aspect ratios W + H = 2
    # W in [0.5, 1.5], H = 2 - W
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for W in aspect_ratios:
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue
        
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            break
        
        time_limit = min(remaining * 0.7, 20.0)
        x, val = _solve_for_aspect(W, H, max_iters=500, time_limit=time_limit)
        
        if val > best_val:
            best_val = val
            best_x = x.copy()
            best_W = W
            best_H = H
    
    # Polish the best solution
    if best_x is not None:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 5:
            best_x, best_val = _polish(best_x, best_W, best_H, time_limit=min(remaining - 5, 15.0))
    
    # If we still have time, try more aspect ratios around the best
    if best_x is not None:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 10:
            # Refine around best aspect ratio
            fine_aspects = np.linspace(best_W - 0.1, best_W + 0.1, 5)
            for W in fine_aspects:
                H = 2.0 - W
                if W <= 0.3 or H <= 0.3:
                    continue
                if time.time() - start_time > DEADLINE - 10:
                    break
                x, val = _solve_for_aspect(W, H, max_iters=300, time_limit=5.0)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_W = W
                    best_H = H
            
            # Final polish
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 5:
                best_x, best_val = _polish(best_x, best_W, best_H, time_limit=min(remaining - 5, 10.0))
    
    # Convert to output format: (n, 3) array with [x, y, r]
    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    n = N
    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    result = np.zeros((n, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = radii
    
    return result
