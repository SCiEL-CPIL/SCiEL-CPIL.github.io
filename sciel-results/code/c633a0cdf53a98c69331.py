"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _objective(x):
    """
    x: (2N+2) array = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}, W, H]
    Minimize negative sum of radii (i.e., maximize sum of radii)
    Subject to:
      - W + H = 2 (perimeter = 2*(W+H) = 4)
      - circles inside rectangle: r_i <= x_i, r_i <= W - x_i, r_i <= y_i, r_i <= H - y_i
      - non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    We use a penalty method for simplicity and robustness.
    """
    N = len(x) // 2 - 1  # Actually x has 2N+2 elements, so N = (len(x)-2)//2
    # Let me restructure: x = [positions (2N), radii (N), W, H]
    # So len(x) = 3N + 2
    n = (len(x) - 2) // 3
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:3*n]
    W = x[3*n]
    H = x[3*n + 1]
    
    # Perimeter constraint: 2*(W + H) = 4 => W + H = 2
    # We'll enforce this via penalty
    
    # Objective: maximize sum(radii) => minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Penalties
    # 1. Perimeter constraint
    perim_pen = 100.0 * (W + H - 2.0)**2
    
    # 2. Circles inside rectangle
    inside_pen = 0.0
    for i in range(n):
        xi, yi = pos[i]
        ri = radii[i]
        # xi - ri >= 0
        if xi - ri < 0:
            inside_pen += 50.0 * (xi - ri)**2
        # W - xi - ri >= 0
        if W - xi - ri < 0:
            inside_pen += 50.0 * (W - xi - ri)**2
        # yi - ri >= 0
        if yi - ri < 0:
            inside_pen += 50.0 * (yi - ri)**2
        # H - yi - ri >= 0
        if H - yi - ri < 0:
            inside_pen += 50.0 * (H - yi - ri)**2
    
    # 3. Non-overlap
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist**2:
                # Penalty for violation
                overlap_pen += 100.0 * (min_dist - np.sqrt(dist_sq))**2
    
    return obj + perim_pen + inside_pen + overlap_pen


def _make_initial_guess(W, H):
    """Create a grid-based initial guess."""
    n = N
    # Use a grid layout
    cols = int(np.ceil(np.sqrt(n * W / H))) if H > 0 else int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        col = i % cols
        row = i // cols
        pos[i, 0] = (col + 0.5) * W / cols
        pos[i, 1] = (row + 0.5) * H / rows
        # Initial radius estimate
        r_est = min(W / cols, H / rows) / 2 * 0.95
        radii[i] = r_est
    
    x = np.concatenate([pos.flatten(), radii, [W, H]])
    return x


def _solve_for_aspect(W, H, time_limit):
    """Solve for a given aspect ratio W x H where W + H = 2."""
    n = N
    x0 = _make_initial_guess(W, H)
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))  # x_i
    for i in range(n):
        bounds.append((0.0, H))  # y_i
    for i in range(n):
        bounds.append((0.01, 1.0))  # r_i
    bounds.append((0.5, 1.5))  # W
    bounds.append((0.5, 1.5))  # H
    
    result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
    
    if result.success or result.fun < 1e6:
        pos = result.x[:2*n].reshape(n, 2)
        radii = result.x[2*n:3*n]
        W_opt = result.x[3*n]
        H_opt = result.x[3*n + 1]
        return pos, radii, W_opt, H_opt, result.fun
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios
    # W + H = 2, so W ranges from ~0.5 to 1.5
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        result = _solve_for_aspect(W, H, deadline - time.time())
        if result is not None:
            pos, radii, W_opt, H_opt, obj_val = result
            sum_radii = np.sum(radii)
            
            # Verify validity
            valid = True
            # Check perimeter
            if abs(2*(W_opt + H_opt) - 4.0) > 0.01:
                valid = False
            # Check circles inside
            for i in range(N):
                xi, yi = pos[i]
                ri = radii[i]
                if xi - ri < -1e-6 or W_opt - xi - ri < -1e-6 or yi - ri < -1e-6 or H_opt - yi - ri < -1e-6:
                    valid = False
            # Check non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    if dx*dx + dy*dy < (radii[i] + radii[j])**2 - 1e-6:
                        valid = False
            
            if valid and sum_radii > best_value:
                best_value = sum_radii
                best_result = (pos, radii, W_opt, H_opt)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    pos, radii, W_opt, H_opt = best_result
    
    # Build output array: (x, y, r) for each circle
    # But we need to scale to perimeter 4. W_opt + H_opt should be ~2.
    # The rectangle is [0, W_opt] x [0, H_opt]
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = pos[i, 0]
        result[i, 1] = pos[i, 1]
        result[i, 2] = radii[i]
    
    return result

# EVOLVE-BLOCK-END
