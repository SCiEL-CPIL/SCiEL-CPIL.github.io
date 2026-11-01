"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed.
    
    z layout: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_N-1, y_N-1, r_N-1, w]
    h is determined by perimeter: h = 2.0 - w
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    
    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6
    
    obj = -np.sum(r)
    
    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)
    
    # Wall constraints (smooth)
    wall_pen = (
        np.maximum(0, r - x) +
        np.maximum(0, x + r - w) +
        np.maximum(0, r - y) +
        np.maximum(0, y + r - h)
    )
    obj += penalty_weight * np.sum(wall_pen * wall_pen)
    
    # Small radius penalty to avoid degenerate solutions
    obj += 0.01 * np.sum(np.maximum(0, 0.005 - r) ** 2)
    
    return obj

def _is_valid(x, y, r, w, h, tol=1e-7):
    """Check if configuration is valid."""
    if w < 0.01 or h < 0.01:
        return False
    if np.any(r < 0):
        return False
    # Wall constraints
    if np.any(x - r < -tol) or np.any(x + r > w + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > h + tol):
        return False
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _get_initial_guesses():
    """Generate multiple initial configurations."""
    guesses = []
    
    # Grid layout 5x5 (with 21 circles, some cells empty)
    side = 5
    w0 = 1.0
    h0 = 1.0
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
    x0 = np.array([p[0] for p in positions])
    y0 = np.array([p[1] for p in positions])
    r0 = np.full(N, 0.5 / side * 0.9)
    z0 = np.zeros(3 * N + 1)
    z0[0::3][:N] = x0
    z0[1::3][:N] = y0
    z0[2::3][:N] = r0
    z0[-1] = w0
    guesses.append(z0)
    
    # Square grid 3x7
    cols, rows = 7, 3
    w1 = 1.4
    h1 = 0.6
    positions1 = []
    for i in range(cols):
        for j in range(rows):
            if len(positions1) < N:
                positions1.append(((i + 0.5) / cols, (j + 0.5) / rows))
    x1 = np.array([p[0] for p in positions1])
    y1 = np.array([p[1] for p in positions1])
    r1 = np.full(N, 0.5 / cols * 0.85)
    z1 = np.zeros(3 * N + 1)
    z1[0::3][:N] = x1
    z1[1::3][:N] = y1
    z1[2::3][:N] = r1
    z1[-1] = w1
    guesses.append(z1)
    
    # Random initializations
    for seed in range(3):
        np.random.seed(seed)
        x_r = np.random.uniform(0.1, 0.9, N)
        y_r = np.random.uniform(0.1, 0.9, N)
        r_r = np.full(N, 0.08)
        z_r = np.zeros(3 * N + 1)
        z_r[0::3][:N] = x_r
        z_r[1::3][:N] = y_r
        z_r[2::3][:N] = r_r
        z_r[-1] = 1.0 + 0.1 * (seed - 1)
        guesses.append(z_r)
    
    return guesses

def _expansion_relaxation_polish(x, y, r, w, h, max_iter=2000, tol=1e-9):
    """Iteratively expand circles and relax positions to remove overlaps."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for iteration in range(max_iter):
        # Try to expand each circle
        for i in range(N):
            # Max possible radius given walls
            r_max_wall = min(x[i], w - x[i], y[i], h - y[i])
            
            # Max radius given other circles
            r_max_other = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    r_max_other = min(r_max_other, dist - r[j])
            
            r_target = min(r_max_wall, r_max_other, r[i] + 0.001)
            r_target = max(r_target, r[i] * 0.9)  # Don't shrink too much
            
            if r_target > r[i] + tol:
                r[i] = r_target
            
            # Relax position to resolve any remaining overlaps
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    x[i] += nx * overlap * 0.5
                    y[i] += ny * overlap * 0.5
                    x[j] -= nx * overlap * 0.5
                    y[j] -= ny * overlap * 0.5
            
            # Clamp to walls
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_z = None
    best_sum_r = -np.inf
    
    initial_guesses = _get_initial_guesses()
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        z0 = guess.copy()
        
        try:
            # SLSQP optimization
            res = minimize(
                _vectorized_objective,
                z0,
                method='SLSQP',
                bounds=[(0.01, 1.99)] * (3 * N) + [(0.3, 1.7)],
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success or res.fun < _vectorized_objective(z0):
                z_res = res.x
                x_res = z_res[0::3][:N]
                y_res = z_res[1::3][:N]
                r_res = z_res[2::3][:N]
                w_res = z_res[-1]
                h_res = 2.0 - w_res
                
                # Check validity
                if _is_valid(x_res, y_res, r_res, w_res, h_res):
                    sum_r = np.sum(r_res)
                    if sum_r > best_sum_r:
                        best_sum_r = sum_r
                        best_z = z_res.copy()
                else:
                    # Try to fix by reducing radii
                    x_fix = x_res.copy()
                    y_fix = y_res.copy()
                    r_fix = r_res.copy()
                    for i in range(N):
                        r_fix[i] = min(r_fix[i], x_fix[i], w_res - x_fix[i], y_fix[i], h_res - y_fix[i])
                        x_fix[i] = np.clip(x_fix[i], r_fix[i], w_res - r_fix[i])
                        y_fix[i] = np.clip(y_fix[i], r_fix[i], h_res - r_fix[i])
                    
                    # Reduce radii to remove overlaps
                    for _ in range(100):
                        max_overlap = 0
                        for i in range(N):
                            for j in range(i + 1, N):
                                dx = x_fix[i] - x_fix[j]
                                dy = y_fix[i] - y_fix[j]
                                dist = np.sqrt(dx * dx + dy * dy)
                                min_dist = r_fix[i] + r_fix[j]
                                if dist < min_dist:
                                    overlap = min_dist - dist
                                    max_overlap = max(max_overlap, overlap)
                                    r_fix[i] = max(0.005, r_fix[i] - overlap / 2)
                                    r_fix[j] = max(0.005, r_fix[j] - overlap / 2)
                        if max_overlap < 1e-8:
                            break
                    
                    x_fix = np.clip(x_fix, r_fix, w_res - r_fix)
                    y_fix = np.clip(y_fix, r_fix, h_res - r_fix)
                    
                    if _is_valid(x_fix, y_fix, r_fix, w_res, h_res):
                        sum_r = np.sum(r_fix)
                        if sum_r > best_sum_r:
                            best_sum_r = sum_r
                            z_fix = np.zeros(3 * N + 1)
                            z_fix[0::3][:N] = x_fix
                            z_fix[1::3][:N] = y_fix
                            z_fix[2::3][:N] = r_fix
                            z_fix[-1] = w_res
                            best_z = z_fix.copy()
        except Exception:
            pass
    
    # If we have a best solution, try expansion polish
    if best_z is not None and time.time() < deadline - 30:
        x_best = best_z[0::3][:N]
        y_best = best_z[1::3][:N]
        r_best = best_z[2::3][:N]
        w_best = best_z[-1]
        h_best = 2.0 - w_best
        
        try:
            x_polished, y_polished, r_polished = _expansion_relaxation_polish(
                x_best, y_best, r_best, w_best, h_best,
                max_iter=1000, tol=1e-10
            )
            
            if _is_valid(x_polished, y_polished, r_polished, w_best, h_best):
                sum_r_polished = np.sum(r_polished)
                if sum_r_polished > best_sum_r:
                    best_sum_r = sum_r_polished
                    z_polished = np.zeros(3 * N + 1)
                    z_polished[0::3][:N] = x_polished
                    z_polished[1::3][:N] = y_polished
                    z_polished[2::3][:N] = r_polished
                    z_polished[-1] = w_best
                    best_z = z_polished
        except Exception:
            pass
    
    # Final fallback: simple grid
    if best_z is None:
        side = 5
        w_f = 1.0
        h_f = 1.0
        x_f = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y_f = np.array([(j + 0.5) / side for j in range(side) for _ in range(side)][:N])
        r_f = np.full(N, 0.5 / side * 0.8)
        return np.column_stack([x_f, y_f, r_f])
    
    # Extract final result
    x_final = best_z[0::3][:N]
    y_final = best_z[1::3][:N]
    r_final = best_z[2::3][:N]
    w_final = best_z[-1]
    h_final = 2.0 - w_final
    
    # Final safety: ensure validity
    for i in range(N):
        r_final[i] = min(r_final[i], x_final[i], w_final - x_final[i], y_final[i], h_final - y_final[i])
        x_final[i] = np.clip(x_final[i], r_final[i], w_final - r_final[i])
        y_final[i] = np.clip(y_final[i], r_final[i], h_final - r_final[i])
    
    # Remove any remaining overlaps by reducing radii
    for _ in range(200):
        max_overlap = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_final[i] - x_final[j]
                dy = y_final[i] - y_final[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r_final[i] + r_final[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    max_overlap = max(max_overlap, overlap)
                    r_final[i] = max(0.005, r_final[i] - overlap / 2)
                    r_final[j] = max(0.005, r_final[j] - overlap / 2)
        if max_overlap < 1e-9:
            break
    
    x_final = np.clip(x_final, r_final, w_final - r_final)
    y_final = np.clip(y_final, r_final, h_final - r_final)
    
    return np.column_stack([x_final, y_final, r_final])
# EVOLVE-BLOCK-END
