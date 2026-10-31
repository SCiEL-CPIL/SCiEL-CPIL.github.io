"""Circle packing in perimeter-4 rectangle: SLSQP + coordinate-descent hybrid.

Strategy:
  1. For several aspect ratios, generate a grid seed of 21 circles.
  2. Use SLSQP to maximize sum of radii subject to non-overlap and wall constraints.
  3. After SLSQP, run a coordinate-descent local refinement to squeeze extra radii.
  4. Return the best feasible packing found.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # safety margin before 150 s kill


def _time_ok():
    return time.time() < DEADLINE


def _make_constraints(w, h):
    """Build constraint list for SLSQP for a fixed rectangle w x h."""
    cons = []
    # r_i >= 1e-4
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - 1e-4,
        })
    # cx_i - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[N + i] - x[i],
        })
    # (w - cx_i) - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (w - x[N + i]) - x[i],
        })
    # cy_i - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2 * N + i] - x[i],
        })
    # (h - cy_i) - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (h - x[2 * N + i]) - x[i],
        })
    # pairwise: dist^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N + i] - x[N + j]) ** 2 +
                    (x[2 * N + i] - x[2 * N + j]) ** 2 -
                    (x[i] + x[j]) ** 2
                ),
            })
    return cons


def _grid_seed(w, h):
    """Generate a grid-based initial layout for 21 circles in w x h rectangle."""
    best_seed = None
    best_score = -1e9
    for cols in range(3, 8):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.98
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
        # Feasibility check
        ok = True
        for i in range(N):
            ri = x[i]
            cx = x[N + i]
            cy = x[2 * N + i]
            if cx < ri - 1e-9 or cx > w - ri + 1e-9 or cy < ri - 1e-9 or cy > h - ri + 1e-9:
                ok = False
                break
        if ok:
            for i in range(N):
                for j in range(i + 1, N):
                    d2 = (x[N + i] - x[N + j]) ** 2 + (x[2 * N + i] - x[2 * N + j]) ** 2
                    if d2 < (x[i] + x[j]) ** 2 - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            score = np.sum(x[:N])
            if score > best_score:
                best_score = score
                best_seed = x.copy()
    if best_seed is None:
        # Fallback: tiny circles in a line
        best_seed = np.zeros(3 * N)
        best_seed[:N] = 1e-4
        xs = np.linspace(0.05, w - 0.05, N)
        best_seed[N:2 * N] = xs
        best_seed[2 * N:3 * N] = h / 2.0
    return best_seed


def _is_feasible(x, w, tol=1e-6):
    """Check if a solution is feasible."""
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol) or np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _local_refine(x, w, h, max_rounds=80, step_size=0.01):
    """Coordinate-descent refinement on centers and radii to increase sum of radii."""
    x = x.copy()
    current_score = np.sum(x[:N])
    
    for _ in range(max_rounds):
        if not _time_ok():
            break
        improved = False
        
        # Try adjusting centers
        for i in range(N):
            ri = x[i]
            cx = x[N + i]
            cy = x[2 * N + i]
            
            # X perturbations
            for dx in (-step_size, step_size):
                nx = cx + dx
                if ri < nx < w - ri:
                    tx = x.copy()
                    tx[N + i] = nx
                    # Check feasibility quickly
                    if _is_feasible(tx, w, tol=1e-7):
                        tscore = np.sum(tx[:N])
                        if tscore > current_score + 1e-12:
                            x = tx
                            current_score = tscore
                            improved = True
            
            # Y perturbations
            for dy in (-step_size, step_size):
                ny = cy + dy
                if ri < ny < h - ri:
                    tx = x.copy()
                    tx[2 * N + i] = ny
                    if _is_feasible(tx, w, tol=1e-7):
                        tscore = np.sum(tx[:N])
                        if tscore > current_score + 1e-12:
                            x = tx
                            current_score = tscore
                            improved = True
        
        # Try adjusting radii (increase)
        for i in range(N):
            ri = x[i]
            cx = x[N + i]
            cy = x[2 * N + i]
            
            # Max possible radius from walls
            max_r = min(cx, w - cx, cy, h - cy)
            
            # Max possible radius from other circles
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(cx - x[N + j], cy - x[2 * N + j])
                max_r = min(max_r, d - x[j])
            
            max_r = max(max_r, 1e-4)
            
            if max_r > ri + 1e-9:
                # Try to increase radius
                new_r = min(ri + step_size, max_r)
                tx = x.copy()
                tx[i] = new_r
                if _is_feasible(tx, w, tol=1e-7):
                    tscore = np.sum(tx[:N])
                    if tscore > current_score + 1e-12:
                        x = tx
                        current_score = tscore
                        improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return x, current_score


def _slsqp_solve(w, h, seed, maxiter=200):
    """Run SLSQP optimization for a fixed rectangle."""
    if not _time_ok():
        return None
    
    cons = _make_constraints(w, h)
    bounds = []
    for i in range(N):
        bounds.append((1e-4, min(w, h) / 2.0))
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))
    
    def objective(x):
        return -np.sum(x[:N])
    
    try:
        res = minimize(
            objective,
            seed,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10},
        )
    except Exception:
        return None
    
    if not _time_ok():
        return None
    
    x = res.x
    x[:N] = np.maximum(x[:N], 1e-4)
    
    if _is_feasible(x, w, tol=1e-5):
        return x
    
    return None


def construct():
    best_score = 0.0
    best_arr = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7, 1.5, 0.65]
    
    for w in aspect_ratios:
        if not _time_ok():
            break
        if w <= 0.1 or w >= 1.9:
            continue
        h = 2.0 - w
        
        # Generate grid seed
        seed = _grid_seed(w, h)
        
        # Run SLSQP
        x = _slsqp_solve(w, h, seed, maxiter=150)
        
        if x is not None:
            # Local refinement
            x, score = _local_refine(x, w, h)
            
            if score > best_score:
                best_score = score
                cx = x[N:2 * N]
                cy = x[2 * N:3 * N]
                r = x[:N]
                # Clamp to be safe
                r = np.maximum(r, 1e-4)
                cx = np.clip(cx, r, w - r)
                cy = np.clip(cy, r, h - r)
                # Re-check after clamping
                x_final = np.concatenate([r, cx, cy])
                if _is_feasible(x_final, w, tol=1e-5):
                    best_arr = np.column_stack((cx, cy, r))
                else:
                    # Shrink radii slightly
                    r = r * 0.99
                    cx = np.clip(cx, r, w - r)
                    cy = np.clip(cy, r, h - r)
                    x_final = np.concatenate([r, cx, cy])
                    if _is_feasible(x_final, w, tol=1e-5):
                        best_arr = np.column_stack((cx, cy, r))
    
    # Fallback if nothing valid found
    if best_arr is None:
        w, h = 1.0, 1.0
        r = 1e-4
        cx = np.linspace(r, w - r, N)
        cy = np.full(N, h / 2.0)
        best_arr = np.column_stack((cx, cy, r))
    
    return best_arr
