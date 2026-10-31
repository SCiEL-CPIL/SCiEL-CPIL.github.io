"""Circle packing in perimeter-4 rectangle: force-directed + L-BFGS-B + multi-start."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

def _rect_from_width(W):
    H = 2.0 - W
    return W, H

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    # clamp
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    # check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _force_directed(x, W, H, steps=500, lr=0.01):
    """Run force-directed relaxation to improve packing."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    for step in range(steps):
        # Compute forces
        fcx = np.zeros(n)
        fcy = np.zeros(n)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req and d > 1e-12:
                    # Push apart
                    f = (req - d) / d
                    fcx[i] += f * dx
                    fcy[i] += f * dy
                    fcx[j] -= f * dx
                    fcy[j] -= f * dy
                elif d <= 1e-12:
                    # Random push
                    angle = random.uniform(0, 2 * np.pi)
                    fcx[i] += 0.1 * np.cos(angle)
                    fcy[i] += 0.1 * np.sin(angle)
                    fcx[j] -= 0.1 * np.cos(angle)
                    fcy[j] -= 0.1 * np.sin(angle)
        
        # Attraction to boundaries (grow circles)
        for i in range(n):
            # Distance to each wall
            d_left = cx[i]
            d_right = W - cx[i]
            d_bottom = cy[i]
            d_top = H - cy[i]
            
            # If circle can grow, push it toward the closest wall
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d > r[i] * 0.95:
                # Circle has room to grow, push toward closest wall
                if min_d == d_left:
                    fcx[i] -= 0.01
                elif min_d == d_right:
                    fcx[i] += 0.01
                elif min_d == d_bottom:
                    fcy[i] -= 0.01
                elif min_d == d_top:
                    fcy[i] += 0.01
        
        # Update positions
        cx += lr * fcx
        cy += lr * fcy
        
        # Clamp to rectangle
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        
        # Grow radii where possible
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r * 1.001 + 0.0001, lo)
        r = np.maximum(r, 1e-6)
        
        # Decay learning rate
        lr *= 0.999
    
    return np.column_stack([cx, cy, r])

def _optimize_l_bfgs(x0, W, H, max_iter=100):
    """Run L-BFGS-B optimization."""
    def objective(x):
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:3*N]
        
        # Sum of radii (we want to maximize, so negate)
        val = -np.sum(r)
        
        # Penalty for boundary violations
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        viol_b = np.maximum(r - lo, 0.0)
        val += 1e4 * np.sum(viol_b ** 2)
        
        # Penalty for negative radii
        viol_r = np.maximum(-r, 0.0)
        val += 1e4 * np.sum(viol_r ** 2)
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                viol = r[i] + r[j] - d
                if viol > 0:
                    val += 1e4 * viol ** 2
        
        return val
    
    x_flat = x0.flatten()
    
    try:
        res = minimize(objective, x_flat, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.x is not None:
            x_opt = res.x.reshape(N, 3)
            s = _evaluate(x_opt, W, H)
            if s is not None:
                return x_opt, s
    except Exception:
        pass
    
    return x0, _evaluate(x0, W, H) or 0.0

def _grid_layout(kx, ky, W, H, shrink=0.99):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _random_layout(W, H):
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _optimize_from(x0, W, H):
    """Multi-stage optimization: force-directed then L-BFGS-B."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0
    
    # Stage 1: Force-directed relaxation
    if time.time() < DEADLINE - 10:
        x_fd = _force_directed(x0, W, H, steps=300, lr=0.01)
        s_fd = _evaluate(x_fd, W, H)
        if s_fd is not None and s_fd > best_sum:
            best_sum = s_fd
            best_x = x_fd
    
    # Stage 2: L-BFGS-B from best
    if time.time() < DEADLINE - 5:
        x_opt, s_opt = _optimize_l_bfgs(best_x, W, H, max_iter=80)
        if s_opt > best_sum:
            best_sum = s_opt
            best_x = x_opt
    
    return best_x, best_sum

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse starting candidates
    candidates = []
    
    # Aspect ratios to try
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5), (6, 3), (3, 6)]
    
    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.98)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
    
    # Add random starts
    for W in Ws:
        H = 2.0 - W
        for _ in range(2):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
    
    # Sort by initial score, best first
    candidates.sort(key=lambda t: -t[0])
    
    # Optimize top candidates
    top_k = min(15, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        x_opt, s_opt = _optimize_from(g0, W, H)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H
    
    # If we still have time, do a few more random restarts from the best found
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(6, int(remaining // 10))
        for _ in range(n_extra):
            if not time_ok():
                break
            # Perturb best
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.03
            x0 = np.clip(x0, 0.0, None)
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_opt, s_opt = _optimize_from(x0, best_W, best_H)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.98)
        best_arr = g
        best_W, best_H = W, H
    
    # Final clamping to ensure validity
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    # Final validation: if invalid, fall back to a known-good grid
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        for s0, W, H, g0 in candidates[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = np.maximum(g0[:, 2], 1e-6)
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.minimum(r2, lo2)
            r2 = np.maximum(r2, 1e-6)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        # Ultimate fallback: 5x5 grid in unit square
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
