"""Circle packing in perimeter-4 rectangle: sequential maximal placement + local search."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE

def _max_radius_at(px, py, cx, cy, r, W, H, margin=1e-9):
    """Max radius for a circle at (px,py) given existing circles and rectangle walls."""
    # Wall constraint
    d_wall = min(px, W - px, py, H - py)
    max_r = d_wall
    # Circle-circle constraint
    n = len(cx)
    for i in range(n):
        d = np.hypot(px - cx[i], py - cy[i])
        # Need: dist >= r_new + r[i] => r_new <= d - r[i]
        avail = d - r[i]
        if avail < max_r:
            max_r = avail
    return max(0.0, max_r - margin)

def _find_best_spot(W, H, cx, cy, r, n_existing, n_samples=2000):
    """Find the spot with maximum possible radius for the next circle."""
    best_r = 0.0
    best_px, best_py = W / 2, H / 2
    # Grid + random sampling
    nx = max(10, int(np.sqrt(n_samples * W / max(H, 1e-6))))
    ny = max(10, int(np.sqrt(n_samples * H / max(W, 1e-6))))
    xs = np.linspace(0.01 * W, 0.99 * W, nx)
    ys = np.linspace(0.01 * H, 0.99 * H, ny)
    for px in xs:
        for py in ys:
            rad = _max_radius_at(px, py, cx, cy, r, W, H)
            if rad > best_r:
                best_r = rad
                best_px, best_py = px, py
    # Random refinement around best spot
    for _ in range(200):
        px = best_px + random.gauss(0, 0.02 * W)
        py = best_py + random.gauss(0, 0.02 * H)
        px = max(0.01 * W, min(0.99 * W, px))
        py = max(0.01 * H, min(0.99 * H, py))
        rad = _max_radius_at(px, py, cx, cy, r, W, H)
        if rad > best_r:
            best_r = rad
            best_px, best_py = px, py
    return best_px, best_py, best_r

def _sequential_pack(W, H, n_circles=N, n_samples=2000, seed=42):
    """Sequentially place circles at maximal radius."""
    random.seed(seed)
    cx = np.zeros(n_circles)
    cy = np.zeros(n_circles)
    r = np.zeros(n_circles)
    for k in range(n_circles):
        px, py, rad = _find_best_spot(W, H, cx[:k], cy[:k], r[:k], k, n_samples)
        if rad < 1e-6:
            break
        cx[k] = px
        cy[k] = py
        r[k] = rad
    return np.column_stack([cx, cy, r])

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _local_search_swap(x, W, H, n_iters=50):
    """Remove smallest circle, reinsert at best spot. Repeat."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = N
    best_sum = np.sum(r)
    
    for it in range(n_iters):
        if not _time_ok():
            break
        # Find smallest circle
        idx = np.argmin(r)
        # Remove it
        cx_rem = np.delete(cx, idx)
        cy_rem = np.delete(cy, idx)
        r_rem = np.delete(r, idx)
        # Find best spot
        px, py, rad = _find_best_spot(W, H, cx_rem, cy_rem, r_rem, n - 1, 1500)
        if rad > r[idx] + 1e-8:
            # Reinsert at new location
            cx[idx] = px
            cy[idx] = py
            r[idx] = rad
            best_sum = np.sum(r)
        else:
            # Try a few random removals
            for _ in range(3):
                idx2 = random.randint(0, n - 1)
                cx_rem2 = np.delete(cx, idx2)
                cy_rem2 = np.delete(cy, idx2)
                r_rem2 = np.delete(r, idx2)
                px2, py2, rad2 = _find_best_spot(W, H, cx_rem2, cy_rem2, r_rem2, n - 1, 1000)
                if rad2 > r[idx2] + 1e-8:
                    cx[idx2] = px2
                    cy[idx2] = py2
                    r[idx2] = rad2
                    best_sum = np.sum(r)
                    break
    return np.column_stack([cx, cy, r])

def _local_search_perturb(x, W, H, n_iters=30):
    """Perturb positions slightly and re-optimize radii."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = N
    best_sum = np.sum(r)
    
    for it in range(n_iters):
        if not _time_ok():
            break
        # Perturb a random circle
        idx = random.randint(0, n - 1)
        cx[idx] += random.gauss(0, 0.02)
        cy[idx] += random.gauss(0, 0.02)
        cx[idx] = max(0.0, min(W, cx[idx]))
        cy[idx] = max(0.0, min(H, cy[idx]))
        # Recompute max radius for this circle
        rad = _max_radius_at(cx[idx], cy[idx], np.delete(cx, idx), np.delete(cy, idx), np.delete(r, idx), W, H)
        if rad > r[idx] + 1e-8:
            r[idx] = rad
            new_sum = np.sum(r)
            if new_sum > best_sum:
                best_sum = new_sum
            else:
                # Revert
                cx[idx] -= random.gauss(0, 0.02)
                cy[idx] -= random.gauss(0, 0.02)
                cx[idx] = max(0.0, min(W, cx[idx]))
                cy[idx] = max(0.0, min(H, cy[idx]))
                r[idx] = _max_radius_at(cx[idx], cy[idx], np.delete(cx, idx), np.delete(cy, idx), np.delete(r, idx), W, H)
        else:
            # Revert
            cx[idx] -= random.gauss(0, 0.02)
            cy[idx] -= random.gauss(0, 0.02)
            cx[idx] = max(0.0, min(W, cx[idx]))
            cy[idx] = max(0.0, min(H, cy[idx]))
            r[idx] = _max_radius_at(cx[idx], cy[idx], np.delete(cx, idx), np.delete(cy, idx), np.delete(r, idx), W, H)
    return np.column_stack([cx, cy, r])

def _optimize_from(x0, W, H):
    """Multi-stage: local search swap + perturb."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0
    
    # Stage 1: swap-based local search
    if _time_ok():
        x_ls = _local_search_swap(best_x, W, H, n_iters=40)
        s_ls = _evaluate(x_ls, W, H)
        if s_ls is not None and s_ls > best_sum:
            best_sum = s_ls
            best_x = x_ls
    
    # Stage 2: perturb-based local search
    if _time_ok():
        x_lp = _local_search_perturb(best_x, W, H, n_iters=30)
        s_lp = _evaluate(x_lp, W, H)
        if s_lp is not None and s_lp > best_sum:
            best_sum = s_lp
            best_x = x_lp
    
    return best_x, best_sum

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Aspect ratios to try
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    # Phase 1: Sequential packing with different seeds and aspect ratios
    candidates = []
    for W in Ws:
        H = 2.0 - W
        for seed in [42, 123, 456]:
            if not _time_ok():
                break
            x = _sequential_pack(W, H, N, n_samples=1500, seed=seed)
            s = _evaluate(x, W, H)
            if s is not None:
                candidates.append((s, W, H, x))
    
    # Sort by initial score
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 2: Local search on top candidates
    top_k = min(8, len(candidates))
    for idx in range(top_k):
        if not _time_ok():
            break
        s0, W, H, x0 = candidates[idx]
        x_opt, s_opt = _optimize_from(x0, W, H)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H
    
    # Phase 3: More diverse sequential packings if time remains
    if _time_ok():
        remaining = DEADLINE - time.time()
        n_extra = min(10, int(remaining // 8))
        for i in range(n_extra):
            if not _time_ok():
                break
            W = random.uniform(0.7, 1.4)
            H = 2.0 - W
            seed = random.randint(1, 10000)
            x = _sequential_pack(W, H, N, n_samples=1200, seed=seed)
            s = _evaluate(x, W, H)
            if s is not None:
                x_opt, s_opt = _optimize_from(x, W, H)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_arr = x_opt
                    best_W = W
                    best_H = H
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        x = _sequential_pack(W, H, N, n_samples=2000, seed=42)
        best_arr = x
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
    
    # Final validation
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Try fallback sequential pack
        W, H = 1.0, 1.0
        x = _sequential_pack(W, H, N, n_samples=2000, seed=999)
        cx = np.clip(x[:, 0], 0.0, W)
        cy = np.clip(x[:, 1], 0.0, H)
        r = np.maximum(x[:, 2], 1e-6)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r, lo)
        r = np.maximum(r, 1e-6)
        out2 = np.column_stack([cx, cy, r])
        if _evaluate(out2, W, H) is not None:
            return out2
        # Ultimate fallback: small circles in grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
