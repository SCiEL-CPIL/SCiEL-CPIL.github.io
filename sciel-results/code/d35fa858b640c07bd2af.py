"""Circle packing in perimeter-4 rectangle: Greedy radius maximization + jitter search."""
import numpy as np
import time
import random

N = 21
START = time.time()
DEADLINE = START + 110.0  # Safety margin

def _time_ok():
    return time.time() < DEADLINE

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
    # Vectorized overlap check
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _maximize_radii(cx, cy, W, H, n_iter=50):
    """
    Given fixed positions, maximize radii greedily.
    This is fast and ensures feasibility if starting from a valid position set.
    """
    n = len(cx)
    r = np.zeros(n)
    
    # Initialize radii to 0
    for it in range(n_iter):
        for i in range(n):
            # Max radius limited by boundaries
            max_r = min(cx[i], W - cx[i], cy[i], H - cy[i])
            
            # Max radius limited by other circles
            # r_i + r_j <= d_ij  =>  r_i <= d_ij - r_j
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                # If current r_j is large, it constrains r_i
                # We want the largest r_i such that r_i + r_j <= d
                # So r_i <= d - r_j
                # But we also need r_i >= 0
                constraint = d - r[j]
                if constraint < max_r:
                    max_r = constraint
            
            r[i] = max(0.0, max_r)
    
    # Final clamp to ensure strict non-overlap (numerical safety)
    # Re-calculate with a small epsilon reduction
    r_final = np.zeros(n)
    for i in range(n):
        max_r = min(cx[i], W - cx[i], cy[i], H - cy[i])
        for j in range(n):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            constraint = d - r_final[j]
            if constraint < max_r:
                max_r = constraint
        r_final[i] = max(0.0, max_r)
        
    # Apply a small shrink factor to guarantee disjoint interiors
    r_final *= 0.9999
    return r_final

def _jitter_search(cx, cy, W, H, n_restarts=50, jitter_scale=0.05):
    """
    Local search: jitter positions, then maximize radii.
    """
    best_sum = -1.0
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = _maximize_radii(cx, cy, W, H)
    best_sum = np.sum(best_r)
    
    current_cx = cx.copy()
    current_cy = cy.copy()
    
    for i in range(n_restarts):
        if not _time_ok():
            break
            
        # Jitter
        new_cx = current_cx + np.random.randn(N) * jitter_scale
        new_cy = current_cy + np.random.randn(N) * jitter_scale
        
        # Clamp to keep centers inside (with some margin to allow radius > 0)
        new_cx = np.clip(new_cx, 0.0, W)
        new_cy = np.clip(new_cy, 0.0, H)
        
        # Maximize radii for new positions
        new_r = _maximize_radii(new_cx, new_cy, W, H)
        new_sum = np.sum(new_r)
        
        if new_sum > best_sum:
            best_sum = new_sum
            best_cx = new_cx.copy()
            best_cy = new_cy.copy()
            best_r = new_r.copy()
            current_cx = new_cx.copy()
            current_cy = new_cy.copy()
            
            # Adapt jitter scale: decrease if improving, increase if stuck?
            # Simple: keep constant or slightly decay
            jitter_scale *= 0.98
    
    return best_cx, best_cy, best_r, best_sum

def _grid_start(W, H, kx, ky):
    """Create a grid starting layout."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            pts.append([xx, yy])
    arr = np.array(pts)
    if len(arr) < N:
        # Pad with small circles at edges
        while len(arr) < N:
            arr = np.vstack([arr, [0.02, 0.02]])
    return arr[:N]

def _hex_start(W, H):
    """Hexagonal packing starting layout."""
    rows_configs = [
        [7, 7, 7],
        [6, 7, 8],
        [5, 7, 9],
        [6, 6, 9],
        [7, 6, 8],
        [8, 7, 6],
        [5, 6, 10],
        [6, 8, 7],
    ]
    
    best = None
    best_sum = 0
    
    for config in rows_configs:
        if sum(config) != N:
            continue
        nr = len(config)
        sx = W / (max(config) + 0.5)
        sy = H / (nr + 0.5)
        
        cx = np.zeros(N)
        cy = np.zeros(N)
        idx = 0
        for ri, cnt in enumerate(config):
            y = (ri + 0.5) * sy
            offset = (ri % 2) * sx * 0.5
            for ci in range(cnt):
                x = (ci + 0.5) * sx + offset
                cx[idx] = x
                cy[idx] = y
                idx += 1
        
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        
        arr = np.column_stack([cx, cy])
        # Evaluate this layout
        r = _maximize_radii(cx, cy, W, H)
        s_val = np.sum(r)
        if s_val > best_sum:
            best_sum = s_val
            best = arr
    
    if best is not None:
        return best
    
    # Fallback to grid
    return _grid_start(W, H, 5, 5)

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Try fewer aspect ratios to save time
    Ws = np.linspace(0.8, 1.2, 9)
    
    for W in Ws:
        if not _time_ok():
            break
        H = 2.0 - W
        if H < 0.5:
            continue
        
        # Generate starting configurations
        starts = []
        
        # Grid starts
        for kx, ky in [(4, 5), (5, 4), (6, 4), (4, 6), (7, 3)]:
            if kx * ky >= N:
                g = _grid_start(W, H, kx, ky)
                starts.append(g)
        
        # Hexagonal start
        h = _hex_start(W, H)
        starts.append(h)
        
        # Relax each start using jitter search
        for s0 in starts:
            if not _time_ok():
                break
            cx0 = s0[:, 0]
            cy0 = s0[:, 1]
            
            # Run jitter search
            cx_res, cy_res, r_res, s_res = _jitter_search(cx0, cy0, W, H, n_restarts=30, jitter_scale=0.05)
            
            if s_res > best_sum:
                best_sum = s_res
                best_arr = np.column_stack([cx_res, cy_res, r_res])
                best_W = W
                best_H = H
        
        # If we have time, do a second pass with finer jitter on the best found so far for this W
        if _time_ok() and best_arr is not None and best_W == W:
            cx0 = best_arr[:, 0]
            cy0 = best_arr[:, 1]
            cx_res, cy_res, r_res, s_res = _jitter_search(cx0, cy0, W, H, n_restarts=50, jitter_scale=0.02)
            if s_res > best_sum:
                best_sum = s_res
                best_arr = np.column_stack([cx_res, cy_res, r_res])
                best_W = W
                best_H = H
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_start(W, H, 5, 5)
        cx0, cy0 = g[:, 0], g[:, 1]
        cx_res, cy_res, r_res, s_res = _jitter_search(cx0, cy0, W, H, n_restarts=10, jitter_scale=0.05)
        best_arr = np.column_stack([cx_res, cy_res, r_res])
        best_W, best_H = W, H
        best_sum = s_res
    
    # Final clamping and validation
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    
    # Ensure validity
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Fallback grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
