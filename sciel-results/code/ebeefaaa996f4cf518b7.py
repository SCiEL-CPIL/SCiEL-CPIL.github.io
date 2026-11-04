"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
import time

# Fixed deadline guard
_DEADLINE = 110.0  # seconds, safety margin from 120s limit

def _make_hex_grid(n_rows, n_cols, aspect):
    """Generate a hexagonal grid of n_rows x n_cols points in a rectangle with given aspect ratio.
    
    Returns array of shape (n_rows*n_cols, 2) with x in [0, aspect], y in [0, 1].
    """
    points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = (j + (0.5 if i % 2 else 0)) * (2.0 / (n_cols + 0.5))
            y = (i + 0.5) * (1.0 / n_rows)
            points.append([x, y])
    return np.array(points)

def _make_rect_grid(n_rows, n_cols, aspect):
    """Generate a rectangular grid of n_rows x n_cols points."""
    points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = (j + 0.5) * (aspect / n_cols)
            y = (i + 0.5) * (1.0 / n_rows)
            points.append([x, y])
    return np.array(points)

def _objective(params, n_circles, aspect):
    """Penalty-based objective: minimize sum of overlaps + boundary violations - sum of radii.
    
    params: array of shape (2*n, n) where first 2*n are (x,y) positions and last n are radii.
    The rectangle has width=aspect, height=1, perimeter = 2*(aspect+1) = 4 => aspect+1=2 => aspect=1.
    Wait, perimeter 4 means 2*(w+h)=4, so w+h=2. Let's parametrize by aspect ratio r = w/h,
    then w + h = 2, so h = 2/(1+r), w = 2r/(1+r).
    """
    # Actually let's use: w + h = 2. Parametrize by r = w/h.
    # h = 2/(1+r), w = 2r/(1+r)
    r = aspect
    h = 2.0 / (1.0 + r)
    w = 2.0 * r / (1.0 + r)
    
    pos = params[:2*n_circles].reshape(n_circles, 2)
    radii = params[2*n_circles:]
    
    # Boundary constraints: each circle must be inside [0,w] x [0,h]
    # x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
    bound_viol = np.maximum(0, -pos[:, 0] + radii) + \
                 np.maximum(0, pos[:, 0] + radii - w) + \
                 np.maximum(0, -pos[:, 1] + radii) + \
                 np.maximum(0, pos[:, 1] + radii - h)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    # Vectorized: compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = radii[:, np.newaxis] + radii[np.newaxis, :]
    # Only upper triangle
    triu_idx = np.triu_indices(n_circles, k=1)
    overlaps = np.maximum(0, min_dists[triu_idx] - dists[triu_idx])
    
    penalty = 100.0 * (np.sum(bound_viol) + np.sum(overlaps))
    value = -np.sum(radii) + penalty
    return value

def _construct_from_params(params, aspect):
    """Convert params to (n, 3) array of (x, y, r)."""
    n = len(params) // 3
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    result = np.zeros((n, 3))
    result[:, 0] = pos[:, 0]
    result[:, 1] = pos[:, 1]
    result[:, 2] = radii
    return result

def _validate_and_scale(result, aspect):
    """Validate the packing and scale if needed to ensure perimeter <= 4."""
    n = result.shape[0]
    if n == 0:
        return None
    
    r = aspect
    h = 2.0 / (1.0 + r)
    w = 2.0 * r / (1.0 + r)
    
    pos = result[:, :2]
    radii = result[:, 2]
    
    # Check if all circles are inside the rectangle
    if np.any(radii < 0):
        radii = np.abs(radii)
    
    # Find max extent
    x_min = np.min(pos[:, 0] - radii)
    x_max = np.max(pos[:, 0] + radii)
    y_min = np.min(pos[:, 1] - radii)
    y_max = np.max(pos[:, 1] + radii)
    
    actual_w = x_max - x_min
    actual_h = y_max - y_min
    
    if actual_w <= 0 or actual_h <= 0:
        return None
    
    # The bounding rectangle of the circles has dimensions actual_w x actual_h
    # Perimeter = 2*(actual_w + actual_h)
    perim = 2.0 * (actual_w + actual_h)
    
    if perim <= 4.0 + 1e-9:
        return result
    
    # Scale down to fit
    scale = 4.0 / perim
    # Shift to origin then scale
    pos_centered = pos - np.array([x_min, y_min])
    pos_scaled = pos_centered * scale
    radii_scaled = radii * scale
    
    result_scaled = np.zeros((n, 3))
    result_scaled[:, 0] = pos_scaled[:, 0]
    result_scaled[:, 1] = pos_scaled[:, 1]
    result_scaled[:, 2] = radii_scaled
    
    return result_scaled

def _optimize_for_aspect(aspect, init_points, deadline):
    """Run optimization for a given aspect ratio and initial configuration."""
    n = init_points.shape[0]
    
    # Initial radii: estimate from nearest neighbor distances
    from scipy.spatial import cKDTree
    tree = cKDTree(init_points)
    d, _ = tree.query(init_points, k=2)
    min_dists = d[:, 1]
    init_radii = 0.5 * min_dists * 0.95  # slightly conservative
    
    # Build initial params
    r = aspect
    h = 2.0 / (1.0 + r)
    w = 2.0 * r / (1.0 + r)
    
    # Scale initial points to fit in [0,w] x [0,h]
    pts_x = init_points[:, 0]
    pts_y = init_points[:, 1]
    
    # Normalize to [0,1] first
    if pts_x.max() > pts_x.min():
        pts_x_norm = (pts_x - pts_x.min()) / (pts_x.max() - pts_x.min())
    else:
        pts_x_norm = np.full(n, 0.5)
    if pts_y.max() > pts_y.min():
        pts_y_norm = (pts_y - pts_y.min()) / (pts_y.max() - pts_y.min())
    else:
        pts_y_norm = np.full(n, 0.5)
    
    init_pos = np.column_stack([pts_x_norm * w, pts_y_norm * h])
    
    # Recompute radii based on scaled positions
    tree2 = cKDTree(init_pos)
    d2, _ = tree2.query(init_pos, k=2)
    min_dists2 = d2[:, 1]
    init_radii = 0.5 * min_dists2 * 0.9
    
    init_params = np.concatenate([init_pos.flatten(), init_radii])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
    for i in range(n):
        bounds.append((0, min(w, h) / 2))
    
    best_obj = _objective(init_params, n, aspect)
    best_params = init_params.copy()
    
    # Use L-BFGS-B with penalty method
    try:
        res = minimize(
            _objective,
            init_params,
            args=(n, aspect),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Second pass with tighter penalty
    if time.time() < deadline:
        try:
            # Reduce penalty by re-optimizing
            res2 = minimize(
                _objective,
                best_params,
                args=(n, aspect),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    result = _construct_from_params(best_params, aspect)
    result = _validate_and_scale(result, aspect)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_result = None
    best_value = -np.inf
    
    # Different grid configurations to try
    configs = [
        (5, 4),  # 5 rows, 4 cols = 20, add 1
        (4, 5),  # 4 rows, 5 cols = 20, add 1
        (5, 5),  # 5x5 = 25, use 21
        (3, 7),  # 3x7 = 21
        (7, 3),  # 7x3 = 21
        (6, 4),  # 6x4 = 24, use 21
        (4, 6),  # 4x6 = 24, use 21
    ]
    
    # Aspect ratios to explore
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    for aspect in aspect_ratios:
        if time.time() >= deadline - 5:
            break
        
        r = aspect
        h = 2.0 / (1.0 + r)
        w = 2.0 * r / (1.0 + r)
        
        for (n_rows, n_cols) in configs:
            if time.time() >= deadline - 3:
                break
            
            total = n_rows * n_cols
            if total < N:
                continue
            
            # Generate hex grid
            try:
                if (n_rows + n_cols) % 2 == 0 or True:  # always try hex
                    init_pts = _make_hex_grid(n_rows, n_cols, 1.0)
                else:
                    init_pts = _make_rect_grid(n_rows, n_cols, 1.0)
            except Exception:
                init_pts = _make_rect_grid(n_rows, n_cols, 1.0)
            
            # Take first N points (or randomly sample if more)
            if len(init_pts) > N:
                # Use a deterministic subset: pick the most "central" points
                # Or just take first N
                idx = np.random.RandomState(42).choice(len(init_pts), N, replace=False)
                init_pts = init_pts[idx]
            elif len(init_pts) < N:
                continue
            
            result = _optimize_for_aspect(aspect, init_pts, deadline)
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result.copy()
    
    # If we didn't find anything, return a basic valid packing
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Scale to fit perimeter 4
        pos = best_result[:, :2]
        radii = best_result[:, 2]
        x_min, y_min = pos.min(axis=0)
        x_max, y_max = pos.max(axis=0)
        w = x_max - x_min
        h = y_max - y_min
        perim = 2 * (w + h)
        if perim > 4:
            scale = 4.0 / perim
            pos = (pos - np.array([x_min, y_min])) * scale
            radii = radii * scale
            best_result[:, :2] = pos
            best_result[:, 2] = radii
    
    return best_result
# EVOLVE-BLOCK-END
