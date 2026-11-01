"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with fixed cell assignments."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _construct_from_grid(rows, cols, w, h, active_mask=None):
    """
    Construct circles from grid dimensions.
    rows, cols: grid dimensions
    w: array of column widths (size cols)
    h: array of row heights (size rows)
    active_mask: boolean array of shape (rows, cols), True if circle is present
    Returns: (x, y, r) arrays of length N
    """
    if active_mask is None:
        active_mask = np.ones((rows, cols), dtype=bool)
    
    if not np.allclose(np.sum(w), w.sum()): # just a sanity check, though w is passed
        pass
        
    x_list = []
    y_list = []
    r_list = []
    
    x_offset = 0.0
    for j in range(cols):
        y_offset = 0.0
        for i in range(rows):
            if active_mask[i, j]:
                cx = x_offset + w[j] / 2.0
                cy = y_offset + h[i] / 2.0
                r = min(w[j] / 2.0, h[i] / 2.0)
                x_list.append(cx)
                y_list.append(cy)
                r_list.append(r)
            y_offset += h[i]
        x_offset += w[j]
        
    return np.array(x_list), np.array(y_list), np.array(r_list)

def _optimize_grid_layout(rows, cols, W, H, active_mask=None):
    """
    Optimize grid line positions for given W, H and topology.
    """
    if active_mask is None:
        active_mask = np.ones((rows, cols), dtype=bool)
        
    n_active = np.sum(active_mask)
    if n_active != N:
        return None

    # Variables:
    # w[0..cols-2], h[0..rows-2]
    # w[cols-1] = W - sum(w[0..cols-2])
    # h[rows-1] = H - sum(h[0..rows-2])
    
    n_vars = (cols - 1) + (rows - 1)
    
    def objective(z):
        w_vars = z[:cols-1]
        h_vars = z[cols-1:]
        
        w = np.empty(cols)
        h = np.empty(rows)
        
        w[:cols-1] = w_vars
        w[cols-1] = W - np.sum(w_vars)
        
        h[:rows-1] = h_vars
        h[rows-1] = H - np.sum(h_vars)
        
        # Check positivity
        if np.any(w <= 0) or np.any(h <= 0):
            return 1e6
            
        # Calculate sum of radii
        total_r = 0.0
        for i in range(rows):
            for j in range(cols):
                if active_mask[i, j]:
                    total_r += min(w[j] / 2.0, h[i] / 2.0)
                    
        return -total_r

    # Initial guess: uniform
    w_init = np.full(cols - 1, W / cols)
    h_init = np.full(rows - 1, H / rows)
    z0 = np.concatenate([w_init, h_init])
    
    # Bounds: small positive to large
    bounds = [(1e-6, W), (1e-6, H)] * n_vars # Rough bounds, SLSQP will handle
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
        
        if res.success or res.fun < objective(z0):
            z_opt = res.x
        else:
            z_opt = z0
            
        w_vars = z_opt[:cols-1]
        h_vars = z_opt[cols-1:]
        
        w = np.empty(cols)
        h = np.empty(rows)
        
        w[:cols-1] = w_vars
        w[cols-1] = W - np.sum(w_vars)
        
        h[:rows-1] = h_vars
        h[rows-1] = H - np.sum(h_vars)
        
        if np.any(w <= 0) or np.any(h <= 0):
            return None
            
        x, y, r = _construct_from_grid(rows, cols, w, h, active_mask)
        return x, y, r, float(np.sum(r))
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_value = 0.0
    best_result = None
    
    # Topologies to try
    # (rows, cols, active_mask)
    topologies = []
    
    # 3x7 full
    topologies.append((3, 7, np.ones((3, 7), dtype=bool)))
    # 7x3 full
    topologies.append((7, 3, np.ones((7, 3), dtype=bool)))
    
    # 5x5 with 4 empty (e.g., corners)
    mask_55 = np.ones((5, 5), dtype=bool)
    # Remove 4 corners to make it symmetric? Or just remove any 4.
    # Let's remove (0,0), (0,4), (4,0), (4,4)
    mask_55[0,0] = False
    mask_55[0,4] = False
    mask_55[4,0] = False
    mask_55[4,4] = False
    if np.sum(mask_55) == N:
        topologies.append((5, 5, mask_55))
        
    # 4x6 with 3 empty
    mask_46 = np.ones((4, 6), dtype=bool)
    # Remove 3 from one corner
    mask_46[0,0] = False
    mask_46[0,1] = False
    mask_46[1,0] = False
    if np.sum(mask_46) == N:
        topologies.append((4, 6, mask_46))
        
    # 6x4 with 3 empty
    mask_64 = np.ones((6, 4), dtype=bool)
    mask_64[0,0] = False
    mask_64[0,1] = False
    mask_64[1,0] = False
    if np.sum(mask_64) == N:
        topologies.append((6, 4, mask_64))

    # Aspect ratios to try
    # W is width, H = 2 - W
    # Square is W=1, H=1
    # We should sample W in [0.4, 1.6]
    
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for W in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        H = 2.0 - W
        
        for rows, cols, mask in topologies:
            if time.time() > deadline - 5:
                break
                
            result = _optimize_grid_layout(rows, cols, W, H, mask)
            
            if result is not None:
                x, y, r, val = result
                
                # Quick validity check
                # Since constructed from grid, they should be valid.
                # But let's verify bounds just in case of numerical issues
                if np.all(x >= 0) and np.all(x <= W) and np.all(y >= 0) and np.all(y <= H):
                    if val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), W)
                        
    # Fallback if no result found (should not happen)
    if best_result is None:
        # Simple 3x7 grid with W=1, H=1
        W, H = 1.0, 1.0
        w = np.full(7, W/7)
        h = np.full(3, H/3)
        mask = np.ones((3, 7), dtype=bool)
        x, y, r = _construct_from_grid(3, 7, w, h, mask)
        best_result = (x, y, r, W)
        
    x, y, r, W = best_result
    H = 2.0 - W
    
    # Final safety shrink
    for _ in range(5):
        r_new = r * (1 - 1e-9)
        # Check validity
        valid = True
        for i in range(N):
            if x[i] < r_new[i] or x[i] > W - r_new[i] or y[i] < r_new[i] or y[i] > H - r_new[i]:
                valid = False
                break
        if not valid:
            break
        r = r_new
        
    return np.column_stack((x, y, r))
