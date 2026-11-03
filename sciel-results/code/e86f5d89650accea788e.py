"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _pairwise_distances(centers):
    """Compute pairwise distances for n circles. Returns (n, n) matrix."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def _validate(centers, radii):
    """Check if circles are valid (non-overlapping, within bounds)."""
    n = len(centers)
    # Check bounds
    if np.any(centers[:, 0] < 0) or np.any(centers[:, 0] > 1):
        return False
    if np.any(centers[:, 1] < 0) or np.any(centers[:, 1] > 1):
        return False
    if np.any(radii < 0):
        return False
    # Check overlaps
    d = _pairwise_distances(centers)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < radii[i] + radii[j] - 1e-9:
                return False
    return True

def _objective(params, aspect_ratio):
    """
    Objective: maximize sum of radii.
    params: array of shape (n, 3) = [x, y, r] for each circle
    aspect_ratio: width/height of rectangle (height = 2/(1+aspect), width = 2*aspect/(1+aspect))
    Actually, perimeter = 2*(w+h) = 4, so w+h = 2.
    Let aspect = w/h, so w = aspect*h, aspect*h + h = 2, h = 2/(aspect+1), w = 2*aspect/(aspect+1).
    We normalize coordinates to [0,1]x[0,1] internally, then scale.
    Actually, let's work in the actual rectangle coordinates for clarity, 
    or normalize. Let's normalize x to [0,1] and y to [0,1] representing the unit square,
    then the actual rectangle is [0, w] x [0, h].
    So actual_x = x * w, actual_y = y * h.
    Distance in actual coords: sqrt((dx*w)^2 + (dy*h)^2).
    Constraint: dist >= r_i + r_j.
    """
    w = 2.0 * aspect_ratio / (1.0 + aspect_ratio)
    h = 2.0 / (1.0 + aspect_ratio)
    
    # params are normalized coords [0,1] and radii in actual units? 
    # No, let's keep radii in actual units to make objective direct.
    # But scaling radii with w/h makes it messy.
    # Let's define params as [x_norm, y_norm, r_actual].
    # Objective: sum(r_actual)
    
    centers_norm = params[:, :2]
    radii = params[:, 2]
    
    # Convert to actual coords
    centers = np.column_stack([centers_norm[:, 0] * w, centers_norm[:, 1] * h])
    
    # Check validity
    if np.any(centers_norm < -1e-6) or np.any(centers_norm > 1 + 1e-6):
        return -1e10
    if np.any(radii < -1e-6):
        return -1e10
        
    d = _pairwise_distances(centers)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if d[i, j] < radii[i] + radii[j] - 1e-7:
                return -1e10
    
    return np.sum(radii)

def _nelder_mead_minimize(objective, x0, bounds=None, max_iter=1000, tol=1e-8):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i+1] = x0.copy()
        simplex[i+1][i] += 0.1 * (bounds[i][1] - bounds[i][0]) if bounds else 0.1
    
    f = [objective(x) for x in simplex]
    
    for it in range(max_iter):
        # Sort simplex
        idx = np.argsort(f)
        simplex = [simplex[i] for i in idx]
        f = [f[i] for i in idx]
        
        if abs(f[-1] - f[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f[-1] = fe
            else:
                simplex[-1] = xr
                f[-1] = fr
        elif fr < f[-2]:
            simplex[-1] = xr
            f[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < f[-1]:
                simplex[-1] = xc
                f[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f[i] = objective(simplex[i])
    
    return simplex[0], f[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.66, 1.5, 0.5, 2.0, 0.75, 1.33]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Initial guess: grid layout
        # Determine grid size
        # For 21 circles, try 4x5 or 3x7 or other
        grid_configs = [(4, 5), (5, 4), (3, 7), (7, 3), (2, 10.5), (10.5, 2), (1, 21), (21, 1), (6, 3.5)]
        
        for gx, gy in grid_configs:
            if time.time() > deadline:
                break
            
            # Create initial positions
            xs = np.linspace(0.5/gx, 1 - 0.5/gx, int(np.ceil(gx))) if gx % 1 != 0 else np.linspace(0.5/gx, 1 - 0.5/gx, int(gx))
            ys = np.linspace(0.5/gy, 1 - 0.5/gy, int(np.ceil(gy))) if gy % 1 != 0 else np.linspace(0.5/gy, 1 - 0.5/gy, int(gy))
            
            # Simpler: just use a grid of 5x5 and take first 21
            # Or better, generate a grid that fits 21
            # Let's use a 4x5 grid (20) + 1 center, or 5x4
            # Actually, let's just use a uniform grid of 21 points
            # 4 cols, 5 rows = 20, add 1 in center?
            # Let's try 5x5 grid, take 21
            side = 5
            xs_grid = np.linspace(0.5/side, 1 - 0.5/side, side)
            ys_grid = np.linspace(0.5/side, 1 - 0.5/side, side)
            
            centers_norm = []
            for i in range(side):
                for j in range(side):
                    centers_norm.append([xs_grid[i], ys_grid[j]])
            centers_norm = np.array(centers_norm[:21])
            
            # Initial radii: small
            # Estimate max radius based on grid spacing
            dx = xs_grid[1] - xs_grid[0] if side > 1 else 1.0
            dy = ys_grid[1] - ys_grid[0] if side > 1 else 1.0
            # Actual distance between neighbors
            dist_x = dx * w
            dist_y = dy * h
            min_dist = min(dist_x, dist_y)
            r_init = min_dist / 2.0 * 0.9  # slightly less to avoid overlap
            
            params = np.column_stack([centers_norm, np.full(21, r_init)])
            
            # Optimize
            # We want to maximize sum(r), so minimize -sum(r)
            # But we have constraints. Let's use a penalty method or just NM on the objective with penalty.
            
            def obj_penalized(p):
                # Penalty for constraint violations
                cn = p[:, :2]
                r = p[:, 2]
                
                # Bounds penalty
                pen = 0
                if np.any(cn < 0) or np.any(cn > 1):
                    pen += 1e5 * (np.sum(np.maximum(0, -cn)) + np.sum(np.maximum(0, cn - 1)))
                if np.any(r < 0):
                    pen += 1e5 * np.sum(np.maximum(0, -r))
                
                # Overlap penalty
                centers = np.column_stack([cn[:, 0] * w, cn[:, 1] * h])
                d = _pairwise_distances(centers)
                for i in range(21):
                    for j in range(i + 1, 21):
                        overlap = r[i] + r[j] - d[i, j]
                        if overlap > 0:
                            pen += 1e5 * overlap ** 2
                
                return -np.sum(r) + pen
            
            # Run Nelder-Mead
            try:
                x_opt, f_opt = _nelder_mead_minimize(obj_penalized, params, max_iter=2000)
                
                if time.time() > deadline:
                    break
                
                # Check if valid
                cn = x_opt[:, :2]
                r = x_opt[:, 2]
                if _validate(cn * np.array([w, h]), r):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        # Store in normalized coords for consistent output? 
                        # The task says "array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
                        # Usually this means [x, y, r] in the actual rectangle coordinates.
                        # Let's return actual coordinates.
                        best_result = np.column_stack([cn[:, 0] * w, cn[:, 1] * h, r])
                        
            except Exception as e:
                pass
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
