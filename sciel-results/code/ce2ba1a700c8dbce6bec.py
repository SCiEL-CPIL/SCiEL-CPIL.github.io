"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

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
    if np.any(centers[:, 0] < -1e-6) or np.any(centers[:, 0] > 1 + 1e-6):
        return False
    if np.any(centers[:, 1] < -1e-6) or np.any(centers[:, 1] > 1 + 1e-6):
        return False
    if np.any(radii < -1e-6):
        return False
    # Check overlaps
    d = _pairwise_distances(centers)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < radii[i] + radii[j] - 1e-8:
                return False
    return True

def _pack_for_aspect(ar, deadline):
    """Pack 21 circles in a rectangle with given aspect ratio."""
    w = 2.0 * ar / (1.0 + ar)
    h = 2.0 / (1.0 + ar)
    
    # Try multiple grid configurations
    best_local = None
    best_local_score = -np.inf
    
    # Grid configurations: (cols, rows)
    # For 21 circles, good options: 3x7, 7x3, 4x5(+1), 5x4(+1), 6x3(+3), etc.
    grid_configs = [(3, 7), (7, 3), (4, 5), (5, 4), (6, 3), (3, 6), (5, 3), (3, 5)]
    
    for cols, rows in grid_configs:
        if time.time() > deadline:
            break
        
        total_cells = cols * rows
        if total_cells < N:
            continue
        
        # Generate grid positions
        xs = np.linspace(1.0/(2*cols), 1.0 - 1.0/(2*cols), cols)
        ys = np.linspace(1.0/(2*rows), 1.0 - 1.0/(2*rows), rows)
        
        # Get first N positions
        centers_norm = []
        for i in range(cols):
            for j in range(rows):
                if len(centers_norm) < N:
                    centers_norm.append([xs[i], ys[j]])
        
        centers_norm = np.array(centers_norm)
        
        # Initial radius estimate
        dx = xs[1] - xs[0] if cols > 1 else 1.0
        dy = ys[1] - ys[0] if rows > 1 else 1.0
        dist_x = dx * w
        dist_y = dy * h
        min_dist = min(dist_x, dist_y)
        r_init = min_dist / 2.0 * 0.95
        
        # Parameters: [x_norm, y_norm, r] for each circle
        x0 = np.column_stack([centers_norm, np.full(N, r_init)])
        
        # Define objective and constraints for SLSQP
        def objective(p):
            """Minimize negative sum of radii."""
            return -np.sum(p[:, 2])
        
        def constraints_overlap(p):
            """Constraint: no overlaps. Returns array of (dist - r_i - r_j) for all pairs."""
            cn = p[:, :2]
            r = p[:, 2]
            centers = np.column_stack([cn[:, 0] * w, cn[:, 1] * h])
            d = _pairwise_distances(centers)
            cons = []
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append(d[i, j] - r[i] - r[j])
            return np.array(cons)
        
        def constraints_radius_nonneg(p):
            """Constraint: radii >= 0."""
            return p[:, 2]
        
        # Bounds: x in [0,1], y in [0,1], r in [0, max_r]
        max_r = min(w, h) / 2.0
        bounds = []
        for _ in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, max_r))  # r
        
        # Flatten parameters
        x0_flat = x0.flatten()
        
        try:
            # Use SLSQP with constraints
            result = minimize(
                objective,
                x0_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: constraints_overlap(p.reshape(N, 3))},
                    {'type': 'ineq', 'fun': lambda p: constraints_radius_nonneg(p.reshape(N, 3))}
                ],
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            if result.success:
                x_opt = result.x.reshape(N, 3)
                cn = x_opt[:, :2]
                r = x_opt[:, 2]
                
                # Validate
                if _validate(cn * np.array([w, h]), r):
                    score = np.sum(r)
                    if score > best_local_score:
                        best_local_score = score
                        best_local = np.column_stack([cn[:, 0] * w, cn[:, 1] * h, r])
                        
        except Exception:
            pass
    
    return best_local, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20 second margin for safety
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        result, score = _pack_for_aspect(ar, deadline)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid in unit square (perimeter = 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
