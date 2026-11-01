import numpy as np
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Hexagonal lattice
    # We want to pack 32 circles. A hexagonal grid is denser than square.
    # Approximate number of rows/cols.
    # For N=32, roughly 6x6 grid minus corners or similar.
    # Let's generate a hex grid and pick the first 32 points that fit in a bounding box, then scale.
    
    # Hex grid parameters
    # Distance between centers in hex grid is 1 unit for now.
    # x spacing: 1.0
    # y spacing: sqrt(3)/2
    # Offset for odd rows: 0.5
    
    points = []
    # Generate enough points
    for row in range(10):
        for col in range(10):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            points.append((x, y))
    
    # Sort by distance from center of this set to get a compact cluster?
    # Or just take the first N.
    # Better: Center the hex grid points.
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    
    # Select the N points closest to the center to form a compact cluster
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    init_pts = pts[idx]
    
    # Scale to fit in unit square with some margin
    # Current extent
    min_x, min_y = init_pts.min(axis=0)
    max_x, max_y = init_pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to fit these into [0,1]x[0,1]
    # The relative positions are preserved.
    # Scale factor
    scale = 0.9 / max(width, height) # 0.9 leaves margin for radii
    
    init_pts = (init_pts - min_x) * scale
    init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
    
    # Center in unit square
    current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
    offset = (1.0 - current_w) / 2
    init_pts += offset
    
    # Initial radii: small, based on min distance
    # Calculate min distance between any pair
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(init_pts[i] - init_pts[j])
            if d < min_dist:
                min_dist = d
    init_r = min_dist / 2 * 0.9
    
    # Variables: x_1, y_1, r_1, x_2, y_2, r_2, ...
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = init_pts[i, 0]
        x0[3*i+1] = init_pts[i, 1]
        x0[3*i+2] = init_r
        
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-5, 0.5)) # r
    
    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        pos = vars[0:3*N:3]
        rad = vars[2:3*N:3]
        
        # Objective: -sum(r)
        obj = -np.sum(rad)
        
        # Penalty for overlaps
        # d_ij >= r_i + r_j
        # violation = max(0, r_i + r_j - d_ij)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                overlap = rad[i] + rad[j] - d
                if overlap > 0:
                    pen += overlap**2
        
        # Penalty for boundary violations
        # x >= r, 1-x >= r, y >= r, 1-y >= r
        for i in range(N):
            d_left = pos[i, 0] - rad[i]
            d_right = 1.0 - pos[i, 0] - rad[i]
            d_bottom = pos[i, 1] - rad[i]
            d_top = 1.0 - pos[i, 1] - rad[i]
            
            if d_left < 0: pen += d_left**2
            if d_right < 0: pen += d_right**2
            if d_bottom < 0: pen += d_bottom**2
            if d_top < 0: pen += d_top**2
            
        # Weight for penalty
        weight = 100.0
        return obj + weight * pen

    def objective_grad(vars):
        # Analytical gradient is complex due to max(0, ...), 
        # but L-BFGS-B can use numerical gradient if not provided.
        # However, providing gradient is faster.
        # Let's stick to numerical gradient for robustness and simplicity, 
        # or implement a smooth approximation.
        # Given N=32, numerical gradient might be slow.
        # Let's implement the gradient for the smooth parts and approximate the penalty.
        
        pos = vars[0:3*N:3]
        rad = vars[2:3*N:3]
        
        grad = np.zeros_like(vars)
        
        # Gradient of -sum(r)
        grad[2:3*N:3] = -1.0
        
        # Gradient of penalty
        weight = 100.0
        
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                overlap = rad[i] + rad[j] - d
                if overlap > 0:
                    # d(pen)/d(pos_i) = 2 * overlap * d(-d)/d(pos_i) = 2 * overlap * (d_vec / d)
                    # d(pen)/d(pos_j) = 2 * overlap * d(-d)/d(pos_j) = 2 * overlap * (-d_vec / d)
                    # d(pen)/d(r_i) = 2 * overlap * 1
                    # d(pen)/d(r_j) = 2 * overlap * 1
                    
                    factor = 2.0 * overlap / d
                    grad[3*i] += weight * factor * d_vec[0]
                    grad[3*i+1] += weight * factor * d_vec[1]
                    grad[3*j] -= weight * factor * d_vec[0]
                    grad[3*j+1] -= weight * factor * d_vec[1]
                    grad[3*i+2] += weight * 2.0 * overlap
                    grad[3*j+2] += weight * 2.0 * overlap

        # Boundary
        for i in range(N):
            # Left: x_i - r_i >= 0
            val = pos[i, 0] - rad[i]
            if val < 0:
                grad[3*i] -= weight * 2.0 * val
                grad[3*i+2] += weight * 2.0 * val
            
            # Right: 1 - x_i - r_i >= 0
            val = 1.0 - pos[i, 0] - rad[i]
            if val < 0:
                grad[3*i] += weight * 2.0 * val
                grad[3*i+2] += weight * 2.0 * val
            
            # Bottom: y_i - r_i >= 0
            val = pos[i, 1] - rad[i]
            if val < 0:
                grad[3*i+1] -= weight * 2.0 * val
                grad[3*i+2] += weight * 2.0 * val
            
            # Top: 1 - y_i - r_i >= 0
            val = 1.0 - pos[i, 1] - rad[i]
            if val < 0:
                grad[3*i+1] += weight * 2.0 * val
                grad[3*i+2] += weight * 2.0 * val

        return grad

    # Run optimization
    # Use L-BFGS-B for bound-constrained optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=objective_grad, bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        res = None
    
    if res is not None and res.success:
        x_opt = res.x
    else:
        x_opt = x0
        
    # 2. Polish: Ensure strict feasibility
    # Extract positions and radii
    pos = x_opt[0:3*N:3].copy()
    rad = x_opt[2:3*N:3].copy()
    
    # Iterate a few times to tighten radii
    for _ in range(10):
        for i in range(N):
            # Max radius for i given others
            min_dist_to_others = np.inf
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(pos[i] - pos[j])
                # We need d >= r_i + r_j => r_i <= d - r_j
                min_dist_to_others = min(min_dist_to_others, d - rad[j])
            
            # Boundary constraints
            min_dist_to_boundary = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            
            max_r = min(min_dist_to_others, min_dist_to_boundary)
            # Add small epsilon to avoid touching
            rad[i] = max(1e-6, max_r - 1e-5)
            
    # Final check and construction
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = pos[i, 0]
        result[i, 1] = pos[i, 1]
        result[i, 2] = rad[i]
        
    return result
# EVOLVE-BLOCK-END
