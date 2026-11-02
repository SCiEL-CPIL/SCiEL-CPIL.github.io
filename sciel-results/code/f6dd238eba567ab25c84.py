import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145.0 # Safety margin

def _hex_init(n_points, aspect):
    """
    Generate a hexagonal lattice initialization for n_points circles.
    aspect = w/h
    """
    # Generate a large hex grid
    # We want roughly n_points points
    # Radius in lattice units approx sqrt(n_points)
    R = int(np.sqrt(n_points)) + 2
    points = []
    for j in range(-R, R+1):
        for i in range(-R, R+1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            points.append((x, y))
    
    points = np.array(points)
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Calculate distances from origin and sort
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    # Take the closest n_points
    # But we want them to fit in a rectangle of given aspect
    # So we project them? No, just take the central cluster.
    # The shape of the cluster might not match the aspect.
    # We will scale the coordinates to fit the rectangle in the optimizer.
    
    # Let's just return the coordinates of the closest n_points
    pts = points[idx[:n_points]]
    
    # Normalize to fit in a unit square initially, optimizer will scale
    # Actually, we can just return them and let the optimizer scale positions/radii.
    # But for a good start, we should scale them to roughly fill a box of aspect 'aspect'.
    
    # Estimate bounding box of pts
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    w_pts = max_x - min_x
    h_pts = max_y - min_y
    
    if w_pts == 0: w_pts = 1
    if h_pts == 0: h_pts = 1
    
    # Scale to fit aspect
    # Target aspect = w/h
    # Current aspect = w_pts/h_pts
    # If current aspect < target aspect, scale x more.
    # If current aspect > target aspect, scale y more.
    
    # Let's just center them at 0,0 and let the optimizer place them.
    # The radii will be determined by the optimizer.
    return pts

def construct():
    start_time = time.time()
    best_score = 0.0
    best_x = None
    
    # Objective function
    def objective(v):
        # v = [x1, y1, r1, ..., xN, yN, rN, w]
        x = v[0:N*3:3]
        y = v[1:N*3:3]
        r = v[2:N*3:3]
        w = v[-1]
        
        h = 2.0 - w
        
        # If w or h are non-positive, return large penalty
        if w <= 0 or h <= 0:
            return 1e10
        
        sum_r = np.sum(r)
        
        # Wall penalties
        pen = 0.0
        # x walls
        pen += np.sum(np.maximum(0, -x + r))**2
        pen += np.sum(np.maximum(0, x + r - w))**2
        # y walls
        pen += np.sum(np.maximum(0, -y + r))**2
        pen += np.sum(np.maximum(0, y + r - h))**2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                pen += np.maximum(0, r[i] + r[j] - d)**2
        
        # We want to maximize sum_r, so minimize -sum_r + lambda * pen
        # Choose lambda large enough to enforce constraints
        lam = 100.0
        return -sum_r + lam * pen

    def objective_grad(v):
        # Numerical gradient is slow but robust for this small dim?
        # Analytical is better.
        x = v[0:N*3:3].copy()
        y = v[1:N*3:3].copy()
        r = v[2:N*3:3].copy()
        w = v[-1]
        
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            return np.zeros_like(v)
        
        grad = np.zeros_like(v)
        
        # d(-sum_r)/dr_i = -1
        grad[2:N*3:3] -= 1.0
        
        lam = 100.0
        
        # Wall gradients
        # -x + r > 0
        v1 = -x + r
        mask1 = v1 > 0
        grad[0:N*3:3][mask1] += 2 * lam * v1[mask1] * (-1) # d(-x+r)/dx = -1
        grad[2:N*3:3][mask1] += 2 * lam * v1[mask1] * (1)  # d(-x+r)/dr = 1
        
        # x + r - w > 0
        v2 = x + r - w
        mask2 = v2 > 0
        grad[0:N*3:3][mask2] += 2 * lam * v2[mask2] * (1)  # d(x+r-w)/dx = 1
        grad[2:N*3:3][mask2] += 2 * lam * v2[mask2] * (1)  # d(x+r-w)/dr = 1
        grad[-1] += np.sum(2 * lam * v2[mask2] * (-1)) # d(x+r-w)/dw = -1
        
        # -y + r > 0
        v3 = -y + r
        mask3 = v3 > 0
        grad[1:N*3:3][mask3] += 2 * lam * v3[mask3] * (-1)
        grad[2:N*3:3][mask3] += 2 * lam * v3[mask3] * (1)
        
        # y + r - h > 0
        # h = 2 - w, so dh/dw = -1
        v4 = y + r - h
        mask4 = v4 > 0
        grad[1:N*3:3][mask4] += 2 * lam * v4[mask4] * (1)
        grad[2:N*3:3][mask4] += 2 * lam * v4[mask4] * (1)
        # d(y+r-h)/dw = -dh/dw = 1
        grad[-1] += np.sum(2 * lam * v4[mask4] * (1))
        
        # Overlap gradients
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-12:
                    continue
                pen = r[i] + r[j] - d
                if pen > 0:
                    # d(pen)/dx_i = -dx/d
                    # d(pen)/dx_j = dx/d
                    # d(pen)/dr_i = 1
                    # d(pen)/dr_j = 1
                    factor = 2 * lam * pen
                    ux = dx / d
                    uy = dy / d
                    
                    idx_i_x = i*3
                    idx_i_y = i*3 + 1
                    idx_i_r = i*3 + 2
                    idx_j_x = j*3
                    idx_j_y = j*3 + 1
                    idx_j_r = j*3 + 2
                    
                    grad[idx_i_x] += factor * (-ux)
                    grad[idx_i_y] += factor * (-uy)
                    grad[idx_i_r] += factor * (1)
                    
                    grad[idx_j_x] += factor * (ux)
                    grad[idx_j_y] += factor * (uy)
                    grad[idx_j_r] += factor * (1)
        
        return grad

    # Initialization strategies
    inits = []
    
    # 1. Hexagonal cluster
    pts_hex = _hex_init(N, 1.0)
    # Scale pts_hex to roughly fit in a 1x1 box
    max_extent = np.max(np.abs(pts_hex))
    if max_extent > 0:
        pts_hex = pts_hex / max_extent * 0.5 # Center at 0, scale to [-0.5, 0.5]
    
    # We need to map these 2D points to the variable vector
    # We'll try different aspect ratios and rotations
    
    aspects = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    rotations = [0.0, np.pi/6, np.pi/4, np.pi/3]
    
    for aspect in aspects:
        w_init = 2.0 * aspect / (aspect + 1.0)
        h_init = 2.0 - w_init
        
        for rot in rotations:
            # Rotate points
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s], [s, c]])
            pts_rot = pts_hex @ R.T
            
            # Scale to fit aspect
            # We want the points to span roughly w_init and h_init
            # Current span:
            min_x, max_x = np.min(pts_rot[:, 0]), np.max(pts_rot[:, 0])
            min_y, max_y = np.min(pts_rot[:, 1]), np.max(pts_rot[:, 1])
            span_x = max_x - min_x
            span_y = max_y - min_y
            
            if span_x < 1e-6: span_x = 1e-6
            if span_y < 1e-6: span_y = 1e-6
            
            # Scale factors
            sx = w_init * 0.9 / span_x
            sy = h_init * 0.9 / span_y
            
            # Use min scale to maintain aspect ratio of cluster? 
            # Or fit both? Let's fit both by scaling x and y independently.
            # This distorts the hex lattice, but gives a good starting point for the optimizer.
            
            pts_scaled = pts_rot.copy()
            pts_scaled[:, 0] = (pts_rot[:, 0] - min_x) * sx + w_init * 0.05
            pts_scaled[:, 1] = (pts_rot[:, 1] - min_y) * sy + h_init * 0.05
            
            # Estimate initial radii
            # Average distance to nearest neighbor
            # For now, just use a fraction of min(w,h)
            r_init = 0.1 * min(w_init, h_init)
            
            v0 = np.zeros(N*3 + 1)
            for i in range(N):
                v0[i*3] = pts_scaled[i, 0]
                v0[i*3+1] = pts_scaled[i, 1]
                v0[i*3+2] = r_init
            v0[-1] = w_init
            
            inits.append(v0)
            
            # Check time
            if time.time() - start_time > DEADLINE - 20:
                break
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # Optimize each initialization
    bounds = []
    for i in range(N):
        bounds.append((0, 2.0)) # x
        bounds.append((0, 2.0)) # y
        bounds.append((0, 1.0)) # r
    bounds.append((0.1, 1.9)) # w
    
    for v0 in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        try:
            res = minimize(objective, v0, jac=objective_grad, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < 1e-5: # Check if penalty is low
                v_best = res.x
                # Extract
                x = v_best[0:N*3:3]
                y = v_best[1:N*3:3]
                r = v_best[2:N*3:3]
                w = v_best[-1]
                h = 2.0 - w
                
                # Verify constraints
                valid = True
                # Walls
                if np.any(x + r > w + 1e-6) or np.any(-x + r > 1e-6) or np.any(y + r > h + 1e-6) or np.any(-y + r > 1e-6):
                    valid = False
                # Overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        # Format output: (N, 3) array
                        best_x = np.column_stack((x, y, r))
                        
        except Exception as e:
            pass

    if best_x is None:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        cols = 5
        rows = 5
        # 21 circles in 5x5 grid
        # 25 spots, remove 4 corners?
        # Just place 21 in a 4x5 grid + 1
        # Let's do 5 cols, 5 rows, skip last 4
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        X, Y = np.meshgrid(x, y)
        X = X.ravel()
        Y = Y.ravel()
        R = np.full(25, 0.08)
        
        # Remove 4
        idx_remove = [0, 24, 4, 20] # Corners
        idx_keep = [i for i in range(25) if i not in idx_remove]
        best_x = np.column_stack((X[idx_keep], Y[idx_keep], R[idx_keep]))
        best_score = np.sum(R[idx_keep])

    return best_x
