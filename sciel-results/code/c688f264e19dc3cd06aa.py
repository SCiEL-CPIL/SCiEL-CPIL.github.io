import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def soft_min_max_ratio(points, beta):
    """
    Compute a smooth surrogate for log((d_min / d_max)^2).
    Returns 2*log(d_min_soft) - 2*log(d_max_soft) for optimization stability.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return -1e6
    
    x_max = dists / beta
    m_max = np.max(x_max)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(x_max - m_max)))
    d_max_soft = beta * log_sum_exp_max
    
    x_min = -dists / beta
    m_min = np.max(x_min)
    log_sum_exp_min = m_min + np.log(np.sum(np.exp(x_min - m_min)))
    d_min_soft = -beta * log_sum_exp_min
    
    if d_max_soft <= 0 or d_min_soft <= 0:
        return -1e6
        
    log_ratio = np.log(d_min_soft) - np.log(d_max_soft)
    return 2 * log_ratio

def points_from_params(params):
    """
    Convert 28-dim param vector to 16x2 points.
    Params: first 2 fixed at (0,0) and (1,0), remaining 28 are free coords.
    """
    n = 16
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    rest = params.reshape(14, 2)
    pts[2:] = rest
    return pts

def points_to_params(pts):
    """
    Convert 16x2 points to 28-dim params, assuming we can identify two points
    to serve as the fixed reference (0,0) and (1,0).
    We pick the two closest points as the reference pair.
    """
    dists = spd.pdist(pts)
    if len(dists) == 0:
        return None
    min_idx = np.argmin(dists)
    # Convert linear index to pair (i, j) for 16 points
    # pdist for n points gives indices: (0,1),(0,2),...,(0,15),(1,2),...
    # Linear index k corresponds to pair (i,j) where:
    # i = floor((-1 + sqrt(1+8*(k+1)*n-8))/2) ... complex. Let's just enumerate.
    pairs = []
    idx = 0
    for i in range(15):
        for j in range(i+1, 16):
            if idx == min_idx:
                break
            idx += 1
    else:
        return None
    i, j = pairs[-1] if pairs else (0, 1)
    # Actually let me just find it properly
    idx = 0
    i_found = j_found = -1
    for i in range(15):
        for j in range(i+1, 16):
            if idx == min_idx:
                i_found, j_found = i, j
                break
            idx += 1
        if i_found >= 0:
            break
    if i_found < 0:
        return None
    
    p0 = pts[i_found].copy()
    p1 = pts[j_found].copy()
    
    # Create a copy with i_found at position 0 and j_found at position 1
    new_pts = pts.copy()
    # Move i_found to 0
    if i_found != 0:
        new_pts[0] = new_pts[i_found]
        # Shift elements between 0 and i_found down
        for k in range(i_found, 0, -1):
            new_pts[k] = new_pts[k-1]
    # Now move j_found to 1 (accounting for possible shift)
    if j_found != 1:
        if j_found == i_found:
            # Shouldn't happen since they're different
            return None
        if j_found > i_found:
            j_found -= 1  # Account for the shift
        if j_found != 1:
            new_pts[1] = new_pts[j_found]
            for k in range(j_found, 1, -1):
                new_pts[k] = new_pts[k-1]
    
    # Now new_pts[0] = p0, new_pts[1] = p1
    p0 = new_pts[0]
    p1 = new_pts[1]
    
    pts_trans = new_pts - p0
    d = np.linalg.norm(p1)
    
    if d < 1e-9:
        return None
    
    angle = np.arctan2(p1[1], p1[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    pts_rot = pts_trans @ rot.T
    pts_scaled = pts_rot / d
    
    # Verify pts_scaled[1] is close to (1,0)
    if not np.allclose(pts_scaled[1], [1.0, 0.0], atol=1e-6):
        return None
        
    params = pts_scaled[2:].flatten()
    return params

def model_1_6_9(params):
    """
    Model: 1 center + 6 ring + 9 (3 basis x 3-fold symmetry) = 16 points
    Params (8): r1, theta1, x1,y1, x2,y2, x3,y3
    """
    r1 = params[0]
    theta1 = params[1]
    
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + theta1
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    
    basis = params[2:8].reshape(3, 2)
    
    pts_r2 = []
    for k in range(3):
        rot_angle = k * 2 * np.pi / 3
        c = np.cos(rot_angle)
        s = np.sin(rot_angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        for p in basis:
            new_p = rot_matrix @ p
            pts_r2.append(new_p)
            
    pts_r2 = np.array(pts_r2)
    
    center = np.array([[0.0, 0.0]])
    
    x_all = np.concatenate([x1, pts_r2[:, 0], [0.0]])
    y_all = np.concatenate([y1, pts_r2[:, 1], [0.0]])
    
    return np.column_stack((x_all, y_all))

def model_1_5_10(params):
    """
    Model: 1 center + 5 ring + 10 ring = 16 points
    Params (4): r1, theta1, r2, theta2
    """
    r1 = params[0]
    theta1 = params[1]
    r2 = params[2]
    theta2 = params[3]
    
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False) + theta1
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    
    angles2 = np.linspace(0, 2*np.pi, 10, endpoint=False) + theta2
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    
    center = np.array([0.0, 0.0])
    
    x_all = np.concatenate([x1, x2, [0.0]])
    y_all = np.concatenate([y1, y2, [0.0]])
    
    return np.column_stack((x_all, y_all))

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    
    global_deadline = time.time() + 620  # Safety margin (630s limit)
    
    def objective_smooth(params, beta):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            return -soft_min_max_ratio(pts, beta)
        except Exception:
            return 1e6

    def objective_exact(params):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            score = calculate_score(pts)
            return -score
        except Exception:
            return 1e6

    def objective_full(x):
        """Objective for full 32-dim space (no constraints)."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return -calculate_score(pts)

    best_params = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Hexagonal lattice-based init
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    d_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(d_center)[:16]
    hex_init = hex_pts[idx]
    candidates.append(hex_init)
    
    # 2. 1+6+9 model init
    basis_hex = np.array([
        [2.0, 0.0],
        [1.5, np.sqrt(3)/2],
        [0.0, np.sqrt(3)]
    ])
    params_hex = np.array([
        1.0, 0.0,
        basis_hex[0,0], basis_hex[0,1],
        basis_hex[1,0], basis_hex[1,1],
        basis_hex[2,0], basis_hex[2,1]
    ])
    hex_model_pts = model_1_6_9(params_hex)
    candidates.append(hex_model_pts)
    
    # 3. 1+5+10 model init
    params_510 = np.array([1.0, 0.0, 1.8, np.pi/10])
    ring_510_pts = model_1_5_10(params_510)
    candidates.append(ring_510_pts)
    
    # 4. 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(grid_points)
    
    # 5. Random
    rand_points = rng.randn(16, 2) * 2.0
    candidates.append(rand_points)
    
    # Convert each candidate to param space for optimization
    inits = []
    for cand in candidates:
        params = points_to_params(cand)
        if params is not None:
            inits.append(params)
    
    # Add some random inits in param space
    for _ in range(3):
        inits.append(rng.randn(28) * 2.0)
    
    start_time = time.time()
    
    # Phase 1: Smooth surrogate optimization with L-BFGS-B
    betas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    current_params = None
    
    for beta in betas:
        if time.time() > global_deadline - 120:
            break
            
        best_local_params = None
        best_local_score = 1e6
        
        for init in inits:
            if time.time() > global_deadline - 120:
                break
                
            try:
                res = minimize(
                    objective_smooth, init, args=(beta,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res.fun < best_local_score:
                    best_local_score = res.fun
                    best_local_params = res.x
            except Exception:
                pass
        
        if best_local_params is not None:
            current_params = best_local_params
            pts = points_from_params(current_params)
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
            
            # Keep best as next init, trim list
            inits.insert(0, best_local_params)
            if len(inits) > 4:
                inits = inits[:4]

    # Phase 2: Exact Nelder-Mead polish in constrained space
    if time.time() < global_deadline - 80:
        if best_params is not None:
            start_polish = best_params
        else:
            start_polish = inits[0]
            
        try:
            res = minimize(
                objective_exact, start_polish, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() < global_deadline - 40 and best_params is not None:
            try:
                res = minimize(
                    objective_exact, best_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'fatol': 1e-12, 'xatol': 1e-12}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass

    # Phase 3: Full-space polish (remove the constraint that pts[0]=(0,0), pts[1]=(1,0))
    if time.time() < global_deadline - 30 and best_params is not None:
        # Get current best points
        best_pts = points_from_params(best_params)
        
        # Also try polishing the structured model outputs directly
        model_candidates = []
        if best_pts is not None:
            model_candidates.append(best_pts)
        model_candidates.append(hex_model_pts)
        model_candidates.append(ring_510_pts)
        model_candidates.append(hex_init)
        
        for cand_pts in model_candidates:
            if time.time() > global_deadline - 30:
                break
            try:
                x0 = cand_pts.flatten()
                res = minimize(
                    objective_full, x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-10}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_pts = res.x.reshape(n, 2)
            except Exception:
                pass
    
    # Final: ensure we have valid output
    if best_params is not None:
        final_points = points_from_params(best_params)
    else:
        # Fallback: use the best structured candidate
        final_points = hex_init
    
    if not np.all(np.isfinite(final_points)):
        final_points = np.zeros((n, d))
        
    return final_points
