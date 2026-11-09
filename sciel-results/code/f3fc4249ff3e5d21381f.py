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

def soft_min_max_ratio_and_grad(points, beta):
    """
    Compute a smooth surrogate for log((d_min / d_max)^2) and its gradient.
    Returns (value, gradient).
    Value: 2*log(d_min_soft) - 2*log(d_max_soft)
    """
    n = points.shape[0]
    d = points.shape[1]
    
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Mask diagonal to avoid division by zero in gradients if needed, 
    # though dists[i,i]=0 so exp(0/beta)=1, but we only care about i!=j for pdist equivalent
    # Actually, for the log-sum-exp over all pairs, including diagonal adds a constant term (log(n) roughly) 
    # but the gradient contribution from i=i is 0 because diff=0.
    # However, pdist excludes diagonal. To match pdist exactly, we should mask diagonal.
    mask = ~np.eye(n, dtype=bool)
    
    # For numerical stability in log-sum-exp, we work with the full matrix but mask diagonal in weights.
    # Let's stick to the logic of the previous soft_min_max_ratio but add gradient.
    
    # d_max_soft = beta * log(sum(exp(dists/beta)))
    # d_min_soft = -beta * log(sum(exp(-dists/beta)))
    
    # We need to be careful with the mask. The original code used pdist.
    # Using full matrix with mask is equivalent up to a constant shift in the log-sum-exp if we are careful.
    # Actually, sum(exp(x_i)) for i in S is the same as sum(exp(x_i)) for all i if we set x_i = -inf for i not in S.
    
    x_max = dists / beta
    x_max[~mask] = -np.inf # Exclude diagonal
    
    # Handle case where all are -inf (n=1, but n=16 here)
    m_max = np.max(x_max)
    if not np.isfinite(m_max):
        return -1e6, np.zeros((n, d))
        
    exp_max = np.exp(x_max - m_max)
    sum_exp_max = np.sum(exp_max)
    
    # d_max_soft = beta * (m_max + log(sum_exp_max))
    log_d_max_soft = np.log(beta) + m_max + np.log(sum_exp_max)
    
    # Gradient of log_d_max_soft w.r.t points
    # d/dp log(sum(exp(d/beta))) = sum( (1/beta) * exp(d/beta) / sum(exp(d/beta)) * d(d)/dp )
    # d(d_ij)/dp_k = (p_k - p_j)/d_ij if k==i, (p_k - p_i)/d_ij if k==j, 0 otherwise
    # Actually, d(||p_i - p_j||)/dp_k = (p_k - p_j)/||p_i - p_j|| for k=i, (p_k - p_i)/||p_i - p_j|| for k=j.
    
    # Let w_max_ij = exp_max[i,j] / sum_exp_max
    # grad_log_d_max = (1/beta) * sum_{i,j} w_max_ij * grad_d_ij
    # grad_d_ij w.r.t p_k:
    # if k==i: (p_i - p_j)/d_ij
    # if k==j: (p_j - p_i)/d_ij = -(p_i - p_j)/d_ij
    
    # We can compute this efficiently.
    # grad_d_ij = diff[i,j] / d_ij (vector)
    # Note: diff[i,j] = p_i - p_j.
    # d/dp_k d_ij:
    # if k==i: diff[i,j]/d_ij
    # if k==j: -diff[i,j]/d_ij
    
    # So contribution to grad p_k from pair (i,j) is:
    # w_max_ij / beta * ( (delta_{ki} - delta_{kj}) * diff[i,j] / d_ij )
    
    # Let's compute the vector field V_ij = diff[i,j] / d_ij
    # Avoid division by zero where d_ij = 0 (diagonal, already masked)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[..., np.newaxis]
    # Set diagonal to 0
    unit_vecs[~mask] = 0.0
    
    # grad p_k = (1/beta) * sum_{i,j} w_max_ij * (delta_{ki} - delta_{kj}) * unit_vecs[i,j]
    # This is (1/beta) * ( sum_i w_max_i,: * unit_vecs[i,:] - sum_j w_max_,:j * unit_vecs[:,j] )? No.
    # sum_{i,j} w_ij * delta_{ki} * u_ij = sum_j w_kj * u_kj
    # sum_{i,j} w_ij * delta_{kj} * u_ij = sum_i w_ik * u_ik
    
    # So grad p_k = (1/beta) * ( sum_j w_max[k,j] * unit_vecs[k,j] - sum_i w_max[i,k] * unit_vecs[i,k] )
    
    w_max = exp_max / sum_exp_max
    # term1: sum_j w_max[k,j] * unit_vecs[k,j]
    # This is row-wise weighted sum
    grad_max_term1 = np.sum(w_max[..., np.newaxis] * unit_vecs, axis=1)
    # term2: sum_i w_max[i,k] * unit_vecs[i,k]
    # This is column-wise weighted sum, but unit_vecs[i,k] = -unit_vecs[k,i]
    # So sum_i w_max[i,k] * (-unit_vecs[k,i]) = - sum_i w_max[i,k] * unit_vecs[k,i]
    # Wait, unit_vecs[i,k] is vector p_i - p_k over dist.
    # The formula derived:
    # grad p_k = (1/beta) * sum_{i,j} w_ij (delta_{ki} - delta_{kj}) u_ij
    # = (1/beta) * ( sum_j w_kj u_kj - sum_i w_ik u_ik )
    
    # Note u_ik = -u_ki.
    # So sum_i w_ik u_ik = sum_i w_ik (-u_ki) = - sum_i w_ik u_ki
    # So grad p_k = (1/beta) * ( sum_j w_kj u_kj + sum_i w_ik u_ki )
    # Since sum is over all indices, let's just compute:
    # A_k = sum_j w_kj u_kj
    # B_k = sum_i w_ik u_ik  (Note: u_ik is the vector from i to k? No, u_ij = (p_i-p_j)/d. So u_ik = (p_i-p_k)/d. )
    
    # Let's stick to the explicit sum to be safe, n=16 is small.
    # grad_max = np.zeros((n, d))
    # for i in range(n):
    #   for j in range(n):
    #     if i==j: continue
    #     w = w_max[i,j]
    #     if w == 0: continue
    #     u = unit_vecs[i,j]
    #     grad_max[i] += (w/beta) * u
    #     grad_max[j] -= (w/beta) * u
    
    # Vectorized version:
    # grad_max = (1/beta) * ( np.sum(w_max[..., np.newaxis] * unit_vecs, axis=1) - np.sum(w_max.T[..., np.newaxis] * unit_vecs, axis=0) )?
    # No.
    # Let's use the loop for clarity and correctness, n=16 is tiny.
    grad_max = np.zeros((n, d))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = w_max[i, j]
            if w > 0:
                u = unit_vecs[i, j]
                grad_max[i] += (w / beta) * u
                grad_max[j] -= (w / beta) * u
    
    # Now for d_min_soft
    # d_min_soft = -beta * log(sum(exp(-dists/beta)))
    # Let y = -dists / beta
    # y_min = y
    # m_min = max(y)
    # log_d_min_soft = log(-beta) + m_min + log(sum(exp(y - m_min)))
    # Note: log(-beta) is constant if beta>0. But beta is positive.
    # Wait, d_min_soft = -beta * LSE. So log(d_min_soft) = log(-beta) + log(LSE).
    # Since beta > 0, -beta < 0. LSE is positive.
    # Actually, the definition was:
    # x_min = -dists / beta
    # m_min = max(x_min)
    # LSE_min = m_min + log(sum(exp(x_min - m_min)))
    # d_min_soft = -beta * LSE_min
    # This implies d_min_soft > 0 requires LSE_min < 0?
    # No, exp(x) is always positive. LSE is positive.
    # In the original code:
    # d_min_soft = -beta * log_sum_exp_min
    # If log_sum_exp_min is positive, d_min_soft is negative.
    # But distances are positive.
    # Let's re-read original soft_min_max_ratio.
    # x_min = -dists / beta.
    # m_min = max(x_min).
    # log_sum_exp_min = m_min + log(sum(exp(x_min - m_min))).
    # d_min_soft = -beta * log_sum_exp_min.
    
    # If dists are positive, x_min are negative.
    # m_min is the largest (closest to 0).
    # log_sum_exp_min is roughly m_min + log(n).
    # If m_min is negative, log_sum_exp_min can be negative or positive.
    # However, d_min_soft must be positive for log to be defined.
    # The original code checks `if d_max_soft <= 0 or d_min_soft <= 0: return -1e6`.
    
    # Let's assume the original logic holds.
    # Gradient of log(d_min_soft) = d/dp log(-beta * LSE_min) = d/dp log(LSE_min)
    # LSE_min = log(sum(exp(-d/beta)))
    # d/dp LSE_min = sum( (1/beta) * exp(-d/beta) / sum(exp(-d/beta)) * d(-d)/dp )
    # d(-d_ij)/dp_k = - d(d_ij)/dp_k
    # So grad = - (1/beta) * sum w_min_ij * grad_d_ij
    # where w_min_ij = exp(-d_ij/beta) / sum(exp(-d/beta))
    
    x_min = -dists / beta
    x_min[~mask] = -np.inf
    
    m_min = np.max(x_min)
    if not np.isfinite(m_min):
         return -1e6, np.zeros((n, d))
         
    exp_min = np.exp(x_min - m_min)
    sum_exp_min = np.sum(exp_min)
    
    log_d_min_soft = np.log(-beta) + m_min + np.log(sum_exp_min) # log(-beta) is constant
    
    w_min = exp_min / sum_exp_min
    
    grad_min = np.zeros((n, d))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = w_min[i, j]
            if w > 0:
                u = unit_vecs[i, j]
                # grad_d_ij is same as before.
                # d(-d)/dp = -grad_d
                # So contribution is - (w/beta) * grad_d
                # grad_d[i] += u, grad_d[j] -= u
                # So -grad_d[i] -= u, -grad_d[j] += u
                grad_min[i] -= (w / beta) * u
                grad_min[j] += (w / beta) * u
    
    value = 2 * (log_d_min_soft - log_d_max_soft)
    grad = 2 * (grad_min - grad_max)
    
    return value, grad

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
    
    # Find the pair (i, j) corresponding to min_idx
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
        for k in range(i_found, 0, -1):
            new_pts[k] = new_pts[k-1]
    # Now move j_found to 1 (accounting for possible shift)
    if j_found != 1:
        if j_found == i_found:
            return None
        if j_found > i_found:
            j_found -= 1
        if j_found != 1:
            new_pts[1] = new_pts[j_found]
            for k in range(j_found, 1, -1):
                new_pts[k] = new_pts[k-1]
    
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
    
    global_deadline = time.time() + 620
    
    def objective_smooth(params, beta):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            val, grad = soft_min_max_ratio_and_grad(pts, beta)
            return -val, -grad
        except Exception:
            return 1e6, np.zeros_like(params)

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
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return -calculate_score(pts)

    best_params = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
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
    
    # 6. Concentric circles (1+4+11 or 1+7+8)
    for n_mid in [4, 7]:
        n_out = 16 - 1 - n_mid
        r_mid = 1.0
        r_out = 1.8
        angles_mid = np.linspace(0, 2*np.pi, n_mid, endpoint=False)
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False) + np.pi/n_out
        pts_c = np.array([[0,0]])
        pts_m = np.column_stack([r_mid*np.cos(angles_mid), r_mid*np.sin(angles_mid)])
        pts_o = np.column_stack([r_out*np.cos(angles_out), r_out*np.sin(angles_out)])
        candidates.append(np.vstack([pts_c, pts_m, pts_o]))
    
    inits = []
    for cand in candidates:
        params = points_to_params(cand)
        if params is not None:
            inits.append(params)
    
    for _ in range(5):
        inits.append(rng.randn(28) * 2.0)
    
    start_time = time.time()
    
    # Phase 1: Smooth surrogate optimization
    betas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    current_params = None
    
    for beta in betas:
        if time.time() > global_deadline - 100:
            break
            
        best_local_params = None
        best_local_score = 1e6
        
        for init in inits:
            if time.time() > global_deadline - 100:
                break
                
            try:
                res = minimize(
                    objective_smooth, init, args=(beta,), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
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
            
            inits.insert(0, best_local_params)
            if len(inits) > 5:
                inits = inits[:5]

    # Phase 2: Exact Nelder-Mead polish
    if time.time() < global_deadline - 80:
        if best_params is not None:
            start_polish = best_params
        else:
            start_polish = inits[0]
            
        try:
            res = minimize(
                objective_exact, start_polish, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10}
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
                    options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass

    # Phase 3: Full-space polish
    if time.time() < global_deadline - 30 and best_params is not None:
        best_pts = points_from_params(best_params)
        
        model_candidates = [best_pts]
        for cand in candidates:
            model_candidates.append(cand)
        
        for cand_pts in model_candidates:
            if time.time() > global_deadline - 30:
                break
            try:
                x0 = cand_pts.flatten()
                res = minimize(
                    objective_full, x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_pts = res.x.reshape(n, 2)
            except Exception:
                pass
    
    if best_params is not None:
        final_points = points_from_params(best_params)
    else:
        final_points = hex_init
    
    if not np.all(np.isfinite(final_points)):
        final_points = np.zeros((n, d))
        
    return final_points
