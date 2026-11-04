import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _project_to_bary_simplex(bary):
    """Project onto the barycentric simplex: u>=0, v>=0, u+v<=1."""
    bary = np.array(bary, dtype=float)
    # Clip negatives
    bary = np.maximum(bary, 0.0)
    # Project onto u+v<=1
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_dets_from_bary(bary):
    """Compute all triple determinants (2*area) from barycentric coords.
    
    For points p_i = u_i*V1 + v_i*V2, the cross product (pj-pi) x (pk-pi)
    can be computed from barycentric differences.
    
    V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
    p = u*V1 + v*V2 = (u + 0.5*v, sqrt(3)/2 * v)
    
    det = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    
    Let du = u_j - u_i, dv = v_j - v_i, etc.
    px = u + 0.5*v, py = sqrt(3)/2 * v
    
    (pj-pi)_x = du_ji + 0.5*dv_ji
    (pj-pi)_y = sqrt(3)/2 * dv_ji
    (pk-pi)_x = du_ki + 0.5*dv_ki
    (pk-pi)_y = sqrt(3)/2 * dv_ki
    
    det = (du_ji + 0.5*dv_ji)*(sqrt(3)/2*dv_ki) - (sqrt(3)/2*dv_ji)*(du_ki + 0.5*dv_ki)
        = sqrt(3)/2 * [du_ji*dv_ki + 0.5*dv_ji*dv_ki - dv_ji*du_ki - 0.5*dv_ji*dv_ki]
        = sqrt(3)/2 * [du_ji*dv_ki - dv_ji*du_ki]
    """
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    du_ji = uj - ui
    dv_ji = vj - vi
    du_ki = uk - ui
    dv_ki = vk - vi
    
    dets = np.sqrt(3) / 2.0 * (du_ji * dv_ki - dv_ji * du_ki)
    return dets

def _min_area_ratio_from_bary(bary):
    """Compute the minimum area (divided by triangle area) from barycentric coords."""
    dets = _all_dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    # Area of triangle formed by 3 points = 0.5 * |det|
    # Ratio to unit triangle area = (0.5 * min_det) / TRI_AREA
    return 0.5 * min_det / TRI_AREA

def _objective_smooth(x, beta=50.0):
    """Smooth objective: negative log-sum-exp of negative areas (approximates -min area).
    
    We want to maximize min(area_i), which is equivalent to minimizing -min(area_i).
    Using soft-min: -log-sum-exp(-beta * area_i) / beta approximates -min(area_i)
    as beta -> infinity.
    
    We work with 2*area = |det| values for numerical stability.
    """
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    # log-sum-exp trick for numerical stability
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0
    
    # We want to maximize min(abs_det). 
    # -min(abs_det) ≈ (1/beta) * log(sum(exp(-beta * abs_det)))
    # = (1/beta) * log(sum(exp(-beta * abs_det)))
    # Use log-sum-exp: log(sum(exp(a_i))) = max(a) + log(sum(exp(a_i - max(a))))
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    lse = max_nb + np.log(np.sum(np.exp(neg_beta_dets - max_nb)))
    return lse / beta  # This approximates -min(abs_det)

def _objective_grad(x, beta=50.0):
    """Objective with gradient for L-BFGS-B."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0, np.zeros_like(x)
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    exp_vals = np.exp(neg_beta_dets - max_nb)
    sum_exp = np.sum(exp_vals)
    lse = max_nb + np.log(sum_exp)
    val = lse / beta
    
    # Gradient: d/dx [log(sum(exp(-beta*|det_i|)))/beta]
    # = (1/beta) * sum(exp(-beta*|det_i|) * (-beta * d|det_i|/dx)) / sum(exp(-beta*|det_i|))
    # = -sum(exp(-beta*|det_i|) * d|det_i|/dx) / sum(exp(-beta*|det_i|))
    # 
    # d|det_i|/dx = sign(det_i) * ddet_i/dx
    # ddet_i/dx can be computed from the barycentric representation
    
    # For gradient, compute d(det_i)/d(bary) for each triple
    idx = _IDX_TRIPLES
    signs = np.sign(dets)
    
    # d(det)/d(u_i) for triple (i,j,k): 
    # det = sqrt(3)/2 * [(u_j-u_i)(v_k-v_i) - (v_j-v_i)(u_k-u_i)]
    # d/d(u_i) = sqrt(3)/2 * [-(v_k-v_i) + (v_j-v_i)] = sqrt(3)/2 * (v_j - v_k)
    # d/d(v_i) = sqrt(3)/2 * [-(u_k-u_i) + (u_j-u_i)] = sqrt(3)/2 * (u_j - u_k)
    # d/d(u_j) = sqrt(3)/2 * [(v_k-v_i)]
    # d/d(v_j) = sqrt(3)/2 * [-(u_k-u_i)]
    # d/d(u_k) = sqrt(3)/2 * [-(v_j-v_i)]
    # d/d(v_k) = sqrt(3)/2 * [(u_j-u_i)]
    
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    sqrt3_2 = np.sqrt(3) / 2.0
    
    # Gradient of det w.r.t. each point's barycentric coords
    # Shape: (N_TRIPLES, N, 2)
    grad_det = np.zeros((N_TRIPLES, N, 2))
    
    # Point i
    grad_det[:, idx[:, 0], 0] = sqrt3_2 * (vj - vk)
    grad_det[:, idx[:, 0], 1] = sqrt3_2 * (uj - uk)
    # Point j
    grad_det[:, idx[:, 1], 0] = sqrt3_2 * (vk - vi)
    grad_det[:, idx[:, 1], 1] = -sqrt3_2 * (uk - ui)
    # Point k
    grad_det[:, idx[:, 2], 0] = -sqrt3_2 * (vj - vi)
    grad_det[:, idx[:, 2], 1] = sqrt3_2 * (uj - ui)
    
    # d|det|/dx = sign(det) * ddet/dx
    sign_det = signs[:, None, None]
    grad_abs_det = sign_det * grad_det  # (N_TRIPLES, N, 2)
    
    # Weight by softmax probabilities
    weights = exp_vals / sum_exp  # (N_TRIPLES,)
    
    # grad_obj = -sum(weights_t * grad_abs_det_t)
    grad_bary = -np.sum(weights[:, None, None] * grad_abs_det, axis=0)  # (N, 2)
    
    # Map back to flat x
    grad_x = np.zeros_like(x)
    grad_x[:N*2] = grad_bary.flatten()
    
    return val, grad_x

def _objective_slsqp(x):
    """Objective for SLSQP (uses exact min area)."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    min_abs_det = np.min(np.abs(dets))
    return -min_abs_det  # Negate for minimization

def _structured_init_fps():
    """Farthest-point-sampling structured initialization in barycentric coords."""
    # Start with vertices
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    
    # Add centroid
    points.append(np.array([1.0/3, 1.0/3]))
    
    # Farthest point sampling for remaining points
    rng = np.random.RandomState(789)
    
    while len(points) < N:
        # Sample many candidates
        candidates = []
        for _ in range(2000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v <= 1.0:
                candidates.append(np.array([u, v]))
        candidates = np.array(candidates)
        
        # For each candidate, compute min distance to existing points
        # Use barycentric distance (approximates geometric distance)
        best_candidate = None
        best_min_dist = -1
        
        for cand in candidates:
            dists = []
            for p in points:
                d = np.linalg.norm(cand - p)
                dists.append(d)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = cand
        
        if best_candidate is not None:
            points.append(best_candidate)
    
    return np.array(points)

def _local_optimize_nm(x0, maxiter=500, xatol=1e-9, fatol=1e-9):
    """Nelder-Mead local optimization."""
    try:
        res = minimize(
            _objective_smooth, x0, method='Nelder-Mead',
            args=(30.0,),
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=300, gtol=1e-8):
    """L-BFGS-B with gradient."""
    try:
        res = minimize(
            _objective_grad, x0, method='L-BFGS-B',
            jac=True,
            args=(30.0,),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """SLSQP with simplex constraints."""
    n = len(x0) // 2
    # Constraints: u_i >= 0, v_i >= 0, u_i + v_i <= 1
    cons = []
    for i in range(n):
        # u_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        # v_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * n)
    
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    # Generate multiple structured starts
    starts = []
    
    # FPS-based start
    starts.append(_structured_init_fps())
    
    # Hexagonal lattice
    bary_hex = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75],
    ])
    starts.append(_project_to_bary_simplex(bary_hex.copy()))
    
    # Symmetric grid
    bary_grid = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.25, 0.0], [0.75, 0.0], [0.0, 0.25], [0.0, 0.75],
        [0.25, 0.25], [0.5, 0.25], [0.25, 0.5], [0.5, 0.5],
    ])
    starts.append(_project_to_bary_simplex(bary_grid.copy()))
    
    # Random structured
    rng = np.random.RandomState(42)
    bary_rand = []
    bary_rand.append([0.0, 0.0])
    bary_rand.append([1.0, 0.0])
    bary_rand.append([0.0, 1.0])
    bary_rand.append([1/3, 1/3])
    for _ in range(7):
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_rand.append([u, v])
        else:
            bary_rand.append([u * 0.7, v * 0.7])
    starts.append(_project_to_bary_simplex(np.array(bary_rand[:N])))
    
    # Evaluate initial starts
    for bary_start in starts:
        if time_left() < 5:
            break
        bary_clamped = _project_to_bary_simplex(bary_start)
        points = _bary_to_cart(bary_clamped)
        score = _min_area_ratio_from_bary(bary_clamped)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary_clamped.copy()
    
    # Multi-start optimization
    max_restarts = 80
    restart = 0
    
    while restart < max_restarts and time_left() > 15.0:
        restart += 1
        
        # Choose starting point
        if restart % 4 == 0:
            # Random restart
            bary_start = []
            for _ in range(N):
                u = rng.uniform(0.05, 0.9)
                v = rng.uniform(0.05, 0.9)
                if u + v < 0.9:
                    bary_start.append([u, v])
                else:
                    bary_start.append([u * 0.8, v * 0.8])
            bary_start = _project_to_bary_simplex(np.array(bary_start[:N]))
        elif restart % 2 == 0:
            # Moderate perturbation of best
            scale = 0.04 + 0.02 * (1.0 - restart / max_restarts)
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _project_to_bary_simplex(bary_pert)
        else:
            # Larger perturbation
            scale = 0.08 + 0.04 * (1.0 - restart / max_restarts)
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _project_to_bary_simplex(bary_pert)
        
        bary_opt = bary_start.copy()
        
        # Try SLSQP (exact min area, constrained)
        if time_left() > 10:
            x_sl = _local_optimize_slsqp(bary_opt.flatten(), maxiter=150)
            bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
            score_sl = _min_area_ratio_from_bary(bary_sl)
            if score_sl > best_score:
                best_score = score_sl
                best_points = _bary_to_cart(bary_sl).copy()
                best_bary = bary_sl.copy()
                bary_opt = bary_sl.copy()
        
        # Try L-BFGS-B with smooth objective and gradient
        if time_left() > 8:
            x_lb = _local_optimize_lbfgs(bary_opt.flatten(), maxiter=200)
            bary_lb = _project_to_bary_simplex(x_lb.reshape(-1, 2))
            score_lb = _min_area_ratio_from_bary(bary_lb)
            if score_lb > best_score:
                best_score = score_lb
                best_points = _bary_to_cart(bary_lb).copy()
                best_bary = bary_lb.copy()
                bary_opt = bary_lb.copy()
        
        # Fine-tune with Nelder-Mead
        if time_left() > 5:
            x_nm = _local_optimize_nm(bary_opt.flatten(), maxiter=300)
            bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
            score_nm = _min_area_ratio_from_bary(bary_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = _bary_to_cart(bary_nm).copy()
                best_bary = bary_nm.copy()
    
    # Final fine-tuning
    if time_left() > 8.0 and best_bary is not None:
        for ft in range(15):
            if time_left() < 4.0:
                break
            # Small perturbation
            bary_ft = best_bary + rng.normal(0, 0.003, size=best_bary.shape)
            bary_ft = _project_to_bary_simplex(bary_ft)
            
            # SLSQP refinement
            x_sl = _local_optimize_slsqp(bary_ft.flatten(), maxiter=100)
            bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
            score_sl = _min_area_ratio_from_bary(bary_sl)
            
            if score_sl > best_score:
                best_score = score_sl
                best_points = _bary_to_cart(bary_sl).copy()
                best_bary = bary_sl.copy()
            
            # NM refinement
            x_nm = _local_optimize_nm(bary_sl.flatten(), maxiter=200, xatol=1e-10, fatol=1e-10)
            bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
            score_nm = _min_area_ratio_from_bary(bary_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_points = _bary_to_cart(bary_nm).copy()
                best_bary = bary_nm.copy()
    
    # Fallback
    if best_points is None:
        bary_fb = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75],
        ])
        best_points = _bary_to_cart(_project_to_bary_simplex(bary_fb))
    
    return best_points
