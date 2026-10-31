import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_spread_objective(params):
    """
    Smooth surrogate objective.
    Minimizes the soft spread of squared distances.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max and soft min using log-sum-exp
    m = np.max(d_sq)
    exp_max = np.exp(d_sq - m)
    soft_max = m + np.log(np.sum(exp_max))
    
    m_neg = np.max(-d_sq)
    exp_min = np.exp(-d_sq + m_neg)
    soft_min = -m_neg - np.log(np.sum(exp_min))
    
    if soft_min < 1e-6:
        return 1e10
    
    return soft_max / soft_min

def _soft_spread_gradient(params):
    """
    Analytical gradient of the soft spread objective.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    m = np.max(d_sq)
    exp_max = np.exp(d_sq - m)
    soft_max = m + np.log(np.sum(exp_max))
    w_max = exp_max / np.sum(exp_max)
    
    m_neg = np.max(-d_sq)
    exp_min = np.exp(-d_sq + m_neg)
    soft_min = -m_neg - np.log(np.sum(exp_min))
    w_min = exp_min / np.sum(exp_min)
    
    if soft_min < 1e-6:
        return np.zeros_like(params)
    
    grad_sm = np.zeros((N, 2))
    grad_smin = np.zeros((N, 2))
    
    iu = np.triu_indices(N, 1)
    pairs_i = iu[0]
    pairs_j = iu[1]
    num_pairs = len(pairs_i)
    
    for idx in range(num_pairs):
        i = pairs_i[idx]
        j = pairs_j[idx]
        diff = P[i] - P[j]
        
        grad_sm[i] += 2 * w_max[idx] * diff
        grad_sm[j] -= 2 * w_max[idx] * diff
        
        grad_smin[i] += 2 * w_min[idx] * diff
        grad_smin[j] -= 2 * w_min[idx] * diff
    
    grad_combined = (grad_sm * soft_min - soft_max * grad_smin) / (soft_min * soft_min)
    return grad_combined[2:].flatten()

def _exact_objective(params):
    """
    Exact objective for local refinement.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    
    # 1. Hexagonal Lattice Subset - evaluate all removal combinations
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    if len(outer_indices) >= 3:
        best_hex_score = float('inf')
        best_hex_pts = None
        for removal_combo in itertools.combinations(outer_indices, 3):
            removal_set = set(removal_combo)
            keep_idx = [i for i in range(len(hex_points)) if i not in removal_set]
            pts_hex = hex_points[keep_idx]
            s = _score(pts_hex)
            if s < best_hex_score:
                best_hex_score = s
                best_hex_pts = pts_hex
        if best_hex_pts is not None:
            candidates.append(best_hex_pts)
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer)
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Regular 16-gon
    pts_poly = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_poly.append([np.cos(ang), np.sin(ang)])
    candidates.append(np.array(pts_poly))
    
    # 5. Two-ring circle: 6 inner at r=0.6, 10 outer at r=1.4
    pts_two_ring = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_two_ring.append([0.6 * np.cos(ang), 0.6 * np.sin(ang)])
    for k in range(10):
        ang = 2 * np.pi * (k + 0.5) / 10
        pts_two_ring.append([1.4 * np.cos(ang), 1.4 * np.sin(ang)])
    candidates.append(np.array(pts_two_ring))
    
    # 6. Random Perturbed Grid
    rng = np.random.default_rng(42)
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    # 7. Square ring (4 corners + 12 on edges)
    pts_ring = []
    # Corners
    pts_ring.append([-1.5, -1.5])
    pts_ring.append([1.5, -1.5])
    pts_ring.append([1.5, 1.5])
    pts_ring.append([-1.5, 1.5])
    # Edge midpoints and quarter points
    for t in [0.25, 0.5, 0.75]:
        pts_ring.append([-1.5 + t*3, -1.5])
        pts_ring.append([1.5, -1.5 + t*3])
        pts_ring.append([-1.5 + t*3, 1.5])
        pts_ring.append([-1.5, -1.5 + t*3])
    # That's 4 + 12 = 16
    candidates.append(np.array(pts_ring))
    
    # 8. Cross pattern
    pts_cross = []
    # Center
    pts_cross.append([0, 0])
    # 4 arms
    for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
        for r in [0.5, 1.0, 1.5]:
            pts_cross.append([r * np.cos(ang), r * np.sin(ang)])
    # Diagonal points
    for ang in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        pts_cross.append([1.2 * np.cos(ang), 1.2 * np.sin(ang)])
    candidates.append(np.array(pts_cross))
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    p, q = np.unravel_index(min_idx, (N, N))
    if p > q:
        p, q = q, p
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None, None
        
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    return init_params, perm

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    best_perm = None
    best_initial_P = None
    
    # Phase 1: Smooth Optimization (L-BFGS-B) with analytical gradient
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 50:
            break
            
        init_params, perm = _prepare_for_optimization(P0)
        if init_params is None:
            continue
            
        try:
            res = minimize(
                _soft_spread_objective,
                init_params,
                jac=_soft_spread_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            P_opt = np.zeros((N, 2))
            P_opt[0] = [0.0, 0.0]
            P_opt[1] = [1.0, 0.0]
            P_opt[2:] = res.x.reshape(-1, 2)
            
            score = _score(P_opt)
            
            if score < best_score:
                best_score = score
                best_params = res.x.copy()
                best_perm = perm
                best_initial_P = P0
                
        except Exception:
            pass

    # Phase 2: Multi-restart local refinement
    if best_params is not None:
        rng = np.random.default_rng(12345)
        num_restarts = 8
        restart_time_budget = min(15, (deadline - 15) / num_restarts)
        
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            scale = 0.01 * (r + 1)
            perturbed = best_params + rng.normal(0, scale, best_params.shape)
            
            try:
                res_pw = minimize(
                    _exact_objective,
                    perturbed,
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': 3000, 'disp': False}
                )
                
                P_pw = np.zeros((N, 2))
                P_pw[0] = [0.0, 0.0]
                P_pw[1] = [1.0, 0.0]
                P_pw[2:] = res_pw.x.reshape(-1, 2)
                
                pw_score = _score(P_pw)
                
                if pw_score < best_score:
                    best_score = pw_score
                    best_params = res_pw.x.copy()
                    
            except Exception:
                pass

    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            
            P_nm = np.zeros((N, 2))
            P_nm[0] = [0.0, 0.0]
            P_nm[1] = [1.0, 0.0]
            P_nm[2:] = res_nm.x.reshape(-1, 2)
            
            nm_score = _score(P_nm)
            
            if nm_score < best_score:
                best_score = nm_score
                best_params = res_nm.x.copy()
                
        except Exception:
            pass

    # Phase 4: Final fine-tuning with very tight tolerances
    if best_params is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _exact_objective,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'maxfev': 5000, 'disp': False, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            
            P_final = np.zeros((N, 2))
            P_final[0] = [0.0, 0.0]
            P_final[1] = [1.0, 0.0]
            P_final[2:] = res_final.x.reshape(-1, 2)
            
            final_score = _score(P_final)
            
            if final_score < best_score:
                best_score = final_score
                best_params = res_final.x.copy()
                
        except Exception:
            pass

    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        if _score(final_pts) > 1e6:
            best_init = None
            best_init_score = float('inf')
            for P0 in candidates:
                s = _score(P0)
                if s < best_init_score:
                    best_init_score = s
                    best_init = P0
            if best_init is not None:
                return best_init
            return candidates[0]
            
        return final_pts
    else:
        best_init = None
        best_init_score = float('inf')
        for P0 in candidates:
            s = _score(P0)
            if s < best_init_score:
                best_init_score = s
                best_init = P0
        if best_init is not None:
            return best_init
        return candidates[0]

# EVOLVE-BLOCK-END
