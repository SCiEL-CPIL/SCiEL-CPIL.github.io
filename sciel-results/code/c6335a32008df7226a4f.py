import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2

def _soft_max_logsumexp(x, T=1.0):
    """Stable soft max via log-sum-exp."""
    x_scaled = x / T
    m = np.max(x_scaled)
    if m < -1000:
        return m * T
    return T * (m + np.log(np.sum(np.exp(x_scaled - m))))

def _soft_min_logsumexp(x, T=1.0):
    """Stable soft min via log-sum-exp on -x."""
    return -_soft_max_logsumexp(-x, T)

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    # P is (N, 2)
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

def _soft_spread_objective(params, T=1.0):
    """
    Smooth surrogate objective using soft max/min.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    soft_max = _soft_max_logsumexp(d_sq, T)
    soft_min = _soft_min_logsumexp(d_sq, T)
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _soft_spread_gradient(params, T=1.0):
    """
    Analytical gradient of the soft spread objective.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Compute all pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq_full = np.sum(diff**2, axis=2)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, 1)
    d_sq = d_sq_full[iu]
    
    # Soft max weights: softmax(d_sq / T)
    d_scaled = d_sq / T
    m = np.max(d_scaled)
    exp_vals = np.exp(d_scaled - m)
    softmax_weights = exp_vals / np.sum(exp_vals)
    
    # Soft min weights: softmax(-d_sq / T)
    neg_d_scaled = -d_sq / T
    m_neg = np.max(neg_d_scaled)
    exp_vals_neg = np.exp(neg_d_scaled - m_neg)
    softmin_weights = exp_vals_neg / np.sum(exp_vals_neg)
    
    soft_max = T * (m + np.log(np.sum(exp_vals)))
    soft_min = -T * (m_neg + np.log(np.sum(exp_vals_neg)))
    
    if soft_min < 1e-8:
        return np.zeros_like(params)
    
    # d(soft_max)/d(d_sq) = softmax_weights / T
    # d(soft_min)/d(d_sq) = -softmin_weights / T
    
    # Objective = soft_max / soft_min
    # d(obj)/d(d_sq) = (d(soft_max)/d(d_sq) * soft_min - soft_max * d(soft_min)/d(d_sq)) / soft_min^2
    #                = (softmax_weights / T * soft_min + soft_max * softmin_weights / T) / soft_min^2
    #                = (softmax_weights * soft_min + soft_max * softmin_weights) / (T * soft_min^2)
    
    grad_d_sq = (softmax_weights * soft_min + soft_max * softmin_weights) / (T * soft_min**2)
    
    # Now compute gradient w.r.t. P
    # d(d_sq_ij)/d(P_k) = 2 * (P_k - P_j) if k == i, 2 * (P_k - P_i) if k == j, 0 otherwise
    # For upper triangle, i < j
    
    grad_P = np.zeros((N, 2))
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        g = grad_d_sq[idx]
        # d(d_sq_ij)/d(P_i) = 2*(P_i - P_j)
        grad_P[i] += 2 * g * (P[i] - P[j])
        # d(d_sq_ij)/d(P_j) = 2*(P_j - P_i)
        grad_P[j] += 2 * g * (P[j] - P[i])
    
    # Gradient w.r.t. params (which are P[2:].flatten())
    grad_params = grad_P[2:].flatten()
    
    return grad_params

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
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal Lattice Subset
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
        # Remove 3 outer points to get 16
        for trial in range(8):
            if trial == 0:
                remove_idx = list(outer_indices[:3])
            else:
                remove_idx = list(rng.choice(outer_indices, 3, replace=False))
            keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
            pts_hex = hex_points[keep_idx]
            candidates.append(pts_hex)

    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer) with optimized radii
    for r_inner, r_outer in [(0.5, 1.5), (0.6, 1.4), (0.7, 1.3), (1.0, 2.0)]:
        pts_circ = []
        for k in range(8):
            ang = 2 * np.pi * k / 8
            pts_circ.append([r_outer * np.cos(ang), r_outer * np.sin(ang)])
        for k in range(8):
            ang = 2 * np.pi * (k + 0.5) / 8
            pts_circ.append([r_inner * np.cos(ang), r_inner * np.sin(ang)])
        candidates.append(np.array(pts_circ))
    
    # 4. Random Perturbed Grid
    perturbed = grid + rng.normal(0, 0.15, grid.shape)
    candidates.append(perturbed)
    
    # 5. Random Gaussian Cluster
    gauss = rng.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([2 * np.cos(ang), 2 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    # 7. Three concentric rings: 6, 6, 4
    for r1, r2, r3 in [(0.5, 1.0, 1.5), (0.6, 1.1, 1.6), (0.4, 0.9, 1.4)]:
        pts_3rings = []
        for k in range(6):
            ang = 2 * np.pi * k / 6
            pts_3rings.append([r1 * np.cos(ang), r1 * np.sin(ang)])
        for k in range(6):
            ang = 2 * np.pi * (k + 0.5) / 6
            pts_3rings.append([r2 * np.cos(ang), r2 * np.sin(ang)])
        for k in range(4):
            ang = 2 * np.pi * k / 4
            pts_3rings.append([r3 * np.cos(ang), r3 * np.sin(ang)])
        candidates.append(np.array(pts_3rings))
    
    # 8. Square grid with alternating offsets (quasi-crystal-like)
    sq_pts = []
    for i in range(4):
        for j in range(4):
            x = i - 1.5
            y = j - 1.5
            if (i + j) % 2 == 1:
                x += 0.1
                y += 0.1
            sq_pts.append([x, y])
    candidates.append(np.array(sq_pts))
    
    # 9. Fibonacci spiral
    fib_pts = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for k in range(16):
        r = np.sqrt(k + 1) * 0.6
        ang = k * golden_angle
        fib_pts.append([r * np.cos(ang), r * np.sin(ang)])
    candidates.append(np.array(fib_pts))
    
    # 10. Random diverse points
    for seed in [123, 456, 789]:
        rng_temp = np.random.default_rng(seed)
        rand_pts = rng_temp.uniform(-2, 2, (N, 2))
        candidates.append(rand_pts)
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    Returns the optimized parameter vector (coords of P[2:] to P[15]) and the permutation used.
    """
    # Find min distance pair
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
        # Degenerate, return original flattened (bad)
        return None, None
        
    # Rotation matrix to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (P - p0) @ R_mat.T / norm_v
    
    # Reorder points so that p is at index 0 and q is at index 1
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Parameters are the coordinates of points 2..15
    init_params = transformed[2:].flatten()
    
    return init_params, perm

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    best_perm = None
    best_initial_P = None
    
    results = []
    
    # Phase 1: Multi-temperature L-BFGS-B with analytical gradients
    temperatures = [5.0, 2.0, 1.0, 0.5, 0.2]
    
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 25:  # Leave 25s for final polish
            break
            
        init_params, perm = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        current_params = init_params
        current_score = float('inf')
        
        # Multi-temperature annealing
        for T in temperatures:
            if time.time() > deadline - 25:
                break
            
            try:
                res = minimize(
                    lambda x: _soft_spread_objective(x, T),
                    lambda x: _soft_spread_gradient(x, T),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                P_opt = np.zeros((N, 2))
                P_opt[0] = [0.0, 0.0]
                P_opt[1] = [1.0, 0.0]
                P_opt[2:] = res.x.reshape(-1, 2)
                
                score = _score(P_opt)
                
                if score < current_score:
                    current_score = score
                    current_params = res.x.copy()
                
                if score < best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_perm = perm
                    best_initial_P = P0
                    
            except Exception:
                pass
        
        results.append((current_score, current_params, perm, P0))

    # Phase 2: Fine-tune top candidates with lower temperature
    results.sort(key=lambda x: x[0])
    
    top_k = min(5, len(results))
    for score, params, perm, P0 in results[:top_k]:
        if time.time() > deadline - 10:
            break
        
        try:
            # Final refinement with very low temperature
            res_final = minimize(
                lambda x: _soft_spread_objective(x, 0.1),
                lambda x: _soft_spread_gradient(x, 0.1),
                params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            P_final = np.zeros((N, 2))
            P_final[0] = [0.0, 0.0]
            P_final[1] = [1.0, 0.0]
            P_final[2:] = res_final.x.reshape(-1, 2)
            
            final_score = _score(P_final)
            
            if final_score < best_score:
                best_score = final_score
                best_params = res_final.x.copy()
                best_perm = perm
                
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish on the very best
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
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

    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Check validity
        if _score(final_pts) > 1e6:
             # Fallback to best initial candidate
             return candidates[0]
             
        return final_pts
    else:
        # Fallback to first candidate
        return candidates[0]

# EVOLVE-BLOCK-END
