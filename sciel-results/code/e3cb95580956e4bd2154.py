# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

# Precompute constants
_HEX_ANGLE_RAD = np.deg2rad(np.arange(6))
_HEX_COS = np.cos(_HEX_ANGLE_RAD)
_HEX_SIN = np.sin(_HEX_ANGLE_RAD)
# Outer hexagon edge normals (for angle 0)
# Vertices at 0, 60, ...
# Edge 0 is between v0 and v1. Normal is at 30 deg.
_NORMAL_ANGLES = np.deg2rad(30 + 60 * np.arange(6))
_NORMAL_COS = np.cos(_NORMAL_ANGLES)
_NORMAL_SIN = np.sin(_NORMAL_ANGLES)
_NORMALS = np.column_stack([_NORMAL_COS, _NORMAL_SIN]) # (6, 2)

def _get_vertices(centers):
    """
    Compute vertices for all hexagons.
    centers: (N, 2)
    Returns: (N, 6, 2)
    """
    N = centers.shape[0]
    # centers[:, np.newaxis, :] + np.column_stack([_HEX_COS, _HEX_SIN])
    verts = centers[:, np.newaxis, :] + np.column_stack([_HEX_COS, _HEX_SIN])[np.newaxis, :, :]
    return verts

def _smooth_max_size(centers, beta=100.0):
    """
    Compute a smooth approximation of the minimum containing hexagon side length.
    Uses Log-Sum-Exp on the projections onto the 6 outward normals.
    """
    verts = _get_vertices(centers) # (N, 6, 2)
    verts_flat = verts.reshape(-1, 2) # (N*6, 2)
    
    # Projections onto 6 normals
    # normals: (6, 2), verts_flat: (N*6, 2)
    # proj: (6, N*6)
    proj = _NORMALS @ verts_flat.T
    
    # We want s = (2/sqrt(3)) * max(proj)
    # Smooth max: (1/beta) * log(sum(exp(beta * proj)))
    # To avoid overflow, subtract max
    max_proj = np.max(proj)
    if max_proj > 500: # Safety
        return (2.0 / np.sqrt(3.0)) * max_proj
    
    exp_proj = np.exp(beta * (proj - max_proj))
    log_sum_exp = max_proj + np.log(np.sum(exp_proj)) / beta
    
    s = (2.0 / np.sqrt(3.0)) * log_sum_exp
    return s

def _overlap_penalty(centers, beta=10.0):
    """
    Compute a smooth penalty for overlapping hexagons.
    Uses Softplus on the overlap depth along each separating axis.
    """
    N = centers.shape[0]
    verts = _get_vertices(centers) # (N, 6, 2)
    
    penalty = 0.0
    # Precompute projections of each vertex onto each normal?
    # proj[i, k, j] = dot(verts[i, j], normals[k])
    # verts: (N, 6, 2), normals: (6, 2)
    # proj = np.einsum('ijk,jl->ikl', verts, _NORMALS) # (N, 6, 6) -> N hexes, 6 normals, 6 vertices
    
    proj = np.einsum('ijk,jl->ikl', verts, _NORMALS) # (N, 6, 6)
    
    for i in range(N):
        for j in range(i + 1, N):
            # For each of the 6 axes
            for k in range(6):
                # Interval i on axis k
                min_i = np.min(proj[i, k, :])
                max_i = np.max(proj[i, k, :])
                # Interval j on axis k
                min_j = np.min(proj[j, k, :])
                max_j = np.max(proj[j, k, :])
                
                # Overlap depth is max(0, max_i - min_j, max_j - min_i)
                # But wait, this is the overlap *if* they overlap on this axis.
                # If they are separated on ANY axis, they are disjoint.
                # The penalty should be 0 if they are separated on at least one axis.
                # This "OR" logic is hard to make smooth directly.
                
                # However, if we sum the squared overlaps for ALL axes, it will be non-zero 
                # even if they are separated on one axis but "overlapping" in projection on others?
                # No, if they are separated on axis k, the gap is positive.
                # Overlap on axis k is defined as:
                # gap = min(max_i - min_j, max_j - min_i)
                # If gap > 0, they are separated on this axis.
                # If gap <= 0, they overlap on this axis.
                # 
                # For two convex polygons, they are disjoint IF AND ONLY IF there EXISTS an axis with gap > 0.
                # They are intersecting IF AND ONLY IF for ALL axes, gap <= 0.
                #
                # So the "intersection" condition is AND of (gap_k <= 0 for all k).
                # The penalty for intersection should be non-zero only if they intersect.
                #
                # A common trick: The minimum gap over all axes is the "separation distance".
                # If min_gap > 0, they are separated.
                # If min_gap < 0, they are intersecting.
                # The "depth of intersection" is -min_gap.
                # We want to penalize -min_gap if it is positive.
                #
                # min_gap = min_k (gap_k)
                # We want penalty = softplus(-min_gap) or similar.
                # But min is non-smooth.
                # We can use a smooth min (Log-Sum-Exp with negative beta).
                
                # Let's compute gaps for all 6 axes for this pair.
                # gap_k = min(max_i_k - min_j_k, max_j_k - min_i_k)
                # Note: This formula assumes we just check if intervals overlap.
                # Actually, the standard SAT gap is:
                # If max_i < min_j, gap = min_j - max_i
                # If max_j < min_i, gap = min_i - max_j
                # Else gap = -inf (overlap)
                
                # This is getting complicated to make smooth.
                # Let's use a simpler robust metric for hexagons.
                # Two hexagons are separated if the distance between centers is large enough.
                # But orientation is fixed.
                
                # Let's stick to the projection intervals.
                # gap_k = min(max_i, max_j) - max(min_i, min_j) ? No.
                
                # Let's calculate the "overlap" on axis k.
                # Overlap_k = max(0, max_i - min_j, max_j - min_i)
                # This is the length of the intersection of the intervals.
                # If Overlap_k > 0 for all k, they intersect.
                # If Overlap_k == 0 for any k, they are separated.
                
                # We want a penalty that is 0 if ANY Overlap_k is 0.
                # Penalty = - log(1 + sum(Overlap_k)) ? No.
                
                # Let's use the fact that for tight packings, they touch on ONE axis.
                # We can assume the "active" constraint is the minimum gap.
                # Let's define the "separation" S_ij = min_k (gap_k).
                # We want S_ij >= 0.
                # Penalty = softplus(-S_ij).
                
                # To compute S_ij smoothly:
                # S_ij = min_k (gap_k).
                # gap_k is piecewise linear.
                # This is still non-smooth.
                
                # Alternative: Just use the L2 distance between centers?
                # No, hexagons are not circles.
                
                # Let's try a different approach: 
                # Since the problem is small (12 hexagons), we can use a non-smooth penalty 
                # but rely on SLSQP's ability to handle some non-smoothness if we provide 
                # gradients carefully? No, SLSQP requires smooth constraints.
                
                # Let's use a "soft" separation constraint.
                # For each axis k, define the gap.
                # If we sum the squares of the negative gaps (where gap < 0), we penalize overlap.
                # But we only want to penalize if they overlap on ALL axes.
                
                # Actually, if they are separated on one axis, the gap on that axis is positive.
                # If we simply penalize max(0, -min_gap)^2, it works.
                
                # Let's compute gaps for all 6 axes.
                gaps = np.zeros(6)
                for k in range(6):
                    min_i_k = np.min(proj[i, k, :])
                    max_i_k = np.max(proj[i, k, :])
                    min_j_k = np.min(proj[j, k, :])
                    max_j_k = np.max(proj[j, k, :])
                    
                    # Calculate gap
                    # If max_i < min_j, gap = min_j - max_i
                    # If max_j < min_i, gap = min_i - max_j
                    # Else, gap = -1 (overlap)
                    
                    # Smooth approximation of gap:
                    # gap = -softplus(softplus(max_i - min_j) - softplus(min_j - max_i)) ?
                    # This is getting messy.
                    
                    # Let's use a simpler heuristic for the penalty.
                    # Penalty = sum_k (max(0, overlap_k))^2
                    # If they are separated on axis k, overlap_k = 0.
                    # If they overlap on axis k, overlap_k > 0.
                    # If they are separated on ANY axis, at least one overlap_k is 0.
                    # But the sum will still be > 0 if they overlap on other axes.
                    # This will force them to be separated on ALL axes, which is stricter than necessary
                    # and might prevent valid tight packings where they touch on one axis and have "shadow" overlap on others?
                    # No, if they touch on one axis, the gap is 0. The overlap on that axis is 0.
                    # On other axes, they might have positive overlap in projection, but they are still disjoint.
                    # So penalizing sum(overlap^2) is WRONG. It will push them apart even when they are validly touching.
                    
                    # We MUST use the min-gap formulation.
                    # gap_k = min(max_i_k, max_j_k) - max(min_i_k, min_j_k) is NOT the gap.
                    
                    # Correct gap:
                    # If max_i_k < min_j_k: gap = min_j_k - max_i_k
                    # If max_j_k < min_i_k: gap = min_i_k - max_j_k
                    # Else: gap = -1 (overlap)
                    
                    # Smooth gap:
                    # gap_k = -softplus(softplus(max_i_k - min_j_k) - softplus(min_j_k - max_i_k)) 
                    # This doesn't look right.
                    
                    # Let's use:
                    # gap_k = min(max_i_k - min_j_k, max_j_k - min_i_k)
                    # Wait, if max_i < min_j, then max_i - min_j is negative. max_j - min_i is positive?
                    # Example: i=[0,1], j=[2,3]. max_i=1, min_j=2. max_i-min_j = -1.
                    # max_j=3, min_i=0. max_j-min_i = 3.
                    # min(-1, 3) = -1. But gap is 1.
                    
                    # Correct formula for gap between intervals [a,b] and [c,d]:
                    # gap = max(0, c - b, a - d)
                    
                    # Smooth max:
                    # gap_k = softmax(0, c - b, a - d) with small beta?
                    # Or just:
                    # gap_k = (1/beta) * log(exp(beta*0) + exp(beta*(c-b)) + exp(beta*(a-d)))
                    # This is a smooth approximation of the max.
                    
                    # We want the MINIMUM gap over all axes.
                    # S_ij = min_k (gap_k)
                    # Smooth min:
                    # S_ij = - (1/beta_min) * log( sum_k exp(-beta_min * gap_k) )
                    
                    # Penalty = softplus(-S_ij)
                    
                    # This is computationally heavy but doable.
                    
                    # Let's simplify. In a tight packing of hexagons, they usually touch.
                    # We can just use the distance between centers and check against a lower bound?
                    # No, that's not exact.
                    
                    # Let's try a different strategy: 
                    # Use Nelder-Mead but with a MUCH better initial guess and a 
                    # "simulated annealing" style restart, but with a 
                    # vectorized objective that is faster.
                    
                    # The current program is slow because it's not vectorized well and uses NM.
                    # Let's switch to SLSQP with a SMOOTH objective and SMOOTH constraints.
                    
                    # Constraint: S_ij >= 0.
                    # S_ij = min_k (gap_k).
                    # gap_k = max(0, min_j_k - max_i_k, min_i_k - max_j_k).
                    
                    # Let's implement smooth max and smooth min.
                    def smooth_max(vals, beta=10.0):
                        m = np.max(vals)
                        if m > 500: return m
                        return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta
                    
                    def smooth_min(vals, beta=10.0):
                        return -smooth_max(-vals, beta)
                    
                    # For this pair i,j, compute gaps for all 6 axes
                    # We need proj[i, k, :] and proj[j, k, :]
                    
                    # Vectorized gap calculation for all 6 axes
                    # min_i: (6,), max_i: (6,)
                    # min_j: (6,), max_j: (6,)
                    
                    min_i_all = np.min(proj[i, :, :], axis=1) # (6,)
                    max_i_all = np.max(proj[i, :, :], axis=1) # (6,)
                    min_j_all = np.min(proj[j, :, :], axis=1) # (6,)
                    max_j_all = np.max(proj[j, :, :], axis=1) # (6,)
                    
                    # gap_k = max(0, min_j_k - max_i_k, min_i_k - max_j_k)
                    # Let's compute the 3 terms for all 6 axes
                    term1 = np.zeros(6)
                    term2 = min_j_all - max_i_all
                    term3 = min_i_all - max_j_all
                    
                    # Smooth max of [0, term2, term3]
                    # Stack into (3, 6)
                    terms = np.vstack([term1, term2, term3])
                    
                    # Smooth max along axis 0
                    beta_gap = 10.0
                    m = np.max(terms, axis=0)
                    # Handle overflow
                    safe_m = np.where(m > 500, 500, m)
                    exp_terms = np.exp(beta_gap * (terms - safe_m[np.newaxis, :]))
                    gaps = safe_m + np.log(np.sum(exp_terms, axis=0)) / beta_gap # (6,)
                    
                    # Now smooth min of gaps
                    beta_min = 10.0
                    min_gap = smooth_min(gaps, beta_min)
                    
                    # Penalty for this pair: softplus(-min_gap)
                    # softplus(x) = log(1 + exp(x))
                    x = -min_gap
                    penalty += np.log1p(np.exp(np.clip(x, -500, 500)))
                    
    return penalty

def _objective(params):
    centers = params.reshape(12, 2)
    size = _smooth_max_size(centers, beta=50.0)
    pen = _overlap_penalty(centers, beta=10.0)
    return size + 1000.0 * pen

def _grad_objective(params):
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def _make_good_initial():
    # Hexagonal lattice
    # Center at 0
    # Neighbors at distance 2*sqrt(3)? No, side 1.
    # Distance between centers of touching hexagons is 2*sqrt(3)/2 * 2? 
    # Apothem is sqrt(3)/2. Distance between centers is 2 * apothem = sqrt(3).
    # Wait, side length 1. Distance between parallel sides is sqrt(3).
    # So centers of touching hexagons are sqrt(3) apart.
    
    d = np.sqrt(3)
    centers = []
    # Ring 0
    centers.append([0.0, 0.0])
    # Ring 1 (6 hexagons)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers.append([d * np.cos(angle), d * np.sin(angle)])
    # Ring 2 (5 more to make 12? No, Ring 2 has 12. We need 5 more)
    # Let's just place them roughly
    # 12 total: 1 + 6 + 5
    # The 5 can be placed in the gaps of the second ring?
    # Or just use a 4x3 grid?
    
    # Let's use a simple grid
    # x: -1.5, -0.5, 0.5, 1.5
    # y: -sqrt(3)/2, 0, sqrt(3)/2
    # This is 12 points.
    # Spacing x: 1.0 (too close, need sqrt(3))
    # Spacing y: sqrt(3)/2 (too close, need sqrt(3))
    
    # Correct lattice:
    # x = i * 1.5 + (j % 2) * 0.75 ?
    # For hexagons side 1:
    # Horizontal distance between centers: 1.5? No.
    # Vertex to vertex is 2. Center to center is sqrt(3).
    # If we align flat-top:
    # dx = 1.5, dy = sqrt(3)/2.
    # Let's check separation.
    # Center (0,0) and (1.5, sqrt(3)/2).
    # dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    
    init = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            x = i * 1.5 + (j % 2) * 0.75
            y = j * np.sqrt(3) / 2
            init.append([x, y])
            
    return np.array(init[:12])

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    init_centers = _make_good_initial()
    best_centers = init_centers.copy()
    
    # Initial size check
    def _exact_size(centers):
        verts = _get_vertices(centers)
        verts_flat = verts.reshape(-1, 2)
        proj = _NORMALS @ verts_flat.T
        max_proj = np.max(proj)
        return (2.0 / np.sqrt(3.0)) * max_proj

    best_s = _exact_size(best_centers)
    
    x0 = init_centers.flatten()
    
    # SLSQP Optimization
    # We don't provide gradients for constraints, so we can't use constraints easily.
    # We rely on the penalty in the objective.
    
    try:
        result = minimize(
            _objective,
            x0,
            method='SLSQP',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'maxfun': 10000
            }
        )
        
        if result.fun < best_s:
            candidate = result.x.reshape(12, 2)
            # Verify separation with exact check
            if _check_separation(candidate):
                s_cand = _exact_size(candidate)
                if s_cand < best_s:
                    best_centers = candidate
                    best_s = s_cand
    except Exception:
        pass
    
    # Polish with Nelder-Mead on the exact objective if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        try:
            result2 = minimize(
                lambda p: _exact_size(p.reshape(12, 2)),
                best_centers.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            if result2.fun < best_s:
                candidate = result2.x.reshape(12, 2)
                if _check_separation(candidate):
                    best_centers = candidate
                    best_s = result2.fun
        except Exception:
            pass

    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def _check_separation(centers, tol=1e-6):
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = np.column_stack([centers[i, 0] + base_cos, centers[i, 1] + base_sin])
            vj = np.column_stack([centers[j, 0] + base_cos, centers[j, 1] + base_sin])
            
            for k in range(6):
                n_vec = normals[k]
                proj_i = np.dot(vi, n_vec)
                proj_j = np.dot(vj, n_vec)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    break
            else:
                return False
    return True

# EVOLVE-BLOCK-END
