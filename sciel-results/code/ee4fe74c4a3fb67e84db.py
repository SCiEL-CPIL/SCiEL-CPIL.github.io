import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return (N, N) matrix of squared pairwise distances."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, 1)
    d_sq_upper = d_sq[iu]
    min_d_sq = np.min(d_sq_upper)
    max_d_sq = np.max(d_sq_upper)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _objective_with_grad(params):
    """
    Objective: (max_d_sq / min_d_sq) with gradient.
    params: (28,) flattened coords of points 2..15.
    P[0]=(0,0), P[1]=(1,0) are fixed.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Compute all pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, 1)
    d_sq_u = d_sq[iu]
    
    min_idx = np.argmin(d_sq_u)
    max_idx = np.argmax(d_sq_u)
    
    min_d_sq = d_sq_u[min_idx]
    max_d_sq = d_sq_u[max_idx]
    
    if min_d_sq < 1e-12:
        return 1e10, np.zeros_like(params)
    
    obj = max_d_sq / min_d_sq
    
    # Gradient: d(obj)/d(P[i]) for the pairs involved
    # obj = max_d_sq / min_d_sq
    # d(obj)/d(P[i]) = (1/min_d_sq) * d(max_d_sq)/d(P[i]) - (max_d_sq/min_d_sq^2) * d(min_d_sq)/d(P[i])
    
    grad_P = np.zeros((N, 2))
    
    # For max pair (a, b): d(max_d_sq)/d(P[i]) = 2*(P[a]-P[b]) if i==a, -2*(P[a]-P[b]) if i==b, 0 otherwise
    a_max, b_max = iu[0][max_idx], iu[1][max_idx]
    diff_max = P[a_max] - P[b_max]
    grad_P[a_max] += 2.0 * diff_max
    grad_P[b_max] -= 2.0 * diff_max
    
    # For min pair (c, d): d(min_d_sq)/d(P[i]) = 2*(P[c]-P[d]) if i==c, -2*(P[c]-P[d]) if i==d, 0 otherwise
    a_min, b_min = iu[0][min_idx], iu[1][min_idx]
    diff_min = P[a_min] - P[b_min]
    grad_P[a_min] += 2.0 * diff_min
    grad_P[b_min] -= 2.0 * diff_min
    
    # Combine
    # d(obj)/d(P[i]) = (1/min_d_sq) * grad_max_P[i] - (max_d_sq/min_d_sq^2) * grad_min_P[i]
    # = (1/min_d_sq) * (grad_max_P[i] - (max_d_sq/min_d_sq) * grad_min_P[i])
    # = (1/min_d_sq) * (grad_max_P[i] - obj * grad_min_P[i])
    
    # But we need to be careful: the gradient of obj w.r.t. P is:
    # d(obj)/d(P[i]) = (1/min_d_sq) * d(max_d_sq)/d(P[i]) - (max_d_sq / min_d_sq^2) * d(min_d_sq)/d(P[i])
    
    coeff_max = 1.0 / min_d_sq
    coeff_min = max_d_sq / (min_d_sq * min_d_sq)
    
    grad_P_total = coeff_max * grad_P
    
    # Wait, I need to separate the contributions. Let me redo:
    grad_P = np.zeros((N, 2))
    grad_P[a_max] += 2.0 * diff_max
    grad_P[b_max] -= 2.0 * diff_max
    grad_max = grad_P.copy()
    
    grad_P = np.zeros((N, 2))
    grad_P[a_min] += 2.0 * diff_min
    grad_P[b_min] -= 2.0 * diff_min
    grad_min = grad_P.copy()
    
    grad_P_total = coeff_max * grad_max - coeff_min * grad_min
    
    # Only points 2..15 are free parameters
    grad_params = grad_P_total[2:].flatten()
    
    return obj, grad_params

def _objective_no_grad(params):
    """Objective without gradient for Nelder-Mead."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _get_initial_candidates():
    """Generate diverse initial configurations as (N,2) arrays."""
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. 4x4 Grid (baseline, score=8)
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    
    # 2. Hexagonal lattice subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.5:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        # Pick 16 points - try to pick spread out ones
        # Just take first 16 or use random selection
        idx = rng.choice(len(hex_points), N, replace=False)
        candidates.append(hex_points[idx])
    
    # 3. Two concentric rings (8 inner, 8 outer)
    pts = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts.append([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts.append([2.0 * np.cos(ang), 2.0 * np.sin(ang)])
    candidates.append(np.array(pts))
    
    # 4. Ring of 16
    pts = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts.append([1.5 * np.cos(ang), 1.5 * np.sin(ang)])
    candidates.append(np.array(pts))
    
    # 5. Random perturbed grid
    perturbed = grid + rng.normal(0, 0.15, grid.shape)
    candidates.append(perturbed)
    
    # 6. Random uniform in square
    rand_pts = rng.uniform(-1.5, 1.5, (N, 2))
    candidates.append(rand_pts)
    
    # 7. Two rings with different radii
    pts = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts.append([0.8 * np.cos(ang), 0.8 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.25) / 8
        pts.append([1.8 * np.cos(ang), 1.8 * np.sin(ang)])
    candidates.append(np.array(pts))
    
    # 8. Square with corners
    pts = []
    # 4 corners
    pts.append([-1.5, -1.5])
    pts.append([1.5, -1.5])
    pts.append([1.5, 1.5])
    pts.append([-1.5, 1.5])
    # 4 edge midpoints
    pts.append([0, -1.5])
    pts.append([1.5, 0])
    pts.append([0, 1.5])
    pts.append([-1.5, 0])
    # 8 inner points in a 3x3 minus center
    for i in [-0.75, 0, 0.75]:
        for j in [-0.75, 0, 0.75]:
            if i == 0 and j == 0:
                continue
            pts.append([i, j])
    # That's 16 points
    candidates.append(np.array(pts[:16]))
    
    # 9. More random variations
    for seed in [7, 13, 21]:
        rng2 = np.random.default_rng(seed)
        pts = rng2.uniform(-1.5, 1.5, (N, 2))
        candidates.append(pts)
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/translating/scaling so that pair becomes (0,0) and (1,0).
    Returns params (28,) or None.
    """
    d_sq = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, 1)
    d_sq_u = d_sq[iu]
    min_idx = np.argmin(d_sq_u)
    p, q = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R_mat.T / norm_v
    
    # Reorder: p -> index 0, q -> index 1
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    init_params = transformed[2:].flatten()
    
    return init_params

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit, leave 20s buffer)
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    
    results = []
    
    # Phase 1: L-BFGS-B with gradient from multiple starts
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 40:
            break
        
        init_params = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        try:
            res = minimize(
                _objective_with_grad,
                init_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            P_opt = np.zeros((N, 2))
            P_opt[0] = [0.0, 0.0]
            P_opt[1] = [1.0, 0.0]
            P_opt[2:] = res.x.reshape(-1, 2)
            
            score = _score(P_opt)
            
            if score < best_score:
                best_score = score
                best_params = res.x.copy()
            
            results.append((score, res.x.copy()))
            
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on top candidates
    results.sort(key=lambda x: x[0])
    
    top_k = min(5, len(results))
    for score, params in results[:top_k]:
        if time.time() > deadline - 10:
            break
        
        try:
            res_nm = minimize(
                _objective_no_grad,
                params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            P_final = np.zeros((N, 2))
            P_final[0] = [0.0, 0.0]
            P_final[1] = [1.0, 0.0]
            P_final[2:] = res_nm.x.reshape(-1, 2)
            
            final_score = _score(P_final)
            
            if final_score < best_score:
                best_score = final_score
                best_params = res_nm.x.copy()
                
        except Exception:
            pass
    
    # Phase 3: If we have time, do another round of L-BFGS-B from the best result
    # with slight perturbations
    if best_params is not None and time.time() < deadline - 15:
        rng = np.random.default_rng(99)
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + rng.normal(0, 0.01, best_params.shape)
            try:
                res = minimize(
                    _objective_with_grad,
                    perturbed,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                P_opt = np.zeros((N, 2))
                P_opt[0] = [0.0, 0.0]
                P_opt[1] = [1.0, 0.0]
                P_opt[2:] = res.x.reshape(-1, 2)
                score = _score(P_opt)
                if score < best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Validate
        final_score = _score(final_pts)
        if final_score > 1e6:
            # Fallback to grid
            return np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
        
        return final_pts
    else:
        # Fallback to grid
        return np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)

# EVOLVE-BLOCK-END
