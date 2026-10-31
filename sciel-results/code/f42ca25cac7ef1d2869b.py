import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2
NP = N * (N - 1) // 2  # number of pairwise distances

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
    Smooth surrogate objective with temperature.
    Minimizes the soft spread of squared distances.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    # Reconstruct P
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max and soft min using log-sum-exp with temperature
    # Scale by 1/T to control sharpness
    
    inv_T = 1.0 / T
    
    # Soft max: (1/T) * log(sum(exp(T * d_sq)))
    m = np.max(d_sq)
    exp_max = np.exp(inv_T * (d_sq - m))
    soft_max = m + T * np.log(np.sum(exp_max))
    
    # Soft min: -(1/T) * log(sum(exp(-T * d_sq)))
    m_neg = np.max(-d_sq)  # which is -min(d_sq)
    exp_min = np.exp(inv_T * (-d_sq + m_neg))
    soft_min = -m_neg - T * np.log(np.sum(exp_min))
    
    # We want to minimize the ratio soft_max / soft_min
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _soft_spread_gradient(params, T=1.0):
    """
    Compute gradient of the soft spread objective.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    inv_T = 1.0 / T
    
    # Soft max weights
    m = np.max(d_sq)
    exp_max = np.exp(inv_T * (d_sq - m))
    w_max = exp_max / np.sum(exp_max)
    soft_max = m + T * np.log(np.sum(exp_max))
    
    # Soft min weights
    m_neg = np.max(-d_sq)
    exp_min = np.exp(inv_T * (-d_sq + m_neg))
    w_min = exp_min / np.sum(exp_min)
    soft_min = -m_neg - T * np.log(np.sum(exp_min))
    
    if soft_min < 1e-8:
        return np.zeros_like(params)
    
    # Objective: f = soft_max / soft_min
    # df/d(soft_max) = 1/soft_min
    # df/d(soft_min) = -soft_max / soft_min^2
    
    # Gradient of soft_max w.r.t. d_sq: T * w_max
    # Gradient of soft_min w.r.t. d_sq: -T * w_min
    
    grad_d_sq = (1.0 / soft_min) * T * w_max + (soft_max / (soft_min ** 2)) * T * w_min
    
    # Now compute gradient of d_sq w.r.t. params
    # d_sq[i,j] = ||P_i - P_j||^2
    # d(d_sq[i,j])/d(P_k) = 2*(P_k - P_j) if k==i, 2*(P_i - P_k) if k==j, 0 otherwise
    
    grad_P = np.zeros((N, 2))
    iu = np.triu_indices(N, 1)
    for idx in range(NP):
        i, j = iu[0][idx], iu[1][idx]
        g = 2.0 * grad_d_sq[idx]
        grad_P[i] += g * (P[i] - P[j])
        grad_P[j] += g * (P[j] - P[i])
    
    # Only points 2..N-1 are free
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
    
    # Identify outer ring (dist^2 == 4)
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # Pick several combinations of 3 removals from outer ring
    if len(outer_indices) >= 3:
        # Deterministic: remove first 3
        remove_idx = list(outer_indices[:3])
        keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
        pts_hex = hex_points[keep_idx]
        candidates.append(pts_hex)
        
        # Random removals for diversity
        rng = np.random.default_rng(42)
        for _ in range(8):
            remove_idx = list(rng.choice(outer_indices, 3, replace=False))
            keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
            pts_hex = hex_points[keep_idx]
            candidates.append(pts_hex)

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
    
    # 4. Random Perturbed Grid
    rng = np.random.default_rng(42)
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    # 5. Random Gaussian Cluster
    rng = np.random.default_rng(123)
    gauss = rng.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([2 * np.cos(ang), 2 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    # 7. Two rings: 8 and 8, different radii
    pts_2rings = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_2rings.append([1.5 * np.cos(ang), 1.5 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_2rings.append([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
    candidates.append(np.array(pts_2rings))
    
    # 8. Three rings: 6, 5, 5
    pts_3rings = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_3rings.append([1.8 * np.cos(ang), 1.8 * np.sin(ang)])
    for k in range(5):
        ang = 2 * np.pi * (k + 0.5) / 5
        pts_3rings.append([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    for k in range(5):
        ang = 2 * np.pi * k / 5
        pts_3rings.append([0.4 * np.cos(ang), 0.4 * np.sin(ang)])
    candidates.append(np.array(pts_3rings))
    
    # 9. Perturbed hexagonal with more points
    rng = np.random.default_rng(99)
    hex_perturbed = hex_points[:16] + rng.normal(0, 0.15, (16, 2))
    candidates.append(hex_perturbed)
    
    # 10. Square spiral pattern
    pts_spiral = []
    angles = np.linspace(0, 4 * np.pi, 16, endpoint=False)
    radii = np.linspace(0.3, 2.0, 16)
    for a, r in zip(angles, radii):
        pts_spiral.append([r * np.cos(a), r * np.sin(a)])
    candidates.append(np.array(pts_spiral))

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

def _refine_with_annealing(params, start_T=1.0, end_T=0.1, n_steps=5):
    """
    Refine params using annealed soft objective with L-BFGS-B.
    """
    current_params = params.copy()
    
    for step in range(n_steps):
        T = start_T * (end_T / start_T) ** (step / max(n_steps - 1, 1))
        
        def obj(p):
            return _soft_spread_objective(p, T=T)
        
        def grad(p):
            return _soft_spread_gradient(p, T=T)
        
        try:
            res = minimize(
                obj,
                current_params,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_params = res.x
        except Exception:
            break
    
    return current_params

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    best_perm = None
    best_initial_P = None
    
    results = []
    
    # Phase 1: Smooth Optimization with annealing (L-BFGS-B with gradient)
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 40:  # Leave 40s for final polish
            break
            
        init_params, perm = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        # Run annealed L-BFGS-B
        try:
            refined_params = _refine_with_annealing(init_params, start_T=1.0, end_T=0.1, n_steps=4)
            
            # Evaluate exact score of the result
            P_opt = np.zeros((N, 2))
            P_opt[0] = [0.0, 0.0]
            P_opt[1] = [1.0, 0.0]
            P_opt[2:] = refined_params.reshape(-1, 2)
            
            score = _score(P_opt)
            
            if score < best_score:
                best_score = score
                best_params = refined_params
                best_perm = perm
                best_initial_P = P0
            
            results.append((score, refined_params, perm, P0))
                
        except Exception:
            pass

    # Phase 2: Local Refinement (Nelder-Mead) on the top K candidates
    # Sort results by score
    results.sort(key=lambda x: x[0])
    
    top_k = 5
    if time.time() < deadline - 15:
        for score, params, perm, P0 in results[:top_k]:
            if time.time() > deadline - 5:
                break
            try:
                # First, do a tight L-BFGS-B with very low temperature
                def tight_obj(p):
                    return _soft_spread_objective(p, T=0.05)
                
                def tight_grad(p):
                    return _soft_spread_gradient(p, T=0.05)
                
                res_tight = minimize(
                    tight_obj,
                    params,
                    jac=tight_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                # Then Nelder-Mead on exact objective
                res_final = minimize(
                    _exact_objective,
                    res_tight.x,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                P_final = np.zeros((N, 2))
                P_final[0] = [0.0, 0.0]
                P_final[1] = [1.0, 0.0]
                P_final[2:] = res_final.x.reshape(-1, 2)
                
                final_score = _score(P_final)
                
                if final_score < best_score:
                    best_score = final_score
                    best_params = res_final.x
                    best_perm = perm
                    
            except Exception:
                pass

    # Phase 3: Final high-precision polish on the best result
    if best_params is not None and time.time() < deadline - 3:
        try:
            # Very tight Nelder-Mead
            res_polish = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            P_polish = np.zeros((N, 2))
            P_polish[0] = [0.0, 0.0]
            P_polish[1] = [1.0, 0.0]
            P_polish[2:] = res_polish.x.reshape(-1, 2)
            
            polish_score = _score(P_polish)
            
            if polish_score < best_score:
                best_score = polish_score
                best_params = res_polish.x
                
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
