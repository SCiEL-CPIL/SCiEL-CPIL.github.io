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

def _hex_cluster(ratio, shear, ring_scale):
    """
    Build a 16-point hexagonal lattice cluster with 3 geometric DOF.
    
    The cluster is organized in rings:
    - Ring 0: center (1 point)
    - Ring 1: 6 points at distance 1 from center
    - Ring 2: 12 points? No, that's 19 total. Let's do a different arrangement.
    
    Actually for 16 points, let's use a different hexagonal arrangement:
    - A 4-3-4-3-2 pattern or similar compact hexagonal number arrangement.
    
    Let me use a specific compact hexagonal cluster for 16 points:
    Ring 0: 1 point (center)
    Ring 1: 6 points (first hexagonal shell)
    Ring 2: 9 more points to make 16 total (partial second shell)
    
    Actually, the hexagonal numbers are 1, 7, 19, 37... So 16 is between 7 and 19.
    Let's use a specific arrangement that's compact.
    
    I'll parameterize it as follows:
    - Base hexagonal lattice with basis vectors e1, e2 at 60 degrees
    - e1 = (1, 0)
    - e2 = (ratio * cos(60+shear), ratio * sin(60+shear))
    
    For the 16-point cluster, I'll use integer coordinates (i, j) in the hexagonal basis:
    (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), 
    (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2), (2,1), (-2,-1)
    
    That's 16 points. Let me verify they form a compact cluster.
    """
    # Hexagonal lattice basis
    angle = np.pi / 3 + shear  # 60 degrees plus shear
    e1 = np.array([1.0, 0.0])
    e2 = np.array([ratio * np.cos(angle), ratio * np.sin(angle)])
    
    # Integer coordinates for 16-point hexagonal cluster
    # These form a compact hexagonal arrangement
    coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1),  # ring 1 (6 points)
        (2, 0), (-2, 0), (1, 1), (-1, -1), (2, -1), (-2, 1),  # partial ring 2
        (1, -2), (-1, 2)  # two more to make 16
    ]
    
    points = np.array([i * e1 + j * e2 for i, j in coords])
    
    # Apply radial scaling to outermost points to break perfect symmetry
    # Points at distance > 1.5 from center get scaled by ring_scale
    dists = np.linalg.norm(points, axis=1)
    outer_mask = dists > 1.5
    points[outer_mask] *= ring_scale
    
    return points

def _hex_objective(params):
    """Objective function for hexagonal cluster optimization."""
    ratio, shear, ring_scale = params
    # Keep parameters in reasonable ranges
    if ratio < 0.5 or ratio > 2.0 or abs(shear) > np.pi/4 or ring_scale < 0.8 or ring_scale > 1.5:
        return 1e10
    P = _hex_cluster(ratio, shear, ring_scale)
    return _score(P)

def _get_hex_candidates():
    """Generate diverse hexagonal cluster candidates."""
    candidates = []
    
    # Try a grid of parameter values
    for ratio in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for shear in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for rs in [0.9, 1.0, 1.1]:
                P = _hex_cluster(ratio, shear, rs)
                candidates.append(P)
    
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

def _exact_objective(params):
    """Exact objective for local refinement."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _soft_spread_objective(params, T=1.0):
    """Smooth surrogate objective using log-sum-exp."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    m = np.max(d_sq)
    exp_max = np.exp(d_sq - m)
    soft_max = m + np.log(np.sum(exp_max))
    
    m_neg = np.max(-d_sq)
    exp_min = np.exp(-d_sq + m_neg)
    soft_min = -m_neg - np.log(np.sum(exp_min))
    
    if soft_min < 1e-6:
        return 1e10
    
    return soft_max / soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = float('inf')
    best_P = None
    
    # Phase 1: Optimize hexagonal cluster with 3 DOF
    hex_candidates = _get_hex_candidates()
    
    # Find best hexagonal candidate
    best_hex_score = float('inf')
    best_hex_params = None
    for P in hex_candidates:
        s = _score(P)
        if s < best_hex_score:
            best_hex_score = s
            # Back-calculate parameters (approximate)
            best_hex_params = np.array([1.0, 0.0, 1.0])  # Default
            best_P = P
    
    # Refine best hexagonal candidate with full optimization
    if best_P is not None and time.time() < deadline - 30:
        init_params, perm = _prepare_for_optimization(best_P)
        if init_params is not None:
            # L-BFGS-B on soft objective
            try:
                res = minimize(
                    _soft_spread_objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                P_opt = np.zeros((N, 2))
                P_opt[0] = [0.0, 0.0]
                P_opt[1] = [1.0, 0.0]
                P_opt[2:] = res.x.reshape(-1, 2)
                
                score = _score(P_opt)
                if score < best_score:
                    best_score = score
                    best_P = P_opt
            except Exception:
                pass
            
            # Powell refinement
            if time.time() < deadline - 10:
                try:
                    res_final = minimize(
                        _exact_objective,
                        init_params,
                        method='Powell',
                        options={'maxiter': 200, 'maxfev': 5000, 'disp': False}
                    )
                    
                    P_final = np.zeros((N, 2))
                    P_final[0] = [0.0, 0.0]
                    P_final[1] = [1.0, 0.0]
                    P_final[2:] = res_final.x.reshape(-1, 2)
                    
                    final_score = _score(P_final)
                    if final_score < best_score:
                        best_score = final_score
                        best_P = P_final
                except Exception:
                    pass
            
            # Nelder-Mead polish
            if time.time() < deadline - 5:
                try:
                    res_nm = minimize(
                        _exact_objective,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                    )
                    
                    P_nm = np.zeros((N, 2))
                    P_nm[0] = [0.0, 0.0]
                    P_nm[1] = [1.0, 0.0]
                    P_nm[2:] = res_nm.x.reshape(-1, 2)
                    
                    nm_score = _score(P_nm)
                    if nm_score < best_score:
                        best_score = nm_score
                        best_P = P_nm
                except Exception:
                    pass
    
    # If hexagonal approach didn't find anything good, try other configurations
    if best_score > 1.5:
        # Try triangular lattice based approaches
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        
        # Generate triangular lattice points
        tri_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                if np.sum(p**2) <= 5.0 + 1e-9:
                    tri_points.append(p)
        
        tri_points = np.array(tri_points)
        
        # Try different subsets of 16 points
        if len(tri_points) >= 16:
            # Sort by distance from center and take various combinations
            dists = np.linalg.norm(tri_points, axis=1)
            sorted_idx = np.argsort(dists)
            
            # Take the 16 closest
            P_tri = tri_points[sorted_idx[:16]]
            s_tri = _score(P_tri)
            if s_tri < best_score:
                best_score = s_tri
                best_P = P_tri
            
            # Try removing different outer points
            outer_mask = dists[sorted_idx] > 1.5
            outer_in_sorted = np.where(outer_mask)[0]
            
            if len(outer_in_sorted) >= 1:
                for remove_idx in outer_in_sorted[:5]:
                    keep = [i for i in range(16) if i != remove_idx]
                    # Add next closest point
                    next_idx = sorted_idx[16] if 16 < len(sorted_idx) else None
                    if next_idx is not None:
                        keep.append(next_idx)
                        keep = keep[:16]
                        P_test = tri_points[keep]
                        s_test = _score(P_test)
                        if s_test < best_score:
                            best_score = s_test
                            best_P = P_test
    
    # Final validation and return
    if best_P is not None:
        final_score = _score(best_P)
        if final_score < 1e6:
            return best_P
    
    # Fallback: simple grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    return grid

# EVOLVE-BLOCK-END
