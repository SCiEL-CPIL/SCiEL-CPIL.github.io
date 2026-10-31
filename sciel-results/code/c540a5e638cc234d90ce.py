import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

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

def _soft_spread_objective(params, T=1.0):
    """
    Smooth surrogate objective for L-BFGS-B.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max and soft min using log-sum-exp
    # We want to minimize soft_max / soft_min
    # To avoid numerical issues with exp, we use standard log-sum-exp tricks
    
    m = np.max(d_sq)
    exp_max = np.exp(d_sq - m)
    soft_max = m + np.log(np.sum(exp_max))
    
    m_neg = np.max(-d_sq)
    exp_min = np.exp(-d_sq + m_neg)
    soft_min = -m_neg - np.log(np.sum(exp_min))
    
    if soft_min < 1e-6:
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
    
    iu = np.triu_indices(N, 1)
    pairs_i = iu[0]
    pairs_j = iu[1]
    num_pairs = len(pairs_i)
    
    grad_sm = np.zeros((N, 2))
    grad_smin = np.zeros((N, 2))
    
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

def _de_objective(params):
    """
    Objective for Differential Evolution.
    params: flattened array of shape (2*N,) representing all N points.
    We fix P[0]=(0,0) and P[1]=(1,0) inside the function to reduce dimensionality?
    No, DE works better with full dimensionality sometimes, but fixing reduces search space.
    Let's fix P[0] and P[1] in the parameterization to save 4 dims.
    So params shape is (2*(N-2),).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Check for distinct points
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    
    if min_d_sq < 1e-4:
        return 1e10 + (1e-4 - min_d_sq)
    
    max_d_sq = np.max(d_sq)
    return max_d_sq / min_d_sq

def _get_hex_seed():
    """
    Generate a seed from a triangular lattice.
    We want 16 points.
    A hexagonal cluster of radius 2 has 1 + 6 + 12 = 19 points.
    We can remove 3 points from the outer ring to get 16.
    The outer ring has 12 points.
    Removing 3 points creates "gaps". The best gaps are likely opposite or symmetric.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate lattice points within radius 2
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                pts.append(p)
    
    pts = np.array(pts)
    
    # Identify outer ring
    d_sqs = np.sum(pts**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    inner_indices = np.where(np.isclose(d_sqs, 1.0) | np.isclose(d_sqs, 0.0))[0] # Center and radius 1
    
    # We have 1 center, 6 radius 1, 12 radius 2. Total 19.
    # We need to remove 3 from the 12 outer points.
    # To maximize min distance, we should remove points that are close to each other?
    # Actually, removing points increases the min distance between remaining points?
    # No, removing points *cannot* decrease the min distance.
    # We want to minimize the max distance.
    # The max distance is the diameter.
    # If we remove points from the outer ring, the diameter might decrease if we remove the "far" points?
    # The diameter is determined by the farthest pair.
    # In a hexagon, the diameter is between opposite vertices.
    # If we remove vertices, we might break the long diagonals.
    
    best_score = float('inf')
    best_pts = None
    
    # Try all combinations of 3 removals from the 12 outer points
    # C(12, 3) = 220 combinations. Very fast.
    for removal_combo in itertools.combinations(outer_indices, 3):
        removal_set = set(removal_combo)
        keep_idx = [i for i in range(len(pts)) if i not in removal_set]
        candidate_pts = pts[keep_idx]
        
        s = _score(candidate_pts)
        if s < best_score:
            best_score = s
            best_pts = candidate_pts
            
    return best_pts

def _prepare_params(P):
    """
    Convert a full (N,2) point array to the reduced parameter vector
    used by the optimizers (fixing P0=(0,0) and P1=(1,0)).
    Returns None if the transformation is degenerate.
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
        return None
        
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
    
    return init_params

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Generate Seed
    P_seed = _get_hex_seed()
    
    if P_seed is None:
        # Fallback to random
        P_seed = np.random.randn(N, 2)
    
    # 2. Prepare Initial Parameters
    init_params = _prepare_params(P_seed)
    
    if init_params is None:
        # Fallback
        P_fallback = np.random.randn(N, 2)
        init_params = _prepare_params(P_fallback)
        if init_params is None:
             # Absolute fallback
             return np.array([[i, j] for i in range(4) for j in range(4)])

    best_params = init_params.copy()
    best_score = _score(np.vstack([np.array([[0,0],[1,0]]), init_params.reshape(-1,2)]))
    
    # 3. Differential Evolution (Global Search)
    # We optimize the 28 variables (14 points * 2 coords)
    # Bounds: The cluster should be within [-5, 5]
    bounds = [(-5.0, 5.0)] * (2 * (N - 2))
    
    if time.time() < deadline - 30: # Leave 30s for polishing
        try:
            de_res = differential_evolution(
                _de_objective,
                bounds=bounds,
                x0=init_params,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False # We will polish separately
            )
            
            de_score = _de_objective(de_res.x)
            if de_score < best_score:
                best_score = de_score
                best_params = de_res.x.copy()
                
        except Exception:
            pass
            
    # 4. Local Refinement (L-BFGS-B)
    # Use the soft objective for smooth gradients
    if time.time() < deadline - 10:
        try:
            lbfgs_res = minimize(
                _soft_spread_objective,
                best_params,
                jac=_soft_spread_gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            lbfgs_score = _soft_spread_objective(lbfgs_res.x)
            # Compare exact scores
            P_lbfgs = np.vstack([np.array([[0,0],[1,0]]), lbfgs_res.x.reshape(-1,2)])
            exact_lbfgs_score = _score(P_lbfgs)
            
            if exact_lbfgs_score < best_score:
                best_score = exact_lbfgs_score
                best_params = lbfgs_res.x.copy()
                
        except Exception:
            pass
            
    # 5. Final Polish (Powell or Nelder-Mead on exact objective)
    # Powell is robust for non-smooth objectives
    if time.time() < deadline - 2:
        try:
            powell_res = minimize(
                _de_objective, # Using the exact ratio objective
                best_params,
                method='Powell',
                options={'maxiter': 100, 'maxfev': 1000, 'disp': False}
            )
            
            powell_score = _de_objective(powell_res.x)
            if powell_score < best_score:
                best_score = powell_score
                best_params = powell_res.x.copy()
                
        except Exception:
            pass

    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Check validity
    if _score(final_pts) > 1e6:
        return P_seed
        
    return final_pts

# EVOLVE-BLOCK-END
