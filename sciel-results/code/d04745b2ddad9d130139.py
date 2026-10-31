"""
Optimized 16-point placement: minimize (max_dist/min_dist)^2.
Strategy:
1. Reduce dimensionality by fixing the closest pair of points to (0,0) and (1,0).
   This removes rigid body degrees of freedom (translation, rotation, scale)
   and fixes the minimum distance to 1.
2. The objective becomes minimizing the maximum squared distance.
3. Use Differential Evolution for global search in the 28-dimensional reduced space.
4. Polish with L-BFGS-B using a soft-max surrogate for the maximum distance.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2
DEADLINE = 145.0  # safety margin before 150s kill

# Indices for pairwise distances
IU = np.triu_indices(N, k=1)

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances. pts: (16,2) -> (120,) array of d^2."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return d2[IU]

def _prepare_params(pts):
    """
    Transform 16x2 points to 28-dim params.
    1. Find closest pair (i, j).
    2. Translate so pts[i] is at origin.
    3. Rotate so vector (pts[j]-pts[i]) is along x-axis.
    4. Scale so distance is 1.
    5. Return coords of remaining 14 points.
    """
    d2 = _pairwise_dists_sq(pts)
    min_idx = np.argmin(d2)
    i_min, j_min = IU[0][min_idx], IU[1][min_idx]
    
    p0 = pts[i_min]
    p1 = pts[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
        
    # Rotation matrix to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], 
                  [-sin_t, cos_t]])
    
    # Transform all points
    # 1. Translate by -p0
    # 2. Rotate by R
    # 3. Scale by 1/norm_v
    transformed = (pts - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, others follow
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # The first two points are now (0,0) and (1,0) (approximately)
    # We only need to store the remaining 14 points
    params = transformed[2:].flatten()
    return params

def _reconstruct(params):
    """
    Reconstruct 16x2 points from 28-dim params.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(14, 2)
    return pts

def _exact_score_reduced(params):
    """
    Calculate (max_dist/min_dist)^2 for the configuration defined by params.
    Since min_dist is fixed to 1, this is just max_dist^2.
    """
    pts = _reconstruct(params)
    d2 = _pairwise_dists_sq(pts)
    return np.max(d2)

def _soft_objective_reduced(params, beta):
    """
    Soft surrogate for minimizing max squared distance.
    Uses log-sum-exp to approximate max.
    """
    pts = _reconstruct(params)
    d2 = _pairwise_dists_sq(pts)
    d2 = np.maximum(d2, 1e-12)
    
    # Soft max: (1/beta) * log(sum(exp(beta * d2)))
    m = np.max(d2)
    soft_max = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d2 - m))))
    
    return soft_max

def _get_initial_candidates_reduced():
    """
    Generate diverse initial candidates in the reduced 28-dim space.
    """
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. 4x4 Grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid = np.column_stack([X.flatten(), Y.flatten()])
    # Center it roughly
    grid -= grid.mean(axis=0)
    p = _prepare_params(grid)
    if p is not None: candidates.append(p)
    
    # 2. Hexagonal Lattice
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Select 16 closest to center
    dists = np.sum(hex_pts**2, axis=1)
    idx = np.argsort(dists)[:16]
    hex_sel = hex_pts[idx]
    hex_sel -= hex_sel.mean(axis=0)
    p = _prepare_params(hex_sel)
    if p is not None: candidates.append(p)
    
    # 3. Concentric Circles
    ang1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ang2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    r1, r2 = 0.5, 1.5
    c1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    c2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    circ = np.vstack([c1, c2])
    circ -= circ.mean(axis=0)
    p = _prepare_params(circ)
    if p is not None: candidates.append(p)
    
    # 4. Random Uniform
    for _ in range(5):
        pts = rng.uniform(-2, 2, (N, 2))
        pts -= pts.mean(axis=0)
        p = _prepare_params(pts)
        if p is not None: candidates.append(p)
        
    # 5. Random on Ring
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.5, 2.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts -= pts.mean(axis=0)
        p = _prepare_params(pts)
        if p is not None: candidates.append(p)
        
    return candidates

def _local_optimize_reduced(params, maxiter=5000):
    """
    Polish a candidate in reduced space using L-BFGS-B on soft objective.
    Uses annealing of beta.
    """
    best_params = params.copy()
    best_score = _exact_score_reduced(best_params)
    
    # Anneal beta from low (smooth) to high (sharp)
    betas = [5.0, 20.0, 100.0, 500.0]
    
    for beta in betas:
        try:
            res = minimize(
                lambda p: _soft_objective_reduced(p, beta),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            score = _exact_score_reduced(res.x)
            if score < best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
            
    return best_params, best_score

def construct():
    t0 = time.time()
    
    # Phase 1: Global Search with Differential Evolution in reduced space (28 dims)
    best_params = None
    best_score = np.inf
    
    # Generate initial candidates
    candidates = _get_initial_candidates_reduced()
    
    # Polish initial candidates
    for p in candidates:
        if time.time() - t0 > DEADLINE - 30:
            break
        try:
            p_opt, score = _local_optimize_reduced(p, maxiter=2000)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            continue
            
    # Run DE if time permits
    if time.time() - t0 < DEADLINE - 40:
        try:
            # Define bounds for 28 params. 
            # Points should be within a reasonable range. 
            # Since min dist is 1, max dist shouldn't be huge. 
            # Let's say points are within [-5, 5] relative to the fixed pair.
            bounds = [(-5, 5)] * 28
            
            # We can use the best found so far to guide DE or just run standard DE
            # Using sobol init for better coverage
            result_de = differential_evolution(
                lambda p: _soft_objective_reduced(p, 10.0), # Use moderate beta for DE
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                popsize=20, 
                polish=True, 
                init='sobol',
                time_limit=30
            )
            
            if result_de.success or result_de.fun < best_score:
                p_de = result_de.x
                # Polish the DE result
                p_opt, score = _local_optimize_reduced(p_de, maxiter=3000)
                if score < best_score:
                    best_score = score
                    best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Random Restarts in Reduced Space
    rng = np.random.default_rng(123)
    while time.time() - t0 < DEADLINE - 10:
        try:
            # Generate random params
            p_rand = rng.uniform(-3, 3, 28)
            p_opt, score = _local_optimize_reduced(p_rand, maxiter=1000)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            break
            
    # Phase 3: Final High-Precision Polish
    if best_params is not None and time.time() - t0 < DEADLINE - 2:
        try:
            # Very high beta for final push
            res_final = minimize(
                lambda p: _soft_objective_reduced(p, 1000.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            score_final = _exact_score_reduced(res_final.x)
            if score_final < best_score:
                best_score = score_final
                best_params = res_final.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: Return a simple grid in full space (though we should return reduced? 
        # No, construct() must return (16,2) points)
        # We need to return full points.
        gx = np.array([0, 1, 2, 3], dtype=float)
        gy = np.array([0, 1, 2, 3], dtype=float)
        X, Y = np.meshgrid(gx, gy)
        best_params = _prepare_params(np.column_stack([X.flatten(), Y.flatten()]))
        if best_params is None:
             return np.column_stack([X.flatten(), Y.flatten()])

    # Reconstruct full points from best reduced params
    final_pts = _reconstruct(best_params)
    
    return final_pts

if __name__ == '__main__':
    pts = construct()
    # Calculate score for verification
    d2 = _pairwise_dists_sq(pts)
    score = np.max(d2) / np.min(d2)
    print("Score:", score)
