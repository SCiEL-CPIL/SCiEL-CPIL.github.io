import numpy as np
import time
import math
from scipy.optimize import minimize, basinhopping

N = 11
SQRT3 = math.sqrt(3)
# Minimum center distance for non-overlapping unit hexagons
MIN_DIST = SQRT3

def _get_vertices(centers, angles):
    """
    Compute vertices for all inner hexagons.
    centers: (N, 2)
    angles: (N,) in radians
    Returns: (N, 6, 2)
    """
    # Relative vertices of a unit hexagon (side 1) centered at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    cos_a = np.cos(base_angles)
    sin_a = np.sin(base_angles)
    
    # Rotate and translate
    # angles is (N, 1)
    ang = angles[:, np.newaxis]
    cos_ang = np.cos(ang)
    sin_ang = np.sin(ang)
    
    # x = cx + cos(ang)*vx - sin(ang)*vy
    # y = cy + sin(ang)*vx + cos(ang)*vy
    # vx, vy are cos_a, sin_a
    
    # Broadcasting: centers (N, 1, 2), cos_a (1, 6, 1)
    cx = centers[:, 0, np.newaxis]
    cy = centers[:, 1, np.newaxis]
    vx = cos_a[np.newaxis, :, np.newaxis]
    vy = sin_a[np.newaxis, :, np.newaxis]
    
    x = cx + cos_ang * vx - sin_ang * vy
    y = cy + sin_ang * vx + cos_ang * vy
    
    verts = np.stack([x, y], axis=-1)
    return verts

def _compute_side_and_penalty(centers, angles, beta, penalty_weight):
    """
    Compute smooth approximation of outer side length and overlap penalty.
    """
    # Normals of the outer hexagon (fixed at 0 degrees rotation for the hexagon itself)
    # A regular hexagon with side S has apothem S*sqrt(3)/2.
    # If vertices are at angles 0, 60..., the edges are perpendicular to normals at 30, 90, ...
    # Let's fix outer hexagon orientation such that normals are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    verts = _get_vertices(centers, angles)
    
    # Project all vertices onto all normals
    # verts: (N, 6, 2)
    # normals: (6, 2)
    # We want (N, 6, 6) projections: N hexagons, 6 vertices each, 6 normals
    # proj[i, j, k] = dot(verts[i, j], normals[k])
    
    # Efficient way:
    # For each normal k, project all vertices.
    # proj_all = np.einsum('ijk,jk->ij', verts, normals.T) # No, normals is (6,2), verts is (N,6,2)
    # Let's do:
    # proj = np.zeros((N, 6, 6))
    # for k in range(6):
    #    proj[:, :, k] = verts @ normals[k]
    
    # Vectorized:
    # verts (N, 6, 2) @ normals.T (2, 6) -> (N, 6, 6)
    projections = np.einsum('ijk,lk->ijl', verts, normals)
    
    abs_projections = np.abs(projections)
    
    # Log-sum-exp for max
    # max_val = (1/beta) * log(sum(exp(beta * abs_projections)))
    # To avoid overflow, subtract max
    max_abs = np.max(abs_projections)
    # If max_abs is very small, exp(beta*abs) might be fine, but let's be safe.
    # We want the global max over all N, 6, 6 values.
    global_max = np.max(abs_projections)
    
    # Soft max
    exp_terms = np.exp(beta * (abs_projections - global_max))
    sum_exp = np.sum(exp_terms)
    if sum_exp > 0:
        soft_max = global_max + (1.0 / beta) * np.log(sum_exp)
    else:
        soft_max = global_max
        
    side_length = 2.0 * soft_max / SQRT3
    
    # Overlap penalty
    # centers: (N, 2)
    # Compute pairwise distances
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
    # dist = np.linalg.norm(diff, axis=-1) # (N, N)
    # Only upper triangle
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(dist_sq + 1e-12)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dist[mask]
    
    # Penalty for d < MIN_DIST
    violations = np.maximum(0.0, MIN_DIST - pair_dists)
    penalty = penalty_weight * np.sum(violations ** 2)
    
    return side_length + penalty

def construct():
    start_time = time.time()
    DEADLINE = 145.0 # Safety margin for 150s limit
    
    # Initial configurations
    # 1. Seed: 1 center, 6 ring, 4 outer
    def _make_seed():
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center
        centers[0] = [0.0, 0.0]
        angles[0] = 0.0
        
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = np.deg2rad(k * 60)
            r = SQRT3
            centers[k+1, 0] = r * np.cos(ang)
            centers[k+1, 1] = r * np.sin(ang)
            angles[k+1] = 0.0
            
        # Ring 2: 4 hexagons. 
        # Place them in the "bays" between ring 1 hexagons to minimize bounding box.
        # Bays are at angles 30, 90, 150, 210, 270, 330.
        # We pick 4. Symmetric choice: 30, 150, 270, 30? No.
        # Let's pick 30, 90, 150, 210? No, that's not symmetric.
        # Let's pick 30, 150, 270, 30 is invalid.
        # Let's pick 30, 90, 210, 270?
        # Actually, just placing them at distance 2*SQRT3 from center at angles 30, 150, 270, 30?
        # Let's try angles 30, 90, 150, 210 at distance ~2.8?
        # A better heuristic: Place them at the vertices of the "super-hexagon" formed by the 7-hex cluster.
        # The 7-hex cluster has "corners" at distance 2*SQRT3? No.
        # Let's just use the same angles as the gaps: 30, 90, 150, 210.
        # And distance 2.5?
        # Let's use the seed from the original code but ensure no overlap.
        # Original code used distance 3.0.
        gap_angles = [30, 90, 150, 210]
        for k in range(4):
            ang = np.deg2rad(gap_angles[k])
            r = 2.6 # Slightly less than 3 to tighten
            centers[k+7, 0] = r * np.cos(ang)
            centers[k+7, 1] = r * np.sin(ang)
            angles[k+7] = 0.0
            
        return centers, angles

    def _make_random():
        centers = np.random.uniform(-2, 2, (N, 2))
        angles = np.random.uniform(0, 2*np.pi, N)
        return centers, angles

    # Objective wrapper for optimizer
    # We need to pass beta and penalty_weight. We will use a closure or global variables.
    # Since we can't easily pass args to basinhopping's local optimizer, we'll use a mutable container or global.
    # Let's use a class or global dict.
    opt_params = {'beta': 10.0, 'penalty_weight': 10.0}

    def objective(params):
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        # Angles are in radians
        return _compute_side_and_penalty(centers, angles, opt_params['beta'], opt_params['penalty_weight'])

    # Initial best
    best_score = float('inf')
    best_params = None
    
    seeds = [
        _make_seed(),
        _make_random(),
        _make_random()
    ]
    
    # Add more diverse seeds
    # Shifted seed
    c, a = _make_seed()
    c += np.array([0.1, 0.1])
    seeds.append((c, a))
    
    for centers, angles in seeds:
        if time.time() > DEADLINE - 10:
            break
        init_params = np.concatenate([centers.flatten(), angles.flatten()])
        
        # Multi-start with Basin Hopping
        # Basin hopping uses a local optimizer (L-BFGS-B) and a global random walk.
        
        # Anneal beta and penalty
        # Start with low beta (smooth) and high penalty (to avoid overlap)
        # Then increase beta (sharp max) and decrease penalty (to allow closer packing)
        
        current_params = init_params.copy()
        
        # Phase 1: Smooth objective, high penalty
        opt_params['beta'] = 5.0
        opt_params['penalty_weight'] = 100.0
        
        try:
            res = basinhopping(objective, current_params, niter=20, T=0.5, stepsize=0.5, 
                               minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
            current_params = res.x.copy()
        except Exception:
            pass
            
        # Phase 2: Sharper objective, lower penalty
        if time.time() > DEADLINE - 20:
            continue
            
        opt_params['beta'] = 50.0
        opt_params['penalty_weight'] = 10.0
        
        try:
            res = basinhopping(objective, current_params, niter=20, T=0.2, stepsize=0.2,
                               minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
            current_params = res.x.copy()
        except Exception:
            pass
            
        # Phase 3: Very sharp objective, minimal penalty (rely on constraints being satisfied)
        if time.time() > DEADLINE - 30:
            continue
            
        opt_params['beta'] = 200.0
        opt_params['penalty_weight'] = 1.0
        
        try:
            res = basinhopping(objective, current_params, niter=10, T=0.1, stepsize=0.1,
                               minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If we still have time, do a final polish on the best candidate found
    if best_params is not None and time.time() < DEADLINE - 5:
        # Use a very high beta and no penalty to get the exact side length
        # But we must ensure no overlap.
        # Let's just evaluate the best candidate with high beta to get a good estimate.
        opt_params['beta'] = 1000.0
        opt_params['penalty_weight'] = 0.0
        
        # Local polish
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract result
    if best_params is None:
        # Fallback to seed
        centers, angles = _make_seed()
        best_params = np.concatenate([centers.flatten(), angles.flatten()])
        opt_params['beta'] = 1000.0
        opt_params['penalty_weight'] = 0.0
        best_score = objective(best_params)

    centers = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    # Calculate actual side length and check validity
    # We need to return the side length of the outer hexagon.
    # The objective with beta=1000 approximates the max projection.
    # Let's compute the exact max projection for the final configuration.
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    verts = _get_vertices(centers, angles)
    projections = np.einsum('ijk,lk->ijl', verts, normals)
    max_proj = np.max(np.abs(projections))
    exact_side = 2.0 * max_proj / SQRT3
    
    # Check for overlaps
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dist[mask]
    if np.min(pair_dists) < MIN_DIST - 1e-9:
        # Overlap detected. We should expand slightly or return a valid seed.
        # For now, if overlap, we return the seed configuration which is valid.
        centers, angles = _make_seed()
        verts = _get_vertices(centers, angles)
        projections = np.einsum('ijk,lk->ijl', verts, normals)
        max_proj = np.max(np.abs(projections))
        exact_side = 2.0 * max_proj / SQRT3

    inner_rows = []
    for i in range(N):
        inner_rows.append([float(centers[i, 0]), float(centers[i, 1]), float(np.rad2deg(angles[i]))])
    
    center = (0.0, 0.0)
    side = float(exact_side)
    angle_deg = 0.0 # Outer hexagon is fixed at 0 degrees rotation in our model
    
    return inner_rows, center, side, angle_deg

if __name__ == "__main__":
    construct()
