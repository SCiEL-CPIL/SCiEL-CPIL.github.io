import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0, 60, ..., 300)
# These define the constraints for the outer hexagon
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

# Unit hexagon vertices (side length 1)
UNIT_VERTS = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

def compute_required_side_and_penalty(C, A):
    """
    C: (N, 2) array of centers
    A: (N,) array of angles in degrees
    
    Returns:
    side: The minimum side length of the outer hexagon (centered at origin, rot 0) to enclose all inner hexagons.
    penalty: A penalty term for overlapping inner hexagons.
    """
    # Convert angles to radians
    A_rad = np.deg2rad(A)
    
    # Compute vertices for all hexagons
    # We need to rotate UNIT_VERTS by A_rad for each hexagon
    # c, s = cos(A_rad), sin(A_rad)
    c = np.cos(A_rad)
    s = np.sin(A_rad)
    
    # Rotation matrix for each hexagon: [[c, -s], [s, c]]
    # Apply to UNIT_VERTS (6, 2)
    # rotated_verts[i] = UNIT_VERTS @ R_i^T
    # R_i = [[c_i, -s_i], [s_i, c_i]]
    # R_i^T = [[c_i, s_i], [-s_i, c_i]]
    # v_rot = v @ R_i^T
    # v_rot[:, 0] = v[:, 0]*c + v[:, 1]*(-s) -> v[:, 0]*c - v[:, 1]*s
    # v_rot[:, 1] = v[:, 0]*s + v[:, 1]*c
    
    # Efficient calculation
    # UNIT_VERTS is (6, 2)
    # c, s are (N,)
    # We want rotated_verts of shape (N, 6, 2)
    
    # v_x = UNIT_VERTS[:, 0] (6,)
    # v_y = UNIT_VERTS[:, 1] (6,)
    
    vx = UNIT_VERTS[:, 0]
    vy = UNIT_VERTS[:, 1]
    
    # rotated_vx[i, j] = vx[j] * c[i] - vy[j] * s[i]
    # rotated_vy[i, j] = vx[j] * s[i] + vy[j] * c[i]
    
    rotated_vx = vx[None, :] * c[:, None] - vy[None, :] * s[:, None]
    rotated_vy = vx[None, :] * s[:, None] + vy[None, :] * c[:, None]
    
    # Add centers
    # C is (N, 2)
    verts = np.stack([
        rotated_vx + C[:, 0:1],
        rotated_vy + C[:, 1:2]
    ], axis=-1) # Shape (N, 6, 2)
    
    # 1. Compute Required Side Length
    # For each of the 6 normals n_k, compute max projection of all vertices.
    # proj[i, k] = max_j (verts[i, j] @ n_k)
    # verts @ n_k -> (N, 6)
    # We need max over the 6 vertices for each hexagon i and normal k.
    
    # proj_max[i, k] = max_j (verts[i, j, 0]*n_k[0] + verts[i, j, 1]*n_k[1])
    
    # Let's compute all projections: (N, 6, 6) -> N hexagons, 6 vertices, 6 normals
    # proj_all[i, j, k] = verts[i, j] @ NORM[k]
    
    # Using einsum or matmul
    # verts (N, 6, 2) @ NORM.T (2, 6) -> (N, 6, 6)
    proj_all = np.einsum('ijk,kl->ijl', verts, NORM)
    
    # Max over vertices (axis=1)
    max_proj = np.max(proj_all, axis=1) # (N, 6)
    
    # The outer hexagon is centered at origin.
    # The constraint is max_proj[i, k] <= apothem_outer
    # apothem_outer = S * S3 / 2
    # So S >= 2 * max(max_proj) / S3
    
    m = np.max(max_proj)
    if m < 0: m = 0 # Should not happen if origin is inside, but safe
    side = 2.0 * m / S3
    
    # 2. Compute Overlap Penalty
    # For each pair (i, j), check if they overlap.
    # Two convex polygons are disjoint iff there is a separating axis.
    # For two regular hexagons, the separating axes are the 6 normals of each hexagon.
    # Total 12 axes per pair.
    
    # This is computationally heavy to do exactly in a vectorized way for the objective function 
    # because the "separating axis" condition involves a max over axes and a check if min separation > 0.
    # 
    # A simpler, differentiable approximation for the objective function is to use the distance between centers
    # and a heuristic based on the minimum distance between edges, or simply use a "soft" overlap penalty.
    #
    # However, a common and effective trick is to compute the "Minkowski Sum" distance or similar.
    # But for simplicity and speed in the objective, we can use the following:
    # For each pair, compute the minimum distance between their boundaries.
    # If they overlap, distance is negative.
    #
    # A cheaper approximation:
    # The distance between centers is d.
    # The "effective radius" in direction u is the support function h(u).
    # If h_i(u) + h_j(-u) > d . u for any u, they might overlap.
    # Actually, they overlap iff max_u (h_i(u) + h_j(-u) - d . u) > 0.
    # This is equivalent to checking if the distance between the two polygons is negative.
    #
    # Let's use a simpler penalty: sum of squared overlaps approximated by center distance vs sum of circumradii? No, that's too loose.
    #
    # Let's use the exact separating axis theorem check but vectorized over pairs?
    # That's hard to vectorize cleanly in a single NumPy op without loops.
    # Given N=11, there are 55 pairs. A Python loop over 55 pairs is fast enough if the inner logic is vectorized over axes.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Axes: 6 normals of i, 6 normals of j
            # Normals of i: R_i @ NORM
            # Normals of j: R_j @ NORM
            
            # R_i = [[c_i, -s_i], [s_i, c_i]]
            # Normals_i = NORM @ R_i.T
            # NORM is (6, 2). R_i.T is (2, 2).
            
            # We already have c, s.
            # Let's compute the 12 axes for this pair.
            
            # Axis 1: NORM (6, 2)
            # Axis 2: Rotated NORM for j relative to i?
            # Actually, the separating axis theorem for two convex polygons P and Q says:
            # They are disjoint iff there exists a normal n to an edge of P or Q such that proj_P(n) and proj_Q(n) are disjoint.
            
            # Let's compute projections of both hexagons onto these 12 axes.
            
            # Projections of hex i onto its own normals:
            # Since hex i is regular, its projection onto its own normals is simply [-1, 1] shifted by center?
            # No, the support function of a unit hexagon in direction n_k (where n_k is a normal) is 1.
            # So proj_i(n_k) = C_i . n_k + 1.
            # And proj_i(-n_k) = C_i . (-n_k) + 1.
            
            # Projections of hex j onto hex i's normals:
            # We need to compute max_dot(j_verts, n_k) and min_dot(j_verts, n_k).
            
            # Let's just use the vertices we already computed.
            # verts_i = verts[i] (6, 2)
            # verts_j = verts[j] (6, 2)
            
            # Axes:
            # 6 normals of i: N_i = NORM (since we defined NORM as the normals of the unit hexagon at angle 0)
            # Wait, the normals rotate with the hexagon.
            # The normals of hex i are: R_i @ NORM.
            # The normals of hex j are: R_j @ NORM.
            
            # Let's compute the 12 axes.
            # Normals_i = R_i @ NORM
            # R_i = [[c_i, -s_i], [s_i, c_i]]
            # Normals_i = [[c_i, -s_i], [s_i, c_i]] @ NORM.T -> (2, 6).T -> (6, 2)
            # Actually, NORM @ R_i.T is easier to compute with existing c, s.
            # n_k = (nx, ny). n_k_rot = (nx*c - ny*s, nx*s + ny*c)
            
            # Let's precompute rotated normals for all hexagons.
            # This is (N, 6, 2).
            # We can do this once outside the loop if we pass it, or compute here.
            # Since we are in the objective, let's compute it efficiently.
            
            # For pair (i, j), we need:
            # Axes = vstack([Normals_i, Normals_j])
            # Proj_i = verts_i @ Axes.T -> (6, 12). Max and Min.
            # Proj_j = verts_j @ Axes.T -> (6, 12). Max and Min.
            
            # If for any axis k, max(Proj_i[:, k]) < min(Proj_j[:, k]) or max(Proj_j[:, k]) < min(Proj_i[:, k]), they are separated.
            # If not, they overlap.
            
            # To make this differentiable-ish for the optimizer, we can penalize the "penetration depth".
            # Penetration depth along axis k = max(0, max(Proj_i[:, k]) - min(Proj_j[:, k]))
            # Total penetration = sum_k max(0, ...)
            
            # Let's compute rotated normals for i and j.
            # n_i_rot = NORM @ [[c_i, s_i], [-s_i, c_i]]
            # n_i_rot[:, 0] = NORM[:, 0]*c_i + NORM[:, 1]*(-s_i)
            # n_i_rot[:, 1] = NORM[:, 0]*s_i + NORM[:, 1]*c_i
            
            ni0 = NORM[:, 0] * c[i] - NORM[:, 1] * s[i]
            ni1 = NORM[:, 0] * s[i] + NORM[:, 1] * c[i]
            Normals_i = np.stack([ni0, ni1], axis=1) # (6, 2)
            
            nj0 = NORM[:, 0] * c[j] - NORM[:, 1] * s[j]
            nj1 = NORM[:, 0] * s[j] + NORM[:, 1] * c[j]
            Normals_j = np.stack([nj0, nj1], axis=1) # (6, 2)
            
            Axes = np.vstack([Normals_i, Normals_j]) # (12, 2)
            
            # Projections
            # verts_i is (6, 2)
            # Projs_i = verts_i @ Axes.T -> (6, 12)
            Projs_i = verts[i] @ Axes.T
            Projs_j = verts[j] @ Axes.T
            
            max_i = np.max(Projs_i, axis=0) # (12,)
            min_i = np.min(Projs_i, axis=0)
            max_j = np.max(Projs_j, axis=0)
            min_j = np.min(Projs_j, axis=0)
            
            # Separation along axis k:
            # If max_i[k] < min_j[k], separated by min_j[k] - max_i[k]
            # If max_j[k] < min_i[k], separated by min_i[k] - max_j[k]
            # Overlap if neither.
            
            # Penetration depth:
            # d1 = min_j - max_i (positive if separated)
            # d2 = min_i - max_j (positive if separated)
            # If both negative, overlap.
            # The "depth" of overlap is max(0, -min(d1, d2))? No.
            # If they overlap, d1 < 0 and d2 < 0.
            # The penetration depth is the minimum amount you need to move to separate.
            # This is roughly -min(d1, d2) if we consider the direction.
            # Actually, if d1 < 0 and d2 < 0, the overlap exists.
            # A simple penalty is sum of (max(0, -d1) + max(0, -d2))? No, that double counts.
            # Let's use: penalty += sum_k max(0, max_i[k] - min_j[k]) + max(0, max_j[k] - min_i[k])
            # This sums the "interference" along each axis.
            
            pen_1 = np.maximum(0, max_i - min_j)
            pen_2 = np.maximum(0, max_j - min_i)
            
            penalty += np.sum(pen_1) + np.sum(pen_2)
            
    return side, penalty

def objective(flat):
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    side, penalty = compute_required_side_and_penalty(C, A)
    return side + 10.0 * penalty

def init_flower():
    """
    1 center, 6 surrounding.
    """
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        # Distance between centers of touching hexagons is 2 * apothem = S3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    # Add 4 more in a second ring
    # Positions at distance 2*S3, angles 30, 90, 150, 210
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A

def init_lattice():
    """
    Hexagonal lattice arrangement.
    """
    C = []
    # Row 0: 3 hexagons
    for i in range(3):
        C.append([(i - 1.0) * S3, 0.0])
    # Row 1: 4 hexagons
    for i in range(4):
        C.append([(i - 1.5) * S3, 1.5])
    # Row 2: 3 hexagons
    for i in range(3):
        C.append([(i - 1.0) * S3, 3.0])
    # Row 3: 1 hexagon
    C.append([0.0, 4.5])
    
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initialize with flower pattern
    C0, A0 = init_flower()
    
    # Flatten
    flat0 = np.empty(3 * N)
    flat0[0::3] = C0[:, 0]
    flat0[1::3] = A0
    flat0[2::3] = C0[:, 1]
    
    best_flat = flat0
    best_score = objective(flat0)
    
    # Try optimization
    # Powell is often better for this type of problem than Nelder-Mead
    # because it doesn't assume smoothness and handles the non-smooth penalty better.
    
    try:
        res = minimize(objective, flat0, method='Powell', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_flat = res.x
            
    except Exception as e:
        pass
    
    # If time allows, try a second run from a slightly perturbed start
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        flat1 = flat0 + rng.normal(0, 0.1, size=flat0.shape)
        
        try:
            res2 = minimize(objective, flat1, method='Powell', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res2.fun < best_score:
                best_score = res2.fun
                best_flat = res2.x
                
        except Exception:
            pass

    # Extract best solution
    C_best = best_flat[0::3].reshape(N, 2)
    A_best = best_flat[1::3]
    
    # Compute final side length
    side, _ = compute_required_side_and_penalty(C_best, A_best)
    
    # Convert to required format
    inner = []
    for i in range(N):
        inner.append([float(C_best[i, 0]), float(C_best[i, 1]), float(A_best[i])])
    
    return inner, (0.0, 0.0), float(side), 0.0

