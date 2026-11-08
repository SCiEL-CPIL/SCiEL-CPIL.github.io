"""Strategy: Direct optimization of hexagon centers and orientations using scipy.optimize.minimize.

Key differences from the current program:
1. Uses scipy's L-BFGS-B optimizer instead of manual force-directed relaxation.
2. Optimizes a continuous penalty function that directly penalizes overlaps and boundary violations.
3. Uses a robust, vectorized penalty calculation based on Separating Axis Theorem (SAT) logic for hexagons.
4. Employs a multi-start strategy with diverse initial configurations (lattice-like, random, ring) to escape local minima.
5. Focuses on finding a valid packing for a fixed, small S (e.g., 4.5-5.0) rather than binary searching S, as the optimal S is likely close to the theoretical minimum for N=11.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute constants
    SQRT3 = np.sqrt(3.0)
    HEX_ANGLES = np.deg2rad([0, 60, 120, 180, 240, 330]) # Wait, 330 is wrong, should be 300.
    HEX_ANGLES = np.deg2rad([0, 60, 120, 180, 240, 300])
    COS_A = np.cos(HEX_ANGLES)
    SIN_A = np.sin(HEX_ANGLES)
    VERTS_UNIT = np.column_stack((COS_A, SIN_A)) # (6, 2)
    
    # Normals for the 6 edges of a hexagon (unit normal vectors)
    # Edge i is between vertex i and i+1.
    # Normal to edge i is at angle 30 + 60*i
    EDGE_NORMAL_ANGLES = np.deg2rad([30, 90, 150, 210, 270, 330])
    EDGE_NORMALS = np.column_stack((np.cos(EDGE_NORMAL_ANGLES), np.sin(EDGE_NORMAL_ANGLES)))

    def get_hex_verts(cx, cy, angle_rad):
        # Returns (6, 2) array of vertices
        rot = angle_rad
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        return cx + VERTS_UNIT @ R.T, cy + VERTS_UNIT @ R.T, VERTS_UNIT @ R.T

    def hex_penalty(i, j, centers, angles, S):
        # Calculate penalty for overlap between hex i and j, and boundary violation for both.
        # This function is called frequently, so it should be efficient.
        # However, for simplicity in the objective function, we will compute all penalties in a loop.
        # Let's try to vectorize where possible or keep it simple first.
        pass

    def objective(params):
        # params: [x0, y0, a0, x1, y1, a1, ...]
        centers = params[0:2*N:3]
        ys = params[1:2*N+1:3]
        angles = params[2:3*N:3]
        
        # Reshape to (N, 2) and (N,)
        c = np.column_stack((centers, ys))
        a = angles
        
        penalty = 0.0
        
        # Boundary penalty
        # For each hex, check if all vertices are inside outer hex.
        # Outer hex: center (0,0), side S, rotation 0.
        # Inradius R_out = S * SQRT3 / 2
        R_out = S * SQRT3 / 2
        
        for i in range(N):
            # Get vertices of hex i
            # cx, cy, verts_local = get_hex_verts(c[i,0], c[i,1], a[i])
            # verts = c[i] + verts_local
            
            # To speed up, we can approximate the boundary check using the maximum projection of the hexagon onto the 6 edge normals.
            # The distance from center to the farthest point in direction n is the support function.
            # For a regular hexagon, the support function h(n) = max_{v in vertices} v . n
            # Since the hexagon is convex and symmetric, we can compute this analytically or precomputed.
            # Actually, the condition "hex i is inside outer hex" is equivalent to:
            # For each of the 6 outer edge normals n_k, max_{v in hex_i} (v . n_k) <= R_out
            
            # Let's compute the max projection for each hex i onto each of the 6 normals.
            # verts_i = c[i] + R(a[i]) @ VERTS_UNIT.T
            # proj_i_k = max_j (verts_i[j] . n_k) = c[i] . n_k + max_j (R(a[i]) @ VERTS_UNIT.T [j] . n_k)
            # The term max_j (R(a[i]) @ VERTS_UNIT.T [j] . n_k) depends only on a[i] and n_k.
            # Let's precompute this term for a given angle? No, angle varies.
            # But we can compute it on the fly.
            
            # Efficient calculation of support function for a rotated hexagon:
            # The support function of a regular hexagon in direction n is:
            # h(n) = max( |n . v_k| for k in 0..5 ) where v_k are the vertices of the *unrotated* hexagon? No.
            # It's easier to just compute the 6 vertex projections.
            
            rot = a[i]
            c_rot, s_rot = np.cos(rot), np.sin(rot)
            R = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
            verts_local = VERTS_UNIT @ R.T # (6, 2)
            
            # Project local vertices onto the 6 outer normals
            # proj_local = verts_local @ EDGE_NORMALS.T # (6, 6)
            # max_proj_local = np.max(proj_local, axis=0) # (6,)
            
            # The actual projection of vertex j onto normal k is:
            # (c[i] + verts_local[j]) . n_k = c[i] . n_k + verts_local[j] . n_k
            # So max over j is c[i] . n_k + max_j (verts_local[j] . n_k)
            
            # We want max_j (verts_local[j] . n_k) for all k.
            # This is the support function of the rotated hexagon in the directions n_k.
            
            # Let's just compute the 6x6 matrix.
            proj_local = verts_local @ EDGE_NORMALS.T
            max_proj_local = np.max(proj_local, axis=0)
            
            # c[i] . n_k for all k
            c_proj = c[i] @ EDGE_NORMALS.T
            
            # Total max projection
            max_proj_total = c_proj + max_proj_local
            
            # Penalty for boundary violation
            # If max_proj_total[k] > R_out, we have a violation.
            # We use a smooth penalty: sum( max(0, max_proj_total - R_out)^2 )
            viol = np.maximum(0, max_proj_total - R_out)
            penalty += np.sum(viol ** 2)
        
        # Overlap penalty
        # For each pair (i, j), check if they overlap.
        # Overlap if they are not disjoint.
        # Disjoint if there exists a separating axis.
        # For convex polygons, if they overlap, the minimum distance between them is 0.
        # We can use a smooth approximation of the minimum distance.
        # Or, we can use the fact that if two convex polygons overlap, the sum of the maximum penetration depths is positive?
        # A common trick: penalty = sum_{pairs} max(0, overlap_depth)^2
        
        # How to calculate overlap depth between two hexagons?
        # It's the minimum translation distance to separate them.
        # This is computationally expensive to do exactly in a smooth way.
        # Alternative: Use the distance between centers? No, rotation matters.
        # Alternative: Check if any vertex of i is inside j or vice versa?
        # If we use a penalty that increases when vertices are inside, it might work.
        # Let's try: For each pair (i, j), calculate the minimum distance between any vertex of i and any edge of j, and vice versa.
        # If min_dist < 0, they overlap.
        # But calculating min distance between segments is complex.
        
        # Let's try a simpler heuristic for the penalty:
        # If the distance between centers is less than 2.0 (diameter), they *might* overlap.
        # A better proxy: The sum of the "radial distances" of the vertices into the other hexagon.
        
        # Let's stick to a robust but slightly slower method for N=11:
        # For each pair, check the 12 separating axes (6 from i, 6 from j).
        # If they overlap, the projection intervals on all axes overlap.
        # The "overlap amount" on an axis is the length of the intersection of the intervals.
        # The minimum overlap amount over all axes is 0 if they are disjoint, and > 0 if they overlap.
        # Actually, if they overlap, the minimum overlap amount over the *separating* axes is 0? No, there are no separating axes.
        # If they overlap, the minimum overlap amount over *all* candidate axes is > 0?
        # No, if they are just touching, it's 0. If they overlap deeply, it's large.
        # The minimum overlap amount over the 12 axes is a measure of how "deep" the overlap is.
        # Let's use this as the penalty term.
        
        for i in range(N):
            for j in range(i+1, N):
                # Get vertices of i and j
                rot_i = a[i]
                c_rot_i, s_rot_i = np.cos(rot_i), np.sin(rot_i)
                R_i = np.array([[c_rot_i, -s_rot_i], [s_rot_i, c_rot_i]])
                verts_i = c[i] + VERTS_UNIT @ R_i.T
                
                rot_j = a[j]
                c_rot_j, s_rot_j = np.cos(rot_j), np.sin(rot_j)
                R_j = np.array([[c_rot_j, -s_rot_j], [s_rot_j, c_rot_j]])
                verts_j = c[j] + VERTS_UNIT @ R_j.T
                
                # Get edge normals for i and j
                # Normals are rotated by the same angle as the vertices
                normals_i = EDGE_NORMALS @ R_i.T # Wait, R_i rotates points. Does it rotate normals the same way? Yes.
                normals_j = EDGE_NORMALS @ R_j.T
                
                # Project vertices onto all 12 axes
                # Axes: normals_i (6) and normals_j (6)
                axes = np.vstack((normals_i, normals_j)) # (12, 2)
                
                # Projections of verts_i onto axes
                proj_i = verts_i @ axes.T # (6, 12)
                min_i = np.min(proj_i, axis=0)
                max_i = np.max(proj_i, axis=0)
                
                # Projections of verts_j onto axes
                proj_j = verts_j @ axes.T # (6, 12)
                min_j = np.min(proj_j, axis=0)
                max_j = np.max(proj_j, axis=0)
                
                # Overlap amount on each axis
                # overlap = min(max_i, max_j) - max(min_i, min_j)
                overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                
                # If they overlap, all overlap values are >= 0.
                # If they are disjoint, at least one overlap value is < 0.
                # The minimum overlap value is the "separation distance" if disjoint, or 0 if touching, or > 0 if overlapping?
                # No, if they overlap, the minimum overlap is still > 0?
                # Let's think. If two squares overlap by 0.1 in x and 0.1 in y, the overlap on x-axis is 0.1, on y-axis is 0.1. Min is 0.1.
                # If they are disjoint, one axis has negative overlap.
                # So, if min(overlap) < 0, they are disjoint.
                # If min(overlap) >= 0, they overlap (or touch).
                # The value min(overlap) is a good measure of the "depth" of overlap if positive, and separation if negative.
                # We want to penalize when min(overlap) > 0.
                # Actually, we want to penalize any overlap.
                # Let's use penalty += max(0, min(overlap))^2
                
                min_overlap = np.min(overlap)
                if min_overlap > 0:
                    penalty += min_overlap ** 2
                else:
                    # They are disjoint or touching. No penalty.
                    # But to make the gradient smoother, we could add a small penalty for close approaches?
                    # No, let's keep it simple.
                    pass
        
        return penalty

    # Define the S value to optimize for.
    # Based on inspirations, S around 4.5-5.0 is promising.
    # Let's try a few values of S and pick the best valid one.
    # But the objective function depends on S.
    # We can run the optimizer for a fixed S.
    
    S_target = 4.8 # Start with a reasonable guess
    
    best_result = None
    best_penalty = np.inf
    
    # Initial configurations
    # 1. Hexagonal lattice
    # 2. Random
    # 3. Ring
    
    def init_lattice():
        # 1 in center, 6 around, 4 more?
        # Let's place 11 hexagons in a tight cluster.
        # Centers at distance d = SQRT3 (touching)
        # Angles 0, 60, 120, 180, 240, 300 for the ring.
        pos = []
        pos.append([0, 0])
        for k in range(6):
            angle = np.deg2rad(60 * k)
            pos.append([SQRT3 * np.cos(angle), SQRT3 * np.sin(angle)])
        # 4 more, placed further out?
        # Let's just add 4 more randomly near the ring.
        for k in range(4):
            angle = np.deg2rad(60 * k + 30)
            pos.append([2 * SQRT3 * np.cos(angle), 2 * SQRT3 * np.sin(angle)])
        
        return np.array(pos)
    
    def init_random():
        return np.random.randn(N, 2) * 2.0

    def init_ring():
        # All on a ring of radius R
        R = 2.0
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack((R * np.cos(angles), R * np.sin(angles)))

    inits = [init_lattice(), init_random(), init_ring()]
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Randomize angles
        init_angles = np.random.uniform(0, 2*np.pi, N)
        
        # Pack into params
        params = np.zeros(3*N)
        params[0:2*N:3] = init_pos[:, 0]
        params[1:2*N+1:3] = init_pos[:, 1]
        params[2:3*N:3] = init_angles
        
        # Run optimization
        # Use L-BFGS-B
        # We need to set bounds? No, positions can be anywhere, angles can be anywhere.
        # But we can restrict angles to [0, 2pi) for stability? No, minimize handles unbounded.
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success or res.fun < best_penalty:
                # Check if the result is actually valid (penalty is very small)
                if res.fun < 1e-4:
                    best_penalty = res.fun
                    best_params = res.x
                    best_result = (best_params, S_target)
        except Exception as e:
            print(f"Optimization failed for init {init_idx}: {e}")
            
        # If we found a very good solution, we can break
        if best_penalty < 1e-6:
            break

    if best_result is None:
        # Fallback: try a larger S
        S_target = 5.5
        # Repeat with larger S?
        # For now, just return a known valid but larger configuration.
        # Or try one more optimization with S=5.5.
        # Let's just return a simple valid packing with large S.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    params, S = best_result
    
    # Extract positions and angles
    centers_x = params[0:2*N:3]
    centers_y = params[1:2*N+1:3]
    angles_rad = params[2:3*N:3]
    angles_deg = np.rad2deg(angles_rad)
    
    inner = []
    for i in range(N):
        inner.append([centers_x[i], centers_y[i], angles_deg[i]])
    
    return inner, (0.0, 0.0), S, 0.0
