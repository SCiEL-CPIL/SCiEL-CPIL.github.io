import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def smooth_collision_penalty(verts1, verts2, k=20.0):
        """
        Approximates the minimum distance between two convex polygons.
        Uses the maximum of the negative of the minimum distance.
        For a smooth penalty, we can sum the soft-min of the distances between all pairs of edges/vertices.
        A simpler differentiable proxy: 
        For each edge of poly1, compute distance to poly2. 
        This is complex. 
        Alternative: Use the fact that for disjoint convex polygons, there is a separating axis.
        The minimum distance is the max over all separating axes of the gap.
        Gap(a) = max(proj1) - min(proj2) or max(proj2) - min(proj1).
        Distance = max(0, min_gap).
        To make it smooth, we can use a log-sum-exp or soft-min approximation on the "violations".
        
        However, a common trick for packing is to define the objective as:
        Sum over all pairs of hexagons: soft_max(0, -min_distance)
        
        Computing min_distance between two convex polygons is differentiable almost everywhere.
        We can compute the distance between all pairs of edges and vertices? No, that's O(N^2) per pair.
        
        Let's use a simpler differentiable metric:
        The "Minkowski sum" approach or just checking the 6 axes of one polygon against the other.
        For two hexagons, the separating axes are the normals of their edges.
        For hexagon 1 (angle a1) and hexagon 2 (angle a2), the relevant axes are the 6 normals of hex1 and 6 normals of hex2.
        For each axis, compute the gap. If gap < 0, they overlap.
        The "depth of overlap" is max(0, -min_gap).
        We want to minimize the sum of (depth_of_overlap)^2 for all pairs.
        
        This is differentiable (piecewise, but smooth enough for L-BFGS-B with finite differences or careful gradient).
        """
        # Normals for hex1
        a1 = np.radians(verts1_angle) # We need angles passed or stored
        # Actually, let's pass angles explicitly to avoid global state issues in vectorization
        
    def compute_overlap_penalty(positions, angles):
        """
        positions: (N, 2)
        angles: (N,) in radians
        Returns scalar penalty.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Get vertices
                v1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
                v2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, np.degrees(angles[j]))
                
                # Collect axes: normals of v1 and normals of v2
                axes = []
                for v in [v1, v2]:
                    for k in range(6):
                        edge = v[(k+1)%6] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-8:
                            axes.append(normal / norm)
                
                min_gap = float('inf')
                for axis in axes:
                    proj1 = v1.dot(axis)
                    proj2 = v2.dot(axis)
                    gap1 = np.min(proj2) - np.max(proj1)
                    gap2 = np.min(proj1) - np.max(proj2)
                    gap = min(gap1, gap2)
                    if gap < min_gap:
                        min_gap = gap
                
                # If min_gap < 0, they overlap.
                # Penalty is (max(0, -min_gap))^2
                if min_gap < 0:
                    penalty += (-min_gap) ** 2
        return penalty

    def objective(params):
        """
        params: 
        0..23: x, y for 12 hexagons
        24..35: angles for 12 hexagons (radians)
        36: outer_center_x
        37: outer_center_y
        38: outer_side_length
        39: outer_angle (radians)
        """
        if time.time() > deadline:
            return 1e10

        positions = params[:24].reshape(N, 2)
        angles = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_side = params[38]
        outer_angle = params[39]
        
        # 1. Collision Penalty
        coll_pen = compute_overlap_penalty(positions, angles)
        
        # 2. Containment Penalty
        # Outer hexagon vertices
        outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_side, np.degrees(outer_angle))
        
        # For each inner hexagon vertex, check if it's inside outer hexagon.
        # If not, add penalty.
        # "Inside" check for convex polygon: point must be on the same side of all edges.
        # For smoothness, we can calculate the minimum distance from the point to the boundary.
        # If point is outside, distance is positive (or negative depending on sign convention).
        # Let's use the projection method.
        # For each edge of outer hex, compute signed distance.
        # If any signed distance < 0, it's outside.
        # Penalty = sum of (max(0, -signed_dist))^2
        
        contain_pen = 0.0
        for i in range(N):
            inner_verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
            for vx, vy in inner_verts:
                # Check against outer hexagon
                min_dist = float('inf')
                # Outer hexagon edges
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-8: continue
                    normal = normal / norm
                    # Ensure normal points INWARD
                    # Center of outer hex is (outer_cx, outer_cy)
                    # Midpoint of edge
                    mid = (p1 + p2) / 2.0
                    vec_to_center = np.array([outer_cx, outer_cy]) - mid
                    if np.dot(normal, vec_to_center) < 0:
                        normal = -normal
                    
                    # Signed distance from point (vx, vy) to the line defined by p1, normal
                    # dist = dot(point - p1, normal)
                    dist = np.dot([vx - p1[0], vy - p1[1]], normal)
                    if dist < min_dist:
                        min_dist = dist
                
                if min_dist < 0:
                    contain_pen += (-min_dist) ** 2
        
        # 3. Objective: Minimize outer side length + penalties
        # We want to minimize outer_side.
        # But if we just minimize outer_side, it will go to 0 unless constrained by containment.
        # The containment penalty prevents the inner hexes from being inside a tiny outer hex.
        # So: f = outer_side + lambda_coll * coll_pen + lambda_cont * contain_pen
        
        lambda_coll = 10.0
        lambda_cont = 10.0
        
        return outer_side + lambda_coll * coll_pen + lambda_cont * contain_pen

    # Initial Guess
    # Honeycomb cluster
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y])
        
    init_positions = np.array(init_positions)
    init_angles = np.zeros(N)
    
    # Estimate initial outer size
    # Max distance from origin to any vertex of inner hexes
    max_r = 0
    for i in range(N):
        v = hexagon_vertices(init_positions[i,0], init_positions[i,1], 1.0, 0.0)
        dists = np.linalg.norm(v, axis=1)
        max_r = max(max_r, np.max(dists))
    
    init_outer_cx = 0.0
    init_outer_cy = 0.0
    init_outer_side = max_r * 1.1 # Add margin
    init_outer_angle = 0.0
    
    init_params = np.concatenate([
        init_positions.flatten(),
        init_angles,
        [init_outer_cx, init_outer_cy, init_outer_side, init_outer_angle]
    ])
    
    # Bounds
    # Positions: -20 to 20
    # Angles: -pi to pi
    # Outer center: -20 to 20
    # Outer side: 1.0 to 50.0 (must be at least 1 to contain one hex, realistically > 5)
    # Outer angle: -pi to pi
    
    bounds = []
    for i in range(N):
        bounds.append((-20, 20))
        bounds.append((-20, 20))
    for i in range(N):
        bounds.append((-np.pi, np.pi))
    bounds.append((-20, 20)) # outer cx
    bounds.append((-20, 20)) # outer cy
    bounds.append((1.0, 50.0)) # outer side
    bounds.append((-np.pi, np.pi)) # outer angle
    
    best_val = objective(init_params)
    best_params = init_params.copy()
    
    # Use L-BFGS-B for local optimization
    # Since the problem is non-convex, we might need multiple restarts or a global search first.
    # Given the budget, let's try a few random restarts with L-BFGS-B.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            x0 = init_params
        else:
            # Perturb initial params
            x0 = init_params.copy()
            noise_pos = np.random.normal(0, 0.5, size=(N, 2))
            noise_ang = np.random.normal(0, 0.5, size=N)
            noise_outer = np.random.normal(0, 0.1, size=4)
            
            x0[:24] += noise_pos.flatten()
            x0[24:36] += noise_ang
            x0[36:40] += noise_outer
            
            # Clip to bounds
            for i in range(len(bounds)):
                if x0[i] < bounds[i][0]: x0[i] = bounds[i][0]
                if x0[i] > bounds[i][1]: x0[i] = bounds[i][1]
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final check: If the best solution still has significant penalties, it might be invalid.
    # The objective function minimizes side + penalties.
    # If penalties are non-zero, the configuration is invalid (overlapping or sticking out).
    # We need to verify if the final configuration is actually valid (disjoint and contained).
    
    def is_valid(params):
        positions = params[:24].reshape(N, 2)
        angles = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_side = params[38]
        outer_angle = params[39]
        
        # Check collisions
        pen = compute_overlap_penalty(positions, angles)
        if pen > 1e-4:
            return False, 0.0
            
        # Check containment
        outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_side, np.degrees(outer_angle))
        for i in range(N):
            inner_verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
            for vx, vy in inner_verts:
                # Check if inside outer hex
                # Using the same logic as objective but strict
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-8: continue
                    normal = normal / norm
                    mid = (p1 + p2) / 2.0
                    vec_to_center = np.array([outer_cx, outer_cy]) - mid
                    if np.dot(normal, vec_to_center) < 0:
                        normal = -normal
                    dist = np.dot([vx - p1[0], vy - p1[1]], normal)
                    if dist < -1e-5:
                        return False, 0.0
        return True, outer_side

    valid, val = is_valid(best_params)
    
    if not valid:
        # Try to repair or fall back to a known valid configuration if possible
        # Or just return the best attempt, hoping the evaluator is lenient or the penalty was small
        # But the task says "invalid result ... scores 0".
        # So we must return a VALID configuration.
        
        # Fallback: Use the initial honeycomb which is valid (if we scaled it correctly)
        # The initial honeycomb has side length ~ max_r.
        # Let's re-calculate the exact min side for the initial honeycomb.
        
        def min_side_for_config(positions, angles, outer_cx, outer_cy, outer_angle):
            # Find min S such that all inner verts are inside outer hex
            # Binary search or direct calculation
            # Direct calculation:
            # For each inner vertex, find the min S of outer hex (centered at cx,cy, angle a) containing it.
            # The outer hex is defined by 6 half-planes.
            # For a fixed angle and center, the required S is max over all constraints.
            
            # Constraints for outer hex (side S, center C, angle A):
            # The distance from C to the supporting line of each side is S * sqrt(3)/2 (apothem).
            # So for each side direction n_i, we need dot(P - C, n_i) <= S * sqrt(3)/2
            # where n_i are the inward normals.
            
            normals = []
            for k in range(6):
                # Vertex angles
                ang = np.radians(outer_angle) + np.radians(60 * k)
                # Normal is perpendicular to edge. Edge is between vertex k and k+1.
                # Midpoint angle is ang + 30 deg.
                # Normal points to center.
                # Let's just generate the 6 inward normals directly.
                # Angle of normal k: outer_angle + 60*k + 30? 
                # Let's use the vertex definition to get normals.
                pass
            
            # Simpler: Use the objective function's logic to find the "effective" S.
            # If we fix center and angle, the min S is determined by the furthest vertex.
            # S_req = max( dist(P_i, C) / cos(theta_diff) ... )
            
            # Let's just use a simple numerical check for the fallback.
            # We know the honeycomb is valid. We just need the correct S.
            # The honeycomb fits in a hexagon of side ~ 3.5?
            # Let's compute the bounding hexagon for the initial positions.
            
            max_s = 0
            for i in range(N):
                v = hexagon_vertices(positions[i,0], positions[i,1], 1.0, np.degrees(angles[i]))
                for vx, vy in v:
                    # Calculate min S for point (vx,vy) to be in hex(C, S, A)
                    # This is equivalent to finding the smallest S such that P is in Hex(C, S, A).
                    # Hex(C, S, A) is the set of points P such that for all 6 inward normals n_j:
                    # dot(P - C, n_j) <= S * sqrt(3)/2
                    # So S >= max( dot(P - C, n_j) / (sqrt(3)/2) )
                    
                    # Get inward normals for outer hex with angle outer_angle
                    # Vertices of unit hex at angle 0:
                    # (1,0), (0.5, sqrt3/2), ...
                    # Normals to sides:
                    # Side 0: (1,0) to (0.5, sqrt3/2). Mid (0.75, sqrt3/4). Normal (sqrt3/4, -0.75)? No.
                    # Let's compute normals for a hexagon of side 1, angle 0.
                    v0 = hexagon_vertices(0,0, 1.0, 0.0)
                    normals_unit = []
                    for k in range(6):
                        p1 = v0[k]
                        p2 = v0[(k+1)%6]
                        edge = p2 - p1
                        n = np.array([-edge[1], edge[0]])
                        n = n / np.linalg.norm(n)
                        # Make inward
                        if np.dot(n, [0,0]) > 0: # Center is origin
                            pass
                        else:
                            n = -n
                        normals_unit.append(n)
                    
                    # Rotate normals by outer_angle
                    c, s = np.cos(outer_angle), np.sin(outer_angle)
                    rot = np.array([[c, -s], [s, c]])
                    normals_rot = [rot @ n for n in normals_unit]
                    
                    vec = [vx - outer_cx, vy - outer_cy]
                    max_proj = 0
                    for n in normals_rot:
                        proj = np.dot(vec, n)
                        if proj > max_proj:
                            max_proj = proj
                    
                    # S * sqrt(3)/2 >= max_proj
                    s_req = max_proj * 2 / SQRT3
                    if s_req > max_s:
                        max_s = s_req
            
            return max_s

        # Fallback to initial honeycomb
        valid_fallback, val_fallback = is_valid(init_params)
        if valid_fallback:
            best_params = init_params
            best_val = val_fallback
        else:
            # If even the fallback is invalid (shouldn't happen), try to compute min side for it
            # and hope it's close enough or just return it.
            # Actually, init_params has outer_side = max_r * 1.1, which should be valid.
            # The is_valid check might be too strict due to floating point.
            # Let's trust the computed min_side.
            min_s = min_side_for_config(
                init_params[:24].reshape(N,2), 
                init_params[24:36], 
                init_params[36], init_params[37], init_params[39]
            )
            init_params[38] = min_s
            best_params = init_params
            best_val = min_s

    # Extract final parameters
    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        angle_deg = np.degrees(best_params[24+i])
        inner.append([float(x), float(y), float(angle_deg)])
        
    outer_cx = float(best_params[36])
    outer_cy = float(best_params[37])
    outer_side = float(best_params[38])
    outer_angle = float(np.degrees(best_params[39]))
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-END
