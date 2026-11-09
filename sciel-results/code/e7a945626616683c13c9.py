import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 570
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Contact Graph Representation
    # We define the 12 hexagons in a fixed topology (4-3-3-2 honeycomb slice)
    # and parameterize their relative positions by distances and angles.
    
    # Topology:
    # Row 0: 4 hexagons (indices 0, 1, 2, 3)
    # Row 1: 3 hexagons (indices 4, 5, 6)
    # Row 2: 3 hexagons (indices 7, 8, 9)
    # Row 3: 2 hexagons (indices 10, 11)
    
    # We will fix hexagon 0 at (0,0) with angle 0.
    # We parameterize the other 11 hexagons relative to their "parents" in the tree/graph.
    # To ensure non-overlap, we enforce minimum distances.
    # For two unit hexagons with relative angle delta, the minimum non-overlapping distance is:
    # d_min(delta) = 2 * cos(min(delta, 60-delta)) * cos(30) ... actually simpler:
    # The support function of a unit hexagon in direction u is h(u).
    # Distance needed for separation along vector v (unit) is h(v) + h(-v) = 2 * h(v) if symmetric?
    # For regular hexagon side 1, h(u) = max(0, cos(u - 60k)) * sqrt(3)/2 * ... 
    # Actually, let's just use a safe lower bound: sqrt(3) is the distance for parallel edges.
    # If rotated, they can get closer? No, rotating usually increases required distance or keeps it same if aligned.
    # Wait, if I rotate one by 30 deg, the vertices touch the edge of the other.
    # Distance for 30 deg rotation: The vertex of one is at distance 1 from center. The edge of other is at distance sqrt(3)/2 = 0.866.
    # So distance = 1 + 0.866 = 1.866 > sqrt(3) = 1.732.
    # So sqrt(3) is indeed the minimum distance for ANY relative angle?
    # Let's check.
    # Hexagon A at origin. Hexagon B at distance d along x-axis, rotated by alpha.
    # Closest points:
    # If alpha=0, closest is edge to edge. Distance sqrt(3).
    # If alpha=30, vertex of B (at 1,0 relative to B center? No, vertices are at 0, 60, 120... for angle 0.
    # If B is rotated 30, vertices are at 30, 90, 150...
    # Vertex of B at (d + cos(30), sin(30)) = (d + 0.866, 0.5).
    # Edge of A: from (0.5, 0.866) to (0.5, -0.866)? No.
    # Vertices of A (angle 0): (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866).
    # Edge from (0.5, 0.866) to (0.5, -0.866) is vertical line x=0.5.
    # Distance from B vertex (d+0.866, 0.5) to line x=0.5 is d+0.866 - 0.5 = d + 0.366.
    # If they touch, d+0.366 = 0 => d = -0.366? No, that's overlap.
    # We need the distance between sets to be >= 0.
    # Distance from B vertex to A edge.
    # If d is small, B vertex might be inside A.
    # Minimum d such that B vertex is on A edge:
    # The x-coordinate of the vertical edge of A is 0.5.
    # The x-coordinate of the right-most vertex of B (rotated 30) is d + cos(30) = d + 0.866.
    # Wait, the vertex at angle 30 is (d+cos30, sin30).
    # The edge of A is x=0.5.
    # So d + 0.866 = 0.5 => d = -0.366. This means at d=0, the vertex is at 0.866, which is > 0.5. So it's outside?
    # No, if d=0, B is at origin. Vertex at (0.866, 0.5). A has vertex at (1,0) and (0.5, 0.866).
    # The edge connecting (1,0) and (0.5, 0.866) is not vertical.
    # I am overcomplicating.
    # Let's just use a penalty function based on the actual SAT check, but in the reduced space.
    # Or better, use the fact that if we enforce d_ij >= sqrt(3) for all pairs, they are guaranteed not to overlap?
    # Yes, because the maximum "radius" of a unit hexagon is 1 (vertex to center).
    # If distance between centers is >= 1 + 1 = 2, they definitely don't overlap.
    # But they can overlap if distance < 2.
    # The minimum non-overlapping distance depends on relative angle.
    # However, we can simply compute the minimum separation for the current configuration and penalize if < 0.
    
    # Strategy:
    # 1. Define a graph of connections.
    # 2. Parameters: For each node i > 0, distance to parent p(i) and angle of vector p(i)->i.
    #    Plus global rotation of the whole cluster.
    # 3. Objective: Minimize outer hexagon side s.
    # 4. Penalty: If any pair overlaps, add large penalty.
    
    # Graph structure:
    # 0 is root.
    # 1 is child of 0.
    # 2 is child of 1.
    # 3 is child of 2.
    # 4 is child of 0.
    # 5 is child of 4.
    # 6 is child of 5.
    # 7 is child of 4.
    # 8 is child of 7.
    # 9 is child of 8.
    # 10 is child of 7.
    # 11 is child of 10.
    
    parents = [-1, 0, 1, 2, 0, 4, 5, 4, 7, 8, 7, 10]
    
    # Initial guess: Standard lattice
    # Lattice vectors a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    # We map the indices to lattice coordinates roughly.
    # This specific tree might not map perfectly to the 4-3-3-2 rows if we just use a chain.
    # Let's use a more natural tree for the 4-3-3-2 shape.
    # Row 0: 0, 1, 2, 3.  1 is child of 0. 2 is child of 1. 3 is child of 2.
    # Row 1: 4, 5, 6.  4 is child of 0 (below it?). 5 is child of 4. 6 is child of 5.
    # Row 2: 7, 8, 9.  7 is child of 4. 8 is child of 7. 9 is child of 8.
    # Row 3: 10, 11. 10 is child of 7. 11 is child of 10.
    
    # Initial positions (x,y) for 12 hexagons in 4-3-3-2 arrangement (approx)
    # h = sqrt(3)/2
    # Row 0: y=0. x = 0, 2h, 4h, 6h ? No, hexagons are wider.
    # Center to center horizontal distance for adjacent in same row is 2*h = sqrt(3).
    # Row 1: y=1.5. x = h, 3h, 5h.
    # Row 2: y=3.0. x = 0, 2h, 4h.
    # Row 3: y=4.5. x = h, 3h.
    
    h = np.sqrt(3)/2
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [4*h, 0],
        [6*h, 0],
        [h, 1.5],
        [3*h, 1.5],
        [5*h, 1.5],
        [0, 3.0],
        [2*h, 3.0],
        [4*h, 3.0],
        [h, 4.5],
        [3*h, 4.5]
    ])
    
    # Calculate initial distances and angles from parents
    init_dists = np.zeros(11)
    init_angles = np.zeros(11)
    for i in range(1, 12):
        p = parents[i]
        vec = init_pos[i] - init_pos[p]
        init_dists[i-1] = np.linalg.norm(vec)
        init_angles[i-1] = np.arctan2(vec[1], vec[0])
        
    def build_positions(params):
        # params: [d1, a1, d2, a2, ..., d11, a11, global_rot]
        # 11 distances, 11 angles, 1 global rotation = 23 variables
        positions = np.zeros((12, 2))
        angles = np.zeros(12)
        
        global_rot = params[22]
        angles[0] = global_rot
        
        for i in range(1, 12):
            d = params[(i-1)*2]
            a_rel = params[(i-1)*2 + 1]
            p = parents[i]
            
            # Vector from parent to i
            vec = np.array([d * np.cos(a_rel), d * np.sin(a_rel)])
            positions[i] = positions[p] + vec
            angles[i] = angles[p] + a_rel # Assuming we want to maintain relative orientation? 
            # Actually, the problem allows any angle for each hexagon.
            # But in a "Contact Graph" approach, we might assume they are aligned or have specific relative angles.
            # Let's assume all hexagons have the same orientation as the global rotation.
            # This is a strong constraint but often optimal for hexagonal packing (tiling).
            # If we allow individual rotations, we add 11 more variables.
            # Let's start with global orientation only.
            angles[i] = global_rot
            
        return positions, angles
    
    def objective(params):
        positions, angles = build_positions(params)
        
        # Check overlaps and compute penalty
        unit_hex_verts = _hex_vertices(0, 0, 1, 0)
        penalty = 0.0
        
        # Pre-compute vertices for all hexagons
        all_verts = []
        for i in range(12):
            v = positions[i] + unit_hex_verts
            all_verts.append(v)
            
        for i in range(12):
            for j in range(i+1, 12):
                vi = all_verts[i]
                vj = all_verts[j]
                
                # Quick check: distance between centers
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist > 2.5: # Max distance for unit hexagons to interact is 2 (vertex to vertex)
                    continue
                
                # SAT Check
                separated = False
                for k in range(6):
                    v1 = vi[k]
                    v2 = vi[(k+1)%6]
                    e = v2 - v1
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12: continue
                    n = n / norm
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                    if gap > 1e-6:
                        separated = True
                        break
                if not separated:
                    for k in range(6):
                        v1 = vj[k]
                        v2 = vj[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap > 1e-6:
                            separated = True
                            break
                if not separated:
                    # Calculate minimum separation (penalty)
                    min_sep = float('inf')
                    for k in range(6):
                        v1 = vi[k]
                        v2 = vi[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap < min_sep:
                            min_sep = gap
                    for k in range(6):
                        v1 = vj[k]
                        v2 = vj[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap < min_sep:
                            min_sep = gap
                    if min_sep < 0:
                        penalty += (-min_sep) ** 2 * 1000
                        
        s = _min_outer_side(positions, 0.0) # Global rotation is in params, but _min_outer_side takes angle
        # Wait, _min_outer_side computes min s for a given outer angle.
        # Here, the outer hexagon orientation is free.
        # We should minimize s over the outer angle as well?
        # The task says "returning ... outer a (3,) array (x, y, angle_deg)".
        # So we can choose the outer angle.
        # In the current program, it optimizes the outer angle too.
        # Let's add outer angle to params.
        # params: [d1, a1, ..., d11, a11, global_rot, outer_angle]
        # Total 24 variables.
        
        s = _min_outer_side(positions, params[23])
        
        return s + penalty
    
    def update_best(positions, angles, outer_angle):
        nonlocal best_score, best_inner, best_outer, best_s
        s = _min_outer_side(positions, outer_angle)
        
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = angles
        
        # Center the configuration at the origin for the outer hexagon
        # The outer hexagon center should be the centroid of the inner hexagons?
        # Or we can just place it at (0,0) and let s adjust?
        # _min_outer_side assumes outer is at (0,0).
        # So we should shift positions so that the "center" of the cluster is at (0,0).
        # Which center? The one that minimizes s?
        # _min_outer_side effectively finds the min s hexagon centered at (0,0) that contains the points.
        # If we shift the points, s changes.
        # To minimize s, we should center the points optimally.
        # However, _min_outer_side as written fixes the center at (0,0).
        # So we must shift the positions such that the optimal enclosing hexagon is centered at (0,0).
        # This is equivalent to finding the center that minimizes the max projection.
        # For a regular hexagon, the optimal center is the centroid of the vertices?
        # Actually, for a regular polygon, the minimum enclosing regular polygon of the same orientation
        # has its center at the centroid of the points? No.
        # It has its center at the point that minimizes the max distance to the edges.
        # For now, let's assume the centroid of the 12 centers is a good approximation.
        # Or better, we can optimize the translation as well?
        # That adds 2 variables.
        # Let's stick to the current program's approach: 
        # The current program optimizes positions freely and then calculates s.
        # It does NOT optimize the translation of the cluster relative to the outer hexagon.
        # Wait, looking at the current program:
        # `positions` are the variables.
        # `_min_outer_side(positions, angle)` assumes outer is at (0,0).
        # So the optimizer is effectively choosing positions such that the cluster is "centered" in a way that minimizes s.
        # This is correct.
        
        outer_candidate = np.array([0, 0, outer_angle])
        
        if _check_validity(inner_candidate, outer_candidate, s):
            score_candidate = 1.0 / s
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate.copy()
                best_outer = outer_candidate.copy()
                best_s = s
    
    # Initial params
    init_params = np.zeros(24)
    for i in range(1, 12):
        init_params[(i-1)*2] = init_dists[i-1]
        init_params[(i-1)*2 + 1] = init_angles[i-1]
    init_params[22] = 0.0 # global rot
    init_params[23] = 0.0 # outer angle
    
    # Optimization
    np.random.seed(42)
    
    # Multi-start
    for r in range(20):
        if time.time() > deadline:
            break
            
        if r == 0:
            x_start = init_params.copy()
        else:
            noise_d = np.random.randn(11) * 0.1
            noise_a = np.random.randn(11) * 0.1
            noise_rot = np.random.randn() * 0.1
            noise_outer = np.random.randn() * 0.1
            
            x_start = init_params.copy()
            for i in range(11):
                x_start[(i)*2] += noise_d[i]
                x_start[(i)*2 + 1] += noise_a[i]
            x_start[22] += noise_rot
            x_start[23] += noise_outer
            
        try:
            res = minimize(
                objective,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < 100: # Reasonable
                positions, angles = build_positions(res.x)
                outer_angle = res.x[23]
                update_best(positions, angles, outer_angle)
        except Exception:
            continue
            
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback
        h = np.sqrt(3)/2
        inner_fallback = np.array([
            [0, 0, 0],
            [2*h, 0, 0],
            [h, 1.5, 0],
            [-h, 1.5, 0],
            [-2*h, 0, 0],
            [-h, -1.5, 0],
            [h, -1.5, 0],
            [3*h, 1.5, 0],
            [0, 3, 0],
            [-3*h, 1.5, 0],
            [-3*h, -1.5, 0],
            [3*h, -1.5, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
