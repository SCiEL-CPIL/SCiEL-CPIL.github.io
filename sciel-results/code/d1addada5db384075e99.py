# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _check_validity(inner, outer, s):
    """
    Check all validity conditions. Returns (is_valid, min_violation).
    inner: (12,3) array (x,y,angle_deg)
    outer: (3,) array (x,y,angle_deg)
    s: float side length
    """
    # Non-finite check
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False, -1.0

    if s < np.sqrt(12.0) - 1e-9:
        return False, -1.0

    # Outer vertices distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-6:
                return False, -1.0

    # Unit edge check: after shifting by outer center, each inner hex edge length should be 1
    shift = np.array([outer[0], outer[1]])
    for k in range(12):
        cx, cy, ang = inner[k]
        cx2, cy2 = cx - shift[0], cy - shift[1]
        verts = _hex_vertices(cx2, cy2, 1.0, ang)
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False, -1.0

    # Separation: SAT for all pairs
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            # Check all edge normals of both hexagons
            all_verts = [vi, vj]
            all_normals = []
            for v in all_verts:
                for e in range(6):
                    p1 = v[e]
                    p2 = v[(e + 1) % 6]
                    edge = p2 - p1
                    # Normal (perpendicular)
                    n = np.array([-edge[1], edge[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    all_normals.append(n)

            for n in all_normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                # Intervals must NOT overlap: gap > 1e-6
                gap = max(min_j - max_i, min_i - max_j)
                if gap < 1e-6:
                    # They overlap on this axis - need to check if they're actually separated
                    # Actually, for separation we need ALL axes to show separation? No.
                    # SAT: if there exists ANY axis where projections are separated, the objects are separated.
                    # For them to NOT be separated (overlap), ALL axes must have overlapping projections.
                    # So if we find one axis with gap >= 1e-6, they ARE separated.
                    pass
            # If we get here, we need to verify: if NO axis gives separation, they overlap
            # Let me redo: check if there's any separating axis
            separated = False
            for n in all_normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                gap = max(min_j - max_i, min_i - max_j)
                if gap >= 1e-6:
                    separated = True
                    break
            if not separated:
                return False, -1.0

    # Inner vertices inside outer hexagon
    for k in range(12):
        vi = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for v in vi:
            for e in range(6):
                p1 = ov[e]
                p2 = ov[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False, -1.0

    return True, 0.0


def _min_outer_side_for_inner(inner):
    """
    Find the minimum side length s of a regular hexagon (any orientation, any center)
    that contains all 12 unit hexagons.
    Returns (min_s, best_outer) where best_outer is (cx, cy, angle_deg).
    """
    # For a given orientation of the outer hexagon, find the minimal side length
    # by checking all vertices of all inner hexagons against the 6 half-planes.
    
    # The outer hexagon with center (cx,cy), side s, angle a has vertices:
    # (cx + s*cos(a+60i), cy + s*sin(a+60i))
    # The half-plane constraints: for each edge, the interior is on one side.
    
    # For a regular hexagon centered at origin with angle 0 (flat-top or pointy-top?):
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center.
    # Edge from vertex i to vertex i+1: the inward normal direction.
    
    # Let's think of it differently. A regular hexagon can be defined as the intersection
    # of 6 half-planes. For a hexagon with center c, side s, and rotation angle a:
    # The 6 supporting lines are at distance s*sqrt(3)/2 from center (apothem = s*sqrt(3)/2)
    # in directions a + 30 + 60*i degrees (i=0..5).
    
    # Wait, let me reconsider. For a regular hexagon with side length s:
    # - The apothem (distance from center to midpoint of edge) = s * sqrt(3) / 2
    # - The circumradius (center to vertex) = s
    # 
    # If vertices are at angles a, a+60, ..., a+300, then edges are between consecutive vertices.
    # The normal to edge i (between vertex i and i+1) points in direction a + 60*i + 30 degrees.
    # The distance from center to edge (apothem) = s * cos(30°) = s * sqrt(3)/2.
    
    # So the hexagon = { p : dot(p - c, n_i) <= s*sqrt(3)/2 for all i }
    # where n_i is the unit normal in direction (a + 30 + 60*i) degrees.
    
    # To find minimal s for a given (c, a):
    # For each inner vertex v, and each direction i:
    #   dot(v - c, n_i) <= s * sqrt(3)/2
    #   => s >= dot(v - c, n_i) / (sqrt(3)/2)
    # So s_min = max over all (v, i) of dot(v - c, n_i) / (sqrt(3)/2)
    # But we also need s >= 0, and we want the max of positive values.
    
    # Actually we need: s >= max_i [max_v dot(v-c, n_i)] / (sqrt(3)/2)
    # But only if the max is positive; if all dots are negative for some i, that constraint is automatically satisfied.
    
    # Let's collect all vertices of all inner hexagons
    all_verts = []
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    best_s = np.inf
    best_c = np.array([0.0, 0.0])
    best_a = 0.0
    
    # Try various orientations
    # The optimal orientation is likely aligned with the inner hexagons
    # Try angles from 0 to 60 in steps of 5 degrees, then refine
    for a_deg in np.arange(0, 60, 5.0):
        a = np.deg2rad(a_deg)
        normals = []
        for i in range(6):
            theta = a + np.deg2rad(30 + 60 * i)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        normals = np.array(normals)  # (6, 2)
        
        # For each normal direction, find the max dot product with all vertices
        # dot(all_verts - c, n_i) for all vertices
        # We need to choose c to minimize the maximum constraint.
        
        # For a fixed orientation, the problem is:
        # minimize s subject to: for all i, max_v dot(v - c, n_i) <= s * sqrt(3)/2
        # => s >= (1/sqrt(3)) * max_i max_v dot(v - c, n_i)
        
        # This is a convex problem in c. The optimal c minimizes:
        # f(c) = max_i max_v dot(v - c, n_i)
        #      = max_i (max_v dot(v, n_i) - dot(c, n_i))
        
        # Let M_i = max_v dot(v, n_i) for each i
        # f(c) = max_i (M_i - dot(c, n_i))
        # We want to minimize this over c.
        
        # This is equivalent to: find c that minimizes the maximum of 6 linear functions.
        # This is a linear programming problem.
        
        # For simplicity, let's try c = centroid of all vertices as a starting point,
        # then do a few iterations of subgradient descent.
        
        # Compute M_i
        dots = all_verts @ normals.T  # (72, 6)
        M = dots.max(axis=0)  # (6,)
        
        # Subgradient descent for c
        c = all_verts.mean(axis=0)
        lr = 0.5
        for it in range(200):
            # f(c) = max_i (M_i - dot(c, n_i))
            vals = M - c @ normals.T  # (6,)
            idx = np.argmax(vals)
            # subgradient: -n_idx
            c = c + lr * normals[idx]
            lr *= 0.95
        
        # Compute s
        vals = M - c @ normals.T
        s_val = vals.max() / (np.sqrt(3) / 2)
        
        if s_val < best_s:
            best_s = s_val
            best_c = c.copy()
            best_a = a_deg
    
    # Refine around best angle
    for a_deg in np.arange(best_a - 5, best_a + 5.1, 0.5):
        a = np.deg2rad(a_deg)
        normals = []
        for i in range(6):
            theta = a + np.deg2rad(30 + 60 * i)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        normals = np.array(normals)
        
        dots = all_verts @ normals.T
        M = dots.max(axis=0)
        
        c = all_verts.mean(axis=0)
        lr = 0.5
        for it in range(300):
            vals = M - c @ normals.T
            idx = np.argmax(vals)
            c = c + lr * normals[idx]
            lr *= 0.95
        
        vals = M - c @ normals.T
        s_val = vals.max() / (np.sqrt(3) / 2)
        
        if s_val < best_s:
            best_s = s_val
            best_c = c.copy()
            best_a = a_deg
    
    return best_s, np.array([best_c[0], best_c[1], best_a])


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing the outer hexagon side length.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit, use 590 for safety
    
    np.random.seed(42)
    
    # Initial configuration: hexagonal cluster
    # Center at origin
    # Ring 1: 6 hexagons at distance 2 (touching)
    # Ring 2: 6 more, but we only need 5 more (total 12)
    # For flat-top hexagons (angle 0), the neighbors are at:
    # (±2, 0), (±1, ±sqrt(3))
    
    sqrt3 = np.sqrt(3)
    
    # Start with a symmetric arrangement
    inner = np.array([
        [0, 0, 0],           # center
        [2, 0, 0],           # right
        [-2, 0, 0],          # left
        [1, sqrt3, 0],       # top-right
        [-1, sqrt3, 0],      # top-left
        [1, -sqrt3, 0],      # bottom-right
        [-1, -sqrt3, 0],     # bottom-left
        [3, sqrt3, 0],       # far top-right
        [-3, sqrt3, 0],      # far top-left
        [3, -sqrt3, 0],      # far bottom-right
        [-3, -sqrt3, 0],     # far bottom-left
        [0, 2*sqrt3, 0],     # top
    ])
    
    # Compute initial outer
    s0, outer0 = _min_outer_side_for_inner(inner)
    
    best_inner = inner.copy()
    best_outer = outer0.copy()
    best_s = s0
    
    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling = 0.995
    n_iter = 200000
    
    current_s = s0
    current_inner = inner.copy()
    
    for it in range(n_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Random perturbation
        idx = np.random.randint(0, 12)
        dx = np.random.randn() * 0.1
        dy = np.random.randn() * 0.1
        da = np.random.randn() * 2.0  # degrees
        
        new_inner = current_inner.copy()
        new_inner[idx, 0] += dx
        new_inner[idx, 1] += dy
        new_inner[idx, 2] += da
        
        # Compute new outer side
        new_s, new_outer = _min_outer_side_for_inner(new_inner)
        
        delta_s = new_s - current_s
        
        if delta_s < 0 or np.random.random() < np.exp(-delta_s / T):
            current_s = new_s
            current_inner = new_inner
            
            if new_s < best_s:
                best_s = new_s
                best_inner = new_inner.copy()
                best_outer = new_outer.copy()
        
        T = max(T * cooling, T_min)
    
    # Final verification
    valid, _ = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        # Fallback to a known valid configuration
        inner = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.17, 0],
            [-1.25, 2.17, 0],
            [1.25, -2.17, 0],
            [-1.25, -2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, -2.17, 0],
            [-3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
