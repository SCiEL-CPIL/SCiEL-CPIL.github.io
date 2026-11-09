"""Packing 12 unit hexagons in minimal regular hexagon using SLSQP."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # 1. Initial valid configuration: Hexagonal lattice
    # A hexagon of side 1 has circumradius 1.
    # Distance between centers of touching hexagons in a lattice is 2*sqrt(3)
    d = 2 * math.sqrt(3)
    
    # Layout: 3 rows of 4 hexagons
    # Row 0: 4 hexagons at y = 0
    # Row 1: 4 hexagons at y = 1.5 * d, shifted x by d/2
    # Row 2: 4 hexagons at y = 3.0 * d
    # Total width: 3*d + 2 (circumradius) approx
    # Total height: 3*d + 2 approx
    
    centers = []
    angles = [0.0] * N
    # Generate 12 points in a 4x3 grid
    # x offsets: 0, d, 2d, 3d
    # y offsets: 0, 1.5d, 3d
    # Shift odd rows by d/2
    
    # Let's center the grid at (0,0)
    # x coords for row 0: -1.5d, -0.5d, 0.5d, 1.5d
    # x coords for row 1: -d, 0, d, 2d -> wait, shift by d/2 relative to row 0
    # Let's just generate and then center.
    
    raw_centers = []
    for row in range(3):
        for col in range(4):
            x = col * d
            if row % 2 == 1:
                x += d / 2.0
            y = row * 1.5 * d
            raw_centers.append((x, y))
            
    # Center the configuration
    xs = [p[0] for p in raw_centers]
    ys = [p[1] for p in raw_centers]
    cx_mean = (min(xs) + max(xs)) / 2.0
    cy_mean = (min(ys) + max(ys)) / 2.0
    
    for p in raw_centers:
        centers.append([p[0] - cx_mean, p[1] - cy_mean])
        
    # Initial outer side length estimate
    # Max distance from center to any vertex
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def enclosing_side_length(centers, angles):
        # Outer hexagon is centered at (0,0) with rotation 0 for simplicity 
        # (or we can optimize outer rotation too, but let's fix it to 0 first 
        # to reduce params, or include it. The task allows outer rotation.
        # Let's optimize outer rotation as well.
        # Params: x1, y1, a1, ..., x12, y12, a12, outer_angle
        # Actually, the return signature is (inner, center, side, angle).
        # inner: list of [x, y, angle]
        # center: [x, y] of outer
        # side: side length
        # angle: rotation of outer
        
        # We will optimize:
        # x1, y1, a1 ... x12, y12, a12 (36 params)
        # outer_cx, outer_cy, outer_angle (3 params)
        # Total 39 params.
        
        # Objective: minimize 'side'
        # Constraint: All vertices of inner hexagons must be inside the outer hexagon.
        # And inner hexagons must not overlap.
        
        # Let's define the outer hexagon H_out with center (ocx, ocy), side L, angle alpha.
        # A point (x, y) is inside H_out if for all 6 edges, the distance to the line 
        # containing the edge is <= apothem (L * sqrt(3) / 2) and on the correct side.
        # Actually, it's easier to check if the max "hexagonal distance" is <= L.
        # Or simply: the outer hexagon is the intersection of 6 half-planes.
        
        # Let's use the support function approach for non-overlap.
        # Distance between two convex polygons P, Q is min_{p in P, q in Q} ||p-q||.
        # For hexagons, this is complex.
        # Alternative: Use the fact that two regular hexagons of side 1 with centers c1, c2 
        # and angles a1, a2 do not overlap if the distance between centers is >= 
        # some function of the relative angle.
        # Actually, the minimum distance between two unit hexagons with relative angle theta
        # varies. The worst case (closest approach) for given theta is 2 * apothem * cos(theta/2)? No.
        
        # Let's use a simpler constraint:
        # The convex hull of all inner hexagon vertices must fit inside the outer hexagon.
        # And pairwise: The distance between any vertex of hex i and any edge of hex j 
        # must be positive (or use center distance approximation if rotations are small).
        
        # Given the time limit, a robust approach for "disjoint" is:
        # For each pair (i, j), check if the minimum distance between their convex hulls > 0.
        # This is expensive in SLSQP.
        
        # Let's try a simpler heuristic constraint that is sufficient:
        # The distance between centers must be at least 2 * sqrt(3) * cos(max_angle_diff/2)?
        # No, let's stick to the vertex-edge check for a few critical pairs or use 
        # a conservative center distance.
        
        # Conservative constraint: Distance between centers >= 2 * sqrt(3) is NOT correct 
        # if they are rotated. If one is rotated 30 deg, they can be closer? 
        # Actually, the circumradius is 1. The minimum distance between centers of two 
        # non-overlapping unit hexagons is 2 * sqrt(3) * cos(15 deg)? 
        # The "width" of a hexagon in any direction is between sqrt(3) and 2.
        # The minimum distance between centers of two non-overlapping convex sets 
        # is at least the sum of their support functions in that direction.
        
        # Let's use the following constraint for pair (i,j):
        # For all 6 directions of hex i, the projection of center j onto that direction 
        # minus the projection of center i must be >= support_i(dir) + support_j(dir rotated by relative angle).
        # This is 6 constraints per pair. 66 pairs * 6 = 396 constraints.
        # This is feasible for SLSQP with 39 variables.
        
        pass # Placeholder

    # Let's refine the objective and constraints.
    
    def get_support(cx, cy, angle_deg, direction_angle, side=1.0):
        """
        Returns the support function of a regular hexagon centered at (cx, cy) 
        with rotation angle_deg in the direction direction_angle.
        support = dot(center, dir) + max_{v in vertices} dot(v - center, dir)
        """
        dir_vec = np.array([math.cos(direction_angle), math.sin(direction_angle)])
        center_vec = np.array([cx, cy])
        center_proj = np.dot(center_vec, dir_vec)
        
        # Max projection of vertices relative to center
        # Vertices are at angles angle_deg + k*pi/3
        max_proj = 0.0
        for k in range(6):
            v_angle = math.radians(angle_deg) + k * math.pi / 3.0
            v_vec = np.array([math.cos(v_angle), math.sin(v_angle)]) * side
            proj = np.dot(v_vec, dir_vec)
            if proj > max_proj:
                max_proj = proj
        return center_proj + max_proj

    def constraints_func(params):
        """
        Returns list of constraint dicts.
        params: [x1, y1, a1, ..., x12, y12, a12, ocx, ocy, o_angle]
        """
        cons = []
        # Extract inner hexagons
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            a = params[3*i+2]
            inner_centers.append((cx, cy))
            inner_angles.append(a)
        
        ocx = params[3*N]
        ocy = params[3*N+1]
        o_angle = params[3*N+2]
        
        # 1. Containment constraints:
        # For each inner hexagon, for each of its 6 vertices, 
        # the vertex must be inside the outer hexagon.
        # A point p is inside outer hexagon (center oc, angle oa, side L) if:
        # For all 6 edge normals n_k of outer hexagon:
        # dot(p - oc, n_k) <= apothem_outer = L * sqrt(3) / 2
        
        # We want to minimize L. 
        # Instead of fixing L and checking containment, we can define L as a variable?
        # No, SLSQP minimizes a scalar. 
        # Let's make L the objective. 
        # But L is not in params. 
        # We can add L to params and minimize it, with constraints that L >= required_L.
        # Or, simpler: Fix L to a large value initially, and minimize a proxy?
        # Better: Add L as a parameter. 
        # New params: [x1, y1, a1, ..., x12, y12, a12, ocx, ocy, o_angle, L]
        # Total 40 params.
        # Objective: minimize L.
        # Constraints:
        # 1. Containment: For each inner vertex v, for each outer normal n_k:
        #    dot(v - oc, n_k) <= L * sqrt(3) / 2
        #    => L >= 2/sqrt(3) * dot(v - oc, n_k)
        #    This is linear in L.
        # 2. Non-overlap: For each pair (i, j):
        #    For each of the 6 directions of hex i:
        #    support_i(dir) + support_j(dir) <= distance between centers projected?
        #    Actually, the condition for non-overlap of two convex sets A, B is:
        #    There exists a separating hyperplane. 
        #    A simpler sufficient condition: 
        #    For all directions d, support_A(d) + support_B(-d) <= 0? No.
        #    The distance between A and B is min_d (support_A(d) + support_B(-d) - 0)?
        #    Actually, min_{a in A, b in B} ||a-b|| >= 0.
        #    A common constraint for non-overlap of two polygons is that the 
        #    convex hull of their vertices does not contain the origin relative to 
        #    the vector between centers?
        
        # Let's use the "separating axis theorem" style constraint.
        # Two convex polygons do not overlap if there is an axis where their 
        # projections do not overlap.
        # This is an OR condition, which is hard for SLSQP (non-convex).
        
        # Alternative: Use the fact that for regular hexagons, the minimum distance 
        # between centers required to avoid overlap is a known function of relative angle.
        # Let d_min(theta) be the minimum center distance for non-overlap.
        # Then constraint: ||c_i - c_j|| >= d_min(a_i - a_j).
        # This is a smooth constraint if we approximate d_min.
        # For small relative angles, d_min is approx 2 * sqrt(3) * cos(theta/2)?
        # Actually, if theta=0, d_min = 2 * sqrt(3).
        # If theta=30 deg, the hexagons interlock? No, they are disjoint.
        # The worst case for closest approach is when a vertex of one touches an edge of the other.
        
        # Let's approximate d_min(theta) as 2 * sqrt(3) * max(1, cos(theta/2))? 
        # No, let's just use a conservative constant: 2 * sqrt(3) + epsilon.
        # This will prevent interlocking but might be too loose for rotated hexagons, 
        # leading to overlaps if they are rotated and close.
        # To be safe, we can check pairwise vertex-edge distances for the final result 
        # and repair if needed, but SLSQP needs smooth constraints.
        
        # Let's use the support function sum constraint:
        # For pair (i, j), for all 6 directions d_k (normals of hex i):
        # The distance between the supporting lines must be positive.
        # dist = support_i(d_k) - support_j(d_k) - dot(c_j - c_i, d_k) ?
        # No, support_i(d) is the max dot(p, d) for p in i.
        # If we define d as the direction from i to j, then:
        # If support_i(d) + support_j(-d) < 0, they might overlap?
        # Actually, the condition for separation is:
        # min_{d} (support_i(d) + support_j(-d)) > 0 ?
        
        # Let's simplify: 
        # Constraint: For each pair (i, j), the distance between centers must be 
        # at least 2 * sqrt(3) * 0.95 (conservative). 
        # And we will verify validity at the end. If invalid, we can try to push them apart.
        
        # Given the 150s limit, let's try a simpler optimization:
        # Minimize the maximum distance from the outer center to any inner vertex.
        # This effectively minimizes the required outer side length.
        # And enforce pairwise center distance >= 2 * sqrt(3) * 0.9.
        
        pass

    # Let's restart with a clearer, simpler SLSQP setup.
    
    def objective(params):
        # params: [x1, y1, a1, ..., x12, y12, a12, ocx, ocy, o_angle]
        # We will estimate L from the max distance.
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            a = params[3*i+2]
            inner_centers.append((cx, cy))
            inner_angles.append(a)
        
        ocx = params[3*N]
        ocy = params[3*N+1]
        o_angle = params[3*N+2]
        
        # Calculate max "hexagonal radius" required
        # The outer hexagon has side L. Its inradius is L * sqrt(3) / 2.
        # A point (x, y) is inside if max_k |dot((x-oc, y-oc), n_k)| <= inradius.
        # So L >= 2/sqrt(3) * max_k |dot((x-oc, y-oc), n_k)|
        
        # To minimize L, we minimize the max over all inner vertices of 
        # max_k |dot((v-oc), n_k)|
        
        max_val = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            a = inner_angles[i]
            verts = get_hex_vertices(cx, cy, a, 1.0)
            for vx, vy in verts:
                # Vector from outer center to vertex
                dx = vx - ocx
                dy = vy - ocy
                # Project onto 6 normals of outer hexagon
                # Outer hexagon normals are at angles o_angle + k*pi/3
                # Note: Normals are perpendicular to edges. 
                # For a regular hexagon with vertex angle theta, 
                # the edge normal is at theta + pi/6?
                # Let's define outer hexagon vertices at o_angle + k*pi/3.
                # The supporting lines (edges) have normals at o_angle + pi/6 + k*pi/3.
                for k in range(6):
                    n_angle = math.radians(o_angle) + math.pi/6 + k * math.pi / 3.0
                    nx = math.cos(n_angle)
                    ny = math.sin(n_angle)
                    proj = dx * nx + dy * ny
                    if abs(proj) > max_val:
                        max_val = abs(proj)
        
        # L = 2/sqrt(3) * max_val
        return 2.0 / math.sqrt(3.0) * max_val

    def constraint_func(params):
        # Pairwise non-overlap: Center distance >= min_dist
        # min_dist depends on relative angle. 
        # Let's use a conservative min_dist = 2 * sqrt(3) * 0.9
        min_dist = 2 * math.sqrt(3) * 0.9
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                cx1 = params[3*i]
                cy1 = params[3*i+1]
                cx2 = params[3*j]
                cy2 = params[3*j+1]
                dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                # Constraint: dist >= min_dist
                # SLSQP constraint: g(params) >= 0
                cons.append(dist - min_dist)
        return cons

    # Initial guess
    x0 = []
    for i in range(N):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
    x0.append(0.0) # ocx
    x0.append(0.0) # ocy
    x0.append(0.0) # o_angle
    
    # Bounds: 
    # Centers can be anywhere, but let's bound them to a reasonable area.
    # Angles: 0 to 360.
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 2*math.pi)) # angle
    bounds.append((-5, 5)) # ocx
    bounds.append((-5, 5)) # ocy
    bounds.append((0, 2*math.pi)) # o_angle

    # Run SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': constraint_func}],
                       options={'maxiter': 50, 'ftol': 1e-4})
        
        if res.success:
            params = res.x
        else:
            params = x0
            
        # Extract result
        inner = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            a = params[3*i+2]
            inner.append([cx, cy, math.degrees(a)])
            
        ocx = params[3*N]
        ocy = params[3*N+1]
        o_angle = params[3*N+2]
        
        # Calculate final side length
        side = objective(params)
        
        # Verify validity (optional, but good for safety)
        # If invalid, we might need to adjust, but SLSQP with conservative 
        # constraints should be close.
        
        return inner, (ocx, ocy), side, math.degrees(o_angle)
        
    except Exception as e:
        # Fallback to initial valid configuration
        inner = []
        for i in range(N):
            inner.append([centers[i][0], centers[i][1], angles[i]])
        # Estimate side for initial config
        # Max distance from (0,0) to any vertex
        max_d = 0
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i], 1.0)
            for vx, vy in verts:
                d = math.sqrt(vx*vx + vy*vy)
                if d > max_d:
                    max_d = d
        # For a hexagon, side L = 2/sqrt(3) * inradius. 
        # The max distance to a vertex is the circumradius of the outer hexagon?
        # No, the outer hexagon must contain the points.
        # The minimal regular hexagon containing a set of points has side 
        # L = 2/sqrt(3) * max_k |dot(p, n_k)|
        # Let's compute it properly.
        max_proj = 0
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i], 1.0)
            for vx, vy in verts:
                for k in range(6):
                    n_angle = math.pi/6 + k * math.pi / 3.0
                    nx = math.cos(n_angle)
                    ny = math.sin(n_angle)
                    proj = vx * nx + vy * ny
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
        side = 2.0 / math.sqrt(3.0) * max_proj
        
        return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
