"""Strategy: Constrained nonlinear optimization using SLSQP to minimize outer hexagon side length."""
import numpy as np
import time
import scipy.optimize as opt
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Helper: get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle (radians)
    def get_hex_vertices(cx, cy, angle):
        # Base angles for flat-top hexagon (vertices at 0, 60, ...) or pointy?
        # Standard math: vertices at angles 0, 60, 120, 180, 240, 300.
        # This corresponds to a "pointy" top if y is up? 
        # Let's stick to consistent definitions.
        # Side length 1. Circumradius 1.
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        rotated_angles = angles + angle
        xs = cx + np.cos(rotated_angles)
        ys = cy + np.sin(rotated_angles)
        return np.column_stack((xs, ys))

    # Helper: Check distance between two convex polygons (hexagons)
    # Returns the minimum distance. If overlapping, returns negative distance (penetration depth approx).
    # We use the Separating Axis Theorem (SAT) for distance.
    # For two convex polygons, if there is a separating axis, the distance is the max over axes of (projection difference).
    # Actually, for non-overlapping convex polygons, the minimum distance is the minimum distance between any edge of A and any edge of B, or vertex to edge.
    # A robust way for small polygons:
    # 1. Check if any vertex of A is inside B. If so, overlap.
    # 2. Check if any vertex of B is inside A. If so, overlap.
    # 3. Check if any edge of A intersects any edge of B. If so, overlap.
    # 4. If no overlap, calculate min distance between all pairs of segments (edges).
    
    def point_in_convex_polygon(p, poly):
        # poly: (N, 2) array, counter-clockwise
        # Returns True if p is inside or on boundary
        n = len(poly)
        sign = None
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            cross = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
            if abs(cross) < 1e-10:
                continue
            if sign is None:
                sign = 1 if cross > 0 else -1
            else:
                if (cross > 0 and sign == -1) or (cross < 0 and sign == 1):
                    return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def dist_point_segment(p, a, b):
        # Distance from point p to segment ab
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-10:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def min_dist_segments(a1, a2, b1, b2):
        # Minimum distance between segment a1-a2 and b1-b2
        # If they intersect, dist is 0.
        if segments_intersect(a1, a2, b1, b2):
            return 0.0
        # Check point-to-segment distances
        d1 = dist_point_segment(a1, b1, b2)
        d2 = dist_point_segment(a2, b1, b2)
        d3 = dist_point_segment(b1, a1, a2)
        d4 = dist_point_segment(b2, a1, a2)
        return min(d1, d2, d3, d4)

    def min_dist_hexagons(hex1, hex2):
        # hex1, hex2: (6, 2) arrays
        # Check for overlap
        for v in hex1:
            if point_in_convex_polygon(v, hex2):
                return -0.1 # Overlapping, return negative to indicate violation
        for v in hex2:
            if point_in_convex_polygon(v, hex1):
                return -0.1
        # Check edge intersections
        for i in range(6):
            for j in range(6):
                if segments_intersect(hex1[i], hex1[(i+1)%6], hex2[j], hex2[(j+1)%6]):
                    return -0.1
        # Calculate min distance
        min_d = float('inf')
        for i in range(6):
            for j in range(6):
                d = min_dist_segments(hex1[i], hex1[(i+1)%6], hex2[j], hex2[(j+1)%6])
                if d < min_d:
                    min_d = d
        return min_d

    # Outer hexagon geometry
    # Center (0,0), side length S, rotation 0.
    # Normals to edges: 30, 90, 150, 210, 270, 330 degrees.
    outer_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    outer_normals = np.column_stack((np.cos(outer_normals), np.sin(outer_normals)))

    def objective(vars):
        # vars: [x0, y0, theta0, x1, y1, theta1, ..., x10, y10, theta10, S]
        # Total size: 3*N + 1 = 34
        S = vars[-1]
        return S

    def constraints(vars):
        cons = []
        S = vars[-1]
        R_out = S * np.sqrt(3) / 2.0
        
        # Extract positions and angles
        centers = vars[:3*N].reshape(N, 3)
        
        # Precompute vertices for all hexagons
        verts_list = []
        for i in range(N):
            x, y, theta = centers[i]
            verts = get_hex_vertices(x, y, theta)
            verts_list.append(verts)
            
        # Constraint 1: Containment in outer hexagon
        # For each hexagon i, and each vertex v, and each normal n_k:
        # dot(v, n_k) <= R_out
        # This is a lot of constraints: 11 * 6 * 6 = 396.
        # We can group them: for each hex i, the max projection of its vertices onto each normal must be <= R_out.
        # Actually, since the hexagon is convex, the max projection of the hexagon onto a normal is the max projection of its vertices.
        for i in range(N):
            verts = verts_list[i]
            # Project vertices onto all 6 normals
            # verts: (6,2), outer_normals: (6,2)
            # We need: for each normal k, max_j (dot(verts[j], normal[k])) <= R_out
            # This is equivalent to: max_j (sum_k? No.)
            # Let's just add the 6 constraints per hexagon.
            # The "worst" vertex for a given normal is the one with max dot product.
            # To keep it simple for SLSQP, we can add the constraint for the vertex that is furthest out in each direction?
            # No, we must ensure ALL vertices are inside.
            # Since the hexagon is convex, if all vertices are inside the outer hexagon (which is convex), the whole hexagon is inside.
            # So we need: for all vertices v of hex i, and all normals n_k: dot(v, n_k) <= R_out.
            # This is 6*6 = 36 constraints per hexagon. Total 396.
            # This might be slow for SLSQP.
            # Optimization: The max projection of a convex polygon onto a direction is achieved at a vertex.
            # So for each normal n_k, we need max_{v in hex_i} dot(v, n_k) <= R_out.
            # We can compute this max and add 6 constraints per hexagon.
            for k in range(6):
                proj = np.dot(verts, outer_normals[k])
                max_proj = np.max(proj)
                # Constraint: max_proj - R_out <= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, mp=max_proj, R=R_out: R - mp
                })
        
        # Constraint 2: Disjointness
        # For each pair (i, j), min_dist >= 0
        # SLSQP requires differentiable functions. min_dist is not differentiable at 0 (cusp) or when overlap changes type.
        # However, SLSQP can often handle non-smooth constraints if we are lucky, or we can use a smooth approximation.
        # Better: use a smooth penalty in the objective? No, we want hard constraints.
        # Alternative: Use 'COBYLA' method which handles non-smooth constraints better?
        # COBYLA is slower but more robust for non-smooth.
        # Let's try SLSQP first. If it fails, switch to COBYLA or use a smooth approximation like -log(dist + eps).
        # Let's use a smooth approximation for disjointness to help SLSQP.
        # Constraint: dist(i, j) >= 0.
        # Smoothed: dist(i, j) + eps * exp(-dist(i, j)) >= 0? No.
        # Let's just pass the raw min_dist. SLSQP uses finite differences. It might struggle.
        # Let's try COBYLA instead, as it doesn't require gradients and handles non-smoothness better.
        
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_hexagons(verts_list[i], verts_list[j])
                # Constraint: d >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, dd=d: dd
                })
        
        return cons

    # Initial guess
    # Hexagonal lattice packing
    # Center (0,0)
    # Neighbors at distance sqrt(3) * 2? No, distance between centers of touching unit hexagons is sqrt(3).
    # Wait, inradius is sqrt(3)/2. Distance between centers of touching hexagons is 2 * inradius = sqrt(3).
    
    def generate_initial_guess():
        # 1 in center
        # 6 around
        # 4 more?
        # Let's place 11 in a 3x3 grid minus 2 corners, or similar.
        # Lattice vectors for hexagonal packing:
        # v1 = (sqrt(3), 0)
        # v2 = (sqrt(3)/2, 1.5)  -- Wait, y spacing for flat top?
        # Let's use standard hex lattice.
        # Centers:
        # (0,0)
        # (sqrt(3), 0), (-sqrt(3), 0)
        # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (sqrt(3)/2, -1.5), (-sqrt(3)/2, -1.5)
        # That's 7.
        # Next ring:
        # (2*sqrt(3), 0)
        # (3*sqrt(3)/2, 3) -- No, distance is sqrt(3).
        # Let's just generate a list of 11 positions.
        pos = []
        # Ring 0
        pos.append((0, 0))
        # Ring 1
        for k in range(6):
            angle = k * np.pi / 3
            pos.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
        # Ring 2 (pick 4)
        # (2*sqrt(3), 0)
        # (sqrt(3)/2, 3*sqrt(3)/2) ? No.
        # Let's just add 4 more in the outer ring.
        # (2*sqrt(3), 0)
        # (sqrt(3), 3*sqrt(3)/2) ?
        # Let's calculate coordinates for ring 2.
        # Distance from center is 2 * sqrt(3) * cos(pi/6)? No.
        # In hex lattice, ring r has distance r * sqrt(3).
        # So ring 2 distance is 2 * sqrt(3).
        # Angles: 0, 60, 120...
        # Let's pick 4 points from ring 2.
        for k in [0, 1, 4, 5]:
            angle = k * np.pi / 3
            pos.append((2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)))
            
        # Total 1 + 6 + 4 = 11.
        
        vars = []
        for x, y in pos:
            vars.extend([x, y, 0.0])
        # Initial S
        # Bounding box of these points.
        # Max radius is 2*sqrt(3) ~ 3.46.
        # Outer hex inradius R_out must be > max projection.
        # Max projection of a point at dist 3.46 is 3.46.
        # R_out = S * sqrt(3) / 2.
        # S * 0.866 > 3.46 => S > 4.0.
        # Let's start with S = 5.0
        vars.append(5.0)
        return np.array(vars)

    best_result = None
    best_S = float('inf')
    
    # Try multiple initializations
    inits = [
        generate_initial_guess(),
        # Random perturbation
        np.array(generate_initial_guess()) + np.random.randn(3*N + 1) * 0.5,
        # Another lattice offset
        np.array(generate_initial_guess()) + np.array([0.1, 0.1, 0]*N + [0])
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use COBYLA for robustness against non-smooth constraints
            # COBYLA does not support bounds directly, so we rely on constraints or penalization.
            # SLSQP supports bounds. Let's try SLSQP first with tight bounds.
            # Bounds: x, y in [-10, 10], theta in [0, 2*pi], S in [4, 10]
            bounds = [(-10, 10), (-10, 10), (0, 2*np.pi)] * N + [(4.0, 10.0)]
            
            res = opt.minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                S_val = res.x[-1]
                # Verify validity manually to be safe
                vars = res.x
                centers = vars[:3*N].reshape(N, 3)
                verts_list = [get_hex_vertices(centers[i,0], centers[i,1], centers[i,2]) for i in range(N)]
                
                valid = True
                R_out = S_val * np.sqrt(3) / 2.0
                # Check containment
                for i in range(N):
                    for v in verts_list[i]:
                        for k in range(6):
                            if np.dot(v, outer_normals[k]) > R_out + 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Check disjointness
                    for i in range(N):
                        for j in range(i+1, N):
                            d = min_dist_hexagons(verts_list[i], verts_list[j])
                            if d < -1e-6: # Overlapping
                                valid = False
                                break
                        if not valid: break
                
                if valid and S_val < best_S:
                    best_S = S_val
                    best_result = res.x
                    print(f"Found valid packing with S={S_val}")
                    
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue

    if best_result is None:
        # Fallback: Return a large valid packing
        # Just use the initial guess with a large S
        init = generate_initial_guess()
        S = 8.0
        init[-1] = S
        best_result = init
        best_S = S

    # Format output
    vars = best_result
    centers = vars[:3*N].reshape(N, 3)
    S_val = vars[-1]
    
    inner = []
    for i in range(N):
        x, y, theta = centers[i]
        # Convert theta to degrees
        angle_deg = np.rad2deg(theta) % 360.0
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), S_val, 0.0

