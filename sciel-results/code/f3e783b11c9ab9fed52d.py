import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Helper to get vertices of a regular hexagon
    def get_hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    # Calculate minimal distance between centers of two unit hexagons 
    # with relative angle delta_theta such that they are disjoint (touching)
    # This is the support function distance.
    def min_center_distance(angle1, angle2):
        # Due to symmetry, we only care about the relative angle modulo 60 degrees
        delta = (angle2 - angle1) % (np.pi / 3.0)
        # The distance is the sum of the support functions in the direction of separation.
        # For a regular hexagon, the support function h(theta) = cos(pi/6) / cos(theta - pi/6) 
        # where theta is angle relative to a vertex normal.
        # However, a simpler geometric fact: 
        # Two regular polygons are disjoint if and only if the distance between centers 
        # is at least the sum of their in-radii (apothems) if aligned, 
        # but more generally, it's the minimum distance between boundaries.
        
        # Let's use the property that the distance between centers must be at least 
        # the minimum over all angles of (h1(alpha) + h2(alpha + pi))
        # where h is the support function.
        # For a regular hexagon of side 1:
        # h(alpha) = (sqrt(3)/2) / cos((alpha % (pi/3)) - pi/6)
        
        # The minimum distance between two such shapes is the maximum over all directions 
        # of the sum of their support functions? No, it's the minimum distance between boundaries.
        # Actually, for two convex bodies A and B, dist(A,B) = max_{u} (h_A(u) - h_B(-u) - u.x) ? No.
        # dist(A,B) = min_{a in A, b in B} ||a-b||.
        
        # A robust way for hexagons:
        # The distance is the maximum of the projections on the 6 normal directions of the "Minkowski Sum" boundary?
        # Let's just compute it numerically once for a grid of relative angles or use an analytic formula.
        
        # Analytic formula for distance between two regular hexagons of side 1 with relative angle delta:
        # If delta = 0, dist = sqrt(3) (side to side)
        # If delta = pi/6, dist = 2 (vertex to vertex)
        
        # The function is periodic with period pi/3 and symmetric around pi/6.
        # It is actually linear? No.
        # Let's approximate or derive.
        # The support function of a unit hexagon (flat top, vertex at pi/3, 2pi/3...)
        # Let's define hexagon with vertex at angle 0.
        # h(t) = cos(pi/6) / cos(t - pi/6) for t in [-pi/6, pi/6] relative to vertex normal?
        # Vertex normals are at 0, pi/3, 2pi/3...
        # Side normals are at pi/6, pi/2, 5pi/6...
        
        # Let's use the fact that the distance is the minimum k such that 
        # Hex1 + k*UnitDisk does not intersect Hex2? No.
        
        # Let's use a simple lookup table or interpolation for safety, or just use 
        # the conservative bound: dist >= sqrt(3) (inradius * 2) is NOT correct for rotated.
        # Wait, if you rotate one hexagon by 30 degrees, the vertex touches the side midpoint.
        # Distance is apothem + apothem? No.
        # Vertex of Hex1 is at dist 1 from center. Side midpoint of Hex2 is at dist sqrt(3)/2 from center.
        # If aligned such that vertex of 1 points to side of 2, the distance is 1 + sqrt(3)/2?
        # No, they must not overlap.
        
        # Let's calculate the exact minimum distance for two unit hexagons with relative angle delta.
        # This value is known as the "contact distance".
        # For delta in [0, pi/6]:
        # The contact happens between a vertex of one and a side of the other?
        # At delta=0, side-side contact. Distance = sqrt(3).
        # At delta=pi/6, vertex-vertex contact. Distance = 2.
        # Is it linear?
        # Let's check delta=pi/12 (15 deg).
        # It turns out the distance function is:
        # d(delta) = sqrt(3) * cos(delta) + 1 * sin(delta) ? No.
        
        # Let's use a robust numerical calculation for the specific pairs during optimization 
        # if we can't derive it easily, but we need it to be fast.
        # Actually, we can pre-compute a lookup table for relative angles.
        
        # Pre-computation for relative angle delta in [0, pi/6]
        # We sample the boundary of Hex1 and Hex2.
        # But doing this inside the objective is too slow.
        
        # Let's use the analytical solution for the distance between two regular hexagons.
        # Reference: "Distance between two regular hexagons"
        # For relative angle alpha in [0, pi/6]:
        # The distance is given by:
        # d(alpha) = (sqrt(3) * cos(alpha) + sin(alpha)) / (cos(alpha) + sqrt(3)*sin(alpha)/3) ? No.
        
        # Let's derive it quickly.
        # Hex1 vertex at (1, 0). Side from (1,0) to (0.5, sqrt(3)/2).
        # Hex2 rotated by alpha.
        # The minimum distance is the maximum of the separation over all 12 normal directions 
        # (6 from each hexagon) of the Minkowski difference?
        # No, the distance between two convex sets is the maximum over all unit vectors u 
        # of (min_{x in A} u.x - max_{y in B} u.y) ? No, that's for separating hyperplanes.
        # If they are disjoint, there exists a separating hyperplane.
        # The distance is max_{u} (h_A(u) - h_B(-u) - u.c)?
        
        # Let's just use a conservative lower bound that is easy to compute, 
        # or a tight bound.
        # A safe lower bound is sqrt(3) (the inradius sum) ONLY if we assume they can't interlock.
        # But they CAN interlock (rotate). The maximum distance required is 2.
        # The minimum distance required is sqrt(3).
        # The function is monotonic from sqrt(3) to 2 as delta goes 0 to pi/6.
        
        # Let's use the formula: d(delta) = 1 / cos(delta - pi/6) * cos(pi/6) + ...?
        
        # Actually, let's use a simple interpolation table generated at startup.
        pass

    # Generate lookup table for min distance between two unit hexagons
    # relative_angle: 0 to pi/6
    # We will compute this by checking separation for a grid of angles.
    delta_grid = np.linspace(0, np.pi/6, 100)
    dist_table = np.zeros_like(delta_grid)
    
    for i, d in enumerate(delta_grid):
        # Hex1 at origin, angle 0
        # Hex2 at distance D, angle d. We want the smallest D such that they are disjoint.
        # Binary search on D? Or just compute the support function sum?
        # The distance is the minimum over all directions of the sum of support functions?
        # No, the distance between two convex sets A and B is:
        # min_{a in A, b in B} ||a-b||
        # This is equivalent to finding the minimum norm vector in the Minkowski Difference A-B that doesn't cross 0?
        # Actually, for disjoint convex sets, the distance is the maximum over all unit vectors u 
        # of (h_A(u) - h_B(-u) - u.p) where p is the vector between centers?
        # This is getting complicated.
        
        # Simpler: The distance is the maximum of the projections of the gap on the 12 normal vectors 
        # of the two hexagons? No.
        
        # Let's use the property that the distance is the minimum D such that 
        # max_{u in N1 union N2} (h1(u) + h2(u) - u.D) <= 0?
        # Where h is support function.
        # h1(u) = max_{v in Hex1} u.v
        # h2(u) = max_{v in Hex2} u.v
        # If Hex2 is shifted by D*u_sep, then...
        
        # Let's just hardcode the known values for hexagons.
        # At delta=0, dist = sqrt(3)
        # At delta=pi/6, dist = 2
        # The function is actually:
        # d(delta) = (sqrt(3) * cos(delta) + sin(delta)) / (cos(delta) + sqrt(3)/3 * sin(delta)) 
        # Let's test delta=0: (sqrt(3) + 0) / (1 + 0) = sqrt(3). Correct.
        # Let's test delta=pi/6: 
        # cos(pi/6)=sqrt(3)/2, sin(pi/6)=1/2
        # Num: sqrt(3)*sqrt(3)/2 + 1/2 = 3/2 + 1/2 = 2
        # Denom: sqrt(3)/2 + sqrt(3)/3 * 1/2 = sqrt(3)/2 + sqrt(3)/6 = 3sqrt(3)/6 + sqrt(3)/6 = 4sqrt(3)/6 = 2sqrt(3)/3
        # Result: 2 / (2sqrt(3)/3) = 3/sqrt(3) = sqrt(3) ?? 
        # Wait, 3/sqrt(3) is sqrt(3). But it should be 2.
        # So the formula is wrong.
        
        # Let's try: d(delta) = 1 / cos(delta) ? No.
        
        # Let's compute it numerically for the table.
        # For a given relative angle d, find min D such that Hex1 and Hex2 shifted by D along X axis are disjoint?
        # No, they can be positioned optimally. The "min center distance" is the minimum D such that 
        # there EXISTS an orientation/positioning where they touch.
        # Actually, the "min center distance" for a FIXED relative angle is the distance when they touch.
        # This is the support function of the Minkowski Sum of Hex1 and -Hex2?
        # The distance is h_{Hex1 \oplus -Hex2}(u) for the optimal u?
        
        # Let's just use the conservative bound: sqrt(3). 
        # This ensures validity but might be suboptimal (hexagons further apart than necessary).
        # To improve, we can allow closer packing by using the exact distance.
        # Let's compute the exact distance for the table.
        
        # Vertices of Hex1 (angle 0):
        v1 = get_hex_vertices(0, 0, 1, 0)
        # Vertices of Hex2 (angle d):
        v2 = get_hex_vertices(0, 0, 1, d)
        
        # The distance between two convex polygons is the minimum distance between any edge of one and any edge of the other, 
        # or vertex to edge.
        # We can compute this by checking all pairs of edges.
        min_d = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                q1 = v2[j]
                q2 = v2[(j+1)%6]
                # Distance between segment p1-p2 and q1-q2
                # Since they are disjoint in the "table" sense (we are finding the touching distance),
                # we can just find the min distance between any point on seg1 and any point on seg2.
                # This is a standard geometry problem.
                # For speed, we only need to do this 100 times.
                
                # Function to compute min distance between two segments
                def seg_seg_dist(p1, p2, q1, q2):
                    # Project q1, q2 onto p1p2
                    # ... complex.
                    # Let's use a simpler approach: 
                    # The distance is the minimum of distances from vertices to the other segment.
                    def point_seg_dist(p, a, b):
                        ab = b - a
                        ap = p - a
                        t = np.dot(ap, ab) / np.dot(ab, ab)
                        t = np.clip(t, 0, 1)
                        closest = a + t * ab
                        return np.linalg.norm(p - closest)
                    
                    d1 = point_seg_dist(q1, p1, p2)
                    d2 = point_seg_dist(q2, p1, p2)
                    d3 = point_seg_dist(p1, q1, q2)
                    d4 = point_seg_dist(p2, q1, q2)
                    return min(d1, d2, d3, d4)
                
                d_val = seg_seg_dist(p1, p2, q1, q2)
                if d_val < min_d:
                    min_d = d_val
                    
        dist_table[i] = min_d

    # Interpolate distance table
    def get_min_dist(angle1, angle2):
        delta = (angle2 - angle1) % (np.pi / 3.0)
        # Ensure delta is in [0, pi/6]
        if delta > np.pi / 6.0:
            delta = np.pi / 3.0 - delta
        return np.interp(delta, delta_grid, dist_table)

    def objective(params):
        if time.time() > deadline:
            return 1e10

        # Unpack
        # params: 12 * 3 = 36 values (x, y, angle)
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]

        # 1. Check constraints (pairwise disjoint)
        # We use the precomputed min distance table.
        # If dist(centers[i], centers[j]) < min_dist(angles[i], angles[j]) + eps, it's invalid.
        # To make it differentiable, we add a penalty for violations.
        penalty = 0.0
        eps = 1e-5
        
        for i in range(N):
            for j in range(i+1, N):
                d_centers = np.linalg.norm(centers[i] - centers[j])
                d_min = get_min_dist(angles[i], angles[j])
                if d_centers < d_min + eps:
                    # Penalty: square of the violation
                    violation = d_min + eps - d_centers
                    penalty += violation ** 2

        if penalty > 1e-4:
            # Strong penalty if invalid
            return penalty * 100.0 + outer_size_estimate(centers, angles)

        # 2. Calculate outer hexagon side length
        # The outer hexagon must contain all inner hexagons.
        # The required side length S is determined by the "hexagonal radius" of the set of all vertices.
        # We want to minimize S.
        # The outer hexagon is centered at (0,0) with angle 0 for simplicity? 
        # The task allows the outer hexagon to rotate. 
        # To minimize S, we should align the outer hexagon optimally.
        # However, for a first pass, let's assume outer hexagon angle is 0.
        # We can add outer_angle to params later if needed, but it adds dimension.
        # Let's stick to outer_angle = 0 for now, but center is (0,0).
        # Actually, the center of the outer hexagon should be the centroid of the cluster to minimize S.
        # Let's assume outer center is (0,0) and we optimize the inner positions relative to it.
        
        # Compute vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts)
        
        # For a regular hexagon with side S, angle 0, centered at origin:
        # A point (x,y) is inside if:
        # |x| <= S
        # |y| <= S * sqrt(3) / 2
        # |x| + |y|/sqrt(3) <= S
        
        # So S must be >= max( |x|, |y|*2/sqrt(3), |x| + |y|/sqrt(3) ) for all points.
        
        s_req = np.max([
            np.max(np.abs(all_verts[:, 0])),
            np.max(np.abs(all_verts[:, 1])) * 2.0 / np.sqrt(3.0),
            np.max(np.abs(all_verts[:, 0]) + np.abs(all_verts[:, 1]) / np.sqrt(3.0))
        ])
        
        return s_req + penalty

    # Initial guess: Honeycomb
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_centers.append([x, y])
    init_centers = np.array(init_centers)
    init_angles = np.zeros(N)
    
    init_params = np.concatenate([init_centers.flatten(), init_angles])
    
    # Bounds
    bounds = [(-5, 5)] * (2*N) + [(-np.pi/6, np.pi/6)] * N
    
    # Optimization
    best_val = objective(init_params)
    best_params = init_params
    
    # Try L-BFGS-B
    try:
        res = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    time_elapsed = time.time() - start_time
    if time_elapsed < 120.0:
        for restart in range(5):
            # Perturb initial guess
            noise = np.random.normal(0, 0.5, 2*N)
            noise_angles = np.random.uniform(-0.1, 0.1, N)
            p = init_params.copy()
            p[:2*N] += noise
            p[2*N:] += noise_angles
            
            try:
                res = minimize(
                    objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
            
            if time.time() > deadline - 10:
                break

    # Extract results
    final_centers = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:3*N]
    
    inner = []
    for i in range(N):
        inner.append([float(final_centers[i, 0]), float(final_centers[i, 1]), float(np.degrees(final_angles[i]))])
        
    # Outer hexagon
    # We assumed center (0,0) and angle 0.
    # Side length is best_val (minus any residual penalty, but if valid, penalty is 0)
    # We should re-calculate the exact side length for the final configuration to be precise.
    
    all_verts = []
    for i in range(N):
        v = get_hex_vertices(final_centers[i, 0], final_centers[i, 1], 1.0, final_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    s_final = np.max([
        np.max(np.abs(all_verts[:, 0])),
        np.max(np.abs(all_verts[:, 1])) * 2.0 / np.sqrt(3.0),
        np.max(np.abs(all_verts[:, 0]) + np.abs(all_verts[:, 1]) / np.sqrt(3.0))
    ])
    
    return inner, (0.0, 0.0), float(s_final), 0.0

# EVOLVE-BLOCK-END
