import numpy as np
import time
from itertools import product, combinations
from scipy.spatial import ConvexHull
import scipy.spatial as sp

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_hexagon_vertices(side, angle_deg):
        return hexagon_vertices(0, 0, side, angle_deg)

    def minkowski_sum(v1, v2):
        """
        Compute the Minkowski sum of two convex polygons (vertices in CCW order).
        Returns vertices of the resulting convex polygon in CCW order.
        """
        # Normalize to start at lowest y, then lowest x
        def normalize(poly):
            min_idx = np.argmin(poly[:, 1])
            # If there are ties in y, pick the one with min x
            min_y = poly[min_idx, 1]
            candidates = np.where(poly[:, 1] == min_y)[0]
            min_x = np.min(poly[candidates, 0])
            final_idx = np.where((poly[:, 1] == min_y) & (poly[:, 0] == min_x))[0][0]
            return np.roll(poly, -final_idx, axis=0)

        p1 = normalize(v1)
        p2 = normalize(v2)
        
        n1, n2 = len(p1), len(p2)
        
        # Edge vectors
        e1 = np.diff(p1, axis=0, append=p1[:1])
        e2 = np.diff(p2, axis=0, append=p2[:1])
        
        # Sort edges by angle
        def edge_angle(e):
            return np.arctan2(e[1], e[0])
        
        a1 = np.array([edge_angle(e) for e in e1])
        a2 = np.array([edge_angle(e) for e in e2])
        
        # Merge edges sorted by angle
        # We need to be careful with the cyclic nature.
        # A simpler robust method for small polygons (n<=6) is to compute all cross sums and take convex hull.
        # Since n1, n2 <= 6, n1*n2 <= 36 points. ConvexHull is very fast.
        points = []
        for p in p1:
            for q in p2:
                points.append(p + q)
        points = np.array(points)
        
        try:
            hull = sp.ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            # Fallback if collinear or degenerate (unlikely for hexagons)
            return points

    def check_disjoint(p1, p2, eps=1e-5):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        p1, p2 are (n, 2) arrays of vertices.
        """
        def project(poly, axis):
            return np.dot(poly, axis)

        # Axes to check: normals of all edges of p1 and p2
        for poly in [p1, p2]:
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                edge = poly[j] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal /= norm
                
                proj1 = project(p1, normal)
                proj2 = project(p2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # If there is a gap, they are disjoint
                if max1 < min2 - eps or max2 < min1 - eps:
                    return True
        return False

    def separate_polygons(p1, p2, minkowski_diff, max_iter=50):
        """
        Translate p2 relative to p1 so that they are disjoint.
        p1 is fixed at origin (or its current pos). p2 is the one to move.
        minkowski_diff is the Minkowski difference p1 - p2 (convex polygon).
        We want to find a translation t such that (p2 + t) is disjoint from p1.
        This is equivalent to t not being in the interior of (p1 - p2).
        Currently, relative pos is 0 (if we consider p2 at origin relative to p1).
        Wait, the Minkowski difference D = p1 - p2 = {a - b | a in p1, b in p2}.
        If p2 is at position q relative to p1, they overlap if q is in D.
        We want to move q to the boundary or outside of D.
        The shortest vector to move q out of D is along the normal to the closest edge of D to q.
        
        Here, we assume p1 and p2 are currently overlapping.
        We compute the Minkowski difference once (it depends only on shapes, not positions).
        Then we iteratively push p2 out.
        """
        # D is the set of vectors v such that p1 and p2+v overlap.
        # We want to find v such that v is NOT in D.
        # Currently v = 0 (relative).
        # We project 0 onto the boundary of D? No, 0 is inside D.
        # We want to move p2 by t. New relative pos is t.
        # We want t to be outside D.
        # The closest point on the boundary of D to 0 is the point to move to?
        # Actually, we want to minimize ||t|| subject to t not in D.
        # This is equivalent to finding the point on the boundary of D closest to 0,
        # and moving slightly beyond it?
        # Or just moving to the boundary? If we move to the boundary, they touch (disjoint with eps).
        
        # Find the point on the boundary of D closest to the origin.
        # Since D is convex and contains 0, the closest point on the boundary to 0
        # is determined by the edge of D that is "closest" to 0.
        
        # For each edge of D, check if the projection of 0 onto the line of the edge
        # falls within the edge segment. If so, the distance is the perpendicular distance.
        # The edge with the minimum such distance is the one to push against.
        
        n = len(minkowski_diff)
        min_dist = float('inf')
        best_normal = None
        best_point = None
        
        for i in range(n):
            p_a = minkowski_diff[i]
            p_b = minkowski_diff[(i+1)%n]
            edge = p_b - p_a
            # Normal pointing inward? We need the normal pointing towards the origin.
            # Since 0 is inside, the vector from edge to 0 should be along the normal.
            # Let's just compute the distance from 0 to the line segment.
            
            # Vector from A to 0
            v = -p_a
            # Project v onto edge
            t = np.dot(v, edge) / (np.dot(edge, edge))
            if 0 <= t <= 1:
                proj = p_a + t * edge
                dist = np.linalg.norm(proj)
                if dist < min_dist:
                    min_dist = dist
                    # Normal pointing from edge to 0
                    normal = -proj / dist
                    best_normal = normal
                    best_point = proj
            else:
                # Closest point is a vertex
                dist_a = np.linalg.norm(p_a)
                if dist_a < min_dist:
                    min_dist = dist_a
                    best_normal = -p_a / dist_a
                    best_point = p_a
                dist_b = np.linalg.norm(p_b)
                if dist_b < min_dist:
                    min_dist = dist_b
                    best_normal = -p_b / dist_b
                    best_point = p_b

        if best_normal is None:
            return np.array([0.0, 0.0])
        
        # We want to move p2 such that the relative position t is just outside D.
        # The vector from 0 to best_point is -best_normal * min_dist.
        # So best_point is in direction -best_normal.
        # To get outside, we should move in direction best_normal?
        # Let's verify.
        # D is the region of overlap. 0 is in D.
        # best_point is on boundary of D.
        # The vector from 0 to best_point points INTO D? No, 0 is inside, best_point is on boundary.
        # The vector best_point - 0 points from center to boundary.
        # To exit D, we must move in the direction of best_point - 0?
        # Yes. If we move t = best_point + epsilon * (best_point/||best_point||), we are outside.
        # Wait, if 0 is the center, and best_point is on the boundary, moving towards best_point takes us to the boundary.
        # To be strictly disjoint (or touching), we can move to best_point.
        # Let's add a small epsilon to ensure separation.
        
        t = best_point + 1e-6 * best_normal # best_normal points from edge to 0?
        # Let's re-evaluate best_normal.
        # In the code above: normal = -proj / dist. proj is the point on edge.
        # So -proj is the vector from proj to 0.
        # So best_normal points from edge to 0 (inward).
        # To move OUT, we need to move in the direction of -best_normal (outward).
        # Direction from 0 to boundary is proj (which is -best_normal * dist).
        # So direction is -best_normal.
        
        t = -best_normal * (min_dist + 1e-6)
        
        return t

    def solve_configuration(angles):
        """
        Given a list of N angles, try to find positions such that hexagons are disjoint.
        Uses iterative separation.
        Returns positions if successful, None otherwise.
        """
        # Initialize positions on a circle
        positions = np.zeros((N, 2))
        for i in range(N):
            a = 2 * np.pi * i / N
            positions[i] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
            
        # Precompute Minkowski differences for all pairs
        # mink_diff[i][j] is the Minkowski difference of hex_i and hex_j
        # Note: mink_diff[i][j] is the set of vectors v such that hex_i and hex_j+v overlap.
        # So if pos_j - pos_i is in mink_diff[i][j], they overlap.
        
        mink_diffs = [[None for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                v_i = get_hexagon_vertices(1.0, angles[i])
                v_j = get_hexagon_vertices(1.0, angles[j])
                # Minkowski sum of v_i and -v_j
                neg_v_j = -v_j
                # Ensure CCW order for ConvexHull? ConvexHull handles it.
                # But minkowski_sum function expects convex polygons.
                # Let's use the ConvexHull approach directly for Minkowski Sum
                points = []
                for p in v_i:
                    for q in neg_v_j:
                        points.append(p + q)
                points = np.array(points)
                try:
                    hull = sp.ConvexHull(points)
                    m_diff = points[hull.vertices]
                except Exception:
                    m_diff = points
                mink_diffs[i][j] = m_diff
                mink_diffs[j][i] = -m_diff # Symmetry? (A-B) = -(B-A). Yes.

        # Iterative separation
        for iteration in range(100):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    # Check if they overlap
                    # Relative position: d = positions[j] - positions[i]
                    # They overlap if d is in mink_diffs[i][j]
                    # Quick check: is d inside mink_diffs[i][j]?
                    # Since mink_diffs[i][j] is convex, we can use point-in-polygon or just SAT.
                    # But SAT is expensive.
                    # Let's just assume if they were separated last time, they might be.
                    # To be safe, check disjointness.
                    v_i = get_hexagon_vertices(1.0, angles[i]) + positions[i]
                    v_j = get_hexagon_vertices(1.0, angles[j]) + positions[j]
                    
                    if not check_disjoint(v_i, v_j):
                        # They overlap. Separate them.
                        # Move j relative to i.
                        # We want to find a translation t for j such that (v_j + t) is disjoint from v_i.
                        # This is equivalent to moving relative pos d by t.
                        # d_new = d + t.
                        # We want d_new to be outside mink_diffs[i][j].
                        # Currently d is inside mink_diffs[i][j].
                        # Use separate_polygons logic on the Minkowski difference.
                        # The function separate_polygons returns the translation to apply to the "second" polygon
                        # relative to the "first" to make them disjoint, assuming they are currently at relative pos 0.
                        # But here they are at relative pos d.
                        # The Minkowski difference M is defined such that overlap happens if relative pos is in M.
                        # We are currently at d. We want to move to d' outside M.
                        # This is the same as: Shift the Minkowski difference by -d.
                        # M' = M - d.
                        # We want to find a point in M' that is outside M'? No.
                        # We want to find a translation t such that d+t is outside M.
                        # This is equivalent to finding a point x in M such that x is "closest to boundary" relative to d?
                        # Actually, just compute the closest point on the boundary of M to the point d.
                        # And move d to that point (plus epsilon).
                        
                        M = mink_diffs[i][j]
                        # Find closest point on boundary of M to d
                        n = len(M)
                        min_dist = float('inf')
                        best_point = None
                        
                        for k in range(n):
                            p_a = M[k]
                            p_b = M[(k+1)%n]
                            edge = p_b - p_a
                            v = d - p_a
                            t_param = np.dot(v, edge) / (np.dot(edge, edge))
                            if 0 <= t_param <= 1:
                                proj = p_a + t_param * edge
                                dist = np.linalg.norm(d - proj)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_point = proj
                            else:
                                dist_a = np.linalg.norm(d - p_a)
                                if dist_a < min_dist:
                                    min_dist = dist_a
                                    best_point = p_a
                                dist_b = np.linalg.norm(d - p_b)
                                if dist_b < min_dist:
                                    min_dist = dist_b
                                    best_point = p_b

                        if best_point is not None:
                            # Move d to best_point + epsilon * normal
                            # Normal pointing out of M at best_point
                            # If best_point is on an edge, normal is perpendicular.
                            # If best_point is a vertex, normal is bisector?
                            # Simplest: move from d to best_point. This puts d on the boundary.
                            # To ensure disjointness, move slightly further.
                            # Direction: (best_point - d) / ||best_point - d||
                            if min_dist > 1e-9:
                                dir_vec = (best_point - d) / min_dist
                            else:
                                dir_vec = np.array([1.0, 0.0])
                            
                            t = (best_point - d) + 1e-6 * dir_vec
                            positions[j] += t
                            changed = True
            if not changed:
                break
        
        # Final verification
        for i in range(N):
            for j in range(i+1, N):
                v_i = get_hexagon_vertices(1.0, angles[i]) + positions[i]
                v_j = get_hexagon_vertices(1.0, angles[j]) + positions[j]
                if not check_disjoint(v_i, v_j, eps=1e-4):
                    return None
        
        return positions

    def calculate_side_length(positions, angles):
        """
        Calculate the minimum side length S of a regular hexagon (centered at origin, angle 0)
        that contains all hexagons.
        """
        # Collect all vertices
        all_verts = []
        for i in range(N):
            v = get_hexagon_vertices(1.0, angles[i]) + positions[i]
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        x = all_verts[:, 0]
        y = all_verts[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    # Discrete search over angles
    # Symmetries: Rotating all hexagons by 30 degrees doesn't change the problem structure relative to the outer hexagon?
    # Actually, the outer hexagon is fixed at angle 0.
    # Inner hexagons can be rotated.
    # Let's try a few combinations.
    
    best_side = float('inf')
    best_config = None
    
    # Candidate angle sets
    # 1. All 0
    # 2. All 30
    # 3. Mix 0 and 30
    # 4. Mix 0, 30, 60 (60 is same as 0 for hexagon shape)
    
    angle_candidates = [0, 30]
    
    # Generate a subset of combinations to keep runtime low
    # There are 2^12 = 4096 combinations. This is small enough to try all if solver is fast.
    # But solver has 100 iterations * 66 pairs * SAT checks.
    # Let's try a heuristic selection first, then random sampling if time permits.
    
    configs_to_try = []
    
    # 1. All 0
    configs_to_try.append([0]*N)
    # 2. All 30
    configs_to_try.append([30]*N)
    
    # 3. Alternating
    configs_to_try.append([0 if i%2==0 else 30 for i in range(N)])
    configs_to_try.append([30 if i%2==0 else 0 for i in range(N)])
    
    # 4. Random samples
    np.random.seed(42)
    for _ in range(20):
        angles = [np.random.choice(angle_candidates) for _ in range(N)]
        configs_to_try.append(angles)
        
    # 5. Structured: Clusters of same angle
    for k in range(1, N):
        configs_to_try.append([0]*k + [30]*(N-k))
        configs_to_try.append([30]*k + [0]*(N-k))
        
    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs_to_try:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
            
    for angles in unique_configs:
        if time.time() > deadline - 5:
            break
            
        positions = solve_configuration(angles)
        if positions is not None:
            side = calculate_side_length(positions, angles)
            if side < best_side:
                best_side = side
                best_config = (positions, angles)
                
    # If no valid config found, fallback to a very large square-like arrangement
    if best_config is None:
        # Simple grid
        positions = np.zeros((N, 2))
        angles = [0]*N
        for i in range(N):
            row = i // 4
            col = i % 4
            positions[i] = [col * 2.1, row * 2.1]
        best_side = calculate_side_length(positions, angles)
        best_config = (positions, angles)

    positions, angles = best_config
    
    # Final check
    if not all(check_disjoint(get_hexagon_vertices(1.0, angles[i]) + positions[i], 
                              get_hexagon_vertices(1.0, angles[j]) + positions[j], eps=1e-4) 
               for i, j in combinations(range(N), 2)):
        # Invalid, expand slightly
        scale = 1.1
        positions = positions * scale
        best_side = calculate_side_length(positions, angles)

    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
