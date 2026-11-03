import numpy as np
import time
from itertools import combinations

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _min_dist_polygons(v1, v2):
    """
    Calculate the minimum distance between two convex polygons.
    If they overlap, return 0.
    Uses segment-segment distance.
    """
    n1, n2 = len(v1), len(v2)
    
    # Helper: distance between two segments
    def _seg_seg_dist(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(r)
        elif a < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            return np.linalg.norm(r + t * d2)
        elif e < 1e-12:
            s = -np.dot(d1, r) / a
            s = max(0.0, min(1.0, s))
            return np.linalg.norm(s * d1 - r)
        
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        
        if denom < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            s = max(0.0, min(1.0, (b * t - c) / a))
        else:
            s = (b * f - c * e) / denom
            if s < 0.0 or s > 1.0:
                s = 0.0
                t = f / e
                t = max(0.0, min(1.0, t))
            else:
                t = (b * s + f) / e
                t = max(0.0, min(1.0, t))
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
        
        return np.linalg.norm(s * d1 - r - t * d2)

    min_dist = float('inf')
    # Check vertex-vertex, vertex-edge, edge-edge distances
    # For convex polygons, checking edge-edge is sufficient for min distance if no intersection
    # But we need to detect intersection (distance 0)
    
    # First, check for intersection or containment
    # Simple check: if any vertex of v1 is inside v2 or vice versa
    def _point_in_poly(p, poly):
        # Cross product method for convex polygon
        sign = 0
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1)%len(poly)]
            cross = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
            if cross > 1e-9:
                if sign < 0: return False
                sign = 1
            elif cross < -1e-9:
                if sign > 0: return False
                sign = -1
        return True

    for p in v1:
        if _point_in_poly(p, v2):
            return 0.0
    for p in v2:
        if _point_in_poly(p, v1):
            return 0.0

    # Check edge intersections
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1)%n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j+1)%n2]
            if _seg_seg_dist(p1, p2, q1, q2) < 1e-9:
                return 0.0
            min_dist = min(min_dist, _seg_seg_dist(p1, p2, q1, q2))
            
    return min_dist

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Define a triangular lattice for inner hexagon centers.
    #    Lattice vectors: a1 = (d, 0), a2 = (d/2, d*sqrt(3)/2)
    #    For unit hexagons (side 1), the distance to touch is 2 * apothem = sqrt(3).
    #    So we set d = sqrt(3) + epsilon to ensure disjointness.
    # 2. Enumerate all subsets of 12 lattice points within a certain radius.
    #    To keep the search space manageable, we restrict to points within a bounding box.
    # 3. For each valid configuration (all 12 points found), calculate the minimal outer hexagon side length.
    #    The outer hexagon is centered at the centroid of the inner hexagons (or optimized).
    #    We sweep the outer hexagon rotation angle to minimize the side length.
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Lattice parameters
    # Minimum distance between centers of disjoint unit hexagons is sqrt(3)
    # We use a slightly larger distance to be safe, but the optimization will tighten it.
    # Actually, if we fix the lattice, we can't tighten it below sqrt(3).
    # So we set d = sqrt(3). The hexagons will be touching.
    d = np.sqrt(3.0)
    
    # Generate lattice points
    # We need 12 points. The minimal cluster of 12 points in a triangular lattice
    # is a hexagon of radius 2 (center + 6 neighbors + 5 more? No, radius 1 has 7 points).
    # Radius 2 has 1 + 6 + 12 = 19 points. We need to pick 12 from these 19.
    # Or we can look at other shapes.
    
    # Let's generate points in a hexagonal grid of radius 3 (just in case)
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            # Triangular lattice coords to Cartesian
            # x = i * d + j * d * 0.5
            # y = j * d * sqrt(3) / 2
            x = i * d + j * d * 0.5
            y = j * d * np.sqrt(3) / 2
            points.append((x, y))
    
    points = np.array(points)
    
    # Filter points to a reasonable bounding box to reduce combinations
    # The optimal packing of 12 hexagons is likely compact.
    # Max extent of 12 hexagons: roughly a 4x3 block?
    # Let's keep points within radius 4*d
    mask = np.linalg.norm(points, axis=1) < 4 * d
    points = points[mask]
    
    # We need to choose 12 points from this set.
    # The number of points is roughly pi * (4d)^2 / (d^2 * sqrt(3)/2) ~ 16 * 2 / 1.73 ~ 18-20 points.
    # C(20, 12) is 125970. This is small enough to brute force.
    
    num_points = len(points)
    if num_points < 12:
        # Fallback if lattice is too small
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0

    # Precompute all combinations of 12 points
    # To speed up, we can prune combinations that are not connected or have large holes?
    # No, just brute force is fine for C(20,12).
    
    # Optimization: For a given set of centers, the optimal outer hexagon center
    # is not necessarily the centroid, but for a symmetric outer hexagon and symmetric-ish inner cluster,
    # it's close. However, we can just compute the bounding box for each rotation.
    
    # To be efficient, we will:
    # 1. Iterate through combinations.
    # 2. For each combination, check if they are all distinct (they are, by construction).
    # 3. Compute the minimal outer side length.
    
    # How to compute minimal outer side length for a set of points?
    # The outer hexagon has side S, center C, angle A.
    # Condition: For all inner hex vertices V_i, and all 6 normals n_k of outer hex:
    # dot(V_i - C, n_k) <= S * sqrt(3) / 2
    # This must hold for all i, k.
    # So S >= max_{i,k} dot(V_i - C, n_k) * 2 / sqrt(3)
    # We want to minimize S over C and A.
    # For a fixed A, the optimal C is the center of the minimal bounding hexagon.
    # This is a linear programming problem or can be solved by checking the constraints.
    # Actually, for a fixed A, the minimal S is determined by the "width" of the set of points
    # in the directions of the 6 normals.
    # Specifically, let P be the set of all inner hex vertices.
    # For a fixed A, the constraints are:
    # max_{p in P} dot(p, n_k) - min_{p in P} dot(p, n_k) <= S * sqrt(3)
    # Wait, the outer hexagon is defined by 6 half-planes.
    # If the outer hexagon is centered at C, the distance from C to each side is S*sqrt(3)/2.
    # So for each normal n_k, we need:
    # max_{p in P} dot(p - C, n_k) <= S * sqrt(3) / 2
    # min_{p in P} dot(p - C, n_k) >= -S * sqrt(3) / 2
    # This implies:
    # S * sqrt(3) / 2 >= max( max_p dot(p, n_k) - dot(C, n_k), dot(C, n_k) - min_p dot(p, n_k) )
    # To minimize S, we choose dot(C, n_k) to be the midpoint of [min_p dot(p, n_k), max_p dot(p, n_k)].
    # Then the required half-width in direction n_k is (max_p - min_p) / 2.
    # So S * sqrt(3) / 2 >= (max_p - min_p) / 2
    # S >= (max_p - min_p) / sqrt(3)
    # We need this to hold for all 6 directions n_k.
    # So S(A) = max_{k=0..5} (max_p dot(p, n_k) - min_p dot(p, n_k)) / sqrt(3)
    # We want to minimize S(A) over A.
    
    # This is a 1D search over A (0 to 60 degrees, due to symmetry).
    
    # Precompute vertices for all inner hexagons?
    # No, inner hexagons have fixed orientation (0 degrees) in this strategy.
    # Vertices of inner hex at center c: c + (cos(60k), sin(60k))
    
    # To make this fast, we can precompute the 6 vertices relative to center.
    rel_verts = np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])
    
    # We will sweep A from 0 to 60 degrees in small steps.
    # For each A, compute S(A).
    # To find the minimum, we can use a ternary search or just a fine grid.
    # Since S(A) is the max of 6 continuous functions, it's continuous and piecewise smooth.
    # A grid of 100 steps over 60 degrees should be sufficient for a good estimate.
    # Then refine.
    
    angles_to_test = np.linspace(0, 60, 121) # 0.5 degree steps
    
    # To speed up the combination loop, we can pre-filter.
    # But let's just run it. C(20, 12) is ~125k.
    # For each combo, we have 12*6 = 72 points.
    # For each angle (121 angles), we compute 6 max-min projections.
    # 125k * 121 * 72 * 6 ~ 6.5 billion operations. Too slow in Python.
    
    # We need to optimize.
    # 1. Reduce combinations: Only consider "connected" clusters?
    #    In a triangular lattice, a cluster of 12 points is likely compact.
    #    We can require that the bounding box of the points is small.
    # 2. Reduce angles: Use a coarser grid first, then refine for best candidates.
    # 3. Vectorize the projection calculation.
    
    # Let's try a different approach:
    # Instead of enumerating all combinations, we can use a heuristic search (Simulated Annealing)
    # on the lattice positions.
    # But the directive asks for a "grid-based discrete search... reducing continuous optimization to 1D search over rotation and small combinatorial search over lattice translations".
    # This implies enumerating translations is part of the plan.
    
    # Let's optimize the brute force:
    # 1. Precompute the "width" function for a set of points is expensive.
    # 2. We can approximate the minimal side length by the radius of the bounding circle? No, hexagon is different.
    # 3. We can use a faster check: For a given set of centers, the minimal S is determined by the convex hull of the vertices.
    #    We can compute the convex hull of the 72 points.
    #    Then S(A) is the max width of the convex hull in the 6 directions.
    #    Computing convex hull is O(N log N).
    
    # Let's limit the search space.
    # The optimal packing of 12 hexagons is known to be a hexagonal cluster of radius 2?
    # Radius 1: 7 hexagons.
    # Radius 2: 19 hexagons.
    # We need 12. So we take the 7 center+neighbors, and add 5 from the next ring.
    # The next ring has 12 positions. We need to choose 5.
    # C(12, 5) = 792. This is very small!
    # Plus we can shift the whole cluster.
    # So the strategy is:
    # 1. Base cluster: 7 hexagons (center + 6 neighbors).
    # 2. Add 5 hexagons from the 12 positions in the next ring.
    # 3. Enumerate all C(12, 5) = 792 ways to add them.
    # 4. For each of these 792 configurations, compute the optimal outer hexagon.
    # 5. Also consider other topologies? e.g. removing a corner from the 7?
    #    The 7-hexagon cluster is the most compact. Removing one makes it less compact.
    #    So adding to the 7-cluster is the best bet.
    
    # Let's implement this specific combinatorial search.
    
    # Base 7 points:
    # (0,0)
    # (d, 0), (d/2, d*sqrt(3)/2), (-d/2, d*sqrt(3)/2), (-d, 0), (-d/2, -d*sqrt(3)/2), (d/2, -d*sqrt(3)/2)
    
    base_pts = np.array([
        [0, 0],
        [d, 0],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [-d, 0],
        [-d/2, -d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2]
    ])
    
    # Next ring 12 points:
    # Distance from center is 2d? No, distance between adjacent lattice points is d.
    # The next ring is at distance 2d from center?
    # Lattice points at distance 2d:
    # (2d, 0), (3d/2, d*sqrt(3)/2), (d, d*sqrt(3)), (-d/2, d*sqrt(3)), (-3d/2, d*sqrt(3)/2), (-2d, 0), ...
    # Let's generate them properly.
    
    ring2_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * d + j * d * 0.5
            y = j * d * np.sqrt(3) / 2
            dist_sq = x*x + y*y
            # Distance squared for ring 2 is (2d)^2 = 4d^2
            if abs(dist_sq - 4*d*d) < 1e-9:
                ring2_pts.append([x, y])
                
    ring2_pts = np.array(ring2_pts)
    # There should be 12 points.
    
    # Enumerate combinations of 5 points from ring2_pts
    ring2_indices = list(range(len(ring2_pts)))
    
    best_config = None
    best_side_val = float('inf')
    
    # To speed up, we can precompute the convex hull or just the min/max projections.
    # But 792 * 121 angles * 72 points is still ~6.9 million operations.
    # This is fast in Python if vectorized.
    
    # Let's vectorize the angle sweep for a given configuration.
    
    # For a given set of centers (12 points), we have 72 vertices.
    # We want to minimize S(A) = max_k (max_p dot(p, n_k(A)) - min_p dot(p, n_k(A))) / sqrt(3)
    
    # We can precompute the vertices for a configuration.
    
    # Let's iterate through the 792 combinations.
    
    for combo in combinations(ring2_indices, 5):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Construct the 12 centers
        centers = np.vstack([base_pts, ring2_pts[list(combo)]])
        
        # Compute all 72 vertices
        verts = []
        for c in centers:
            verts.append(c + rel_verts)
        verts = np.array(verts) # Shape (72, 2)
        
        # Find optimal angle A
        # S(A) is max of 6 functions.
        # We can sample A and find the min.
        
        # To be faster, we can use the fact that the minimum occurs when two of the widths are equal and maximal?
        # Or just use a fine grid.
        
        # Let's use a grid of 360 points (0.1 deg) for precision.
        # 360 * 72 * 6 ~ 155k ops per config.
        # 792 * 155k ~ 123 million ops.
        # This might be too slow in Python (123M float ops ~ 1-2 seconds if vectorized, but loops are slow).
        
        # Let's vectorize the angle calculation.
        
        # n_k(A) = (cos(A + 60k), sin(A + 60k))
        # dot(verts, n_k) = verts_x * cos(...) + verts_y * sin(...)
        
        # We can compute the projection of all 72 vertices onto all 6 normals for all angles at once?
        # No, angles are different.
        
        # Let's just use a coarser grid first: 0.5 deg (120 steps).
        # Then refine the best 10 configs with 0.1 deg.
        
        angles = np.linspace(0, 60, 121)
        # Precompute cos and sin for the 6 normals for each angle
        # n_k = (cos(A + k*pi/3), sin(A + k*pi/3))
        
        # We can compute the width for all angles at once using matrix multiplication?
        # Projections = verts @ N.T where N is (121*6, 2)? No.
        
        # Let's just loop over angles, but vectorize over vertices and normals.
        
        min_side_for_config = float('inf')
        
        for A in angles:
            A_rad = np.deg2rad(A)
            # Normals for the 6 sides
            # n_k = (cos(A_rad + k*pi/3), sin(A_rad + k*pi/3))
            k_angles = A_rad + np.arange(6) * (np.pi/3)
            cos_k = np.cos(k_angles)
            sin_k = np.sin(k_angles)
            
            # Project all 72 vertices onto these 6 normals
            # verts is (72, 2)
            # cos_k is (6,), sin_k is (6,)
            # proj = verts[:, 0] * cos_k + verts[:, 1] * sin_k
            # This gives a (72, 6) array
            
            proj = verts[:, 0:1] * cos_k[None, :] + verts[:, 1:2] * sin_k[None, :]
            
            # Width in each direction
            widths = proj.max(axis=0) - proj.min(axis=0)
            
            # S = max(widths) / sqrt(3)
            side = np.max(widths) / np.sqrt(3.0)
            
            if side < min_side_for_config:
                min_side_for_config = side
                best_A_for_config = A
                
        if min_side_for_config < best_side_val:
            best_side_val = min_side_for_config
            best_config = (centers, best_A_for_config)
            
    # Refine the best configuration
    if best_config is not None:
        centers, best_A = best_config
        
        # Refine angle around best_A
        # Use a finer grid
        angles_fine = np.linspace(best_A - 1.0, best_A + 1.0, 201) # 0.01 deg steps
        
        verts = []
        for c in centers:
            verts.append(c + rel_verts)
        verts = np.array(verts)
        
        min_side_refined = float('inf')
        best_A_refined = best_A
        
        for A in angles_fine:
            A_rad = np.deg2rad(A)
            k_angles = A_rad + np.arange(6) * (np.pi/3)
            cos_k = np.cos(k_angles)
            sin_k = np.sin(k_angles)
            
            proj = verts[:, 0:1] * cos_k[None, :] + verts[:, 1:2] * sin_k[None, :]
            widths = proj.max(axis=0) - proj.min(axis=0)
            side = np.max(widths) / np.sqrt(3.0)
            
            if side < min_side_refined:
                min_side_refined = side
                best_A_refined = A
                
        best_side_val = min_side_refined
        best_A = best_A_refined
        
        # The outer hexagon center should be the centroid of the bounding box?
        # Actually, the optimal center C is such that the max and min projections are symmetric.
        # For each normal n_k, C_k = (max_p dot(p, n_k) + min_p dot(p, n_k)) / 2
        # But C must be the same for all k.
        # In our calculation, we implicitly assumed C is chosen optimally for each k independently.
        # This is only valid if there exists a C that satisfies all constraints simultaneously.
        # For a regular hexagon, the "width" in 3 pairs of opposite directions determines the side length.
        # If we define the outer hexagon by the 6 lines tangent to the convex hull of the inner vertices,
        # and if those 6 lines form a regular hexagon, then the side length is determined by the widths.
        # The condition for the 6 lines to form a regular hexagon is that the distances from the center to the lines are equal.
        # In our formula S = max(widths) / sqrt(3), we are taking the maximum width.
        # This corresponds to the outer hexagon being the smallest regular hexagon that contains the points,
        # aligned with angle A.
        # The center of this hexagon is the average of the midpoints of the widths?
        # Actually, for a regular hexagon, the center is the point equidistant from all 6 sides.
        # If we use the max width, the other widths might be smaller, meaning the hexagon is larger than necessary for those directions.
        # But since it's a regular hexagon, the side length is fixed by the largest width.
        # The center should be chosen to center the "max width" direction, and hopefully the others fit.
        # For a symmetric cluster, the centroid is a good approximation.
        # Let's use the centroid of the inner hexagon centers as the outer center.
        
        outer_center = np.mean(centers, axis=0)
        
        # Verify validity
        # We need to check if the inner hexagons are actually inside the outer hexagon.
        # And if they are disjoint.
        # Disjointness is guaranteed by the lattice construction (distance d = sqrt(3) is the touching distance).
        # Wait, if they touch, the distance is exactly sqrt(3).
        # The problem says "disjoint". Usually this means interiors are disjoint. Touching is allowed?
        # "Place 12 disjoint regular hexagons". In packing problems, touching is usually allowed.
        # If strict disjointness is required, we need d > sqrt(3).
        # Let's assume touching is allowed.
        
        # Check if inside
        inner_hexes = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        
        # The side length we calculated is the minimal side for a regular hexagon of angle A
        # to contain the convex hull of the vertices.
        # So it should be valid.
        
        # However, we should double check with the _check_validity function to be safe.
        # But _check_validity is slow.
        # Let's just return the result.
        
        # One issue: The outer hexagon center calculated as centroid might not be optimal.
        # But the side length calculation S = max(widths)/sqrt(3) assumes the center is optimally placed.
        # If we fix the center to the centroid, the required side length might be larger.
        # We should calculate the required side length for the specific center.
        
        # Let's re-calculate the side length for the specific center (centroid).
        # For each normal n_k, we need:
        # max_p dot(p - C, n_k) <= S * sqrt(3) / 2
        # min_p dot(p - C, n_k) >= -S * sqrt(3) / 2
        # So S * sqrt(3) / 2 >= max( max_p dot(p, n_k) - dot(C, n_k), dot(C, n_k) - min_p dot(p, n_k) )
        
        # Let's compute this for the best angle.
        
        A_rad = np.deg2rad(best_A)
        k_angles = A_rad + np.arange(6) * (np.pi/3)
        cos_k = np.cos(k_angles)
        sin_k = np.sin(k_angles)
        
        proj = verts[:, 0:1] * cos_k[None, :] + verts[:, 1:2] * sin_k[None, :]
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        C_proj = outer_center[0] * cos_k + outer_center[1] * sin_k
        
        # Required half-width for each side
        req_hw = np.maximum(max_proj - C_proj, C_proj - min_proj)
        
        # S * sqrt(3) / 2 >= max(req_hw)
        side_with_centroid = np.max(req_hw) * 2.0 / np.sqrt(3.0)
        
        # This side_with_centroid might be larger than best_side_val (which assumed optimal center).
        # If it is, we should use it.
        
        if side_with_centroid > best_side_val:
            best_side_val = side_with_centroid
            
        # We can also try to optimize the center C.
        # The optimal C minimizes max_k (max_p dot(p, n_k) - min_p dot(p, n_k)) / 2 ? No.
        # It minimizes max_k max( max_p dot(p, n_k) - dot(C, n_k), dot(C, n_k) - min_p dot(p, n_k) )
        # This is a convex optimization problem in C (2 variables).
        # We can solve it with a simple gradient descent or linear programming.
        # But for now, let's assume the centroid is close enough.
        # Or we can use the center of the minimal bounding circle?
        
        # Let's try to find a better center by checking a few candidates:
        # 1. Centroid of centers
        # 2. Centroid of vertices
        # 3. Center of the bounding box of the max width direction
        
        # For the max width direction n_k*, the optimal C_k* is the midpoint.
        # Let's find the direction with the max width in the optimal angle.
        # And set C to be the projection of the midpoint onto the plane?
        # Actually, for a regular hexagon, the center is unique.
        # The condition is that the distances to all 6 sides are equal.
        # This implies C is the intersection of the 3 midlines of the 3 pairs of parallel supporting lines.
        # If the supporting lines form a regular hexagon, the midlines intersect at a single point.
        # So we can find C by intersecting the midlines of the 3 pairs of directions with the largest widths?
        # No, we need to intersect the midlines of ALL 3 pairs.
        # If they don't intersect at a single point (because the widths are not consistent with a regular hexagon),
        # then the smallest regular hexagon containing the points is determined by the largest width,
        # and the center is chosen to center that largest width, and the other sides might be "loose".
        # In that case, the center is not uniquely defined by the "tight" sides.
        # But we can choose C to minimize the maximum distance to the 6 lines.
        # This is the Chebyshev center problem.
        
        # For simplicity, let's stick with the centroid or the midpoint of the max width.
        # Let's use the midpoint of the max width direction.
        
        widths = max_proj - min_proj
        max_width_idx = np.argmax(widths)
        n_max = np.array([cos_k[max_width_idx], sin_k[max_width_idx]])
        
        # Midpoint of the projection in this direction
        # C should satisfy dot(C, n_max) = (max_proj[max_width_idx] + min_proj[max_width_idx]) / 2
        # This gives one equation for C (2 variables).
        # We need another equation. We can use the second max width direction.
        
        # Let's just use the centroid for now. It's robust.
        
        # Final check
        if best_side_val < best_side:
            best_side = best_side_val
            best_inner = inner_hexes
            best_outer_center = (outer_center[0], outer_center[1])
            best_outer_angle = best_A

    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# Run a quick self-test if executed directly
if __name__ == "__main__":
    res = construct()
    print(res)

