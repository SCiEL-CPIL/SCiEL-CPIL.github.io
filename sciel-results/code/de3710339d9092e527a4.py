"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s limit
    
    # Flat-top hexagon vertices for side length 1, centered at origin
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Axial coordinate to Cartesian conversion for flat-top hexagons
    # Axial (q, r) -> Cartesian (x, y)
    # x = size * (q + r/2)
    # y = size * (sqrt(3) * r/2)
    # Here size=1.
    def axial_to_cart(q, r):
        x = q + 0.5 * r
        y = (np.sqrt(3) / 2.0) * r
        return np.array([x, y])

    # Pre-compute a set of candidate sets of 11 hexagons on the axial grid.
    # We want compact clusters.
    # A 3x3 hexagon block (radius 1) has 1 + 6 = 7 hexagons.
    # A 4x4 hexagon block (radius 2) has 1 + 6 + 12 = 19 hexagons.
    # We need 11.
    
    # Let's generate all subsets of 11 hexagons from a central region, 
    # but that's too many. Instead, let's look for known efficient packings or 
    # simply try to fill a "stadium" or "ellipse" shape on the grid.
    
    # A simpler heuristic: Try all 11-subsets of a 5x5 axial grid centered at 0.
    # But 5x5 grid has ~25 points. C(25,11) is huge.
    
    # Better: Use a greedy construction or specific patterns.
    # Pattern 1: A "lozenge" or rhombus of 11?
    # Pattern 2: A central hexagon, surrounded by 6, and 4 more attached.
    
    # Let's define a function to calculate the minimal outer side length S 
    # for a given set of centers, assuming outer hex is centered at (0,0) and angle 0.
    # The outer hexagon contains the inner hexagons if every vertex of every inner hexagon
    # is inside the outer hexagon.
    
    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + (side * base_vertices @ R.T)

    def min_outer_side_for_centers(centers, outer_angle_deg=0.0):
        """
        Calculate the minimum side length S of a regular hexagon centered at origin,
        rotated by outer_angle_deg, that contains all hexagons at 'centers' 
        (each side 1, angle 0).
        """
        # The condition for a point p to be inside a hexagon of side S, center 0, angle 0 is:
        # max(|p_x|, |0.5*p_x + sqrt(3)/2*p_y|, |-0.5*p_x + sqrt(3)/2*p_y|) <= S * (1 if axis aligned? No)
        
        # Let's use the support function / projection approach.
        # A convex polygon P is contained in a scaled convex polygon Q (centered at 0) 
        # if and only if the support function of P is <= support function of Q in all directions.
        # However, checking vertices is sufficient for containment of convex sets if the containing set is convex?
        # Yes, if all vertices of the inner set are inside the outer set, and the outer set is convex, 
        # then the entire inner set is inside.
        
        # So we just need to find the smallest S such that all vertices of all inner hexagons
        # are inside the outer hexagon of side S.
        
        # Vertices of outer hexagon (side S, angle 0):
        # The boundary is defined by 6 lines.
        # Line 1: x = S
        # Line 2: 0.5x + sqrt(3)/2 y = S * sqrt(3)/2 ? No.
        # Let's derive the half-plane inequalities for a flat-top hexagon of side S.
        # Vertices: (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S,0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # Inequalities:
        # 1. x <= S
        # 2. 0.5x + sqrt(3)/2 y <= S*sqrt(3)/2  => x + sqrt(3)y <= S*sqrt(3)
        # 3. -0.5x + sqrt(3)/2 y <= S*sqrt(3)/2 => -x + sqrt(3)y <= S*sqrt(3)
        # 4. x >= -S
        # 5. -0.5x - sqrt(3)/2 y <= S*sqrt(3)/2 => x - sqrt(3)y <= S*sqrt(3)
        # 6. 0.5x - sqrt(3)/2 y <= S*sqrt(3)/2 => -x - sqrt(3)y <= S*sqrt(3)
        
        # Wait, let's verify the normals.
        # Normal to side connecting (S,0) and (S/2, S*sqrt(3)/2) is (1, sqrt(3))/2?
        # Vector: (-S/2, S*sqrt(3)/2). Normal: (S*sqrt(3)/2, S/2) -> (sqrt(3), 1).
        # Equation: sqrt(3)x + y = S*sqrt(3) + 0 = S*sqrt(3).
        # Check (S,0): sqrt(3)S = S*sqrt(3). Correct.
        # Check (S/2, S*sqrt(3)/2): sqrt(3)S/2 + S*sqrt(3)/2 = S*sqrt(3). Correct.
        
        # So the 6 inequalities for containment (point (x,y) inside) are:
        # 1. x <= S
        # 2. -x <= S
        # 3. sqrt(3)x + y <= S*sqrt(3)
        # 4. -sqrt(3)x + y <= S*sqrt(3)
        # 5. sqrt(3)x - y <= S*sqrt(3)
        # 6. -sqrt(3)x - y <= S*sqrt(3)
        
        # This can be rewritten as:
        # S >= max( |x|, (sqrt(3)|x| + |y|) / sqrt(3) )? No.
        # From 3: S >= (sqrt(3)x + y) / sqrt(3) = x + y/sqrt(3)
        # From 4: S >= (-sqrt(3)x + y) / sqrt(3) = -x + y/sqrt(3)
        # From 5: S >= (sqrt(3)x - y) / sqrt(3) = x - y/sqrt(3)
        # From 6: S >= (-sqrt(3)x - y) / sqrt(3) = -x - y/sqrt(3)
        
        # So S >= max( |x|, |x + y/sqrt(3)|, |-x + y/sqrt(3)| )
        # Note that max(A, -A) = |A|.
        # Let u = x + y/sqrt(3) and v = -x + y/sqrt(3).
        # Then S >= max( |x|, |u|, |v| ).
        
        min_S = 0.0
        # We need to check all vertices of all inner hexagons.
        # To be efficient, we can pre-compute the max required S for a set of points.
        
        for center in centers:
            verts = get_hex_verts(center, 0.0)
            for pt in verts:
                x, y = pt
                req_S = max(abs(x), abs(x + y/np.sqrt(3)), abs(-x + y/np.sqrt(3)))
                if req_S > min_S:
                    min_S = req_S
        return min_S

    # Generate candidate clusters of 11 hexagons.
    # We will search over subsets of a small grid.
    # Grid points: (q, r) where q, r in [-2, 2].
    # Total points: 5x5 = 25.
    # We need to choose 11.
    # This is C(25, 11) which is ~4.4 million. Too many to brute force with geometry checks if slow.
    # But we can optimize:
    # 1. Filter by bounding box?
    # 2. Use a heuristic search?
    
    # Alternative: Known optimal packings for small N.
    # For N=11, a "flower" pattern (1+6) plus 4 more is likely good.
    # Let's try to generate a few specific patterns and evaluate them.
    
    patterns = []
    
    # Pattern 1: Central 7 (radius 1) + 4 attached to the "outside" of the ring.
    # The ring is at distance 1. The next layer is distance 2.
    # Let's just take the 11 closest points to the origin on the axial grid?
    # Distances in axial coords:
    # (0,0) d=0
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) d=1
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1) d=2
    
    # Let's collect all grid points in a 3x3x3 area (radius 2) and sort by distance.
    grid_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Axial distance is roughly max(|q|, |r|, |q+r|)?
            # Actually, axial distance is (|q| + |r| + |q+r|) / 2
            dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
            grid_points.append((dist, q, r))
    
    grid_points.sort()
    
    # Take the first 11 unique points?
    # This gives the most compact cluster centered at 0.
    # Let's verify if these 11 are disjoint.
    # On a hex grid, adjacent points (distance 1) are tangent.
    # Points at distance > 1 are separated.
    # So any set of grid points is a valid disjoint packing (interiors disjoint).
    
    # The "first 11" by distance will be:
    # 1 at d=0
    # 6 at d=1
    # 4 at d=2
    # This forms a compact shape.
    
    top_11 = grid_points[:11]
    centers_1 = [axial_to_cart(q, r) for _, q, r in top_11]
    patterns.append(centers_1)
    
    # Pattern 2: Maybe a slightly different arrangement of the 4 outer points?
    # The 12 points at d=2 form a ring. We pick 4.
    # Symmetry suggests picking 4 that are spaced out or clustered?
    # Clustered might reduce the max radius?
    # Let's try all combinations of 4 from the 12 points at d=2?
    # C(12, 4) = 495. This is very fast.
    
    ring_d2 = [ (q,r) for d, q, r in grid_points if abs(d-2.0) < 1e-9 ]
    
    best_S = float('inf')
    best_centers = None
    best_pattern_idx = 0
    
    # Check Pattern 1 (first 11)
    S1 = min_outer_side_for_centers(centers_1)
    if S1 < best_S:
        best_S = S1
        best_centers = centers_1
        best_pattern_idx = 1
        
    # Check combinations of 4 from ring_d2 added to the core 7
    core_7 = [axial_to_cart(q, r) for d, q, r in grid_points[:7]]
    
    if time.time() < deadline - 10:
        for combo in itertools.combinations(range(12), 4):
            if time.time() > deadline:
                break
            outer_4 = [axial_to_cart(ring_d2[i][0], ring_d2[i][1]) for i in combo]
            centers = core_7 + outer_4
            S = min_outer_side_for_centers(centers)
            if S < best_S:
                best_S = S
                best_centers = centers
                best_pattern_idx = 2

    # If we have time, we can also try rotating the entire cluster?
    # The problem allows inner rotation. But we fixed inner angle to 0.
    # Does rotating the cluster help?
    # The outer hexagon is fixed at angle 0.
    # If we rotate the inner cluster, the "bounding box" in the hex metric might change.
    # However, the hexagon has 6-fold symmetry.
    # Rotating by 60 degrees maps the grid to itself.
    # Rotating by 30 degrees might align vertices differently.
    # Let's test a few rotations of the best cluster found.
    
    def rotate_points(points, angle_deg):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return [R @ p for p in points]
        
    if best_centers is not None:
        # Try rotating the cluster by 0, 15, 30, 45, 60 degrees
        # Note: rotating the cluster changes the positions relative to the outer hex.
        # The inner hexagons themselves are NOT rotated, only their centers.
        # Wait, the problem says "each inner hexagon may translate and rotate".
        # If we rotate the *centers* but keep the *hexagons* at angle 0, 
        # the hexagons are still axis-aligned.
        # Is it beneficial to rotate the hexagons themselves?
        # Usually, for packing hexagons in a hexagon, aligning them is optimal.
        # Let's stick to axis-aligned inner hexagons.
        
        # But we can rotate the *set of centers*.
        for rot in [0, 15, 30, 45, 60, 75, 90]:
            rotated_centers = rotate_points(best_centers, rot)
            S_rot = min_outer_side_for_centers(rotated_centers)
            if S_rot < best_S:
                best_S = S_rot
                best_centers = rotated_centers

    # Final check: Ensure we have a valid result
    if best_centers is None:
        # Fallback: linear arrangement (should be worse)
        best_centers = [np.array([4.0 * (i - 5), 0.0]) for i in range(11)]
        best_S = 40.0 # Huge penalty

    # Construct the return value
    # inner: list of (x, y, angle_degrees)
    # center: (x, y) of outer hexagon (0,0)
    # side: best_S
    # angle: 0.0
    
    inner = []
    for c in best_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
