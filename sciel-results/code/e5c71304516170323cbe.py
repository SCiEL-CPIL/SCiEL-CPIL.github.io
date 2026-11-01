import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate candidate lattice configurations
    # A "lattice configuration" is a set of (i, j) coordinates in a hexagonal grid
    # that contains N cells.
    # We generate sets by taking a central cluster and adding neighbors, 
    # or by taking rectangular patches in the axial coordinate system.
    
    candidates = set()
    
    # Strategy A: Rectangular patches in axial coords (i, j)
    # Axial coords: (q, r). Neighbors are (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
    # We look for sets of 12 points that form a "convex" or compact shape.
    
    # Let's enumerate small bounding boxes in axial coords and check subsets? 
    # No, simpler: generate "polyhexes" of size 12? Too many (4282 free polyhexes of order 12 is high, but fixed are higher).
    # Instead, let's use a heuristic: 
    # 1. Start with a 3x4 rectangle in axial coords? 
    # 2. Start with a hexagon of radius 2 (19 cells) and remove 7?
    
    # Better approach: 
    # Generate all subsets of a 4x4 axial grid that have size 12 and are "connected" and "compact"?
    # This might be too slow if done naively.
    
    # Let's use a specific set of known good "blocks" for hex packing.
    # Common dense packings of 12 hexagons:
    # 1. 3x4 parallelogram (area 12 in axial sense? No, 3x4 axial rectangle has 12 cells)
    #    Axial rectangle: q in [0,2], r in [0,3] -> 3*4 = 12 cells.
    # 2. Hexagon of radius 1 (7 cells) + 5 neighbors?
    
    # Let's explicitly construct a few topologies:
    
    # Topology 1: 3x4 Axial Rectangle
    # q: 0, 1, 2; r: 0, 1, 2, 3
    t1 = [(q, r) for q in range(3) for r in range(4)]
    candidates.add(frozenset(t1))
    
    # Topology 2: 4x3 Axial Rectangle
    t2 = [(q, r) for q in range(4) for r in range(3)]
    candidates.add(frozenset(t2))
    
    # Topology 3: "Hexagon-like" cluster
    # Center (0,0). Neighbors.
    # Radius 1 hex has 7 cells: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Add 5 more to make 12.
    # Let's just take the 12 cells closest to origin in a 5x5 axial box, centered?
    box = [(q, r) for q in range(-2, 3) for r in range(-2, 3)]
    # Sort by distance from center (0,0) in axial metric? 
    # Axial distance: (|q| + |r| + |q+r|) / 2
    def axial_dist(q, r):
        return (abs(q) + abs(r) + abs(q+r)) / 2.0
    
    box_sorted = sorted(box, key=lambda p: axial_dist(p[0], p[1]))
    # Take first 12? This gives a hexagon of radius 1 (7 cells) + next ring.
    # The next ring has 12 cells. We need 5 more.
    # This creates an irregular shape.
    t3 = frozenset(box_sorted[:12])
    candidates.add(t3)
    
    # Topology 4: Two 3x2 blocks?
    # Let's stick to the 3x4 and 4x3 rectangles and the "radius 1.5" blob.
    
    # Topology 5: 2x6 rectangle
    t5 = [(q, r) for q in range(2) for r in range(6)]
    candidates.add(frozenset(t5))
    
    # Topology 6: 6x2 rectangle
    t6 = [(q, r) for q in range(6) for r in range(2)]
    candidates.add(frozenset(t6))
    
    # Topology 7: 1x12 line (bad, but include for completeness)
    t7 = [(q, 0) for q in range(12)]
    candidates.add(frozenset(t7))
    
    # Topology 8: "L" shape or other? 
    # Let's add a "thick" hexagon. 
    # Cells with axial_dist <= 1 (7 cells) + 5 cells from dist=2 ring that are "compact".
    # Just take the first 5 from the sorted list of dist=2 cells?
    dist2_cells = [p for p in box if axial_dist(p[0], p[1]) == 2]
    # Sort dist2 by angle or something to keep it compact? 
    # Let's just pick the ones with smallest q, then r?
    dist2_sorted = sorted(dist2_cells, key=lambda p: (p[0], p[1]))
    t8 = frozenset(box_sorted[:7] + dist2_sorted[:5])
    candidates.add(t8)
    
    # Convert candidates to list of lists for processing
    lattice_configs = [list(c) for c in candidates]
    
    best_side = float('inf')
    best_config = None
    best_params = None
    
    # Precompute hexagon vertices for unit side length
    # Vertices of a regular hexagon with side 1, centered at origin, angle 0 (flat top? pointy top?)
    # Standard math: vertices at angles 0, 60, 120...
    # If angle 0, vertex at (1, 0).
    # Let's use vertices relative to center.
    hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    
    def get_hex_verts(center, angle_deg, scale=1.0):
        """
        Returns vertices of a hexagon with given center, rotation, and scale.
        """
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        verts = (hex_verts * scale) @ R.T
        verts += center
        return verts
    
    def point_in_hexagon(p, center, angle_deg, side_len):
        """
        Check if point p is inside the hexagon defined by center, angle, side_len.
        Uses distance to edges.
        """
        # Transform p to hexagon's local frame
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        theta = math.radians(-angle_deg) # Rotate back
        c, s = math.cos(theta), math.sin(theta)
        x = c * dx - s * dy
        y = s * dx + c * dy
        
        # In local frame, hexagon is axis-aligned (pointy top or flat top depending on base def)
        # Our base hex_verts has vertex at (1,0) which is "pointy right".
        # The supporting lines are at angles 30, 90, 150, 210, 270, 330.
        # Distance from center to each side is apothem = side_len * sqrt(3)/2.
        # Wait, the normal vectors for the sides of a hexagon with vertex at (1,0) are at 30, 90, ...
        # Let's check.
        # Side between (1,0) and (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4). Normal is (sqrt(3)/2, 0.5) -> angle 30.
        # So yes, normals are at 30, 90, 150...
        
        normals = []
        for a in range(30, 360, 60):
            rad = math.radians(a)
            normals.append([math.cos(rad), math.sin(rad)])
        normals = np.array(normals)
        
        # Project local point onto normals
        projections = locals_ = np.dot(normals, [x, y])
        apothem = side_len * math.sqrt(3) / 2.0
        # If any projection > apothem, it's outside
        if np.any(projections > apothem):
            return False
        return True

    # We need to find the minimum side length S of the OUTER hexagon
    # such that all inner hexagons (side 1) are inside.
    # This is equivalent to finding the smallest S such that the union of inner hexagons
    # fits in a hexagon of side S.
    
    # For a fixed lattice configuration (set of relative centers) and global rotation theta:
    # The centers of the inner hexagons are fixed relative to each other.
    # Let C_i be the center of inner hex i in the global frame (after rotation).
    # Each inner hexagon occupies a region.
    # The condition "inner hex i is inside outer hex" is:
    # For all vertices v of inner hex i, v is inside outer hex.
    # Since hexagons are convex, it's sufficient to check vertices.
    
    # However, we can simplify. The outer hexagon is defined by 6 half-planes.
    # H_k: n_k . x <= d_k, where n_k are normals of outer hex, d_k is distance from center to side (apothem_outer).
    # Apothem_outer = S * sqrt(3) / 2.
    # So we need max_{i, v} (n_k . v) <= S * sqrt(3) / 2 for all k.
    # Thus S = (2/sqrt(3)) * max_{k} (max_{i, v} (n_k . v)).
    
    # We can optimize S for a given theta.
    
    # Let's iterate over theta.
    # The lattice geometry has 6-fold symmetry? No, the lattice itself might not, but the container does.
    # We can search theta in [0, 60) degrees.
    
    # To speed up, we can calculate the "width" of the cluster in the 6 normal directions.
    
    # Precompute relative centers for each lattice config
    # Axial to Cartesian:
    # x = q * 1.5
    # y = r * sqrt(3) + q * sqrt(3)/2
    # This assumes unit side length hexagons in a honeycomb.
    # Distance between centers of adjacent hexagons in honeycomb is sqrt(3).
    # Let's verify.
    # Hex side 1. Apothem sqrt(3)/2.
    # Center to center across a shared edge = 2 * apothem = sqrt(3).
    # In axial coords:
    # (0,0) to (1,0): dx = 1.5, dy = sqrt(3)/2. dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    # (0,0) to (0,1): dx = 0, dy = sqrt(3). dist = sqrt(3). Correct.
    
    for config in lattice_configs:
        if time.time() > deadline:
            break
            
        # Calculate relative centers
        centers = []
        for q, r in config:
            x = 1.5 * q
            y = math.sqrt(3) * r + math.sqrt(3)/2 * q
            centers.append([x, y])
        centers = np.array(centers)
        
        # Center the configuration at origin to minimize bounding box?
        # The problem allows translating the whole cluster.
        # So we can shift centers so that the "center of mass" or "centroid" is at origin.
        # This minimizes the maximum distance from origin for symmetric shapes.
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        # Now search for optimal rotation theta
        # We want to minimize S(theta) = (2/sqrt(3)) * max_k ( max_{i} ( max_{v in hex_i} n_k . (R(theta) c_i + v_i) ) )
        # Note: v_i are vertices of unit hexagon.
        # max_{v in hex_i} n_k . v is the support function of the unit hexagon in direction n_k.
        # Let h(n_k) be the max dot product of unit hex vertices with n_k.
        # Then max_{v in hex_i} n_k . (R c_i + v_i) = n_k . R c_i + h(n_k - R c_i)? No.
        # v_i is a vertex of the hexagon centered at R c_i.
        # The set of vertices is R c_i + V, where V is the set of vertices of unit hex at origin.
        # Max over v in R c_i + V of n_k . v = n_k . R c_i + max_{v' in V} n_k . v'
        # Let h(n_k) = max_{v' in V} n_k . v'. This is the support function of the unit hexagon.
        # So for a given theta, the constraint for normal k is:
        # max_i ( n_k . R(theta) c_i + h(n_k) ) <= S * sqrt(3) / 2
        # S(theta) = (2/sqrt(3)) * max_k ( h(n_k) + max_i ( n_k . R(theta) c_i ) )
        
        # h(n) for unit hexagon:
        # If n is a normal to a side, h(n) = apothem = sqrt(3)/2.
        # If n is along a vertex, h(n) = 1.
        # Generally, h(n) is piecewise linear.
        # But since we are checking specific n_k (normals of the OUTER hexagon),
        # and the outer hexagon can also rotate? 
        # The problem says "outer hexagon center, side length and rotation".
        # So we can rotate the OUTER hexagon too!
        # This means we can align the outer hexagon's normals with the "best" directions for the cluster.
        # Effectively, we can rotate the cluster relative to the outer hexagon freely.
        # So we just need to find the minimum S such that the cluster fits in a hexagon of side S.
        # This is the "minimum enclosing hexagon" problem for the set of points (centers) expanded by the unit hexagon.
        
        # Since we can rotate both, it's equivalent to fixing the outer hexagon (normals at 30, 90...) 
        # and rotating the cluster.
        
        # So for a fixed cluster (centers), we search theta in [0, 60).
        # S(theta) = (2/sqrt(3)) * max_{k=0..5} ( h(n_k) + max_i ( n_k . R(theta) c_i ) )
        
        # h(n_k) is constant for fixed outer orientation.
        # Let's calculate h(n_k) for n_k = [cos(30+60k), sin(30+60k)]
        # n_0 is at 30 deg.
        # Unit hex vertices are at 0, 60, ...
        # Dot product of n_0 (30 deg) with vertices:
        # v_0 (0 deg): cos(30) = sqrt(3)/2
        # v_1 (60 deg): cos(30) = sqrt(3)/2
        # v_2 (120 deg): cos(90) = 0
        # ...
        # Max is sqrt(3)/2.
        # So h(n_k) = sqrt(3)/2 for all k if outer normals are at 30, 90...
        # Wait, if outer hex is "flat top" (normals at 0, 60...), then h(n) would be different?
        # The shape of the outer hexagon is regular. Its orientation only matters relative to the cluster.
        # So yes, we can fix outer normals and rotate cluster.
        
        h_val = math.sqrt(3) / 2.0
        
        # Define function to compute S for a given theta
        def calc_S(theta_rad):
            c, s = math.cos(theta_rad), math.sin(theta_rad)
            R = np.array([[c, -s], [s, c]])
            rotated_centers = centers @ R.T
            
            max_proj = 0.0
            for k in range(6):
                angle_deg = 30 + 60 * k
                rad = math.radians(angle_deg)
                n = np.array([math.cos(rad), math.sin(rad)])
                # max_i n . rotated_c_i
                projs = rotated_centers @ n
                max_i = np.max(projs)
                total = max_i + h_val
                if total > max_proj:
                    max_proj = total
            return max_proj * 2.0 / math.sqrt(3)
        
        # Optimize theta. 
        # S(theta) is a sum of max of sinusoids, so it's smooth-ish but non-differentiable at intersections.
        # We can do a grid search + local refinement.
        
        # Grid search
        best_local_S = float('inf')
        best_local_theta = 0
        
        # 360 points in 60 degrees
        thetas = np.linspace(0, math.pi/3, 360, endpoint=False)
        for theta in thetas:
            S = calc_S(theta)
            if S < best_local_S:
                best_local_S = S
                best_local_theta = theta
        
        # Refine around best
        # Golden section or just fine grid
        start = best_local_theta - math.pi/360
        end = best_local_theta + math.pi/360
        fine_thetas = np.linspace(start, end, 100)
        for theta in fine_thetas:
            S = calc_S(theta)
            if S < best_local_S:
                best_local_S = S
                best_local_theta = theta
        
        if best_local_S < best_side:
            best_side = best_local_S
            best_config = config
            best_params = (best_local_theta, centroid)
            
    # Construct the final answer
    # We need to return:
    # inner: list of [x, y, angle_deg] for each hex
    # center: [x, y] of outer hex
    # side: outer side length
    # angle: outer rotation
    
    if best_config is None:
        # Fallback
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 40.0, 0.0
        
    theta, centroid = best_params
    
    # Generate inner hexagons
    inner = []
    for q, r in best_config:
        x = 1.5 * q
        y = math.sqrt(3) * r + math.sqrt(3)/2 * q
        # Rotate and translate
        c, s = math.cos(theta), math.sin(theta)
        rx = c * x - s * y
        ry = s * x + c * y
        # Note: The centroid was subtracted before rotation in the optimization.
        # So the final position is (rx, ry) relative to the outer center.
        # We can place the outer center at (0,0).
        inner.append([rx, ry, math.degrees(theta)])
        
    # Outer hexagon
    # We fixed outer normals at 30, 90...
    # This corresponds to a hexagon with vertices at 0, 60... ?
    # If normals are at 30, the sides are perpendicular to 30, so sides are at 120/300?
    # A regular hexagon with flat top has vertices at 0, 60... and normals at 30, 90...
    # So angle 0.0 is fine for the outer hexagon rotation.
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
