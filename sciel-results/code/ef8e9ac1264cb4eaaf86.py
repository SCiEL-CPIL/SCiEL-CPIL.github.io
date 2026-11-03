import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Hexagonal lattice: unit hexagons with side 1
    # Center-to-center distance for edge-adjacent: sqrt(3)
    # Axial coordinates (q, r) -> Cartesian:
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    
    sqrt3 = math.sqrt(3)
    
    # Generate compact clusters of 11 hexagons using axial coordinates
    # Start from center, add neighbors greedily for compactness
    
    # Precompute a set of candidate cluster shapes
    # A "hexagonal" number sequence: 1, 7, 19, 37...
    # 11 is between 7 and 19, so we take 7 (center + ring 1) + 4 from ring 2
    
    # Ring 1 (6 neighbors of origin):
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    # Ring 2: hexagons at distance 2 from center
    # In axial coords, ring 2 has 12 hexagons
    ring2 = []
    # Directions for ring 2
    dirs = [(2, 0), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1),
            (-1, 2), (1, 2), (2, 1), (1, 1), (0, 2), (0, -2)]
    # Actually let me generate properly
    # Ring k hexagons: walk around
    def generate_ring(k):
        if k == 0:
            return [(0, 0)]
        hexes = []
        # Start at (k, 0) and walk
        # The 6 directions in axial coords
        d = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        pos = [k, 0]
        for i in range(6):
            for j in range(k):
                hexes.append(tuple(pos))
                pos = [pos[0] + d[i][0], pos[1] + d[i][1]]
        return hexes
    
    ring2 = generate_ring(2)
    # ring2 should have 12 hexagons
    
    # All candidate positions for 11 hexagons: must include (0,0) and 6 from ring1, plus 4 from ring2
    # For compactness, we want the 4 ring-2 hexagons to be adjacent to the ring-1 ones
    
    # Strategy: try all combinations of 4 from ring2 (12 choose 4 = 495), 
    # plus all 6 from ring1 (fixed for now), compute enclosing hexagon
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    def hexagon_vertices(cx, cy, side, angle_deg=0.0):
        """Return 6 vertices of regular hexagon centered at (cx,cy) with given side and rotation."""
        vertices = []
        for i in range(6):
            theta = math.radians(angle_deg + 60 * i)
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            vertices.append((vx, vy))
        return vertices
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
        """Check if point (px,py) is inside regular hexagon."""
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        theta = math.radians(angle_deg)
        # Rotate point by -theta
        rx = dx * math.cos(-theta) - dy * math.sin(-theta)
        ry = dx * math.sin(-theta) + dy * math.cos(-theta)
        # Check against 6 half-planes
        for i in range(6):
            angle = math.radians(60 * i)
            nx = math.cos(angle)
            ny = math.sin(angle)
            # The hexagon has apothem = side * sqrt(3)/2
            apothem = side * sqrt3 / 2.0
            if rx * nx + ry * ny > apothem:
                return False
        return True
    
    def min_enclosing_hexagon_side(points, angle_deg=0.0):
        """
        Given a set of points, find the minimum side length of a regular hexagon 
        (with given orientation) that can enclose all points, centered optimally.
        
        For a fixed orientation, the minimum enclosing hexagon can be found by
        considering the support functions in the 6 normal directions.
        """
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # The 6 outward normals of the hexagon
        normals = []
        for i in range(6):
            a = math.radians(60 * i)
            normals.append((math.cos(a), math.sin(a)))
        
        # For each normal direction, compute the range of projections
        # The hexagon centered at (cx, cy) with side s has the property that
        # for each normal n_i, the half-width is s * sqrt(3)/2
        # So we need: max_i (max_p p·n_i - min_p p·n_i) / 2 <= s * sqrt(3)/2
        # => s >= max_i (max_p p·n_i - min_p p·n_i) / sqrt(3)
        
        max_range = 0.0
        for nx, ny in normals:
            projections = []
            for px, py in points:
                # Project onto normal (in world frame, but hexagon is rotated)
                # Actually, we need to project onto the hexagon's local normals
                # The hexagon's normal i in world frame is:
                #   rotate (cos(60i), sin(60i)) by theta
                wx = nx * cos_t - ny * sin_t
                wy = nx * sin_t + ny * cos_t
                proj = px * wx + py * wy
                projections.append(proj)
            pmin = min(projections)
            pmax = max(projections)
            range_val = pmax - pmin
            if range_val > max_range:
                max_range = range_val
        
        # s * sqrt(3) >= max_range  =>  s >= max_range / sqrt(3)
        # But wait, this gives the minimum side for a hexagon centered at the 
        # midpoint of the bounding box in each direction. The optimal center 
        # for a fixed orientation minimizes the max over directions of 
        # (half_range_i + |center_proj_i - mid_proj_i|)
        # For simplicity, assume center is at centroid, then:
        # s >= max_i half_range_i / (sqrt(3)/2) = max_i half_range_i * 2/sqrt(3)
        
        # Actually, the correct formula: for a regular hexagon with side s,
        # the distance from center to each edge (apothem) is s*sqrt(3)/2.
        # For the hexagon to contain all points, for each normal direction n_i:
        #   max_p (p - c)·n_i <= s*sqrt(3)/2
        #   min_p (p - c)·n_i >= -s*sqrt(3)/2
        # So: max_p (p·n_i) - c·n_i <= s*sqrt(3)/2
        #      c·n_i - min_p (p·n_i) <= s*sqrt(3)/2
        # Adding: max_p(p·n_i) - min_p(p·n_i) <= s*sqrt(3)
        # So s >= (max_p(p·n_i) - min_p(p·n_i)) / sqrt(3) for all i
        # The minimum s for fixed orientation and optimal center is:
        # s = max_i (range_i) / sqrt(3)
        # where range_i = max_p(p·n_i) - min_p(p·n_i)
        
        return max_range / sqrt3
    
    # Try different orientations of the container hexagon
    # For hexagonal lattice packing, orientation 0, 30, 60, 90, 120, 150 degrees might be relevant
    # But due to 6-fold symmetry, we only need 0 to 60 degrees
    
    best_side = float('inf')
    best_cluster = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Generate all candidate clusters: (0,0) + all 6 ring1 + 4 from ring2
    from itertools import combinations
    
    ring2_list = generate_ring(2)
    # Verify ring2 has 12 elements
    # print(len(ring2_list))  # should be 12
    
    # For each combination of 4 from ring2
    combo_count = 0
    for combo in combinations(range(len(ring2_list)), 4):
        if time.time() > deadline:
            break
            
        cluster_axial = [(0, 0)] + ring1 + [ring2_list[i] for i in combo]
        
        # Convert to Cartesian
        cluster_cart = [axial_to_cartesian(q, r) for q, r in cluster_axial]
        
        # Try a few container orientations
        for angle_deg in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline:
                    break
            side = min_enclosing_hexagon_side(cluster_cart, angle_deg)
            
            if side < best_side:
                best_side = side
                best_cluster = cluster_axial
                best_angle = angle_deg
                # Compute center as centroid
                cx = sum(p[0] for p in cluster_cart) / len(cluster_cart)
                cy = sum(p[1] for p in cluster_cart) / len(cluster_cart)
                best_center = (cx, cy)
        
        combo_count += 1
    
    if best_cluster is None:
        # Fallback: linear arrangement
        inner = [[2.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 22.0, 0.0
    
    # Build inner list: for each hexagon in cluster, give (x, y, angle_deg)
    # The inner hexagons are axis-aligned (angle 0) on the lattice
    inner = []
    for q, r in best_cluster:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    # The outer hexagon: center at best_center, side = best_side, rotation = best_angle
    # But we need to verify the center is correct. The min_enclosing_hexagon_side 
    # assumed the center is chosen optimally. Let's compute the actual optimal center.
    
    # For the optimal center, for each normal direction, the center projection
    # should be at the midpoint of the range
    # c·n_i = (max_p(p·n_i) + min_p(p·n_i)) / 2 for all i
    # This is generally over-determined, but for symmetric clusters it works
    
    # Let's compute the center as the average of the midpoints in each normal direction
    theta = math.radians(best_angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    cluster_cart = [axial_to_cartesian(q, r) for q, r in best_cluster]
    
    center_x = 0.0
    center_y = 0.0
    for i in range(6):
        a = math.radians(60 * i)
        nx = math.cos(a)
        ny = math.sin(a)
        wx = nx * cos_t - ny * sin_t
        wy = nx * sin_t + ny * cos_t
        projections = [px * wx + py * wy for px, py in cluster_cart]
        mid = (max(projections) + min(projections)) / 2.0
        # The center's projection onto this normal should be mid
        # c·(wx,wy) = mid
        # This gives one linear equation per direction. Average them.
        center_x += mid * wx
        center_y += mid * wy
    
    center_x /= 6.0
    center_y /= 6.0
    
    # Verify the side length with this center
    # Recompute to be safe
    final_side = 0.0
    for i in range(6):
        a = math.radians(60 * i)
        nx = math.cos(a)
        ny = math.sin(a)
        wx = nx * cos_t - ny * sin_t
        wy = nx * sin_t + ny * cos_t
        projections = [(px - center_x) * wx + (py - center_y) * wy for px, py in cluster_cart]
        max_proj = max(abs(p) for p in projections)
        required_side = max_proj / (sqrt3 / 2.0)  # apothem = side * sqrt(3)/2
        if required_side > final_side:
            final_side = required_side
    
    # Use the more conservative estimate
    outer_side = max(best_side, final_side)
    
    return inner, (center_x, center_y), outer_side, best_angle
# EVOLVE-BLOCK-END
