import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Unit hexagon vertices (flat-top, side length 1)
    hex_verts_base = np.array([
        [1.0, 0.0],
        [0.5, sqrt3 / 2.0],
        [-0.5, sqrt3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -sqrt3 / 2.0],
        [0.5, -sqrt3 / 2.0]
    ])

    # Outer hexagon normals (pointing outward from center)
    # For a regular hexagon with vertex at angle 0, the edge normals are at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [math.cos(math.pi / 6 + k * math.pi / 3), math.sin(math.pi / 6 + k * math.pi / 3)]
        for k in range(6)
    ])

    def get_hex_verts(angle_rad):
        """Get vertices of a unit hexagon rotated by angle_rad."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return hex_verts_base @ rot.T

    def compute_side_length(positions, angles):
        """
        Compute the minimum side length of the outer hexagon (centered at origin, angle 0)
        that contains all inner hexagons.
        positions: (N, 2) array of centers
        angles: (N,) array of rotation angles in radians
        Returns: minimum side length S
        """
        max_s = 0.0
        for i in range(N):
            verts = get_hex_verts(angles[i])
            all_pts = positions[i] + verts
            # For each vertex, project onto each outer normal
            # Constraint: dot(P, n_k) <= S * sqrt(3) / 2
            # So S >= 2 * max_k dot(P, n_k) / sqrt(3)
            # But we need max over all P and all k
            # dot(P, n_k) for all P, k
            # = max over k of (max over P of dot(P, n_k))
            # = max over k of dot(max_P(P, n_k), n_k) ... no
            # Actually: for a fixed k, max over P of dot(P, n_k) = max over P of (P_x * n_kx + P_y * n_ky)
            # This is just the support function in direction n_k
            for k in range(6):
                proj = all_pts[:, 0] * outer_normals[k, 0] + all_pts[:, 1] * outer_normals[k, 1]
                max_proj = np.max(proj)
                if max_proj > 0:
                    s_req = 2.0 * max_proj / sqrt3
                    if s_req > max_s:
                        max_s = s_req
        return max_s

    def check_disjointness(positions, angles, min_dist_sq=None):
        """
        Check that all pairs of hexagons are disjoint.
        Two unit hexagons are disjoint if the distance between their centers
        is greater than the maximum "reach" in any direction... 
        Actually, for regular hexagons of side 1, the minimum distance between centers
        for non-overlap depends on orientation.
        
        A simpler sufficient condition: if the distance between centers > 2 * (circumradius) = 2 * 1 = 2,
        they are definitely disjoint. But this is conservative.
        
        Better: two convex polygons are disjoint if no edge of one intersects any edge of the other
        and neither contains the other. For efficiency, we can use a simpler check.
        
        For two regular hexagons of side 1, the maximum distance from center to any point is 1 (circumradius).
        So if center distance > 2, they are disjoint. But they can be closer if rotated.
        
        Actually, the inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866.
        The minimum distance between centers for two non-overlapping unit hexagons
        (regardless of rotation) is at least 2 * inradius = sqrt(3) ≈ 1.732? No, that's not right either.
        
        Let me think more carefully. Two unit regular hexagons. The "width" of a unit hexagon
        (distance between parallel edges) is sqrt(3). The "diameter" (distance between opposite vertices) is 2.
        
        For two hexagons to be disjoint, for every direction, the projections must not overlap.
        This is complex. Let's use a conservative but practical check:
        - If center distance > 2.0, definitely disjoint.
        - If center distance < sqrt(3) - epsilon, definitely overlapping (since the inradius is sqrt(3)/2,
          so two hexagons with centers closer than 2*inradius = sqrt(3) will have overlapping interiors
          if aligned, but rotation can help).
        
        Actually, let's just do a proper polygon intersection check for close pairs.
        """
        eps = 1e-9
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist_sq = dx * dx + dy * dy
                # Quick reject: if distance > 2 + eps, definitely disjoint
                if dist_sq > (2.0 + eps) ** 2:
                    continue
                # If distance < sqrt(3) - eps, likely overlapping, but let's check properly
                # Get vertices
                verts_i = positions[i] + get_hex_verts(angles[i])
                verts_j = positions[j] + get_hex_verts(angles[j])
                # Check if polygons intersect
                if polygons_intersect(verts_i, verts_j):
                    return False
        return True

    def polygons_intersect(poly1, poly2):
        """Check if two convex polygons intersect."""
        n1 = len(poly1)
        n2 = len(poly2)
        # Check if any vertex of poly1 is inside poly2
        for i in range(n1):
            if point_in_convex_poly(poly1[i], poly2):
                return True
        # Check if any vertex of poly2 is inside poly1
        for j in range(n2):
            if point_in_convex_poly(poly2[j], poly1):
                return True
        # Check for edge intersections
        for i in range(n1):
            for j in range(n2):
                if segments_intersect(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                    return True
        return False

    def point_in_convex_poly(point, poly):
        """Check if point is inside a convex polygon."""
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            # Cross product to check if point is to the left of edge
            cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            if cross < -1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hex_grid_init():
        """Initialize with a hexagonal grid arrangement."""
        positions = []
        angles = [0.0] * N
        
        # Center
        positions.append([0.0, 0.0])
        
        # 6 neighbors
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
        for q, r in neighbors:
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            positions.append([x, y])
        
        # 5 more from next ring
        extra = [(2,0), (2,-1), (1,1), (-1,2), (-2,1)]
        for q, r in extra:
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            positions.append([x, y])
        
        return np.array(positions), np.array(angles)

    def optimize(positions, angles, max_iter=500, step_init=0.1):
        """
        Local optimization using a simple gradient-free method.
        We want to minimize the outer side length while maintaining disjointness.
        """
        best_side = compute_side_length(positions, angles)
        best_positions = positions.copy()
        best_angles = angles.copy()
        
        if not check_disjointness(positions, angles):
            return None, None, None
        
        step = step_init
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(N):
                # Try moving hexagon i in random direction
                for _ in range(4):
                    if time.time() > deadline:
                        break
                    dx = (np.random.rand() - 0.5) * step
                    dy = (np.random.rand() - 0.5) * step
                    dangle = (np.random.rand() - 0.5) * step * 0.5
                    
                    new_pos = positions.copy()
                    new_ang = angles.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    new_ang[i] += dangle
                    
                    if not check_disjointness(new_pos, new_ang):
                        continue
                    
                    new_side = compute_side_length(new_pos, new_ang)
                    if new_side < best_side - 1e-10:
                        best_side = new_side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        positions = new_pos
                        angles = new_ang
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 20:
                    step *= 0.8
                    if step < 1e-5:
                        break
            else:
                step = max(step * 1.05, 0.01)
        
        return best_positions, best_angles, best_side

    # Main search
    best_result = None
    best_side = float('inf')
    
    # Start with hex grid initialization
    positions, angles = hex_grid_init()
    # Scale slightly to ensure strict disjointness
    positions *= 1.001
    
    # Run local optimization
    pos, ang, side = optimize(positions, angles, max_iter=300, step_init=0.05)
    if pos is not None and side < best_side:
        best_side = side
        best_result = (pos, ang, side)
    
    # Try a few more random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Perturb the initial configuration
        positions = hex_grid_init()[0] * 1.001
        angles = np.zeros(N)
        
        # Add random perturbations
        for i in range(N):
            positions[i, 0] += (np.random.rand() - 0.5) * 0.3
            positions[i, 1] += (np.random.rand() - 0.5) * 0.3
            angles[i] = (np.random.rand() - 0.5) * 0.5
        
        pos, ang, side = optimize(positions, angles, max_iter=200, step_init=0.03)
        if pos is not None and side < best_side:
            best_side = side
            best_result = (pos, ang, side)
    
    if best_result is None:
        # Fallback: use the initial hex grid
        positions, angles = hex_grid_init()
        positions *= 1.001
        side = compute_side_length(positions, angles)
        best_result = (positions, angles, side)
        best_side = side
    
    positions, angles, side = best_result
    
    # Convert to the required output format
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i] * 180.0 / math.pi)])
    
    center = (0.0, 0.0)
    angle = 0.0
    side = float(side) * 1.001  # small margin for safety
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
