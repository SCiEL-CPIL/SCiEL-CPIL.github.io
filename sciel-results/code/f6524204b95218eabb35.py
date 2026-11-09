# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0  # seconds

    # Generate candidate positions on a hexagonal grid
    # For unit hexagons to not overlap, centers must be at least 2*sqrt(3)/2 = sqrt(3) apart
    # Actually for flat-top hexagons with side 1, the distance between centers of adjacent hexagons is 2*sqrt(3)/2 = sqrt(3)
    # Wait, let me recalculate. For regular hexagons with side length 1:
    # - The "radius" (center to vertex) is 1
    # - The "apothem" (center to edge midpoint) is sqrt(3)/2
    # - Two hexagons are tangent when their centers are 2*sqrt(3)/2 = sqrt(3) apart? No...
    # Actually, two unit regular hexagons (side=1) are non-overlapping if the distance between centers
    # is at least 2*sqrt(3)/2 = sqrt(3) when they share an edge orientation.
    # But actually, the minimum distance for non-overlap depends on relative orientation.
    # For identical orientations, the minimum center distance is 2*apothem = sqrt(3) ≈ 1.732
    # Wait no. Let me think again.
    # A regular hexagon with side s=1 has:
    #   - circumradius (center to vertex) = 1
    #   - inradius (center to edge) = sqrt(3)/2 ≈ 0.866
    # Two hexagons with the same orientation are non-overlapping iff the distance between centers
    # is at least 2 * inradius = sqrt(3) ≈ 1.732
    # This is because the closest points are on parallel edges.

    # Actually, I need to be more careful. Let me use the separation test from the task:
    # "every pair of inner hexagons is separated: on at least one of their 12 edge normals 
    #  the projected intervals are apart by more than 1e-6"
    
    # For now, use a conservative minimum distance of 2.0 between centers (slightly more than sqrt(3))
    # to ensure non-overlap with same orientation.

    min_dist = 2.0  # conservative minimum center distance for non-overlapping unit hexagons
    
    # Generate grid positions
    # Use a hexagonal lattice with spacing slightly more than min_dist
    spacing = 2.05  # slightly more than min_dist to ensure separation
    
    # Generate positions in a region large enough
    max_coord = 10.0
    positions = []
    for i in range(-int(max_coord / spacing), int(max_coord / spacing) + 1):
        for j in range(-int(max_coord / spacing), int(max_coord / spacing) + 1):
            # Hexagonal lattice offset
            x = i * spacing
            y = j * spacing * np.sqrt(3)
            if abs(x) <= max_coord and abs(y) <= max_coord:
                positions.append((x, y))
    
    positions = np.array(positions)
    
    # Also add some additional candidate positions for better packing
    # Try different spacings and offsets
    for sp in [1.8, 1.9, 2.0, 2.1, 2.2]:
        for offset in [0.0, 0.1, 0.2, 0.3]:
            for i in range(-int(max_coord / sp), int(max_coord / sp) + 1):
                for j in range(-int(max_coord / sp), int(max_coord / sp) + 1):
                    x = i * sp + offset
                    y = j * sp * np.sqrt(3) + offset * 0.5
                    if abs(x) <= max_coord and abs(y) <= max_coord:
                        positions = np.vstack([positions, [[x, y]]])
    
    # Remove duplicates
    positions = np.unique(positions, axis=0)
    
    # Function to check if a set of positions is valid (non-overlapping)
    def is_valid_packing(pos):
        """Check if all pairs of hexagons at these positions are non-overlapping."""
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist - 1e-9:
                    return False
        return True
    
    # Function to compute the minimal enclosing regular hexagon (with fixed orientation)
    def compute_min_enclosing_hexagon(pos, angle_deg=0.0):
        """
        Compute the minimal side length of a regular hexagon (with given orientation)
        that encloses all points in pos.
        
        A regular hexagon with center (cx, cy), side s, and angle a has vertices at:
        (cx + s*cos(a + 60*i), cy + s*sin(a + 60*i)) for i = 0..5
        
        The hexagon can be described by 6 half-planes (one for each edge).
        For a point p to be inside, it must satisfy all 6 inequalities.
        
        For a hexagon centered at origin with angle 0 (flat-top), the edges are at angles
        0, 60, 120, 180, 240, 300 degrees. The normal to each edge points outward.
        
        Actually, let's think of it differently. A regular hexagon with circumradius s
        (side length = s for regular hexagon) centered at (cx, cy) with rotation angle a
        has the property that a point (x, y) is inside if and only if:
        
        For each of the 6 directions at angles (a + 30 + 60*k) for k=0..5 (these are the
        edge normals), the projection of (x-cx, y-cy) onto that direction is <= s * cos(30°) = s * sqrt(3)/2
        
        Wait, I need to be more careful. Let me use the vertex definition.
        
        Vertices: v_k = (cx + s*cos(a + 60*k), cy + s*sin(a + 60*k)) for k=0..5
        
        The edge from v_k to v_{k+1} has a normal direction. The inward normal points toward the center.
        
        For a regular hexagon, the distance from center to each edge (apothem) is s * sqrt(3)/2.
        The edge normals point in directions (a + 30 + 60*k) for k=0..5.
        
        So a point p is inside iff for all k:
        (p - center) · n_k <= apothem = s * sqrt(3)/2
        where n_k = (cos(a + 30 + 60*k), sin(a + 30 + 60*k))
        
        To find the minimal s such that all points are inside, we need:
        s >= max over all points p and all k of: (p - center) · n_k / (sqrt(3)/2)
        
        But we can also vary the center. So we want to minimize s subject to:
        (p_i - c) · n_k <= s * sqrt(3)/2 for all i, k
        
        This is a linear program in variables (c_x, c_y, s):
        Minimize s
        Subject to:
        (p_i - c) · n_k <= s * sqrt(3)/2 for all i, k
        => c · n_k + (s * sqrt(3)/2 - p_i · n_k) >= 0
        => -c · n_k + p_i · n_k <= s * sqrt(3)/2
        
        Let me set up the LP properly.
        Variables: x = [cx, cy, s]
        Objective: minimize s => c = [0, 0, 1]
        
        Constraints for each point i and direction k:
        (p_ix - cx) * n_kx + (p_iy - cy) * n_ky <= s * sqrt(3)/2
        => -cx * n_kx - cy * n_ky - s * sqrt(3)/2 <= -p_ix * n_kx - p_iy * n_ky
        => cx * n_kx + cy * n_ky + s * sqrt(3)/2 >= p_ix * n_kx + p_iy * n_ky
        
        In standard form for linprog (minimize c^T x subject to A_ub x <= b_ub):
        -cx * n_kx - cy * n_ky - s * sqrt(3)/2 <= -p_ix * n_kx - p_iy * n_ky
        
        So A_ub row: [-n_kx, -n_ky, -sqrt(3)/2]
        b_ub: -p_ix * n_kx - p_iy * n_ky
        """
        n_points = len(pos)
        # 6 edge normal directions
        base_angle = np.radians(angle_deg + 30)  # offset by 30 for edge normals
        normals = np.array([
            [np.cos(base_angle + np.radians(60 * k)), np.sin(base_angle + np.radians(60 * k))]
            for k in range(6)
        ])
        
        apothem_factor = np.sqrt(3) / 2
        
        # Build constraint matrix
        A_ub = []
        b_ub = []
        for i in range(n_points):
            for k in range(6):
                A_ub.append([-normals[k, 0], -normals[k, 1], -apothem_factor])
                b_ub.append(-(pos[i, 0] * normals[k, 0] + pos[i, 1] * normals[k, 1]))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Objective: minimize s
        c_obj = np.array([0, 0, 1])
        
        # Bounds: cx, cy can be anything, s >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            s_min = result.x[2]
            cx = result.x[0]
            cy = result.x[1]
            return s_min, cx, cy
        else:
            return None, None, None
    
    # Function to compute minimal enclosing hexagon trying multiple orientations
    def compute_min_enclosing_hexagon_multi(pos):
        """Try multiple orientations and return the best (smallest side length)."""
        best_s = float('inf')
        best_cx = 0
        best_cy = 0
        best_angle = 0
        
        for angle in range(0, 30, 5):  # Try angles 0, 5, 10, 15, 20, 25 degrees
            s, cx, cy = compute_min_enclosing_hexagon(pos, angle)
            if s is not None and s < best_s:
                best_s = s
                best_cx = cx
                best_cy = cy
                best_angle = angle
        
        return best_s, best_cx, best_cy, best_angle
    
    # Find the best subset of 11 positions
    n_positions = len(positions)
    
    # For efficiency, use a greedy approach: try different starting points and grow the set
    # Or use random sampling of 11 positions
    
    best_result = None
    best_score = -float('inf')
    
    # First, try the compact "flower" arrangement: 1 center + 6 around + 4 more
    # This is known to be efficient for hexagonal packings
    
    # Generate candidate 11-point configurations
    configs = []
    
    # Configuration 1: Flower pattern (1 center, 6 ring, 4 outer)
    # Center at (0,0)
    # Ring 1 at distance ~2.0 in 6 directions
    # Ring 2 at distance ~4.0 in 4 directions
    
    flower_configs = []
    for r1 in [1.9, 2.0, 2.1]:
        for r2 in [3.8, 3.9, 4.0, 4.1]:
            for angle_offset in [0, 30]:
                pts = [[0, 0]]
                for k in range(6):
                    a = np.radians(60 * k + angle_offset)
                    pts.append([r1 * np.cos(a), r1 * np.sin(a)])
                # Add 4 more points at the "gaps" of the hexagonal ring
                for k in [0, 1, 3, 4]:  # 4 of the 6 gap positions
                    a = np.radians(60 * k + 30 + angle_offset)
                    pts.append([r2 * np.cos(a), r2 * np.sin(a)])
                configs.append(np.array(pts))
    
    # Configuration 2: Linear arrangement
    for spacing in [2.0, 2.1]:
        pts = []
        for i in range(11):
            pts.append([i * spacing, 0])
        configs.append(np.array(pts))
    
    # Configuration 3: 3x4 grid minus one corner
    for spacing in [2.0, 2.1]:
        pts = []
        for i in range(4):
            for j in range(3):
                pts.append([i * spacing, j * spacing * np.sqrt(3)])
        pts = pts[:11]
        configs.append(np.array(pts))
    
    # Random configurations
    np.random.seed(42)
    for trial in range(200):
        if time.time() > deadline - safety_margin:
            break
        # Random 11 points
        pts = np.random.randn(11, 2) * 5
        # Ensure minimum distance
        valid = True
        for i in range(11):
            for j in range(i + 1, 11):
                if np.linalg.norm(pts[i] - pts[j]) < min_dist:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            configs.append(pts)
    
    # Evaluate each configuration
    for config in configs:
        if time.time() > deadline - safety_margin:
            break
        
        if len(config) != 11:
            continue
        
        if not is_valid_packing(config):
            continue
        
        s, cx, cy, angle = compute_min_enclosing_hexagon_multi(config)
        
        if s is None or s < np.sqrt(11) - 1e-9:
            continue
        
        score = 1.0 / s
        if score > best_score:
            best_score = score
            best_result = (config, cx, cy, angle, s)
    
    # If we found a good result, use it; otherwise fall back to the original
    if best_result is not None:
        config, cx, cy, angle, s = best_result
        
        # Center the configuration so the outer hexagon is centered at origin
        # The outer hexagon center is (cx, cy), so shift inner hexagons by (-cx, -cy)
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i, 0] = config[i, 0] - cx
            inner_hex_data[i, 1] = config[i, 1] - cy
            inner_hex_data[i, 2] = 0.0  # angle in degrees
        
        outer_hex_data = np.array([0.0, 0.0, float(angle)])
        outer_hex_side_length = float(s)
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    else:
        # Fallback to the original configuration
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
