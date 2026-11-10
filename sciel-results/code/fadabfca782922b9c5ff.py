# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _edge_normals(hex_verts):
    """Return unit outward normals for each edge of a convex polygon (vertices in CCW order).
    For a regular hexagon with vertices in order, outward normal of edge i is direction from center to edge midpoint."""
    # Edge i is from vertex i to vertex i+1
    # Outward normal: perpendicular to edge, pointing away from center
    edges = np.roll(hex_verts, -1, axis=0) - hex_verts
    # For CCW polygon, outward normal is (edge_y, -edge_x) normalized
    # Wait: for CCW, inward normal is (edge_y, -edge_x)? Let me think.
    # Edge direction e = v2 - v1. For CCW polygon, the interior is to the left of the edge.
    # So outward normal is to the right: (e_y, -e_x)
    nx = edges[:, 1]
    ny = -edges[:, 0]
    norms = np.sqrt(nx**2 + ny**2)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return np.column_stack([nx / norms, ny / norms])


def _sat_gap(hex1_verts, hex2_verts):
    """Compute the minimum gap between two convex hexagons using SAT.
    Returns the minimum distance (positive if separated, negative if overlapping).
    We check all edges of both hexagons."""
    min_gap = np.inf
    for hverts in [hex1_verts, hex2_verts]:
        normals = _edge_normals(hverts)
        for normal in normals:
            # Project both hexagons onto this axis
            proj1 = hex1_verts @ normal
            proj2 = hex2_verts @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Gap between the two intervals
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0  # overlapping on this axis
            if gap < min_gap:
                min_gap = gap
    return min_gap


def _point_in_hexagon(v, hex_verts):
    """Check if point v is inside or on the boundary of convex polygon hex_verts (CCW).
    For each edge (p1, p2), cross(p2-p1, v-p1) should be >= -tol for CCW."""
    tol = 1e-6
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True


def _compute_outer_side(inner_data):
    """Compute the minimum outer hexagon side length that contains all inner hexagons.
    The outer hexagon is centered at origin with angle 0 (for simplicity, we can also optimize).
    Actually, let's find the minimum s such that all inner vertices are inside the outer hexagon
    centered at origin with side s and angle 0."""
    # For a regular hexagon centered at origin with side s and angle 0:
    # Vertices at s * (cos(60i), sin(60i)) for i=0..5
    # The hexagon contains point (x,y) iff for all 6 edge normals, the projection is within bounds.
    # Edge normals for angle 0 hexagon: the edges are between consecutive vertices.
    # Let's just compute the max over all inner vertices of the required s.
    
    max_s = 0.0
    # For a hexagon with center (0,0), side s, angle 0:
    # Vertices: s*(1,0), s*(0.5, sqrt(3)/2), s*(-0.5, sqrt(3)/2), s*(-1,0), s*(-0.5,-sqrt(3)/2), s*(0.5,-sqrt(3)/2)
    # The supporting lines: the hexagon is the intersection of 6 half-planes.
    # For angle 0, the edge normals point in directions: 30, 90, 150, 210, 270, 330 degrees (outward)
    # Actually, let me compute properly.
    
    # For a regular hexagon with side s centered at origin, the distance from center to each edge (apothem) is s * sqrt(3)/2.
    # The outward normals are at angles 30, 90, 150, 210, 270, 330 degrees (for angle 0 orientation).
    # A point (x,y) is inside if for each normal direction n_i, dot((x,y), n_i) <= s * sqrt(3)/2.
    # So s >= 2/sqrt(3) * max_i dot((x,y), n_i)
    
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    for i in range(12):
        cx, cy, ang = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            projections = verts @ normals.T  # (6, 6) but we want per-vertex
            # Actually, for each vertex v, compute dot with each normal
            dots = v @ normals.T  # (6,)
            max_dot = dots.max()
            required_s = 2.0 / np.sqrt(3.0) * max_dot
            if required_s > max_s:
                max_s = required_s
    
    return max_s


def _check_validity(inner_data, outer_data, s):
    """Check all validity conditions. Returns (is_valid, reason)."""
    # Check finite values
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False, "non-finite values"
    
    if s < np.sqrt(12) - 1e-9:
        return False, "s < sqrt(12)"
    
    ox, oy, oang = outer_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False, "outer vertices not distinct"
    
    # Check unit edges of inner hexagons
    for i in range(12):
        cx, cy, ang = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j+1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"inner hex {i} edge {j} length {edge_len}"
    
    # Check separation between inner hexagons
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        inner_verts_list.append(verts)
    
    for i in range(12):
        for j in range(i+1, 12):
            gap = _sat_gap(inner_verts_list[i], inner_verts_list[j])
            if gap < -1e-6:
                return False, f"inner hex {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, outer_verts):
                return False, f"inner hex {i} vertex outside outer"
    
    return True, "valid"


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Start with a reasonable initial configuration
    # Use a hexagonal lattice arrangement
    # Unit hexagons in a honeycomb pattern
    # Adjacent hexagons in a honeycomb are at distance 2*sqrt(3) apart? No.
    # For unit hexagons (side 1), the distance between centers of adjacent hexagons in a honeycomb is 2*sqrt(3)/... 
    # Actually, for two unit hexagons sharing an edge, the center-to-center distance is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
    # Wait, apothem of unit hexagon is sqrt(3)/2. Two hexagons sharing an edge: centers are 2*apothem = sqrt(3) apart.
    
    # Let's use a triangular lattice with spacing sqrt(3)
    dx = np.sqrt(3)
    dy = 1.5  # For hexagonal packing, vertical spacing is 1.5 for unit hexagons? 
    # Actually, for a honeycomb of unit hexagons:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5 (for rows offset by sqrt(3)/2)
    
    # Let's place 12 hexagons in a 3x4 or similar arrangement
    # Try a 4x3 grid in hexagonal lattice
    positions = []
    # Row 0: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * dx
        y = 0
        positions.append([x, y, 0])
    # Row 1: 3 hexagons (offset)
    for i in range(3):
        x = (i - 1) * dx
        y = 1.5
        positions.append([x, y, 0])
    # Row 2: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * dx
        y = 3.0
        positions.append([x, y, 0])
    # That's 11, need one more
    positions.append([0, -1.5, 0])
    
    inner_data = np.array(positions[:12])
    
    # Compute initial outer side
    outer_data = np.array([0.0, 0.0, 0.0])
    s = _compute_outer_side(inner_data)
    
    # Validate initial
    is_valid, reason = _check_validity(inner_data, outer_data, s)
    if not is_valid:
        # Try to fix by increasing s
        s = s * 1.1
        is_valid, reason = _check_validity(inner_data, outer_data, s)
    
    best_s = s
    best_inner = inner_data.copy()
    best_outer = outer_data.copy()
    
    # Gradient descent / simulated annealing to minimize s
    # Objective: minimize s, subject to validity constraints
    
    # Use a penalty method: minimize s + penalty * (violations)
    def objective(x):
        """x is a flattened array of 12*3 + 3 = 39 values: 12 inner (x,y,angle) + outer (x,y,angle) + s"""
        inner = x[:36].reshape(12, 3)
        outer = x[36:39]
        s = x[39] if x.size > 39 else _compute_outer_side(inner)
        
        # Compute actual s needed
        s_needed = _compute_outer_side(inner)
        
        # Penalty for s being too small
        penalty = 0.0
        if s < s_needed:
            penalty += (s_needed - s) ** 2 * 100
        
        # Check separation
        inner_verts_list = []
        for i in range(12):
            cx, cy, ang = inner[i]
            verts = _hex_vertices(cx, cy, 1.0, ang)
            inner_verts_list.append(verts)
        
        for i in range(12):
            for j in range(i+1, 12):
                gap = _sat_gap(inner_verts_list[i], inner_verts_list[j])
                if gap < 0:
                    penalty += gap ** 2 * 1000
        
        # Check containment
        ov, oa = outer[0], outer[1], outer[2]
        outer_verts = _hex_vertices(ov, oa, s, outer[2])
        for i in range(12):
            for v in inner_verts_list[i]:
                if not _point_in_hexagon(v, outer_verts):
                    # Estimate violation
                    penalty += 1.0
        
        return s + penalty
    
    # Simple gradient descent with random restarts
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 60:
            break
        
        # Perturb initial configuration
        if restart == 0:
            x = np.concatenate([inner_data.flatten(), outer_data, [s]])
        else:
            x = np.concatenate([inner_data.flatten(), outer_data, [s]]) + np.random.randn(40) * 0.5
        
        # Gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 30:
                break
            
            x_old = x.copy()
            f_old = objective(x)
            
            # Numerical gradient
            grad = np.zeros(40)
            h = 1e-5
            for i in range(40):
                x_plus = x.copy()
                x_plus[i] += h
                f_plus = objective(x_plus)
                grad[i] = (f_plus - f_old) / h
            
            x = x - lr * grad
            
            # Update s to be the minimum needed
            inner = x[:36].reshape(12, 3)
            s_needed = _compute_outer_side(inner)
            x[39] = max(x[39], s_needed)
            
            if time.time() > deadline - 30:
                break
        
        # Final polish
        inner = x[:36].reshape(12, 3)
        outer = x[36:39]
        s = x[39]
        
        is_valid, reason = _check_validity(inner, outer, s)
        if is_valid and s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    # Return the best result
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
