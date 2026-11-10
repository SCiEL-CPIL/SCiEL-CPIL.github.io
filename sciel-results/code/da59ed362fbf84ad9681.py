# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    
    Returns
        inner_hex_data: np.ndarray of shape (12,3), rows (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    
    def hex_vertices(cx, cy, d, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60) * np.arange(6)
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])
    
    def get_edge_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15)
        return normals
    
    def separation_violation(verts1, verts2):
        """Return max violation of separation (positive means overlap)."""
        n1 = get_edge_normals(verts1)
        n2 = get_edge_normals(verts2)
        all_normals = np.vstack([n1, n2])
        
        max_violation = 0.0
        for normal in all_normals:
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If intervals overlap, violation is the overlap amount
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_violation:
                max_violation = overlap
        return max_violation
    
    def point_in_hexagon(v, outer_verts):
        """Check if point v is inside or on outer hexagon using cross products."""
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
        return True
    
    def compute_required_outer_side(inner_verts_list, outer_center):
        """Compute the minimum outer hex side length needed to contain all inner hexagons."""
        # For a regular hexagon centered at outer_center with side s and angle 0,
        # the vertices are at distance s from center at angles 0, 60, ..., 300
        # A point (x,y) relative to center is inside if for each edge, the cross product test passes.
        
        # For a regular hexagon with side s centered at origin, angle 0:
        # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # The hexagon can be described as the intersection of 6 half-planes.
        # For angle 0, the edges are at angles 30, 90, 150, 210, 270, 330 from center.
        # Each edge is at distance s*sqrt(3)/2 from center.
        
        # Actually, for a regular hexagon with side s, the apothem (distance from center to edge) is s*sqrt(3)/2.
        # The hexagon is the set of points whose distance to each of the 6 edge lines is <= apothem.
        
        # For a hexagon centered at origin with "flat top" orientation (angle 0 in our convention):
        # The 6 edge normals point at angles 0, 60, 120, 180, 240, 300 degrees.
        # Wait, let me think more carefully.
        
        # Vertices at angles 0, 60, 120, 180, 240, 300 from center, at distance s.
        # Edge between vertex at angle 0 and vertex at angle 60:
        #   V0 = (s, 0), V1 = (s/2, s*sqrt(3)/2)
        #   Edge vector: (-s/2, s*sqrt(3)/2)
        #   Normal (inward): (s*sqrt(3)/2, s/2) normalized -> (sqrt(3)/2, 1/2) which is angle 30
        #   Distance from origin to this edge: s*sqrt(3)/2
        
        # So the 6 inward normals are at angles 30, 90, 150, 210, 270, 330 degrees.
        # The hexagon contains point p (relative to center) iff for all 6 normals n_i:
        #   p · n_i <= s * sqrt(3) / 2
        
        # So s >= max_i (p_i · n_i) / (sqrt(3)/2) = max_i (p_i · n_i) * 2/sqrt(3)
        
        sqrt3 = np.sqrt(3)
        # Inward normal angles: 30, 90, 150, 210, 270, 330
        normal_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        required_s = 0.0
        for verts in inner_verts_list:
            # Get all vertices relative to outer_center
            rel_verts = verts - outer_center
            # For each vertex, compute max dot product with any normal
            dots = rel_verts @ normals.T  # shape (6, 6)
            max_dot = dots.max(axis=0)  # max over vertices for each normal
            s_needed = np.max(max_dot) * 2.0 / sqrt3
            if s_needed > required_s:
                required_s = s_needed
        
        return required_s
    
    def objective(params):
        """Objective: outer hex side length + penalty for violations."""
        n_inner = 12
        # Parse parameters: 12*(x,y,angle) + outer_x + outer_y + outer_angle = 36 + 3 = 39
        inner_params = params[:36].reshape(12, 3)
        outer_x = params[36]
        outer_y = params[37]
        outer_angle = params[38]
        
        # Compute inner hexagon vertices
        inner_verts_list = []
        for i in range(n_inner):
            x, y, a = inner_params[i]
            verts = hex_vertices(x, y, 1.0, a)
            inner_verts_list.append(verts)
        
        # Compute required outer side
        outer_center = np.array([outer_x, outer_y])
        s = compute_required_outer_side(inner_verts_list, outer_center)
        
        # Penalty for separation violations
        penalty = 0.0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                viol = separation_violation(inner_verts_list[i], inner_verts_list[j])
                if viol > 0:
                    penalty += viol ** 2
        
        # Penalty for inner vertices outside outer hexagon
        outer_verts = hex_vertices(outer_x, outer_y, s, outer_angle)
        for verts in inner_verts_list:
            for v in verts:
                if not point_in_hexagon(v, outer_verts):
                    # Compute how far outside
                    min_cross = float('inf')
                    for i in range(6):
                        p1 = outer_verts[i]
                        p2 = outer_verts[(i + 1) % 6]
                        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                        if cross < min_cross:
                            min_cross = cross
                    if min_cross < 0:
                        penalty += min_cross ** 2
        
        return s + 100.0 * penalty
    
    # Initial configuration: honeycomb-like arrangement
    # 12 hexagons in a roughly hexagonal cluster
    # Let's use a pattern: center + 6 around + 5 more
    initial_inner = []
    # Center
    initial_inner.append([0, 0, 0])
    # 6 neighbors at distance sqrt(3) ~ 1.732
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        initial_inner.append([x, y, 0])
    # 5 more: extend the hexagonal pattern
    # Add hexagons at the "corners" of the 7-hex cluster
    for k in range(6):
        angle = np.radians(60 * k + 30)
        x = 3 * np.sqrt(3) * np.cos(angle)
        y = 3 * np.sqrt(3) * np.sin(angle)
        if len(initial_inner) < 12:
            initial_inner.append([x, y, 0])
    
    # Trim to exactly 12
    initial_inner = initial_inner[:12]
    
    initial_params = np.array(initial_inner).flatten()
    initial_outer = np.array([0, 0, 0])
    initial_params = np.concatenate([initial_params, initial_outer])
    
    # Optimize
    result = minimize(objective, initial_params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_params = result.x
    
    # Extract final configuration
    inner_params = best_params[:36].reshape(12, 3)
    outer_x = best_params[36]
    outer_y = best_params[37]
    outer_angle = best_params[38]
    
    # Compute final outer side
    inner_verts_list = []
    for i in range(12):
        x, y, a = inner_params[i]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
    
    outer_center = np.array([outer_x, outer_y])
    s = compute_required_outer_side(inner_verts_list, outer_center)
    
    # Ensure s is at least sqrt(12)
    s = max(s, np.sqrt(12) + 1e-9)
    
    inner_hex_data = inner_params
    outer_hex_data = np.array([outer_x, outer_y, outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
