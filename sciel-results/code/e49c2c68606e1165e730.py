# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _check_separation(v1, v2, tol=1e-6):
    """SAT for two convex hexagons (6,2) arrays. Returns True if separated."""
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = verts @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
    return False


def _check_inside(inner_verts, outer_verts, tol=1e-6):
    """Check all inner vertices are inside outer hexagon."""
    for v in inner_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            if cross < -tol:
                return False
    return True


def _is_valid(inner_data, outer_data, s, tol=1e-6):
    """Full validity check."""
    # Check finite
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-12:
        return False
    # Check outer vertices distinct
    ov = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False
    # Check inner hexagons have unit edges
    inner_verts_list = []
    for row in inner_data:
        verts = _hex_vertices(row[0], row[1], 1.0, row[2])
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i + 1) % 6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)
    # Check separation
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j], tol):
                return False
    # Check inside
    if not _check_inside(np.vstack(inner_verts_list), ov, tol):
        return False
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 630s wall-clock, leave margin

    # Initial cluster arrangement
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
            [0, -4, 0],
        ]
    )

    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0

    # Compute tight outer hexagon that contains all inner hexagons
    def compute_tight_outer(inner_data, outer_center, outer_angle_deg):
        """Find minimum side length s such that all inner hexagons fit inside outer hexagon."""
        # Get all inner vertices
        all_verts = []
        for row in inner_data:
            verts = _hex_vertices(row[0], row[1], 1.0, row[2])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # For a regular hexagon with center c, angle a, side s:
        # The distance from center to each edge is s * sqrt(3)/2
        # A point p is inside if for all 6 edge normals n_i, dot(p - c, n_i) <= s * sqrt(3)/2
        # So s >= max over all points and edges of dot(p - c, n_i) / (sqrt(3)/2)
        
        a = np.deg2rad(outer_angle_deg)
        edge_normals = []
        for i in range(6):
            angle = a + np.deg2rad(60 * i)
            # Edge normal points outward
            n = np.array([np.cos(angle), np.sin(angle)])
            edge_normals.append(n)
        edge_normals = np.array(edge_normals)
        
        # For each vertex, compute max dot product with each normal
        # p - c for each vertex
        rel = all_verts - outer_center
        # (72, 2) @ (2, 6) -> (72, 6)
        dots = rel @ edge_normals
        max_dot = dots.max()
        
        # s * sqrt(3)/2 >= max_dot
        s_min = max_dot / (np.sqrt(3) / 2)
        return max(s_min, np.sqrt(12))
    
    # Try different outer hexagon angles to find tightest fit
    best_s = 8.0
    best_angle = 0.0
    for angle_deg in np.linspace(0, 60, 121):
        s = compute_tight_outer(inner_hex_data, outer_hex_data[:2], angle_deg)
        if s < best_s:
            best_s = s
            best_angle = angle_deg
    
    outer_hex_side_length = best_s
    outer_hex_data = np.array([0, 0, best_angle])
    
    # Verify validity
    if not _is_valid(inner_hex_data, outer_hex_data, outer_hex_side_length):
        # Fallback to safe value
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
    
    # Local search: try to shrink outer hexagon further by adjusting angle
    current_s = outer_hex_side_length
    current_angle = outer_hex_data[2]
    
    for _ in range(500):
        if time.time() > deadline - 30:
            break
        
        # Try small angle perturbations
        improved = False
        for da in np.linspace(-5, 5, 21):
            if da == 0:
                continue
            new_angle = current_angle + da
            new_s = compute_tight_outer(inner_hex_data, outer_hex_data[:2], new_angle)
            if new_s < current_s - 1e-10:
                # Verify
                test_outer = np.array([0, 0, new_angle])
                if _is_valid(inner_hex_data, test_outer, new_s):
                    current_s = new_s
                    current_angle = new_angle
                    improved = True
                    break
        
        if not improved:
            break
    
    outer_hex_side_length = current_s
    outer_hex_data = np.array([0, 0, current_angle])
    
    # Final validity check
    if not _is_valid(inner_hex_data, outer_hex_data, outer_hex_side_length):
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
