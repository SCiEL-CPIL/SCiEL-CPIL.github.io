# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _hex_vertices(cx, cy, s, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_min_enclosing_hex_side(points, angle_deg):
    """
    Compute the minimum side length of a regular hexagon centered at (0,0) with given angle
    that contains all points.
    """
    # Outer hexagon vertices for unit side
    outer_v_template = _hex_vertices(0, 0, 1.0, angle_deg)
    
    max_s = 0.0
    for e in range(6):
        p1 = outer_v_template[e]
        p2 = outer_v_template[(e + 1) % 6]
        edge = p2 - p1
        # Outward normal
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-12:
            continue
        n = np.array([nx / norm, ny / norm])
        
        # Project all points onto this normal
        # Points are assumed to be relative to center (0,0)
        projs = points @ n
        max_proj = projs.max()
        if max_proj > 0:
            # The edge is at distance s * sqrt(3) / 2 from center along normal n
            # So s >= max_proj * 2 / sqrt(3)
            required_s = max_proj * 2.0 / np.sqrt(3.0)
            max_s = max(max_s, required_s)
            
    return max_s

def _check_separation(inner_hexes):
    """Check if all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        # Compute normals
        nm = np.zeros((6, 2))
        for k in range(6):
            p1 = v[k]
            p2 = v[(k + 1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.linalg.norm([nx, ny])
            if norm > 1e-12:
                nm[k] = [nx / norm, ny / norm]
        vertices_list.append(v)
        normals_list.append(nm)
    
    for i in range(n):
        for j in range(i + 1, n):
            all_normals = np.vstack([normals_list[i], normals_list[j]])
            separated = False
            for k in range(12):
                n = all_normals[k]
                proj1 = vertices_list[i] @ n
                proj2 = vertices_list[j] @ n
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Central 7 hexagons
    center = np.array([0.0, 0.0, 0.0])
    ring1 = [
        np.array([2.0, 0.0, 0.0]),
        np.array([1.0, np.sqrt(3), 0.0]),
        np.array([-1.0, np.sqrt(3), 0.0]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([-1.0, -np.sqrt(3), 0.0]),
        np.array([1.0, -np.sqrt(3), 0.0])
    ]
    
    base_hexes = np.vstack([center] + ring1)
    
    # Ring 2 positions (12 positions)
    # Distance 2*sqrt(3) from center
    r2 = 2 * np.sqrt(3)
    angles_deg = np.arange(0, 360, 30) # 0, 30, 60...
    ring2_positions = []
    for a in angles_deg:
        rad = np.deg2rad(a)
        x = r2 * np.cos(rad)
        y = r2 * np.sin(rad)
        ring2_positions.append(np.array([x, y, 0.0]))
    
    best_side = float('inf')
    best_params = None
    best_angle = 0.0
    
    # Try different outer hexagon angles
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline - 10:
            break
            
        # We need to pick 4 positions from ring2_positions
        # To reduce search space, we can assume symmetry or just iterate all combinations
        # 12 choose 4 = 495 combinations. Very fast.
        for comb in combinations(range(12), 4):
            extra_hexes = np.vstack([ring2_positions[i] for i in comb])
            all_hexes = np.vstack([base_hexes, extra_hexes])
            
            # Check separation
            if not _check_separation(all_hexes):
                continue
                
            # Compute min enclosing side
            # Get all vertices
            verts = []
            for h in all_hexes:
                v = _hex_vertices(h[0], h[1], 1.0, h[2])
                verts.append(v)
            all_verts = np.vstack(verts)
            
            side = _compute_min_enclosing_hex_side(all_verts, outer_angle)
            
            if side < best_side:
                best_side = side
                best_params = all_hexes.copy()
                best_angle = outer_angle
                
    # Fallback if no valid packing found (should not happen)
    if best_params is None:
        best_params = base_hexes
        best_side = _compute_min_enclosing_hex_side(np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in best_params]), 0.0)
        best_angle = 0.0

    return best_params, np.array([0.0, 0.0, float(best_angle)]), float(best_side)

# EVOLVE-BLOCK-END
