# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, rotation angle_deg."""
    angles = np.deg2rad(angle_deg) + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_verts, outer_center, outer_angle_deg):
    """
    For a set of inner vertices (N,2) and an outer hexagon defined by center and angle,
    find the minimum side length s such that all inner vertices are inside the outer hexagon.
    """
    oc = np.array(outer_center)
    oa = np.deg2rad(outer_angle_deg)
    base_angles = oa + np.deg2rad(60.0 * np.arange(6))
    unit_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    rel = inner_verts - oc  # (N, 2)
    edge_vecs = np.roll(unit_verts, -1, axis=0) - unit_verts  # (6, 2)
    
    # A[i, p] = cross(edge_vecs[i], rel[p])
    A = edge_vecs[:, 0:1] * rel[1:2, :] - edge_vecs[:, 1:2] * rel[0:1, :]  # (6, N)
    
    s_min = np.max(-2.0 * A / np.sqrt(3.0))
    s_min = max(s_min, 0.0)
    
    return s_min


def _get_all_configurations():
    """
    Generate multiple candidate rigid-body configurations for 12 unit hexagons.
    Returns a list of (12, 3) arrays.
    """
    sqrt3 = np.sqrt(3.0)
    configs = []
    
    # Configuration 1: Center + ring 1 + 6 from ring 2 (original)
    axial1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1), (2, -1),
    ]
    pos1 = []
    for q, r in axial1:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos1.append([x, y, 0.0])
    configs.append(np.array(pos1))
    
    # Configuration 2: More symmetric - center + ring 1 + 5 from ring 2 (different selection)
    axial2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 0), (0, -2), (2, -1),
    ]
    pos2 = []
    for q, r in axial2:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos2.append([x, y, 0.0])
    configs.append(np.array(pos2))
    
    # Configuration 3: 3-4-3-2 row arrangement
    # Rows of hexagons with alternating offsets
    row_y = [0, sqrt3, 2*sqrt3, 3*sqrt3]
    row_counts = [3, 4, 3, 2]
    pos3 = []
    for i, (y, count) in enumerate(zip(row_y, row_counts)):
        # Center the row
        x_start = -1.5 * (count - 1) / 2.0
        for j in range(count):
            x = x_start + 1.5 * j
            pos3.append([x, y, 0.0])
    configs.append(np.array(pos3))
    
    # Configuration 4: 4-4-4 arrangement
    pos4 = []
    for i in range(3):
        y = i * sqrt3
        for j in range(4):
            x = -2.25 + 1.5 * j
            pos4.append([x, y, 0.0])
    configs.append(np.array(pos4))
    
    # Configuration 5: Compact hexagonal cluster - try different ring 2 selections
    axial5 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1),
    ]
    pos5 = []
    for q, r in axial5:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos5.append([x, y, 0.0])
    configs.append(np.array(pos5))
    
    # Filter out configurations with overlaps
    valid_configs = []
    for cfg in configs:
        if _check_separation_fast(cfg):
            valid_configs.append(cfg)
    
    return valid_configs


def _check_separation_fast(inner_positions):
    """
    Quick check that no two inner hexagons overlap.
    For two regular hexagons with the same orientation, they don't overlap
    if and only if the center-to-center distance >= sqrt(3) (for unit side length).
    """
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(inner_positions[i, :2] - inner_positions[j, :2])
            if dist < np.sqrt(3.0) - 1e-9:
                return False
    return True


def _compute_s_for_params(inner_positions, outer_center, outer_angle_deg):
    """Compute the minimum outer hexagon side length for given parameters."""
    all_verts = []
    for pos in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    s = _min_containing_hex_side(all_verts, outer_center, outer_angle_deg)
    return s


def _validate_result(inner_hex_data, outer_hex_data, s):
    """Validate the result against the task constraints."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12.0):
        return False
    
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    oc = outer_hex_data[:2]
    shifted_inner = inner_hex_data.copy()
    shifted_inner[:, :2] -= oc
    
    for pos in shifted_inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i + 1) % 6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    if not _check_separation_fast(shifted_inner):
        return False
    
    all_verts = []
    for pos in shifted_inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    ov = _hex_vertices(0, 0, s, outer_hex_data[2])
    
    for v in all_verts:
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i + 1) % 6]
            edge = p2 - p1
            to_v = v - p1
            cross_val = edge[0] * to_v[1] - edge[1] * to_v[0]
            if cross_val < -1e-6:
                return False
    
    return True


def _optimize_for_config(inner_positions, deadline, time_budget=120.0):
    """
    Optimize the outer hexagon parameters for a given configuration.
    Returns (best_s, best_params) where best_params = (angle_deg, cx, cy).
    """
    start_time = time.time()
    
    # Precompute all vertices
    all_verts = []
    for pos in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # Centroid of inner hexagons
    centroid = np.mean(inner_positions[:, :2], axis=0)
    
    best_s = np.inf
    best_params = (0.0, 0.0, 0.0)
    
    # Coarse grid search
    angle_range = np.linspace(0, 60, 13)  # 13 points, step 5 degrees
    center_range = np.linspace(-1.5, 1.5, 7)  # 7 points, step 0.5
    
    for angle in angle_range:
        if time.time() > deadline - time_budget * 0.3:
            break
        for cx in center_range:
            for cy in center_range:
                s = _min_containing_hex_side(all_verts, [cx, cy], angle)
                if s < best_s:
                    best_s = s
                    best_params = (angle, cx, cy)
    
    # Local refinement using scipy
    if time.time() < deadline - time_budget * 0.2:
        best_angle, best_cx, best_cy = best_params
        
        def objective(params):
            ang, cx, cy = params
            return _min_containing_hex_side(all_verts, [cx, cy], ang)
        
        x0 = np.array([best_angle, best_cx, best_cy])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_s:
                best_s = result.fun
                best_params = tuple(result.x)
        except:
            pass
        
        # Further local search with smaller steps
        if time.time() < deadline - time_budget * 0.1:
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
            for step in step_sizes:
                if time.time() > deadline - time_budget * 0.1:
                    break
                improved = True
                while improved and time.time() < deadline - time_budget * 0.1:
                    improved = False
                    neighbors = [
                        (best_angle + step, best_cx, best_cy),
                        (best_angle - step, best_cx, best_cy),
                        (best_angle, best_cx + step, best_cy),
                        (best_angle, best_cx - step, best_cy),
                        (best_angle, best_cx, best_cy + step),
                        (best_angle, best_cx, best_cy - step),
                    ]
                    for ang, cx, cy in neighbors:
                        if time.time() > deadline - time_budget * 0.1:
                            break
                        s = _min_containing_hex_side(all_verts, [cx, cy], ang)
                        if s < best_s - 1e-10:
                            best_s = s
                            best_angle, best_cx, best_cy = ang, cx, cy
                            improved = True
    
    return best_s, best_params


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Generate multiple configurations
    configs = _get_all_configurations()
    
    if not configs:
        # Fallback
        configs = [np.array([
            [0, 0, 0],
            [1.5, 0, 0], [-1.5, 0, 0],
            [0.75, 1.5, 0], [-0.75, 1.5, 0],
            [0.75, -1.5, 0], [-0.75, -1.5, 0],
            [3.0, 0, 0], [-3.0, 0, 0],
            [1.5, 3.0, 0], [-1.5, 3.0, 0],
            [1.5, -3.0, 0], [-1.5, -3.0, 0],
        ])]
    
    best_s = np.inf
    best_config = None
    best_params = (0.0, 0.0, 0.0)
    
    # Allocate time budget per configuration
    n_configs = len(configs)
    time_per_config = min(120.0, (deadline - start_time - 30.0) / n_configs)
    
    for i, config in enumerate(configs):
        if time.time() > deadline - 30.0:
            break
        
        s, params = _optimize_for_config(config, deadline, time_budget=time_per_config)
        
        if s < best_s:
            best_s = s
            best_config = config
            best_params = params
    
    # If no valid config found, use fallback
    if best_config is None:
        best_config = configs[0]
        centroid = np.mean(best_config[:, :2], axis=0)
        best_params = (0.0, centroid[0], centroid[1])
        best_s = _compute_s_for_params(best_config, centroid, 0.0)
    
    outer_angle, outer_cx, outer_cy = best_params
    
    inner_hex_data = best_config.copy()
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])
    
    # Validate
    if not _validate_result(inner_hex_data, outer_hex_data, best_s):
        # Try fallback: center outer hexagon at centroid
        centroid = np.mean(best_config[:, :2], axis=0)
        outer_hex_data = np.array([centroid[0], centroid[1], 0.0])
        best_s = _compute_s_for_params(best_config, centroid, 0.0)
        
        if not _validate_result(inner_hex_data, outer_hex_data, best_s):
            # Last resort
            inner_hex_data = np.array([
                [0, 0, 0],
                [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0],
                [0, -4, 0],
            ])
            outer_hex_data = np.array([0, 0, 0])
            best_s = 8.0
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
