# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon (side s) centered at (cx,cy) with angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(cx, cy, s, a_deg):
    """Return 6 outward edge normals (unit vectors) for a hexagon."""
    verts = _hex_vertices(cx, cy, s, a_deg)
    normals = np.zeros((6, 2))
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # outward normal: rotate edge by -90 deg, normalize, point away from center
        n = np.array([edge[1], -edge[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # ensure outward: dot with (midpoint - center) should be positive
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - np.array([cx, cy])) < 0:
            n = -n
        normals[i] = n
    return normals


def _check_pair_separated(verts1, verts2):
    """
    Check if two hexagons are separated on at least one of their 12 edge normals.
    Returns True if separated (valid), False if overlapping.
    """
    # 6 normals from hex1, 6 normals from hex2
    for i in range(6):
        n = verts1[i + 1] - verts1[i]
        n = np.array([n[1], -n[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # ensure outward for hex1
        c1 = np.mean(verts1, axis=0)
        if np.dot(n, (verts1[i] + verts1[i+1]) / 2 - c1) < 0:
            n = -n
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        if proj1.max() - proj2.min() > 1e-6 or proj2.max() - proj1.min() > 1e-6:
            return True
    for i in range(6):
        n = verts2[i + 1] - verts2[i]
        n = np.array([n[1], -n[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        c2 = np.mean(verts2, axis=0)
        if np.dot(n, (verts2[i] + verts2[i+1]) / 2 - c2) < 0:
            n = -n
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        if proj1.max() - proj2.min() > 1e-6 or proj2.max() - proj1.min() > 1e-6:
            return True
    return False


def _check_inside_outer(verts_inner, outer_verts):
    """Check all inner vertices are inside or on the outer hexagon."""
    for p in verts_inner:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            # cross product: (p2-p1) x (p-p1) should be >= -1e-6 (inside or on)
            cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
            if cross < -1e-6:
                return False
    return True


def _compute_penalty(pos, side, outer_center):
    """
    Compute penalty for a configuration.
    pos: (11, 2) array of inner hex centers
    side: outer hex side length
    outer_center: (2,) array
    Returns (penalty, is_valid)
    """
    penalty = 0.0
    is_valid = True
    
    # Check inner hexagons don't overlap
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
            v2 = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, 0.0)
            if not _check_pair_separated(v1, v2):
                # Compute overlap penalty
                # Use minimum distance between vertices as proxy
                min_dist = np.inf
                for a in v1:
                    for b in v2:
                        d = np.linalg.norm(a - b)
                        if d < min_dist:
                            min_dist = d
                penalty += max(0, 1e-6 - min_dist) * 1000
                is_valid = False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, 0.0)
    for i in range(11):
        v_inner = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
        if not _check_inside_outer(v_inner, outer_verts):
            # Compute how far outside
            for p in v_inner:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    edge = p2 - p1
                    cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    if cross < -1e-6:
                        # Distance to edge
                        edge_len = np.linalg.norm(edge)
                        dist = -cross / edge_len
                        penalty += max(0, dist) * 100
                        is_valid = False
    
    return penalty, is_valid


def _objective(x):
    """
    Objective function for optimization.
    x: (23,) array = [side, x1, y1, ..., x11, y11]
    """
    side = x[0]
    pos = x[1:].reshape(11, 2)
    
    # Keep side within reasonable bounds
    if side < np.sqrt(11) - 0.1 or side > 15:
        return 1e6
    
    # Center outer hex at origin for simplicity
    outer_center = np.array([0.0, 0.0])
    
    penalty, is_valid = _compute_penalty(pos, side, outer_center)
    
    # Objective: minimize side + penalty
    obj = side + penalty
    return obj


def _hexagon_packing_11_inner():
    """Main optimization routine."""
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: honeycomb pattern
    # Center hex at origin, 6 around it, 4 more
    hex_height = np.sqrt(3) / 2  # ~0.866
    
    # Honeycomb: 1 center, 6 in first ring, 4 in second ring (partial)
    # Actually for 11: 1 + 6 + 4 = 11
    # First ring: 6 hexagons at distance 2*hex_height = sqrt(3) from center
    # Second ring: 4 hexagons
    
    r1 = np.sqrt(3)  # distance to first ring centers
    r2 = 2 * np.sqrt(3)  # distance to second ring centers
    
    initial_pos = np.zeros((11, 2))
    initial_pos[0] = [0, 0]  # center
    # First ring: 6 hexagons
    for i in range(6):
        angle = np.pi / 3 * i
        initial_pos[i + 1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    # Second ring: 4 hexagons (at angles 0, 60, 180, 240 degrees, but only 4)
    # Actually let's place them at 30, 90, 150, 210, 270, 330 - but we only need 4
    # Better: place at the "gaps" of the first ring
    for i in range(4):
        angle = np.pi / 3 * (i * 2 + 1) + np.pi / 6  # offset angles
        initial_pos[i + 7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    
    # Initial side estimate
    max_extent = np.max(np.linalg.norm(initial_pos, axis=1)) + 1.0  # +1 for hex radius
    initial_side = max_extent + 0.5
    
    # Define bounds for DE
    # side: [sqrt(11), 12]
    # positions: [-8, 8] each
    bounds = [(np.sqrt(11) - 0.5, 12.0)]
    for _ in range(22):
        bounds.append((-8.0, 8.0))
    
    # Initial point for DE
    x0 = np.concatenate([[initial_side], initial_pos.flatten()])
    
    best_obj = _objective(x0)
    best_x = x0.copy()
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            x0=x0,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol',
            maxfev=5000
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Check time
    if time.time() > deadline - 60:
        # Not enough time for polish
        pass
    else:
        # Polish with Nelder-Mead
        try:
            result2 = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_x = result2.x.copy()
        except Exception as e:
            print(f"NM failed: {e}")
    
    # Extract best solution
    side = best_x[0]
    pos = best_x[1:].reshape(11, 2)
    
    # Ensure side is at least sqrt(11)
    side = max(side, np.sqrt(11))
    
    # Build inner hex data: (x, y, angle_deg)
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = pos[i, 0]
        inner_hex_data[i, 1] = pos[i, 1]
        inner_hex_data[i, 2] = 0.0  # angle
    
    # Outer hex: centered at origin
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    try:
        inner, outer, side = _hexagon_packing_11_inner()
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback: simple grid
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
        side = 8.0
        inner, outer, side = inner_hex_data, outer_hex_data, side
    
    # Final validity check and fix if needed
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
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
        side = 8.0
        inner, outer, side = inner_hex_data, outer_hex_data, side
    
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
