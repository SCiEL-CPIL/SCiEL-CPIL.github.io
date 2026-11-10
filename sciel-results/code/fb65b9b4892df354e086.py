# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _all_vertices(inner_data):
    """Compute all vertices for all hexagons. Returns shape (11, 6, 2)."""
    n = len(inner_data)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
    return verts


def _edge_normals(verts):
    """Compute edge normals for all hexagons. Returns shape (n, 6, 2) where each row is a unit normal."""
    n = verts.shape[0]
    normals = np.zeros((n, 6, 2))
    for i in range(n):
        for j in range(6):
            e = verts[i, (j+1) % 6] - verts[i, j]
            nn = np.array([-e[1], e[0]])
            norm = np.linalg.norm(nn)
            if norm > 1e-10:
                normals[i, j] = nn / norm
    return normals


def _separated_fast(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays of shape (6,2)) are separated using all 12 edge normals."""
    # Check normals from hex 1
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    # Check normals from hex 2
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False


def _is_valid_fast(inner_data):
    """Check if all pairs of hexagons are separated."""
    n = len(inner_data)
    verts = _all_vertices(inner_data)
    for i in range(n):
        for j in range(i+1, n):
            if not _separated_fast(verts[i], verts[j]):
                return False
    return True


def _compute_outer_side(inner_data, outer_angle=0.0):
    """Compute minimum outer hex side to contain all inner hexagons."""
    a = np.radians(outer_angle)
    all_verts = _all_vertices(inner_data)  # shape (11, 6, 2)
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n  # shape (11, 6)
        max_proj = max(max_proj, np.max(proj))
    return 2.0 * max_proj / np.sqrt(3)


def _check_containment(inner_data, outer_center, outer_angle, side):
    """Check if all inner hexagons are contained in the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    all_verts = _all_vertices(inner_data)  # shape (11, 6, 2)
    for i in range(11):
        for j in range(6):
            p = all_verts[i, j]
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True


def _local_search(best_cfg, best_angle, deadline, rng):
    """Local search with random perturbations."""
    best_side = _compute_outer_side(best_cfg, best_angle)
    step_size = 0.3
    n_improvements = 0
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        new_cfg = best_cfg.copy()
        mask = rng.random(11) < 0.4
        n_mask = np.sum(mask)
        if n_mask > 0:
            new_cfg[mask, :2] += rng.normal(0, step_size, size=(n_mask, 2))
        
        if _is_valid_fast(new_cfg):
            s = _compute_outer_side(new_cfg, best_angle)
            if s < best_side - 1e-8:
                best_side = s
                best_cfg = new_cfg.copy()
                n_improvements += 1
                step_size = max(0.02, step_size * 0.95)
                if n_improvements % 100 == 0:
                    step_size *= 0.9
            else:
                step_size = min(0.6, step_size * 1.03)
    
    return best_cfg, best_side


def _refine(best_cfg, best_angle, deadline, rng):
    """High-precision refinement with small perturbations."""
    best_side = _compute_outer_side(best_cfg, best_angle)
    step_size = 0.05
    
    for it in range(10000):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, 11)
        new_cfg = best_cfg.copy()
        new_cfg[idx, :2] += rng.normal(0, step_size, size=2)
        
        if _is_valid_fast(new_cfg):
            s = _compute_outer_side(new_cfg, best_angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.005, step_size * 0.9)
            else:
                step_size = min(0.15, step_size * 1.05)
    
    return best_cfg, best_side


def _initial_config_hexgrid():
    """Honeycomb: 1 center + 6 ring + 4 outer."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    
    return np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
        [0.5*dx, dy, 0.0],
        [-0.5*dx, dy, 0.0],
        [-dx, 0.0, 0.0],
        [-0.5*dx, -dy, 0.0],
        [0.5*dx, -dy, 0.0],
        [1.5*dx, 1.5*dy, 0.0],
        [-1.5*dx, 1.5*dy, 0.0],
        [1.5*dx, -1.5*dy, 0.0],
        [-1.5*dx, -1.5*dy, 0.0],
    ])


def _initial_config_ring():
    """Ring configuration: center + 6 + 4 in a different arrangement."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    
    # Center
    config = [[0.0, 0.0, 0.0]]
    # 6 around center
    for i in range(6):
        angle = np.radians(60 * i)
        config.append([dx * np.cos(angle), dx * np.sin(angle), 0.0])
    # 4 more in outer ring
    for i in range(4):
        angle = np.radians(90 + 90 * i)
        r = 2.5 * dx
        config.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    
    return np.array(config)


def _initial_config_compact():
    """More compact initial configuration."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    
    # Try a tighter packing
    config = np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
        [0.5*dx, dy, 0.0],
        [-0.5*dx, dy, 0.0],
        [-dx, 0.0, 0.0],
        [-0.5*dx, -dy, 0.0],
        [0.5*dx, -dy, 0.0],
        [1.5*dx, 0.5*dy, 0.0],
        [-1.5*dx, 0.5*dy, 0.0],
        [1.5*dx, -0.5*dy, 0.0],
        [-1.5*dx, -0.5*dy, 0.0],
    ])
    return config


def _mechanical_settle(positions, outer_angle, deadline, rng, n_steps=5000):
    """
    Simulate hexagons as particles with repulsive forces and containment in outer hex.
    positions: (11, 2) array of centers
    Returns settled positions (11, 2).
    """
    pos = positions.copy().astype(float)
    sqrt3 = np.sqrt(3)
    
    # Precompute outer hex edge data for containment
    # Outer hex with side S, center at origin, angle outer_angle
    # Vertices at distance S from center
    # Edge normals point inward
    a_rad = np.radians(outer_angle)
    # The 6 inward normals of a regular hexagon centered at origin
    # For a hexagon with vertices at angles a + 60*i, the inward normals of edges
    # are at angles a + 60*i + 30 (pointing from center toward edge midpoint... actually inward)
    # Edge between vertex i and i+1: normal points inward
    # For vertex at angle theta, next at theta+60, the edge midpoint direction is theta+30
    # Inward normal is opposite to outward normal
    # Outward normal of edge i (between v_i and v_{i+1}) points in direction of angle a + 60*i + 30
    # Inward normal points in direction a + 60*i + 30 + 180 = a + 60*i + 210
    # But for containment, we need: for each edge, the projection of all points onto the
    # inward normal must be <= S (the apothem... wait)
    
    # Actually, for a regular hexagon with side length S, the apothem (distance from center to edge) is S*sqrt(3)/2
    # The containment condition: for each of the 6 edges, the maximum projection of all vertices
    # onto the outward normal must be <= apothem = S * sqrt(3)/2
    
    # Let's use the same approach as _compute_outer_side:
    # For each of 6 directions (outward normals), max_proj <= apothem
    # outward normal direction for edge i: angle = a_rad + 30 + 60*i degrees
    
    outward_normals = []
    for i in range(6):
        theta = a_rad + np.radians(30 + 60 * i)
        outward_normals.append(np.array([np.cos(theta), np.sin(theta)]))
    outward_normals = np.array(outward_normals)  # (6, 2)
    
    dt = 0.01
    damping = 0.92
    k_repel = 50.0
    k_contain = 200.0
    min_center_dist = 2.0 * sqrt3  # 2 * sqrt(3) is the minimum center-to-center distance for touching hexagons
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        forces = np.zeros_like(pos)
        
        # Repulsive forces between all pairs
        for i in range(11):
            for j in range(i+1, 11):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-6:
                    # Random tiny push
                    d = rng.normal(0, 0.01, size=2)
                    dist = np.linalg.norm(d)
                if dist < min_center_dist:
                    # Overlapping: repel
                    overlap = min_center_dist - dist
                    dir_vec = d / dist
                    f = k_repel * overlap
                    forces[i] += f * dir_vec
                    forces[j] -= f * dir_vec
        
        # Containment forces: push inward if any vertex would be outside
        # For each hexagon, check if any of its 6 vertices would violate containment
        # For a hexagon at position pos[i], its vertices are at pos[i] + unit_hex_vertices
        # We need: for each outward normal n_k, max over all vertices of (vertex . n_k) <= apothem
        # Equivalently: for each outward normal, (pos[i] . n_k) + max_vertex_offset . n_k <= apothem
        # The max_vertex_offset for a unit hexagon in direction n is the support function
        
        # Unit hexagon vertices (angle 0):
        unit_angles = np.radians(60 * np.arange(6))
        unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
        
        apothem_est = 5.0  # Will be computed from current best_side, but for simulation use a large value initially
        # Actually, we want to minimize the required apothem, so we should use a tight bound
        # Let's compute current required side and use that as the boundary
        # For simulation, we'll use a fixed boundary that's slightly larger than current best
        current_side_est = 6.0  # Rough estimate
        apothem = current_side_est * sqrt3 / 2.0
        
        for i in range(11):
            # For each outward normal, check if hexagon i violates
            for k in range(6):
                n = outward_normals[k]
                # Max projection of unit hex vertices onto n
                max_offset = np.max(unit_verts @ n)
                # Position projection
                pos_proj = pos[i] @ n
                # Total max projection
                total = pos_proj + max_offset
                if total > apothem:
                    # Push inward
                    excess = total - apothem
                    forces[i] -= k_contain * excess * n
        
        # Update velocities (implicit in force-based update with damping)
        pos += dt * forces
        # Apply damping by scaling position update... actually let's use velocity
        # Simpler: just do position update with damping on the step
        # The above already does pos += dt * forces, which is fine for small dt
        
    return pos


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_config_hexgrid(),
        _initial_config_ring(),
        _initial_config_compact(),
    ]
    
    best_cfg = None
    best_side = float('inf')
    best_angle = 0.0
    
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        
        # Try different outer angles
        for angle in np.arange(0, 30, 0.5):
            if time.time() > deadline:
                break
            s = _compute_outer_side(cfg, angle)
            if s < best_side:
                best_side = s
                best_cfg = cfg.copy()
                best_angle = angle
        
        # Local search from this configuration
        if best_cfg is not None and time.time() < deadline - 100:
            cfg_search, side_search = _local_search(best_cfg, best_angle, deadline, rng)
            if side_search < best_side:
                best_side = side_search
                best_cfg = cfg_search
    
    # Refinement phase
    if best_cfg is not None and time.time() < deadline - 50:
        best_cfg, best_side = _refine(best_cfg, best_angle, deadline, rng)
    
    # Mechanical settling: try to improve via physics simulation
    if best_cfg is not None and time.time() < deadline - 30:
        try:
            # Run mechanical simulation from current best
            settled_pos = _mechanical_settle(best_cfg[:, :2], best_angle, deadline, rng, n_steps=3000)
            settled_cfg = np.column_stack([settled_pos, np.zeros(11)])
            if _is_valid_fast(settled_cfg):
                s = _compute_outer_side(settled_cfg, best_angle)
                if s < best_side:
                    best_side = s
                    best_cfg = settled_cfg
        except Exception:
            pass
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment and adjust if needed
    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            for _ in range(200):
                if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                    break
                best_side *= 1.001
    
    if best_cfg is None:
        # Fallback
        best_cfg = _initial_config_hexgrid()
        best_angle = 0.0
        best_side = _compute_outer_side(best_cfg, best_angle)
        outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return best_cfg, outer_hex_data, best_side


# EVOLVE-BLOCK-END
