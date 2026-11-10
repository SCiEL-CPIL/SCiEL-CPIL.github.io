# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _all_vertices(cfg):
    """Compute all 11*6 = 66 vertices. Returns (11, 6, 2)."""
    cx = cfg[:, 0]
    cy = cfg[:, 1]
    a = np.radians(cfg[:, 2])
    angles = a[:, None] + np.radians(60 * np.arange(6))[None, :]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    vx = cx[:, None] + cos_a
    vy = cy[:, None] + sin_a
    return np.stack([vx, vy], axis=-1)


def _edge_normals(verts):
    """Compute 6 edge normals per hexagon. verts (11,6,2) -> normals (11,6,2)."""
    edges = np.roll(verts, -1, axis=1) - verts
    normals = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return normals / norms


def _separated_all(cfg, tol=1e-6):
    """Check all pairs of hexagons for separation using 12 edge normals."""
    verts = _all_vertices(cfg)
    n1 = _edge_normals(verts)
    n_pairs = 11
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            v1 = verts[i]
            v2 = verts[j]
            sep = False
            # Check normals from hex i
            for k in range(6):
                n = n1[i, k]
                p1 = v1 @ n
                p2 = v2 @ n
                if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                    sep = True
                    break
            if sep:
                continue
            # Check normals from hex j
            for k in range(6):
                n = n1[j, k]
                p1 = v1 @ n
                p2 = v2 @ n
                if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                    sep = True
                    break
            if not sep:
                return False
    return True


def _is_valid_fast(cfg, side, outer_center, outer_angle):
    """Fast validity check: separation + containment."""
    if not _separated_all(cfg):
        return False
    if not _check_containment(cfg, outer_center, outer_angle, side):
        return False
    return True


def _compute_outer_side(cfg, outer_angle_deg=0.0):
    """Compute minimum outer hex side to contain all inner hexagons."""
    verts = _all_vertices(cfg)
    a = np.radians(outer_angle_deg)
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = verts.reshape(-1, 2) @ n
        max_proj = max(max_proj, np.max(proj))
    return 2.0 * max_proj / np.sqrt(3)


def _check_containment(cfg, outer_center, outer_angle, side):
    """Check if all inner hexagons are contained in the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    inner_verts = _all_vertices(cfg)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        for h in range(11):
            for v_idx in range(6):
                p = inner_verts[h, v_idx]
                cross_val = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True


def _local_search(cfg, angle, rng, time_budget, start_step=0.3):
    """Simulated-annealing style local search to minimize outer side length."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side(best_cfg, angle)
    step_size = start_step
    current_side = best_side
    current_cfg = best_cfg.copy()
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        new_cfg = current_cfg.copy()
        n_perturb = int(rng.integers(1, 5))
        indices = rng.choice(11, size=n_perturb, replace=False)
        new_cfg[indices, :2] += rng.normal(0, step_size, size=(n_perturb, 2))
        
        if _separated_all(new_cfg):
            s = _compute_outer_side(new_cfg, angle)
            if s < current_side:
                current_cfg = new_cfg
                current_side = s
                if s < best_side - 1e-10:
                    best_side = s
                    best_cfg = new_cfg.copy()
                step_size = max(0.01, step_size * 0.98)
            else:
                progress = (time.time() - start_t) / time_budget
                T = max(0.01, (1 - progress) * 0.5)
                if rng.random() < np.exp(-(s - current_side) / T):
                    current_cfg = new_cfg
                    current_side = s
                    step_size = max(0.01, step_size * 0.99)
                else:
                    step_size = min(1.0, step_size * 1.03)
        else:
            step_size = min(1.0, step_size * 1.05)
    
    return best_cfg, best_side


def _optimize_rigid_cluster(rng, time_budget):
    """
    Optimize a rigid hex cluster by varying inter-ring distances and global rotation.
    Uses parameterized search over structural parameters.
    """
    sqrt3 = np.sqrt(3)
    best_side = float('inf')
    best_cfg = None
    best_angle = 0.0
    
    start_t = time.time()
    n_iter = 0
    
    while time.time() - start_t < time_budget:
        n_iter += 1
        # Parameters: r1 (inner ring radius), r2 (outer ring radius), 
        # angle offsets for outer ring, global rotation
        r1 = 2.0 * sqrt3 + rng.normal(0, 0.15)
        r2_base = r1 * (1.4 + rng.uniform(-0.1, 0.3))
        
        # Try different outer ring patterns
        pattern = int(rng.integers(0, 4))
        global_rot = rng.uniform(0, np.pi/3)
        
        config = np.zeros((11, 3))
        config[0] = [0, 0, 0]
        
        # Inner ring of 6
        for i in range(6):
            a = global_rot + 2 * np.pi * i / 6
            config[1+i] = [r1 * np.cos(a), r1 * np.sin(a), 0]
        
        # Outer ring of 4 - different patterns
        if pattern == 0:
            # Square-like
            outer_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
            r2s = [r2_base + rng.normal(0, 0.1) for _ in range(4)]
        elif pattern == 1:
            # Rhombus
            outer_angles = [0, np.pi/3, np.pi, 4*np.pi/3]
            r2s = [r2_base + rng.normal(0, 0.1) for _ in range(4)]
        elif pattern == 2:
            # Asymmetric
            outer_angles = [rng.uniform(0, np.pi/6), rng.uniform(np.pi/2, 2*np.pi/3),
                          rng.uniform(np.pi, 7*np.pi/6), rng.uniform(4*np.pi/3, 5*np.pi/3)]
            r2s = [r2_base + rng.normal(0, 0.15) for _ in range(4)]
        else:
            # Tight cluster
            outer_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
            r2s = [r2_base * 0.95 + rng.normal(0, 0.1) for _ in range(4)]
        
        for i in range(4):
            a = global_rot + outer_angles[i]
            config[7+i] = [r2s[i] * np.cos(a), r2s[i] * np.sin(a), 0]
        
        if not _separated_all(config):
            continue
        
        # Find best outer angle
        best_a = 0.0
        best_s = float('inf')
        for angle in np.arange(0, 30, 1.0):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        # Fine-tune angle
        for angle in np.arange(best_a - 1.0, best_a + 1.0, 0.1):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        if best_s < best_side:
            best_side = best_s
            best_cfg = config.copy()
            best_angle = best_a
    
    return best_cfg, best_side, best_angle


def _refine(cfg, angle, rng, time_budget):
    """Fine-grained coordinate descent refinement with multiple hexagon updates."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side(best_cfg, angle)
    step_size = 0.05
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        # Sometimes update multiple hexagons
        if rng.random() < 0.3:
            n_upd = int(rng.integers(2, 4))
            indices = rng.choice(11, size=n_upd, replace=False)
            new_cfg = best_cfg.copy()
            new_cfg[indices, :2] += rng.normal(0, step_size * 0.7, size=(n_upd, 2))
        else:
            idx = int(rng.integers(0, 11))
            new_cfg = best_cfg.copy()
            new_cfg[idx, :2] += rng.normal(0, step_size, size=2)
        
        if _separated_all(new_cfg):
            s = _compute_outer_side(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.001, step_size * 0.93)
            else:
                step_size = min(0.2, step_size * 1.05)
        else:
            step_size = min(0.2, step_size * 1.08)
    
    return best_cfg, best_side


def _initial_config_hexgrid():
    """Honeycomb: 1 center + 6 ring + 4 more."""
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


def _build_staggered_11():
    """Staggered 2-row configuration for compact packing."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    config = np.array([
        # Row 1: 6 hexagons
        [-2.5*dx, 0.0, 0.0],
        [-1.5*dx, 0.0, 0.0],
        [-0.5*dx, 0.0, 0.0],
        [0.5*dx, 0.0, 0.0],
        [1.5*dx, 0.0, 0.0],
        [2.5*dx, 0.0, 0.0],
        # Row 2: 5 hexagons staggered
        [-2.0*dx, dy, 0.0],
        [-1.0*dx, dy, 0.0],
        [0.0, dy, 0.0],
        [1.0*dx, dy, 0.0],
        [2.0*dx, dy, 0.0],
    ])
    return config


def _build_compact_cluster_11():
    """Compact 3x3-ish cluster with 2 extra."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    config = np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
        [0.5*dx, dy, 0.0],
        [-0.5*dx, dy, 0.0],
        [-dx, 0.0, 0.0],
        [-0.5*dx, -dy, 0.0],
        [0.5*dx, -dy, 0.0],
        [1.5*dx, 1.5*dy, 0.0],
        [-1.5*dx, 1.5*dy, 0.0],
        [0.0, 3.0*dy, 0.0],
        [0.0, -3.0*dy, 0.0],
    ])
    return config


def _build_linear_chain_11():
    """Linear chain with slight stagger."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    config = np.zeros((11, 3))
    for i in range(11):
        x = (i - 5) * dx
        y = (dy if i % 2 == 0 else -dy) * 0.3
        config[i] = [x, y, 0.0]
    return config


def _initial_config_ring(rng, n_ring1=6, n_ring2=4):
    """Ring-based initial config with random angles."""
    config = np.zeros((11, 3))
    config[0] = [0.0, 0.0, 0.0]
    r1 = 2.0 * np.sqrt(3)
    for i in range(n_ring1):
        a = 2 * np.pi * i / n_ring1 + rng.normal(0, 0.1)
        config[1 + i] = [r1 * np.cos(a), r1 * np.sin(a), 0.0]
    r2 = 3.5 * np.sqrt(3)
    for i in range(n_ring2):
        a = 2 * np.pi * i / n_ring2 + np.pi / n_ring2 + rng.normal(0, 0.1)
        config[1 + n_ring1 + i] = [r2 * np.cos(a), r2 * np.sin(a), 0.0]
    return config


def _initial_config_compact(rng):
    """Compact flower + 4 satellites in asymmetric positions."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    config = np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
        [0.5*dx, dy, 0.0],
        [-0.5*dx, dy, 0.0],
        [-dx, 0.0, 0.0],
        [-0.5*dx, -dy, 0.0],
        [0.5*dx, -dy, 0.0],
    ])
    # 4 satellites at asymmetric positions
    sat_positions = [
        [1.5*dx + rng.normal(0, 0.3), 0.5*dy + rng.normal(0, 0.3)],
        [-1.5*dx + rng.normal(0, 0.3), 0.5*dy + rng.normal(0, 0.3)],
        [0.5*dx + rng.normal(0, 0.3), 2.0*dy + rng.normal(0, 0.3)],
        [-0.5*dx + rng.normal(0, 0.3), -2.0*dy + rng.normal(0, 0.3)],
    ]
    for i, (sx, sy) in enumerate(sat_positions):
        config = np.vstack([config, [[sx, sy, 0.0]]])
    return config


def _rigid_cluster_search(rng, time_budget):
    """
    Structural approach: fix the topological contact graph, optimize only
    global rotation and a few inter-ring distances.
    """
    sqrt3 = np.sqrt(3)
    best_side = float('inf')
    best_params = None
    
    start_t = time.time()
    # Parameters: r1 (inner ring radius), r2 (outer ring radius), angle_offset, global_rotation
    # r1 = 2*sqrt(3) is the natural honeycomb distance
    # We'll search over small perturbations
    
    n_iter = 0
    while time.time() - start_t < time_budget:
        n_iter += 1
        # Random parameter set
        r1 = 2.0 * sqrt3 + rng.normal(0, 0.2)
        r2 = r1 * (1.5 + rng.normal(0, 0.15))
        angle_off = rng.uniform(0, np.pi/3)
        global_rot = rng.uniform(0, np.pi/6)
        
        # Build configuration
        config = np.zeros((11, 3))
        config[0] = [0, 0, 0]
        # Inner ring of 6
        for i in range(6):
            a = global_rot + 2 * np.pi * i / 6
            config[1+i] = [r1 * np.cos(a), r1 * np.sin(a), 0]
        # Outer ring of 4 (asymmetric)
        outer_angles = [angle_off, angle_off + np.pi/2, angle_off + np.pi, angle_off + 3*np.pi/2]
        for i in range(4):
            a = global_rot + outer_angles[i]
            config[7+i] = [r2 * np.cos(a), r2 * np.sin(a), 0]
        
        if not _separated_all(config):
            continue
        
        # Find best outer angle
        best_a = 0.0
        best_s = float('inf')
        for angle in np.arange(0, 30, 1.0):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        if best_s < best_side:
            best_side = best_s
            best_params = (r1, r2, angle_off, global_rot, best_a, config.copy())
    
    if best_params is None:
        return None, float('inf')
    
    _, _, _, _, best_a, best_cfg = best_params
    return best_cfg, best_side


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 575  # Safety margin
    
    rng = np.random.default_rng(42)
    best_side = float('inf')
    best_cfg = None
    best_angle = 0.0
    
    # Generate diverse initial configurations
    inits = [
        _initial_config_hexgrid(),
        _build_staggered_11(),
        _build_compact_cluster_11(),
    ]
    for _ in range(2):
        inits.append(_initial_config_ring(rng))
    for _ in range(2):
        inits.append(_initial_config_compact(rng))
    # Perturbed variants
    base = _initial_config_hexgrid()
    for _ in range(2):
        p = base.copy()
        p[:, :2] += rng.normal(0, 0.4, size=(11, 2))
        inits.append(p)
    
    # Evaluate initial configs
    for config in inits:
        if time.time() > deadline - 120:
            break
        
        if not _separated_all(config):
            continue
        
        # Find best outer angle
        best_a = 0.0
        best_s = float('inf')
        for angle in np.arange(0, 30, 1.0):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        # Fine-tune angle
        for angle in np.arange(best_a - 1.0, best_a + 1.0, 0.1):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        if best_s < best_side:
            best_side = best_s
            best_cfg = config.copy()
            best_angle = best_a
    
    # Rigid cluster optimization (structural search)
    remaining = deadline - time.time()
    if remaining > 120:
        t_budget = min(remaining * 0.25, 100)
        rigid_cfg, rigid_side, rigid_angle = _optimize_rigid_cluster(rng, t_budget)
        if rigid_cfg is not None and rigid_side < best_side:
            best_side = rigid_side
            best_cfg = rigid_cfg
            best_angle = rigid_angle
    
    # Local search on best config
    remaining = deadline - time.time()
    if remaining > 60 and best_cfg is not None:
        t_budget = min(remaining * 0.4, 150)
        cfg2, s2 = _local_search(best_cfg, best_angle, rng, t_budget)
        if s2 < best_side:
            best_side = s2
            best_cfg = cfg2
    
    # Refinement
    remaining = deadline - time.time()
    if remaining > 30 and best_cfg is not None:
        t_budget = min(remaining * 0.5, 100)
        best_cfg, best_side = _refine(best_cfg, best_angle, rng, t_budget)
    
    # Final angle sweep
    for angle in np.arange(best_angle - 2.0, best_angle + 2.0, 0.05):
        s = _compute_outer_side(best_cfg, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
    
    # Additional local search passes if time remains
    remaining = deadline - time.time()
    if remaining > 40 and best_cfg is not None:
        t_budget = min(remaining * 0.5, 120)
        cfg2, s2 = _local_search(best_cfg, best_angle, rng, t_budget, start_step=0.1)
        if s2 < best_side:
            best_side = s2
            best_cfg = cfg2
        
        remaining = deadline - time.time()
        if remaining > 20:
            t_budget = min(remaining * 0.6, 60)
            best_cfg, best_side = _refine(best_cfg, best_angle, rng, t_budget)
    
    # Final fine angle sweep
    for angle in np.arange(best_angle - 1.0, best_angle + 1.0, 0.02):
        s = _compute_outer_side(best_cfg, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment with safety margin
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
        for _ in range(1000):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                break
            best_side *= 1.0001
    
    return best_cfg, outer_hex_data, best_side


# EVOLVE-BLOCK-END
