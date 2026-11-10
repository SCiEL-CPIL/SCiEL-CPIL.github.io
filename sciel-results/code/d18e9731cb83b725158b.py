# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays) are separated using all 12 edge normals."""
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


def _compute_outer_side_fast(inner_data, outer_angle=0.0):
    """Faster version: precompute all vertices once."""
    a = np.radians(outer_angle)
    # Precompute all 11*6 = 66 vertices
    all_verts = np.array([_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner_data])
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
    for hex_row in inner_data:
        v = _hex_vertices(hex_row[0], hex_row[1], 1.0, hex_row[2])
        for p in v:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i+1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True


def _is_valid_fast(cfg, min_dist=2.0*np.sqrt(3) - 1e-9):
    """Fast validity check using center distances for hexagons with same orientation.
    For flat-top hexagons with same angle, separation requires center distance >= 2*sqrt(3)."""
    # For hexagons with the same orientation, the minimum center distance for
    # non-overlapping is 2*sqrt(3) when aligned on the hex grid.
    # But for arbitrary positions, we need the full check.
    # This is a quick rejection: if all pairs have center dist >= min_dist, likely valid.
    # We'll do full check only if this passes.
    for i in range(11):
        for j in range(i+1, 11):
            d2 = (cfg[i,0]-cfg[j,0])**2 + (cfg[i,1]-cfg[j,1])**2
            if d2 < (min_dist - 0.5)**2:
                # Need full check
                v1 = _hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2])
                v2 = _hex_vertices(cfg[j,0], cfg[j,1], 1.0, cfg[j,2])
                if not _hex_separated(v1, v2):
                    return False
            # If d2 >= (min_dist-0.5)^2, they're likely separated
    return True


def _is_valid(cfg):
    """Full validity check."""
    for i in range(11):
        for j in range(i+1, 11):
            v1 = _hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2])
            v2 = _hex_vertices(cfg[j,0], cfg[j,1], 1.0, cfg[j,2])
            if not _hex_separated(v1, v2):
                return False
    return True


def _local_search(cfg, angle, rng, time_budget, start_step=0.3):
    """Local search to minimize outer side length."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    step_size = start_step
    
    start_t = time.time()
    it = 0
    while time.time() - start_t < time_budget:
        it += 1
        new_cfg = best_cfg.copy()
        # Randomly perturb a subset of hexagons
        mask = rng.random(11) < 0.35
        n_mask = np.sum(mask)
        if n_mask == 0:
            mask[rng.integers(0, 11)] = True
            n_mask = 1
        new_cfg[mask, :2] += rng.normal(0, step_size, size=(n_mask, 2))
        
        # Quick validity check
        if _is_valid_fast(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.015, step_size * 0.96)
            else:
                step_size = min(0.8, step_size * 1.03)
        else:
            step_size = min(0.8, step_size * 1.06)
    
    return best_cfg, best_side


def _refine(cfg, angle, rng, time_budget, start_step=0.08):
    """Fine-grained refinement of individual hexagon positions."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    step_size = start_step
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        idx = rng.integers(0, 11)
        new_cfg = best_cfg.copy()
        new_cfg[idx, :2] += rng.normal(0, step_size, size=2)
        
        if _is_valid_fast(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.002, step_size * 0.92)
            else:
                step_size = min(0.2, step_size * 1.05)
        else:
            step_size = min(0.2, step_size * 1.07)
    
    return best_cfg, best_side


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

    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3  # Horizontal center distance for touching flat-top hexagons
    dy = 1.5 * sqrt3  # Vertical offset for staggered rows

    # Initial honeycomb config: 1 center + 6 ring + 4 additional
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
        [1.5*dx, -1.5*dy, 0.0],
        [-1.5*dx, -1.5*dy, 0.0],
    ])

    best_side = float('inf')
    best_cfg = config.copy()
    best_angle = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Find best outer angle for initial config
    for angle in np.arange(0, 30, 0.5):
        s = _compute_outer_side_fast(config, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
            best_cfg = config.copy()

    # Fine-tune angle
    for angle in np.arange(best_angle - 0.5, best_angle + 0.5, 0.05):
        s = _compute_outer_side_fast(config, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
            best_cfg = config.copy()

    # Phase 2: Local search
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    if remaining > 60:
        t_budget = min(remaining * 0.5, 180)
        best_cfg, best_side = _local_search(best_cfg, best_angle, rng, t_budget)

    # Phase 3: Refine angle again
    if time.time() < deadline - 30:
        for angle in np.arange(best_angle - 1.0, best_angle + 1.0, 0.1):
            s = _compute_outer_side_fast(best_cfg, angle)
            if s < best_side:
                best_side = s
                best_angle = angle

    # Phase 4: Fine refinement
    remaining = deadline - time.time()
    if remaining > 30:
        t_budget = min(remaining * 0.7, 150)
        best_cfg, best_side = _refine(best_cfg, best_angle, rng, t_budget)

    # Phase 5: Try different initial configurations with random perturbations
    remaining = deadline - time.time()
    if remaining > 45:
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            # Perturb initial config
            perturbed = config.copy()
            perturbed[:, :2] += rng.normal(0, 1.0, size=(11, 2))
            
            if _is_valid(perturbed):
                # Find best angle for this config
                best_a = best_angle
                best_s = _compute_outer_side_fast(perturbed, best_a)
                for angle in np.arange(0, 30, 1.0):
                    s = _compute_outer_side_fast(perturbed, angle)
                    if s < best_s:
                        best_s = s
                        best_a = angle
                
                if best_s < best_side:
                    # Local search from here
                    t_budget = min((deadline - time.time()) * 0.3, 60)
                    if t_budget > 10:
                        cfg2, s2 = _local_search(perturbed, best_a, rng, t_budget)
                        if s2 < best_side:
                            best_side = s2
                            best_cfg = cfg2
                            best_angle = best_a

    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment and adjust if needed
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
        for _ in range(200):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                break
            best_side *= 1.001
    
    # Final high-precision polish pass
    if time.time() < deadline - 10:
        t_budget = min(deadline - time.time() - 5, 30)
        best_cfg, best_side = _refine(best_cfg, best_angle, rng, t_budget, start_step=0.02)
        
    # Final containment check and adjustment
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
        for _ in range(500):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                break
            best_side *= 1.0001
    
    return best_cfg, outer_hex_data, best_side


# EVOLVE-BLOCK-END
