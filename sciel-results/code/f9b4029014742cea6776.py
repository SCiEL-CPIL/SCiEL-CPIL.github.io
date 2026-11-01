"""Improved program using vectorized simulated annealing + Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

# Precompute unit hexagon vertices and normals
_V1 = np.column_stack([np.cos(np.radians(60 * np.arange(6))), np.sin(np.radians(60 * np.arange(6)))])
_N1 = np.column_stack([np.cos(np.radians(60 * np.arange(6))), np.sin(np.radians(60 * np.arange(6)))])

def _all_pair_separations(centers):
    """
    Vectorized computation of minimum separation for all pairs of unit hexagons.
    centers: list of [x, y, angle_deg]
    Returns array of shape (N, N) with min_sep[i,j] for i<j (symmetric).
    """
    n = len(centers)
    min_seps = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dtheta = np.radians(centers[j][2] - centers[i][2])
            
            # Hex1 vertices (unit, angle 0)
            v1 = _V1  # (6, 2)
            # Hex2 vertices (unit, rotated by dtheta)
            ct, st = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[ct, -st], [st, ct]])
            v2 = _V1 @ R.T  # (6, 2)
            
            # Axes: normals of both hexagons
            n1 = _N1  # (6, 2)
            n2 = _N1 @ R.T  # (6, 2)
            axes = np.vstack([n1, n2])  # (12, 2)
            
            # Project vertices onto each axis
            proj1 = v1 @ axes.T  # (6, 12)
            proj2 = v2 @ axes.T + (dx * axes[:, 0] + dy * axes[:, 1])  # (6, 12)
            
            sep1 = proj1.min(axis=0) - proj2.max(axis=0)  # (12,)
            sep2 = proj2.min(axis=0) - proj1.max(axis=0)  # (12,)
            seps = np.maximum(sep1, sep2)
            
            min_sep = seps.min()
            min_seps[i, j] = min_sep
            min_seps[j, i] = min_sep
    
    return min_seps

def _required_side(centers, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    oa = np.radians(outer_angle_deg)
    cos_oa, sin_oa = np.cos(oa), np.sin(oa)
    
    max_extent = 0.0
    for (px, py, pa) in centers:
        a = np.radians(pa)
        v_angles = a + np.radians(60 * np.arange(6))
        vx = px + np.cos(v_angles)
        vy = py + np.sin(v_angles)
        
        rx = vx - outer_center[0]
        ry = vy - outer_center[1]
        
        rx_local = rx * cos_oa - ry * sin_oa
        ry_local = rx * sin_oa + ry * cos_oa
        
        projs = rx_local[:, None] * _N1[None, :, 0] + ry_local[:, None] * _N1[None, :, 1]
        max_extent = max(max_extent, np.max(np.abs(projs)))
    
    return max_extent / (np.sqrt(3) / 2)

def _energy(centers, outer_center, outer_angle_deg):
    """
    Compute energy: outer_side + penalty for overlaps.
    Lower is better. Overlaps contribute quadratic penalty.
    """
    side = _required_side(centers, outer_center, outer_angle_deg)
    
    min_seps = _all_pair_separations(centers)
    # Get upper triangle
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if min_seps[i, j] < 0:
                penalty += min_seps[i, j] ** 2
    
    return side + 50.0 * penalty

def _is_valid(centers, outer_center, outer_angle_deg):
    """Check if configuration is valid (no overlaps, all contained)."""
    side = _required_side(centers, outer_center, outer_angle_deg)
    min_seps = _all_pair_separations(centers)
    for i in range(N):
        for j in range(i + 1, N):
            if min_seps[i, j] < -1e-9:
                return False
    return True

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration - 1 center + 6 ring + 5 more."""
    # Hexagonal lattice: spacing for non-overlapping unit hexagons
    # Two unit hexagons (flat-top) don't overlap if center distance >= 2*sqrt(3) (for same orientation)
    # But with 12 hexagons, we need a compact arrangement
    
    # Use a "flower" pattern: 1 center + 6 surrounding + 5 more
    s3 = np.sqrt(3)
    # Distance between centers of adjacent non-overlapping hexagons (same orientation)
    # For flat-top hexagons: min distance = 2 * s3 * cos(30°) = 2 * s3 * s3/2 = 3
    # Actually for pointy-top: min dist = 2 * s3
    # Let's use pointy-top (vertex at top)
    # Adjacent hex centers: dx = s3, dy = 1.5 (for one direction)
    # Or dx = 0, dy = 2 (vertical neighbors)
    
    # Let me use the standard hex grid spacing:
    # For unit hexagons (circumradius = 1), inradius = s3/2
    # Two hexagons side-by-side (horizontal): center dist = 2 * s3
    # Two hexagons at 60°: center dist = 2 * s3
    
    d = 2.0 * s3  # ~3.464
    
    positions = [
        (0, 0),
        (d, 0), (-d, 0),
        (d/2, d * s3/2), (-d/2, d * s3/2),
        (d/2, -d * s3/2), (-d/2, -d * s3/2),
        # Second ring - pick 5 from the 6 possible
        (2*d, 0),
        (3*d/2, d * s3/2),
        (-3*d/2, d * s3/2),
        (3*d/2, -d * s3/2),
        (-3*d/2, -d * s3/2),
        (0, d * s3),
    ]
    
    centers = []
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.8
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = []
    for i in range(N):
        x = rng.uniform(-4, 4)
        y = rng.uniform(-4, 4)
        a = rng.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    return _required_side(centers, outer_center, outer_angle)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = _required_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-7:
        return False
    
    min_seps = _all_pair_separations(centers)
    for i in range(N):
        for j in range(i + 1, N):
            if min_seps[i, j] < -1e-7:
                return False
    return True

def simulated_annealing(centers, outer_center, outer_angle, time_limit=20.0, seed=0):
    """
    Run simulated annealing to minimize the energy (side + overlap penalty).
    Returns (centers, outer_center, outer_angle, side) for best valid solution found,
    or the best energy solution if no valid one found.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Current state
    cur_centers = [list(c) for c in centers]
    cur_oc = list(outer_center)
    cur_oa = outer_angle
    
    cur_energy = _energy(cur_centers, cur_oc, cur_oa)
    
    # Best state (by energy)
    best_centers = [list(c) for c in cur_centers]
    best_oc = list(cur_oc)
    best_oa = cur_oa
    best_energy = cur_energy
    
    # Best valid state
    best_valid_side = float('inf')
    best_valid_centers = None
    best_valid_oc = None
    best_valid_oa = None
    
    if _is_valid(cur_centers, cur_oc, cur_oa):
        best_valid_side = _required_side(cur_centers, cur_oc, cur_oa)
        best_valid_centers = [list(c) for c in cur_centers]
        best_valid_oc = list(cur_oc)
        best_valid_oa = cur_oa
    
    # Annealing schedule
    T_start = 0.5
    T_end = 0.001
    n_steps = 50000  # target number of steps
    # We'll adjust based on time
    
    for step in range(n_steps):
        elapsed = time.time() - start
        if elapsed > time_limit - 0.5:
            break
        
        # Temperature
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        # Generate perturbation
        new_centers = [list(c) for c in cur_centers]
        new_oc = list(cur_oc)
        new_oa = cur_oa
        
        # Perturb a random hexagon
        idx = rng.randint(N)
        scale = 0.1 * (1 + T)
        new_centers[idx][0] += rng.normal(0, scale)
        new_centers[idx][1] += rng.normal(0, scale)
        new_centers[idx][2] += rng.normal(0, scale * 2)
        
        # Occasionally perturb outer
        if rng.random() < 0.3:
            new_oc[0] += rng.normal(0, 0.05)
            new_oc[1] += rng.normal(0, 0.05)
            new_oa += rng.normal(0, 0.05)
        
        # Occasionally do a larger move
        if rng.random() < 0.05:
            idx2 = rng.randint(N)
            new_centers[idx2][0] += rng.normal(0, 0.5)
            new_centers[idx2][1] += rng.normal(0, 0.5)
        
        new_energy = _energy(new_centers, new_oc, new_oa)
        
        # Accept?
        delta = new_energy - cur_energy
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur_centers = new_centers
            cur_oc = new_oc
            cur_oa = new_oa
            cur_energy = new_energy
            
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_centers = [list(c) for c in cur_centers]
                best_oc = list(cur_oc)
                best_oa = cur_oa
            
            # Check validity periodically
            if step % 100 == 0 and cur_energy < best_valid_side * 0.95:
                if _is_valid(cur_centers, cur_oc, cur_oa):
                    s = _required_side(cur_centers, cur_oc, cur_oa)
                    if s < best_valid_side:
                        best_valid_side = s
                        best_valid_centers = [list(c) for c in cur_centers]
                        best_valid_oc = list(cur_oc)
                        best_valid_oa = cur_oa
        
        # Check time more frequently
        if step % 500 == 0:
            if time.time() - start > time_limit - 0.5:
                break
    
    # Final check
    if _is_valid(best_centers, best_oc, best_oa):
        s = _required_side(best_centers, best_oc, best_oa)
        if s < best_valid_side:
            best_valid_side = s
            best_valid_centers = [list(c) for c in best_centers]
            best_valid_oc = list(best_oc)
            best_valid_oa = best_oa
    
    if best_valid_centers is not None:
        return best_valid_centers, best_valid_oc, best_valid_oa, best_valid_side
    else:
        return best_centers, best_oc, best_oa, _required_side(best_centers, best_oc, best_oa)

def nelder_mead_polish(centers, outer_center, outer_angle, time_limit=10.0):
    """
    Use Nelder-Mead to polish the best valid configuration.
    Only optimizes positions (not rotations) to keep things simple and fast.
    """
    start = time.time()
    n = N
    
    # Parameterize: positions only (2N params), keep angles and outer fixed
    def obj(params):
        c = [list(centers[i]) for i in range(n)]
        for i in range(n):
            c[i][0] = params[2*i]
            c[i][1] = params[2*i+1]
        # Check validity - if invalid, return large penalty
        min_seps = _all_pair_separations(c)
        for i in range(n):
            for j in range(i+1, n):
                if min_seps[i,j] < -1e-6:
                    return 100.0 + 10.0 * min_seps[i,j]**2
        return _required_side(c, outer_center, outer_angle)
    
    x0 = np.zeros(2*n)
    for i in range(n):
        x0[2*i] = centers[i][0]
        x0[2*i+1] = centers[i][1]
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
    except Exception:
        return centers, outer_center, outer_angle, _required_side(centers, outer_center, outer_angle)
    
    if time.time() - start > time_limit:
        return centers, outer_center, outer_angle, _required_side(centers, outer_center, outer_angle)
    
    new_centers = [list(c) for c in centers]
    for i in range(n):
        new_centers[i][0] = result.x[2*i]
        new_centers[i][1] = result.x[2*i+1]
    
    if _is_valid(new_centers, outer_center, outer_angle):
        side = _required_side(new_centers, outer_center, outer_angle)
        return new_centers, outer_center, outer_angle, side
    else:
        return centers, outer_center, outer_angle, _required_side(centers, outer_center, outer_angle)

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Multiple initial configurations
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(999)),
        ("random4", initial_config_random(777)),
    ]
    
    # For each config, run simulated annealing
    for idx, (name, config) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        time_limit = min(18, DEADLINE - elapsed - 8)
        if time_limit < 4:
            break
        
        try:
            centers = [list(c) for c in config]
            outer_center = [0.0, 0.0]
            outer_angle = 0.0
            
            opt_centers, opt_oc, opt_ao, opt_side = simulated_annealing(
                centers, outer_center, outer_angle,
                time_limit=time_limit, seed=1000 + idx
            )
            
            # Validate
            if check_validity(opt_centers, tuple(opt_oc), opt_side, opt_ao):
                if opt_side < best_score:
                    best_score = opt_side
                    best_result = (opt_centers, tuple(opt_oc), opt_side, opt_ao)
        except Exception:
            pass
    
    # Polish the best result with Nelder-Mead
    if best_result is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            time_limit = min(12, DEADLINE - elapsed - 3)
            if time_limit > 3:
                try:
                    bc, boc, bs, ba = best_result
                    pc, poc, ps, pa = nelder_mead_polish(
                        bc, boc, ba,
                        time_limit=time_limit
                    )
                    if check_validity(pc, tuple(poc), ps, pa):
                        if ps < best_score:
                            best_score = ps
                            best_result = (pc, tuple(poc), ps, pa)
                except Exception:
                    pass
    
    # Second round: run SA from the best result with different seed
    if best_result is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 8:
            time_limit = min(10, DEADLINE - elapsed - 4)
            if time_limit > 3:
                try:
                    bc, boc, bs, ba = best_result
                    pc, poc, ps, pa = simulated_annealing(
                        bc, list(boc), ba,
                        time_limit=time_limit, seed=9999
                    )
                    if check_validity(pc, tuple(poc), ps, pa):
                        if ps < best_score:
                            best_score = ps
                            best_result = (pc, tuple(poc), ps, pa)
                except Exception:
                    pass
    
    # Final polish
    if best_result is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 3:
            time_limit = min(5, DEADLINE - elapsed - 1)
            if time_limit > 2:
                try:
                    bc, boc, bs, ba = best_result
                    pc, poc, ps, pa = nelder_mead_polish(
                        bc, boc, ba,
                        time_limit=time_limit
                    )
                    if check_validity(pc, tuple(poc), ps, pa):
                        if ps < best_score:
                            best_score = ps
                            best_result = (pc, tuple(poc), ps, pa)
                except Exception:
                    pass
    
    # Fallback: return best found or a valid default
    if best_result is not None:
        return best_result
    
    # Default valid packing (larger outer hexagon)
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 4.0 * np.cos(theta)
        y = 4.0 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
# EVOLVE-BLOCK-END
