"""Morse cluster global minima: lattice-based initialization with multi-restart relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    # FCC lattice vectors
    a = spacing * 0.7071  # half diagonal for unit spacing
    # Generate points in a cube and filter
    # Estimate needed radius
    # FCC has 4 atoms per cubic cell of side a*sqrt(2)
    # Density: 4 / (a*sqrt(2))^3 = 4/(2*sqrt(2)*a^3)
    # Volume needed: n / density
    # Radius: (3*V/(4*pi))^(1/3)
    cell_vol = (a * np.sqrt(2)) ** 3
    density = 4.0 / cell_vol
    vol_needed = n / density
    r_est = (3.0 * vol_needed / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_est *= 1.2  # add margin
    
    # Generate FCC points
    points = []
    # FCC: corners + face centers
    # Generate in a grid
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # Corner
                p = np.array([i, j, k]) * a * np.sqrt(2)
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                # Face centers
                if i < extent and j < extent:
                    p2 = np.array([i + 0.5, j + 0.5, k]) * a * np.sqrt(2)
                    if np.linalg.norm(p2) <= r_est:
                        points.append(p2)
                if i < extent and k < extent:
                    p3 = np.array([i + 0.5, j, k + 0.5]) * a * np.sqrt(2)
                    if np.linalg.norm(p3) <= r_est:
                        points.append(p3)
                if j < extent and k < extent:
                    p4 = np.array([i, j + 0.5, k + 0.5]) * a * np.sqrt(2)
                    if np.linalg.norm(p4) <= r_est:
                        points.append(p4)
    
    points = np.array(points)
    if len(points) == 0:
        # Fallback: random
        rng = np.random.default_rng(42)
        points = rng.uniform(-r_est, r_est, (n, 3))
        return points
    
    # Remove duplicates and sort by distance from origin
    # Use a simple dedup
    unique_points = []
    for p in points:
        is_dup = False
        for q in unique_points:
            if np.linalg.norm(p - q) < 0.1 * a:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)
    
    unique_points = np.array(unique_points)
    # Sort by distance from origin
    dists = np.linalg.norm(unique_points, axis=1)
    idx = np.argsort(dists)
    unique_points = unique_points[idx]
    
    if len(unique_points) >= n:
        return unique_points[:n]
    else:
        # Pad with random points if needed
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(unique_points), 3))
        return np.vstack([unique_points, extra])


def _bcp_points(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    a = spacing * 0.5774  # for BCC, nearest neighbor distance = a*sqrt(3)/2
    # Adjust: we want nearest neighbor ~ spacing
    # BCC: nearest neighbor = a*sqrt(3)/2, so a = 2*spacing/sqrt(3)
    a = 2.0 * spacing / np.sqrt(3.0)
    
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # Corner
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                # Body center
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    
    points = np.array(points)
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = rng.uniform(-r_est, r_est, (n, 3))
        return points
    
    # Dedup
    unique_points = []
    for p in points:
        is_dup = False
        for q in unique_points:
            if np.linalg.norm(p - q) < 0.1 * a:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)
    
    unique_points = np.array(unique_points)
    dists = np.linalg.norm(unique_points, axis=1)
    idx = np.argsort(dists)
    unique_points = unique_points[idx]
    
    if len(unique_points) >= n:
        return unique_points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(unique_points), 3))
        return np.vstack([unique_points, extra])


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice centered at origin."""
    a = spacing
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = rng.uniform(-r_est, r_est, (n, 3))
        return points
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _random_compact(n, spacing=1.0, seed=0):
    """Generate n random points in a compact sphere with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fill remaining randomly, allowing closer distances
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points)


def _perturb(positions, magnitude=0.1):
    """Add small random perturbation to positions."""
    rng = np.random.default_rng()
    return positions + rng.uniform(-magnitude, magnitude, positions.shape)


def construct(instance):
    """Find low-energy Morse cluster configurations via multi-restart lattice-based search."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate per-instance time budget
    # We have 6 instances total, ~285s total
    # Give each instance roughly 45s, but adjust based on call order
    per_instance_budget = 45.0
    # First instances get slightly more time to establish good results
    if _instance_call_count[0] <= 2:
        per_instance_budget = 50.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
    
    instance_deadline = time.time() + per_instance_budget
    
    # Generate multiple starting configurations from different lattices
    start_configs = []
    
    # FCC
    try:
        fcc = _fcc_points(size, spacing=1.0)
        start_configs.append(fcc)
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _bcc_points(size, spacing=1.0)
        start_configs.append(bcc)
    except Exception:
        pass
    
    # Simple cubic
    try:
        sc = _sc_points(size, spacing=1.0)
        start_configs.append(sc)
    except Exception:
        pass
    
    # Random compact with different seeds
    for seed in [0, 1, 2, 3, 4]:
        try:
            rc = _random_compact(size, spacing=1.0, seed=seed)
            start_configs.append(rc)
        except Exception:
            pass
    
    # Perturbed versions of best lattice starts
    if len(start_configs) > 0:
        for i in range(min(3, len(start_configs))):
            try:
                pert = _perturb(start_configs[i], magnitude=0.05)
                start_configs.append(pert)
            except Exception:
                pass
    
    # Ensure we have at least some starts
    if len(start_configs) == 0:
        start_configs.append(_random_compact(size, spacing=1.0, seed=0))
    
    # Relax each starting configuration, keeping the best
    best_energy = np.inf
    best_positions = None
    
    for idx, config in enumerate(start_configs):
        # Check time
        if time.time() > instance_deadline - 2.0:
            break
        
        try:
            energy, positions = relax(config, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, do additional perturbations of the best result
    if best_positions is not None and time.time() < instance_deadline - 5.0:
        rng = np.random.default_rng(12345)
        for _ in range(10):
            if time.time() > instance_deadline - 3.0:
                break
            try:
                perturbed = best_positions + rng.uniform(-0.08, 0.08, best_positions.shape)
                energy, positions = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # Final safety check: ensure minimum distance constraint
    if best_positions is None:
        # Fallback: return a simple lattice
        best_positions = _sc_points(size, spacing=1.0)
    
    # Verify minimum distance
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.5:
                # Separate the two atoms
                mid = 0.5 * (best_positions[i] + best_positions[j])
                direction = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                    norm = 1.0
                direction /= norm
                offset = 0.26 * direction
                best_positions[i] = mid + offset
                best_positions[j] = mid - offset
    
    return best_positions
# EVOLVE-BLOCK-END
