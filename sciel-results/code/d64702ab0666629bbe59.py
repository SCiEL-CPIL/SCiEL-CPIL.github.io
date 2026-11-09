"""Morse cluster global minima search with diverse initializations and time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 310.0  # 20s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster by taking the n closest FCC lattice points to origin."""
    points = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                # FCC lattice: positions where (i+j+k) is even, scaled so nearest neighbor = spacing
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * spacing * 0.7071
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                p = np.array([i, j, k]) * spacing * 0.577  # BCC nearest neighbor ~ 0.577*2 = 1.15, adjust
                points.append(p)
                # Add body center
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing * 0.577
                points.append(pc)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _build_icosahedral_shells(n, spacing=1.0):
    """Build cluster using icosahedral shell geometry."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm_ico = np.linalg.norm(ico12[0])
    ico12 /= norm_ico
    
    points = [np.zeros(3)]
    
    # Shell radii and point counts for icosahedral shells
    shells = [
        (0.0, 1),      # center
        (1.0, 12),     # first icosahedral shell
        (1.633, 30),   # second shell (approx)
        (2.0, 56),     # third shell (approx)
        (2.449, 92),   # fourth shell (approx)
    ]
    
    for radius, count in shells:
        if radius == 0.0:
            points.append(np.zeros(3))
            continue
        for v in ico12:
            if len(points) >= n:
                break
            points.append(v * radius)
        # For shells with more than 12 points, add more points on the shell
        if count > 12 and len(points) < n:
            rng = np.random.default_rng(radius * 1000)
            while len(points) < min(n, len(points) + count - 12):
                p = rng.normal(0, 1, 3)
                p = p / np.linalg.norm(p) * radius
                # Check minimum distance
                if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                    points.append(p)
    
    # If still not enough, add more shells
    shell_radius = 3.0
    while len(points) < n:
        rng = np.random.default_rng(int(shell_radius * 1000))
        added = 0
        for _ in range(100):
            if len(points) >= n:
                break
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
                added += 1
        if added == 0:
            shell_radius += 0.5
        else:
            shell_radius += 0.3
    
    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 10000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.65 * spacing for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _build_greedy_placement(n, spacing=1.0, seed=123):
    """Build a cluster with good local packing using greedy placement."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    radius = spacing * n ** (1.0/3.0) * 0.9
    
    for _ in range(n - 1):
        best_p = None
        best_score = -1
        for _ in range(30):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            min_dist = min(np.linalg.norm(p - q) for q in points)
            if min_dist < 0.55 * spacing:
                continue
            score = min_dist
            if score > best_score:
                best_score = score
                best_p = p
        if best_p is not None:
            points.append(best_p)
        else:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points[:n])

def _build_spherical_shells(n, spacing=1.0):
    """Build cluster using concentric spherical shells with optimal packing."""
    points = [np.zeros(3)]
    
    # Approximate shell radii for optimal packing
    shell_configs = [
        (1.0, 12),
        (1.633, 30),
        (2.0, 56),
        (2.449, 92),
        (2.828, 140),
    ]
    
    for radius, count in shell_configs:
        if len(points) >= n:
            break
        # Use Fibonacci sphere for even distribution
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        for i in range(count):
            if len(points) >= n:
                break
            z = 1.0 - 2.0 * i / max(1, count - 1)
            r = np.sqrt(1.0 - z * z)
            theta = 2.0 * np.pi * i / golden
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.45 * spacing for q in points):
                points.append(p)
    
    # Fill remaining
    shell_radius = 3.2
    while len(points) < n:
        rng = np.random.default_rng(int(shell_radius * 1000))
        added = 0
        for _ in range(200):
            if len(points) >= n:
                break
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
                added += 1
        if added == 0:
            shell_radius += 0.4
        else:
            shell_radius += 0.25
    
    return np.array(points[:n])

def _ensure_min_distance(pos, min_d=0.5, max_iter=20):
    """Ensure all pairs are at least min_d apart by nudging."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0, 0])
                    diff /= np.linalg.norm(diff)
                    shift = (min_d - d) * 0.5 + 0.01
                    pos[i] += diff * shift
                    pos[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return pos

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check time budget
    tl = _time_left()
    if tl < 5.0:
        # Very little time left, just return a quick result
        pos = _build_random_compact(size, spacing=1.0, seed=size)
        try:
            _, pos = relax(pos, potential, parameter)
        except Exception:
            pass
        return _ensure_min_distance(pos)
    
    # Generate multiple diverse starting configurations
    start_configs = []
    
    builders = [
        lambda: _build_fcc_cluster(size, 1.0),
        lambda: _build_bcc_cluster(size, 1.0),
        lambda: _build_icosahedral_shells(size, 1.0),
        lambda: _build_random_compact(size, 1.0, seed=size),
        lambda: _build_greedy_placement(size, 1.0, seed=size + 7),
        lambda: _build_spherical_shells(size, 1.0),
        lambda: _build_random_compact(size, 1.0, seed=size + 100),
        lambda: _build_fcc_cluster(size, 1.0) + np.random.default_rng(size).normal(0, 0.1, (size, 3)),
    ]
    
    for builder in builders:
        try:
            cfg = builder()
            if len(cfg) == size:
                start_configs.append(cfg)
        except Exception:
            pass
    
    if not start_configs:
        start_configs = [_build_random_compact(size, 1.0, seed=size)]
    
    # Determine how many relaxations we can afford
    tl = _time_left()
    est_per_relax = 4.0 if size <= 50 else 6.0
    n_attempts = max(1, min(len(start_configs), int(tl / (est_per_relax * 1.2))))
    
    best_energy = float('inf')
    best_pos = start_configs[0].copy()
    
    for i in range(n_attempts):
        tl = _time_left()
        if tl < 8.0:
            break
        
        config = start_configs[i].copy()
        
        # Do perturbation restarts
        rng = np.random.default_rng(i * 1000 + size)
        n_restarts = 2 if size <= 50 else 1
        max_restart_time = tl * 0.7 / n_restarts
        
        for r in range(n_restarts):
            tl = _time_left()
            if tl < 5.0:
                break
            
            if r == 0:
                p = config.copy()
            else:
                p = config + rng.normal(0, 0.12, config.shape)
            
            try:
                energy, relaxed = relax(p, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Final verification: ensure minimum distance constraint
    pos = _ensure_min_distance(best_pos)
    
    return pos
# EVOLVE-BLOCK-END
