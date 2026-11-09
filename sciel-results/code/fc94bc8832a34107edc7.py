"""Morse cluster global minima search with structured initializations and time budget."""
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
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _build_icosahedral_shell(n, spacing=1.0):
    """Build a cluster using icosahedral shell geometry."""
    # Icosahedron vertices (12)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    # Octahedron vertices (6) for inner shell
    oct6 = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    
    # Cuboctahedron (12) for another shell
    cubo12 = np.array([
        [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0],
        [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1],
        [0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1]
    ], dtype=float)
    cubo12 /= np.linalg.norm(cubo12[0])
    
    points = []
    # Center
    points.append(np.zeros(3))
    # Octahedron shell (6)
    for v in oct6:
        points.append(v * 0.8)
    # Icosahedron shell (12)
    for v in ico12:
        points.append(v * 1.5)
    # Cuboctahedron shell (12)
    for v in cubo12:
        points.append(v * 2.2)
    
    # If we need more, add additional shells
    shell_radius = 3.0
    while len(points) < n:
        # Add points on a spherical shell
        rng = np.random.default_rng(hash(tuple(points[0])) % 10000 + len(points))
        for _ in range(n - len(points)):
            while True:
                p = rng.normal(0, 1, 3)
                p = p / np.linalg.norm(p) * shell_radius
                if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                    points.append(p)
                    break
                # If can't place, increase radius slightly
                if rng.integers(0, 10) == 0:
                    shell_radius += 0.3
                    break
        if shell_radius > 10:
            break
        shell_radius += 0.5
    
    return np.array(points[:n])

def _build_fcc_like(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    # Generate FCC lattice points
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC: all (i,j,k) where i+j+k is even
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * spacing * 0.7071  # FCC nearest neighbor = 1
                    if np.linalg.norm(p) <= 4.0 * spacing:
                        points.append(p)
    
    points = np.array(points)
    # Keep closest n points to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 5000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
            points.append(p)
    
    # If not enough, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _build_delaunay_like(n, spacing=1.0):
    """Build a cluster with good local packing using greedy placement."""
    rng = np.random.default_rng(123)
    points = [np.zeros(3)]
    radius = spacing * n ** (1.0/3.0) * 0.9
    
    for _ in range(n - 1):
        best_p = None
        best_score = -1
        for _ in range(20):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            min_dist = min(np.linalg.norm(p - q) for q in points)
            if min_dist < 0.6 * spacing:
                continue
            # Prefer points that are close to existing cluster but not too close
            score = min_dist
            if score > best_score:
                best_score = score
                best_p = p
        if best_p is not None:
            points.append(best_p)
        else:
            # Fallback: random
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points[:n])

def _relax_with_restart(pos, potential, param, n_restarts=3, max_time=30.0):
    """Relax with multiple perturbations to escape local minima."""
    best_energy = float('inf')
    best_pos = pos.copy()
    start = time.time()
    
    rng = np.random.default_rng(999)
    
    for restart in range(n_restarts):
        if time.time() - start > max_time:
            break
        # Perturb
        if restart == 0:
            p = pos.copy()
        else:
            p = pos + rng.normal(0, 0.15, pos.shape)
        
        try:
            energy, relaxed = relax(p, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 1.0  # Morse equilibrium distance
    
    # Check time budget
    tl = _time_left()
    if tl < 5.0:
        # Very little time left, just return a quick result
        pos = _build_random_compact(size, spacing, seed=size)
        try:
            _, pos = relax(pos, potential, parameter)
        except Exception:
            pass
        return pos
    
    # Generate multiple diverse starting configurations
    start_configs = []
    
    # 1. Icosahedral shell
    try:
        start_configs.append(_build_icosahedral_shell(size, spacing))
    except Exception:
        pass
    
    # 2. FCC-like
    try:
        start_configs.append(_build_fcc_like(size, spacing))
    except Exception:
        pass
    
    # 3. Random compact
    try:
        start_configs.append(_build_random_compact(size, spacing, seed=size))
    except Exception:
        pass
    
    # 4. Greedy placement
    try:
        start_configs.append(_build_delaunay_like(size, spacing))
    except Exception:
        pass
    
    # 5. Additional random with different seed
    try:
        start_configs.append(_build_random_compact(size, spacing, seed=size + 100))
    except Exception:
        pass
    
    # Determine how many relaxations we can afford
    tl = _time_left()
    # Estimate: each relax call takes ~2-10s depending on n
    est_per_relax = 5.0 if size <= 50 else 8.0
    n_attempts = max(1, min(len(start_configs), int(tl / (est_per_relax * 1.5))))
    
    best_energy = float('inf')
    best_pos = start_configs[0] if start_configs else np.zeros((size, 3))
    
    for i in range(n_attempts):
        tl = _time_left()
        if tl < 10.0:
            break
        
        config = start_configs[i]
        
        # Do a few perturbation restarts
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
    # If any pair is too close, nudge them apart
    pos = best_pos.copy()
    for _ in range(10):
        min_d = float('inf')
        for i in range(size):
            for j in range(i+1, size):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
                if d < 0.5:
                    # Nudge apart
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0, 0])
                    diff /= np.linalg.norm(diff)
                    pos[i] += diff * (0.5 - d) * 0.5
                    pos[j] -= diff * (0.5 - d) * 0.5
        if min_d >= 0.5:
            break
    
    return pos
# EVOLVE-BLOCK-END
