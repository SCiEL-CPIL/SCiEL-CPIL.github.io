"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_fcc(n, rng):
    """Generate FCC-like cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    a = 2.2
    points = []
    max_coord = int(R / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for basis in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + basis[0], j*a + basis[1], k*a + basis[2]])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    # Shuffle and take first n
    rng.shuffle(points)
    if len(points) < n:
        # Fill with random points if not enough
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                points.append(p)
    return np.array(points[:n])


def _generate_icosahedral(n, rng):
    """Generate icosahedral-like cluster."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Generate layers of icosahedra
    points = []
    layer_radius = 0.0
    atoms_per_layer = [1, 12, 42, 92]  # Approximate
    
    for layer_idx, target_atoms in enumerate(atoms_per_layer):
        if len(points) >= n:
            break
        
        if layer_idx == 0:
            # Center atom
            points.append(np.zeros(3))
        else:
            # Add atoms on icosahedral shell
            shell_radius = layer_idx * 2.2
            remaining = min(target_atoms - len(points), n - len(points))
            for _ in range(remaining):
                # Pick random vertex and add small perturbation
                v = verts[rng.integers(0, 12)] * shell_radius
                v = v + rng.normal(0, 0.1, 3)
                points.append(v)
    
    # Fill remaining with random points
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            points.append(p)
    
    return np.array(points[:n])


def _generate_random(n, rng):
    """Generate random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
    return np.array(points)


def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential, parameter = args
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return (energy, relaxed)
    except Exception:
        return (np.inf, positions)


def construct(instance):
    """Best of multiple structural starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size)
    
    # Generate diverse starting structures
    candidates = []
    
    # FCC-like
    for _ in range(3):
        candidates.append(_generate_fcc(size, rng))
    
    # Icosahedral-like
    for _ in range(3):
        candidates.append(_generate_icosahedral(size, rng))
    
    # Random
    for _ in range(2):
        candidates.append(_generate_random(size, rng))
    
    # Parallel relaxation
    time_remaining = _get_time_remaining()
    if time_remaining <= 0:
        # Return best available immediately
        return candidates[0]
    
    # Use up to 4 threads
    max_workers = min(4, len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_relax_worker, (pos, potential, parameter)) for pos in candidates]
        
        best_energy = np.inf
        best_positions = candidates[0]
        
        for future in concurrent.futures.as_completed(futures):
            if _get_time_remaining() <= 0:
                break
            try:
                energy, positions = future.result(timeout=1)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    return best_positions
# EVOLVE-BLOCK-END
