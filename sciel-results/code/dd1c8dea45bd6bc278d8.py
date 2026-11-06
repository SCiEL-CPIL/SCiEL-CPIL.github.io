"""Dzugutov cluster global minima: Mackay icosahedral seeds + parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 290.0  # seconds, safety margin before 330s kill
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    # Mask upper triangle
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    # Compute energy
    e = np.zeros(len(r))
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    if np.any(mask_a):
        e[mask_a] = A * (r[mask_a] ** (-m) - B) * np.exp(c / (r[mask_a] - a))
    if np.any(mask_b):
        e[mask_b] = B * np.exp(d / (r[mask_b] - b))
    return float(np.sum(e))


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit circumscribed radius)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit circumscribed radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _build_mackay_icosahedron(n_atoms, scale=1.0, rng=None):
    """Build a Mackay icosahedral cluster with n_atoms atoms.
    
    Mackay icosahedra have shells:
    - 1 atom (center)
    - 12 atoms (first shell, icosahedron vertices)
    - 42 atoms (second shell)
    - 92 atoms (third shell)
    ...
    For n between these, we take the largest shell that fits and fill remaining
    by adding atoms on the next shell.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Shell sizes for Mackay icosahedra: 1, 12, 42, 92, ...
    shell_sizes = [1, 12, 42, 92]
    
    # Determine which shells to use
    positions = []
    remaining = n_atoms
    
    # Center atom
    if remaining > 0:
        positions.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 12 icosahedron vertices
    ico_verts = _icosahedron_vertices()
    r1 = 1.0 * scale  # First shell radius
    if remaining > 0:
        n_first = min(12, remaining)
        for i in range(n_first):
            positions.append(ico_verts[i] * r1)
        remaining -= n_first
    
    # Second shell: 42 atoms (face centers of icosahedron faces, etc.)
    if remaining > 0:
        # Generate second shell positions: face centers of icosahedron
        # Icosahedron has 20 faces
        # For simplicity, generate points on a sphere of radius r2
        r2 = 1.0 * np.sqrt(3) * scale  # Approximate second shell radius
        
        # Use golden ratio spiral for even distribution
        n_second = min(42, remaining)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_second):
            z = 1 - 2 * (i + 0.5) / n_second
            theta = golden_angle * i
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r2)
        remaining -= n_second
    
    # Third shell and beyond: fill with more spherical shells
    if remaining > 0:
        r3 = 1.0 * np.sqrt(5) * scale
        n_third = min(92, remaining)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_third):
            z = 1 - 2 * (i + 0.5) / n_third
            theta = golden_angle * i
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r3)
        remaining -= n_third
    
    # If still more needed, add on larger shells
    shell_radius = 1.0 * np.sqrt(7) * scale
    shell_idx = 4
    while remaining > 0:
        golden_angle = np.pi * (3 - np.sqrt(5))
        n_shell = min(remaining, 120)
        for i in range(n_shell):
            z = 1 - 2 * (i + 0.5) / n_shell
            theta = golden_angle * i
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            positions.append(np.array([x, y, z]) * shell_radius)
        remaining -= n_shell
        shell_radius += scale
        shell_idx += 1
    
    return np.array(positions[:n_atoms])


def _random_compact_cluster(n_atoms, rng):
    """Generate a random compact cluster with minimum distance constraint."""
    spacing = 1.0
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.85
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.65 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, fill remaining randomly
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _kick_and_relax(positions, kick_scale, rng, potential='dzugutov', parameter=None):
    """Apply a random kick and relax."""
    kicked = positions + rng.normal(0, kick_scale, positions.shape)
    try:
        energy, relaxed = relax(kicked, potential, parameter)
        return energy, relaxed
    except:
        return np.inf, positions


def _basin_hopping(positions, n_kicks, kick_scale_base, rng, potential='dzugutov', parameter=None, deadline_fn=None):
    """Perform basin hopping from a starting position."""
    best_energy, best_pos = _kick_and_relax(positions, kick_scale_base, rng, potential, parameter)
    
    for i in range(n_kicks):
        if deadline_fn is not None and deadline_fn() < 5.0:
            break
        
        # Adaptive kick scale
        kick_scale = kick_scale_base * (0.8 + 0.4 * np.random.random())
        new_energy, new_pos = _kick_and_relax(best_pos, kick_scale, rng, potential, parameter)
        
        if new_energy < best_energy:
            best_energy, best_pos = new_energy, new_pos
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov clusters using icosahedral seeds and parallel basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting configurations
    seeds = []
    
    # 1. Mackay icosahedral seeds with different scales
    for scale in [0.9, 1.0, 1.1]:
        try:
            seed = _build_mackay_icosahedron(size, scale=scale, rng=rng)
            if len(seed) == size:
                seeds.append(seed)
        except:
            pass
    
    # 2. Deformed icosahedra
    for deform in [0.1, 0.2, 0.3]:
        try:
            seed = _build_mackay_icosahedron(size, scale=1.0, rng=rng)
            if len(seed) == size:
                # Apply anisotropic deformation
                deform_matrix = np.diag([1 + deform, 1 - deform/2, 1 - deform/2])
                seed_deformed = seed @ deform_matrix.T
                seeds.append(seed_deformed)
        except:
            pass
    
    # 3. Random compact clusters
    for _ in range(3):
        try:
            seed = _random_compact_cluster(size, rng)
            seeds.append(seed)
        except:
            pass
    
    # Relax all seeds to get initial best
    best_energy = np.inf
    best_positions = None
    
    for seed in seeds:
        if _remaining() < 10.0:
            break
        try:
            energy, relaxed = relax(seed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            pass
    
    if best_positions is None:
        # Fallback: return a simple arrangement
        best_positions = _random_compact_cluster(size, rng)
        best_energy = _energy(best_positions)
    
    # Parallel basin hopping from the best seed
    n_workers = min(4, max(1, int(_remaining() / 20)))  # Use up to 4 threads
    kicks_per_worker = max(5, int(_remaining() / (n_workers * 10)))
    
    # Distribute work
    work_items = []
    for i in range(n_workers):
        kick_scale = 0.3 + 0.2 * i  # Different kick scales for diversity
        work_items.append((best_positions.copy(), kicks_per_worker, kick_scale, i))
    
    results = [(best_energy, best_positions)]
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for pos, n_kicks, kick_scale, idx in work_items:
            worker_rng = np.random.default_rng(size * 1000 + idx)
            future = executor.submit(
                _basin_hopping, pos, n_kicks, kick_scale, worker_rng,
                potential, parameter, _remaining
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                energy, positions = future.result(timeout=5)
                if energy < results[0][0]:
                    results[0] = (energy, positions)
            except:
                pass
    
    # Final polish: one more relaxation with tighter criteria
    if _remaining() > 10.0:
        try:
            final_energy, final_pos = relax(results[0][1], potential, parameter)
            if final_energy < results[0][0]:
                results[0] = (final_energy, final_pos)
        except:
            pass
    
    # Ensure minimum distance constraint
    pos = results[0][1]
    # Check and fix any violations
    for i in range(size):
        for j in range(i + 1, size):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 0.5 and d > 0:
                direction = (pos[i] - pos[j]) / d
                correction = (0.5 - d) / 2
                pos[i] += direction * correction
                pos[j] -= direction * correction
    
    return pos
# EVOLVE-BLOCK-END
