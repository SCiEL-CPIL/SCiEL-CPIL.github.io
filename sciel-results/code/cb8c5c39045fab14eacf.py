"""Dzugutov cluster global minima: icosahedral shell seeds + basin hopping with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 300.0
_THREAD_LOCK = threading.Lock()
_best_results = {}


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy for a set of positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    E = 0.0
    mask1 = r_ij < a
    mask2 = (r_ij >= a) & (r_ij < b)
    if np.any(mask1):
        r1 = r_ij[mask1]
        E += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    if np.any(mask2):
        r2 = r_ij[mask2]
        E += np.sum(B * np.exp(d / (r2 - b)))
    return float(E)


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron with edge length ~1."""
    t = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    # Normalize to have a reasonable size
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * 1.0
    return verts


def _generate_icosahedral_cluster(n, rng):
    """Generate an icosahedral shell cluster of n atoms."""
    # Core: 1 atom at center
    # First shell: 12 atoms (icosahedron)
    # Second shell: 42 atoms (truncated icosahedron-like)
    # Third shell: 92 atoms
    
    shells = [1, 12, 42, 92, 120]
    # For n up to 60, we use core + first shell + part of second shell
    
    positions = [np.zeros(3)]  # center atom
    
    # First shell: icosahedron with radius ~1.1
    ico = _icosahedron_vertices()
    r1 = 1.1
    for v in ico:
        positions.append(v * r1)
    
    remaining = n - len(positions)
    if remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: place atoms on a larger icosahedron or dodecahedron
    # Use a larger icosahedron rotated and scaled
    r2 = 1.9
    # Generate points on a sphere with some symmetry
    # Use Fibonacci sphere for uniform distribution
    def fibonacci_sphere(n_points, radius, rng):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            z = 1 - 2 * i / max(n_points - 1, 1)
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i + rng.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append(np.array([x, y, z]) * radius)
        return points
    
    # Add second shell atoms
    second_shell = fibonacci_sphere(remaining, r2, rng)
    for p in second_shell:
        positions.append(p)
    
    return np.array(positions[:n])


def _generate_dodecahedral_cluster(n, rng):
    """Generate a dodecahedral-like cluster."""
    positions = [np.zeros(3)]
    
    # First shell: 20 vertices of dodecahedron (approximate with icosahedron dual)
    ico = _icosahedron_vertices()
    r1 = 1.1
    for v in ico:
        positions.append(v * r1)
    
    remaining = n - len(positions)
    if remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell
    r2 = 2.0
    def fibonacci_sphere(n_points, radius, rng):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            z = 1 - 2 * i / max(n_points - 1, 1)
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i + rng.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append(np.array([x, y, z]) * radius)
        return points
    
    second_shell = fibonacci_sphere(remaining, r2, rng)
    for p in second_shell:
        positions.append(p)
    
    return np.array(positions[:n])


def _generate_cuboctahedral_cluster(n, rng):
    """Generate a cuboctahedral-like cluster."""
    positions = [np.zeros(3)]
    
    # First shell: 12 vertices (cuboctahedron)
    # Cuboctahedron vertices: permutations of (0, 1, 1)
    verts = []
    for perm in [(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1),
                 (1,0,1),(1,0,-1),(1,-1,1),(1,-1,-1),
                 (-1,0,1),(-1,0,-1),(-1,1,1),(-1,1,-1)]:
        verts.append(np.array(perm, dtype=float))
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * 1.1
    
    for v in verts:
        positions.append(v)
    
    remaining = n - len(positions)
    if remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell
    r2 = 2.0
    def fibonacci_sphere(n_points, radius, rng):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            z = 1 - 2 * i / max(n_points - 1, 1)
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i + rng.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append(np.array([x, y, z]) * radius)
        return points
    
    second_shell = fibonacci_sphere(remaining, r2, rng)
    for p in second_shell:
        positions.append(p)
    
    return np.array(positions[:n])


def _basin_hopping(start_pos, potential, parameter, rng, max_iter=50, kick_scale=0.3):
    """Simple basin hopping: kick and relax."""
    best_pos = start_pos.copy()
    best_energy = _dzugutov_energy(best_pos)
    
    for i in range(max_iter):
        if _remaining() < 10:
            break
        
        # Kick
        kick = rng.normal(0, kick_scale, size=start_pos.shape)
        kicked = best_pos + kick
        
        # Relax
        try:
            energy, relaxed = relax(kicked, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
        except Exception:
            continue
    
    return best_energy, best_pos


def _relax_and_score(start_pos, potential, parameter):
    """Relax a starting position and return energy and positions."""
    try:
        energy, positions = relax(start_pos, potential, parameter)
        return energy, positions
    except Exception:
        return np.inf, start_pos


def construct(instance):
    """Find low-energy Dzugutov clusters using icosahedral seeds and basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if size in _best_results:
        return _best_results[size]
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse seeds
    seeds = []
    
    # Icosahedral seed
    seeds.append(_generate_icosahedral_cluster(size, rng))
    
    # Dodecahedral seed
    seeds.append(_generate_dodecahedral_cluster(size, rng))
    
    # Cuboctahedral seed
    seeds.append(_generate_cuboctahedral_cluster(size, rng))
    
    # Random compact seed
    radius = 1.1 * size ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    seeds.append(np.array(points))
    
    # Additional random seeds with different radii
    for r_mult in [0.6, 0.9, 1.2]:
        radius = 1.1 * size ** (1.0 / 3.0) * r_mult
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
                points.append(p)
        if len(points) == size:
            seeds.append(np.array(points))
    
    # Relax all seeds and find best
    best_energy = np.inf
    best_pos = seeds[0]
    
    for seed in seeds:
        if _remaining() < 15:
            break
        energy, pos = _relax_and_score(seed, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Basin hopping from the best seed
    if _remaining() > 30:
        kick_scales = [0.3, 0.2, 0.15]
        for ks in kick_scales:
            if _remaining() < 15:
                break
            e, pos = _basin_hopping(best_pos, potential, parameter, rng, max_iter=30, kick_scale=ks)
            if e < best_energy:
                best_energy = e
                best_pos = pos
    
    # Final relaxation to ensure convergence
    if _remaining() > 5:
        try:
            final_energy, final_pos = relax(best_pos, potential, parameter)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except Exception:
            pass
    
    _best_results[size] = best_pos
    return best_pos
# EVOLVE-BLOCK-END
