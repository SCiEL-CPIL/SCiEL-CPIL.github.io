"""Portfolio bootstrap: structured high-symmetry seeds + local relax + scale/optimize."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 325.0  # wall-clock safety margin below the 330s kill
_T0 = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _cuboctahedron_verts():
    # 12 vertices: permutations of (0, ±1, ±1)
    v = []
    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            for z in [0, 1, -1]:
                if abs(x) + abs(y) + abs(z) == 2:
                    v.append([x, y, z])
    v = np.array(v, float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _rhombicubocta_verts():
    # 24 vertices of rhombicuboctahedron
    a = 1.0
    b = 1.0 + math.sqrt(2)
    v = []
    for sign in [1, -1]:
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    v.append([sign * a, sx * b, sy * a])
                    v.append([sx * a, sign * b, sy * a])
                    v.append([sx * a, sy * b, sign * a])
    v = np.array(v[:24], float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _fcc_positions(n, scale=0.72):
    """Generate n atoms in FCC-like arrangement."""
    pts = []
    # Generate FCC lattice points within a sphere
    max_idx = int(math.ceil(n ** (1.0 / 3.0)) + 3)
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: (i,j,k) where i+j+k is even, plus face centers
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
                # Face centers
                if abs(i) <= max_idx and abs(j) <= max_idx and abs(k) <= max_idx:
                    if (i + j + k) % 2 != 0:
                        pass  # skip, we'll add face centers separately
    # Add face centers
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 != 0:
                    pts.append([i + 0.5, j + 0.5, k])
                    pts.append([i + 0.5, j, k + 0.5])
                    pts.append([i, j + 0.5, k + 0.5])
    
    pts = np.array(pts, float)
    # Filter to sphere
    norms = np.linalg.norm(pts, axis=1)
    radius = scale * n ** (1.0 / 3.0)
    pts = pts[norms <= radius]
    
    if len(pts) < n:
        # Pad with random points
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                pts = np.vstack([pts, p])
    else:
        # Randomly select n points
        idx = np.random.default_rng(42).permutation(len(pts))[:n]
        pts = pts[idx]
    
    return pts[:n] * 0.5


def _hcp_positions(n, scale=0.72):
    """Generate n atoms in HCP-like arrangement."""
    pts = []
    a = 1.0
    c = math.sqrt(8.0 / 3.0) * a
    max_layer = int(math.ceil(n ** (1.0 / 3.0)) + 3)
    
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        # Hexagonal arrangement in each layer
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * math.sqrt(3.0) / 2.0 * j
                pts.append([x, y, z])
    
    pts = np.array(pts, float)
    norms = np.linalg.norm(pts, axis=1)
    radius = scale * n ** (1.0 / 3.0)
    pts = pts[norms <= radius]
    
    if len(pts) < n:
        rng = np.random.default_rng(43)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                pts = np.vstack([pts, p])
    else:
        idx = np.random.default_rng(43).permutation(len(pts))[:n]
        pts = pts[idx]
    
    return pts[:n] * 0.5


def _bcc_positions(n, scale=0.72):
    """Generate n atoms in BCC-like arrangement."""
    pts = []
    a = 1.0
    max_idx = int(math.ceil(n ** (1.0 / 3.0)) + 3)
    
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                pts.append([i * a, j * a, k * a])
                # Body center
                pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    
    pts = np.array(pts, float)
    norms = np.linalg.norm(pts, axis=1)
    radius = scale * n ** (1.0 / 3.0)
    pts = pts[norms <= radius]
    
    if len(pts) < n:
        rng = np.random.default_rng(44)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                pts = np.vstack([pts, p])
    else:
        idx = np.random.default_rng(44).permutation(len(pts))[:n]
        pts = pts[idx]
    
    return pts[:n] * 0.5


def _random_compact(n, seed, spacing=0.72):
    """Generate n atoms in a compact random arrangement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    pts = []
    min_dist = 0.8 * spacing
    attempts = 0
    while len(pts) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                pts.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, just use random points
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    
    return np.array(pts[:n], float)


def _scale_and_optimize(positions, potential, kind, max_iter=1000):
    """Scale positions to optimal density and relax."""
    n = len(positions)
    
    # Try different scales
    best_energy = np.inf
    best_positions = positions
    
    for scale in [0.5, 0.6, 0.7, 0.72, 0.75, 0.8, 0.9, 1.0]:
        scaled = positions * scale
        try:
            energy, relaxed = relax(scaled, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            continue
    
    return best_energy, best_positions


def construct(instance):
    """Portfolio of structured seeds + local relax."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse starting configurations
    seeds = []
    
    # 1. Icosahedron-based
    ico = _icosahedron_verts()
    for scale in [0.5, 0.6, 0.7, 0.8]:
        pts = np.tile(ico, (size // len(ico) + 1, 1))[:size] * scale
        seeds.append(pts)
    
    # 2. Cuboctahedron-based
    cubo = _cuboctahedron_verts()
    for scale in [0.5, 0.6, 0.7, 0.8]:
        pts = np.tile(cubo, (size // len(cubo) + 1, 1))[:size] * scale
        seeds.append(pts)
    
    # 3. FCC
    for scale in [0.5, 0.6, 0.7]:
        seeds.append(_fcc_positions(size, scale))
    
    # 4. HCP
    for scale in [0.5, 0.6, 0.7]:
        seeds.append(_hcp_positions(size, scale))
    
    # 5. BCC
    for scale in [0.5, 0.6, 0.7]:
        seeds.append(_bcc_positions(size, scale))
    
    # 6. Random compact
    for seed_idx in range(6):
        seeds.append(_random_compact(size, seed_idx))
    
    # 7. Spherical shell distributions
    for shell in [0.4, 0.5, 0.6]:
        rng = np.random.default_rng(100 + shell * 10)
        pts = []
        radius = 0.72 * size ** (1.0 / 3.0) * 0.8
        for _ in range(size):
            # Random point in sphere
            u = rng.random()
            v = rng.random()
            r = radius * u ** (1.0 / 3.0)
            theta = 2.0 * np.pi * v
            phi = np.arccos(2.0 * u - 1.0)
            p = [r * np.sin(phi) * np.cos(theta),
                 r * np.sin(phi) * np.sin(theta),
                 r * np.cos(phi)]
            pts.append(p)
        seeds.append(np.array(pts, float))
    
    # Relax each seed and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, seed in enumerate(seeds):
        if _time_left() < 15.0:
            break
        
        try:
            energy, relaxed = relax(seed, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            continue
    
    # If we still have time, try to improve the best solution
    if best_positions is not None and _time_left() > 30.0:
        try:
            # Perturb and re-relax
            rng = np.random.default_rng(200)
            for _ in range(3):
                if _time_left() < 15.0:
                    break
                perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
                energy, relaxed = relax(perturbed, potential, kind)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except:
            pass
    
    if best_positions is None:
        # Fallback: random
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0) * 0.75
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
