"""Dzugutov cluster global minima: icosahedral seeds + parallel basin hopping with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 295.0  # 5s safety margin before 300s kill
_LOCK = threading.Lock()
_BEST_RESULTS = {}  # n -> (energy, positions)


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy for a cluster."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    r = np.triu(r, 1)
    e = np.zeros_like(r)
    mask_a = r < a
    mask_b = (r < b) & (~mask_a)
    if np.any(mask_a):
        e[mask_a] = A * (r[mask_a] ** (-m) - B) * np.exp(c / (r[mask_a] - a))
    if np.any(mask_b):
        e[mask_b] = B * np.exp(d / (r[mask_b] - b))
    return float(np.sum(e))


def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron (unit edge length scaled)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    # Normalize to unit edge length
    edge = np.linalg.norm(verts[0] - verts[1])
    verts /= edge
    return verts


def _mckay_icosahedron(n_atoms):
    """Generate a Mackay icosahedral cluster with n_atoms atoms."""
    ico = _icosahedron_vertices()
    # Mackay icosahedra: 13 (n=1), 55 (n=2), 147 (n=3)
    # For arbitrary n, use shells: core (1), 12-vertex shell, 30-edge shell, 20-face shell
    # Shell radii: r1=1, r2=sqrt(3), r3=2*sqrt(2), r4=...
    # For 38-60 atoms, we use: 1 core + 12 (icosahedron) + 30 (edge midpoints of octahedron) + rest
    # Actually, let's use a more flexible approach:
    # Core: 1 atom at center
    # Shell 1: 12 atoms at icosahedron vertices (radius r1)
    # Shell 2: 30 atoms at edge midpoints of the icosahedron (radius r2)
    # Shell 3: 20 atoms at face centers (radius r3)
    # Total: 1+12+30+20 = 63, so for n<=60 we truncate
    
    # Better approach: use known Mackay numbers and interpolate
    # 13, 55, 147 -> for n in [38,60], use 13 + partial 55-shell
    # 13 = 1 + 12
    # 55 = 1 + 12 + 30 + 12 (second shell of icosahedron)
    # Actually 55 = 1 + 12 + 30 + 12? No: 1+12=13, +30=43, +12=55. Yes!
    # So for n in [38,60]: start with 13 (core+icosahedron), add from 30-edge shell
    
    positions = [np.zeros(3)]  # core
    # icosahedron shell
    r1 = 1.0
    for v in ico:
        positions.append(r1 * v)
    # Edge midpoints of icosahedron (30 edges)
    # The 12 vertices form edges; find all pairs at minimum distance
    verts = ico
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(verts[i] - verts[j]) - min_dist) < 1e-10:
                edges.append((i, j))
    
    r2 = np.linalg.norm(verts[0] + verts[1]) / 2.0  # midpoint radius
    for (i, j) in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * r2
        positions.append(mid)
    
    # Face centers (20 faces)
    # Find faces: triangles of vertices at mutual min_dist
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if (abs(d1 - min_dist) < 1e-10 and abs(d2 - min_dist) < 1e-10 and abs(d3 - min_dist) < 1e-10):
                    faces.append((i, j, k))
    
    r3 = np.linalg.norm(verts[0] + verts[1] + verts[2]) / 3.0
    for (i, j, k) in faces:
        face_center = (verts[i] + verts[j] + verts[k]) / 3.0
        face_center = face_center / np.linalg.norm(face_center) * r3
        positions.append(face_center)
    
    # Truncate or pad to n_atoms
    positions = np.array(positions[:n_atoms])
    if len(positions) < n_atoms:
        # Add atoms on a sphere
        rng = np.random.default_rng(42)
        while len(positions) < n_atoms:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * (r3 + 0.3)
            positions = np.vstack([positions, p])
    
    # Scale to reasonable size for Dzugutov (optimal distance ~1.0-1.2)
    # The Dzugutov potential has minimum around r~1.0-1.1
    # Our icosahedron edge is 1.0, so scale by ~1.0
    return positions


def _twisted_icosahedron(n_atoms, twist_angle, rng):
    """Generate a twisted icosahedral cluster."""
    base = _mckay_icosahedron(n_atoms)
    # Apply a small rotation to outer shells
    # Identify shells by radius
    radii = np.linalg.norm(base, axis=1)
    # Sort by radius
    order = np.argsort(radii)
    # Twist outer atoms
    theta = twist_angle
    z_axis = np.array([0, 0, 1.0])
    # Rotation matrix around z
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    
    # Apply increasing twist with radius
    result = base.copy()
    for idx in range(n_atoms):
        r = radii[idx]
        if r > 0.5:  # not the core
            twist_factor = r / max(radii)
            angle = theta * twist_factor
            c2, s2 = np.cos(angle), np.sin(angle)
            R2 = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1.0]])
            result[idx] = R2 @ result[idx]
    
    # Add small random perturbation
    result += rng.normal(0, 0.05, size=(n_atoms, 3))
    return result


def _random_compact_cluster(n_atoms, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.5
    attempts = 0
    while len(points) < n_atoms and attempts < n_atoms * 100:
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
    if len(points) < n_atoms:
        # Fallback: grid
        pts = []
        step = spacing * 0.6
        for x in np.arange(-radius, radius, step):
            for y in np.arange(-radius, radius, step):
                for z in np.arange(-radius, radius, step):
                    p = np.array([x, y, z])
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
                    if len(pts) >= n_atoms:
                        break
                if len(pts) >= n_atoms:
                    break
            if len(pts) >= n_atoms:
                break
        points = pts[:n_atoms]
    return np.array(points)


def _basin_hop(start_pos, n_atoms, rng, max_iter=50, kick_scale=0.3):
    """Perform basin hopping: kick + relax, keep best."""
    best_energy = _dzugutov_energy(start_pos)
    best_pos = start_pos.copy()
    current_pos = start_pos.copy()
    current_energy = best_energy
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        # Kick
        kick = rng.normal(0, kick_scale, size=(n_atoms, 3))
        kicked = current_pos + kick
        # Relax
        try:
            energy, relaxed = relax(kicked, 'dzugutov', None)
        except Exception:
            break
        if energy < current_energy:
            current_pos = relaxed
            current_energy = energy
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
        # Anneal kick scale
        kick_scale *= 0.95
    
    return best_energy, best_pos


def _worker(n_atoms, seed_idx, n_icos, n_twist, n_random):
    """Worker for parallel basin hopping."""
    rng = np.random.default_rng(n_atoms * 1000 + seed_idx)
    best_energy = np.inf
    best_pos = None
    
    # Icosahedral seeds
    for i in range(n_icos):
        if _remaining() < 10.0:
            break
        pos = _mckay_icosahedron(n_atoms)
        # Small random perturbation
        pos += rng.normal(0, 0.02, size=(n_atoms, 3))
        e, p = _basin_hop(pos, n_atoms, rng, max_iter=40, kick_scale=0.2)
        if e < best_energy:
            best_energy = e
            best_pos = p
    
    # Twisted icosahedral seeds
    for i in range(n_twist):
        if _remaining() < 10.0:
            break
        twist = rng.uniform(-0.3, 0.3)
        pos = _twisted_icosahedron(n_atoms, twist, rng)
        e, p = _basin_hop(pos, n_atoms, rng, max_iter=40, kick_scale=0.25)
        if e < best_energy:
            best_energy = e
            best_pos = p
    
    # Random compact seeds
    for i in range(n_random):
        if _remaining() < 10.0:
            break
        pos = _random_compact_cluster(n_atoms, rng, spacing=1.0)
        e, p = _basin_hop(pos, n_atoms, rng, max_iter=30, kick_scale=0.3)
        if e < best_energy:
            best_energy = e
            best_pos = p
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n_atoms = instance["n"]
    
    # Allocate time budget
    # Total available: ~295s for all 6 instances
    # Each instance gets roughly 295/6 ≈ 49s, but we'll use parallelism
    # Use 4 threads per instance
    
    n_icos = 3
    n_twist = 3
    n_random = 3
    
    # Run in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for seed_idx in range(4):
            f = executor.submit(_worker, n_atoms, seed_idx, n_icos, n_twist, n_random)
            futures.append(f)
        
        for f in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                e, p = f.result(timeout=max(1.0, _remaining()))
                results.append((e, p))
            except Exception:
                pass
    
    # Pick best
    if not results:
        # Fallback: simple icosahedron
        pos = _mckay_icosahedron(n_atoms)
        try:
            energy, pos = relax(pos, 'dzugutov', None)
        except Exception:
            pass
        return pos
    
    best = min(results, key=lambda x: x[0])
    return best[1]
# EVOLVE-BLOCK-END
