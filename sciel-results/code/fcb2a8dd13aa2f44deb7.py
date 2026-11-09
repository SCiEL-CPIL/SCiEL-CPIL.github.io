"""Bootstrap spawn: direct approach for SC transition-metal cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    # Vectorized pairwise distance check
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _generate_fcc_cluster(n, lattice=1.0):
    """Generate a compact FCC-like cluster."""
    fcc_points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                fcc_points.append([i * lattice, j * lattice, k * lattice])
    # Add FCC offset points
    offsets = [
        [0, 0, 0],
        [lattice / 2, lattice / 2, 0],
        [lattice / 2, 0, lattice / 2],
        [0, lattice / 2, lattice / 2],
    ]
    all_points = []
    for base in fcc_points:
        for off in offsets:
            all_points.append(np.array(base) + np.array(off))
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    points = all_points[mask]
    if len(points) >= n:
        d = np.linalg.norm(points, axis=1)
        order = np.argsort(d)
        return points[order[:n]]
    # Fallback
    rng = np.random.default_rng(42)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-max_r, max_r, 3)
        pts.append(p)
    return np.array(pts)


def _generate_hcp_cluster(n, lattice=1.0):
    """Generate a compact HCP-like cluster."""
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    pts = points[mask]
    if len(pts) >= n:
        d = np.linalg.norm(pts, axis=1)
        order = np.argsort(d)
        return pts[order[:n]]
    rng = np.random.default_rng(43)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-max_r, max_r, 3)
        pts.append(p)
    return np.array(pts)


def _generate_icosahedral_cluster(n, scale=1.0):
    """Generate an icosahedral-like cluster with configurable scale."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * scale
    
    points = [np.zeros(3)]
    for v in ico_verts:
        points.append(v)
    
    # Add mid-edge points (30 edges of icosahedron)
    n_verts = len(ico_verts)
    shell2 = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            # Icosahedron edges have length ~1.051 at unit scale
            if d < 1.2 * scale:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                norm = np.linalg.norm(mid)
                if norm > 1e-8:
                    mid = mid / norm * 1.35 * scale
                    shell2.append(mid)
    for p in shell2:
        points.append(p)
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.normal(size=3)
            norm = np.linalg.norm(p)
            if norm > 1e-8:
                p = p / norm * 2.0 * scale
            points.append(p)
    return np.array(points[:n])


def _generate_bcc_cluster(n, lattice=1.0):
    """Generate a compact BCC-like cluster."""
    points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                points.append([i * lattice, j * lattice, k * lattice])
                points.append([(i + 0.5) * lattice, (j + 0.5) * lattice, (k + 0.5) * lattice])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    pts = points[mask]
    if len(pts) >= n:
        d = np.linalg.norm(pts, axis=1)
        order = np.argsort(d)
        return pts[order[:n]]
    rng = np.random.default_rng(45)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-max_r, max_r, 3)
        pts.append(p)
    return np.array(pts)


def _generate_random_cluster(n, seed, min_dist=0.8):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
    return np.array(points)


def _try_relax(pos, potential, kind):
    """Try to relax a configuration, return (energy, positions) or (inf, None)."""
    try:
        if not _validate_positions(pos):
            return np.inf, None
        energy, relaxed = relax(pos, potential, kind)
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def construct(instance):
    """
    Multi-start optimization with deadline guard for SC cluster global minima.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s kill
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Generate diverse starting configurations
    candidates = []
    
    # Structured initial guesses with varied scales
    candidates.append(_generate_fcc_cluster(n, lattice=0.70))
    candidates.append(_generate_fcc_cluster(n, lattice=0.80))
    candidates.append(_generate_fcc_cluster(n, lattice=0.90))
    candidates.append(_generate_hcp_cluster(n, lattice=0.70))
    candidates.append(_generate_hcp_cluster(n, lattice=0.80))
    candidates.append(_generate_hcp_cluster(n, lattice=0.90))
    candidates.append(_generate_bcc_cluster(n, lattice=0.70))
    candidates.append(_generate_bcc_cluster(n, lattice=0.80))
    candidates.append(_generate_bcc_cluster(n, lattice=0.90))
    candidates.append(_generate_icosahedral_cluster(n, scale=0.9))
    candidates.append(_generate_icosahedral_cluster(n, scale=1.1))
    candidates.append(_generate_icosahedral_cluster(n, scale=1.3))
    
    # Random clusters with varying seeds and min distances
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    min_dists = [0.75, 0.85, 0.95, 1.05]
    
    for seed in seeds:
        for md in min_dists:
            if time.time() > deadline - 10:
                break
            pos = _generate_random_cluster(n, seed=seed, min_dist=md)
            if _validate_positions(pos):
                candidates.append(pos)
    
    # Relax each candidate, keep the best
    best_energy = np.inf
    best_positions = None
    second_best_energy = np.inf
    second_best_positions = None
    
    for pos in candidates:
        if time.time() > deadline - 10:
            break
        energy, relaxed = _try_relax(pos, potential, kind)
        if energy < best_energy:
            if best_positions is not None:
                second_best_energy = best_energy
                second_best_positions = best_positions.copy()
            best_energy = energy
            best_positions = relaxed.copy()
        elif energy < second_best_energy and relaxed is not None:
            second_best_energy = energy
            second_best_positions = relaxed.copy()
    
    # Polish pass: re-relax the best with tighter tolerance
    if best_positions is not None and time.time() < deadline - 5:
        try:
            energy2, relaxed2 = relax(best_positions, potential, kind)
            if np.isfinite(energy2) and _validate_positions(relaxed2) and energy2 < best_energy:
                best_energy = energy2
                best_positions = relaxed2.copy()
        except Exception:
            pass
    
    # Second polish: try re-relaxing the second-best from a perturbed start
    if second_best_positions is not None and time.time() < deadline - 5:
        try:
            rng = np.random.default_rng(12345)
            perturbed = second_best_positions + rng.normal(0, 0.05, size=second_best_positions.shape)
            energy3, relaxed3 = relax(perturbed, potential, kind)
            if np.isfinite(energy3) and _validate_positions(relaxed3) and energy3 < best_energy:
                best_energy = energy3
                best_positions = relaxed3.copy()
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: simple random
        pos = _generate_random_cluster(n, seed=99, min_dist=0.8)
        energy, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos
    
    return best_positions
# EVOLVE-BLOCK-END
