"""Sutton-Chen cluster global minima via structure-motif seeding + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _icosa_shell(n, scale=1.0):
    """Generate an icosahedral shell cluster."""
    # Icosahedron vertices (unit sphere)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]  # center atom
    # Add atoms on shells
    shell_radii = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.2]
    for r in shell_radii:
        if len(pts) >= n:
            break
        for v in verts:
            if len(pts) >= n:
                break
            p = r * v * scale
            # Check minimum distance
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.4 * scale:
                    ok = False
                    break
            if ok:
                pts.append(p)
    
    # Fill remaining with random points on sphere
    rng = np.random.default_rng(42)
    while len(pts) < n:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        r = shell_radii[min(len(pts) // 12, len(shell_radii) - 1)] * scale
        p = r * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < 0.4 * scale:
                ok = False
                break
        if ok:
            pts.append(p)
        else:
            # Allow closer if we must fill
            if len(pts) < n:
                pts.append(p)
    
    return np.array(pts[:n])


def _fcc_cluster(n, scale=1.0):
    """Generate an FCC-like cluster."""
    pts = []
    a = scale * 0.72
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for offset in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + offset[0], j*a + offset[1], k*a + offset[2]])
                    if np.linalg.norm(p) < 4.5 * scale:
                        pts.append(p)
    
    pts = np.array(pts)
    # Sort by distance from center and take n closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, scale=1.0):
    """Generate an HCP-like cluster."""
    pts = []
    a = scale * 0.72
    c = a * np.sqrt(8/3)
    max_idx = 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # HCP layers
                if k % 2 == 0:
                    p = np.array([i*a, j*a*np.sqrt(3)/2, k*c/2])
                else:
                    p = np.array([i*a + a/2, j*a*np.sqrt(3)/2 + a*np.sqrt(3)/4, k*c/2])
                if np.linalg.norm(p) < 4.0 * scale:
                    pts.append(p)
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _random_compact(n, scale=1.0, seed=0):
    """Random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = scale * n**(1.0/3.0) * 0.8
    pts = []
    for _ in range(50 * n):
        if len(pts) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.5 * scale:
                    ok = False
                    break
            if ok:
                pts.append(p)
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    
    return np.array(pts[:n])


def _basin_hopping(pos, kind, max_iters=20, step_size=0.5):
    """Simple basin hopping: perturb, relax, accept if lower."""
    energy, pos = relax(pos.copy(), 'sutton_chen', kind)
    
    for i in range(max_iters):
        if _time_left() < 5:
            break
        # Perturb
        rng = np.random.default_rng(i)
        delta = rng.normal(0, step_size, pos.shape)
        new_pos = pos + delta
        
        # Relax
        new_energy, new_pos = relax(new_pos, 'sutton_chen', kind)
        
        # Accept if better
        if new_energy < energy:
            energy = new_energy
            pos = new_pos
        else:
            # Metropolis acceptance
            if energy < 0:
                dE = new_energy - energy
                if rng.random() < np.exp(-dE / 0.05):
                    pos = new_pos
                    energy = new_energy
    
    return energy, pos


def construct(instance):
    """Multi-start structure-motif + basin hopping for Sutton-Chen clusters."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse starting structures
    starts = []
    
    # Icosahedral
    starts.append(_icosa_shell(n, scale=1.0))
    # FCC
    starts.append(_fcc_cluster(n, scale=1.0))
    # HCP
    starts.append(_hcp_cluster(n, scale=1.0))
    # Random compact with different seeds
    for seed in range(6):
        starts.append(_random_compact(n, scale=1.0, seed=seed))
    
    # Relax all starting structures
    best_energy = np.inf
    best_pos = None
    
    for i, start in enumerate(starts):
        if _time_left() < 10:
            break
        
        # First, do a quick local relaxation
        energy, pos = relax(start, 'sutton_chen', kind)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
        
        # Basin hopping from this starting point
        if _time_left() > 30:
            bh_energy, bh_pos = _basin_hopping(pos, kind, max_iters=15, step_size=0.4)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
    
    # If we still have time, do more basin hopping from the best structure
    if best_pos is not None and _time_left() > 20:
        bh_energy, bh_pos = _basin_hopping(best_pos, kind, max_iters=30, step_size=0.3)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Validate
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        # Fallback: random compact
        best_pos = _random_compact(n, scale=1.0, seed=999)
        _, best_pos = relax(best_pos, 'sutton_chen', kind)
    
    return best_pos
# EVOLVE-BLOCK-END
