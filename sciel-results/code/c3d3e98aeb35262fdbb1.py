"""Morse cluster global minima finder with multiple restart strategies."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)


def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9


def _fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    pts = []
    # FCC lattice vectors
    a = spacing
    # Generate points in a cube and keep closest n to origin
    max_r = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                # FCC: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    # Keep n closest to origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)  # ideal HCP c/a
    max_r = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                # HCP: two atoms per unit cell
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 1/3, j + 2/3, k + 0.5]) * a
                # Adjust for c axis
                p1[2] *= c / a
                p2[2] *= c / a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _icosahedron_shell(n, spacing=1.0):
    """Generate icosahedral shell cluster."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    pts = [np.zeros(3)]  # center
    if n > 1:
        pts.extend([v.copy() for v in verts])  # 12 vertices
    if n > 13:
        # Add face centers (20 faces)
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            center = center / np.linalg.norm(center) * spacing * 0.87
            pts.append(center)
    if n > 33:
        # Add edge midpoints (30 edges) - approximate
        edges = set()
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * spacing:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * spacing * 0.76
                    pts.append(mid)
    
    pts = np.array(pts[:n])
    return pts


def _random_packed(n, rng, spacing=1.0):
    """Random packed cluster with minimum distance constraint."""
    pts = []
    radius = spacing * n ** (1/3) * 1.2
    min_d = 0.55 * spacing
    attempts = 0
    max_attempts = 50000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_d for q in pts):
                pts.append(p)
        attempts += 1
    # If we didn't get enough, relax constraints
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _perturb_cluster(base, n, rng, scale=0.3):
    """Perturb a base cluster and resize to n atoms."""
    if len(base) >= n:
        # Keep n closest to centroid
        centroid = base.mean(axis=0)
        dists = np.linalg.norm(base - centroid, axis=1)
        idx = np.argsort(dists)[:n]
        pts = base[idx].copy()
    else:
        pts = base.copy()
        # Add random atoms near surface
        centroid = pts.mean(axis=0)
        while len(pts) < n:
            # Pick a random existing point and add neighbor
            i = rng.integers(len(pts))
            new_p = pts[i] + rng.normal(0, scale, 3)
            pts = np.vstack([pts, new_p])
    
    # Add small random perturbation
    pts = pts + rng.normal(0, 0.05, pts.shape)
    return pts


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin
    
    rng = np.random.default_rng(size * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC-based
    try:
        fcc = _fcc_cluster(size, spacing=1.0)
        initial_configs.append(('fcc', fcc))
    except Exception:
        pass
    
    # 2. HCP-based
    try:
        hcp = _hcp_cluster(size, spacing=1.0)
        initial_configs.append(('hcp', hcp))
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _icosahedron_shell(size, spacing=1.0)
        initial_configs.append(('ico', ico))
    except Exception:
        pass
    
    # 4. Random packed (multiple variants)
    for i in range(3):
        try:
            rp = _random_packed(size, rng, spacing=1.0)
            initial_configs.append((f'rand{i}', rp))
        except Exception:
            pass
    
    # 5. Perturbed FCC
    try:
        fcc_base = _fcc_cluster(max(size, 20), spacing=1.0)
        pert = _perturb_cluster(fcc_base, size, rng, scale=0.4)
        initial_configs.append(('fcc_pert', pert))
    except Exception:
        pass
    
    # 6. Perturbed random
    try:
        rand_base = _random_packed(max(size, 15), rng, spacing=1.0)
        pert2 = _perturb_cluster(rand_base, size, rng, scale=0.5)
        initial_configs.append(('rand_pert', pert2))
    except Exception:
        pass
    
    # Additional random restarts for larger clusters
    if size >= 50:
        for i in range(2):
            try:
                rp = _random_packed(size, rng, spacing=1.0)
                initial_configs.append((f'rand_extra{i}', rp))
            except Exception:
                pass
    
    # Relax each initial configuration
    for name, pos in initial_configs:
        if time.time() > deadline - 10:
            break
        try:
            if not _min_dist_check(pos):
                # Try to fix by adding small noise and re-checking
                pos = pos + rng.normal(0, 0.1, pos.shape)
                if not _min_dist_check(pos):
                    continue
            
            energy, relaxed_pos = relax(pos, potential, rho)
            
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # If we have time, do additional random restarts
    if best_pos is not None and time.time() < deadline - 20:
        n_extra = min(10, int((deadline - time.time()) / 5))
        for i in range(n_extra):
            if time.time() > deadline - 10:
                break
            try:
                rp = _random_packed(size, rng, spacing=1.0)
                if not _min_dist_check(rp):
                    rp = rp + rng.normal(0, 0.15, rp.shape)
                energy, relaxed_pos = relax(rp, potential, rho)
                if _min_dist_check(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
            except Exception:
                continue
    
    # Final fallback: if no valid result, create a simple valid config
    if best_pos is None:
        # Create a linear chain or simple valid config
        spacing = 0.6
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i] = [i * spacing, 0, 0]
        # Verify
        if not _min_dist_check(best_pos):
            # Spread them out more
            radius = spacing * size ** (1/3) * 1.5
            best_pos = rng.uniform(-radius, radius, (size, 3))
            # Ensure minimum distance by rejection
            valid = [best_pos[0]]
            for i in range(1, size):
                attempts = 0
                while attempts < 1000:
                    p = best_pos[i]
                    if all(np.linalg.norm(p - v) >= 0.5 for v in valid):
                        valid.append(p)
                        break
                    p = rng.uniform(-radius, radius, 3)
                    best_pos[i] = p
                    attempts += 1
                else:
                    valid.append(best_pos[i])
            best_pos = np.array(valid[:size])
    
    return best_pos
# EVOLVE-BLOCK-END
