"""Morse cluster global minima finder - Basin Hopping with structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
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


def _fix_min_dist(pos, min_d=0.51, max_iter=200):
    """Separate atoms that are too close using iterative pushing."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= min_d:
            break
        i, j = min_pair
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[i] += push * rij / r
        pos[j] -= push * rij / r
    return pos


def _fcc_cluster(n, spacing=1.0):
    """Generate FCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _hcp_cluster(n, spacing=1.0):
    """Generate HCP lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _bcc_cluster(n, spacing=1.0):
    """Generate BCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing * np.sqrt(3.0) / 2.0
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _icosahedron_shell(n, spacing=1.0):
    """Generate icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edge_len = 2.0
    scale = spacing / edge_len
    verts = verts * scale
    
    pts = [np.zeros(3)]
    
    if n > 1:
        for v in verts:
            pts.append(v.copy())
    
    if n > 13:
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                center = center / r * r * 1.15
            pts.append(center)
    
    if n > 33:
        edges = set()
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * scale:
                    mid = (verts[i] + verts[j]) / 2.0
                    r = np.linalg.norm(mid)
                    if r > 1e-10:
                        mid = mid / r * r * 1.3
                    pts.append(mid)
    
    if n > 63:
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        if n > 75:
            for f in faces:
                center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
                pts.append(center)
    
    pts = np.array(pts[:n])
    return pts


def _random_packed(n, rng, spacing=1.0):
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.9
    min_d = 0.55 * spacing
    max_attempts = 80000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    
    return np.array(pts)


def _optimize_lbfgs(pos, rho, max_iter=500, tol=1e-10):
    """Optimize using L-BFGS-B."""
    n = len(pos)
    
    def objective(x):
        return _morse_energy(x.reshape(n, 3), rho)
    
    def gradient(x):
        return morse_energy_gradient(x.reshape(n, 3), rho).reshape(-1)
    
    result = minimize(
        objective,
        pos.reshape(-1),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.success:
        return result.x.reshape(n, 3), result.fun
    else:
        return pos, _morse_energy(pos, rho)


def _optimize_bfgs(pos, rho, max_iter=300, tol=1e-10):
    """Optimize using BFGS."""
    n = len(pos)
    
    def objective(x):
        return _morse_energy(x.reshape(n, 3), rho)
    
    def gradient(x):
        return morse_energy_gradient(x.reshape(n, 3), rho).reshape(-1)
    
    result = minimize(
        objective,
        pos.reshape(-1),
        jac=gradient,
        method='BFGS',
        options={'maxiter': max_iter, 'gtol': 1e-8}
    )
    
    if result.success:
        return result.x.reshape(n, 3), result.fun
    else:
        return pos, _morse_energy(pos, rho)


def _optimize_relax(pos, rho):
    """Use the built-in relax function."""
    try:
        energy, relaxed_pos = relax(pos, 'morse', rho)
        return relaxed_pos, energy
    except Exception:
        return pos, _morse_energy(pos, rho)


def _basin_hopping_step(pos, rho, rng, kick_scale=0.3):
    """Perform a single basin hopping step: kick + local optimization."""
    # Random kick
    kicked_pos = pos + rng.normal(0, kick_scale, pos.shape)
    
    # Fix minimum distance
    kicked_pos = _fix_min_dist(kicked_pos)
    
    if not _min_dist_check(kicked_pos):
        return pos, _morse_energy(pos, rho)
    
    # Local optimization
    optimized_pos, energy = _optimize_lbfgs(kicked_pos, rho)
    
    if not _min_dist_check(optimized_pos):
        return pos, _morse_energy(pos, rho)
    
    return optimized_pos, energy


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin before 330s kill
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial seeds based on rho
    seeds = []
    
    if rho <= 3:
        # Long range: dense packing
        seeds.append(_fcc_cluster(size))
        seeds.append(_hcp_cluster(size))
        seeds.append(_bcc_cluster(size))
        seeds.append(_random_packed(size, rng))
        seeds.append(_random_packed(size, rng))
    elif rho <= 6:
        # Medium range: mix of FCC and icosahedral
        seeds.append(_fcc_cluster(size))
        seeds.append(_hcp_cluster(size))
        seeds.append(_icosahedron_shell(size))
        seeds.append(_random_packed(size, rng))
        seeds.append(_random_packed(size, rng))
    else:
        # Short range: icosahedral dominates
        seeds.append(_icosahedron_shell(size))
        seeds.append(_fcc_cluster(size))
        seeds.append(_hcp_cluster(size))
        seeds.append(_random_packed(size, rng))
        seeds.append(_random_packed(size, rng))
    
    # Ensure minimum distance for all seeds
    for i in range(len(seeds)):
        seeds[i] = _fix_min_dist(seeds[i])
        if not _min_dist_check(seeds[i]):
            seeds[i] = _random_packed(size, rng)
            seeds[i] = _fix_min_dist(seeds[i])
    
    # Initial optimization of each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try:
            # Try multiple optimization methods
            pos1, e1 = _optimize_lbfgs(seed, rho)
            if _min_dist_check(pos1) and e1 < best_energy:
                best_energy = e1
                best_pos = pos1.copy()
            
            pos2, e2 = _optimize_relax(seed, rho)
            if _min_dist_check(pos2) and e2 < best_energy:
                best_energy = e2
                best_pos = pos2.copy()
        except Exception:
            continue
    
    # Basin hopping from best configuration
    if best_pos is not None:
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        # Number of basin hopping steps based on time available
        max_bh_steps = 50
        kick_scales = [0.3, 0.2, 0.15, 0.1, 0.05]
        
        for step in range(max_bh_steps):
            if time.time() > deadline - 20:
                break
            
            # Choose kick scale (decrease over time)
            kick_scale = kick_scales[min(step // 10, len(kick_scales) - 1)]
            
            # Perform basin hopping step
            new_pos, new_energy = _basin_hopping_step(current_pos, rho, rng, kick_scale)
            
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Metropolis acceptance for escape
                delta_e = new_energy - current_energy
                if delta_e < 0.1 and rng.random() < np.exp(-delta_e / 0.05):
                    current_pos = new_pos
                    current_energy = new_energy
    
    # Additional random restarts if time permits
    if time.time() < deadline - 30:
        n_restarts = min(10, int((deadline - time.time()) / 10))
        for i in range(n_restarts):
            if time.time() > deadline - 20:
                break
            try:
                seed = _random_packed(size, rng)
                seed = _fix_min_dist(seed)
                if _min_dist_check(seed):
                    pos, energy = _optimize_lbfgs(seed, rho)
                    if _min_dist_check(pos) and energy < best_energy:
                        best_energy = energy
                        best_pos = pos.copy()
            except Exception:
                continue
    
    # Final refinement
    if best_pos is not None and time.time() < deadline - 10:
        try:
            refined_pos, refined_e = _optimize_lbfgs(best_pos, rho, max_iter=1000)
            if _min_dist_check(refined_pos) and refined_e < best_energy:
                best_energy = refined_e
                best_pos = refined_pos.copy()
        except Exception:
            pass
    
    # Fallback: ensure we have a valid result
    if best_pos is None:
        best_pos = _random_packed(size, rng)
        best_pos = _fix_min_dist(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
