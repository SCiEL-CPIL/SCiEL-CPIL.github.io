"""Morse cluster global minima: basin hopping with L-BFGS refinement and diverse starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Energy and gradient of the Morse potential (vectorized)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    energy = float(np.sum(e))
    
    # dE/dr_ij = 2*rho*ex*(ex-1)
    de_dr = 2.0 * rho * ex * (ex - 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = de_dr / r_ij
    factor = np.where(r_ij > 1e-10, factor, 0.0)
    
    # Build weight matrix
    W = np.zeros((n, n))
    ii, jj = iu
    W[ii, jj] = factor
    W[jj, ii] = -factor
    
    row_sum = W.sum(axis=1)
    grad = pos * row_sum[:, None] - W @ pos
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return float(np.min(r)) >= min_dist - 1e-9


def _repair_distances(pos, min_dist=0.5):
    """Nudge overlapping atoms apart to satisfy minimum distance constraint."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(30):
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(r, np.inf)
        too_close = r < min_dist
        if not np.any(too_close):
            break
        for i in range(n):
            for j in range(i + 1, n):
                if too_close[i, j]:
                    d = diff[i, j]
                    r_ij = r[i, j]
                    if r_ij < 1e-10:
                        d = np.array([1.0, 0.0, 0.0])
                        r_ij = 1.0
                    push = (min_dist - r_ij + 1e-6) / 2.0
                    unit = d / r_ij
                    pos[i] += unit * push
                    pos[j] -= unit * push
    return pos


def _local_minimize(pos, rho, deadline):
    """Local minimization using L-BFGS-B; fall back to relax."""
    pos = pos.astype(np.float64)
    
    def fun_g(x):
        if time.time() > deadline:
            return 1e30, np.zeros_like(x)
        e, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
        return e, g
    
    try:
        res = minimize(fun_g, pos.ravel(), method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8,
                                "maxfun": 5000})
        newpos = res.x.reshape(-1, 3)
        if _is_valid(newpos):
            return newpos
    except Exception:
        pass
    
    # Fallback to provided relax
    try:
        e, newpos = relax(pos, "morse", rho)
        if _is_valid(newpos):
            return newpos
    except Exception:
        pass
    return pos


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    a = spacing
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    R = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 2
    coords = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    coords.append((a * (i + dx), a * (j + dy), a * (k + dz)))
    coords = np.array(coords)
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if k % 2 == 0:
                    p1 = np.array([i, j, k]) * a
                    p2 = np.array([i + 0.5, j + 0.5, k]) * a
                else:
                    p1 = np.array([i + 1/3.0, j + 2/3.0, k]) * a
                    p2 = np.array([i + 2/3.0, j + 1/3.0, k]) * a
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= np.mean(points, axis=0)
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    a = spacing
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    R = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 2
    coords = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    coords.append((a * (i + dx), a * (j + dy), a * (k + dz)))
    coords = np.array(coords)
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _generate_icosahedral(n, spacing=1.0):
    """Generate an icosahedral cluster with Mackay-like shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]  # center
    shell_radii = [1.0, np.sqrt(2.0 + 2.0 / phi), 2.0]
    
    # Shell 1: 12 vertices
    for v in verts:
        points.append(shell_radii[0] * v * spacing)
    if len(points) >= n:
        return np.array(points[:n]) - np.mean(points[:n], axis=0)
    
    # Shell 2: 30 points - use permutations of (0, 1, phi) type directions
    shell2_dirs = []
    for perm in [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                 (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                 (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
                 (1, 0, phi), (-1, 0, phi), (1, 0, -phi), (-1, 0, -phi),
                 (0, phi, 1), (0, -phi, 1), (0, phi, -1), (0, -phi, -1),
                 (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0),
                 (1, phi, 0), (-1, -phi, 0), (phi, 0, 1), (-phi, 0, -1),
                 (1, 0, -phi), (-1, 0, phi)]:
        p = np.array(perm, dtype=float)
        norm = np.linalg.norm(p)
        if norm > 1e-6:
            shell2_dirs.append(p / norm)
    # Deduplicate
    shell2_dirs = np.unique(np.round(shell2_dirs, 8), axis=0)
    for d in shell2_dirs:
        points.append(shell_radii[1] * d * spacing)
    if len(points) >= n:
        return np.array(points[:n]) - np.mean(points[:n], axis=0)
    
    # Shell 3: face centers of icosahedron (20 faces)
    shell3_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                p = verts[i] + verts[j] + verts[k]
                norm = np.linalg.norm(p)
                if norm > 1e-6:
                    shell3_dirs.append(p / norm)
    shell3_dirs = np.unique(np.round(shell3_dirs, 8), axis=0)
    for d in shell3_dirs:
        points.append(shell_radii[2] * d * spacing)
    
    # Pad if needed
    while len(points) < n:
        rng = np.random.default_rng(len(points))
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        points.append(shell_radii[2] * d * spacing * rng.uniform(0.95, 1.05))
    
    points = np.array(points[:n])
    return points - points.mean(axis=0)


def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster with minimum separation."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_sep = 0.8 * spacing
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_sep:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _basin_hopping(pos, rho, max_steps=30, step_size=0.3, rng=None, deadline=None):
    """Perform basin hopping with adaptive step size and Metropolis acceptance."""
    if rng is None:
        rng = np.random.default_rng()
    
    energy, pos = _morse_energy_and_grad(pos, rho)
    current_step = step_size
    accept_count = 0
    total_count = 0
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        if total_count > 5:
            frac = accept_count / total_count
            if frac > 0.6:
                current_step *= 0.85
            elif frac < 0.2:
                current_step *= 1.15
            current_step = np.clip(current_step, 0.1, 1.0)
        
        delta = rng.normal(0, current_step, size=pos.shape)
        new_pos = pos + delta
        
        if not _is_valid(new_pos):
            new_pos = _repair_distances(new_pos)
        
        new_energy, _ = _morse_energy_and_grad(new_pos, rho)
        total_count += 1
        
        if new_energy < energy:
            energy, pos = new_energy, new_pos
            accept_count += 1
        else:
            delta_e = new_energy - energy
            if delta_e < 20.0:
                prob = np.exp(-delta_e / 1.5)
                if rng.random() < prob:
                    energy, pos = new_energy, new_pos
                    accept_count += 1
    
    # Final local relaxation with L-BFGS
    if deadline is None or time.time() < deadline:
        pos = _local_minimize(pos, rho, deadline)
        energy, _ = _morse_energy_and_grad(pos, rho)
    
    return energy, pos


def construct(instance):
    """Multi-restart basin hopping with diverse initializations and L-BFGS refinement."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline with 40s safety margin
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting configurations
    starts = []
    
    for gen_fn in [_generate_fcc, _generate_hcp, _generate_bcc, _generate_icosahedral]:
        try:
            pts = gen_fn(size, spacing=1.0)
            if len(pts) == size:
                starts.append(pts)
        except Exception:
            pass
    
    for i in range(4):
        try:
            rand = _generate_random_compact(size, spacing=1.0, rng=rng)
            starts.append(rand)
        except Exception:
            pass
    
    if starts:
        for i in range(2):
            try:
                base = starts[0].copy()
                base += rng.normal(0, 0.3, size=base.shape)
                base = _repair_distances(base)
                starts.append(base)
            except Exception:
                pass
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Relax all starts using L-BFGS to find good local minima
    relaxed_starts = []
    for start_pos in starts:
        if time.time() > deadline:
            break
        try:
            relaxed_pos = _local_minimize(start_pos, parameter, deadline)
            energy, _ = _morse_energy_and_grad(relaxed_pos, parameter)
            relaxed_starts.append((energy, relaxed_pos))
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # Phase 2: Basin hopping from the best relaxed starts
    relaxed_starts.sort(key=lambda x: x[0])
    n_bh = min(len(relaxed_starts), 5)
    
    for idx in range(n_bh):
        if time.time() > deadline:
            break
        e0, pos0 = relaxed_starts[idx]
        try:
            bh_energy, bh_pos = _basin_hopping(pos0, parameter, max_steps=25,
                                                step_size=0.3, rng=rng, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            continue
    
    # Phase 3: If time remains, do additional basin hopping from best
    if best_pos is not None and time.time() < deadline - 15:
        try:
            bh_energy, bh_pos = _basin_hopping(best_pos, parameter, max_steps=20,
                                                step_size=0.2, rng=rng, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS if time permits
    if best_pos is not None and time.time() < deadline - 5:
        try:
            polished = _local_minimize(best_pos, parameter, deadline)
            e_polished, _ = _morse_energy_and_grad(polished, parameter)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pos = polished
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        try:
            rand = _generate_random_compact(size, spacing=1.0, rng=rng)
            best_pos = _repair_distances(rand)
            best_pos = _local_minimize(best_pos, parameter, deadline)
        except Exception:
            best_pos = np.zeros((size, 3))
            best_pos[:, 0] = np.arange(size) * 1.0
    
    return best_pos
# EVOLVE-BLOCK-END
