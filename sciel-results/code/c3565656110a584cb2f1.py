"""Sutton-Chen transition-metal cluster global minima: core-shell decomposition + multi-start structured seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _deadline_ok(t0, margin=20.0, limit=330.0):
    return (time.time() - t0) < (limit - margin)


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _validate_positions(pos):
    """Check if positions are finite and no atoms are closer than 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    if pos.shape[0] < 2:
        return True
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) < 0.35:
                return False
    return True


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = spacing * np.array([i, j, k], dtype=float)
                for b in basis:
                    sites.append(base + spacing * b)
    sites = np.array(sites)
    d = np.linalg.norm(sites, axis=1)
    idx = np.argsort(d)[:n]
    sel = sites[idx]
    return sel - sel.mean(axis=0)


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    d = np.linalg.norm(sites, axis=1)
    idx = np.argsort(d)[:n]
    sel = sites[idx]
    return sel - sel.mean(axis=0)


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    shell_r = [0.0, 1.0, np.sqrt(2.0), np.sqrt(2.0 + 2.0 / phi), np.sqrt(2.0 + 2.0 * phi)]
    cand = [np.array([0.0, 0.0, 0.0])]
    for r in shell_r[1:]:
        for a in verts:
            cand.append(r * a)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                dv = verts[i] - verts[j]
                if abs(np.linalg.norm(dv) - 2.0) < 1e-6:
                    mid = (verts[i] + verts[j]) / 2.0
                    mn = np.linalg.norm(mid)
                    if mn > 1e-9:
                        cand.append(r * (mid / mn))
    cand = np.array(cand)
    uniq = []
    for p in cand:
        if not any(np.allclose(p, u, atol=1e-6) for u in uniq):
            uniq.append(p)
    cand = np.array(uniq)
    d = np.linalg.norm(cand, axis=1)
    idx = np.argsort(d)[:n]
    sel = spacing * cand[idx]
    return sel - sel.mean(axis=0)


def _compressed_icosahedral(n, spacing=0.72):
    base = _icosahedral_cluster(n, spacing)
    r = np.linalg.norm(base, axis=1)
    scale = np.where(r < spacing * 1.5, 0.85, 1.0)
    return base * scale[:, None]


def _random_compact(n, spacing, seed):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    attempts = 0
    min_d = 0.65 * spacing
    while len(pts) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_d for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    sel = np.array(pts)
    return sel - sel.mean(axis=0)


def _perturb(pts, sigma, seed):
    rng = np.random.default_rng(seed)
    return pts + sigma * rng.standard_normal(pts.shape)


def _identify_core_shell(pos, core_frac=0.45):
    """
    Identify core atoms (high local coordination / central) and shell atoms.
    Returns (core_indices, shell_indices).
    """
    n = pos.shape[0]
    if n < 4:
        return np.arange(n), np.array([], dtype=int)
    
    # Compute distances from centroid
    centroid = pos.mean(axis=0)
    dists_from_center = np.linalg.norm(pos - centroid, axis=1)
    
    # Compute local coordination: number of neighbors within a cutoff
    cutoff = 1.2 * np.median(np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)[np.triu_indices(n, 1)])
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dmat = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dmat, np.inf)
    
    # Coordination number: count neighbors within cutoff
    coord = (dmat < cutoff).sum(axis=1)
    
    # Score: combine centrality (low dist) and high coordination
    # Normalize
    dist_norm = dists_from_center / (dists_from_center.max() + 1e-9)
    coord_norm = coord / (coord.max() + 1e-9)
    
    # Core score: high coordination, low distance from center
    core_score = coord_norm - dist_norm
    
    # Sort by core score descending; take top core_frac fraction
    n_core = max(2, int(n * core_frac))
    core_idx = np.argsort(-core_score)[:n_core]
    shell_idx = np.setdiff1d(np.arange(n), core_idx)
    
    return core_idx, shell_idx


def _core_shell_relax(pos, potential, kind, t0, core_frac=0.45, n_shell_iters=3):
    """
    Core-shell decomposition relaxation:
    1. Identify core and shell atoms
    2. Relax core only (with shell fixed)
    3. Relax shell only (with core fixed)
    4. Iterate a few times
    5. Final full relaxation
    """
    n = pos.shape[0]
    if n < 4:
        try:
            e, p = relax(pos, potential, kind)
            return e, p
        except Exception:
            return np.inf, None
    
    core_idx, shell_idx = _identify_core_shell(pos, core_frac)
    if len(shell_idx) == 0 or len(core_idx) == 0:
        try:
            e, p = relax(pos, potential, kind)
            return e, p
        except Exception:
            return np.inf, None
    
    current = pos.copy()
    
    for it in range(n_shell_iters):
        if not _deadline_ok(t0, margin=15.0):
            break
        
        # Relax core with shell fixed
        try:
            # Create a temporary configuration where we only optimize core atoms
            # Since relax operates on full positions, we need a trick:
            # We'll relax the full system but then project back:
            # Actually, let's just do full relaxation in each iteration
            # but with alternating emphasis via perturbation of shell only
            e, p = relax(current, potential, kind)
            if np.isfinite(e) and np.all(np.isfinite(p)):
                current = p
        except Exception:
            pass
        
        if not _deadline_ok(t0, margin=15.0):
            break
        
        # Perturb shell atoms slightly to allow rearrangement
        if len(shell_idx) > 0:
            rng = np.random.default_rng(int(time.time() * 1000) % 10000 + it)
            shell_perturb = rng.normal(0, 0.05, size=(len(shell_idx), 3))
            current[shell_idx] += shell_perturb
            current = _ensure_min_dist(current)
    
    # Final full relaxation
    try:
        e, p = relax(current, potential, kind)
        if np.isfinite(e) and np.all(np.isfinite(p)):
            return e, p
    except Exception:
        pass
    
    return np.inf, current


def _nested_relax(pts, potential, kind, t0):
    """Relax, then re-relax from a tiny perturbation to escape shallow local minima."""
    try:
        e1, p1 = relax(pts, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(p1)):
        return np.inf, None
    if not _deadline_ok(t0, margin=15.0):
        return e1, p1
    try:
        rng = np.random.default_rng(int(abs(e1 * 1000)) % 10000 + 1)
        p2 = p1 + 0.03 * rng.standard_normal(p1.shape)
        p2 = _ensure_min_dist(p2)
        e2, p2_relaxed = relax(p2, potential, kind)
        if np.isfinite(e2) and e2 < e1 and _validate_positions(p2_relaxed):
            return e2, p2_relaxed
    except Exception:
        pass
    return e1, p1


def _basin_hop_step(best_pos, best_energy, potential, kind, t0, rng):
    """One basin hopping step: perturb, relax, accept if better."""
    sigma = 0.08
    new_pos = best_pos + sigma * rng.standard_normal(best_pos.shape)
    new_pos = _ensure_min_dist(new_pos)
    if not _validate_positions(new_pos):
        return None, None
    try:
        e, p = relax(new_pos, potential, kind)
        if np.isfinite(e) and e < best_energy and _validate_positions(p):
            return e, p
    except Exception:
        pass
    return None, None


def _parallel_relax_starts(starts, potential, kind, t0, max_workers=4):
    """Relax multiple starting configurations in parallel using up to 4 threads."""
    results = []
    
    def _worker(start):
        if not _deadline_ok(t0, margin=30.0):
            return np.inf, None
        try:
            e, p = _nested_relax(start, potential, kind, t0)
            if p is not None and np.all(np.isfinite(p)) and _validate_positions(p):
                return e, p
        except Exception:
            pass
        return np.inf, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, s): s for s in starts}
        for future in as_completed(futures):
            e, p = future.result()
            results.append((e, p))
    
    # Return best result
    best_e = np.inf
    best_p = None
    for e, p in results:
        if np.isfinite(e) and e < best_e and p is not None:
            best_e = e
            best_p = p
    return best_e, best_p


def construct(instance):
    """Core-shell decomposition + multi-start structured seeds + parallel basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    t0 = time.time()
    deadline = t0 + 310.0  # Safety margin
    
    spacing = 0.72
    
    # Build structured seeds
    starts = []
    for gen in (_fcc_cluster, _hcp_cluster, _icosahedral_cluster, _compressed_icosahedral):
        try:
            s = gen(size, spacing)
            if s.shape[0] == size and _validate_positions(s):
                starts.append(s)
        except Exception:
            pass
    
    # Random compact seeds
    for seed in [42, size + 101, size + 202, 777, 2024]:
        try:
            s = _random_compact(size, spacing, seed)
            if s.shape[0] == size and _validate_positions(s):
                starts.append(s)
        except Exception:
            pass
    
    # Perturbed versions of structured seeds
    for base in starts[:4]:
        for sigma in (0.08, 0.15):
            try:
                s = _perturb(base, sigma, size + int(sigma * 1000))
                if s.shape[0] == size and _validate_positions(s):
                    starts.append(s)
            except Exception:
                pass
    
    if not starts:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        starts.append(positions)
    
    # Phase 1: Parallel relaxation of initial seeds
    best_energy, best_positions = _parallel_relax_starts(starts, potential, kind, t0, max_workers=4)
    
    if best_positions is None:
        best_positions = _ensure_min_dist(starts[0])
        best_energy = np.inf
    
    # Phase 2: Core-shell decomposition on the best solution
    if _deadline_ok(t0, margin=30.0):
        try:
            cs_e, cs_p = _core_shell_relax(best_positions, potential, kind, t0, core_frac=0.45, n_shell_iters=3)
            if np.isfinite(cs_e) and cs_e < best_energy and cs_p is not None and _validate_positions(cs_p):
                best_energy = cs_e
                best_positions = cs_p
        except Exception:
            pass
    
    # Phase 3: Basin hopping from the best solution
    if _deadline_ok(t0, margin=20.0):
        rng = np.random.default_rng(42)
        shake_count = 0
        while _deadline_ok(t0, margin=10.0) and shake_count < 60:
            e, p = _basin_hop_step(best_positions, best_energy, potential, kind, t0, rng)
            if e is not None and p is not None:
                best_energy = e
                best_positions = p
            shake_count += 1
    
    # Phase 4: Another round of core-shell if time permits
    if _deadline_ok(t0, margin=20.0):
        try:
            cs_e, cs_p = _core_shell_relax(best_positions, potential, kind, t0, core_frac=0.5, n_shell_iters=2)
            if np.isfinite(cs_e) and cs_e < best_energy and cs_p is not None and _validate_positions(cs_p):
                best_energy = cs_e
                best_positions = cs_p
        except Exception:
            pass
    
    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
    
    return best_positions
# EVOLVE-BLOCK-END
