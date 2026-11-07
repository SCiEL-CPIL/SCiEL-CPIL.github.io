"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _scale_to_spacing(pts, spacing=0.72):
    if len(pts) < 2:
        return pts
    rmax = np.max(np.linalg.norm(pts, axis=1))
    if rmax > 0:
        target_r = spacing * len(pts) ** (1.0 / 3.0) * 0.55
        pts = pts * (target_r / rmax)
    return pts


def _make_fcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_bcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_mackay(n, kind, rng):
    """Generate a cluster by sequential growth, mimicking icosahedral/decahedral hierarchy."""
    if n <= 13:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
            [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
            [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
        verts /= np.linalg.norm(verts[0])
        pts = np.vstack([np.zeros(3), verts])
        return _scale_to_spacing(pts[:n])

    # Start with 13-atom icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    current_pts = np.vstack([np.zeros(3), verts])
    current_pts = _scale_to_spacing(current_pts)
    
    # Determine potential parameters
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41

    for k in range(13, n):
        if _time_left() < 5.0:
            break
            
        # Generate candidates
        candidates = []
        
        # 1. Midpoints of shortest bonds
        N = len(current_pts)
        dists = np.linalg.norm(current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        flat_idx = np.argsort(dists, axis=None)[:10]
        i_idx, j_idx = np.unravel_index(flat_idx, (N, N))
        for i, j in zip(i_idx, j_idx):
            mid = (current_pts[i] + current_pts[j]) / 2
            candidates.append(mid)
            
        # 2. Random positions
        radius = np.max(np.linalg.norm(current_pts, axis=1))
        for _ in range(5):
            r = radius + rng.uniform(0.2, 0.5)
            theta = rng.uniform(0, 2 * np.pi)
            phi_ang = rng.uniform(-np.pi/2, np.pi/2)
            p = np.array([r * np.cos(phi_ang) * np.cos(theta), 
                          r * np.cos(phi_ang) * np.sin(theta), 
                          r * np.sin(phi_ang)])
            candidates.append(p)
            
        candidates = np.array(candidates)
        
        # Precompute S_j for existing cluster
        dists_matrix = np.linalg.norm(current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists_matrix, np.inf)
        dists_matrix_safe = np.where(dists_matrix < 1e-6, 1e-6, dists_matrix)
        
        inv_r_m = dists_matrix_safe ** -m_exp
        S_j = np.sum(inv_r_m, axis=1)
        
        # Calculate Delta E for each candidate
        dists_jP = np.linalg.norm(current_pts[:, np.newaxis, :] - candidates[np.newaxis, :, :], axis=2)
        dists_jP_safe = np.where(dists_jP < 1e-6, 1e-6, dists_jP)
        
        inv_r_jP_n = dists_jP_safe ** -n_exp
        inv_r_jP_m = dists_jP_safe ** -m_exp
        
        sum_inv_r_n_new = np.sum(inv_r_jP_n, axis=0)
        sum_inv_r_m_new = np.sum(inv_r_jP_m, axis=0)
        term_new = 0.5 * sum_inv_r_n_new - c * np.sqrt(sum_inv_r_m_new)
        
        S_j_expanded = S_j[:, np.newaxis]
        delta_existing = 0.5 * inv_r_jP_n - c * (np.sqrt(S_j_expanded + inv_r_jP_m) - np.sqrt(S_j_expanded))
        
        sum_delta_existing = np.sum(delta_existing, axis=0)
        
        total_delta = term_new + sum_delta_existing
        
        best_idx = np.argmin(total_delta)
        best_candidate = candidates[best_idx]
        
        current_pts = np.vstack([current_pts, best_candidate])
        
    return _scale_to_spacing(current_pts)


def _make_decahedral(n):
    """Decahedral cluster: FCC with 5-fold twinning approximation."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    # Apply slight icosahedral distortion
    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Small perturbation to break cubic symmetry
    pts[:, 0] += 0.05 * np.cos(5 * theta) * r
    pts[:, 1] += 0.05 * np.sin(5 * theta) * r
    return _scale_to_spacing(pts)


def _make_random_compact(n, rng, spacing=0.72):
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale=0.1):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop_adaptive(positions, kind, rng, n_hops=10, base_scale=0.15):
    """Adaptive basin hopping with step size adjustment."""
    current = positions.copy()
    current_energy = np.inf
    try:
        current_energy, _ = relax(current, 'sutton_chen', kind)
    except:
        pass

    scale = base_scale
    for _ in range(n_hops):
        if _time_left() < 5.0:
            break
        perturbed = current + rng.normal(0, scale, current.shape)
        if not _is_valid(perturbed):
            scale *= 0.7
            continue
        try:
            energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < current_energy:
                    current = relaxed
                    current_energy = energy
                    scale = min(scale * 1.2, 0.3)
                else:
                    scale *= 0.8
        except Exception:
            scale *= 0.8
    return current, current_energy


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None

    starts = []

    # Generate diverse starting structures
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            starts.append(fcc)
            for s in [0.03, 0.07, 0.12, 0.2]:
                pert = _perturb(fcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            starts.append(bcc)
            for s in [0.05, 0.1, 0.15]:
                pert = _perturb(bcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        mackay = _make_mackay(n, kind, rng)
        if _is_valid(mackay):
            starts.append(mackay)
            for s in [0.03, 0.07, 0.12, 0.18]:
                pert = _perturb(mackay, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            starts.append(deca)
            for s in [0.05, 0.1, 0.15]:
                pert = _perturb(deca, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    # Random compact starts
    for _ in range(6):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            starts.append(rc)

    # Relax each start
    for i, start in enumerate(starts):
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # If we have time, do adaptive basin hopping from the best result
    if best_positions is not None and _time_left() > 25.0:
        try:
            bh_result, bh_energy = _basin_hop_adaptive(best_positions, kind, rng, n_hops=12, base_scale=0.12)
            energy, positions = relax(bh_result, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Additional basin hopping from perturbed best if time permits
    if best_positions is not None and _time_left() > 40.0:
        try:
            pert_best = _perturb(best_positions, rng, 0.15)
            bh2, _ = _basin_hop_adaptive(pert_best, kind, rng, n_hops=8, base_scale=0.1)
            energy, positions = relax(bh2, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # More basin hopping from second-best perturbation
    if best_positions is not None and _time_left() > 55.0:
        try:
            pert_best2 = _perturb(best_positions, rng, 0.2)
            bh3, _ = _basin_hop_adaptive(pert_best2, kind, rng, n_hops=6, base_scale=0.08)
            energy, positions = relax(bh3, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
