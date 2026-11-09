"""Parallel multi-start L-BFGS-B on true Sutton-Chen potential with diverse seeds and final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

_MAX_N = 80
_NTHREADS = 4


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                sites.append(base + np.array([0, 0, z]))
                sites.append(base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 1.0 < d < 2.0:
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42, min_dist_frac=0.7):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_d = min_dist_frac * spacing
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempt += 1
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
        else:
            base = np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _try_relax(pos, potential, kind):
    """Relax positions using the provided relax function."""
    try:
        if not _validate_positions(pos):
            return np.inf, None
        energy, relaxed = relax(pos, potential, kind)
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def _parallel_relax(seeds, potential, kind, deadline):
    """
    Run L-BFGS-B relaxations in parallel on multiple seeds.
    Returns list of (energy, positions) for valid results.
    """
    results = []
    remaining_time = deadline - time.time()
    if remaining_time <= 5:
        return results
    
    n_seeds = len(seeds)
    if n_seeds == 0:
        return results
    
    # Allocate time per seed based on number of seeds and threads
    per_seed_time = max(2.0, remaining_time * 0.8 / max(1, n_seeds // _NTHREADS + 1))
    
    def relax_one(idx_pos):
        idx, pos = idx_pos
        start = time.time()
        e, p = _try_relax(pos, potential, kind)
        elapsed = time.time() - start
        return (e, p, elapsed)
    
    with ThreadPoolExecutor(max_workers=_NTHREADS) as executor:
        futures = {executor.submit(relax_one, (i, pos)): i for i, pos in enumerate(seeds)}
        for future in as_completed(futures):
            if time.time() > deadline - 10:
                break
            try:
                e, p, elapsed = future.result()
                if p is not None and np.isfinite(e):
                    results.append((e, p))
            except Exception:
                pass
    
    return results


def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin: evaluator kills at 330s

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    rng = np.random.default_rng(42)

    # Generate diverse structural seeds - focus on quality over quantity
    candidates = []
    
    # Core structural prototypes at multiple scales
    for scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
        candidates.append(_fcc_cluster(n, spacing * scale))
        candidates.append(_hcp_cluster(n, spacing * scale))
        candidates.append(_bcc_cluster(n, spacing * scale))
        candidates.append(_icosahedral_cluster(n, spacing * scale))
    
    # Add random compact seeds for diversity
    for seed in range(1, 10):
        for md_frac in [0.7, 0.8, 0.9]:
            pos = _random_compact(n, spacing, seed=seed * 100 + int(md_frac * 100), min_dist_frac=md_frac)
            if _validate_positions(pos):
                candidates.append(pos)
    
    # Deduplicate: remove seeds that are very close to existing ones
    unique_candidates = [candidates[0]]
    for pos in candidates[1:]:
        if not _validate_positions(pos):
            continue
        is_dup = False
        for existing in unique_candidates:
            # Check if minimum distance between any pair is too small
            diff = pos - existing
            min_d = np.min(np.linalg.norm(diff, axis=1))
            if min_d < 0.5 * spacing:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(pos)
    
    candidates = unique_candidates[:20]  # Limit to 20 seeds for parallel processing

    best_energy = np.inf
    best_positions = None

    # Phase 1: Parallel multi-start L-BFGS-B on true potential
    results = _parallel_relax(candidates, potential, kind, deadline)
    
    for e, p in results:
        if e < best_energy:
            best_energy = e
            best_positions = p.copy()

    # Phase 2: If we have time, do additional random starts from best region
    if best_positions is not None and time.time() < deadline - 20:
        # Generate jittered versions of the best solution
        jitter_seeds = []
        for sigma in [0.15, 0.25, 0.4]:
            for _ in range(3):
                jittered = best_positions + rng.normal(0, sigma, best_positions.shape)
                if _validate_positions(jittered):
                    jitter_seeds.append(jittered)
        
        if jitter_seeds:
            results2 = _parallel_relax(jitter_seeds, potential, kind, deadline)
            for e, p in results2:
                if e < best_energy:
                    best_energy = e
                    best_positions = p.copy()

    # Phase 3: High-precision final polish
    if best_positions is not None and time.time() < deadline - 5:
        try:
            # Use tighter tolerances for final polish
            from scipy.optimize import minimize
            
            n_exp, m_exp, c_val = _KIND_PARAMS[kind]
            
            def sc_energy_grad(x):
                x = x.reshape(n, 3)
                i_idx, j_idx = np.triu_indices(n, k=1)
                diff = x[i_idx] - x[j_idx]
                r2 = np.sum(diff * diff, axis=1)
                r2 = np.maximum(r2, 1e-12)
                r = np.sqrt(r2)
                
                r_inv_n = r ** (-n_exp)
                E_rep = np.sum(r_inv_n)
                
                r_inv_m = r ** (-m_exp)
                S = np.zeros(n)
                np.add.at(S, i_idx, r_inv_m)
                np.add.at(S, j_idx, r_inv_m)
                sqrt_S = np.sqrt(np.maximum(S, 1e-30))
                E_att = -c_val * np.sum(sqrt_S)
                E = E_rep + E_att
                
                grad = np.zeros((n, 3))
                coeff_rep = -n_exp * r ** (-n_exp - 2)
                forces_rep = coeff_rep[:, np.newaxis] * diff
                np.add.at(grad, i_idx, forces_rep)
                np.add.at(grad, j_idx, -forces_rep)
                
                inv_2sqrt_S = np.zeros(n)
                mask = S > 1e-20
                inv_2sqrt_S[mask] = 1.0 / (2.0 * np.sqrt(S[mask]))
                
                coeff_att_base = c_val * m_exp * r ** (-m_exp - 2)
                weights = coeff_att_base * (inv_2sqrt_S[i_idx] + inv_2sqrt_S[j_idx])
                forces_att = weights[:, np.newaxis] * diff
                np.add.at(grad, i_idx, forces_att)
                np.add.at(grad, j_idx, -forces_att)
                
                return E, grad.reshape(-1)
            
            res = minimize(
                fun=sc_energy_grad,
                x0=best_positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            
            if np.isfinite(res.fun):
                polished = res.x.reshape(n, 3)
                if _validate_positions(polished) and res.fun < best_energy:
                    best_energy = res.fun
                    best_positions = polished.copy()
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        pos = _random_compact(n, spacing, seed=999)
        e, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos

    return best_positions
# EVOLVE-BLOCK-END
