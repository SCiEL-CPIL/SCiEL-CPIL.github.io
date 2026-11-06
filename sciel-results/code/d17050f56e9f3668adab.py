"""Dzugutov cluster global minima: parameterized icosahedral manifold + parallel basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 280.0  # safety margin under 300s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return TIME_LIMIT - _elapsed()


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    rv = r[iu]
    if rv.size == 0:
        return 0.0
    e = np.zeros_like(rv)
    m1 = rv < _a
    m2 = (rv >= _a) & (rv < _b)
    if m1.any():
        r1 = rv[m1]
        e[m1] = _A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a))
    if m2.any():
        r2 = rv[m2]
        e[m2] = _B * np.exp(_d / (r2 - _b))
    return float(np.sum(e))


def _relax(positions, time_left=None):
    """Relax using the provided relaxer; fall back to our own if it fails/slow."""
    pos = np.asarray(positions, dtype=float)
    try:
        energy, out = relax(pos, "dzugutov")
        if out is not None:
            out = np.asarray(out, dtype=float)
            if out.shape == pos.shape and np.all(np.isfinite(out)):
                return float(energy), out
    except Exception:
        pass
    # Fallback: adaptive gradient descent
    p = pos.copy()
    best_e = _energy(p)
    best_p = p.copy()
    for it in range(600):
        if time_left is not None and _remaining() < 10.0:
            break
        try:
            g = dzugutov_energy_gradient(p)
        except Exception:
            break
        g = np.asarray(g, dtype=float)
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        s = min(0.3, 0.05 * gn)
        new_p = p - s * g / gn
        if not np.all(np.isfinite(new_p)):
            s *= 0.1
            new_p = p - s * g / gn
            if not np.all(np.isfinite(new_p)):
                break
        e_new = _energy(new_p)
        if e_new < best_e:
            best_e = e_new
            best_p = new_p.copy()
            p = new_p
        else:
            break
    return best_e, best_p


def _polish(positions):
    """High-precision polish."""
    p = np.asarray(positions, dtype=float).copy()
    best_e = _energy(p)
    best_p = p.copy()
    for it in range(400):
        if _remaining() < 5.0:
            break
        try:
            g = dzugutov_energy_gradient(p)
        except Exception:
            break
        g = np.asarray(g, dtype=float)
        gn = np.linalg.norm(g)
        if gn < 1e-14:
            break
        s = min(0.1, 0.01 * gn)
        new_p = p - s * g / gn
        if not np.all(np.isfinite(new_p)):
            break
        e_new = _energy(new_p)
        if e_new < best_e:
            best_e = e_new
            best_p = new_p.copy()
            p = new_p
        else:
            break
    return best_e, best_p


def _icosahedron_vertices():
    """Standard icosahedron vertices (12), normalized to unit edge length."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    edge = np.linalg.norm(v[0] - v[1])
    return v / edge


def _icosahedral_directions(n_dirs):
    """Generate n_dirs unit vectors with icosahedral-like distribution."""
    if n_dirs <= 12:
        ico = _icosahedron_vertices()
        if n_dirs == 12:
            return ico
        # Subsample
        idx = np.linspace(0, 11, n_dirs, dtype=int)
        return ico[idx]
    
    # For more directions, use a combination of icosahedral shells
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = _icosahedron_vertices()
    
    # Shell 1: 12 vertices
    dirs = list(ico)
    
    # Shell 2: 30 edge midpoints + 20 face centers = 50 directions
    edges = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 11],
        [1, 2], [1, 5], [1, 6], [1, 8],
        [2, 3], [2, 5], [2, 11],
        [3, 7], [3, 9], [3, 11],
        [4, 9], [4, 10], [4, 11],
        [5, 6], [5, 7],
        [6, 7], [6, 10],
        [7, 9], [7, 10],
        [8, 6], [8, 10],
        [9, 10],
    ]
    for e in edges:
        c = (ico[e[0]] + ico[e[1]]) / 2.0
        c /= np.linalg.norm(c)
        dirs.append(c)
    
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 11], [0, 11, 2], [0, 2, 1],
        [1, 2, 5], [2, 11, 3], [11, 4, 9], [4, 8, 10], [8, 1, 6],
        [5, 2, 3], [3, 11, 7], [7, 9, 3], [9, 4, 10], [10, 8, 6],
        [6, 1, 5], [5, 3, 7], [7, 9, 10], [10, 6, 5],
    ]
    for f in faces:
        c = (ico[f[0]] + ico[f[1]] + ico[f[2]]) / 3.0
        c /= np.linalg.norm(c)
        dirs.append(c)
    
    dirs = np.array(dirs)
    if len(dirs) >= n_dirs:
        return dirs[:n_dirs]
    
    # Fill with Fibonacci sphere
    remaining = n_dirs - len(dirs)
    fib_pts = []
    for i in range(remaining):
        z = 1 - 2 * (i + 0.5) / remaining
        r = np.sqrt(1 - z * z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        fib_pts.append([r * np.cos(theta), r * np.sin(theta), z])
    fib_pts = np.array(fib_pts)
    fib_pts /= np.linalg.norm(fib_pts, axis=1, keepdims=True)
    return np.vstack([dirs, fib_pts])


def _build_parameterized_icosahedral(n, shell_radii=None, twist_angles=None, scale=1.0):
    """
    Build an n-atom cluster using a parameterized icosahedral manifold.
    shell_radii: list of shell radii [r1, r2, r3, ...]
    twist_angles: list of twist angles for each shell [t1, t2, t3, ...]
    """
    if shell_radii is None:
        # Default Mackay-like shell radii
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        shell_radii = [1.0, 1.0 + 1.0/phi, 1.0 + 2.0/phi, 1.0 + 3.0/phi]
    
    if twist_angles is None:
        twist_angles = [0.0] * len(shell_radii)
    
    # Shell atom counts: 1, 12, 42, 92, 120, ...
    shell_counts = [1, 12, 42, 92, 120, 180]
    
    pts = []
    for i, radius in enumerate(shell_radii):
        if len(pts) >= n:
            break
        count = shell_counts[i] if i < len(shell_counts) else 120
        need = min(count, n - len(pts))
        
        if need <= 0:
            continue
        
        if i == 0:
            # Center atom
            pts.append(np.zeros(3) * scale)
        else:
            # Get directions for this shell
            dirs = _icosahedral_directions(need)
            # Apply twist
            if twist_angles[i] != 0.0:
                # Rotate around z-axis
                theta = twist_angles[i]
                R = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                dirs = dirs @ R.T
            
            for d in dirs:
                pts.append(d * radius * scale)
    
    # If we still need more atoms, add on outer shell
    while len(pts) < n:
        outer_r = shell_radii[-1] * scale
        rng = np.random.default_rng(len(pts) * 7 + n)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * outer_r * 1.05)
    
    return np.array(pts[:n])


def _build_icosahedral(n, scale=1.0, rng=None):
    """Build an n-atom Mackay icosahedral cluster."""
    return _build_parameterized_icosahedral(n, scale=scale)


def _fcc_cluster(n, scale=1.0):
    """Build an FCC-like cluster."""
    pts = [np.zeros(3)]
    a = scale
    s = a / np.sqrt(2.0)
    shell1 = []
    for (x, y, z) in [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                      (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]:
        shell1.append(np.array([x, y, z]) * s)
    for p in shell1:
        if len(pts) < n:
            pts.append(p)
    s2 = a
    shell2 = []
    for (x, y, z) in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
                      (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0),
                      (0, 0, 2), (0, 0, -2)]:
        shell2.append(np.array([x, y, z]) * a / 2.0)
    rng = np.random.default_rng(0)
    for p in shell2:
        if len(pts) < n:
            pts.append(p)
    while len(pts) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * s2 * 1.6)
    return np.array(pts[:n])


def _random_compact(n, spacing=1.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(n)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    pts = []
    tries = 0
    while len(pts) < n and tries < 4000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.75 * spacing for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:n])


def _seeds(n):
    """List of candidate starting geometries for cluster size n."""
    rng = np.random.default_rng(1000 + n)
    seeds = []
    
    # Primary: proper Mackay icosahedral
    try:
        seeds.append(_build_icosahedral(n, scale=1.0))
    except Exception:
        pass
    
    # Parameterized icosahedral with varied shell radii and twists
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base_radii = [1.0, 1.0 + 1.0/phi, 1.0 + 2.0/phi, 1.0 + 3.0/phi]
    
    # Vary shell radii slightly
    for dr1 in (-0.05, 0.0, 0.05):
        for dr2 in (-0.05, 0.0, 0.05):
            radii = [base_radii[0], base_radii[1] + dr1, base_radii[2] + dr2, base_radii[3]]
            try:
                seeds.append(_build_parameterized_icosahedral(n, shell_radii=radii, scale=1.0))
            except Exception:
                pass
    
    # Vary twist angles
    for t1 in (-0.1, 0.0, 0.1):
        for t2 in (-0.1, 0.0, 0.1):
            twists = [0.0, t1, t2, 0.0]
            try:
                seeds.append(_build_parameterized_icosahedral(n, shell_radii=base_radii, twist_angles=twists, scale=1.0))
            except Exception:
                pass
    
    # Slightly perturbed icosahedral structures at various scales
    for scale in (0.92, 0.96, 1.04, 1.08):
        try:
            seeds.append(_build_icosahedral(n, scale=scale))
        except Exception:
            pass
    
    # Add perturbed versions
    if seeds:
        base = seeds[0]
        for amp in (0.03, 0.07, 0.12):
            perturbed = base + rng.normal(0, amp, size=base.shape)
            seeds.append(perturbed)
    
    # fcc / polycrystalline
    try:
        seeds.append(_fcc_cluster(n, scale=1.0))
    except Exception:
        pass
    
    # random compact
    seeds.append(_random_compact(n, spacing=1.1, rng=rng))
    
    return seeds


def _basin_hop(positions, n, time_budget, rng):
    """Basin hopping: iteratively perturb the best structure and re-relax."""
    pos = np.asarray(positions, dtype=float).copy()
    best_e = np.inf
    best_pos = pos.copy()

    try:
        e, p = _relax(pos)
        if np.all(np.isfinite(p)):
            best_e = e
            best_pos = p.copy()
    except Exception:
        best_e = _energy(pos)
        best_pos = pos.copy()

    amp = 0.12
    t = 0.8
    t_decay = 0.96

    while _remaining() > 20.0:
        noise = rng.normal(0, amp, size=pos.shape)
        perturbed = best_pos + noise

        # Ensure minimum separation
        for i in range(n):
            for j in range(i + 1, n):
                d = perturbed[i] - perturbed[j]
                dist = np.linalg.norm(d)
                if dist < 0.5 and dist > 1e-10:
                    push = (0.5 - dist) / dist * 0.5
                    perturbed[i] += push * d
                    perturbed[j] -= push * d
                elif dist <= 1e-10:
                    v = rng.normal(size=3)
                    v /= np.linalg.norm(v)
                    perturbed[i] += 0.25 * v
                    perturbed[j] -= 0.25 * v

        try:
            e_new, p_new = _relax(perturbed)
        except Exception:
            continue

        if not np.all(np.isfinite(p_new)):
            continue

        if e_new < best_e:
            best_e = e_new
            best_pos = p_new.copy()
            amp = min(amp * 1.08, 0.35)
        else:
            delta = e_new - best_e
            if rng.random() < np.exp(-delta / (t + 1e-10)):
                best_pos = p_new.copy()
            amp *= 0.94

        t *= t_decay

    return best_e, best_pos


def _parallel_search(seed_list, n, time_budget, n_threads=4):
    """Parallel basin hopping from multiple seeds."""
    results = []
    rng_base = np.random.default_rng(12345 + n)
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {}
        for i, seed in enumerate(seed_list[:n_threads]):
            rng = np.random.default_rng(rng_base.integers(0, 2**32) + i)
            future = executor.submit(_basin_hop, seed, n, time_budget, rng)
            futures[future] = (i, seed)
        
        for future in as_completed(futures, timeout=max(1, time_budget - 5)):
            i, seed = futures[future]
            try:
                e, p = future.result(timeout=1)
                results.append((e, p))
            except Exception:
                pass
    
    if not results:
        return None
    
    results.sort(key=lambda x: x[0])
    return results[0]


def construct(instance):
    """Structure-based search: parameterized icosahedral manifold + parallel basin hopping + polish."""
    size = instance["n"]
    potential = instance["potential"]
    best_e = np.inf
    best_pos = None
    rng = np.random.default_rng(42 + size)

    seeds = _seeds(size)
    
    # Phase 1: Relax all seeds to find the best initial basin
    seed_results = []
    for s in seeds:
        if _remaining() < 40.0:
            break
        s = np.asarray(s, dtype=float)
        try:
            e, p = _relax(s)
            if np.all(np.isfinite(p)):
                seed_results.append((e, p.copy()))
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
        except Exception:
            continue

    if best_pos is None:
        best_pos = _build_icosahedral(size, scale=1.0)
        try:
            _, best_pos = relax(best_pos, potential)
        except Exception:
            pass
        best_e = _energy(best_pos)
        seed_results = [(best_e, best_pos.copy())]

    # Sort seed results by energy
    seed_results.sort(key=lambda x: x[0])

    # Phase 2: Parallel basin hopping from top seeds
    if _remaining() > 60.0 and len(seed_results) >= 2:
        # Use top 4 seeds for parallel basin hopping
        top_seeds = [sr[1] for sr in seed_results[:4]]
        try:
            result = _parallel_search(top_seeds, size, _remaining() - 30.0, n_threads=4)
            if result is not None:
                bh_e, bh_pos = result
                if bh_e < best_e:
                    best_e = bh_e
                    best_pos = bh_pos
        except Exception:
            pass

    # Phase 3: Sequential basin hopping from best seed if time permits
    if _remaining() > 40.0:
        try:
            bh_e, bh_pos = _basin_hop(seed_results[0][1], size, _remaining() - 25.0, rng)
            if bh_e < best_e:
                best_e = bh_e
                best_pos = bh_pos
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if _remaining() > 10.0:
        try:
            pol_e, pol_pos = _polish(best_pos)
            if pol_e < best_e:
                best_e = pol_e
                best_pos = pol_pos
        except Exception:
            pass

    # final safety: ensure finite
    best_pos = np.asarray(best_pos, dtype=float)
    if not np.all(np.isfinite(best_pos)):
        best_pos = _build_icosahedral(size, scale=1.0)
    return best_pos
# EVOLVE-BLOCK-END
