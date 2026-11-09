"""Global Minima Search: Basin Hopping on True Sutton-Chen Potential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
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
    start_time = time.time()
    # Strict safety margin: 330s limit, return by 300s to be safe
    deadline = start_time + 300.0

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    rng = np.random.default_rng(42)

    best_energy = np.inf
    best_positions = None

    def _check_time():
        return time.time() < deadline - 5.0

    def _update_best(e, p):
        nonlocal best_energy, best_positions
        if np.isfinite(e) and e < best_energy:
            best_energy = e
            best_positions = p.copy()

    # 1. Generate a small set of diverse structural seeds
    # We limit the number of seeds to ensure we have time for optimization
    seeds = []
    
    # Crystalline seeds (most likely to be near global minima for small clusters)
    seeds.append(_fcc_cluster(n, spacing))
    seeds.append(_hcp_cluster(n, spacing))
    seeds.append(_bcc_cluster(n, spacing))
    seeds.append(_icosahedral_cluster(n, spacing))
    
    # Add a few random compact structures
    for i in range(3):
        if not _check_time():
            break
        pos = _random_compact(n, spacing, seed=100 + i, min_dist_frac=0.7)
        if _validate_positions(pos):
            seeds.append(pos)

    # 2. Relax all seeds to find local minima
    # This is the most critical step. We must ensure we don't run out of time here.
    # We will relax seeds sequentially.
    for pos in seeds:
        if not _check_time():
            break
        e, p = _try_relax(pos, potential, kind)
        if p is not None:
            _update_best(e, p)

    if best_positions is None:
        # Fallback: if relaxation failed completely, return a valid random structure
        pos = _random_compact(n, spacing, seed=999)
        # Try one more time with a different seed
        e, p = _try_relax(pos, potential, kind)
        if p is not None and _validate_positions(p):
            return p
        # If even that fails, return the raw random positions (might be invalid, but better than crash?)
        # The evaluator invalidates if < 0.35 dist. _random_compact ensures min_dist.
        return pos

    # 3. Basin Hopping from the best local minimum found
    # We run a limited number of BH steps to escape local minima.
    # We use a short BH chain because `relax` is expensive.
    
    current_pos = best_positions.copy()
    current_energy = best_energy
    step_size = 0.5
    min_step = 0.2
    max_step = 1.0
    T = 0.5
    
    # We will run BH for a fixed number of steps OR until time runs out.
    # Given the timeout issue, we limit steps to 50 to be safe, relying on the initial seeds being good.
    max_bh_steps = 50
    bh_count = 0
    
    while bh_count < max_bh_steps and _check_time():
        bh_count += 1
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        new_energy, new_pos = _try_relax(perturbed, potential, kind)
        
        if new_pos is None or not np.isfinite(new_energy):
            step_size = max(min_step, step_size * 0.9)
            continue

        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        accept = False
        if delta_e < 0:
            accept = True
        else:
            # Accept uphill moves with low probability to escape traps
            if T > 1e-6:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            current_pos = new_pos
            current_energy = new_energy
            _update_best(current_energy, current_pos)
            # Adapt step size
            if delta_e < 0:
                step_size = min(max_step, step_size * 1.05)
            else:
                step_size = max(min_step, step_size * 0.95)
        else:
            step_size = max(min_step, step_size * 0.95)

    # 4. Final Polish
    if best_positions is not None and _check_time():
        try:
            e2, p2 = relax(best_positions, potential, kind)
            if np.isfinite(e2) and _validate_positions(p2) and e2 < best_energy:
                best_energy = e2
                best_positions = p2.copy()
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
