"""Global Minima Search: LAMMPS-style cluster growth on True Sutton-Chen Potential."""
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


def _lammps_cluster_growth(n, kind, spacing, deadline):
    """
    LAMMPS-style cluster growth: build cluster atom by atom.
    For each new atom, find its optimal position by minimizing
    the energy contribution of that atom to the existing core.
    """
    start_time = time.time()
    
    # Start with 2 atoms at optimal spacing
    if n <= 2:
        pos = np.array([[0, 0, 0], [spacing, 0, 0]])[:n]
        return pos
    
    # Build up from a small seed
    current = np.array([[0, 0, 0], [spacing, 0, 0]], dtype=float)
    current_n = 2
    
    # For each additional atom, find optimal position
    for target_n in range(3, n + 1):
        if time.time() > deadline - 30.0:
            break
        
        # Generate candidate positions for the new atom
        # Use positions on a sphere around the cluster center of mass
        com = np.mean(current, axis=0)
        
        # Multiple strategies for placing the new atom
        candidates = []
        
        # Strategy 1: Closest to surface of existing cluster
        # Find the point on the cluster that is "most exposed"
        for i in range(min(current_n, 10)):
            atom = current[i]
            # Place new atom at various distances and angles from this atom
            for dist in [spacing * 0.9, spacing, spacing * 1.1]:
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    # Random direction
                    z = np.cos(angle)
                    r = np.sqrt(1 - z * z)
                    theta = angle * 1.618  # Golden angle for coverage
                    direction = np.array([r * np.cos(theta), r * np.sin(theta), z])
                    new_pos = atom + dist * direction
                    candidates.append(new_pos)
        
        # Strategy 2: Symmetric placements
        for axis in range(3):
            for sign in [1, -1]:
                new_pos = com + sign * spacing * np.eye(3)[axis]
                candidates.append(new_pos)
        
        # Strategy 3: Fill in gaps - place atom where density is lowest
        if current_n > 4:
            # Sample points on a sphere and pick the one with lowest local density
            num_samples = 20
            for _ in range(num_samples):
                z = 2 * np.random.random() - 1
                theta = 2 * np.pi * np.random.random()
                r = np.sqrt(1 - z * z)
                direction = np.array([r * np.cos(theta), r * np.sin(theta), z])
                new_pos = com + spacing * 1.2 * direction
                candidates.append(new_pos)
        
        candidates = np.array(candidates)
        
        # Relax each candidate with the core and pick the best
        best_e = np.inf
        best_pos = None
        
        # Limit candidates to avoid timeout
        max_candidates = min(len(candidates), 30)
        rng = np.random.default_rng(target_n)
        indices = rng.choice(len(candidates), size=max_candidates, replace=False)
        
        for idx in indices:
            if time.time() > deadline - 25.0:
                break
            
            candidate = candidates[idx]
            new_cluster = np.vstack([current, candidate[None, :]])
            
            # Quick validation
            if not _validate_positions(new_cluster):
                continue
            
            # Relax the full cluster
            e, p = _try_relax(new_cluster, 'sutton_chen', kind)
            
            if p is not None and e < best_e:
                best_e = e
                best_pos = p
        
        if best_pos is None:
            # Fallback: add atom at a reasonable default position
            default_pos = com + spacing * np.array([1, 0, 0])
            new_cluster = np.vstack([current, default_pos[None, :]])
            if _validate_positions(new_cluster):
                e, p = _try_relax(new_cluster, 'sutton_chen', kind)
                if p is not None:
                    best_pos = p
                else:
                    best_pos = new_cluster
            else:
                # Try another position
                default_pos = com + spacing * np.array([0, 1, 0])
                new_cluster = np.vstack([current, default_pos[None, :]])
                if _validate_positions(new_cluster):
                    e, p = _try_relax(new_cluster, 'sutton_chen', kind)
                    if p is not None:
                        best_pos = p
                    else:
                        best_pos = new_cluster
                else:
                    default_pos = com + spacing * np.array([0, 0, 1])
                    new_cluster = np.vstack([current, default_pos[None, :]])
                    best_pos = new_cluster
        else:
            # Use the relaxed position
            current = best_pos
            current_n = target_n
    
    # Final relaxation of the complete cluster
    if time.time() < deadline - 10.0:
        e, p = _try_relax(current, 'sutton_chen', kind)
        if p is not None:
            current = p
    
    return current


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

    # 3. LAMMPS-style cluster growth as primary search strategy
    # This builds the cluster atom-by-atom, which is more systematic than basin hopping
    if _check_time():
        try:
            grown = _lammps_cluster_growth(n, kind, spacing, deadline)
            if grown is not None and len(grown) == n and _validate_positions(grown):
                e_grown, p_grown = _try_relax(grown, potential, kind)
                if p_grown is not None:
                    _update_best(e_grown, p_grown)
        except Exception:
            pass
    
    # 4. Basin Hopping from the best local minimum found (if time remains)
    if best_positions is not None and _check_time():
        current_pos = best_positions.copy()
        current_energy = best_energy
        step_size = 0.5
        min_step = 0.2
        max_step = 1.0
        T = 0.5
        
        # Run BH for remaining time, but limit steps
        max_bh_steps = 30
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
                if T > 1e-6:
                    prob = np.exp(-delta_e / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                current_pos = new_pos
                current_energy = new_energy
                _update_best(current_energy, current_pos)
                if delta_e < 0:
                    step_size = min(max_step, step_size * 1.05)
                else:
                    step_size = max(min_step, step_size * 0.95)
            else:
                step_size = max(min_step, step_size * 0.95)

    # 5. Final Polish
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
