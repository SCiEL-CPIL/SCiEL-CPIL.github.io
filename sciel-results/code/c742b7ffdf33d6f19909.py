"""
Strategy: Multi-start global optimization for Sutton-Chen transition-metal clusters.
Uses diverse structural templates, parallel basin-hopping with Metropolis acceptance,
and a final high-precision polish. Structural change: use a two-phase approach where
phase 1 uses a coarse-grained energy (truncated pair sum) to rapidly screen many
starts, and phase 2 refines the top candidates with full Sutton-Chen relaxation.
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 310.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _liquid_start(n, rng, spacing=1.0):
    """Generate a liquid-like random start with minimum distance."""
    radius = spacing * (n / 4.0) ** (1/3.0) * 2.8
    points = []
    min_dist = 0.55 * spacing
    max_attempts = n * 100
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        existing = np.array(points)
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _fcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, compactly centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    R = (3 * n * a**3 / (16 * np.pi)) ** (1/3) * 1.4
    n_side = int(np.ceil(R / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        R2 = R * 1.5
        n_side2 = int(np.ceil(R2 / a)) + 2
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _hcp_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 4
    n_side = int(np.ceil(np.sqrt(n))) + 4
    R2 = (n / 4.0) ** (2/3) * a**2 * 3.5
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift_x, shift_y = 0.0, 0.0
        else:
            shift_x = a / 2
            shift_y = a * np.sqrt(3) / 6
        
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift_x
                y = j * a * np.sqrt(3) / 2.0 + shift_y
                if x**2 + y**2 + z**2 <= R2:
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        while len(points) < n:
            p = rng.uniform(-3*a, 3*a, 3)
            points = np.vstack([points, p])
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None):
    """Random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * (n / 4.0) ** (1/3.0) * 2.5
    points = []
    min_dist = 0.65 * spacing
    max_attempts = n * 80
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        existing = np.array(points)
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Icosahedral arrangement: center + 12 + 42 + ... shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    if n > 1:
        shell1 = ico_verts * spacing
        for v in shell1:
            if len(points) < n:
                points.append(v)
    
    if n > 13:
        a = spacing * np.sqrt(2)
        R2 = 2.2 * spacing
        n_side = 5
        second_shell = []
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                for k in range(-n_side, n_side + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        r = np.linalg.norm(p)
                        if 1.4 * spacing < r < 2.6 * spacing:
                            second_shell.append(p / r * 2.0 * spacing)
        
        seen = set()
        unique_second = []
        for p in second_shell:
            key = tuple(np.round(p, 3))
            if key not in seen:
                seen.add(key)
                unique_second.append(p)
        
        for p in unique_second:
            if len(points) < n:
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-3.5*spacing, 3.5*spacing, 3)
        if np.linalg.norm(p) < 4.0 * spacing:
            points.append(p)
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _bcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * 2.0 / np.sqrt(3)
    R = (3 * n * a**3 / (8 * np.pi)) ** (1/3) * 1.4
    n_side = int(np.ceil(R / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        R2 = R * 1.5
        n_side2 = int(np.ceil(R2 / a)) + 2
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _perturb(positions, rng, scale=0.3):
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise

def _coarse_energy(positions, kind):
    """
    Fast coarse-grained energy: only nearest-neighbor pairs (within 2.5 sigma)
    with simplified Sutton-Chen. Used for rapid screening.
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask out self and far pairs
    mask = (dists > 0.5) & (dists < 2.5)
    
    if not np.any(mask):
        return 1e10
    
    # Get the relevant distances
    valid_dists = dists[mask]
    
    if len(valid_dists) == 0:
        return 1e10
    
    # Simplified: just use r^-12 - c*r^-6 type
    e_rep = np.sum(valid_dists ** -12)
    e_att = -144.41 * np.sum(valid_dists ** -6)
    
    return e_rep + e_att

def _basin_hopping(positions, potential, kind, rng, n_iters=100, base_scale=0.4, min_time=5.0, use_metropolis=True, temperature_scale=0.02):
    """Run basin hopping from initial positions with optional Metropolis acceptance."""
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy, current_pos = best_energy, best_pos
    
    for i in range(n_iters):
        if _time_left() < min_time:
            break
        
        scale = base_scale * (0.95 ** (i // 15)) + 0.08
        new_pos = _perturb(current_pos, rng, scale)
        
        try:
            e, pos = relax(new_pos, potential, kind)
            if not np.isfinite(e):
                continue
            if e < current_energy or (use_metropolis and rng.random() < np.exp(-(e - current_energy) / (temperature_scale * (1 + i/30)))):
                current_energy, current_pos = e, pos
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except:
            pass
    
    return best_energy, best_pos

def _high_precision_polish(positions, potential, kind):
    """Final high-precision polish pass with tighter tolerance."""
    try:
        from scipy.optimize import minimize
        
        def obj(x):
            pos = x.reshape(-1, 3)
            e, _ = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                return 1e10
            return e
        
        def grad(x):
            pos = x.reshape(-1, 3)
            _, g = sutton_chen_energy_gradient(pos, kind)
            if not np.all(np.isfinite(g)):
                return np.zeros_like(g).flatten()
            return g.flatten()
        
        x0 = positions.flatten()
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 500}
        )
        
        if result.fun < np.inf and np.all(np.isfinite(result.x)):
            final_pos = result.x.reshape(-1, 3)
            # Check minimum distance constraint
            min_d = np.inf
            for i in range(len(final_pos)):
                for j in range(i+1, len(final_pos)):
                    d = np.linalg.norm(final_pos[i] - final_pos[j])
                    if d < min_d:
                        min_d = d
            if min_d > 0.35:
                return result.fun, final_pos
        return relax(positions, potential, kind)
    except:
        return relax(positions, potential, kind)

def _screen_and_relax(positions, potential, kind, rng, time_budget=10.0):
    """
    Phase 1: Coarse screen. Phase 2: Full relax if passes.
    Returns (energy, positions) or (inf, positions) if screened out.
    """
    if _time_left() < time_budget:
        return np.inf, positions
    
    # Coarse energy check
    e_coarse = _coarse_energy(positions, kind)
    
    # If coarse energy is very bad, skip full relax
    # Threshold is relative: if e_coarse > 5 * typical, skip
    # Typical for these clusters is around -n * 1.5 to -n * 2.5
    n = len(positions)
    threshold = 5.0 * n * 2.0  # conservative
    
    if e_coarse > threshold:
        return np.inf, positions
    
    # Full relaxation
    try:
        e, pos = relax(positions, potential, kind)
        if not np.isfinite(e):
            return np.inf, positions
        return e, pos
    except:
        return np.inf, positions

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = np.inf
    best_positions = None
    
    templates = []
    
    try:
        fcc = _fcc_cluster(size, 1.0, rng)
        templates.append(fcc)
    except:
        pass
    
    try:
        hcp = _hcp_cluster(size, 1.0, rng)
        templates.append(hcp)
    except:
        pass
    
    try:
        ico = _icosahedral_shell(size, 1.0, rng)
        templates.append(ico)
    except:
        pass
    
    try:
        bcc = _bcc_cluster(size, 1.0, rng)
        templates.append(bcc)
    except:
        pass
    
    # More random starts for diversity
    for _ in range(12):
        try:
            rand = _random_compact(size, 1.0, rng)
            templates.append(rand)
        except:
            pass
    
    # Add liquid-like starts
    for _ in range(5):
        try:
            liq = _liquid_start(size, rng, 1.0)
            templates.append(liq)
        except:
            pass
    
    # Phase 1: Screen and relax each template in parallel (up to 4 threads)
    # We'll process them sequentially with coarse screening first
    screened_results = []
    
    for t in templates:
        if _time_left() < 20.0:
            break
        e, pos = _screen_and_relax(t, potential, kind, rng, time_budget=8.0)
        if np.isfinite(e) and e < best_energy:
            best_energy = e
            best_positions = pos
        if np.isfinite(e):
            screened_results.append((e, pos))
    
    if best_positions is None and templates:
        best_positions = templates[0]
        best_energy = np.inf
    
    # Sort screened results by energy
    screened_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from top candidates
    # Use parallel basin hopping with 4 threads
    n_top = min(6, len(screened_results))
    
    def _bh_task(args):
        e_init, pos_init, seed_offset = args
        local_rng = np.random.default_rng(seed_offset)
        n_iters = max(40, int((_time_left() - 30.0) * 2.0))
        n_iters = min(n_iters, 150)
        e, pos = _basin_hopping(
            pos_init, potential, kind, local_rng,
            n_iters=n_iters, base_scale=0.5, min_time=15.0, temperature_scale=0.02
        )
        return e, pos
    
    bh_tasks = []
    for i, (e_init, pos_init) in enumerate(screened_results[:n_top]):
        bh_tasks.append((e_init, pos_init, 1000 + i * 7919))
    
    # Also add basin hopping from the best found so far
    if best_positions is not None:
        bh_tasks.append((best_energy, best_positions, 2000))
    
    # Run basin hopping tasks
    # Use ThreadPoolExecutor with up to 4 workers
    max_workers = min(4, len(bh_tasks))
    
    if max_workers > 0 and _time_left() > 30.0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_bh_task, task): task for task in bh_tasks}
            for future in as_completed(futures):
                if _time_left() < 25.0:
                    break
                try:
                    e, pos = future.result(timeout=60)
                    if np.isfinite(e) and e < best_energy:
                        best_energy = e
                        best_positions = pos
                except:
                    pass
    
    # Additional basin hopping from other good templates if time allows
    if _time_left() > 50.0 and len(screened_results) > n_top:
        for e_t, pos_t in screened_results[n_top:n_top+4]:
            if _time_left() < 25.0:
                break
            if e_t > best_energy + 10.0:
                continue
            try:
                local_rng = np.random.default_rng(5000 + int(e_t * 100))
                e, pos = _basin_hopping(pos_t, potential, kind, local_rng, n_iters=30, base_scale=0.45, min_time=8.0, temperature_scale=0.015)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_positions = pos
            except:
                pass
    
    # Final high-precision polish
    if _time_left() > 10.0 and best_positions is not None:
        try:
            e_polished, pos_polished = _high_precision_polish(best_positions, potential, kind)
            if np.isfinite(e_polished) and e_polished < best_energy:
                best_energy = e_polished
                best_positions = pos_polished
        except:
            pass
    
    # Validate final result
    if best_positions is None:
        # Fallback: return a simple compact arrangement
        best_positions = _random_compact(size, 1.0, rng)
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except:
            pass
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        for i in range(len(best_positions)):
            for j in range(i+1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    # Push atoms apart
                    if d > 1e-10:
                        direction = (best_positions[i] - best_positions[j]) / d
                        shift = (0.35 - d) / 2 + 0.01
                        best_positions[i] += direction * shift
                        best_positions[j] -= direction * shift
                    else:
                        best_positions[i] += np.array([0.2, 0, 0])
                        best_positions[j] -= np.array([0.2, 0, 0])
    
    return best_positions
