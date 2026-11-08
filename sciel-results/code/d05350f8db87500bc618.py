"""
Strategy: Topology-preserving basin hopping for Sutton-Chen transition-metal clusters.
Uses neighbor-list constraints to perform structured isomer walks,
combined with parallel multi-start from diverse templates.
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 320.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _get_neighbor_list(positions, cutoff=None):
    """Compute neighbor list with a cutoff distance."""
    n = len(positions)
    if cutoff is None:
        # Use 1.5x the mean nearest-neighbor distance
        dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        cutoff = 1.5 * np.mean(nn_dists)
    
    neighbors = []
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    for i in range(n):
        nbrs = np.where((dists[i] < cutoff) & (np.arange(n) != i))[0]
        neighbors.append(set(nbrs.tolist()))
    return neighbors

def _topology_perturb(positions, neighbors, rng, scale=0.3):
    """
    Perturb positions while trying to preserve neighbor topology.
    Atoms that are neighbors get smaller perturbations; non-neighbors get larger.
    This biases the walk toward topology-preserving moves.
    """
    n = len(positions)
    new_pos = positions.copy()
    
    for i in range(n):
        nbr_set = neighbors[i]
        # Smaller perturbation for atoms with well-defined local environment
        local_scale = scale * 0.7 if len(nbr_set) > 0 else scale * 1.3
        noise = rng.normal(0, local_scale, 3)
        new_pos[i] += noise
    
    # Additional constraint: try to maintain approximate distances to neighbors
    # by doing a gentle relaxation step that preserves topology
    for i in range(n):
        nbrs = list(neighbors[i])
        if not nbrs:
            continue
        # Compute desired displacement to maintain average neighbor distance
        avg_dist = np.mean([np.linalg.norm(positions[i] - positions[j]) for j in nbrs])
        current_avg = np.mean([np.linalg.norm(new_pos[i] - new_pos[j]) for j in nbrs])
        if current_avg > 0:
            correction = (avg_dist - current_avg) / current_avg * 0.3
            for j in nbrs:
                diff = new_pos[i] - new_pos[j]
                d = np.linalg.norm(diff)
                if d > 1e-10:
                    new_pos[i] += diff / d * correction * 0.5
                    new_pos[j] -= diff / d * correction * 0.5
    
    return new_pos

def _topology_basin_hopping(positions, potential, kind, rng, n_iters=80, base_scale=0.4, min_time=5.0, temperature_scale=0.02):
    """
    Basin hopping with topology-preserving perturbations.
    Maintains a neighbor list and uses it to bias perturbations.
    """
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy, current_pos = best_energy, best_pos
    neighbors = _get_neighbor_list(current_pos)
    
    for i in range(n_iters):
        if _time_left() < min_time:
            break
        
        scale = base_scale * (0.95 ** (i // 15)) + 0.08
        new_pos = _topology_perturb(current_pos, neighbors, rng, scale)
        
        try:
            e, pos = relax(new_pos, potential, kind)
            if not np.isfinite(e):
                continue
            
            # Check if topology changed significantly
            new_neighbors = _get_neighbor_list(pos)
            topo_change = sum(len(neighbors[k].symmetric_difference(new_neighbors[k])) for k in range(len(pos))) / (2 * len(pos))
            
            # Metropolis acceptance with topology-aware temperature
            eff_temp = temperature_scale * (1 + i/30) * (1 + topo_change * 5)
            if e < current_energy or (rng.random() < np.exp(-(e - current_energy) / eff_temp)):
                current_energy, current_pos = e, pos
                neighbors = new_neighbors
            if e < best_energy:
                best_energy, best_pos = e, pos
        except:
            pass
    
    return best_energy, best_pos

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

def _parallel_basin_hopping(positions_list, potential, kind, rng, n_workers=3, n_iters_each=30, base_scale=0.45, temperature_scale=0.015):
    """
    Run multiple basin-hopping chains in parallel using ThreadPoolExecutor.
    Each chain gets its own RNG stream.
    """
    best_energy = np.inf
    best_pos = None
    
    def _run_chain(idx, pos_init):
        local_rng = np.random.default_rng(rng.integers(0, 2**31) + idx)
        e, p = _topology_basin_hopping(
            pos_init, potential, kind, local_rng,
            n_iters=n_iters_each, base_scale=base_scale,
            min_time=5.0, temperature_scale=temperature_scale
        )
        return e, p
    
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i, pos in enumerate(positions_list):
                if _time_left() < 15.0:
                    break
                futures.append(executor.submit(_run_chain, i, pos))
            
            for future in as_completed(futures):
                if _time_left() < 10.0:
                    break
                try:
                    e, p = future.result(timeout=60)
                    if np.isfinite(e) and e < best_energy:
                        best_energy = e
                        best_pos = p
                except:
                    pass
    except:
        pass
    
    return best_energy, best_pos

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
    for _ in range(8):
        try:
            rand = _random_compact(size, 1.0, rng)
            templates.append(rand)
        except:
            pass
    
    # Add liquid-like starts
    for _ in range(3):
        try:
            liq = _liquid_start(size, rng, 1.0)
            templates.append(liq)
        except:
            pass
    
    # Relax each template to get initial energies
    relaxed_templates = []
    for t in templates:
        if _time_left() < 20.0:
            break
        try:
            e, pos = relax(t, potential, kind)
            if np.isfinite(e):
                relaxed_templates.append((e, pos))
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
        except:
            pass
    
    if best_positions is None and templates:
        best_positions = templates[0]
        best_energy = np.inf
    
    # Sort by energy and pick top candidates for parallel topology-preserving BH
    relaxed_templates.sort(key=lambda x: x[0])
    top_candidates = [pos for e, pos in relaxed_templates[:min(6, len(relaxed_templates))]]
    
    # Parallel topology-preserving basin hopping
    if _time_left() > 40.0 and len(top_candidates) > 0:
        n_workers = min(3, len(top_candidates))
        e_par, pos_par = _parallel_basin_hopping(
            top_candidates, potential, kind, rng,
            n_workers=n_workers,
            n_iters_each=max(20, int((_time_left() - 30.0) / 3.0)),
            base_scale=0.45,
            temperature_scale=0.015
        )
        if np.isfinite(e_par) and e_par < best_energy:
            best_energy = e_par
            best_positions = pos_par
    
    # Sequential topology-preserving BH from best structure
    if _time_left() > 20.0 and best_positions is not None:
        n_bh = max(30, int((_time_left() - 20.0) * 2.5))
        e_seq, pos_seq = _topology_basin_hopping(
            best_positions, potential, kind, rng,
            n_iters=n_bh, base_scale=0.4,
            min_time=8.0, temperature_scale=0.018
        )
        if np.isfinite(e_seq) and e_seq < best_energy:
            best_energy = e_seq
            best_positions = pos_seq
    
    # Additional standard BH from second-best for diversity
    if _time_left() > 30.0 and len(relaxed_templates) > 1:
        e2, pos2 = relaxed_templates[1]
        if e2 < best_energy + 3.0:
            n_bh2 = max(20, int((_time_left() - 25.0) * 1.5))
            e_alt, pos_alt = _basin_hopping(
                pos2, potential, kind, rng,
                n_iters=n_bh2, base_scale=0.4,
                min_time=8.0, temperature_scale=0.015
            )
            if np.isfinite(e_alt) and e_alt < best_energy:
                best_energy = e_alt
                best_positions = pos_alt
    
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
