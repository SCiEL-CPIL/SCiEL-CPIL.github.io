"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 285.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = c * k
                p1 = np.array([x, y, z])
                candidates.append((np.linalg.norm(p1), p1))
                
                x2 = a * (i + 0.5 * j + 1.0 / np.sqrt(3.0))
                y2 = a * np.sqrt(3) / 2 * (j + 1.0 / 3.0)
                z2 = c * (k + 1.0 / 3.0)
                p2 = np.array([x2, y2, z2])
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a cluster with icosahedral symmetry (Mackay-like)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    pts.extend(verts.tolist())
    
    # Add edge midpoints (12 edges)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.5:
                edges.append((i, j))
    
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid = mid / np.linalg.norm(mid) * 1.05
        pts.append(mid)
    
    # Add face centroids
    while len(pts) < n:
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    if len(pts) >= n:
                        break
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 0.5 and d2 < 0.5 and d3 < 0.5:
                        centroid = (verts[i] + verts[j] + verts[k]) / 3
                        centroid = centroid / np.linalg.norm(centroid) * 1.15
                        pts.append(centroid)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) < n:
            for v in verts:
                if len(pts) >= n:
                    break
                pts.append(v * 1.3)
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, radius=2.0):
    """Generate n random points within a sphere."""
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=(20, 3))
        norms = np.linalg.norm(p, axis=1)
        valid = p[norms <= 1.0]
        for v in valid:
            if len(pts) < n:
                pts.append(v * radius)
    return np.array(pts[:n])


def _generate_deformed(n, rng, base='fcc'):
    """Generate a deformed version of a base structure."""
    if base == 'fcc':
        pos = _generate_fcc(n)
    elif base == 'bcc':
        pos = _generate_bcc(n)
    elif base == 'hcp':
        pos = _generate_hcp(n)
    else:
        pos = _generate_fcc(n)
    
    # Apply random deformation
    scale = 0.8 + 0.4 * rng.random()
    pos = pos * scale
    pos += rng.normal(0, 0.1, pos.shape)
    return pos


def _polish(positions, kind):
    """High-precision polish pass."""
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if not np.isfinite(energy):
            return None, np.inf
        
        # Additional refinement
        for _ in range(20):
            e, g = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                break
            if np.max(np.abs(g)) < 1e-12:
                break
            lr = 0.0001
            new_pos = pos - lr * g
            if _is_valid(new_pos):
                pos = new_pos
            else:
                break
        
        energy, pos = relax(pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return pos, energy
    except Exception:
        pass
    return None, np.inf


def _basin_hopping(positions, kind, rng, n_steps=50, max_time=60.0):
    """
    Basin hopping with adaptive noise scale.
    """
    start_t = time.time()
    current_energy, current_pos = relax(positions, 'sutton_chen', kind)
    best_energy, best_pos = current_energy, current_pos
    
    for step in range(n_steps):
        if time.time() - start_t > max_time:
            break
        
        # Adaptive noise: start small, increase if stuck
        base_noise = 0.05 + 0.05 * (step // 10)
        noise_scale = base_noise * (1.0 + 0.3 * rng.random())
        perturbed = current_pos + rng.normal(0, noise_scale, current_pos.shape)
        
        if not _is_valid(perturbed):
            continue
        
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not np.isfinite(new_energy) or not _is_valid(new_pos):
            continue
        
        # Metropolis acceptance at low temperature
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
        else:
            # Occasionally accept worse moves to escape local minima
            if rng.random() < 0.1:
                current_energy = new_energy
                current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos
    
    return best_pos, best_energy


def _differential_evolution(positions, kind, rng, n_pop=10, n_gen=50, max_time=80.0):
    """
    Differential evolution for global optimization.
    """
    start_t = time.time()
    n = len(positions)
    dim = n * 3
    
    # Initialize population
    pop = [positions.copy()]
    for i in range(n_pop - 1):
        p = positions + rng.normal(0, 0.3, positions.shape)
        if _is_valid(p):
            pop.append(p)
        else:
            pop.append(positions.copy())
    
    # Evaluate initial population
    energies = []
    for p in pop:
        try:
            e, _ = relax(p, 'sutton_chen', kind)
            if np.isfinite(e):
                energies.append(e)
            else:
                energies.append(np.inf)
        except Exception:
            energies.append(np.inf)
    
    best_idx = np.argmin(energies)
    best_energy = energies[best_idx]
    best_pos = pop[best_idx].copy()
    
    for gen in range(n_gen):
        if time.time() - start_t > max_time:
            break
        
        for i in range(n_pop):
            # Select three distinct individuals
            a, b, c = rng.choice(n_pop, size=3, replace=False)
            
            # Differential mutation
            mutant = pop[a] + 0.8 * (pop[b] - pop[c])
            
            # Clip to reasonable bounds
            mutant = np.clip(mutant, -5, 5)
            
            if not _is_valid(mutant):
                continue
            
            try:
                new_energy, new_pos = relax(mutant, 'sutton_chen', kind)
            except Exception:
                continue
            
            if not np.isfinite(new_energy) or not _is_valid(new_pos):
                continue
            
            # Selection
            if new_energy < energies[i]:
                energies[i] = new_energy
                pop[i] = new_pos
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    return best_pos, best_energy


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse initial structures
    base_structures = []
    base_structures.append(_generate_fcc(n))
    base_structures.append(_generate_bcc(n))
    base_structures.append(_generate_hcp(n))
    base_structures.append(_generate_icosahedron(n))
    base_structures.append(_generate_random_compact(n, rng, radius=2.0))
    base_structures.append(_generate_deformed(n, rng, 'fcc'))
    base_structures.append(_generate_deformed(n, rng, 'hcp'))
    
    # Relax all base structures
    relaxed_bases = []
    for base in base_structures:
        if _time_left() < 30.0:
            break
        try:
            energy, relaxed_pos = relax(base, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                relaxed_bases.append(relaxed_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Strategy 1: Differential evolution from best base
    if best_pos is not None and _time_left() > 60.0:
        try:
            de_pos, de_energy = _differential_evolution(
                best_pos, kind, rng, n_pop=8, n_gen=40, max_time=50.0
            )
            if _is_valid(de_pos) and de_energy < best_energy:
                best_energy = de_energy
                best_pos = de_pos
        except Exception:
            pass
    
    # Strategy 2: Basin hopping from multiple bases
    if best_pos is not None and _time_left() > 50.0:
        try:
            bh_pos, bh_energy = _basin_hopping(
                best_pos, kind, rng, n_steps=40, max_time=40.0
            )
            if _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            pass
    
    # Strategy 3: More basin hopping from other bases
    for base in relaxed_bases[1:]:
        if _time_left() < 30.0:
            break
        try:
            bh_pos, bh_energy = _basin_hopping(
                base, kind, rng, n_steps=20, max_time=20.0
            )
            if _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            continue
    
    # Strategy 4: Perturbed restarts
    max_restarts = 30
    for i in range(max_restarts):
        if _time_left() < 20.0:
            break
        
        if relaxed_bases:
            base = relaxed_bases[i % len(relaxed_bases)].copy()
        elif best_pos is not None:
            base = best_pos.copy()
        else:
            base = base_structures[i % len(base_structures)].copy()
        
        noise_scale = 0.05 + 0.1 * ((i // 5) % 4)
        pos = base + rng.normal(0, noise_scale, base.shape)
        
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Final polish
    if best_pos is not None and _time_left() > 15.0:
        try:
            polished_pos, polished_energy = _polish(best_pos, kind)
            if polished_pos is not None and polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
