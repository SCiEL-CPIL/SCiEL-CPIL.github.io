# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s safety margin before 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_spacing(kind):
    """
    Calculate approximate equilibrium bond length for the pair part of Sutton-Chen.
    """
    if kind == '12-6':
        n, m, c = 12, 6, 144.41
    elif kind == '9-6':
        n, m, c = 9, 6, 39.432
    elif kind == '10-8':
        n, m, c = 10, 8, 34.408
    else:
        return 1.0
    return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))

def _fcc_positions(n, a=1.0):
    """Generate n atoms in FCC lattice positions (truncated)."""
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    dist = np.linalg.norm(pos)
                    sites.append((dist, pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    """Generate n atoms in HCP lattice positions (truncated)."""
    c = a * np.sqrt(8/3)
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([i * a, j * a * np.sqrt(3)/2, z])
                sites.append((np.linalg.norm(pos), pos))
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([(i + 0.5) * a, (j + 1/6) * a * np.sqrt(3), z + c/2])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _bcc_positions(n, a=1.0):
    """Generate n atoms in BCC lattice positions (truncated)."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    dist = np.linalg.norm(pos)
                    sites.append((dist, pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _icosahedral_positions(n, a=1.0):
    """
    Generate n atoms in an icosahedral cluster (Mackay icosahedron truncated).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a / np.sqrt(1 + phi**2)
    
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v)
    
    if n > 13:
        faces = [
            [0, 1, 8], [0, 2, 9], [1, 3, 8], [2, 4, 9],
            [1, 0, 5], [3, 2, 6], [0, 4, 5], [2, 6, 7],
            [5, 8, 10], [6, 9, 11], [8, 1, 10], [9, 3, 11],
            [10, 5, 7], [11, 6, 4], [8, 10, 0], [9, 11, 2],
            [5, 7, 1], [6, 4, 3], [10, 8, 0], [11, 9, 2]
        ]
        for f in faces:
            if len(positions) >= n:
                break
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            positions.append(fc)
        
        if n > 33:
            edges = []
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(verts[i] - verts[j])
                    if d < 2.1 * a / np.sqrt(1+phi**2):
                        edges.append((i, j))
            for e in edges:
                if len(positions) >= n:
                    break
                em = (verts[e[0]] + verts[e[1]]) / 2.0
                positions.append(em)
    
    while len(positions) < n:
        idx = np.random.randint(0, 12)
        pos = verts[idx] + np.random.normal(0, 0.1 * a, 3)
        positions.append(pos)
    
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    """Random rejection-sampling cluster."""
    if min_dist is None:
        min_dist = spacing * 0.9
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _molten_cluster(n, spacing, seed, temp_scale=1.0):
    """Generate a 'molten' cluster: random positions with soft repulsion."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    min_d = spacing * 0.5 * temp_scale
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_d for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _deformed_fcc_positions(n, a=1.0, seed=0):
    """Generate n atoms in FCC lattice with slight random deformation."""
    rng = np.random.default_rng(seed)
    pos = _fcc_positions(n, a)
    shear = rng.uniform(-0.05, 0.05, (3, 3))
    np.fill_diagonal(shear, 0)
    pos = pos @ (np.eye(3) + shear)
    return pos

def _gradient_guided_insertion(n, spacing, kind, seed=0):
    """
    Sequentially insert atoms by finding the position that minimizes
    the incremental energy contribution, using a coarse-to-fine grid search.
    """
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    
    # First few atoms: use grid search around existing atoms
    for i in range(1, min(n, 8)):
        if _remaining() < 30.0:
            break
        best_pos = None
        best_e = np.inf
        
        # Search in a shell around existing atoms
        search_radius = spacing * 1.5
        n_grid = 8  # grid resolution per axis
        
        # Candidate positions: grid points in a box around each existing atom
        candidates = []
        for existing in positions:
            for dx in np.linspace(-search_radius, search_radius, n_grid):
                for dy in np.linspace(-search_radius, search_radius, n_grid):
                    for dz in np.linspace(-search_radius, search_radius, n_grid):
                        p = existing + np.array([dx, dy, dz])
                        # Check minimum distance constraint
                        if all(np.linalg.norm(p - q) >= 0.4 for q in positions):
                            candidates.append(p)
        
        if not candidates:
            # Fallback: random position
            p = rng.uniform(-spacing, spacing, 3)
            positions.append(p)
            continue
        
        # Evaluate energy contribution for a subset of candidates
        # To save time, only evaluate top ~200 candidates by proximity
        # Sort by distance to nearest existing atom (prefer close but not too close)
        scored = []
        for p in candidates[:500]:
            min_d = min(np.linalg.norm(p - q) for q in positions)
            # Prefer distances near the equilibrium spacing
            score = abs(min_d - spacing) + 0.1 * min_d
            scored.append((score, p))
        
        scored.sort(key=lambda x: x[0])
        top_candidates = [p for _, p in scored[:50]]
        
        # Evaluate incremental energy for each candidate
        for p in top_candidates:
            test_pos = np.array(positions + [p])
            try:
                e, _ = sutton_chen_energy_gradient(test_pos, kind)
                if e < best_e and np.all(np.isfinite(p)):
                    best_e = e
                    best_pos = p
            except Exception:
                continue
        
        if best_pos is None:
            # Fallback: use the first candidate
            best_pos = top_candidates[0] if top_candidates else rng.uniform(-spacing, spacing, 3)
        
        positions.append(best_pos)
    
    # For remaining atoms, use a faster heuristic: place near the surface
    while len(positions) < n:
        if _remaining() < 30.0:
            break
        
        # Find the centroid and add an atom in a random direction from the surface
        centroid = np.mean(positions, axis=0)
        # Pick a random existing atom as anchor
        anchor_idx = rng.integers(0, len(positions))
        anchor = positions[anchor_idx]
        
        # Direction away from centroid
        direction = anchor - centroid
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-6:
            direction = rng.normal(0, 1, 3)
            dir_norm = np.linalg.norm(direction)
        direction /= dir_norm
        
        # Place at equilibrium distance from anchor, in the outward direction
        p = anchor + direction * spacing * 0.95
        p += rng.normal(0, spacing * 0.1, 3)
        
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.36 for q in positions):
            positions.append(p)
        else:
            # Try a few random alternatives
            for _ in range(5):
                p_alt = anchor + rng.normal(0, spacing * 0.5, 3)
                if all(np.linalg.norm(p_alt - q) >= 0.36 for q in positions):
                    positions.append(p_alt)
                    break
            else:
                positions.append(p)  # Add anyway to avoid infinite loop
    
    return np.array(positions[:n])

def _min_distance_check(pos):
    """Check minimum distance between any two atoms."""
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(energy, pos):
    """Check if a relaxation result is valid."""
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_task(args):
    """Task for parallel execution: relax a single candidate."""
    pos, potential, kind = args
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(energy, relaxed_pos):
            return energy, relaxed_pos
        else:
            return np.inf, relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    global _START_TIME, _DEADLINE
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    spacing = _get_spacing(kind)
    
    # Generate diverse starting points
    candidates = []
    
    # 1. Gradient-guided sequential insertion (primary strategy per directive)
    for seed in [0, 42, 123]:
        try:
            if _remaining() < 60.0:
                break
            gg = _gradient_guided_insertion(n, spacing, kind, seed)
            if len(gg) == n and _min_distance_check(gg) > 0.36:
                candidates.append(gg)
        except Exception:
            pass
    
    # 2. FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(fcc)
    except Exception:
        pass
    
    # 3. HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n:
            candidates.append(hcp)
    except Exception:
        pass
    
    # 4. BCC
    try:
        bcc = _bcc_positions(n, spacing)
        if len(bcc) == n:
            candidates.append(bcc)
    except Exception:
        pass
    
    # 5. Icosahedral
    try:
        ico = _icosahedral_positions(n, spacing)
        if len(ico) == n:
            candidates.append(ico)
    except Exception:
        pass
        
    # 6. Deformed FCC
    try:
        dfcc = _deformed_fcc_positions(n, spacing, seed=7)
        if len(dfcc) == n:
            candidates.append(dfcc)
    except Exception:
        pass
    
    # 7-8. Perturbed FCC/HCP (small)
    try:
        fcc = _fcc_positions(n, spacing)
        rng = np.random.default_rng(1)
        noise = rng.normal(0, spacing * 0.08, fcc.shape)
        candidates.append(fcc + noise)
    except Exception:
        pass
        
    try:
        hcp = _hcp_positions(n, spacing)
        rng = np.random.default_rng(2)
        noise = rng.normal(0, spacing * 0.08, hcp.shape)
        candidates.append(hcp + noise)
    except Exception:
        pass
        
    # 9-10. Perturbed FCC/HCP (medium)
    try:
        fcc = _fcc_positions(n, spacing)
        rng = np.random.default_rng(3)
        noise = rng.normal(0, spacing * 0.2, fcc.shape)
        candidates.append(fcc + noise)
    except Exception:
        pass
        
    try:
        hcp = _hcp_positions(n, spacing)
        rng = np.random.default_rng(4)
        noise = rng.normal(0, spacing * 0.2, hcp.shape)
        candidates.append(hcp + noise)
    except Exception:
        pass
        
    # 11-14. Molten clusters
    for seed, ts in [(555, 1.0), (666, 1.5), (777, 0.8), (888, 2.0)]:
        try:
            molten = _molten_cluster(n, spacing, seed, temp_scale=ts)
            candidates.append(molten)
        except Exception:
            pass
            
    # 15-18. Random clusters
    for seed, md in [(42, None), (123, None), (333, spacing * 1.2), (444, spacing * 1.2)]:
        try:
            rand = _random_cluster(n, spacing, seed, min_dist=md)
            candidates.append(rand)
        except Exception:
            pass
            
    # Filter out invalid candidates
    valid_candidates = []
    for c in candidates:
        if len(c) == n and np.all(np.isfinite(c)) and _min_distance_check(c) > 0.36:
            valid_candidates.append(c)
    
    if not valid_candidates:
        # Last resort fallback
        valid_candidates = [_random_cluster(n, spacing, 999, min_dist=0.4)]
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Parallel Relaxation of Initial Candidates
    if _remaining() > 20.0:
        tasks = [(pos, potential, kind) for pos in valid_candidates]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_task, task): task for task in tasks}
            for future in as_completed(futures):
                if _remaining() < 5.0:
                    break
                try:
                    energy, pos = future.result()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    pass

    # If no valid result yet, try sequential fallback
    if best_positions is None:
        for pos in valid_candidates:
            if _remaining() < 10.0:
                break
            try:
                energy, relaxed_pos = relax(pos, potential, kind)
                if _valid_result(energy, relaxed_pos) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass

    # Phase 2: Iterative Perturbation Refinement
    if best_positions is not None and _remaining() > 30.0:
        rng = np.random.default_rng(999)
        iteration = 0
        no_improve_count = 0
        while _remaining() > 40.0:
            iteration += 1
            # Multi-scale kicks: start larger, decay
            base_kick = spacing * 0.6 * (0.7 ** iteration)
            kicks = [base_kick * 0.3, base_kick * 0.5, base_kick * 0.8, base_kick * 1.0, base_kick * 1.4, base_kick * 1.8]
            
            tasks = []
            for kick in kicks:
                noise = rng.normal(0, kick, best_positions.shape)
                perturbed = best_positions + noise
                tasks.append((perturbed, potential, kind))
                
            improved_this_round = False
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_relax_task, task): task for task in tasks}
                for future in as_completed(futures):
                    if _remaining() < 10.0:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = pos
                            improved_this_round = True
                    except Exception:
                        pass
            
            if not improved_this_round:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
            else:
                no_improve_count = 0
    
    # Phase 3: Additional diversification
    if best_positions is not None and _remaining() > 50.0:
        rng2 = np.random.default_rng(2024)
        for kick_scale in [0.3, 0.5, 0.8]:
            if _remaining() < 20.0:
                break
            noise = rng2.normal(0, spacing * kick_scale, best_positions.shape)
            perturbed = best_positions + noise
            try:
                energy, pos = _relax_task((perturbed, potential, kind))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass

    # Phase 4: Final High-Precision Polish
    if best_positions is not None and _remaining() > 10.0:
        try:
            energy, relaxed_pos = relax(best_positions, potential, kind)
            if _valid_result(energy, relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass

    # Fallback if everything failed
    if best_positions is None:
        try:
            pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
            energy, relaxed_pos = relax(pos, potential, kind)
            if _valid_result(energy, relaxed_pos):
                best_positions = relaxed_pos
            else:
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
        except Exception:
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.8, 0, 0]
            
    return best_positions
# EVOLVE-BLOCK-END
