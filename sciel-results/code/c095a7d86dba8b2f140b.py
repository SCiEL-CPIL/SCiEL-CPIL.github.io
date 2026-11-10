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
    
    # 1. FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(fcc)
    except Exception:
        pass
    
    # 2. HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n:
            candidates.append(hcp)
    except Exception:
        pass
    
    # 3. BCC
    try:
        bcc = _bcc_positions(n, spacing)
        if len(bcc) == n:
            candidates.append(bcc)
    except Exception:
        pass
    
    # 4. Icosahedral
    try:
        ico = _icosahedral_positions(n, spacing)
        if len(ico) == n:
            candidates.append(ico)
    except Exception:
        pass
    
    # 5-8. Random clusters with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            rand = _random_cluster(n, spacing, seed)
            candidates.append(rand)
        except Exception:
            pass
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    # Use thread pool for parallel relaxation (max 4 threads)
    tasks = [(pos, potential, kind) for pos in candidates]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_task, task): task for task in tasks}
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                energy, relaxed_pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass
    
    # If no valid result, try one more random attempt
    if best_positions is None:
        if _remaining() > 5.0:
            try:
                pos = _random_cluster(n, spacing, 999)
                energy, relaxed_pos = relax(pos, potential, kind)
                if _valid_result(energy, relaxed_pos):
                    best_positions = relaxed_pos
            except Exception:
                # Last resort: simple grid
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
    
    # Multi-scale perturbation refinement around the best structure
    if best_positions is not None and _remaining() > 20.0:
        # Try perturbation refinement with decreasing scales
        scales = [0.4, 0.2, 0.1, 0.05, 0.03]
        current_best = best_positions.copy()
        current_energy = best_energy
        
        for scale in scales:
            if _remaining() < 10.0:
                break
            
            # Generate perturbed versions
            rng = np.random.default_rng(42)
            improved = False
            
            for attempt in range(5):
                if _remaining() < 5.0:
                    break
                
                # Perturb positions
                perturbed = current_best + rng.normal(0, scale, current_best.shape)
                
                # Relax the perturbed structure
                try:
                    energy, relaxed_pos = relax(perturbed, potential, kind)
                    if _valid_result(energy, relaxed_pos):
                        if energy < current_energy:
                            current_energy = energy
                            current_best = relaxed_pos
                            improved = True
                except Exception:
                    pass
            
            if not improved:
                # If no improvement at this scale, try a slightly larger perturbation
                # or break if we've converged
                if scale < 0.1:
                    break
    
    # Final polish: high-precision relaxation of the best structure
    if best_positions is not None and _remaining() > 10.0:
        try:
            energy, polished_pos = relax(current_best, potential, kind)
            if _valid_result(energy, polished_pos):
                best_positions = polished_pos
                best_energy = energy
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
