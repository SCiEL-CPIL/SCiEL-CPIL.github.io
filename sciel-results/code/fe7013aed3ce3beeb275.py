# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 295.0  # 300s total budget, return before kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_icosahedral_shell(n_shell, rng, radius=1.9):
    """Generate atoms on an icosahedral shell of given radius."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    # Generate directions: vertices, edge midpoints, face centers
    directions = list(ico_verts)
    
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] + ico_verts[j]
            norm = np.linalg.norm(d)
            if norm > 0.01:
                directions.append(d / norm)
    
    ico_faces = [
        [0,1,8],[0,8,4],[0,4,2],[0,2,6],[0,6,1],
        [1,6,9],[1,9,3],[1,3,5],[1,5,0],[2,4,10],[2,10,7],[2,7,6],
        [3,9,11],[3,11,5],[3,5,0],[4,8,10],[4,10,7],[4,7,2],
        [5,11,9],[5,9,1],[5,0,3],[6,7,10]
    ]
    for face in ico_faces:
        d = ico_verts[face[0]] + ico_verts[face[1]] + ico_verts[face[2]]
        norm = np.linalg.norm(d)
        if norm > 0.01:
            directions.append(d / norm)
    
    # Deduplicate
    unique_dirs = [directions[0]]
    for d in directions[1:]:
        if all(np.linalg.norm(d - ud) > 0.05 for ud in unique_dirs):
            unique_dirs.append(d)
    directions = unique_dirs
    
    # Select n_shell directions
    if n_shell <= len(directions):
        selected = directions[:n_shell]
    else:
        selected = directions + [rng.normal(0, 1, 3) for _ in range(n_shell - len(directions))]
        for i in range(len(selected)):
            selected[i] = selected[i] / np.linalg.norm(selected[i])
    
    points = []
    for d in selected:
        noise = rng.normal(0, 0.02, 3)
        p = radius * d + noise
        points.append(p)
    
    return np.array(points)


def _generate_icosahedron(n, rng):
    """Generate an icosahedral cluster with center + shell."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # For small clusters, just use shell atoms
    if n <= 12:
        return _generate_icosahedral_shell(n, rng, radius=1.9)
    
    center = np.array([0.0, 0.0, 0.0])
    shell_n = n - 1
    shell = _generate_icosahedral_shell(shell_n, rng, radius=1.9)
    
    points = np.vstack([center, shell])
    return points[:n]


def _generate_bimodal(n, rng):
    """Generate a bimodal two-core icosahedral structure."""
    # Split into two cores
    n1 = n // 2
    n2 = n - n1
    
    core1 = _generate_icosahedron(n1, rng)
    core2 = _generate_icosahedron(n2, rng)
    
    # Place cores at some distance
    separation = 4.0 + rng.uniform(0, 1)
    angle = rng.uniform(0, 2 * np.pi)
    offset = np.array([separation * np.cos(angle), separation * np.sin(angle), rng.uniform(-1, 1)])
    
    points = np.vstack([core1, core2 + offset])
    return points[:n]


def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    a = 2.0
    points = []
    max_coord = int((n / 4.0) ** (1.0/3.0) * a) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for basis in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a, j*a, k*a]) + np.array(basis)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    while len(points) < n:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.2 * n ** (1.0/3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            for _ in range(100):
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
                    break
                p = p + rng.normal(0, 0.1, 3)
            else:
                p = np.array([len(points)*0.5, 0, 0])
                points.append(p)
    
    return np.array(points[:n])


def _fix_min_distance(positions, min_dist=0.5, max_iters=50):
    """Fix minimum distance violations by pushing atoms apart."""
    n = len(positions)
    for _ in range(max_iters):
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist_val = np.min(dists)
        if min_dist_val >= min_dist:
            break
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i_idx, j_idx = min_idx
        if i_idx == j_idx:
            continue
        
        diff = positions[i_idx] - positions[j_idx]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            diff = np.array([0.1, 0.1, 0.1])
            dist = np.linalg.norm(diff)
        
        push = (min_dist - dist + 0.01) * 0.5 * (diff / dist)
        positions[i_idx] += push
        positions[j_idx] -= push
    
    return positions


def _basin_hopping(positions, potential, rng, time_limit=40.0):
    """Basin Hopping algorithm with adaptive step size and subset moves."""
    current_energy, current_pos = relax(positions, potential)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    start_time = time.time()
    n_atoms = len(current_pos)
    
    # Adaptive step size
    base_step = 0.5 + 0.3 * (38.0 / n_atoms)
    min_step = 0.2
    max_step = 1.5
    step_size = base_step
    consecutive_rejects = 0
    
    while (time.time() - start_time) < time_limit:
        if time.time() - _START_TIME > _DEADLINE - 3:
            break
        
        # Perturbation: move a random subset of atoms
        # Use smaller subsets for larger clusters
        n_move = max(3, n_atoms // 5)
        move_idx = rng.choice(n_atoms, n_move, replace=False)
        
        delta = rng.normal(0, step_size, (n_move, 3))
        perturbed = current_pos.copy()
        perturbed[move_idx] += delta
        
        # Fix minimum distance
        perturbed = _fix_min_distance(perturbed, min_dist=0.5)
        
        new_energy, new_pos = relax(perturbed, potential)
        
        # Acceptance criterion: Metropolis with adaptive temperature
        elapsed = time.time() - start_time
        T = 0.3 + 0.7 * max(0, (time_limit - elapsed) / time_limit)
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            consecutive_rejects = 0
            step_size = min(max_step, step_size * 1.03)
        else:
            delta_e = new_energy - current_energy
            if rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                if consecutive_rejects > 8:
                    step_size = max(min_step, step_size * 0.93)
                    consecutive_rejects = 0
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
    return best_energy, best_pos


def _assemble_and_relax(n, rng, potential):
    """
    Divide-and-conquer assembly: build cluster from icosahedral sub-clusters,
    optimize their relative positions, then refine with basin hopping.
    """
    # Choose a decomposition strategy
    strategy = rng.choice(['single', 'bimodal', 'trimodal'], p=[0.4, 0.35, 0.25])
    
    if strategy == 'single':
        # Single icosahedral core
        pos = _generate_icosahedron(n, rng)
    elif strategy == 'bimodal':
        # Two icosahedral cores
        n1 = max(13, n // 2)
        n2 = n - n1
        core1 = _generate_icosahedron(n1, rng)
        core2 = _generate_icosahedron(n2, rng)
        
        # Optimize relative position by trying several separations and angles
        best_combo_energy = np.inf
        best_combo_pos = None
        
        for sep in [3.5, 4.0, 4.5, 5.0]:
            for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
                offset = np.array([sep * np.cos(angle), sep * np.sin(angle), rng.uniform(-0.5, 0.5)])
                combo = np.vstack([core1, core2 + offset])
                combo = _fix_min_distance(combo, min_dist=0.5)
                try:
                    e, pos_relaxed = relax(combo, potential)
                    if e < best_combo_energy:
                        best_combo_energy = e
                        best_combo_pos = pos_relaxed
                except:
                    pass
        
        if best_combo_pos is not None:
            pos = best_combo_pos
        else:
            pos = _generate_bimodal(n, rng)
    else:
        # Three icosahedral cores
        n1 = max(13, n // 3)
        n2 = max(13, (n - n1) // 2)
        n3 = n - n1 - n2
        
        core1 = _generate_icosahedron(n1, rng)
        core2 = _generate_icosahedron(n2, rng)
        core3 = _generate_icosahedron(n3, rng)
        
        # Place in triangular arrangement
        sep = 4.0
        offsets = [
            np.array([sep, 0, 0]),
            np.array([-sep/2, sep*np.sqrt(3)/2, 0]),
            np.array([-sep/2, -sep*np.sqrt(3)/2, 0])
        ]
        
        combo = np.vstack([core1, core2 + offsets[0], core3 + offsets[1]])
        combo = _fix_min_distance(combo, min_dist=0.5)
        try:
            _, pos = relax(combo, potential)
        except:
            pos = _generate_random(n, rng)
    
    return pos


def construct(instance):
    """Find low-energy structures using assembly + Basin Hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size)
    
    # Time budget: ~45s per instance (300s / 6 instances with safety margin)
    time_per_instance = 44.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures using assembly strategy
    # Run multiple assembly attempts with different random seeds
    n_attempts = 4
    time_per_attempt = time_per_instance / n_attempts
    
    for attempt in range(n_attempts):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Each attempt uses a slightly different random seed
        attempt_rng = np.random.default_rng(size * 1000 + attempt)
        
        try:
            # Assembly phase: build initial structure
            pos = _assemble_and_relax(size, attempt_rng, potential)
            
            if pos.shape[0] != size:
                pos = _generate_random(size, attempt_rng)
            
            # Basin hopping refinement
            energy, pos_result = _basin_hopping(pos, potential, attempt_rng, time_limit=time_per_attempt)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = pos_result.copy()
        except Exception:
            pass
    
    # If no valid result, use fallback
    if best_positions is None:
        best_positions = _generate_random(size, rng)
        try:
            _, best_positions = relax(best_positions, potential)
        except:
            pass
    
    # Final relaxation to ensure minimum distance
    if best_positions is not None:
        best_positions = _fix_min_distance(best_positions, min_dist=0.5)
        try:
            _, best_positions = relax(best_positions, potential)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
