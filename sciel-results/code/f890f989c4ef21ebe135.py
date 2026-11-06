"""Dzugutov cluster global minima search using diverse icosahedral-seeded basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

WORKING_LIMIT = 305.0
START_TIME = time.time()

def _remaining_time():
    return WORKING_LIMIT - (time.time() - START_TIME)

def _icosahedron_vertices():
    """12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return v

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

def _build_icosahedral_seed(n, spacing=1.0, rng=None):
    """
    Build an n-atom seed based on icosahedral packing.
    Uses concentric icosahedral shells: 13 (1+12), 55 (13+42), etc.
    For arbitrary n, we build a core icosahedron and add atoms on an outer shell.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Core: 13-atom icosahedron
    verts = _icosahedron_vertices()
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * spacing
    
    positions = [np.zeros(3)]  # center
    positions.extend(verts)  # 12 vertices -> 13 total
    
    if n <= 13:
        # Just take first n atoms
        return np.array(positions[:n])
    
    # Add outer shell atoms
    n_outer = n - 13
    r_outer = spacing * 1.9
    
    # Fibonacci sphere distribution for outer atoms
    golden_angle = np.pi * (3 - np.sqrt(5))
    outer = np.zeros((n_outer, 3))
    for i in range(n_outer):
        z = 1 - 2 * (i + 0.5) / n_outer
        r = np.sqrt(max(0, 1 - z*z))
        theta = golden_angle * i
        outer[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r_outer
    
    # Randomly rotate the outer shell to avoid symmetry artifacts
    R = _random_rotation_matrix(rng)
    outer = outer @ R.T
    
    positions.extend(outer.tolist())
    return np.array(positions[:n])

def _build_random_compact_seed(n, rng=None):
    """Build a random compact seed."""
    if rng is None:
        rng = np.random.default_rng()
    # Random points in a small sphere
    positions = rng.normal(0, 0.7, (n, 3))
    return positions

def _relax(positions, potential, parameter):
    """Relax with error handling."""
    try:
        energy, pos = relax(positions, potential, parameter)
        return energy, pos
    except Exception:
        return np.inf, positions

def _ensure_min_distance(positions, min_dist=0.5, max_iter=50):
    """Ensure all pairs are at least min_dist apart."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break
        # Vectorized minimum distance
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        min_d = dist[min_idx]
        
        if min_d >= min_dist:
            break
        
        i, j = min_idx
        if i > j:
            i, j = j, i
        direction = (positions[i] - positions[j]) / (min_d + 1e-10)
        push = (min_dist - min_d) / 2 + 0.01
        positions[i] += direction * push
        positions[j] -= direction * push
    
    return positions

def _basin_hopping_kick(positions, potential, parameter, rng, max_iterations=40, kick_interval=15):
    """
    Basin hopping with periodic large kicks to escape local minima.
    Returns (best_energy, best_positions).
    """
    current_energy, current_pos = _relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    amplitude = 0.3
    beta = 1.0 / 0.3
    kick_amplitude = 1.2
    
    for it in range(max_iterations):
        if _remaining_time() < 15.0:
            break
        
        # Periodic kick
        if it % kick_interval == kick_interval - 1:
            perturbation = rng.normal(0, kick_amplitude, positions.shape)
        else:
            perturbation = rng.normal(0, amplitude, positions.shape)
        
        perturbed = current_pos + perturbation
        new_energy, new_pos = _relax(perturbed, potential, parameter)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-beta * delta_e):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        amplitude *= 1.03
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov clusters using diverse icosahedral-seeded basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(n * 7 + 13)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy: multiple diverse seeds, each followed by basin hopping with kicks
    # Alternate between icosahedral and random seeds for diversity
    
    n_starts = 6  # More starts, each with shorter BH
    
    for start_idx in range(n_starts):
        if _remaining_time() < 30.0:
            break
        
        # Choose seed type: alternate icosahedral and random
        if start_idx % 2 == 0:
            # Icosahedral seed with random rotation
            seed = _build_icosahedral_seed(n, spacing=1.0 + rng.uniform(-0.05, 0.05), rng=rng)
        else:
            # Random compact seed
            seed = _build_random_compact_seed(n, rng=rng)
        
        # Relax seed
        energy, relaxed_seed = _relax(seed, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_seed.copy()
        
        # Basin hopping with kicks
        if _remaining_time() > 25.0:
            bh_energy, bh_pos = _basin_hopping_kick(
                relaxed_seed, potential, parameter, rng, 
                max_iterations=30, kick_interval=12
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
    
    # Final relaxation of best
    if best_positions is not None:
        final_energy, final_pos = _relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _ensure_min_distance(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
