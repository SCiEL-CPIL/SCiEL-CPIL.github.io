"""
Rewritten program for Dzugutov cluster global minima search.
Uses a structural approach based on icosahedral growth patterns,
parallel basin hopping with diverse initial structures, and softness annealing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    dist = np.triu(dist, 1)
    dist = dist.flatten()
    
    # Mask for r < a and r < b
    mask_a = dist < a
    mask_b = dist < b
    
    energy = 0.0
    if np.any(mask_a):
        ra = dist[mask_a]
        energy += np.sum((ra**(-m) - B) * np.exp(c / (ra - a)))
    if np.any(mask_b):
        rb = dist[mask_b]
        energy += np.sum(B * np.exp(d / (rb - b)))
    return energy

def _generate_icosahedral_seed(n, rng):
    """Generate an icosahedral seed structure."""
    # 12 vertices of an icosahedron
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Scale to appropriate distance
    scale = 1.05
    positions = [vertices[0]]
    
    # Add remaining vertices
    for i in range(1, min(12, n)):
        positions.append(vertices[i])
    
    # If we need more atoms, add them in a shell
    if n > 12:
        # Generate points on a sphere for the outer shell
        shell_radius = 2.1
        while len(positions) < n:
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = np.array([
                shell_radius * sin_phi * np.cos(theta),
                shell_radius * sin_phi * np.sin(theta),
                shell_radius * cos_phi
            ])
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.8 for q in positions):
                positions.append(p)
    
    return np.array(positions[:n])

def _generate_dense_random(n, rng):
    """Generate a dense random cluster."""
    radius = 1.2 * n ** (1.0 / 3.0)
    positions = []
    attempts = 0
    while len(positions) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.75 for q in positions):
                positions.append(p)
        attempts += 1
    
    # Fill remaining if needed
    while len(positions) < n:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        r = rng.uniform(0, radius)
        p = np.array([
            r * sin_phi * np.cos(theta),
            r * sin_phi * np.sin(theta),
            r * cos_phi
        ])
        positions.append(p)
    
    return np.array(positions)

def _generate_fcc_seed(n, rng):
    """Generate an FCC-like seed."""
    positions = []
    a = 1.1  # lattice constant
    max_coord = int(np.ceil((n / 4) ** (1/3)))
    
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if len(positions) >= n:
                    break
                p = np.array([x * a, y * a, z * a])
                if np.linalg.norm(p) <= 2.5 * a:
                    if all(np.linalg.norm(p - q) >= 0.7 for q in positions):
                        positions.append(p)
        if len(positions) >= n:
            break
        if len(positions) >= n:
            break
    
    # Fill remaining
    while len(positions) < n:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        r = rng.uniform(0, 3 * a)
        p = np.array([
            r * sin_phi * np.cos(theta),
            r * sin_phi * np.sin(theta),
            r * cos_phi
        ])
        positions.append(p)
    
    return np.array(positions[:n])

def _relax_with_annealing(positions, potential, max_iter=200, temp_start=0.5, temp_end=0.05):
    """Relax with softness annealing."""
    pos = positions.copy()
    for i in range(max_iter):
        temp = temp_start * (temp_end / temp_start) ** (i / max_iter)
        # Get gradient
        try:
            grad = dzugutov_energy_gradient(pos)
            # Small random perturbation
            perturbation = temp * 0.05 * np.random.randn(*pos.shape)
            pos_new = pos - 0.01 * grad + perturbation
            # Check minimum distance
            if all(np.linalg.norm(pos_new[i] - pos_new[j]) >= 0.5 for i in range(len(pos_new)) for j in range(i+1, len(pos_new))):
                pos = pos_new
        except:
            break
    
    # Final relaxation
    try:
        energy, pos = relax(pos, potential)
    except:
        pass
    
    return pos

def _basin_hopping(positions, potential, n_iterations=15, step_size=0.1):
    """Perform basin hopping."""
    current_pos = positions.copy()
    try:
        current_energy = _dzugutov_energy(current_pos)
    except:
        current_energy = float('inf')
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for _ in range(n_iterations):
        # Random perturbation
        new_pos = current_pos + step_size * np.random.randn(*current_pos.shape)
        
        # Relax
        try:
            energy, new_pos = relax(new_pos, potential)
        except:
            continue
        
        # Check minimum distance
        valid = all(np.linalg.norm(new_pos[i] - new_pos[j]) >= 0.5 for i in range(len(new_pos)) for j in range(i+1, len(new_pos)))
        if not valid:
            continue
        
        # Metropolis acceptance
        if energy < current_energy:
            current_pos = new_pos
            current_energy = energy
        else:
            delta = energy - current_energy
            if np.random.rand() < np.exp(-delta / 0.1):
                current_pos = new_pos
                current_energy = energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy

def _worker(args):
    """Worker function for parallel basin hopping."""
    seed_idx, n, potential, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    
    # Generate diverse initial structures
    if seed_idx % 4 == 0:
        positions = _generate_icosahedral_seed(n, rng)
    elif seed_idx % 4 == 1:
        positions = _generate_dense_random(n, rng)
    elif seed_idx % 4 == 2:
        positions = _generate_fcc_seed(n, rng)
    else:
        positions = _generate_icosahedral_seed(n, rng)
    
    # Basin hopping
    best_pos, best_energy = _basin_hopping(positions, potential, n_iterations=12, step_size=0.12)
    
    return best_energy, best_pos

def construct(instance):
    """Main entry point: find low-energy Dzugutov cluster."""
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Number of parallel workers
    n_workers = min(4, n)
    
    # Create worker arguments with different seeds
    rng = np.random.default_rng(n)
    worker_args = [(i, n, potential, int(rng.integers(0, 1000000))) for i in range(n_workers)]
    
    best_energy = float('inf')
    best_positions = None
    
    # Run parallel basin hopping
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, args): args for args in worker_args}
        
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 30:
                break
            try:
                energy, positions = future.result(timeout=10)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                pass
    
    # If we have time, do additional refinement
    if best_positions is not None and time.time() < deadline - 60:
        # Multiple relaxation passes
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            try:
                energy, positions = relax(best_positions, potential)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                break
    
    # If no valid positions found, generate fallback
    if best_positions is None:
        rng = np.random.default_rng(42)
        best_positions = _generate_dense_random(n, rng)
        try:
            best_energy, best_positions = relax(best_positions, potential)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
