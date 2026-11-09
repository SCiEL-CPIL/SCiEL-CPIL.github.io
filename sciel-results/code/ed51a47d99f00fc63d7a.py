"""Morse cluster global minima search using structured initializations and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_pair = r[iu]
    x = rho * (1.0 - r_pair)
    exp_x = np.exp(x)
    e_pair = exp_x * (exp_x - 2.0)
    return float(np.sum(e_pair))


def _generate_icosahedral(n):
    """Generate n points on concentric icosahedral shells."""
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    shell_radius = 1.0
    shell_count = 0
    while len(points) < n:
        # Number of points on this shell (approximate)
        if shell_count == 0:
            num_on_shell = min(n - len(points), 1)
        else:
            # Scale with shell radius squared
            num_on_shell = min(n - len(points), int(12 * (shell_radius ** 2 / 1.0)))
            num_on_shell = max(num_on_shell, 1)
        
        for _ in range(num_on_shell):
            # Pick a random vertex direction and add small perturbation
            idx = np.random.randint(0, 12)
            p = verts[idx] * shell_radius
            p += np.random.normal(0, 0.05, 3)
            points.append(p)
        
        shell_radius *= 1.4
        shell_count += 1
    
    return np.array(points[:n])


def _generate_fcc_like(n):
    """Generate n points in an FCC-like arrangement."""
    points = []
    # FCC lattice points
    a = 1.0  # lattice constant
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC basis
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) < 4.0:
                        lattice_points.append(p)
    
    # Shuffle and take n
    np.random.shuffle(lattice_points)
    points = lattice_points[:n]
    
    if len(points) < n:
        # Add random points if needed
        while len(points) < n:
            p = np.random.uniform(-2, 2, 3)
            points.append(p)
    
    return np.array(points[:n])


def _generate_random_compact(n, seed=42):
    """Generate n points in a compact random cluster."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.6
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.4:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _basin_hopping(positions, rho, n_iter=20, step_size=0.3, temperature=1.0, rng=None):
    """Simple basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = relax(positions.copy(), 'morse', rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for i in range(n_iter):
        # Generate perturbation
        noise = rng.normal(0, step_size, positions.shape)
        perturbed = current_pos + noise
        
        # Relax the perturbed structure
        new_energy, new_pos = relax(perturbed, 'morse', rho)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temperature):
            current_energy = new_energy
            current_pos = new_pos
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        # Decrease temperature
        temperature *= 0.95
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    best_energy = np.inf
    best_pos = None
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Icosahedral shells
    ico_pos = _generate_icosahedral(size)
    initializations.append(ico_pos)
    
    # 2. FCC-like
    fcc_pos = _generate_fcc_like(size)
    initializations.append(fcc_pos)
    
    # 3. Random compact (multiple seeds)
    for seed in [42, 123, 456, 789]:
        rand_pos = _generate_random_compact(size, seed=seed)
        initializations.append(rand_pos)
    
    # Process each initialization with basin hopping
    for idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # First, just relax to get a local minimum
        energy, relaxed_pos = relax(init_pos, 'morse', parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed_pos.copy()
        
        # Apply basin hopping if we have time
        if time.time() < deadline - 60:
            bh_energy, bh_pos = _basin_hopping(
                relaxed_pos, parameter, 
                n_iter=15, step_size=0.25, 
                temperature=0.5
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
    
    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 10:
        final_energy, final_pos = relax(best_pos, 'morse', parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos
    
    if best_pos is None:
        # Fallback: return the first relaxed initialization
        energy, pos = relax(initializations[0], 'morse', parameter)
        return pos
    
    return best_pos
# EVOLVE-BLOCK-END
