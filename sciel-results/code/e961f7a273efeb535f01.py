"""Dzugutov cluster global minima via multi-start basin hopping with polytetrahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


def _dzugutov_energy(positions):
    """Compute Dzugutov energy for given positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    V = np.zeros_like(r)
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    V[mask_a] = A * (r[mask_a]**(-m) - B) * np.exp(c / (r[mask_a] - a))
    V[mask_b] = B * np.exp(d / (r[mask_b] - b))
    return np.sum(V)


def _dzugutov_gradient(positions):
    """Compute Dzugutov energy gradient for given positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                dVdr = A * (-m * r**(-m-1) - c * (r**(-m) - B) * np.exp(c/(r-a)) / (r-a)) * np.exp(c/(r-a))
            elif r < b:
                dVdr = B * (-d * np.exp(d/(r-b)) / (r-b))
            else:
                continue
            force = dVdr * rij / r
            grad[i] += force
            grad[j] -= force
    return grad


def _generate_icosahedron(n_atoms, rng):
    """Generate an icosahedral cluster with n_atoms atoms."""
    if n_atoms == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    vertices /= np.linalg.norm(vertices[0])
    
    positions = [np.zeros(3)]  # center
    if n_atoms > 1:
        positions.append(vertices[0].copy())
    
    # Add second shell (octahedral positions)
    second_shell = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            v = vertices[i] + vertices[j]
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v /= norm
                # Check if this is a new direction
                is_new = True
                for existing in second_shell:
                    if np.abs(np.dot(v, existing)) > 0.99:
                        is_new = False
                        break
                if is_new:
                    second_shell.append(v)
    
    second_shell = np.array(second_shell)
    # Scale second shell to appropriate distance
    second_shell *= 1.28  # typical second shell distance
    
    idx = 2
    for v in second_shell:
        if idx >= n_atoms:
            break
        positions.append(v.copy())
        idx += 1
    
    # If we still need more atoms, add random positions in a shell
    while idx < n_atoms:
        # Random point on sphere
        theta = rng.uniform(0, 2*np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        v = np.array([r*np.cos(theta), r*np.sin(theta), z])
        v *= 1.5 + rng.uniform(0, 0.5)
        positions.append(v)
        idx += 1
    
    return np.array(positions[:n_atoms])


def _generate_fcc_cluster(n_atoms, rng):
    """Generate an FCC-like cluster with n_atoms atoms."""
    # FCC lattice points
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i, j, k]) + np.array(offset)
                    lattice_points.append(p)
    
    lattice_points = np.array(lattice_points)
    lattice_points *= 1.1  # scale
    
    # Sort by distance from origin
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    lattice_points = lattice_points[order]
    
    # Take first n_atoms
    positions = lattice_points[:n_atoms].copy()
    # Center
    positions -= np.mean(positions, axis=0)
    
    # Add small random perturbation
    positions += rng.normal(0, 0.01, positions.shape)
    
    return positions


def _generate_random_compact(n_atoms, rng, spacing=1.1):
    """Generate a random compact cluster."""
    radius = spacing * n_atoms ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n_atoms and attempts < n_atoms * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place enough, fill with random positions
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _relax_with_gradient(positions, max_iter=500, tol=1e-8):
    """Relax positions using gradient descent with adaptive step size."""
    n = len(positions)
    positions = positions.copy()
    step_size = 0.05
    energy = _dzugutov_energy(positions)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        grad = _dzugutov_gradient(positions)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        new_pos = positions - step_size * grad
        new_energy = _dzugutov_energy(new_pos)
        
        if new_energy < energy:
            positions = new_pos
            energy = new_energy
            step_size = min(step_size * 1.05, 0.5)
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return energy, positions


def _basin_hopping(positions, n_iterations=20, noise_scale=0.3):
    """Perform basin hopping optimization."""
    best_energy = _dzugutov_energy(positions)
    best_positions = positions.copy()
    current_energy = best_energy
    current_positions = positions.copy()
    rng = np.random.default_rng()
    
    for it in range(n_iterations):
        if time.time() > _DEADLINE:
            break
        
        # Perturb
        noise = rng.normal(0, noise_scale, current_positions.shape)
        perturbed = current_positions + noise
        
        # Relax
        energy, relaxed = _relax_with_gradient(perturbed, max_iter=300)
        
        # Metropolis acceptance
        delta_e = energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.5):
            current_energy = energy
            current_positions = relaxed
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
    
    return best_energy, best_positions


def construct(instance):
    """Find low-energy Dzugutov cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    
    if potential != 'dzugutov':
        # Fallback for other potentials
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0/3.0) * 0.75
        best = None
        for _ in range(4):
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, instance.get("rho", instance.get("kind")))
            if best is None or energy < best[0]:
                best = (energy, positions)
        return best[1]
    
    rng = np.random.default_rng(size)
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    candidates = []
    
    # 1. Icosahedral seeds
    for _ in range(3):
        ico = _generate_icosahedron(size, rng)
        ico += rng.normal(0, 0.05, ico.shape)
        candidates.append(ico)
    
    # 2. FCC seeds
    for _ in range(3):
        fcc = _generate_fcc_cluster(size, rng)
        candidates.append(fcc)
    
    # 3. Random compact seeds
    for _ in range(4):
        rand = _generate_random_compact(size, rng)
        candidates.append(rand)
    
    # 4. Mixed seeds
    for _ in range(2):
        ico = _generate_icosahedron(size, rng)
        fcc = _generate_fcc_cluster(size, rng)
        mixed = 0.5 * ico + 0.5 * fcc
        mixed += rng.normal(0, 0.1, mixed.shape)
        candidates.append(mixed)
    
    # Process each candidate
    for idx, candidate in enumerate(candidates):
        if time.time() > _DEADLINE - 30:
            break
        
        # First do a quick relaxation
        energy, relaxed = _relax_with_gradient(candidate, max_iter=200)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
        
        # If time permits, do basin hopping on the best so far
        if time.time() < _DEADLINE - 60:
            bh_energy, bh_positions = _basin_hopping(relaxed, n_iterations=10, noise_scale=0.2)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
    
    # Final relaxation on the best configuration
    if best_positions is not None and time.time() < _DEADLINE - 10:
        final_energy, final_positions = _relax_with_gradient(best_positions, max_iter=500)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        # Fallback: use the library's relax
        rand = _generate_random_compact(size, rng)
        _, best_positions = relax(rand, 'dzugutov')
    
    return best_positions
# EVOLVE-BLOCK-END
