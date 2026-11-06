"""Morse cluster global minima search with structured initialization and multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return float(energy)


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    # Gradient: dE/dr_ij = -rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    # dE/dpos_i = sum_{j!=i} dE/dr_ij * (pos_i - pos_j) / r_ij
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = dists[i, j]
            if rij < 1e-10:
                continue
            xij = rho * (1.0 - rij)
            exij = np.exp(xij)
            dE_dr = -rho * exij * (exij - 1.0)
            direction = (pos[i] - pos[j]) / rij
            grad[i] += dE_dr * direction
            grad[j] -= dE_dr * direction
    return float(energy), grad


def _lattice_init(n, rng):
    """Generate a compact lattice-like initial configuration."""
    # Try different lattice structures
    structures = []
    
    # FCC-like packing
    a = 1.0
    fcc_basis = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ]
    count = 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in fcc_basis:
                    if count >= n:
                        break
                    structures.append(np.array([i + b[0], j + b[1], k + b[2]]) * a)
                    count += 1
                if count >= n:
                    break
            if count >= n:
                break
        if count >= n:
            break
    if len(structures) >= n:
        structures = structures[:n]
        # Center
        centroid = np.mean(structures, axis=0)
        structures = np.array(structures) - centroid
        # Add small random perturbation
        structures += rng.normal(0, 0.05, structures.shape)
        return structures
    
    # BCC-like
    bcc_basis = [[0, 0, 0], [0.5, 0.5, 0.5]]
    count = 0
    structures = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in bcc_basis:
                    if count >= n:
                        break
                    structures.append(np.array([i + b[0], j + b[1], k + b[2]]))
                    count += 1
                if count >= n:
                    break
            if count >= n:
                break
        if count >= n:
            break
    if len(structures) >= n:
        structures = structures[:n]
        centroid = np.mean(structures, axis=0)
        structures = np.array(structures) - centroid
        structures += rng.normal(0, 0.05, structures.shape)
        return structures
    
    # Fallback: random compact
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
            points.append(p)
    return np.array(points)


def _random_packed_init(n, rng, min_dist=0.8):
    """Generate a random packed initial configuration."""
    radius = n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
    # If we couldn't place enough, fill with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _fix_min_distance(pos, min_dist=0.5):
    """Ensure all atoms are at least min_dist apart."""
    n = len(pos)
    for _ in range(100):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    violated = True
                    if d < 1e-10:
                        pos[j] += np.array([min_dist, 0, 0])
                    else:
                        direction = (pos[i] - pos[j]) / d
                        push = (min_dist - d) / 2.0 + 1e-6
                        pos[i] += direction * push
                        pos[j] -= direction * push
        if not violated:
            break
    return pos


def construct(instance):
    """Multi-start search with lattice and random initializations, optimized with L-BFGS-B."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    best_energy = np.inf
    best_positions = None
    
    def fun_and_grad(x):
        """Objective function for scipy minimize."""
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_grad(pos, rho)
        return e, g.flatten()
    
    # Generate diverse initial structures
    initial_structures = []
    
    # Lattice-based
    for _ in range(3):
        initial_structures.append(_lattice_init(size, rng))
    
    # Random packed
    for _ in range(3):
        initial_structures.append(_random_packed_init(size, rng))
    
    # Shuffled lattice
    lat = _lattice_init(size, rng)
    idx = rng.permutation(size)
    initial_structures.append(lat[idx])
    
    # Optimize each initial structure
    for init_pos in initial_structures:
        if time.time() >= deadline:
            break
        
        init_pos = _fix_min_distance(init_pos, 0.5)
        x0 = init_pos.flatten()
        
        try:
            result = minimize(
                fun_and_grad, x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < best_energy:
                pos = result.x.reshape(-1, 3)
                pos = _fix_min_distance(pos, 0.5)
                e = _morse_energy(pos, rho)
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
        except Exception:
            continue
    
    # If we still have time, do additional random restarts
    if best_positions is None:
        # Fallback: use the first initialization
        best_positions = _fix_min_distance(initial_structures[0], 0.5)
        best_energy = _morse_energy(best_positions, rho)
    
    # Additional refinement if time permits
    if time.time() < deadline - 30:
        for _ in range(5):
            if time.time() >= deadline:
                break
            # Perturb best solution
            perturbed = best_positions + rng.normal(0, 0.3, best_positions.shape)
            perturbed = _fix_min_distance(perturbed, 0.5)
            x0 = perturbed.flatten()
            try:
                result = minimize(
                    fun_and_grad, x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if result.fun < best_energy:
                    pos = result.x.reshape(-1, 3)
                    pos = _fix_min_distance(pos, 0.5)
                    e = _morse_energy(pos, rho)
                    if e < best_energy:
                        best_energy = e
                        best_positions = pos
            except Exception:
                continue
    
    # Final check
    best_positions = _fix_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
