"""
Strategy: Basin Hopping with Crystalline Initialization.
Instead of pure gradient descent (L-BFGS-B), we use Basin Hopping: a global
optimization method that combines local minimization with random perturbations
to escape local minima. This is structurally different from the current
program's approach of just doing multiple L-BFGS relaxations from fixed
initializations. Basin hopping is known to be very effective for molecular
cluster problems like Morse clusters.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin


def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = -dV_dr * inv_r
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad


def _local_minimize(positions, rho, maxiter=200):
    """Perform local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        return e, g.flatten()
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return positions, _morse_energy_and_grad(positions, rho)[0]


def _basin_hopping(positions, rho, n_iterations=10, perturbation=0.5, rng=None):
    """
    Basin hopping: alternate between local minimization and random perturbation.
    Accept new states based on Metropolis criterion to escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pos, energy = _local_minimize(positions, rho)
    best_pos = pos.copy()
    best_energy = energy
    
    for it in range(n_iterations):
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        # Perturb
        new_pos = pos + rng.normal(0, perturbation, pos.shape)
        
        # Local minimize from perturbed position
        new_pos, new_energy = _local_minimize(new_pos, rho)
        
        # Metropolis acceptance
        if new_energy < energy:
            pos, energy = new_pos, new_energy
        else:
            # T = 0.1 * (energy - new_energy) scale
            delta = new_energy - energy
            T = max(0.1, 0.01 * abs(energy))
            if rng.random() < np.exp(-delta / T):
                pos, energy = new_pos, new_energy
        
        if energy < best_energy:
            best_pos = pos.copy()
            best_energy = energy
    
    return best_pos, best_energy


def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
    a = np.sqrt(2.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = []
    
    fcc = _generate_fcc(size, rng)
    configs.append(fcc)
    
    bcc = _generate_bcc(size, rng)
    configs.append(bcc)
    
    # Random compact
    radius = 1.0 * size ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < size and attempts < size * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
            points.append(p)
        attempts += 1
    if len(points) == size:
        configs.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
    # Allocate time budget: ~40s per instance, 6 instances = 240s
    # Use basin hopping from each initial config
    time_per_config = 30.0  # seconds per initial config
    
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        # Do basin hopping from this initial config
        bh_pos, bh_energy = _basin_hopping(pos, parameter, n_iterations=5, 
                                           perturbation=0.3, rng=rng)
        
        if _is_valid(bh_pos):
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
    
    # If we have time left, do more basin hopping iterations from the best
    if best_positions is not None and time.time() - _START_TIME < _TIME_LIMIT - 40:
        # Additional basin hopping from best position
        extra_pos, extra_energy = _basin_hopping(best_positions, parameter, 
                                                  n_iterations=3, perturbation=0.2, rng=rng)
        if _is_valid(extra_pos) and extra_energy < best_energy:
            best_energy = extra_energy
            best_positions = extra_pos
    
    # Fallback if no valid position found
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 200:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            attempts += 1
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
