# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START = None
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill


def _get_remaining_time():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return _DEADLINE_SECONDS - (time.time() - _GLOBAL_START)


def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Morse potential: exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0))
    return energy


def _morse_energy_and_grad(pos, rho):
    """Compute energy and gradient for L-BFGS-B."""
    energy, grad = morse_energy_gradient(pos, rho)
    return energy, grad


def _enforce_min_distance(pos, min_dist=0.5, max_iter=50):
    """Vectorized enforcement of minimum distance."""
    n = len(pos)
    if n < 2:
        return pos
    
    pos = pos.copy()
    for _ in range(max_iter):
        # Compute all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Find pairs closer than min_dist
        # We only care about i < j
        triu_indices = np.triu_indices(n, k=1)
        pair_dists = dist[triu_indices]
        
        violating_mask = pair_dists < min_dist
        if not np.any(violating_mask):
            break
            
        # Get the indices of violating pairs
        violating_pairs = triu_indices[violating_mask]
        
        # For each violating pair, push them apart
        # To avoid sequential dependency issues in vectorization, we can just 
        # apply a small kick to all atoms involved in violations.
        # A simpler robust approach: iterate over violations but vectorize the distance calc.
        # Since n is small (<80), iterating over violating pairs is fine, but let's try to be efficient.
        
        for i, j in violating_pairs.T:
            d = pair_dists[violating_mask][np.where(violating_mask)[0].tolist().index((i,j)) if (i,j) in violating_pairs.T else 0] # This is inefficient
            # Better: just re-calculate for the specific pair or use the stored value
            # Actually, let's just use a simple loop for the correction, it's O(k) where k is number of violations, usually small.
            pass
            
        # Let's do a simpler vectorized push:
        # Calculate displacement vectors for violating pairs
        # This is tricky to do purely vectorized with in-place updates without race conditions.
        # Given N <= 80, a simple loop over the violating pairs is fast enough.
        
        violating_pairs_list = list(zip(*violating_pairs))
        for i, j in violating_pairs_list:
            d = dist[i, j]
            if d < 1e-6:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = (pos[i] - pos[j]) / d
            
            # Push apart by half the deficit + small epsilon
            push = (min_dist - d + 0.01) * 0.5
            pos[i] += direction * push
            pos[j] -= direction * push
            
    return pos


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster with n atoms."""
    a = spacing
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    side = int(math.ceil(n ** (1.0 / 3.0))) + 2
    lattice_points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * a
                    lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    com = np.mean(lattice_points, axis=0)
    lattice_points -= com
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    lattice_points = lattice_points[order]
    if len(lattice_points) >= n:
        return lattice_points[:n].copy()
    else:
        remaining = n - len(lattice_points)
        extra = np.random.randn(remaining, 3) * spacing * 0.5
        return np.vstack([lattice_points, extra])


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster with n atoms."""
    a = spacing
    basis = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]
    side = int(math.ceil(n ** (1.0 / 3.0))) + 2
    lattice_points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * a
                    lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    com = np.mean(lattice_points, axis=0)
    lattice_points -= com
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    lattice_points = lattice_points[order]
    if len(lattice_points) >= n:
        return lattice_points[:n].copy()
    else:
        remaining = n - len(lattice_points)
        extra = np.random.randn(remaining, 3) * spacing * 0.5
        return np.vstack([lattice_points, extra])


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster with n atoms."""
    a = spacing
    c = math.sqrt(8.0 / 3.0) * a
    def hex_to_cart(hx, hy, hz):
        x = a * (hx + 0.5 * hy)
        y = a * (math.sqrt(3) / 2.0) * hy
        z = c * hz
        return np.array([x, y, z])
    
    points = []
    side = int(math.ceil(n ** (1.0 / 3.0))) + 2
    for layer in range(-side, side + 1):
        if layer % 2 == 0:
            for i in range(-side, side + 1):
                for j in range(-side, side + 1):
                    p = hex_to_cart(i, j, layer)
                    points.append(p)
        else:
            for i in range(-side, side + 1):
                for j in range(-side, side + 1):
                    p = hex_to_cart(i + 1.0 / 3.0, j + 2.0 / 3.0, layer)
                    points.append(p)
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    if len(points) >= n:
        return points[:n].copy()
    else:
        remaining = n - len(points)
        extra = np.random.randn(remaining, 3) * spacing * 0.5
        return np.vstack([points, extra])


def _generate_random(n, spacing=1.0, seed=None):
    """Generate a random cluster with n atoms."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) > 0:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5 * spacing:
                    points.append(p)
            else:
                points.append(p)
    return np.array(points)


def _basin_hopping(pos, rho, potential, n_steps=20, perturbation_scale=0.5, seed=42):
    """Perform basin hopping optimization."""
    rng = np.random.default_rng(seed)
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    
    # Initial relaxation
    try:
        energy, relaxed_pos = relax(pos, potential, rho)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed_pos.copy()
    except Exception:
        pass

    for i in range(n_steps):
        remaining = _get_remaining_time()
        if remaining < 2.0:
            break
        
        # Perturb
        perturbed = best_pos + rng.normal(0, perturbation_scale, best_pos.shape)
        
        # Enforce min distance
        perturbed = _enforce_min_distance(perturbed, 0.5)
        
        # Relax
        try:
            energy, relaxed_pos = relax(perturbed, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            # Fallback to L-BFGS-B if relax fails
            try:
                result = minimize(
                    _morse_energy_and_grad,
                    perturbed,
                    args=(rho,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-6},
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.copy()
            except Exception:
                pass
                
    return best_pos, best_energy


def construct(instance):
    """Find low-energy arrangements of atoms bound by Morse potentials."""
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Generate diverse initializations
    inits = []
    
    # Crystalline templates
    inits.append(_generate_fcc(size, spacing=1.0))
    inits.append(_generate_bcc(size, spacing=1.0))
    inits.append(_generate_hcp(size, spacing=1.0))
    
    # Random starts
    inits.append(_generate_random(size, spacing=1.0, seed=100))
    inits.append(_generate_random(size, spacing=1.0, seed=200))
    
    # Optimize each initialization using Basin Hopping
    best_energy = np.inf
    best_positions = None
    
    for i, init in enumerate(inits):
        remaining = _get_remaining_time()
        if remaining < 10.0:
            break
        
        # Enforce min distance on init
        init = _enforce_min_distance(init, 0.5)
        
        # Run Basin Hopping
        # Number of steps depends on time available
        # Estimate time per step ~ 0.5s for N=80? 
        # Let's allocate 10-20s per init if time permits.
        steps = 10
        if remaining > 50:
            steps = 20
            
        pos, energy = _basin_hopping(init, rho, potential, n_steps=steps, seed=42+i)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
            
    # If no optimization succeeded, use the provided relax on the best init
    if best_positions is None:
        init = inits[0]
        init = _enforce_min_distance(init, 0.5)
        try:
            energy, best_positions = relax(init, potential, rho)
            best_energy = energy
        except Exception:
            best_positions = init
            best_energy = _morse_energy(init, rho)
            
    return best_positions
# EVOLVE-BLOCK-END
