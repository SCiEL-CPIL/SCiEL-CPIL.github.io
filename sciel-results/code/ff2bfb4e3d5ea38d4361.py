"""Morse cluster global minima search: multi-start basin hopping with L-BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# Aim to finish by 280s to be safe, leaving 20s buffer
_DEADLINE = _START_TIME + 280.0


def _time_remaining():
    return _DEADLINE - time.time()


def _check_validity(positions, min_dist=0.5):
    """Check if positions are valid (no overlaps)."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist


def _fibonacci_sphere(n, radius, rng):
    """
    Generate n points on a sphere of given radius using Fibonacci lattice.
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n) + 0.5
    cos_theta = 1.0 - (i / n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    
    # Random rotation to avoid alignment with axes
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    theta = i * phi
    x_coords = sin_theta * np.cos(theta)
    y_coords = sin_theta * np.sin(theta)
    z_coords = cos_theta
    
    points = np.column_stack((x_coords, y_coords, z_coords))
    points = points @ R.T
    points *= radius
    return points


def _random_sphere(n, radius, rng):
    """Generate n random points inside a sphere of given radius."""
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    return np.array(points)


def _lbfgs_refine(positions, rho, max_iter=200):
    """
    Use L-BFGS-B with the provided gradient to refine positions.
    Returns (energy, refined_positions).
    """
    def objective(x):
        pos = x.reshape(-1, 3)
        try:
            e, g = morse_energy_gradient(pos, rho)
            return e, g.flatten()
        except:
            return 1e6, np.zeros_like(x)
    
    x0 = positions.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        refined = res.x.reshape(-1, 3)
        e, _ = morse_energy_gradient(refined, rho)
        return e, refined
    except:
        return np.inf, positions


def _basin_hopping(initial_pos, potential, parameter, size, rng, max_iter=30, step_size=0.3):
    """
    Perform basin hopping starting from initial_pos.
    Uses L-BFGS for local refinement after each kick.
    Returns (best_energy, best_positions).
    """
    best_energy = np.inf
    best_positions = initial_pos.copy()
    
    # First, refine the initial position using L-BFGS
    try:
        energy, refined = _lbfgs_refine(initial_pos, parameter, max_iter=100)
        if _check_validity(refined):
            best_energy = energy
            best_positions = refined
    except:
        pass
    
    # Adaptive step size: start larger, shrink if stuck
    current_step = step_size
    no_improve_count = 0
    max_no_improve = 6
    
    for i in range(max_iter):
        if _time_remaining() < 5.0:
            break
            
        # Adaptive step size logic
        if no_improve_count >= max_no_improve:
            # If stuck, try a larger kick to escape basin
            current_step = step_size * 1.8
            no_improve_count = 0
        else:
            # Gradually reduce step size for fine-tuning
            current_step = step_size * (1.0 - 0.4 * i / max_iter)
        
        kick = rng.normal(0, current_step, size=(size, 3))
        new_pos = best_positions + kick
        
        try:
            # Use L-BFGS for refinement after kick
            energy, refined = _lbfgs_refine(new_pos, parameter, max_iter=80)
            
            if not _check_validity(refined):
                no_improve_count += 1
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = refined
                no_improve_count = 0
            else:
                no_improve_count += 1
        except:
            no_improve_count += 1
            continue
            
    return best_energy, best_positions


def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses multi-start basin hopping with L-BFGS refinement and varied initializations.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate radius for initial configurations
    # For Morse with equilibrium distance 1, clusters are roughly cubic
    radius = 1.0 * size ** (1.0 / 3.0) * 0.95
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Determine number of restarts based on time budget
    # L-BFGS is faster than full relax, so we can do more restarts
    # With 280s budget and 6 instances, ~46s per instance
    # Each L-BFGS refinement ~0.1-0.5s, each BH iter ~0.2-0.8s
    # Let's try 8 restarts with 25 BH iterations each
    
    n_restarts = 8
    max_bh_iter = 25
    
    for i in range(n_restarts):
        if _time_remaining() < 8.0:
            break
            
        # Alternate between structured and random initializations
        if i % 4 == 0:
            # Fibonacci sphere - good for compact clusters
            points = _fibonacci_sphere(size, radius, rng)
        elif i % 4 == 1:
            # Random points in sphere
            points = _random_sphere(size, radius, rng)
        elif i % 4 == 2:
            # Slightly perturbed Fibonacci sphere
            points = _fibonacci_sphere(size, radius * 1.1, rng)
            points += rng.normal(0, 0.15, size=(size, 3))
        else:
            # Random points in a slightly larger sphere
            points = _random_sphere(size, radius * 1.15, rng)
        
        # Check initial validity
        if not _check_validity(points):
            # Try to fix by scaling
            points *= 0.85
            if not _check_validity(points):
                continue

        # Run basin hopping
        try:
            e_bh, pos_bh = _basin_hopping(points, potential, parameter, size, rng, 
                                          max_iter=max_bh_iter)
            
            if e_bh < best_energy:
                best_energy = e_bh
                best_positions = pos_bh
        except:
            continue

    # Fallback if no valid configuration found
    if best_positions is None:
        # Create a simple linear chain as fallback
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.6
        if _time_remaining() > 2.0:
            try:
                energy, relaxed = relax(best_positions, potential, parameter)
                if _check_validity(relaxed):
                    best_positions = relaxed
            except:
                pass

    return best_positions
# EVOLVE-BLOCK-END
