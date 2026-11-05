# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit to be safe (330s hard limit)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_flat(x, rho):
    """
    Compute Morse energy for a flat array of coordinates.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, _ = morse_energy_gradient(positions, rho)
        return energy
    except Exception:
        return 1e10

def _morse_energy_grad_flat(x, rho):
    """
    Compute Morse energy and gradient for a flat array.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _min_distance(positions):
    """
    Compute the minimum pairwise distance in a cluster.
    """
    n = len(positions)
    if n < 2:
        return np.inf
    # Use cdist for efficiency
    dists = cdist(positions, positions)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _generate_fcc_cluster(n, seed=42):
    """
    Generate an FCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0)) + 1.0
    lattice_points = []
    
    # Generate FCC points in a box
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        # Add small jitter to break symmetry
        jitter = rng.uniform(-0.01, 0.01, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        # Fill with random points if not enough lattice points
        points = np.array(lattice_points)
        remaining = n - len(points)
        random_pts = rng.uniform(-radius, radius, (remaining, 3))
        return np.vstack([points, random_pts])

def _generate_bcc_cluster(n, seed=42):
    """
    Generate a BCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0)) + 1.0
    lattice_points = []
    
    # BCC basis: (0,0,0), (0.5,0.5,0.5)
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        jitter = rng.uniform(-0.01, 0.01, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        points = np.array(lattice_points)
        remaining = n - len(points)
        random_pts = rng.uniform(-radius, radius, (remaining, 3))
        return np.vstack([points, random_pts])

def _generate_icosahedral_cluster(n, seed=42):
    """
    Generate an icosahedral shell-like cluster.
    """
    rng = np.random.default_rng(seed)
    points = []
    
    # Icosahedron vertices (normalized to radius 1)
    phi = (1 + math.sqrt(5)) / 2
    ico_verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
        (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (-1, phi, 0) # Duplicates removed in set
    ]
    # Unique icosahedron vertices
    unique_verts = set()
    for v in ico_verts:
        unique_verts.add(tuple(v))
    ico_verts = [np.array(v) for v in unique_verts]
    # Normalize to unit radius
    ico_verts = [v / np.linalg.norm(v) for v in ico_verts]
    
    # Add center
    points.append(np.zeros(3))
    
    # Add shells
    radius = 1.0
    shell_count = 0
    while len(points) < n:
        # Add points on sphere of radius 'radius'
        # Use Fibonacci sphere for even distribution
        num_points = min(12 * (shell_count + 1) ** 2, n - len(points))
        for i in range(num_points):
            z = 1 - (i + 0.5) * 2 / num_points
            r = np.sqrt(1 - z*z)
            theta = np.pi * (1 + 5**0.5) * i # Golden angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            p = np.array([x, y, z]) * radius
            points.append(p)
        
        radius += 0.8 # Approximate next shell spacing
        shell_count += 1
    
    points = np.array(points[:n])
    # Jitter
    jitter = rng.uniform(-0.05, 0.05, (n, 3))
    return points + jitter

def _generate_random_cluster(n, seed=42):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0)) + 1.0
    points = []
    attempts = 0
    max_attempts = 5000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _local_minimize(positions, rho, potential, time_budget=5.0):
    """
    Perform local minimization using L-BFGS-B.
    Returns (energy, positions).
    """
    n = len(positions)
    x0 = positions.flatten()
    
    def func(x):
        return _morse_energy_flat(x, rho)
    
    def jac(x):
        e, g = _morse_energy_grad_flat(x, rho)
        return g
    
    # We don't use bounds directly for distance, but we can use a penalty if needed.
    # However, L-BFGS-B is fast. We rely on the initial structure being valid and
    # the potential being repulsive at short range to maintain distance.
    # If distance < 0.5, energy is high, so gradient will push apart.
    
    start_time = time.time()
    try:
        res = minimize(
            func, x0, jac=jac, method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success:
            energy = res.fun
            pos = res.x.reshape(n, 3)
            return energy, pos
        else:
            # Try a few more steps if it didn't fully converge but improved
            energy = res.fun
            pos = res.x.reshape(n, 3)
            return energy, pos
    except Exception:
        return _morse_energy_flat(x0, rho), positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate diverse initial structures
    starts = []
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # FCC
    for s in seeds[:2]:
        try:
            starts.append(_generate_fcc_cluster(n, seed=s))
        except Exception:
            pass
            
    # BCC
    for s in seeds[2:4]:
        try:
            starts.append(_generate_bcc_cluster(n, seed=s))
        except Exception:
            pass
            
    # Icosahedral
    for s in seeds[4:6]:
        try:
            starts.append(_generate_icosahedral_cluster(n, seed=s))
        except Exception:
            pass
            
    # Random
    for s in seeds[6:8]:
        try:
            starts.append(_generate_random_cluster(n, seed=s))
        except Exception:
            pass
    
    if not starts:
        starts.append(_generate_random_cluster(n, seed=0))
        
    # 2. Local Minimization of all starts
    relaxed_starts = []
    for start in starts:
        if _get_remaining_time() < 10.0:
            break
        try:
            energy, positions = _local_minimize(start, rho, potential)
            # Check validity
            if _min_distance(positions) >= 0.5:
                relaxed_starts.append((energy, positions))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    if not relaxed_starts:
        # Fallback
        best_positions = starts[0]
        best_energy = _morse_energy_flat(best_positions.flatten(), rho)
        return best_positions

    # Sort by energy
    relaxed_starts.sort(key=lambda x: x[0])
    
    # 3. Firefly-inspired Global Search
    # We maintain a population of the best K structures.
    # Each "firefly" (structure) moves towards the best structure with a probability
    # proportional to the energy difference, but the move is a local minimization step.
    # This is essentially a deterministic basin hopping but guided by the global best.
    
    K = 5 # Keep top 5
    population = relaxed_starts[:K]
    
    # If we have fewer than K, pad with random perturbations of the best
    while len(population) < K:
        perturbed = best_positions + np.random.normal(0, 0.5, best_positions.shape)
        try:
            e, p = _local_minimize(perturbed, rho, potential)
            if _min_distance(p) >= 0.5:
                population.append((e, p))
        except Exception:
            pass
    
    # Iterative improvement
    max_iterations = 50
    for i in range(max_iterations):
        if _get_remaining_time() < 10.0:
            break
            
        # Get the best structure
        best_e, best_p = min(population, key=lambda x: x[0])
        
        new_population = [(best_e, best_p)]
        
        for e, p in population:
            if e == best_e and np.allclose(p, best_p):
                continue
                
            # Firefly rule: move towards best if best is better
            if best_e < e:
                # Probability of moving towards best
                # Simple heuristic: always move if best is significantly better,
                # otherwise random walk
                prob = 0.8 # High probability to follow leader
                
                if np.random.rand() < prob:
                    # Interpolate towards best
                    alpha = np.random.uniform(0.1, 0.5)
                    new_p = (1 - alpha) * p + alpha * best_p
                    # Add small noise to avoid exact duplicates
                    new_p += np.random.normal(0, 0.1, new_p.shape)
                else:
                    # Random perturbation
                    new_p = p + np.random.normal(0, 0.5, p.shape)
            else:
                # If this structure is better than the current "best" (shouldn't happen if sorted, 
                # but maybe due to floating point or if best_p was updated)
                # Just keep it or perturb slightly
                new_p = p + np.random.normal(0, 0.1, p.shape)
                
            # Local minimize the new position
            try:
                new_e, new_pos = _local_minimize(new_p, rho, potential)
                if _min_distance(new_pos) >= 0.5:
                    new_population.append((new_e, new_pos))
            except Exception:
                pass
            
            # Update global best
            if new_e < best_energy:
                best_energy = new_e
                best_positions = new_pos
        
        # Update population: keep top K
        new_population.sort(key=lambda x: x[0])
        population = new_population[:K]
        
        # If the best structure hasn't changed in a while, we might stagnate.
        # But Firefly tends to converge.
        
    # 4. Final polish on the best structure
    if _get_remaining_time() > 5.0:
        try:
            final_e, final_p = _local_minimize(best_positions, rho, potential, time_budget=5.0)
            if _min_distance(final_p) >= 0.5 and final_e < best_energy:
                best_energy = final_e
                best_positions = final_p
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
