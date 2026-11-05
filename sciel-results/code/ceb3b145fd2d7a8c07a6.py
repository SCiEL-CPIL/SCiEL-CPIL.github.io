# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit to be safe

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
    Compute Morse energy and gradient for a flat array of coordinates.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _generate_fcc_cluster(n, seed=42):
    """
    Generate an FCC-like cluster by taking the n closest points in an FCC lattice 
    to the origin. This is a strong prior for Morse cluster minima.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate radius needed to contain n atoms in FCC
    # Density of FCC is 4 atoms per unit volume (lattice constant 1).
    # Volume V ~ n/4. Radius R ~ (3V/4pi)^(1/3).
    # Let's be generous with radius to ensure we have enough points.
    radius = 1.5 * (n ** (1.0/3.0))
    
    # Generate FCC points in a cubic box [-radius, radius]^3
    # FCC basis vectors: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # Lattice constant a=1.
    
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    if not points:
        return _generate_random_cluster(n, seed)
        
    points = np.array(points)
    
    # Sort by distance to origin
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Take the n closest
    cluster = points[sorted_indices[:n]]
    
    # Add small random jitter to break symmetry and help escape saddle points if any
    jitter = rng.uniform(-0.05, 0.05, (n, 3))
    cluster = cluster + jitter
    
    return cluster

def _generate_bcc_cluster(n, seed=43):
    """
    Generate a BCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # BCC basis: (0,0,0), (0.5,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    if not points:
        return _generate_random_cluster(n, seed)
        
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    
    cluster = points[sorted_indices[:n]]
    jitter = rng.uniform(-0.05, 0.05, (n, 3))
    cluster = cluster + jitter
    
    return cluster

def _generate_random_cluster(n, seed=44):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 10000
    
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

def _refine_structure(positions, rho, potential, time_limit):
    """
    Refine a structure using L-BFGS-B for high precision.
    """
    n = positions.shape[0]
    x0 = positions.flatten()
    
    def objective(x):
        return _morse_energy_flat(x, rho)
    
    def jac(x):
        e, g = _morse_energy_grad_flat(x, rho)
        return g
        
    # Use L-BFGS-B for better convergence near minima
    try:
        res = minimize(
            objective, 
            x0, 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success:
            return res.fun, res.x.reshape(n, 3)
        else:
            # If it fails, try to relax with the provided relax function
            e, p = relax(res.x.reshape(n, 3), potential, rho)
            return e, p
    except Exception:
        try:
            e, p = relax(positions, potential, rho)
            return e, p
        except Exception:
            return 1e10, positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate diverse high-quality templates
    templates = []
    
    # FCC is the primary candidate for these cluster sizes
    try:
        templates.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
        
    try:
        templates.append(_generate_fcc_cluster(n, seed=123))
    except Exception:
        pass
        
    # BCC as secondary
    try:
        templates.append(_generate_bcc_cluster(n, seed=43))
    except Exception:
        pass
        
    # Random as fallback
    try:
        templates.append(_generate_random_cluster(n, seed=44))
    except Exception:
        pass
        
    if not templates:
        templates.append(_generate_random_cluster(n, seed=0))
        
    # 2. Refine each template
    for t_idx, template in enumerate(templates):
        if _get_remaining_time() < 10.0:
            break
            
        # Check initial validity
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(template[i] - template[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.5:
            continue
            
        # Refine
        try:
            energy, refined = _refine_structure(template, rho, potential, _get_remaining_time() - 5.0)
            
            # Verify refined structure
            min_dist_refined = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(refined[i] - refined[j])
                    if d < min_dist_refined:
                        min_dist_refined = d
                        
            if min_dist_refined >= 0.5:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = refined
        except Exception:
            pass
            
    # 3. If we have time, do a light basin hopping from the best structure to polish
    if best_positions is not None and _get_remaining_time() > 20.0:
        current_positions = best_positions.copy()
        current_energy = best_energy
        step_size = 0.3
        temp = 0.5
        max_steps = 50
        
        for _ in range(max_steps):
            if _get_remaining_time() < 5.0:
                break
                
            noise = np.random.normal(0, step_size, current_positions.shape)
            new_positions = current_positions + noise
            
            # Quick relax to check if it's better
            try:
                new_energy, new_relaxed = relax(new_positions, potential, rho)
                
                min_dist = np.inf
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(new_relaxed[i] - new_relaxed[j])
                        if d < min_dist:
                            min_dist = d
                            
                if min_dist < 0.5:
                    new_energy = float('inf')
                    
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_positions = new_relaxed
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = current_positions
                        temp *= 0.9
                        step_size *= 0.95
                else:
                    delta_e = new_energy - current_energy
                    if delta_e < 0:
                        accept = True
                    else:
                        if temp > 1e-6:
                            accept_prob = math.exp(-delta_e / temp)
                            accept = np.random.rand() < accept_prob
                        else:
                            accept = False
                            
                    if accept:
                        current_energy = new_energy
                        current_positions = new_relaxed
                    temp *= 0.95
                    step_size *= 0.98
                    
            except Exception:
                pass
                
    if best_positions is None:
        # Absolute fallback
        best_positions = templates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
