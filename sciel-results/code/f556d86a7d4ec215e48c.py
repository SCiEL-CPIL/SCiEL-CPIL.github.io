"""
Rewritten program for Dzugutov cluster global minima.
Focuses on icosahedral symmetry and high-density random packings,
relaxed in parallel with strict time management.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 330.0
_SAFETY_MARGIN = 10.0  # Stop accepting new work 10s before hard limit

def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _remaining_time():
    return _HARD_LIMIT - _SAFETY_MARGIN - _get_elapsed()

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedral cluster.
    Layers: 1, 12, 42, 120, 300...
    We generate a large enough icosahedron and trim to n.
    """
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms (vertices)
    # Distance ~ 1.9-2.0
    r1 = 1.95
    shell1 = verts * r1
    points.extend(shell1)
    
    if n <= len(points):
        return np.array(points[:n])

    # Shell 2: 42 atoms
    # These are located at the midpoints of edges and other specific positions.
    # A simpler approximation for a dense icosahedral-like cluster is to add
    # points on a slightly larger sphere, distributed to fill gaps.
    # For Mackay, the next shell is 42.
    # We can approximate by placing atoms on a sphere of radius r2 ~ 2 * r1 * (sqrt(5)-1)/2 ? 
    # Actually, let's just do a dense random packing on a sphere with icosahedral bias.
    
    # To be safe and fast, we generate a dense random cluster with high symmetry bias.
    # We will generate a "liquid" drop and let relax do the work, but seed it with icosahedral core.
    
    # Add remaining atoms in a shell around the core
    target_n = n
    current_n = len(points)
    
    if target_n > current_n:
        # Estimate radius for the rest
        # Volume ~ N * v0. R ~ (3N/4pi v0)^(1/3)
        # For Dzugutov, density is high.
        r_outer = (3.0 * target_n / (4.0 * np.pi * 10.0)) ** (1.0/3.0) * 1.5
        # Ensure r_outer is larger than r1
        r_outer = max(r_outer, r1 * 1.5)
        
        # Generate random points in the shell [r1*1.1, r_outer]
        # Use rejection sampling
        attempts = 0
        max_attempts = 10000
        while len(points) < target_n and attempts < max_attempts:
            # Random point in sphere
            u = rng.uniform(0, 1)
            r = r1 * 1.1 + (r_outer - r1 * 1.1) * u**(1.0/3.0)
            # Actually, uniform in volume is r = R * u^(1/3)
            # We want uniform in the shell volume.
            # Let's just sample in a big sphere and reject if too close to center or existing points.
            
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * r
            # r is uniform in [r_min, r_max]
            
            # Check distance to existing points
            if points:
                dists = np.linalg.norm(points - p, axis=1)
                if np.min(dists) < 1.8: # Keep them apart
                    attempts += 1
                    continue
            points.append(p)
            attempts += 1
            
        # If we still don't have enough, just add random points far away (bad, but valid)
        while len(points) < target_n:
            p = rng.uniform(-r_outer, r_outer, 3)
            points.append(p)
            
    points = np.array(points[:target_n])
    points -= np.mean(points, axis=0)
    return points

def _generate_dense_random(n, rng, min_dist=1.8):
    """
    Generate a dense random cluster using rejection sampling.
    """
    # Estimate radius
    # Assume packing fraction ~ 0.7, atom volume ~ pi/6 * d^3
    # Let's just use a heuristic radius
    r_est = (3.0 * n / (4.0 * np.pi * 8.0)) ** (1.0/3.0) * 1.5
    # Ensure it's large enough
    r_est = max(r_est, 3.0)
    
    points = []
    attempts = 0
    max_attempts = 5000 * n
    
    # Start with a central atom to help packing
    points.append(np.array([0.0, 0.0, 0.0]))
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-r_est, r_est, 3)
        # Optional: bias towards center? No, uniform is fine.
        if np.linalg.norm(p) > r_est:
            attempts += 1
            continue
            
        if len(points) > 0:
            dists = np.linalg.norm(points - p, axis=1)
            if np.min(dists) < min_dist:
                attempts += 1
                continue
                
        points.append(p)
        attempts += 1
        
    # Fallback if we failed to pack enough
    while len(points) < n:
        p = rng.uniform(-r_est, r_est, 3)
        points.append(p)
        
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _relax_worker(args):
    """
    Worker for parallel relaxation.
    """
    positions, potential, parameter, timeout = args
    try:
        # The relax function signature is relax(positions, potential, parameter)
        # It returns (energy, relaxed_positions)
        energy, relaxed_positions = relax(positions, potential, parameter)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def construct(instance):
    """
    Find the lowest energy structure for the given cluster size.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check if we have time
    if _remaining_time() < 5.0:
        # Emergency fallback: return a simple random cluster
        rng = np.random.default_rng(size)
        r = 3.0
        points = rng.uniform(-r, r, (size, 3))
        return points

    # Generate a portfolio of initial guesses
    rng = np.random.default_rng(size)
    initial_clusters = []
    
    # 1. Icosahedral-like (Mackay approximation)
    try:
        ico = _generate_mackay_icosahedron(size, rng)
        initial_clusters.append(ico)
    except Exception:
        pass
        
    # 2. Dense Random Packings (multiple seeds)
    # We want to try a few different random packings to escape local minima
    n_randoms = 3
    for i in range(n_randoms):
        try:
            # Vary the min_dist slightly to get different packing structures
            min_d = 1.8 + 0.1 * i
            rand = _generate_dense_random(size, rng, min_dist=min_d)
            initial_clusters.append(rand)
        except Exception:
            pass
            
    # If we somehow have no clusters (shouldn't happen), add one
    if not initial_clusters:
        rand = _generate_dense_random(size, rng)
        initial_clusters.append(rand)
        
    # Relax in parallel
    # We have 4 threads max.
    # We have ~4-5 candidates.
    
    args_list = []
    for cluster in initial_clusters:
        # Ensure cluster is (n, 3)
        if cluster.shape != (size, 3):
            continue
        args_list.append((cluster, potential, parameter, _remaining_time()))
        
    if not args_list:
        return initial_clusters[0]
        
    best_energy = float('inf')
    best_positions = initial_clusters[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, args) for args in args_list]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            
            # Check deadline
            if _remaining_time() < 2.0:
                break
                
    return best_positions
