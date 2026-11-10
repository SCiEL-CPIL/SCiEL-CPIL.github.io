"""
Rewritten program for Dzugutov cluster global minima.
Uses a portfolio of structural heuristics (FCC, Random, Icosahedral-like)
relaxed in parallel to find low-energy configurations.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit to stop

def _generate_fcc_chunk(n, rng):
    """Generate an FCC lattice chunk roughly spherical."""
    # Estimate radius
    # Density of FCC is high. Let's say volume per atom ~ 10-15 in reduced units.
    # R ~ (3*N / 4*pi * 12)^(1/3)
    R = (3.0 * n / (4.0 * np.pi) * 12.0) ** (1.0 / 3.0) * 1.1
    a = 2.2  # Lattice constant approx equilibrium distance
    
    # Generate FCC points in a box
    # FCC primitive vectors
    # Points: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) * a
    points = []
    # We need to generate points within a sphere of radius R
    # Iterate over a cubic grid
    max_idx = int(R / a) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis vectors
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return None
    
    # If we have more points than needed, select the closest to the center
    if len(points) > n:
        # Calculate distances from center
        dists = np.linalg.norm(points, axis=1)
        # Get indices of the n smallest distances
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # If we don't have enough, add random points
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                # Check minimum distance
                if len(points) > 0:
                    d = np.linalg.norm(points - p, axis=1)
                    if np.min(d) >= 0.5:
                        points = np.vstack([points, [p]])
                else:
                    points = np.array([[p]])
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    return points

def _generate_random_cluster(n, rng):
    """Generate a random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    max_attempts = 1000 * n
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fallback: just place them randomly without distance check
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _generate_icosahedral_like(n, rng):
    """Generate an icosahedral-like cluster."""
    # Start with a central atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Add icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    r = 1.1  # Distance from center to first shell
    
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r
    points.extend(verts)
    
    # Add more shells if needed
    shell_radius = r
    while len(points) < n:
        shell_radius += 1.1
        # Generate points on a sphere of radius shell_radius
        num_to_add = min(n - len(points), 20)
        for _ in range(num_to_add):
            # Random direction
            u = rng.uniform(-1, 1)
            v = rng.uniform(0, 2 * np.pi)
            p = np.array([
                np.sqrt(1 - u**2) * np.cos(v),
                np.sqrt(1 - u**2) * np.sin(v),
                u
            ]) * shell_radius
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential, parameter = args
    try:
        energy, relaxed_positions = relax(positions, potential, parameter)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def construct(instance):
    """Best of multiple random compact starting clusters, each relaxed locally."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > 300 - _DEADLINE_MARGIN:
        # Return a simple random cluster if we're out of time
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    # Generate different types of initial clusters
    rng = np.random.default_rng(size)
    
    initial_clusters = []
    
    # 1. FCC-like cluster
    fcc = _generate_fcc_chunk(size, rng)
    if fcc is not None:
        initial_clusters.append(fcc)
    
    # 2. Random clusters
    for _ in range(3):
        random_cluster = _generate_random_cluster(size, rng)
        initial_clusters.append(random_cluster)
    
    # 3. Icosahedral-like cluster
    ico = _generate_icosahedral_like(size, rng)
    initial_clusters.append(ico)
    
    # Relax all clusters in parallel
    args_list = [(cluster, potential, parameter) for cluster in initial_clusters]
    
    best_energy = float('inf')
    best_positions = initial_clusters[0]
    
    # Use a thread pool for parallel relaxation
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
            if time.time() - _START_TIME > 300 - _DEADLINE_MARGIN:
                break
    
    return best_positions
