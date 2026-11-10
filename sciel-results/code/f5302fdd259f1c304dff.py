"""
Rewritten program for Dzugutov cluster global minima.
Implements a hierarchical polytetrahedral assembly strategy with adaptive basin hopping.
Focuses on Mackay icosahedra and bi-icosahedral structures which are known to be
low-energy motifs for this potential.
"""
import numpy as np
import time
import math

# We assume the evaluation environment provides these
try:
    from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
except ImportError:
    # Fallback stubs for local testing if the module is missing, 
    # but in the real eval they will be present.
    def relax(positions, potential):
        return 0.0, positions
    def dzugutov_energy_gradient(positions):
        return 0.0, np.zeros_like(positions)

_START_TIME = None
_DEADLINE_MARGIN = 10.0  # Safety margin before the 330s limit

def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _remaining_time():
    return 330.0 - _DEADLINE_MARGIN - _get_elapsed()

def _generate_icosahedron_seed(n, rng, core_radius=2.0):
    """
    Generate a Mackay icosahedral seed.
    Shell 1: 12 atoms (vertices of icosahedron)
    Shell 2: 42 atoms (faces and edges)
    Shell 3: 92 atoms
    ...
    We approximate the shells by placing points on a sphere with icosahedral symmetry constraints 
    or simply using a dense Fibonacci sphere and then snapping to symmetry if possible, 
    but for simplicity and robustness, we use a "Fibonacci Sphere" with icosahedral 
    distortion for the first shell and standard spherical shells for the rest.
    """
    points = []
    
    # Center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    if n == 1:
        return np.array(points)
    
    # 1st Shell: 12 Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    r1 = core_radius
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r1
    
    # Add vertices
    for v in verts:
        if len(points) < n:
            points.append(v)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # 2nd Shell: 42 positions
    # In a perfect Mackay icosahedron, the 2nd shell radius is r1 * sqrt(2) approx? 
    # Actually, the distance between center and 1st shell is R1. 
    # Distance between 1st shell neighbors is ~ R1.
    # The 2nd shell atoms are at the centers of the triangular faces of the 1st shell.
    # Radius R2 = R1 * sqrt(1 + 1/3) = R1 * sqrt(4/3) ? No.
    # Let's use a geometric progression for shell radii. 
    # Typical spacing is ~2.0-2.2.
    # R1 = 2.0. R2 ~ 3.0? R3 ~ 4.0?
    
    # Let's generate points on a sphere of radius R_shell
    # We will use a Fibonacci sphere distribution to ensure even coverage, 
    # then relax will fix the details.
    
    shell_idx = 1
    current_count = len(points)
    
    while current_count < n:
        shell_idx += 1
        # Increase radius. The density of atoms in icosahedral shells increases.
        # R_k ~ k * R1
        R_shell = core_radius * (1 + 0.8 * shell_idx)
        
        # Number of atoms to add
        # Mackay numbers: 1, 12, 42, 92, 142...
        # We don't need exact Mackay numbers, just enough to fill n.
        # Add a batch of points on this shell
        num_to_add = min(n - current_count, 20 + shell_idx * 10)
        
        # Fibonacci sphere points
        num_pts = num_to_add
        golden_angle = math.pi * (3 - math.sqrt(5))
        for i in range(num_pts):
            if len(points) >= n:
                break
            y = 1 - (i / float(num_pts - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y*y)
            theta = golden_angle * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            p = np.array([x, y, z]) * R_shell
            points.append(p)
            
        current_count = len(points)
        
        # If we are still far from n, increase the shell size more aggressively
        if current_count < n * 0.5 and shell_idx < 5:
             R_shell *= 1.2 # Expand slightly to avoid overlap
             # Re-generate the last batch with larger radius? 
             # For simplicity, just proceed. Relax will expand.

    points = np.array(points[:n])
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points

def _generate_bi_icosahedron_seed(n, rng, core_radius=2.0):
    """
    Generate a seed consisting of two icosahedral cores sharing a face.
    This is often the global minimum for n ~ 50-60.
    """
    if n < 45:
        # For smaller n, single icosahedron is better
        return _generate_icosahedron_seed(n, rng, core_radius)
    
    # Two cores of size ~ n/2
    n1 = n // 2
    n2 = n - n1
    
    # Generate two separate icosahedra
    ico1 = _generate_icosahedron_seed(n1, rng, core_radius)
    ico2 = _generate_icosahedron_seed(n2, rng, core_radius)
    
    # Place them side by side.
    # The distance between centers should be roughly 2 * R_core
    # R_core is the radius of the 1st shell, which is core_radius.
    # The "width" of the cluster is approx 2 * R_shell_max.
    # For a 20-atom icosahedron, R_shell_max ~ 3.0.
    # Distance between centers ~ 2 * 3.0 = 6.0?
    # Let's estimate the radius of the generated cluster.
    r1 = np.max(np.linalg.norm(ico1, axis=1))
    r2 = np.max(np.linalg.norm(ico2, axis=1))
    
    # Place ico2 at distance d along x-axis
    # To share a face, they should overlap slightly or be close.
    # Let's try d = r1 + r2 - core_radius (overlap by one shell depth)
    d = r1 + r2 - core_radius * 0.5
    
    # Shift ico2
    ico2[:, 0] += d
    
    # Combine
    points = np.vstack([ico1, ico2])
    points -= np.mean(points, axis=0)
    
    return points

def _generate_fcc_seed(n, rng):
    """
    Generate an FCC-like seed.
    """
    # Estimate lattice constant
    # Volume ~ n * (2.0)^3 / 4 (FCC packing)
    # Side length L ~ (4*n)^(1/3)
    a = 2.1
    L = int((4.0 * n) ** (1.0/3.0)) + 2
    
    points = []
    for i in range(-L//2, L//2 + 1):
        for j in range(-L//2, L//2 + 1):
            for k in range(-L//2, L//2 + 1):
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) < 10.0: # Keep compact
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return _generate_icosahedron_seed(n, rng)
        
    if len(points) > n:
        # Keep closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add random points
        while len(points) < n:
            p = rng.uniform(-5, 5, 3)
            if np.linalg.norm(p) < 5.0:
                points = np.vstack([points, [p]])
                
    points -= np.mean(points, axis=0)
    return points

def _basin_hopping(positions, potential, rng, max_time=30.0, initial_step=0.5):
    """
    Perform basin hopping on the given positions.
    Returns the best energy and positions found.
    """
    start_time = time.time()
    best_energy = float('inf')
    best_positions = positions.copy()
    
    try:
        energy, relaxed = relax(positions, potential)
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    except Exception:
        return best_energy, best_positions
        
    current_positions = best_positions
    current_energy = best_energy
    step_size = initial_step
    stagnation_count = 0
    
    while (time.time() - start_time) < max_time and _remaining_time() > 15.0:
        # Perturb
        n = len(current_positions)
        # Move a subset of atoms to reduce dimensionality of perturbation
        num_move = max(1, n // 5)
        idx = rng.choice(n, size=num_move, replace=False)
        
        perturbed = current_positions.copy()
        perturbed[idx] += rng.normal(0, step_size, (num_move, 3))
        
        try:
            new_energy, new_positions = relax(perturbed, potential)
        except Exception:
            continue
            
        if new_energy < current_energy:
            # Accept
            current_energy = new_energy
            current_positions = new_positions
            stagnation_count = 0
            # Slightly increase step size if we improved significantly? 
            # Or keep same. Let's keep same for stability.
        else:
            # Reject
            stagnation_count += 1
            # If stagnating, increase step size to escape local minimum
            if stagnation_count > 5:
                step_size *= 1.2
                stagnation_count = 0
            elif stagnation_count > 10:
                step_size *= 1.5
                stagnation_count = 0
        
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions
            
    return best_energy, best_positions

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    n = instance["n"]
    potential = instance["potential"]
    
    # If we are running out of time, return a quick valid structure
    if _remaining_time() < 5.0:
        rng = np.random.default_rng(n)
        radius = 1.0 * n ** (1.0 / 3.0) * 0.8
        points = rng.uniform(-radius, radius, (n, 3))
        return points

    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Single Icosahedron (Primary)
    seeds.append(_generate_icosahedron_seed(n, rng))
    
    # 2. Bi-Icosahedron (Crucial for n > 45)
    if n > 40:
        seeds.append(_generate_bi_icosahedron_seed(n, rng))
    
    # 3. FCC (Control)
    seeds.append(_generate_fcc_seed(n, rng))
    
    # 4. Random Compact
    radius = 1.0 * n ** (1.0 / 3.0) * 0.9
    random_seed = rng.uniform(-radius, radius, (n, 3))
    seeds.append(random_seed)
    
    # 5. Another Icosahedron with different randomization (different shell expansion)
    # We can simulate this by slightly deforming the icosahedron
    ico2 = _generate_icosahedron_seed(n, rng)
    ico2 += rng.normal(0, 0.2, ico2.shape)
    seeds.append(ico2)
    
    best_energy_global = float('inf')
    best_positions_global = seeds[0]
    
    # Time budget allocation
    # We have ~320s. 
    # Let's allocate 60s to the top 2 seeds (Ico, Bi-Ico) and 20s to others.
    # Or just run BH sequentially until time is up.
    
    # Sort seeds by initial relaxed energy? 
    # Relaxation is expensive. Let's just run BH on each seed for a fixed time.
    
    time_per_seed = 40.0 # 5 seeds * 40s = 200s. Leaves 120s buffer.
    
    for i, seed in enumerate(seeds):
        if _remaining_time() < 10.0:
            break
            
        # Reduce time per seed if we are running late
        current_budget = min(time_per_seed, _remaining_time() - 10.0)
        
        if current_budget < 5.0:
            break
            
        e, pos = _basin_hopping(seed, potential, rng, max_time=current_budget)
        
        if e < best_energy_global:
            best_energy_global = e
            best_positions_global = pos
            
        # If we found a very good energy (close to theoretical), we might stop early
        # But we don't know the theoretical minimum exactly.
        # Let's just continue.
        
    # Final Polish: Run BH on the best found structure for the remaining time
    if _remaining_time() > 10.0:
        e, pos = _basin_hopping(best_positions_global, potential, rng, max_time=_remaining_time() - 5.0)
        if e < best_energy_global:
            best_energy_global = e
            best_positions_global = pos
            
    return best_positions_global
