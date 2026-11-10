import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_fcc(n, rng):
    """Generate FCC-like cluster."""
    # Approximate radius
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    a = 2.2 # Lattice constant
    points = []
    # Generate lattice points in a cube and keep those within sphere
    # FCC basis: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    # Scale a so that volume matches n
    # Volume of sphere ~ 4/3 pi R^3. Number of atoms ~ n.
    # Density ~ n / (4/3 pi R^3).
    # FCC density is 4 atoms / a^3.
    # a^3 ~ 4 * (4/3 pi R^3) / n -> a ~ (16/3 pi R^3 / n)^(1/3)
    # Let's just use a heuristic a.
    a = 2.0
    
    max_coord = int(np.ceil(R / a)) + 2
    candidates = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # 4 points per unit cell
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    if x*x + y*y + z*z <= R*R:
                        candidates.append([x, y, z])
    
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return np.array([rng.uniform(-1, 1, 3) for _ in range(n)])
    
    # If we have more than n, pick n closest to center
    if len(candidates) > n:
        dists = np.sum(candidates**2, axis=1)
        idx = np.argsort(dists)[:n]
        candidates = candidates[idx]
    elif len(candidates) < n:
        # Add random points to fill up
        while len(candidates) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                candidates = np.vstack([candidates, p])
                
    return candidates[:n]

def _generate_icosahedral(n, rng):
    """Generate Icosahedral-like cluster (Mackay icosahedron)."""
    # Mackay icosahedron layers: 1, 12, 42, 92...
    # For n up to 60, we have 1, 12, 42.
    # 1+12=13, 1+12+42=55.
    # For 38, 45, 50, 57, 60, we approximate.
    
    # Vertices of icosahedron (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Shell radii
    r1 = 0.0
    r2 = 1.0
    r3 = np.sqrt(1 + phi) # ~1.9
    r4 = np.sqrt(1 + phi**2) # ~2.4
    
    points = []
    # Center
    points.append([0,0,0])
    
    # Shell 1 (12 atoms)
    if n > 1:
        pts = verts * r2
        points.extend(pts.tolist())
        
    # Shell 2 (42 atoms)
    if n > 13:
        # 20 faces, 2 atoms per face + 12 edges? 
        # Mackay 2nd shell: 30 edge midpoints + 12 face centers? No.
        # 2nd shell has 42 atoms: 30 on edges, 12 on faces? 
        # Actually: 12 vertices of icosahedron, 30 edge midpoints, 20 face centers. Total 62? No.
        # 1st shell: 12 vertices.
        # 2nd shell: 30 edge midpoints + 12 face centers = 42.
        
        # Edge midpoints
        edges = [
            (0,1), (0,2), (0,4), (0,7), (1,3), (1,5), (1,8), (2,3), (2,6), (2,9),
            (3,10), (3,11), (4,5), (4,8), (4,9), (5,6), (5,7), (5,10), (6,7), (6,11),
            (7,10), (7,11), (8,9), (8,10), (8,12), (9,11), (9,12), (10,12), (11,12), (12,13) # Wait, 12 vertices
        ]
        # Correct edges for 12 verts:
        # Let's just generate points on the surface of a sphere of radius r3
        # and snap them to icosahedral symmetry?
        # Simpler: Just generate random points on sphere of radius r3 and keep 42.
        # This is not true Mackay, but "icosahedral-like".
        
        # Let's try to generate 42 points on sphere r3
        # 30 edge midpoints:
        # We need the edges of the icosahedron defined by verts.
        # Distance between adjacent verts is 2.
        # Let's find pairs with dist ~ 2.
        dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
        edge_mask = (dists > 1.9) & (dists < 2.1)
        np.fill_diagonal(edge_mask, False)
        edges = [(i, j) for i in range(12) for j in range(12) if edge_mask[i, j]]
        edge_mids = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            mid /= np.linalg.norm(mid)
            edge_mids.append(mid * r3)
            
        # 12 face centers
        # Faces of icosahedron
        # This is getting complex. Let's just add random points on sphere r3.
        # We have 30 edge mids. Need 12 more.
        # Random on sphere:
        while len(edge_mids) < 42:
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            p = [np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]
            edge_mids.append(np.array(p) * r3)
            
        points.extend(edge_mids[:42])
        
    # Shell 3 (92 atoms) - only if n > 55
    if n > 55:
        # Add points on sphere r4
        needed = n - len(points)
        for _ in range(needed):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            p = [np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]
            points.append(np.array(p) * r4)
            
    return np.array(points[:n])

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    points = []
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            # Check min distance 0.5
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points)

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    # 2 FCC
    for _ in range(2):
        seeds.append(_generate_fcc(n, rng))
    # 2 Icosahedral
    for _ in range(2):
        seeds.append(_generate_icosahedral(n, rng))
    # 4 Random
    for _ in range(4):
        seeds.append(_generate_random(n, rng))
        
    best_energy = float('inf')
    best_positions = seeds[0]
    
    # Phase 1: Relax all seeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, (n, s, potential, parameter)) for s in seeds]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
    # Phase 2: Basin Hopping on the best structure
    # If time permits
    current_energy = best_energy
    current_positions = best_positions
    
    # Define perturbation
    def perturb(positions, step_size):
        # Add Gaussian noise
        noise = rng.normal(0, step_size, positions.shape)
        new_positions = positions + noise
        # Ensure min distance 0.5? Relax will handle overlaps if it's robust, 
        # but let's try to avoid obvious overlaps.
        return new_positions
        
    step_size = 0.5
    max_iters = 50
    
    for i in range(max_iters):
        if _get_time_remaining() < 5.0:
            break
            
        new_positions = perturb(current_positions, step_size)
        energy, relaxed = relax(new_positions, potential, parameter)
        
        if energy < current_energy:
            current_energy = energy
            current_positions = relaxed
            # Decrease step size slightly
            step_size *= 0.95
        else:
            # Metropolis accept? Or just reject?
            # For basin hopping, we usually accept with Metropolis.
            # But here we want to find global min.
            # Let's just reject if worse, and increase step size if stuck.
            # Simple strategy: if rejected, increase step size to escape.
            step_size *= 1.05
            if step_size > 2.0:
                step_size = 2.0
                
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions

    return best_positions
