"""
Strategy: Multi-start Basin Hopping with diverse geometric seeds
- Uses icosahedral, decahedral, and random compact seeds.
- Implements Basin Hopping with adaptive step size.
- Deadline-guarded to ensure valid output within 300s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 295.0  # Safety margin below 300s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _icosahedron_edges_and_faces():
    """Find edges and faces of the unit icosahedron."""
    v = _icosahedron_vertices()
    n = 12
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i,j] = np.linalg.norm(v[i] - v[j])
    edge_len = np.min(d[d > 0])
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(d[i,j] - edge_len) < 1e-10:
                edges.append((i, j))
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if tuple(sorted((i,j))) in edge_set and tuple(sorted((i,k))) in edge_set and tuple(sorted((j,k))) in edge_set:
                    faces.append((i, j, k))
    return edges, faces

def _generate_decahedral_seed(n, rng):
    """Generate a seed based on FCC/decahedral packing."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    a = 1.0
    fcc_points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                fcc_points.append(np.array([i*a, j*a, k*a]))
                fcc_points.append(np.array([(i+0.5)*a, (j+0.5)*a, k*a]))
                fcc_points.append(np.array([(i+0.5)*a, j*a, (k+0.5)*a]))
                fcc_points.append(np.array([i*a, (j+0.5)*a, (k+0.5)*a]))
    centroid = np.mean(fcc_points, axis=0)
    fcc_points = [p - centroid for p in fcc_points]
    fcc_points.sort(key=lambda p: np.linalg.norm(p))
    pts = [fcc_points[i] for i in range(min(n, len(fcc_points)))]
    while len(pts) < n:
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        pts.append(dir * (2.0 + 0.2 * rng.random()))
    return np.array(pts[:n])

def _icosahedron_edges_and_faces():
    """Find edges and faces of the unit icosahedron. Returns (edges, faces) as lists of tuples."""
    v = _icosahedron_vertices()
    n = 12
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i,j] = np.linalg.norm(v[i] - v[j])
    edge_len = np.min(d[d > 0])
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(d[i,j] - edge_len) < 1e-10:
                edges.append((i, j))
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if tuple(sorted((i,j))) in edge_set and tuple(sorted((i,k))) in edge_set and tuple(sorted((j,k))) in edge_set:
                    faces.append((i, j, k))
    return edges, faces

def _icosahedron_edges_and_faces():
    """Find edges and faces of the unit icosahedron."""
    v = _icosahedron_vertices()
    n = 12
    # Compute all pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i,j] = np.linalg.norm(v[i] - v[j])
    # Minimum distance is the edge length
    edge_len = np.min(d[d > 0])
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(d[i,j] - edge_len) < 1e-10:
                edges.append((i, j))
    # Find faces: triples of vertices where all three pairs are edges
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if (tuple(sorted((i,j))), tuple(sorted((i,k))), tuple(sorted((j,k)))):
                    if tuple(sorted((i,j))) in edge_set and tuple(sorted((i,k))) in edge_set and tuple(sorted((j,k))) in edge_set:
                        faces.append((i, j, k))
    return edges, faces

def _generate_random_compact(n, rng, seed_scale=0.8):
    """
    Generate a random compact cluster using rejection sampling.
    Ensures minimum distance to avoid immediate invalidity.
    """
    points = []
    radius = 0.8 * n ** (1.0 / 3.0) * 1.5 
    
    max_attempts = 5000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.5:
                attempts += 1
                continue
        points.append(p)
        attempts += 1
    
    if len(points) < n:
        for i in range(len(points), n):
            p = np.array([i * 1.0, 0, 0])
            points.append(p)
            
    return np.array(points)

def _generate_decahedral_seed(n, rng):
    """
    Generate a seed based on decahedral (FCC) packing.
    Decahedra are common local minima for transition metal clusters.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Build an FCC-like decahedron
    # Start with a central atom and add shells
    pts = [np.array([0.0, 0.0, 0.0])]
    
    # FCC lattice vectors
    a = 1.0  # lattice constant
    # FCC positions in a cube of side a:
    # Corners: (0,0,0), (a,0,0), (0,a,0), (0,0,a), (a,a,0), (a,0,a), (0,a,a), (a,a,a)
    # Face centers: (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2), (a/2,a/2,a), (a/2,a,a/2), (a,a/2,a/2)
    
    # For a decahedron, we use 10 tetrahedral FCC grains meeting at the center
    # Simplified: use a BCC-like or just FCC sphere
    # Let's use a simple approach: place atoms on FCC lattice points within a sphere
    
    # FCC lattice points in a cube [0, a]^3
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corners
                fcc_points.append(np.array([i*a, j*a, k*a]))
                # Face centers
                if i == 0 or i == 2:
                    fcc_points.append(np.array([i*a, (j+0.5)*a, (k+0.5)*a]))
                if j == 0 or j == 2:
                    fcc_points.append(np.array([(i+0.5)*a, j*a, (k+0.5)*a]))
                if k == 0 or k == 2:
                    fcc_points.append(np.array([(i+0.5)*a, (j+0.5)*a, k*a]))
    
    # Center the FCC lattice
    centroid = np.mean(fcc_points, axis=0)
    fcc_points = [p - centroid for p in fcc_points]
    
    # Sort by distance from center
    fcc_points.sort(key=lambda p: np.linalg.norm(p))
    
    # Take the first n points
    for p in fcc_points[:n]:
        pts.append(p)
    
    # If we don't have enough, add random
    while len(pts) < n:
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        pts.append(dir * (2.0 + 0.2 * rng.random()))
    
    # Trim if needed
    pts = pts[:n]
    
    return np.array(pts)

def _generate_icosahedral_seed(n, rng):
    """
    Generate a seed based on Mackay icosahedra.
    For n=13, 55, 147, use exact structures.
    For other n, start with the largest magic number <= n and add random surface atoms.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Magic numbers
    magic = [1, 13, 55, 147]
    
    # Find largest magic <= n
    base_n = 1
    for mn in magic:
        if mn <= n:
            base_n = mn
        else:
            break
            
    if base_n == 1:
        pts = [np.array([0.0, 0.0, 0.0])]
    elif base_n == 13:
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])] + [v[i] for i in range(12)]
    elif base_n == 55:
        # Construct 55-atom Mackay icosahedron
        # Center
        pts = [np.array([0.0, 0.0, 0.0])]
        # Shell 1: 12 vertices
        v1 = _icosahedron_vertices()
        pts.extend([v1[i] for i in range(12)])
        
        # Shell 2: 42 atoms
        # This is complex to generate exactly from scratch without library support.
        # Approximation: Use the 13-atom icosahedron, scale it up, and add atoms
        # at the "midpoints" of edges and "centers" of faces of the 13-atom core,
        # then relax. Or simpler: take the 13-atom icosahedron, and for each of the 12 vertices,
        # add 3 atoms around it? 
        # Actually, a common approximation for 55 is to take 5 interpenetrating 13-atom icosahedra? No.
        # Let's use a robust heuristic: 
        # Start with 13-atom icosahedron.
        # Add 42 atoms on a sphere of radius ~ 1.8 (scaled from 1.0) distributed roughly uniformly.
        # This will likely relax into the 55-atom structure if it's the global min.
        
        # Better: Use the fact that 55 = 1 + 12 + 42.
        # The 42 atoms in shell 2 of a Mackay icosahedron are located at:
        # 30 edge midpoints of the icosahedron formed by shell 1? No, that's for the 13-atom shell.
        # In a 3-shell icosahedron (1-12-42), the 42 atoms are at the vertices of a truncated icosahedron-like structure?
        # Actually, the 55-atom cluster is often approximated by relaxing a "fullerene-like" or "icosahedral" packing.
        
        # Let's use a simpler, effective seed: 
        # Take the 13-atom icosahedron.
        # Add 42 random points on a sphere of radius 1.5 (slightly larger than 1.0).
        # This provides a good "ball" shape that can relax into the 55-atom minimum.
        
        v1 = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])]
        pts.extend([v1[i] for i in range(12)])
        
        # Add 42 random points on sphere radius 1.5
        for _ in range(42):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts.append(dir * 1.5)
            
    else: # base_n > 55 or other
        # Fallback to random compact if we don't have a specific generator
        return _generate_random_compact(n, rng)
        
    pts = np.array(pts)
    
    # If n > base_n, add random atoms on surface
    if n > len(pts):
        current_radius = 1.5 if base_n >= 13 else 0.5
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts = np.vstack([pts, [dir * (current_radius + 0.2)]])
            
    return pts

def _basin_hopping(positions, potential, kind, rng, max_iters=50, temp=0.5):
    """
    Simple Basin Hopping implementation.
    """
    n = len(positions)
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy, current_pos = best_energy, best_pos.copy()
    
    t = temp
    
    for i in range(max_iters):
        if _time_left() < 5.0:
            break
            
        # Adaptive step size: start larger, decrease
        step_size = 0.3 * (1.0 + 0.5 * np.cos(np.pi * i / max_iters))
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except:
            continue
            
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / t):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        t *= 0.95
        
    return best_energy, best_pos

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse seeds
    seeds = []
    
    # Seed 1: Icosahedral based
    try:
        seed_ico = _generate_icosahedral_seed(n, rng)
        if seed_ico is not None and len(seed_ico) == n and np.all(np.isfinite(seed_ico)):
            seeds.append(('ico', seed_ico))
    except:
        pass
        
    # Seed 2: Decahedral based
    try:
        seed_dec = _generate_decahedral_seed(n, rng)
        if seed_dec is not None and len(seed_dec) == n and np.all(np.isfinite(seed_dec)):
            seeds.append(('dec', seed_dec))
    except:
        pass
        
    # Seed 3: Random Compact
    try:
        seed_rand = _generate_random_compact(n, rng)
        if len(seed_rand) == n:
            seeds.append(('rand', seed_rand))
    except:
        pass
        
    # Seed 4: Another Random Compact (different RNG state)
    try:
        rng2 = np.random.default_rng(123)
        seed_rand2 = _generate_random_compact(n, rng2)
        if len(seed_rand2) == n:
            seeds.append(('rand2', seed_rand2))
    except:
        pass
        
    # If no seeds generated, fallback to linear chain
    if not seeds:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Run Basin Hopping on each seed
    for i, (name, seed) in enumerate(seeds):
        if _time_left() < 10.0:
            break
            
        bh_rng = np.random.default_rng(1000 + i)
        
        try:
            e, pos = _basin_hopping(seed, potential, kind, bh_rng, max_iters=40, temp=0.5)
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except Exception:
            try:
                e, pos = relax(seed, potential, kind)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except:
                pass

    # If we still have time, try more random relaxations
    if best_pos is None or _time_left() > 20.0:
        for i in range(5):
            if _time_left() < 5.0:
                break
            try:
                seed = _generate_random_compact(n, np.random.default_rng(2000 + i))
                e, pos = relax(seed, potential, kind)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except:
                pass

    # Final Fallback
    if best_pos is None:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Validate
    if not np.all(np.isfinite(best_pos)):
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Check minimum distance
    d = np.linalg.norm(best_pos[:, None, :] - best_pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.35:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END
