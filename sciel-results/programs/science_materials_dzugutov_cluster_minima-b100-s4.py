"""Dzugutov cluster global minima: Divide-and-conquer assembly + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 310.0  # Safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _rotmat(angle, axis):
    """Generate rotation matrix for angle around axis."""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + axis[0]*axis[0]*C, axis[0]*axis[1]*C - axis[2]*s, axis[0]*axis[2]*C + axis[1]*s],
        [axis[1]*axis[0]*C + axis[2]*s, c + axis[1]*axis[1]*C, axis[1]*axis[2]*C - axis[0]*s],
        [axis[2]*axis[0]*C - axis[1]*s, axis[2]*axis[1]*C + axis[0]*s, c + axis[2]*axis[2]*C]
    ])
    return R

def _generate_icosahedron(size, rng):
    """Generate a Mackay icosahedral cluster of 'size' atoms."""
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # Scale to typical Dzugutov equilibrium distance ~1.9-2.0
    # First shell radius is usually around 1.0 in reduced units if lattice constant is ~1.9?
    # Let's assume the "unit" icosahedron has radius ~1.0 for now, relax will adjust.
    scale = 1.0 
    
    positions = [np.zeros(3)] # Center
    if size <= 1:
        return np.array(positions[:size])
        
    # Add 12 vertices
    for v in verts:
        if len(positions) >= size:
            break
        positions.append(v * scale)
        
    if size <= 13:
        return np.array(positions[:size])
        
    # Add 30 edge midpoints for Mackay shell 2 (total 55)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2: # Threshold for edges in unit sphere
                edges.append((i, j))
                
    for i, j in edges:
        if len(positions) >= size:
            break
        mid = (verts[i] + verts[j]) / 2
        positions.append(mid * scale)
        
    if size <= 55:
        return np.array(positions[:size])
        
    # Add face centers for larger clusters
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    faces.append((i, j, k))
                    
    for i, j, k in faces:
        if len(positions) >= size:
            break
        centroid = (verts[i] + verts[j] + verts[k]) / 3
        centroid = centroid / np.linalg.norm(centroid)
        positions.append(centroid * scale)
        
    # Fill remaining with random points on surface
    while len(positions) < size:
        p = rng.normal(size=3)
        p /= np.linalg.norm(p)
        positions.append(p * scale)
        
    return np.array(positions[:size])

def _generate_decahedron(size, rng):
    """Generate a 12-atom decahedron (truncated icosahedron shell) or similar polytetrahedral motif."""
    # A simple decahedron can be approximated by an icosahedron with 12 vertices.
    # For diversity, we might rotate it or use a different packing.
    # Here we return a 12-atom icosahedron (vertices only) as a "decahedron-like" sub-cluster.
    return _generate_icosahedron(12, rng)

def _assemble_cluster(n, rng, strategy='icos'):
    """Assemble a cluster of n atoms from sub-clusters."""
    if strategy == 'icos':
        # Use 13-atom icosahedra as building blocks
        block_size = 13
        num_blocks = n // block_size
        remainder = n % block_size
        
        positions = []
        
        # Place first block at origin
        if num_blocks > 0:
            block1 = _generate_icosahedron(block_size, rng)
            positions.extend(block1)
            centers = [np.zeros(3)]
            orientations = [np.eye(3)]
        else:
            # n < 13, just generate a single cluster
            return _generate_icosahedron(n, rng)

        # Place remaining blocks
        for i in range(1, num_blocks):
            # Place new block near the previous ones
            # Random direction from origin, distance ~2*radius
            dir_vec = rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            dist = 2.0 # Approx distance between centers
            center = centers[-1] + dir_vec * dist
            
            # Random orientation
            axis = rng.normal(size=3)
            angle = rng.uniform(0, 2*np.pi)
            R = _rotmat(angle, axis)
            
            block = _generate_icosahedron(block_size, rng)
            rotated_block = block @ R.T
            shifted_block = rotated_block + center
            
            positions.extend(shifted_block)
            centers.append(center)
            orientations.append(R)
            
        # Add remainder atoms randomly near the surface
        if remainder > 0:
            for _ in range(remainder):
                # Pick a random existing center and add an atom nearby
                c = rng.choice(centers)
                p = c + rng.normal(0, 0.5, 3)
                positions.append(p)
                
        return np.array(positions[:n])
    else:
        # Fallback: random compact
        radius = n ** (1/3) * 0.8
        points = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _basin_hop(positions, n_iter, rng, potential, parameter):
    """Perform basin hopping to find lower energy minima."""
    best_energy, best_pos = relax(positions, potential, parameter)
    
    for i in range(n_iter):
        if _get_time_remaining() < 10:
            break
            
        # Perturb
        noise = rng.normal(0, 0.2, size=positions.shape)
        perturbed = positions + noise
        
        # Relax
        energy, relaxed = relax(perturbed, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
            
    return best_energy, best_pos

def _worker(args):
    """Worker for parallel assembly and optimization."""
    n, strategy, seed, potential, parameter = args
    rng = np.random.default_rng(seed)
    
    try:
        # 1. Assemble initial structure
        pos = _assemble_cluster(n, rng, strategy)
        
        # 2. Initial relaxation
        energy, pos = relax(pos, potential, parameter)
        
        # 3. Basin hopping
        # Budget: ~20-50 steps depending on time
        n_iter = 30
        energy, pos = _basin_hop(pos, n_iter, rng, potential, parameter)
        
        return energy, pos
    except Exception:
        return float('inf'), pos

def construct(instance):
    """Construct the lowest energy cluster structure."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Define strategies to explore in parallel
    # Strategy 0: Icosahedral assembly
    # Strategy 1: Random compact (fallback)
    # Strategy 2: Icosahedral assembly with different seed
    strategies = [
        (n, 'icos', 100, potential, parameter),
        (n, 'icos', 200, potential, parameter),
        (n, 'random', 300, potential, parameter),
        (n, 'icos', 400, potential, parameter),
    ]
    
    best_energy = float('inf')
    best_pos = None
    
    # Use ThreadPoolExecutor for parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, s) for s in strategies]
        
        for future in concurrent.futures.as_completed(futures):
            if _get_time_remaining() < 20:
                break
            try:
                energy, pos = future.result(timeout=30)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                pass
                
    # If we have time, do a final polish on the best result
    if _get_time_remaining() > 30 and best_pos is not None:
        rng = np.random.default_rng(12345)
        final_energy, final_pos = _basin_hop(best_pos, 20, rng, potential, parameter)
        if final_energy < best_energy:
            best_pos = final_pos
            
    if best_pos is None:
        # Fallback if everything failed
        rng = np.random.default_rng(0)
        best_pos = _assemble_cluster(n, rng, 'random')
        _, best_pos = relax(best_pos, potential, parameter)
        
    return best_pos
# EVOLVE-BLOCK-END
