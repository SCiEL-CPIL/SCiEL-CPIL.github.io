"""Dzugutov cluster global minima search using parallel-replica basin hopping with icosahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

WORKING_LIMIT = 305.0
START_TIME = time.time()

def _remaining_time():
    return WORKING_LIMIT - (time.time() - START_TIME)

def _icosahedron_vertices():
    """12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return v

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix using quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

def _build_icosahedral_seed(n, rng):
    """Build an n-atom seed based on icosahedral packing."""
    # Start with 13-atom icosahedron
    verts = _icosahedron_vertices()
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * 1.0
    
    positions = [np.array([0.0, 0.0, 0.0])]  # center
    positions.extend([v.copy() for v in verts])  # 12 vertices
    
    # If n > 13, add atoms on outer shells
    if n > 13:
        # Add atoms at icosahedral face centers and further shells
        # Face centers of icosahedron: 20 faces
        # Compute face centers
        from itertools import combinations
        face_centers = []
        for i, j, k in combinations(range(12), 3):
            # Check if these 3 vertices form a face (edge length ~ 1.051)
            d1 = np.linalg.norm(verts[i] - verts[j])
            d2 = np.linalg.norm(verts[j] - verts[k])
            d3 = np.linalg.norm(verts[k] - verts[i])
            if abs(d1 - 1.051) < 0.05 and abs(d2 - 1.051) < 0.05 and abs(d3 - 1.051) < 0.05:
                center = (verts[i] + verts[j] + verts[k]) / 3.0
                face_centers.append(center)
        
        # Add face centers (20 atoms)
        n_faces = min(len(face_centers), n - 13)
        for i in range(n_faces):
            fc = face_centers[i].copy()
            # Push slightly outward
            fc = fc * 1.15
            positions.append(fc)
        
        # If still need more, add random atoms in compact region
        n_remaining = n - len(positions)
        for _ in range(n_remaining):
            r = rng.uniform(1.5, 2.5)
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.uniform(0, np.pi)
            pos = np.array([
                r*np.sin(phi)*np.cos(theta),
                r*np.sin(phi)*np.sin(theta),
                r*np.cos(phi)
            ])
            positions.append(pos)
    
    positions = np.array(positions[:n])
    
    # Apply random rotation
    R = _random_rotation_matrix(rng)
    positions = positions @ R.T
    
    return positions

def _build_random_compact_seed(n, rng):
    """Build a random compact seed."""
    positions = rng.normal(0, 1.2, (n, 3))
    return positions

def _ensure_min_distance(positions, min_dist=0.5, max_iter=50):
    """Ensure all pairs are at least min_dist apart."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break
        # Find closest pair
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        
        if min_d >= min_dist:
            break
        
        i, j = min_pair
        direction = (positions[i] - positions[j]) / (min_d + 1e-10)
        push = (min_dist - min_d) / 2 + 0.01
        positions[i] += direction * push
        positions[j] -= direction * push
    
    return positions

def _basin_hopping_kick(positions, potential, parameter, max_iterations=40, rng=None):
    """Basin hopping with adaptive kick size."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    amplitude = 0.3
    beta = 1.0 / 0.3
    stagnation_count = 0
    
    for it in range(max_iterations):
        if _remaining_time() < 15.0:
            break
        
        # Adaptive kick: increase if stuck
        perturbed = current_pos + rng.normal(0, amplitude, positions.shape)
        new_energy, new_pos = relax(perturbed, potential, parameter)
        
        delta_e = new_energy - current_energy
        accept = delta_e < 0 or rng.random() < np.exp(-beta * delta_e)
        
        if accept:
            current_energy = new_energy
            current_pos = new_pos
            stagnation_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            stagnation_count += 1
            # Increase kick size when stuck
            if stagnation_count > 5:
                amplitude *= 1.1
                stagnation_count = 0
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov clusters using parallel-replica basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(n * 17 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Number of parallel replicas
    n_replicas = 6
    
    replica_energies = []
    replica_positions = []
    
    for rep in range(n_replicas):
        if _remaining_time() < 30.0:
            break
        
        # Alternate between icosahedral and random seeds
        if rep % 2 == 0:
            seed_pos = _build_icosahedral_seed(n, rng)
        else:
            seed_pos = _build_random_compact_seed(n, rng)
        
        # Ensure minimum distance before relaxation
        seed_pos = _ensure_min_distance(seed_pos)
        
        # Basin hopping
        energy, pos = _basin_hopping_kick(seed_pos, potential, parameter, max_iterations=35, rng=rng)
        
        replica_energies.append(energy)
        replica_positions.append(pos)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_pos = relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _ensure_min_distance(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
