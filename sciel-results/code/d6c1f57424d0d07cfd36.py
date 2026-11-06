import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_GLOBAL_START_TIME = None

def _get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_icosahedron_edges(verts):
    """
    Find the 30 edges of the icosahedron.
    An edge connects two vertices if their distance is the minimum distance.
    """
    dists = np.linalg.norm(verts[:, np.newaxis] - verts[np.newaxis, :], axis=2)
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    return edges

def _generate_mackay_55():
    """
    Generate the 55-atom Mackay icosahedron.
    """
    verts = _get_icosahedron_vertices()
    R1 = 1.2
    core = np.vstack([np.zeros(3), verts * R1])
    
    edges = _get_icosahedron_edges(verts)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid) # Normalize to unit sphere
        midpoints.append(mid)
    
    midpoints = np.array(midpoints)
    R2 = R1 * 1.5 # Approximate second shell radius
    shell2 = midpoints * R2
    
    positions = np.vstack([core, shell2])
    return positions

def _generate_seed(n, rng):
    """
    Generate a seed for n atoms.
    """
    if n == 55:
        return _generate_mackay_55()
    
    # For other n, we can start with the 55-atom cluster and modify it.
    # Or start with 13-atom and add.
    # Let's start with 13-atom and add random points on the surface.
    verts = _get_icosahedron_vertices()
    R1 = 1.2
    core = np.vstack([np.zeros(3), verts * R1])
    
    if n <= 13:
        # For n < 13, just take a subset of the core
        return core[:n]
    
    # Add n-13 atoms
    extra = []
    for _ in range(n - 13):
        # Random point on the surface of the icosahedron
        # Pick a random face and a random point on it
        # Or just random on sphere
        while True:
            p = rng.uniform(-1, 1, 3)
            if 0.5 < np.linalg.norm(p) < 1.0:
                p = p / np.linalg.norm(p)
                extra.append(p * 1.8) # Radius for extra atoms
                break
    
    positions = np.vstack([core, np.array(extra)])
    return positions

def construct(instance):
    global _GLOBAL_START_TIME
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    
    size = instance["n"]
    deadline = _GLOBAL_START_TIME + 300.0 - 5.0
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate initial seed
    current_positions = _generate_seed(size, rng)
    
    # Relax
    try:
        energy, positions = relax(current_positions, 'dzugutov')
        best_energy = energy
        best_positions = positions.copy()
    except Exception:
        energy = float('inf')
        positions = current_positions
    
    # 2. Basin Hopping
    step_size = 0.2
    temperature = 0.5
    
    while time.time() < deadline:
        perturbation = rng.normal(0, step_size, size=(size, 3))
        new_positions = best_positions + perturbation
        
        try:
            new_energy, new_relaxed_positions = relax(new_positions, 'dzugutov')
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed_positions.copy()
            else:
                delta_e = new_energy - best_energy
                if delta_e < 0:
                    best_energy = new_energy
                    best_positions = new_relaxed_positions.copy()
                else:
                    acceptance_prob = np.exp(-delta_e / temperature)
                    if rng.random() < acceptance_prob:
                        best_positions = new_relaxed_positions.copy()
                        
        except Exception:
            continue
            
        temperature *= 0.95
        if temperature < 1e-4:
            temperature = 1e-4
            
    return best_positions
