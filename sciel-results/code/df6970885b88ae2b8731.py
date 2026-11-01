"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import science_chemistry

def _calc_energy(pos):
    """Calculate Lennard-Jones energy."""
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    mask = dists > 0.5
    r2 = dists[mask]**2
    r6 = r2**3
    r12 = r6**2
    return 4.0 * np.sum(r12 - r6)

def _get_icosahedron_vectors():
    """Generate normalized vectors for icosahedron vertices, edge midpoints, and face centers."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 Vertices
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    # 15 Edge Midpoints
    # Edges connect vertices with distance 2 (in unnormalized coords, dist is 2).
    # Normalized dist is 2/sqrt(1+phi^2).
    # We can just find pairs with minimum distance.
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (v[i] + v[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
    e = np.array(edges)
    
    # 10 Face Centers
    # Faces are triangles of 3 vertices.
    # We can find triples with mutual distance min_dist.
    faces = []
    seen = set()
    for i in range(12):
        for j in range(12):
            if i == j: continue
            for k in range(12):
                if i == k or j == k: continue
                if abs(dists[i, j] - min_dist) < 1e-6 and abs(dists[j, k] - min_dist) < 1e-6 and abs(dists[k, i] - min_dist) < 1e-6:
                    key = tuple(sorted([i, j, k]))
                    if key not in seen:
                        seen.add(key)
                        center = (v[i] + v[j] + v[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        faces.append(center)
    f = np.array(faces)
    
    return v, e, f

def construct():
    v, e, f = _get_icosahedron_vectors()
    
    # Center
    center = np.array([0.0, 0.0, 0.0])
    
    # Inner Shell (12 atoms)
    R_inner = 1.1
    inner_shell = v * R_inner
    
    # Outer Shell (42 atoms)
    # Pool of directions: 12 vertices, 15 edges, 10 faces = 37 directions
    pool = np.vstack([v, e, f])
    
    best_positions = None
    best_energy = float('inf')
    
    # Try a few variations to ensure convergence to global min
    for trial in range(5):
        np.random.seed(trial * 100 + 42)
        
        # Select 42 directions from the pool
        # We can just repeat the pool and shuffle
        indices = np.random.choice(len(pool), 42)
        outer_dirs = pool[indices]
        
        # Add small random perturbation to avoid exact degeneracies
        noise = np.random.normal(0, 0.01, outer_dirs.shape)
        outer_dirs = outer_dirs + noise
        outer_dirs = outer_dirs / np.linalg.norm(outer_dirs, axis=1, keepdims=True)
        
        R_outer = 2.2
        outer_shell = outer_dirs * R_outer
        
        positions = np.vstack([center, inner_shell, outer_shell])
        
        # Relax
        try:
            relaxed = science_chemistry.relax(positions)
            energy = _calc_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
