"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

START_TIME = time.time()
TIME_LIMIT = 115.0 # Leave 5s buffer

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for a set of positions."""
    dists = pdist(pos, 'euclidean')
    if np.any(dists < 1e-12):
        return 1e10
    inv_r2 = 1.0 / (dists * dists)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy

def _lj_energy_and_grad(x):
    """Compute Lennard-Jones energy and gradient efficiently."""
    pos = x.reshape(-1, 3)
    n = pos.shape[0]
    
    # Compute pairwise distances
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
    
    inv_r2 = 1.0 / dists_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: sum over i < j
    # The matrix inv_r12 - inv_r6 has 2x the energy because it sums i!=j
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    
    # Gradient
    # dE/dx_i = sum_{j != i} 4 * (-12 * r_ij^-14 * (x_i - x_j) + 6 * r_ij^-8 * (x_i - x_j))
    #         = sum_{j != i} (x_i - x_j) * 4 * (6 * r_ij^-8 - 12 * r_ij^-14)
    #         = sum_{j != i} (x_i - x_j) * (24 * r_ij^-8 - 48 * r_ij^-14)
    
    # coeff[i, j] = 24 * r_ij^-8 - 48 * r_ij^-14
    # Note: r^-8 = (r^-2)^4, r^-14 = (r^-2)^7
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    coeff = 24.0 * inv_r8 - 48.0 * inv_r14
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i, j] * diff[i, j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def _relax(pos):
    """Relax a structure using L-BFGS-B with the efficient gradient."""
    if time.time() - START_TIME > TIME_LIMIT:
        return pos
    
    pos = pos.copy()
    try:
        result = minimize(
            _lj_energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        relaxed = result.x.reshape(-1, 3)
        
        # Check validity
        dists = pdist(relaxed, 'euclidean')
        if np.min(dists) < 0.5:
             # If invalid, try to push atoms apart or return original
             # For now, return original if relaxation broke constraints
             return pos
        return relaxed
    except Exception:
        return pos

def _valid(pos):
    """Check if positions are valid (min distance >= 0.5)."""
    dists = pdist(pos, 'euclidean')
    return np.min(dists) >= 0.5

def _make_mackay_seed():
    """
    Create a Mackay icosahedron seed for 55 atoms.
    1 center + 12 (shell 1) + 42 (shell 2)
    Shell 1: 12 vertices of icosahedron.
    Shell 2: 30 atoms at edge midpoints of icosahedron, 12 atoms at face centers?
    Actually, the 42 atoms in the second shell of a Mackay icosahedron are:
    - 30 atoms located at the midpoints of the 30 edges of the icosahedron (scaled).
    - 12 atoms located at the vertices of a dodecahedron inscribed in the icosahedron?
    Let's use a simpler geometric construction that is close enough for a seed.
    We will place 12 vertices of an icosahedron.
    We will place 30 edge midpoints.
    We will place 12 points on the sphere at the "face centers" directions?
    Icosahedron has 20 faces. Dodecahedron has 12 faces.
    The dual of the icosahedron is the dodecahedron. The vertices of the dodecahedron correspond to the face centers of the icosahedron.
    So, 12 dodecahedron vertices = 12 face centers of icosahedron.
    Total 30 + 12 = 42.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Scale Shell 1
    R1 = 1.12
    shell1 = verts * R1
    
    # Find edges: pairs of vertices with distance ~ 2*sin(pi/5) in unit sphere?
    # Distance between adjacent vertices in unit icosahedron is 2/sqrt(phi^2+1) ?
    # Let's just find all pairs with small distance.
    dists = pdist(verts)
    # The 30 edges are the shortest distances.
    # Reconstruct pairs
    pairs = []
    min_dist = np.min(dists)
    # There are 30 edges.
    # We need to map pdist indices back to (i, j)
    n = len(verts)
    idx = 0
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(dists[idx] - min_dist) < 1e-6:
                edges.append((i, j))
            idx += 1
            
    # Calculate midpoints of these edges on the unit sphere, then scale
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid = mid / mid_norm
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Find face centers (vertices of dual dodecahedron)
    # A face is defined by 3 vertices. The center is the normalized sum of the 3 vertices.
    # We need to identify the 20 faces.
    # In an icosahedron, each face is a triangle of mutually connected vertices.
    face_centers = []
    # Iterate over all triplets? Too slow for 12 vertices (220 triplets).
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Check if i,j,k form a face (all pairwise connected)
                # We have the edge list. Let's create a set of edges for fast lookup
                # Actually, we can just check distances.
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                if abs(d_ij - min_dist) < 1e-6 and abs(d_ik - min_dist) < 1e-6 and abs(d_jk - min_dist) < 1e-6:
                    center = verts[i] + verts[j] + verts[k]
                    center_norm = np.linalg.norm(center)
                    if center_norm > 1e-6:
                        center = center / center_norm
                        face_centers.append(center)
    
    face_centers = np.array(face_centers)
    # We should have 20 face centers. But we only need 12 for the Mackay second shell?
    # Wait, the Mackay icosahedron second shell has 42 atoms.
    # 30 edge midpoints + 12 ???
    # The 12 atoms are often described as being at the vertices of a dodecahedron.
    # But a dodecahedron has 12 vertices.
    # Perhaps the 12 atoms are a subset of the 20 face centers?
    # Or maybe my count is wrong.
    # Let's check: 1 + 12 + 42 = 55.
    # If I can't get the perfect Mackay seed, I will use the 30 edge midpoints and 12 random points on the sphere, or just use the 20 face centers and remove 8?
    # Let's try to use all 20 face centers + 22 others? No.
    # Let's just use the 30 edge midpoints and 12 points from the 20 face centers (e.g., every other one? or just first 12).
    # Given the GA will relax, an approximate seed is fine.
    
    # Scale Shell 2
    R2 = 2.24 # Approximate
    
    # We have 30 midpoints.
    # We have 20 face centers.
    # We need 42 total.
    # Let's take all 30 midpoints and 12 of the face centers.
    # To choose 12 out of 20, we can just take the first 12.
    # This breaks symmetry slightly but is a good seed.
    
    shell2_part1 = midpoints * R2
    shell2_part2 = face_centers[:12] * R2
    
    shell2 = np.vstack([shell2_part1, shell2_part2])
    
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, shell1, shell2])
    
    # Relax this seed to get a better starting point for the GA population
    relaxed_seed = _relax(all_points)
    if _valid(relaxed_seed):
        return relaxed_seed
    else:
        return all_points

def construct():
    """Genetic algorithm to find the global minimum of 55-atom LJ cluster."""
    np.random.seed(42)
    
    # Initial population
    pop_size = 40
    population = []
    energies = []
    
    # Add Mackay seed
    mackay_seed = _make_mackay_seed()
    population.append(mackay_seed)
    energies.append(_lj_energy(mackay_seed))
    
    # Add other seeds: FCC fragment, HCP fragment, Random
    def make_fcc_seed():
        # Generate FCC lattice points
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    pts.append([i, j, k])
        # FCC points are (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        fcc_pts = []
        for p in pts:
            fcc_pts.append(p)
            fcc_pts.append([p[0]+0.5, p[1]+0.5, p[2]])
            fcc_pts.append([p[0]+0.5, p[1], p[2]+0.5])
            fcc_pts.append([p[0], p[1]+0.5, p[2]+0.5])
        fcc_pts = np.array(fcc_pts)
        # Scale to LJ distance ~1.12
        # FCC nearest neighbor distance is a/sqrt(2).
        # If a=1, dist = 0.707. We want dist = 1.12.
        # Scale factor = 1.12 / 0.707 = 1.58
        fcc_pts *= 1.58
        
        # Select 55 closest to center
        dists = np.linalg.norm(fcc_pts, axis=1)
        idxs = np.argsort(dists)[:55]
        return fcc_pts[idxs]
        
    def make_random_seed():
        # Random points in a sphere of radius 4
        while True:
            pts = np.random.randn(N, 3) * 2.5
            if _valid(pts):
                return pts
                
    seeds = [make_fcc_seed(), make_random_seed(), make_random_seed()]
    for s in seeds:
        population.append(s)
        energies.append(_lj_energy(s))
        
    # Fill rest with random seeds
    while len(population) < pop_size:
        s = make_random_seed()
        population.append(s)
        energies.append(_lj_energy(s))
        
    # Relax all initial population members
    relaxed_population = []
    relaxed_energies = []
    for i, pos in enumerate(population):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        relaxed = _relax(pos)
        if _valid(relaxed):
            relaxed_population.append(relaxed)
            relaxed_energies.append(_lj_energy(relaxed))
        else:
            relaxed_population.append(pos)
            relaxed_energies.append(energies[i])
            
    # Ensure we have enough population
    while len(relaxed_population) < pop_size:
        s = make_random_seed()
        relaxed_population.append(s)
        relaxed_energies.append(_lj_energy(s))
        
    best_energy = min(relaxed_energies)
    best_pos = relaxed_population[np.argmin(relaxed_energies)]
    
    # GA Loop
    max_generations = 100
    gen = 0
    while gen < max_generations:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        gen += 1
        
        # Sort by energy
        sorted_indices = np.argsort(relaxed_energies)
        sorted_pop = [relaxed_population[i] for i in sorted_indices]
        sorted_energies = [relaxed_energies[i] for i in sorted_indices]
        
        # Elitism: Keep top 2
        new_population = [sorted_pop[0].copy(), sorted_pop[1].copy()]
        new_energies = [sorted_energies[0], sorted_energies[1]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select(pop, k=3):
                idxs = np.random.choice(len(pop), k, replace=False)
                best_idx = idxs[np.argmin([relaxed_energies[i] for i in idxs])]
                return pop[best_idx].copy()
                
            parent1 = tournament_select(relaxed_population)
            parent2 = tournament_select(relaxed_population)
            
            # Crossover: Exchange subsets
            child = parent1.copy()
            n_exchange = np.random.randint(10, 30)
            idx1 = np.random.choice(N, n_exchange, replace=False)
            idx2 = np.random.choice(N, n_exchange, replace=False)
            child[idx1] = parent2[idx2]
            
            # Mutation
            mutation_scale = 0.3 + 0.5 * (gen / max_generations) # Decreasing mutation
            child += np.random.randn(N, 3) * mutation_scale
            
            # Relax
            relaxed_child = _relax(child)
            if _valid(relaxed_child):
                new_population.append(relaxed_child)
                new_energies.append(_lj_energy(relaxed_child))
            else:
                new_population.append(child)
                new_energies.append(_lj_energy(child))
                
        relaxed_population = new_population
        relaxed_energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(relaxed_energies)
        if relaxed_energies[gen_best_idx] < best_energy:
            best_energy = relaxed_energies[gen_best_idx]
            best_pos = relaxed_population[gen_best_idx].copy()
            
    # Final relaxation of the best
    if time.time() - START_TIME < TIME_LIMIT:
        best_pos = _relax(best_pos)
        if not _valid(best_pos):
             # Fallback
             best_pos = relaxed_population[np.argmin(relaxed_energies)]
             
    return best_pos
# EVOLVE-BLOCK-END
