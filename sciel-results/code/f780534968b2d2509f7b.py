"""Lennard-Jones 55-atom cluster global minimum via genetic algorithm."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 55


def lj_energy(pos):
    """Compute total Lennard-Jones energy for a set of positions."""
    r = pdist(pos)
    mask = r > 1e-12
    r = r[mask]
    r6 = r ** 6
    r12 = r6 * r6
    return 4.0 * np.sum(1.0 / r12 - 1.0 / r6)


def lj_forces(pos):
    """Compute forces (negative gradient) for L-BFGS. Returns -grad so minimize works."""
    n = len(pos)
    forces = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[j] - pos[i]
            r2 = np.dot(d, d)
            if r2 < 1e-12:
                continue
            r6 = r2 ** 3
            r12 = r6 * r6
            # Force on i from j: 4*(12/r^13 - 6/r^7) * (pos_i - pos_j)/r
            # = 4*(12/r^12 - 6/r^6) * d / r  ... let's be careful
            # E = 4*(r^-12 - r^-6), dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # Force on i = -dE/dr * (pos_i - pos_j)/r = -4*(-12/r^13 + 6/r^7) * (pos_i-pos_j)/r
            # = 4*(12/r^13 - 6/r^7) * (pos_i - pos_j)/r
            # = 4*(12/r^12 - 6/r^6) * (pos_i - pos_j) / r^2
            # Since d = pos_j - pos_i, pos_i - pos_j = -d
            # Force on i = 4*(12/r^12 - 6/r^6) * (-d) / r2 = -4*(12/r^12 - 6/r^6) * d / r2
            coeff = 4.0 * (12.0 / r12 - 6.0 / r6) / r2
            forces[i] -= coeff * d
            forces[j] += coeff * d
    return forces


def energy_and_grad(pos):
    """Return (energy, gradient) for scipy minimize."""
    e = lj_energy(pos)
    f = lj_forces(pos)
    return e, -f  # gradient = -force


def relax(pos):
    """L-BFGS relaxation."""
    try:
        res = minimize(energy_and_grad, pos, method='L-BFGS-B',
                       jac=True, options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8})
        return res.x, res.fun
    except Exception:
        return pos, lj_energy(pos)


def enforce_min_distance(pos, min_dist=0.5):
    """Nudge atoms that are too close apart."""
    pos = pos.copy()
    for _ in range(100):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[j] - pos[i]
                r = np.linalg.norm(d)
                if r < min_dist:
                    if r < 1e-10:
                        d = np.random.randn(3) * 0.01
                        r = np.linalg.norm(d)
                    # Push apart
                    push = (min_dist - r + 0.01) / r * d * 0.5
                    pos[i] -= push
                    pos[j] += push
                    moved = True
        if not moved:
            break
    return pos


def mackay_icosahedron():
    """Generate the Mackay icosahedron for 55 atoms.
    Shell 0: 1 atom at center
    Shell 1: 12 atoms at icosahedral vertices, radius r1
    Shell 2: 42 atoms at icosahedral edges and face centers, radius r2
    """
    # Icosahedral vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # normalize to unit sphere

    # Shell 1: 12 vertices at radius ~1.05 (tuned)
    r1 = 1.05
    shell1 = verts * r1

    # Shell 2: 42 atoms
    # 30 edge midpoints (edges of icosahedron) and 12 face centers
    # Icosahedron has 30 edges and 20 faces
    # For Mackay icosahedron, shell 2 has 42 atoms: 30 on edges + 12 on faces? 
    # Actually Mackay icosahedron M(2) has 1 + 12 + 42 = 55 atoms.
    # The 42 are arranged as: 30 edge positions and 12 face positions? 
    # More precisely: the 42 atoms in shell 2 are at the midpoints of the 30 edges 
    # (projected to shell 2 radius) and the centers of the 20 faces... that's 50.
    # Actually for Mackay icosahedron, shell 2 consists of atoms at:
    # - 30 positions corresponding to edge midpoints of the icosahedron (but at shell 2 radius)
    # - 12 positions corresponding to... 
    # Let me use a simpler approach: generate all icosahedral symmetry positions.
    
    # Alternative: use known coordinates or generate via symmetry operations.
    # Icosahedral group has 60 operations. The 12 vertices generate orbit of 12.
    # For shell 2, we can use the 30 edge midpoints and 12 "face center" type positions.
    
    # Let me construct shell 2 differently:
    # The 42 atoms in the second shell of Mackay icosahedron are located at:
    # 30 atoms at edge midpoints (normalized to shell 2 radius) 
    # 12 atoms at positions corresponding to... 
    
    # Actually, let me just use a simpler construction:
    # Shell 2 atoms: take the 30 edge midpoints of the icosahedron and 12 face centers,
    # but that gives 42? 30 + 12 = 42. Yes!
    
    # Find all edges of icosahedron (pairs of vertices at minimum distance)
    dists = pdist(verts)
    min_dist = np.min(dists)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_dist) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    edge_mids /= np.linalg.norm(edge_mids[0])  # normalize to unit sphere

    # 12 face centers: pick 12 of the 20 faces
    # Faces are triangles of mutually connected vertices
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                dij = np.linalg.norm(verts[i] - verts[j])
                djk = np.linalg.norm(verts[j] - verts[k])
                dik = np.linalg.norm(verts[i] - verts[k])
                if (abs(dij - min_dist) < 1e-10 and abs(djk - min_dist) < 1e-10 and abs(dik - min_dist) < 1e-10):
                    faces.append((i, j, k))
    
    # We have 20 faces; pick 12 (e.g., first 12)
    face_centers = []
    for idx, (i, j, k) in enumerate(faces[:12]):
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    face_centers /= np.linalg.norm(face_centers[0])  # normalize

    r2 = 1.9  # shell 2 radius
    shell2 = np.vstack([edge_mids * r2, face_centers * r2])

    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])

    return np.vstack([center, shell1, shell2])


def random_cubic():
    """Random cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    np.random.shuffle(points)
    points = points[:N]
    return np.array(points, dtype=float) * 1.12


def random_gaussian():
    """Random Gaussian distribution of atoms."""
    return np.random.randn(N, 3) * 2.0


def crossover(parent1, parent2):
    """Exchange a subset of atoms between parents."""
    n_swap = np.random.randint(5, 25)
    idx1 = np.random.choice(N, n_swap, replace=False)
    idx2 = np.random.choice(N, n_swap, replace=False)
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    child1[idx1] = parent2[idx2]
    child2[idx2] = parent1[idx1]
    
    return child1, child2


def mutate(pos, sigma=0.3):
    """Apply small Gaussian displacements."""
    return pos + np.random.randn(N, 3) * sigma


def genetic_algorithm(pop_size=20, generations=30, time_limit=110):
    """Genetic algorithm to find low-energy LJ 55-atom cluster."""
    import time
    start_time = time.time()
    
    # Initialize population
    population = []
    energies = []
    
    # Seed with Mackay icosahedron
    mackay = mackay_icosahedron()
    mackay, e_mackay = relax(mackay)
    population.append(mackay)
    energies.append(e_mackay)
    
    # Add other initial structures
    for _ in range(pop_size - 1):
        if np.random.rand() < 0.3:
            pos = random_cubic()
        else:
            pos = random_gaussian()
        pos = enforce_min_distance(pos)
        pos, e = relax(pos)
        population.append(pos)
        energies.append(e)
    
    best_pos = population[0]
    best_energy = min(energies)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        new_population = []
        new_energies = []
        
        # Keep best
        best_idx = np.argmin(energies)
        new_population.append(population[best_idx])
        new_energies.append(energies[best_idx])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
            
            # Select parents (tournament selection)
            i1 = np.random.randint(len(population))
            i2 = np.random.randint(len(population))
            i3 = np.random.randint(len(population))
            
            # Tournament: pick best of 3
            candidates = [i1, i2, i3]
            parent_idx1 = min(candidates, key=lambda x: energies[x])
            candidates2 = [np.random.randint(len(population)) for _ in range(3)]
            parent_idx2 = min(candidates2, key=lambda x: energies[x])
            
            p1 = population[parent_idx1]
            p2 = population[parent_idx2]
            
            # Crossover
            c1, c2 = crossover(p1, p2)
            
            # Mutation
            c1 = mutate(c1)
            c2 = mutate(c2)
            
            # Enforce min distance and relax
            c1 = enforce_min_distance(c1)
            c1, e1 = relax(c1)
            new_population.append(c1)
            new_energies.append(e1)
            
            c2 = enforce_min_distance(c2)
            c2, e2 = relax(c2)
            new_population.append(c2)
            new_energies.append(e2)
        
        population = new_population[:pop_size]
        energies = new_energies[:pop_size]
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy:
            best_energy = energies[gen_best_idx]
            best_pos = population[gen_best_idx]
    
    return best_pos, best_energy


def construct():
    """Construct initial positions for 55-atom LJ cluster."""
    # Try genetic algorithm with time limit
    try:
        pos, energy = genetic_algorithm(pop_size=15, generations=25, time_limit=110)
        pos = enforce_min_distance(pos, min_dist=0.5)
        return pos
    except Exception:
        # Fallback to Mackay icosahedron
        mackay = mackay_icosahedron()
        mackay, _ = relax(mackay)
        return enforce_min_distance(mackay, min_dist=0.5)
