# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # grad_i = sum_{j != i} 4 * (-12 * r_ij^(-14) + 6 * r_ij^(-8)) * (pos_i - pos_j)
    # In terms of r2: r^(-14) = r2^(-7), r^(-8) = r2^(-4)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def get_mackay_directions():
    """Returns directions for the Mackay icosahedron shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron (distance between adjacent vertices)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    return verts, midpoints

def make_mackay_55():
    """
    Create a 55-atom cluster based on Mackay icosahedron geometry.
    Structure: 
    - 1 atom at center
    - 12 atoms at icosahedron vertices (radius r1)
    - 30 atoms at edge midpoints (radius r2)
    - 12 atoms at icosahedron vertices (radius r3)
    
    We optimize r1, r2, r3 to minimize energy of this symmetric configuration.
    """
    verts, mids = get_mackay_directions()
    
    def symmetric_energy(params):
        r1, r2, r3 = params
        if r1 < 0.5 or r2 < 0.5 or r3 < 0.5:
            return 1e10
        pos = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            verts * r1,
            mids * r2,
            verts * r3
        ])
        return lj_energy(pos)

    # Initial guess based on literature values for LJ 55
    # Typical radii: r1 ~ 1.0, r2 ~ 1.8, r3 ~ 2.6
    x0 = np.array([1.0, 1.8, 2.6])
    
    # Optimize radii using Nelder-Mead
    res = minimize(symmetric_energy, x0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    r1, r2, r3 = res.x
    r1 = max(0.5, r1)
    r2 = max(0.5, r2)
    r3 = max(0.5, r3)
    
    pos = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        verts * r1,
        mids * r2,
        verts * r3
    ])
    return pos

def make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.0
    return pos

def make_perturbed_mackay(seed, scale=0.1):
    """Generate a perturbed Mackay icosahedron."""
    base = make_mackay_55()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def relax(pos, maxiter=500, gtol=1e-6):
    """Relax using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with optimized Mackay
    best_pos = make_mackay_55()
    best_e = lj_energy(best_pos)
    
    # Relax the initial Mackay structure
    if time.time() < deadline - 30:
        pos_relaxed, e_relaxed = relax(best_pos, maxiter=500, gtol=1e-6)
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
    
    # Differential evolution parameters
    pop_size = 12
    mutation_scale = 0.5
    crossover_rate = 0.8
    max_generations = 8
    
    # Initialize population with diverse Mackay-based structures
    rng = np.random.RandomState(42)
    population = [best_pos.copy()]
    population_energies = [best_e]
    
    # Add diverse initial members
    for i in range(pop_size - 1):
        if time.time() > deadline - 25:
            break
        scale = 0.05 + 0.15 * (i / max(1, pop_size - 2))
        pos = make_perturbed_mackay(i, scale=scale)
        pos_relaxed, e_relaxed = relax(pos, maxiter=300, gtol=1e-5)
        population.append(pos_relaxed.copy())
        population_energies.append(e_relaxed)
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
    
    # Sort population by energy
    indices = np.argsort(population_energies)
    population = [population[i] for i in indices]
    population_energies = [population_energies[i] for i in indices]
    
    # Differential evolution loop
    for gen in range(max_generations):
        if time.time() > deadline - 20:
            break
        
        # Generate offspring
        offspring = []
        offspring_energies = []
        
        for i in range(pop_size):
            if time.time() > deadline - 15:
                break
            
            # Select three distinct individuals for mutation
            mask = np.ones(pop_size, dtype=bool)
            mask[i] = False
            candidates_idx = np.where(mask)[0]
            a, b, c = rng.choice(candidates_idx, size=3, replace=False)
            
            # Mutation: base + scale * (a - b)
            mutant = population[a] + mutation_scale * (population[b] - population[c])
            
            # Ensure minimum distance constraint is satisfied
            # Add small random jitter to break symmetries
            mutant += rng.randn(N, 3) * 0.01
            
            # Relax the mutant
            pos_relaxed, e_relaxed = relax(mutant, maxiter=400, gtol=1e-6)
            offspring.append(pos_relaxed.copy())
            offspring_energies.append(e_relaxed)
            
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
        
        if len(offspring) == 0:
            break
        
        # Selection: keep best from parent + offspring
        combined = population + offspring
        combined_e = population_energies + offspring_energies
        
        # Sort and keep best pop_size individuals
        combined_indices = np.argsort(combined_e)
        new_population = [combined[i] for i in combined_indices[:pop_size]]
        new_population_energies = [combined_e[i] for i in combined_indices[:pop_size]]
        
        population = new_population
        population_energies = new_population_energies
        
        # Update best
        if population_energies[0] < best_e:
            best_e = population_energies[0]
            best_pos = population[0].copy()
        
        # Early stopping if no improvement in last generation
        if len(population_energies) > 0 and population_energies[0] >= best_e:
            # Check if we've converged
            if gen > 2 and abs(population_energies[0] - best_e) < 1e-6:
                break
    
    # Final high-precision polish with multiple passes
    if time.time() < deadline - 10:
        # First pass: moderate tolerance
        pos_final, e_final = relax(best_pos, maxiter=2000, gtol=1e-8)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
        
        # Second pass: very tight tolerance if time permits
        if time.time() < deadline - 3:
            pos_final2, e_final2 = relax(best_pos, maxiter=5000, gtol=1e-10)
            if e_final2 < best_e:
                best_e = e_final2
                best_pos = pos_final2
    
    return best_pos

# EVOLVE-BLOCK-END
