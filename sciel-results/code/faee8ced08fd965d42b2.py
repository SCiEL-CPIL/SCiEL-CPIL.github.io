"""Design a C4v-symmetric photonic-crystal unit cell to maximize the TM band gap between bands 4 and 5."""
import time
import math
import random
import numpy as np

def _enforce_c4v(cell):
    """Enforce C4v symmetry: 180-degree rotation (and by construction transpose) about the center."""
    size = len(cell)
    # Enforce 180-degree rotational symmetry: cell[i][j] == cell[size-1-i][size-1-j]
    for i in range(size // 2):
        for j in range(size // 2):
            # Take the OR or average? Let's use majority or just one. For binary, we can set both to 1 if either is 1 (union) or 0 if both 0.
            # Actually, to be safe and simple, let's just enforce that if one is 1, the other is 1.
            # But we want to preserve the design as much as possible.
            # Let's just set them equal to the value of cell[i][j] for now, and then later enforce transpose.
            # Actually, the simplest way to enforce C4v is to define the fundamental domain (1/8 of the cell) and mirror it.
            pass
    # Better approach: define the fundamental domain (i in [0, size//2), j in [i, size//2)) and mirror.
    # For a 32x32 cell, the center is at (15.5, 15.5).
    # The fundamental domain for C4v is the triangle i <= j and i + j < size.
    # Let's just enforce 180 rotation and transpose.
    # 180 rotation: cell[i][j] = cell[size-1-i][size-1-j]
    for i in range(size):
        for j in range(size):
            ri = size - 1 - i
            rj = size - 1 - j
            # If we encounter a pair, set both to 1 if either is 1 (to maximize dielectric) or 0 if either is 0?
            # Let's just set them equal. We'll pick the value from the "upper-left" part of the rotation pair.
            if (i < ri) or (i == ri and j < rj):
                cell[ri][rj] = cell[i][j]
    # Transpose: cell[i][j] = cell[j][i]
    for i in range(size):
        for j in range(i + 1, size):
            # Set both to 1 if either is 1? Or just copy?
            # Let's copy from the upper triangle to the lower.
            cell[j][i] = cell[i][j]
    return cell

def _compute_gap(cell, problem):
    """
    Compute the gap-midgap ratio using a simplified plane-wave method.
    This is a fast approximation for the search loop.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Convert cell to a 2D array of epsilons
    eps_map = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
    
    # For TM polarization, we solve:
    # sum_k' (G x G')^2 / eps(G-G') * E(k+G') = (omega^2 / c^2) E(k+G)
    # where (G x G')^2 = G^2 + G'^2 - 2 G.G' for TM (scalar product of transverse vectors)
    # Actually, for TM (electric field perpendicular to plane), the eigenvalue problem is:
    # sum_{G'} [ (G+G) . (G'+G) / eps(G-G') ] E(k+G') = (omega^2) E(k+G)
    # Wait, the standard form is:
    # sum_{G'} [ (k+G) . (k+G') / eps(G-G') ] E(k+G') = (omega^2) E(k+G)
    
    # We need the Fourier coefficients of 1/eps.
    # 1/eps(G) = (1/A) sum_{i,j} (1/eps(x_i, y_j)) exp(-i G . r_{ij})
    # For a 32x32 pixel map, we can compute this exactly.
    
    # Let's use a small number of plane waves for speed during search.
    # The evaluator uses 289 plane waves. We can use fewer, say 49 or 81, for a fast estimate.
    # Let's try 49 (7x7) plane waves.
    n_pw = 7  # 7x7 = 49 plane waves
    G_list = []
    for gx in range(-n_pw//2, n_pw//2 + 1):
        for gy in range(-n_pw//2, n_pw//2 + 1):
            G_list.append((gx, gy))
    n_pw_total = len(G_list)
    
    # Compute 1/eps Fourier coefficients
    inv_eps = np.zeros((n_pw_total, n_pw_total), dtype=complex)
    # Precompute pixel positions
    pixel_coords = [(i + 0.5) / size - 0.5 for i in range(size)]
    for idx1, G1 in enumerate(G_list):
        for idx2, G2 in enumerate(G_list):
            dGx = G1[0] - G2[0]
            dGy = G1[1] - G2[1]
            if dGx == 0 and dGy == 0:
                # Average of 1/eps
                inv_eps[idx1, idx2] = np.sum(1.0 / eps_map) / (size * size)
            else:
                # Sum over all pixels
                s = 0.0
                for i in range(size):
                    for j in range(size):
                        x = pixel_coords[i]
                        y = pixel_coords[j]
                        s += (1.0 / eps_map[i, j]) * np.exp(-2j * np.pi * (dGx * x + dGy * y))
                inv_eps[idx1, idx2] = s / (size * size)
    
    # Define the irreducible Brillouin zone wave vectors
    # The square lattice BZ is [-pi, pi] x [-pi, pi] (in units of 2pi/a, so actually [-0.5, 0.5] in reciprocal lattice units)
    # The irreducible zone is the triangle with vertices (0,0), (0.5, 0), (0.5, 0.5)
    # Let's sample 30 points in this triangle.
    k_points = []
    for i in range(6):
        for j in range(6):
            kx = 0.5 * i / 5
            ky = 0.5 * j / 5
            if kx + ky <= 0.5:
                k_points.append((kx, ky))
    # Add some more points to make it 30
    while len(k_points) < 30:
        kx = random.random() * 0.5
        ky = random.random() * 0.5 * (1 - 2 * kx)  # This is wrong, let's just do uniform in the triangle
        # Uniform in triangle: kx in [0, 0.5], ky in [0, 0.5 - kx]
        kx = random.random() * 0.5
        ky = random.random() * (0.5 - kx)
        k_points.append((kx, ky))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        # Build the eigenvalue matrix
        # H[G, G'] = (k+G) . (k+G') * inv_eps(G-G')
        # Note: k is in reciprocal lattice units, so k+G is also in reciprocal lattice units.
        # The dot product is (kx+Gx)(kx'+Gx') + (ky+Gy)(ky'+Gy')
        H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        for i in range(n_pw_total):
            for j in range(n_pw_total):
                kpi = (kx + G_list[i][0], ky + G_list[i][1])
                kpj = (kx + G_list[j][0], ky + G_list[j][1])
                dot = kpi[0] * kpj[0] + kpi[1] * kpj[1]
                H[i, j] = dot * inv_eps[i, j]
        
        # Solve the eigenvalue problem
        # We need the 4th and 5th eigenvalues (1-indexed), so indices 3 and 4 (0-indexed)
        # The eigenvalues are omega^2
        try:
            eigvals = np.linalg.eigvalsh(H)
            # Sort in ascending order
            eigvals = np.sort(eigvals.real)
            # Take the 4th and 5th smallest
            if len(eigvals) >= 5:
                omega4_sq = eigvals[3]
                omega5_sq = eigvals[4]
                omega4 = math.sqrt(max(0, omega4_sq))
                omega5 = math.sqrt(max(0, omega5_sq))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if min_omega5 == float('inf'):
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap == 0:
        return 0.0
    return gap / midgap

def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for the 120s budget
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Initialize with the seed design (centered rod)
    best_cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        best_cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    best_cell = _enforce_c4v(best_cell)
    best_score = _compute_gap(best_cell, problem)
    
    # Genetic algorithm
    pop_size = 20
    mutation_rate = 0.1
    
    # Initialize population
    population = [best_cell]
    for _ in range(pop_size - 1):
        # Random C4v-symmetric cell
        cell = [[0 for _ in range(size)] for _ in range(size)]
        # Fill the fundamental domain
        for i in range(size // 2):
            for j in range(i, size // 2):
                if random.random() < 0.5:
                    cell[i][j] = 1
                    cell[j][i] = 1
                    cell[size-1-i][size-1-j] = 1
                    cell[size-1-j][size-1-i] = 1
        population.append(cell)
    
    # Evaluate initial population
    scores = []
    for cell in population:
        score = _compute_gap(cell, problem)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
    
    # Evolution loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        indexed_pop = sorted(zip(scores, population), key=lambda x: -x[0])
        scores = [s for s, _ in indexed_pop]
        population = [c for _, c in indexed_pop]
        
        # Elitism: keep the top 2
        new_population = [row[:] for row in population[0]]
        new_population.append([row[:] for row in population[1]])
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = random.choice(population[:5])
            p2 = random.choice(population[:5])
            
            # Crossover: uniform crossover
            child = [[0 for _ in range(size)] for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    if random.random() < 0.5:
                        child[i][j] = p1[i][j]
                    else:
                        child[i][j] = p2[i][j]
            
            # Mutation
            for i in range(size):
                for j in range(size):
                    if random.random() < mutation_rate:
                        child[i][j] = 1 - child[i][j]
            
            # Enforce C4v symmetry
            child = _enforce_c4v(child)
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = []
        for cell in population:
            score = _compute_gap(cell, problem)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
        
        # Adapt mutation rate
        if generation % 5 == 0:
            mutation_rate = max(0.01, mutation_rate * 0.9)
    
    return best_cell
