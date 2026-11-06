"""
Strategy: Genetic Algorithm with fast approximate band-structure evaluation.
- Symmetry constraint: 32x32 cell, invariant under x->-x, y->-y, x<->y.
- Genome: 16x16 upper-right quadrant (including diagonals/edges as needed for symmetry).
- Fitness: Approximate plane-wave method (smaller basis) to evaluate gap ratio quickly.
- Search: GA with mutation and crossover to explore complex patterns.
"""

import time
import random
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # Symmetry: cell[i][j] == cell[size-1-i][j] == cell[i][size-1-j] == cell[j][i]
    # We only need to define the top-right quadrant.
    # Indices 0..15 in the quadrant map to the full cell.
    # Let's define a helper to expand a 16x16 quadrant to 32x32.
    
    def expand_quadrant(q):
        # q is 16x16
        # Top-right quadrant of 32x32 corresponds to indices:
        # i in [0, 15], j in [16, 31] ? No.
        # Symmetry about center.
        # For N=32, center is between 15 and 16.
        # cell[i][j] == cell[N-1-i][N-1-j] (180 rot)
        # cell[i][j] == cell[N-1-i][j] (y flip)
        # cell[i][j] == cell[i][N-1-j] (x flip)
        # cell[i][j] == cell[j][i] (transpose)
        
        # The fundamental domain is i in [0, 15], j in [i, 15]? No, j goes to N-1.
        # Actually, for N=32, the symmetric part is i in [0, 15] and j in [0, 15] is NOT enough 
        # because j can go up to 31. 
        # Wait, cell[i][j] == cell[i][N-1-j]. So column j is determined by column N-1-j.
        # The unique columns are 0..15. The unique rows are 0..15.
        # But we also have transpose symmetry.
        # So cell[i][j] for i<=15, j<=15 determines the rest?
        # Let's check: 
        # cell[i][j] (i<=15, j<=15)
        # cell[i][N-1-j] = cell[i][j] (x flip) -> determines j in [16,31] for i in [0,15]
        # cell[N-1-i][j] = cell[i][j] (y flip) -> determines i in [16,31] for j in [0,15]
        # cell[N-1-i][N-1-j] = cell[i][j] (both) -> determines i in [16,31], j in [16,31]
        # Transpose: cell[j][i] = cell[i][j]. 
        # If i<=15 and j<=15, transpose is within the quadrant.
        # So the unique data is in the 16x16 top-left quadrant?
        # Let's verify symmetry conditions on the 16x16 block Q (i,j in 0..15).
        # The full cell C is built from Q.
        # C[i][j] for i<16, j<16 is Q[i][j].
        # C[i][j] for i<16, j>=16 is Q[i][N-1-j].
        # C[i][j] for i>=16, j<16 is Q[N-1-i][j].
        # C[i][j] for i>=16, j>=16 is Q[N-1-i][N-1-j].
        
        # Does this satisfy transpose?
        # C[i][j] should equal C[j][i].
        # Case 1: i<16, j<16. C[i][j]=Q[i][j], C[j][i]=Q[j][i]. Requires Q symmetric.
        # Case 2: i<16, j>=16. C[i][j]=Q[i][N-1-j]. C[j][i] (j>=16, i<16) = Q[N-1-j][i].
        # Requires Q[i][N-1-j] == Q[N-1-j][i].
        # Let j' = N-1-j. Since j>=16, j' <= 15.
        # So requires Q[i][j'] == Q[j'][i]. This is the same as Q being symmetric.
        # So if Q is a symmetric 16x16 matrix, the expanded 32x32 cell has all 4 symmetries.
        
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = q[i][j]
                cell[i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][j] = val
                cell[size-1-i][size-1-j] = val
        return cell

    # Approximate Band Structure Evaluator
    # Using a smaller plane-wave basis for speed.
    # The exact evaluator uses 289 PWs (17x17). We can use 9x9=81 or 11x11=121 for speed.
    # Let's use 11x11=121 for a balance.
    N_pw = 11
    # G vectors: (m, n) for m, n in -5..5
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(m, n) for m in G_vals for n in G_vals], dtype=float)
    num_pw = len(G_vectors)
    
    # Precompute G^2
    G2 = np.sum(G_vectors**2, axis=1)
    
    # Fourier coefficients of 1/eps
    # 1/eps(x) = 1/eps_high if dielectric, 1/eps_low if air.
    # 1/eps(x) = sum_{G} (1/eps)_G e^{i G . x}
    # (1/eps)_G = (1/Area) * Integral (1/eps(x)) e^{-i G . x} dx
    
    def compute_fourier_1over_eps(cell):
        # cell is 32x32, 1=dielectric, 0=air
        # Normalize coordinates to [0, 2pi] or [-pi, pi]?
        # Standard: x in [0, 1] or [0, a]. Let's use [0, 1] for pixel indices.
        # G vectors are in units of 2pi/L. If L=1, G = 2pi * (m, n).
        # The integral is over the unit cell.
        # (1/eps)_G = sum_{i,j} (1/eps(i,j)) * exp(-i G . (x_i, y_j)) * dx dy
        # dx = dy = 1/32.
        
        coeffs = np.zeros(num_pw, dtype=complex)
        # Pixel centers
        # i=0 -> x=0.5/32, i=31 -> x=31.5/32
        x_coords = (np.arange(size) + 0.5) / size
        y_coords = x_coords
        
        # Vectorize for speed
        # Create meshgrid of pixel positions
        X, Y = np.meshgrid(x_coords, x_coords)
        
        # 1/eps values
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        area = 1.0 / (size * size)
        
        for k, G in enumerate(G_vectors):
            # G is (m, n). The phase is exp(-i 2pi (m x + n y))
            phase = -2 * np.pi * (G[0] * X + G[1] * Y)
            integral = np.sum(eps_inv * np.exp(phase)) * area
            coeffs[k] = integral
        return coeffs

    def solve_bands(eps_inv_coeffs, k_point):
        # k_point is (kx, ky) in units of 2pi/L?
        # The evaluator uses wave vectors. Let's assume k is in [0, 1] range mapped to BZ.
        # Standard plane wave expansion: E(r) = sum_G c_G exp(i (k+G) . r)
        # Hamiltonian: H_{G', G} = (k+G') . (k+G) * (1/eps)_{G'-G}
        # Wait, for TM, the operator is - (1/eps) * (1/kz^2) * (kx^2 + ky^2 + ...)?
        # Simplified TM band structure equation:
        # sum_G' (k+G) . (k+G') (1/eps)_{G-G'} c_{G'} = omega^2 c_G
        # Actually, the standard form is:
        # sum_G' (k+G) . (k+G') (1/eps)_{G-G'} c_{G'} = (omega/c)^2 c_G
        # Note: (1/eps)_{G-G'} is the Fourier coefficient of 1/eps at G-G'.
        
        # Construct Matrix H
        H = np.zeros((num_pw, num_pw), dtype=complex)
        
        # k is in reciprocal lattice units? 
        # Let's assume k is given in units of 2pi/a.
        # k_vector = (kx, ky) * 2pi.
        # G_vector = (m, n) * 2pi.
        # Dot product (k+G).(k+G') = 4pi^2 (kx+m)(kx+m') + 4pi^2 (ky+n)(ky+n')
        # We can factor out 4pi^2.
        
        kx, ky = k_point
        
        for i in range(num_pw):
            Gi = G_vectors[i]
            ki = (kx + Gi[0], ky + Gi[1])
            for j in range(num_pw):
                Gj = G_vectors[j]
                kj = (kx + Gj[0], ky + Gj[1])
                dot = np.dot(ki, kj)
                # Fourier coeff index: G_i - G_j
                # We need to find index of (Gi - Gj) in G_vectors
                # Since G_vectors is symmetric, we can precompute a map.
                # For now, linear search or precomputed dict.
                # Let's precompute a dict for speed.
                diff = Gi - Gj
                # Find index of diff in G_vectors
                # Since N_pw is small, we can do it, but let's optimize.
                # Actually, the coefficients are periodic.
                # Let's just use a precomputed index map.
                
                H[i, j] = dot * eps_inv_coeffs[diff_idx[diff]]
        
        # Eigenvalues
        # H is Hermitian
        eigvals, eigvecs = np.linalg.eigh(H)
        # omega^2 is proportional to eigvals.
        # We only need the order, so we can return sqrt(eigvals) or just eigvals.
        # The gap ratio depends on frequency, so sqrt.
        return np.sqrt(eigvals)

    # Precompute diff indices
    diff_idx = {}
    for i in range(num_pw):
        for j in range(num_pw):
            diff = G_vectors[i] - G_vectors[j]
            # Find j such that G_vectors[j] == diff
            # Since G_vectors is generated from -N//2 to N//2, 
            # and diff is in -N..N, it might be outside the basis?
            # If the basis is truncated, (1/eps)_{G-G'} for large G-G' is approx 0 or wrapped?
            # In a truncated basis, we usually set coeffs to 0 if G-G' is outside the basis.
            # Let's check if diff is in G_vectors.
            found = False
            for k in range(num_pw):
                if np.allclose(G_vectors[k], diff):
                    diff_idx[diff] = k
                    found = True
                    break
            if not found:
                diff_idx[diff] = -1 # Indicates zero

    def evaluate_fitness(cell):
        # Compute Fourier coeffs
        coeffs = compute_fourier_1over_eps(cell)
        
        # Sample k-points in irreducible BZ
        # Irreducible BZ for square lattice: 0 <= kx <= ky <= 1/2
        # Or 0 <= kx <= 1/2, 0 <= ky <= kx?
        # Standard: Triangle with vertices (0,0), (0.5, 0), (0.5, 0.5).
        # Let's sample a few points.
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.25),
            (0.25, 0.0),
            (0.5, 0.25)
        ]
        
        max_omega4 = 0
        min_omega5 = float('inf')
        
        for k in k_points:
            # Solve bands
            # This is the slow part.
            # We can cache or optimize.
            # For now, direct solve.
            
            # Build H
            H = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                Gi = G_vectors[i]
                ki = (k[0] + Gi[0], k[1] + Gi[1])
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    kj = (k[0] + Gj[0], k[1] + Gj[1])
                    dot = np.dot(ki, kj)
                    diff = Gi - Gj
                    idx = diff_idx.get(diff, -1)
                    if idx != -1:
                        H[i, j] = dot * coeffs[idx]
            
            eigvals = np.linalg.eigvalsh(H)
            # eigvals are omega^2
            omegas = np.sqrt(eigvals)
            
            # We need bands 4 and 5 (1-indexed? "fourth and fifth TM bands")
            # Usually bands are 1-indexed in physics. 
            # In 0-indexed array: index 3 and 4.
            # But there might be degeneracies. 
            # The evaluator uses "bands 4 and 5". 
            # Let's assume the 4th lowest frequency is omega_4 and 5th is omega_5.
            
            if len(omegas) >= 5:
                # Find the 4th and 5th lowest frequencies
                # Sort is done by eigvalsh
                o4 = omegas[3]
                o5 = omegas[4]
                
                # Update global max of o4 and min of o5
                if o4 > max_omega4:
                    max_omega4 = o4
                if o5 < min_omega5:
                    min_omega5 = o5
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap == 0:
            return 0
        ratio = 2 * gap / midgap
        return ratio

    # GA Setup
    pop_size = 20
    num_generations = 50
    mutation_rate = 0.1
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random symmetric 16x16
        q = np.random.randint(0, 2, (16, 16))
        # Symmetrize q
        q = (q + q.T) // 2
        population.append(q)
    
    best_q = None
    best_score = -float('inf')
    
    # Evaluate initial population
    for q in population:
        cell = expand_quadrant(q)
        score = evaluate_fitness(cell)
        if score > best_score:
            best_score = score
            best_q = q.copy()
    
    # Check time
    if time.time() > deadline:
        return expand_quadrant(best_q).tolist()
        
    # GA Loop
    for gen in range(num_generations):
        # Select parents
        # Simple tournament selection
        new_population = []
        
        # Elitism: keep best
        new_population.append(best_q.copy())
        
        while len(new_population) < pop_size:
            # Select two parents
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            # Uniform crossover on the 16x16 grid
            child = np.zeros((16, 16), dtype=int)
            for i in range(16):
                for j in range(16):
                    if random.random() < 0.5:
                        child[i][j] = p1[i][j]
                    else:
                        child[i][j] = p2[i][j]
            
            # Symmetrize
            child = (child + child.T) // 2
            
            # Mutation
            for i in range(16):
                for j in range(16):
                    if random.random() < mutation_rate:
                        child[i][j] = 1 - child[i][j]
            
            # Symmetrize again after mutation
            child = (child + child.T) // 2
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        for q in population:
            cell = expand_quadrant(q)
            score = evaluate_fitness(cell)
            if score > best_score:
                best_score = score
                best_q = q.copy()
                
        if time.time() > deadline - 20:
            break
            
    # Final result
    final_cell = expand_quadrant(best_q)
    return final_cell.tolist()

# Helper to ensure return type is nested list of 0/1
# The expand_quadrant returns a numpy array. .tolist() converts to nested list.
# The values are 0 or 1 integers.

# One edge case: if the GA finds a very bad solution, we should have a fallback.
# But the initial random population will likely have some reasonable gap.
# The "seed" score is 5.78. We hope to beat that.

# Let's add a specific known good pattern as part of the initial population to ensure we don't start from scratch.
# The 4-rod pattern from the current program.
def _four_rod_pattern():
    size = 32
    cell = np.zeros((size, size), dtype=int)
    radius = 0.15
    centers = [
        (-0.25, -0.25),
        ( 0.25, -0.25),
        (-0.25,  0.25),
        ( 0.25,  0.25)
    ]
    for i in range(size):
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            inside = False
            for cx, cy in centers:
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    inside = True
                    break
            if inside:
                cell[i][j] = 1
    # Extract quadrant
    q = cell[:16, :16].copy()
    # Ensure symmetry in q
    q = (q + q.T) // 2
    return q

# Modify design to include this in initial population
def design(problem):
    start_time = time.time()
    deadline = start_time + 180.0
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    def expand_quadrant(q):
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = q[i][j]
                cell[i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][j] = val
                cell[size-1-i][size-1-j] = val
        return cell

    N_pw = 11
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(m, n) for m in G_vals for n in G_vals], dtype=float)
    num_pw = len(G_vectors)
    
    diff_idx = {}
    for i in range(num_pw):
        for j in range(num_pw):
            diff = G_vectors[i] - G_vectors[j]
            found = False
            for k in range(num_pw):
                if np.allclose(G_vectors[k], diff):
                    diff_idx[diff] = k
                    found = True
                    break
            if not found:
                diff_idx[diff] = -1

    def compute_fourier_1over_eps(cell):
        coeffs = np.zeros(num_pw, dtype=complex)
        x_coords = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x_coords, x_coords)
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        area = 1.0 / (size * size)
        
        for k, G in enumerate(G_vectors):
            phase = -2 * np.pi * (G[0] * X + G[1] * Y)
            integral = np.sum(eps_inv * np.exp(phase)) * area
            coeffs[k] = integral
        return coeffs

    def evaluate_fitness(cell):
        coeffs = compute_fourier_1over_eps(cell)
        k_points = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
            (0.25, 0.25), (0.25, 0.0), (0.5, 0.25)
        ]
        max_omega4 = 0
        min_omega5 = float('inf')
        
        for k in k_points:
            H = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                Gi = G_vectors[i]
                ki = (k[0] + Gi[0], k[1] + Gi[1])
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    kj = (k[0] + Gj[0], k[1] + Gj[1])
                    dot = np.dot(ki, kj)
                    diff = Gi - Gj
                    idx = diff_idx.get(diff, -1)
                    if idx != -1:
                        H[i, j] = dot * coeffs[idx]
            eigvals = np.linalg.eigvalsh(H)
            omegas = np.sqrt(eigvals)
            if len(omegas) >= 5:
                o4 = omegas[3]
                o5 = omegas[4]
                if o4 > max_omega4:
                    max_omega4 = o4
                if o5 < min_omega5:
                    min_omega5 = o5
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap == 0:
            return 0
        return 2 * gap / midgap

    # Init Population
    pop_size = 20
    population = []
    
    # Add 4-rod pattern
    population.append(_four_rod_pattern())
    
    # Add randoms
    for _ in range(pop_size - 1):
        q = np.random.randint(0, 2, (16, 16))
        q = (q + q.T) // 2
        population.append(q)
        
    best_q = None
    best_score = -float('inf')
    
    for q in population:
        cell = expand_quadrant(q)
        score = evaluate_fitness(cell)
        if score > best_score:
            best_score = score
            best_q = q.copy()
            
    if time.time() > deadline:
        return expand_quadrant(best_q).tolist()
        
    mutation_rate = 0.1
    num_generations = 50
    
    for gen in range(num_generations):
        new_population = [best_q.copy()]
        
        while len(new_population) < pop_size:
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            child = np.zeros((16, 16), dtype=int)
            for i in range(16):
                for j in range(16):
                    if random.random() < 0.5:
                        child[i][j] = p1[i][j]
                    else:
                        child[i][j] = p2[i][j]
            
            child = (child + child.T) // 2
            
            for i in range(16):
                for j in range(16):
                    if random.random() < mutation_rate:
                        child[i][j] = 1 - child[i][j]
            
            child = (child + child.T) // 2
            new_population.append(child)
            
        population = new_population
        for q in population:
            cell = expand_quadrant(q)
            score = evaluate_fitness(cell)
            if score > best_score:
                best_score = score
                best_q = q.copy()
                
        if time.time() > deadline - 20:
            break
            
    return expand_quadrant(best_q).tolist()
