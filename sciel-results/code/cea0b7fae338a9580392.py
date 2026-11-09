"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _make_symmetric_cell_from_ur_quadrant(ur, size=32):
    """Expand a 16x16 upper-right quadrant to a full 32x32 symmetric cell.
    The full cell has 4-fold rotational symmetry and mirror symmetry.
    ur[i,j] for i in [0,16), j in [0,16) maps to 4 positions.
    """
    cell = np.zeros((size, size), dtype=np.int32)
    half = size // 2  # 16
    
    for i in range(half):
        for j in range(half):
            v = ur[i, j]
            # 4-fold: (i,j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
            cell[i, j] = v
            cell[i, size-1-j] = v
            cell[size-1-i, j] = v
            cell[size-1-i, size-1-j] = v
    
    return cell

def _fast_pw_solver(cell, n_pw=13, n_kpts=8):
    """Fast plane-wave solver for TM bands.
    cell: 32x32 array of 0/1
    Returns: (max_omega4, min_omega5) over sampled k-points
    """
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute 1/eps
    inv_eps = np.where(cell > 0.5, 1.0/12.0, 1.0)
    
    # Plane wave indices
    n = n_pw
    indices = list(range(-n//2, n//2 + 1))
    n_pw_total = len(indices) ** 2
    
    # Precompute Fourier coefficients of 1/eps using FFT
    # inv_eps_G = (1/N^2) * sum_{x,y} inv_eps(x,y) * exp(-i*G*x)
    # For G = 2*pi*(m,n), this is the 2D DFT
    # We can use numpy FFT
    fft_2d = np.fft.fft2(inv_eps)
    fft_2d /= (size * size)
    
    # Map FFT indices to G indices
    # FFT output: index [i,j] corresponds to frequency (i if i < size/2 else i-size, j if j < size/2 else j-size)
    # We need frequencies in range [-n//2, n//2]
    fourier_coeffs = {}
    for m in indices:
        for n_idx in indices:
            # FFT index for frequency m: m % size
            mi = m % size
            ni = n_idx % size
            fourier_coeffs[(m, n_idx)] = fft_2d[mi, ni]
    
    # K-points in irreducible BZ
    # Square lattice BZ: [-pi, pi] x [-pi, pi]
    # Irreducible wedge: 0 <= kx <= ky <= pi
    # Key high-symmetry points and paths
    k_points = []
    # Gamma (0,0) to X (pi/2, pi/2)
    for t in np.linspace(0, 1, 3):
        k_points.append((t * np.pi / 2, t * np.pi / 2))
    # X (pi/2, pi/2) to M (pi, 0)
    for t in np.linspace(0, 1, 3):
        k_points.append((np.pi/2 + t * np.pi/2, np.pi/2 - t * np.pi/2))
    # M (pi, 0) to Gamma (0, 0)
    for t in np.linspace(0, 1, 3):
        k_points.append((np.pi - t * np.pi, 0))
    # Add interior point
    k_points.append((np.pi/3, np.pi/3))
    k_points.append((np.pi/4, np.pi/2))
    
    # Deduplicate
    seen = set()
    unique_k = []
    for kx, ky in k_points:
        key = (round(kx, 4), round(ky, 4))
        if key not in seen:
            seen.add(key)
            unique_k.append((kx, ky))
    k_points = unique_k
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    # Precompute G vectors
    G_list = [(m, n_idx) for m in indices for n_idx in indices]
    
    for kx, ky in k_points:
        # Build eigenvalue problem for TM mode
        # A * c = omega^2 * B * c
        # A is diagonal: A[i,i] = |k - G_i|^2
        # B[i,j] = inv_eps(G_i - G_j)
        
        A_diag = np.zeros(n_pw_total, dtype=np.float64)
        for idx, (mi, ni) in enumerate(G_list):
            kx_G = kx - 2 * np.pi * mi
            ky_G = ky - 2 * np.pi * ni
            A_diag[idx] = kx_G**2 + ky_G**2
        
        B = np.zeros((n_pw_total, n_pw_total), dtype=np.complex128)
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                dm = mi - mj
                dn = ni - nj
                B[i, j] = fourier_coeffs.get((dm, dn), 0.0)
        
        # Solve generalized eigenvalue problem using scipy
        try:
            from scipy.linalg import eig
            A_full = np.diag(A_diag).astype(np.complex128)
            vals, vecs = eig(A_full, B)
            vals_real = np.real(vals)
            vals_pos = vals_real[vals_real > 0]
            vals_pos.sort()
            
            if len(vals_pos) >= 5:
                omega4 = np.sqrt(vals_pos[3])
                omega5 = np.sqrt(vals_pos[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except Exception:
            continue
    
    return max_omega4, min_omega5

def _score_cell(cell):
    """Compute the gap-midgap ratio for a given cell."""
    try:
        max_omega4, min_omega5 = _fast_pw_solver(cell, n_pw=13, n_kpts=8)
        if min_omega5 > max_omega4:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return gap / midgap
            else:
                return 0.0
        else:
            return -1.0
    except Exception:
        return -1.0

def _random_ur_quadrant(size=32, density=0.3):
    """Generate a random upper-right quadrant with given dielectric density."""
    half = size // 2
    ur = np.zeros((half, half), dtype=np.int32)
    for i in range(half):
        for j in range(half):
            ur[i, j] = 1 if random.random() < density else 0
    return ur

def _crossover(ur1, ur2, size=32):
    """Uniform crossover of two UR quadrants."""
    half = size // 2
    child = np.zeros((half, half), dtype=np.int32)
    for i in range(half):
        for j in range(half):
            if random.random() < 0.5:
                child[i, j] = ur1[i, j]
            else:
                child[i, j] = ur2[i, j]
    return child

def _mutate(ur, size=32, mutation_rate=0.05):
    """Bit-flip mutation on UR quadrant."""
    half = size // 2
    child = ur.copy()
    for i in range(half):
        for j in range(half):
            if random.random() < mutation_rate:
                child[i, j] = 1 - child[i, j]
    return child

def design(problem):
    """Genetic algorithm search over symmetry-constrained binary cells."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s limit
    
    half = size // 2
    random.seed(42)
    np.random.seed(42)
    
    # Population size
    pop_size = 20
    elite_size = 4
    
    # Initialize population with diverse densities
    population = []
    for i in range(pop_size):
        density = 0.15 + 0.6 * (i / (pop_size - 1)) if pop_size > 1 else 0.3
        ur = _random_ur_quadrant(size, density)
        cell = _make_symmetric_cell_from_ur_quadrant(ur, size)
        score = _score_cell(cell)
        population.append((ur, cell, score))
    
    # Also add some structured initial candidates
    # Concentric ring
    ur_ring = np.zeros((half, half), dtype=np.int32)
    for i in range(half):
        for j in range(half):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                ur_ring[i, j] = 1
    cell_ring = _make_symmetric_cell_from_ur_quadrant(ur_ring, size)
    score_ring = _score_cell(cell_ring)
    population.append((ur_ring, cell_ring, score_ring))
    
    # Cross shape
    ur_cross = np.zeros((half, half), dtype=np.int32)
    for i in range(half):
        for j in range(half):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            if abs(x - 0.25) < 0.08 or abs(y - 0.25) < 0.08:
                ur_cross[i, j] = 1
    cell_cross = _make_symmetric_cell_from_ur_quadrant(ur_cross, size)
    score_cross = _score_cell(cell_cross)
    population.append((ur_cross, cell_cross, score_cross))
    
    # Sort by score
    population.sort(key=lambda x: x[2], reverse=True)
    
    best_ur = population[0][0].copy()
    best_cell = population[0][1].copy()
    best_score = population[0][2]
    
    # Check time
    if time.time() > deadline - 30:
        result = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(1 if best_cell[i, j] > 0.5 else 0)
            result.append(row)
        return result
    
    # Genetic algorithm loop
    gen = 0
    while time.time() < deadline - 30:
        gen += 1
        
        # Selection: tournament selection
        def tournament_select(pop):
            t_size = 3
            candidates = random.sample(pop, min(t_size, len(pop)))
            return max(candidates, key=lambda x: x[2])
        
        # Create new population
        new_population = []
        
        # Elitism: keep top elites
        for i in range(min(elite_size, len(population))):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size + 2:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            if random.random() < 0.5:
                child_ur = _crossover(parent1[0], parent2[0], size)
            else:
                # Pick one parent randomly
                if random.random() < 0.5:
                    child_ur = parent1[0].copy()
                else:
                    child_ur = parent2[0].copy()
            
            child_ur = _mutate(child_ur, size, mutation_rate=0.03)
            child_cell = _make_symmetric_cell_from_ur_quadrant(child_ur, size)
            
            # Time check
            if time.time() > deadline - 15:
                break
            
            child_score = _score_cell(child_cell)
            new_population.append((child_ur, child_cell, child_score))
            
            if child_score > best_score:
                best_score = child_score
                best_ur = child_ur.copy()
                best_cell = child_cell.copy()
        
        # Replace population
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        # Adaptive mutation rate: decrease over generations
        # Already handled by fixed rate; could make it adaptive
        
        # Check if we've found a good score
        if best_score > 0.5:
            # Do local refinement
            break
    
    # Local refinement of best solution
    if time.time() < deadline - 10:
        local_ur = best_ur.copy()
        local_cell = best_cell.copy()
        local_score = best_score
        
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Try flipping a few random pixels
            flipped = local_ur.copy()
            n_flips = random.randint(1, 3)
            for _ in range(n_flips):
                fi = random.randint(0, half - 1)
                fj = random.randint(0, half - 1)
                flipped[fi, fj] = 1 - flipped[fi, fj]
            
            flipped_cell = _make_symmetric_cell_from_ur_quadrant(flipped, size)
            flipped_score = _score_cell(flipped_cell)
            
            if flipped_score > local_score:
                local_ur = flipped
                local_cell = flipped_cell
                local_score = flipped_score
                
                if local_score > best_score:
                    best_score = local_score
                    best_ur = local_ur.copy()
                    best_cell = local_cell.copy()
    
    # Convert to nested list of 0/1
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 if best_cell[i, j] > 0.5 else 0)
        result.append(row)
    
    return result
# EVOLVE-BLOCK-END
