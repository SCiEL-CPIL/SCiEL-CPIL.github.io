import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 200  # 30s safety margin for 230s budget

    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    # Precompute the irreducible BZ path: Gamma -> M -> X -> Gamma
    # 30 points
    k_points = []
    # Gamma to M: (0,0) to (0.5,0)
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    # M to X: (0.5,0) to (0.5,0.5)
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # X to Gamma: (0.5,0.5) to (0,0)
    for t in np.linspace(0.5, 0, 10):
        k_points.append((t, t))
    k_points = np.array(k_points)

    # Fast PWE Solver using N=5 plane waves (5x5 grid of G-vectors)
    # G = (nx, ny) where nx, ny in [-2, -1, 0, 1, 2]
    N_pw_half = 2
    G_vectors = []
    for nx in range(-N_pw_half, N_pw_half + 1):
        for ny in range(-N_pw_half, N_pw_half + 1):
            G_vectors.append((nx, ny))
    G_vectors = np.array(G_vectors, dtype=float)
    N_pw = len(G_vectors) # 25

    def compute_eps_inv_fourier(cell):
        # Compute Fourier coefficients of 1/eps on the G-grid
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        # We need coefficients c_G = (1/N^2) sum_{r} (1/eps)_r exp(-i 2pi G r)
        # The evaluator uses "exact Fourier coefficients".
        # Let's use np.fft.fft2
        # Note: The G-vectors in the PWE sum correspond to the reciprocal lattice vectors.
        # In the pixel basis, the reciprocal vectors are (2pi/a) * (nx, ny).
        # Our FFT grid corresponds to frequencies -N/2 ... N/2-1.
        # We need the coefficients for G in [-2..2] x [-2..2].
        
        # To be precise, we should compute the coefficients for the specific G-vectors used.
        # c_G = (1/N^2) sum_{i,j} (1/eps)_{ij} exp(-i 2pi (nx * i/N + ny * j/N))
        
        coeffs = np.zeros((N_pw, N_pw), dtype=complex)
        # We can use matvec or just direct sum for speed since N_pw is small and grid is 32x32
        # But FFT is faster.
        # Let's use FFT and extract the needed coefficients.
        # fft2 gives coefficients for G in [-16..15].
        # We need G in [-2..2].
        fft_inv = np.fft.fft2(inv_eps)
        fft_inv /= (size * size)
        fft_inv = np.fft.fftshift(fft_inv)
        
        # Extract the 5x5 block centered at 0
        # fftshifted index for G=(nx,ny) is (nx + size//2, ny + size//2)
        # Wait, fftshift puts 0 in the middle.
        # Index 0 is -16. Index 16 is 0.
        # So G=(nx, ny) is at index (nx+16, ny+16).
        start = size // 2 - N_pw_half
        end = size // 2 + N_pw_half + 1
        coeffs_block = fft_inv[start:end, start:end]
        
        # Map coeffs_block to G_vectors order
        # G_vectors are ordered:
        # (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2)
        # ...
        # coeffs_block[0,0] corresponds to G=(-2,-2)
        # So they match directly if we arrange G_vectors in the same order.
        # Our G_vectors loop:
        # for nx in -2..2:
        #   for ny in -2..2:
        # This matches the row-major order of the 5x5 block if rows are nx and cols are ny.
        # fft_inv[start:end, start:end] has rows corresponding to y (ny) and cols to x (nx)?
        # No, np.fft.fft2(inv_eps) where inv_eps is (32,32) with axes (y,x).
        # Axis 0 is y, Axis 1 is x.
        # So fft_inv[i,j] is coeff for G=(j, i) in (nx, ny)?
        # Let's check: exp(-i 2pi (nx x + ny y)).
        # Numpy FFT: sum_{x,y} f(y,x) exp(-i 2pi (kx x/N + ky y/N)).
        # So fft_inv[ky, kx] is the coefficient for G=(kx, ky).
        # We need to be careful with indexing.
        
        # Let's just compute directly for safety and speed is okay for 32x32.
        # 32*32*25 = 25600 ops. Very fast.
        coeffs = np.zeros((N_pw, N_pw), dtype=complex)
        ys = np.arange(size) / size
        xs = np.arange(size) / size
        YY, XX = np.meshgrid(ys, xs, indexing='ij')
        
        for idx, (nx, ny) in enumerate(G_vectors):
            # phase = exp(-i 2pi (nx * x + ny * y))
            phase = -2j * math.pi * (nx * XX + ny * YY)
            coeffs[idx] = np.sum(inv_eps * np.exp(phase)) / (size * size)
            
        return coeffs

    def solve_bands_fast(cell, kx, ky, coeffs, num_bands=5):
        # H_ij = |k_i|^2 * (1/eps)_{G_i - G_j}  ? 
        # Standard PWE for TM:
        # H_ij = (1/eps)_{G_i - G_j} * (k + G_i)^2
        # Wait, the operator is 1/eps * (-nabla^2).
        # In plane waves: 1/eps * (k+G)^2.
        # So H_ij = (1/eps)_{G_i - G_j} * |k + G_i|^2 ?
        # No, the product 1/eps * E is convolved.
        # (1/eps * E)_k = sum_G (1/eps)_{k-G} E_G.
        # Then -nabla^2 acts on E_G to give |k+G|^2 E_G.
        # So the equation is: sum_G (1/eps)_{G_i - G} |k+G_i|^2 E_G = omega^2 E_{G_i}.
        # Wait, the order matters.
        # The eigenvalue equation is:
        # sum_j (1/eps)_{G_i - G_j} |k + G_i|^2 c_j = omega^2 c_i ?
        # No.
        # Let's derive:
        # E(r) = sum c_j e^{i(k+G_j)r}
        # 1/eps(r) E(r) = sum_{m} (1/eps)_m sum_j c_j e^{i(k+G_j+G_m)r}
        # Let i = j+m. So c_{i-m}.
        # Term is sum_i [ sum_m (1/eps)_m c_{i-m} ] e^{i(k+G_i)r}
        # Then -nabla^2 gives |k+G_i|^2 times that coefficient.
        # So: |k+G_i|^2 sum_m (1/eps)_m c_{i-m} = omega^2 c_i.
        # Let j = i-m. m = i-j.
        # |k+G_i|^2 sum_j (1/eps)_{G_i-G_j} c_j = omega^2 c_i.
        
        # So H[i,j] = |k+G_i|^2 * (1/eps)_{G_i-G_j}.
        
        k_vec = np.array([kx, ky])
        # k_i = k + G_i
        k_vectors = k_vec[None, :] + G_vectors
        k_mags_sq = np.sum(k_vectors**2, axis=1)
        
        H = np.zeros((N_pw, N_pw), dtype=complex)
        
        for i in range(N_pw):
            for j in range(N_pw):
                G_diff = G_vectors[i] - G_vectors[j]
                # Find index of G_diff in coeffs
                # coeffs is ordered by G_vectors
                # We need to find idx such that G_vectors[idx] == G_diff
                # Since G_vectors is small, we can just search or precompute a map.
                # Let's precompute a map outside the loop for efficiency in the caller, or just do it here since it's small.
                # For now, simple search.
                found = False
                for k_idx in range(N_pw):
                    if G_vectors[k_idx][0] == G_diff[0] and G_vectors[k_idx][1] == G_diff[1]:
                        coeff = coeffs[k_idx]
                        found = True
                        break
                if not found:
                    # If G_diff is outside our truncation, coeff is 0
                    coeff = 0.0
                H[i, j] = k_mags_sq[i] * coeff
                
        vals, vecs = np.linalg.eigh(H)
        # Filter small vals
        valid = vals > 1e-6
        if np.sum(valid) < num_bands:
            return None, None
        return vals[valid][:num_bands], vecs[:, valid][:, :num_bands]

    def evaluate_gap_fast(cell):
        coeffs = compute_eps_inv_fourier(cell)
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            vals, vecs = solve_bands_fast(cell, kx, ky, coeffs, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
        if min_omega_5 == float('inf'):
            return -1.0, None, None, None, None
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0, None, None, None, None
            
        ratio = gap / midgap
        return ratio, max_omega_4, min_omega_5, coeffs, k_points

    def enforce_symmetry(cell):
        # C4v symmetry:
        # 1. Rotate 180: cell[i,j] == cell[size-1-i, size-1-j]
        # 2. Reflect X: cell[i,j] == cell[i, size-1-j]
        # 3. Reflect Y: cell[i,j] == cell[size-1-i, j]
        # 4. Transpose: cell[i,j] == cell[j,i]
        
        # To enforce, we average over the group orbits and threshold.
        # Orbits are sets of up to 8 pixels (or fewer on boundaries).
        # Let's just do the operations sequentially on a float array and threshold.
        # This might not be strictly C4v invariant if not done carefully, but averaging is good.
        
        c = cell.astype(float)
        c = c + c[::-1, ::-1] # 180 rot
        c = c + c[:, ::-1]    # Ref X
        c = c + c[::-1, :]    # Ref Y
        c = c + c.T           # Transpose
        
        # The sum should be 8 * cell_value for interior points.
        # Boundary points have fewer partners.
        # To be safe, let's just do:
        # c1 = cell
        # c2 = cell[::-1, ::-1]
        # c3 = cell[:, ::-1]
        # c4 = c3[::-1, :]
        # c5 = cell.T
        # c6 = cell[::-1, ::-1].T
        # c7 = cell[:, ::-1].T
        # c8 = c7[::-1, :]
        
        c = cell.astype(float)
        c += c[::-1, ::-1]
        c += c[:, ::-1]
        c += c[::-1, :]
        c += c.T
        c += c[::-1, ::-1].T
        c += c[:, ::-1].T
        c += c[::-1, ::-1, :].T # This is wrong syntax, let's be explicit
        
        # Let's do it properly.
        c = cell.astype(float)
        c1 = c
        c2 = c[::-1, ::-1]
        c3 = c[:, ::-1]
        c4 = c3[::-1, :]
        c5 = c.T
        c6 = c2.T
        c7 = c3.T
        c8 = c4.T
        
        c_avg = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) / 8.0
        
        # Threshold
        # We want to keep the "mass" similar?
        # If we just threshold at 0.5, we might lose dielectric.
        # Let's threshold at 0.5.
        c_bin = (c_avg > 0.5).astype(int)
        
        return c_bin

    # Initialize population
    # Start with the "4 rods" design from the seed, but optimized.
    # Actually, let's start with a random population of symmetric cells.
    pop_size = 10
    population = []
    
    # Seed 1: 4 rods
    y, x = np.mgrid[0:size, 0:size]
    y = y + 0.5
    x = x + 0.5
    centers = [(size/4, size/4), (size/4, 3*size/4), (3*size/4, size/4), (3*size/4, 3*size/4)]
    r = 5.5
    cell0 = np.zeros((size, size), dtype=int)
    for cx, cy in centers:
        dist = (x - cx)**2 + (y - cy)**2
        cell0[dist < r**2] = 1
    cell0 = enforce_symmetry(cell0)
    population.append(cell0)
    
    # Seed 2-10: Random
    for _ in range(pop_size - 1):
        c_rand = np.zeros((size, size), dtype=int)
        # Generate on fundamental domain (upper right quadrant?)
        # Due to symmetry, we only need to generate 1/8th?
        # No, 1/4 is easier if we reflect.
        # Let's generate full grid random and enforce symmetry.
        c_rand = (np.random.rand(size, size) > 0.5).astype(int)
        c_rand = enforce_symmetry(c_rand)
        population.append(c_rand)
        
    # Evaluate initial population
    scores = []
    for c in population:
        score, *_ = evaluate_gap_fast(c)
        scores.append(score)
        
    best_idx = np.argmax(scores)
    best_cell = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Optimization Loop
    # Genetic Algorithm with Local Search
    
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        # Selection: Keep top half
        sorted_indices = np.argsort(scores)[::-1]
        parents = [population[i] for i in sorted_indices[:pop_size//2]]
        
        new_population = [best_cell.copy()] # Elitism
        
        # Crossover and Mutation
        while len(new_population) < pop_size and time.time() < deadline - 10:
            # Pick two parents
            p1_idx = np.random.randint(0, len(parents))
            p2_idx = np.random.randint(0, len(parents))
            p1 = parents[p1_idx]
            p2 = parents[p2_idx]
            
            # Crossover: Uniform
            mask = (np.random.rand(size, size) > 0.5).astype(int)
            child = p1 * mask + p2 * (1 - mask)
            
            # Enforce symmetry
            child = enforce_symmetry(child)
            
            # Mutation: Flip random symmetric orbits
            # Pick a few random pixels and flip their orbits
            num_mutations = np.random.randint(1, 5)
            for _ in range(num_mutations):
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                
                # Flip orbit
                # (i,j), (size-1-i, size-1-j), (i, size-1-j), (size-1-i, j)
                # (j,i), (size-1-j, size-1-i), (j, size-1-i), (size-1-j, i)
                
                # Just flip (i,j) and then enforce symmetry?
                # No, that changes the mass.
                # Let's flip the whole orbit.
                # It's easier to just pick a random pixel, flip it, and then enforce symmetry.
                # This might break the "orbit flip" logic if we don't flip all partners.
                # Let's flip (i,j) and its 7 partners explicitly.
                
                partners = [
                    (i, j),
                    (size-1-i, size-1-j),
                    (i, size-1-j),
                    (size-1-i, j),
                    (j, i),
                    (size-1-j, size-1-i),
                    (j, size-1-i),
                    (size-1-j, i)
                ]
                
                # Only flip if they are distinct
                unique_partners = list(set(partners))
                val = 1 - child[i, j]
                for pi, pj in unique_partners:
                    child[pi, pj] = val
                    
                # Enforce symmetry again (might have broken it if we didn't cover all)
                # Actually, flipping the full orbit preserves symmetry.
                # But let's enforce just in case.
                child = enforce_symmetry(child)
            
            new_population.append(child)
            
        # Evaluate new population
        new_scores = []
        for c in new_population:
            score, *_ = evaluate_gap_fast(c)
            new_scores.append(score)
            
        # Update population
        population = new_population
        scores = new_scores
        
        # Check for best
        best_idx_new = np.argmax(scores)
        if scores[best_idx_new] > best_score:
            best_score = scores[best_idx_new]
            best_cell = population[best_idx_new].copy()
            
        # Local Search on Best
        # Try flipping single orbits in the best cell
        if time.time() < deadline - 5:
            # Get a list of all unique orbits
            orbits = set()
            for i in range(size):
                for j in range(size):
                    # Generate orbit
                    partners = [
                        (i, j),
                        (size-1-i, size-1-j),
                        (i, size-1-j),
                        (size-1-i, j),
                        (j, i),
                        (size-1-j, size-1-i),
                        (j, size-1-i),
                        (size-1-j, i)
                    ]
                    # Normalize orbit for set key (e.g., min tuple)
                    key = min(partners)
                    if key not in orbits:
                        orbits.add(key)
                        # Store the full orbit for this key
                        orbits[key] = list(set(partners))
                        
            # Try a subset of orbits
            # Randomly sample 20 orbits
            orbit_keys = list(orbits.keys())
            np.random.shuffle(orbit_keys)
            
            for key in orbit_keys[:20]:
                if time.time() > deadline - 5:
                    break
                partners = orbits[key]
                test_cell = best_cell.copy()
                val = 1 - test_cell[partners[0][0], partners[0][1]]
                for pi, pj in partners:
                    test_cell[pi, pj] = val
                test_cell = enforce_symmetry(test_cell)
                
                score, *_ = evaluate_gap_fast(test_cell)
                if score > best_score:
                    best_score = score
                    best_cell = test_cell

    return best_cell.tolist()
