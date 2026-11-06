import numpy as np
import time
import random
import copy

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave margin for evaluator overhead (limit 120s/230s)

    # 1. High-speed, high-accuracy band solver
    # Plane waves: 289 = 17x17 grid (m, n in [-8, 8])
    half_pw = 8
    G_vecs = np.array([(m, n) for m in range(-half_pw, half_pw + 1) 
                       for n in range(-half_pw, half_pw + 1)], dtype=float) * (2.0 * np.pi)
    nG = len(G_vecs) # 289
    
    # K-points: 17x17 grid in the irreducible BZ triangle (0,0)-(pi,0)-(pi,pi)
    # This is a dense sampling to ensure we catch the global band edges.
    # 17*18/2 = 153 points. 
    k_points = []
    n_k = 16
    for i in range(n_k + 1):
        for j in range(n_k + 1 - i):
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    k_points = np.array(k_points)

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score.
        cell: 32x32 list of 0/1
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # Fourier coefficients of eps
        # eps_F[k] = sum eps(x) exp(-i G_k x) / V
        # For a pixel grid, this is essentially the FFT.
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        
        # The matrix A for the TM eigenvalue problem:
        # A_{mn} = (k+G_m)^2 delta_{mn} * (1/eps_{n-m}) ? 
        # No, the standard form is:
        # sum_m (k+G_m)^2 delta_{mn} E_m = omega^2 sum_m eps_{n-m} E_m
        # This is a generalized eigenvalue problem: D E = omega^2 Eps E
        # Or: Eps^{-1} D E = omega^2 E
        # Since Eps is circulant, Eps^{-1} is also circulant with coefficients 1/eps_F.
        # Wait, the inverse of a convolution is a convolution with the inverse Fourier coefficients.
        # So A = Eps^{-1} D is not a simple matrix multiplication, it's an operator.
        # A E = Eps^{-1} (D E)
        # Let v = D E. Then u = Eps^{-1} v.
        # u_k = sum_m (1/eps_F)_{k-m} v_m
        # This is a convolution, which is a matrix multiplication by a circulant matrix C.
        # C_{km} = (1/eps_F)_{k-m}
        
        # We can construct the matrix A explicitly for each k-point.
        # A = C(k) @ D(k) where D(k) is diagonal.
        # C is circulant, so it can be constructed from the first column.
        # First column of C corresponds to m=0. C_{k0} = (1/eps_F)_k.
        # So the first column is just 1/eps_F.
        
        inv_eps_F = 1.0 / eps_F
        # Handle potential zeros (should not happen for physical eps)
        inv_eps_F[np.abs(inv_eps_F) < 1e-10] = 0.0

        min_omega5 = np.inf
        max_omega4 = 0.0
        
        # Precompute G^2
        G2 = np.sum(G_vecs**2, axis=1)

        for kx, ky in k_points:
            kG = G_vecs + np.array([kx, ky])
            kG2 = np.sum(kG**2, axis=1)
            
            # Construct D matrix (diagonal)
            D = np.diag(kG2)
            
            # Construct C matrix (circulant)
            # C is a 289x289 matrix.
            # C_{ij} = inv_eps_F[(i - j) % nG] ? 
            # The convolution index is n-m.
            # Let's build it using the property of circulant matrices.
            # The first column is inv_eps_F.
            # We can use np.roll or just construct it directly for speed if needed, 
            # but 289x289 is small.
            # Actually, we can just do:
            # A = C @ D
            # Since D is diagonal, A_{ij} = C_{ij} * D_{jj} = C_{ij} * kG2[j]
            
            # Efficient construction of C:
            # C is circulant with first column inv_eps_F.
            # C_{ij} = inv_eps_F[(i - j) % nG]
            # We can generate the indices.
            # But even faster: A = C @ D.
            # Let's just build C once? No, inv_eps_F is constant, but D changes with k.
            # So C is constant!
            # Let's precompute C outside the k-loop.
            
            pass # Placeholder, logic moved below

        # Precompute C matrix
        # C_{ij} = inv_eps_F[(i - j) % nG]
        # Note: The FFT indexing in numpy is 0..N-1.
        # The G vector indices are -8..8.
        # We need to map the G vector index (m, n) to the FFT index.
        # G_vecs is ordered: 
        # m=-8, n=-8..8 (17)
        # m=-7, n=-8..8 (17)
        # ...
        # m=8, n=-8..8 (17)
        # Total 289.
        # The FFT of the 32x32 grid gives frequencies -16..15.
        # We need the coefficients for -8..8.
        # np.fft.fft2 returns coefficients in order 0..31.
        # Frequency 0 is at index 0.
        # Frequency -1 is at index 31 (or -1).
        # Frequency -8 is at index 24 (or -8).
        # Frequency 8 is at index 8.
        
        # Let's extract the correct slice from eps_F.
        # eps_F is 32x32.
        # We want m in -8..8, n in -8..8.
        # In numpy fft2, axis 0 is m, axis 1 is n.
        # Indices:
        # m=0 -> 0
        # m=1 -> 1
        # ...
        # m=8 -> 8
        # m=-1 -> 31
        # ...
        # m=-8 -> 24
        
        # Slice for m:
        # 0..8 and 24..31
        m_slice = np.concatenate([np.arange(0, 9), np.arange(24, 32)])
        n_slice = np.concatenate([np.arange(0, 9), np.arange(24, 32)])
        
        eps_F_crop = eps_F[np.ix_(m_slice, n_slice)]
        
        # Now eps_F_crop is 17x17.
        # We need to flatten it to match G_vecs order.
        # G_vecs order:
        # (m=-8, n=-8), (m=-8, n=-7), ..., (m=-8, n=8)
        # (m=-7, n=-8), ...
        
        # Let's map G_vecs to crop indices.
        # G_vecs[i] = (m, n).
        # In crop, m=0 is at index 0, m=1 at 1, ..., m=8 at 8.
        # m=-1 is at index 9, ..., m=-8 at index 16.
        # So m_crop_idx = m if m>=0 else m+17.
        # Same for n.
        
        inv_eps_F_crop = 1.0 / eps_F_crop
        
        # Flatten inv_eps_F_crop to match G_vecs order
        # G_vecs is generated by:
        # for m in -8..8:
        #   for n in -8..8:
        # So the index i corresponds to m = i // 17 - 8, n = i % 17 - 8.
        
        # We need to construct the circulant matrix C of size 289.
        # C_{ij} = inv_eps_F_crop[m_i - m_j, n_i - n_j]
        # where m_i, n_i are the G vector components of index i.
        
        # This is a 2D circulant matrix.
        # We can construct it using FFT properties or just explicitly.
        # 289x289 is small.
        
        # Let's get the m, n components for each G vector index.
        G_m = G_vecs[:, 0] / (2 * np.pi)
        G_n = G_vecs[:, 1] / (2 * np.pi)
        
        # Build C
        # C_{ij} = inv_eps_F_crop[(m_i - m_j) % 17, (n_i - n_j) % 17]
        # Note: The convolution is in the 32x32 grid, but we are using a 17x17 crop.
        # Is the 17x17 crop sufficient?
        # The convolution sum is over all n-m.
        # If we only keep -8..8, we are truncating the convolution.
        # However, the "exact Fourier coefficients" of the pixel map are defined on the 32x32 grid.
        # The plane wave expansion uses G vectors.
        # The matrix element (1/eps)_{n-m} is the Fourier coefficient of 1/eps?
        # No, for TM, the operator is Eps^{-1}.
        # Eps is the convolution matrix of eps.
        # Eps^{-1} is the convolution matrix of 1/eps_F?
        # Yes, if Eps is circulant, its inverse is circulant with inverse Fourier coefficients.
        # The Fourier coefficients of eps are eps_F.
        # So the Fourier coefficients of Eps^{-1} are 1/eps_F.
        # We need these coefficients for all n-m.
        # But n-m can be any integer.
        # However, the matrix is 289x289, so we only need indices 0..288.
        # The index in the 32x32 grid corresponds to the frequency.
        # The frequency m in the G vector corresponds to the frequency m in the 32x32 FFT.
        # So we need 1/eps_F at indices m_i - m_j.
        # m_i, m_j are in -8..8. So m_i - m_j is in -16..16.
        # The 32x32 FFT has frequencies -16..15.
        # So we need the full 32x32 1/eps_F, not just the crop!
        
        inv_eps_F_full = 1.0 / eps_F
        inv_eps_F_full[np.abs(inv_eps_F_full) < 1e-10] = 0.0
        
        # Now build C
        # C_{ij} = inv_eps_F_full[(m_i - m_j) % 32, (n_i - n_j) % 32]
        # Wait, the G vector m is the frequency index.
        # If m_i = -8 and m_j = 8, m_i - m_j = -16.
        # In numpy, -16 % 32 = 16.
        # The FFT index for -16 is 16 (since 32-16=16).
        # So yes, modulo 32 works.
        
        # Vectorized construction of C
        # m_diff = G_m[:, None] - G_m[None, :]
        # n_diff = G_n[:, None] - G_n[None, :]
        # m_diff_idx = m_diff % 32
        # n_diff_idx = n_diff % 32
        # C = inv_eps_F_full[m_diff_idx, n_diff_idx]
        
        m_diff = G_m[:, None] - G_m[None, :]
        n_diff = G_n[:, None] - G_n[None, :]
        m_diff_idx = m_diff.astype(int) % 32
        n_diff_idx = n_diff.astype(int) % 32
        C = inv_eps_F_full[m_diff_idx, n_diff_idx]
        
        # Now loop over k-points
        for kx, ky in k_points:
            kG = G_vecs + np.array([kx, ky])
            kG2 = np.sum(kG**2, axis=1)
            
            # A = C @ D, where D is diag(kG2)
            # A_{ij} = C_{ij} * kG2[j]
            A = C * kG2[None, :]
            
            # Solve eigenvalue problem A v = omega^2 v
            # A is not symmetric/Hermitian in this form?
            # C is Hermitian (if eps is real and symmetric).
            # D is real diagonal.
            # A = C D.
            # This is not a standard Hermitian eigenvalue problem.
            # We need to solve C D v = lambda v.
            # This is equivalent to D C D v = (lambda * D^2) v? No.
            # Let u = D v. Then C u = lambda D^{-1} u.
            # Or: D C D u = lambda D^2 u?
            # Let's use the standard form:
            # C D v = lambda v
            # Multiply by D^{-1/2} on left and right?
            # D^{-1/2} C D^{1/2} (D^{1/2} v) = lambda D^{1/2} v? No.
            
            # Actually, the matrix M = D^{1/2} C D^{1/2} is Hermitian.
            # Let's check:
            # M_{ij} = sqrt(kG2_i) C_{ij} sqrt(kG2_j)
            # M_{ji} = sqrt(kG2_j) C_{ji} sqrt(kG2_i)
            # Since C is Hermitian, C_{ji} = conj(C_{ij}).
            # So M is Hermitian.
            # The eigenvalues of M are the same as the eigenvalues of C D?
            # C D v = lambda v
            # D^{1/2} (C D) D^{-1/2} u = lambda u  (where u = D^{1/2} v)
            # D^{1/2} C D^{1/2} u = lambda u
            # Yes!
            
            D_half = np.sqrt(kG2)
            M = C * np.outer(D_half, D_half)
            
            # Solve for 5 smallest eigenvalues
            # np.linalg.eigvalsh is fast for 289x289
            try:
                vals = np.linalg.eigvalsh(M)
                vals = vals.real
                # Filter out negative values (numerical noise)
                vals = vals[vals > 0]
                
                if len(vals) < 5:
                    continue
                
                omega4 = np.sqrt(vals[3])
                omega5 = np.sqrt(vals[4])
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                continue

        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    # 2. Genetic Algorithm over the pixel bitmap
    # Symmetry: D4. Fundamental domain is 8 pixels (1/32 of the cell).
    # We represent the cell as a 32x32 array, but only optimize 8 pixels.
    # The other pixels are determined by symmetry.
    
    def generate_cell_from_domain(domain):
        # domain: list of 8 booleans (0/1)
        # The fundamental domain is the top-left 4x4 corner?
        # No, the symmetry group D4 acts on the 32x32 grid.
        # The fundamental domain is 1/8 of the grid.
        # Let's define the 8 pixels in the first octant.
        # Pixels (i, j) with i < j and i < 16? 
        # Actually, the center is (15.5, 15.5).
        # The octant is i < 16 and j < i? 
        # Let's just use the top-left 4x4 block? No, that's not a fundamental domain for D4 on 32x32.
        # A fundamental domain for D4 on an N x N grid (N even) is N^2 / 8 pixels.
        # For N=32, that's 128 pixels.
        # My previous thought of 8 pixels was wrong.
        # 32*32 = 1024. 1024/8 = 128.
        
        # Let's define the 128 pixels.
        # We can just pick one pixel from each orbit.
        # Or, simpler: generate a random 32x32 grid, and enforce symmetry.
        
        cell = np.zeros((size, size), dtype=int)
        # For each pixel (i, j), if it's in the fundamental domain, set it.
        # Fundamental domain: i < 16 and j < 16 and i < j? 
        # No, the symmetry is about the center (15.5, 15.5).
        # Let's use the condition: i < 16 and j < 16. (Top-left quadrant)
        # Then mirror to top-right, bottom-left, bottom-right.
        # Then transpose to handle the diagonal symmetry?
        # D4 includes:
        # 1. Identity
        # 2. 90 deg rotation
        # 3. 180 deg rotation
        # 4. 270 deg rotation
        # 5. Reflection across x-axis
        # 6. Reflection across y-axis
        # 7. Reflection across main diagonal
        # 8. Reflection across anti-diagonal
        
        # If we define the top-left 16x16 quadrant, we can generate the rest.
        # But the top-left 16x16 quadrant is not a fundamental domain for D4.
        # It is a fundamental domain for the subgroup of rotations? No.
        # The fundamental domain is 1/8 of the square.
        # Let's define the pixels (i, j) such that 0 <= i < 16 and 0 <= j < i.
        # This is a triangle of 120 pixels? 15*16/2 = 120.
        # Plus the diagonal pixels (i, i) for i in 0..15 (16 pixels).
        # Total 136? 
        # Let's just use a simple approach:
        # Generate a random 32x32 grid.
        # Enforce symmetry by averaging/rounding.
        
        # For the GA, we will represent the state as a 128-bit string (or 128 integers).
        # We will map these 128 bits to the 32x32 grid.
        
        # Let's define the 128 indices.
        # We can just use the first 128 pixels in row-major order? No, that breaks symmetry.
        # We need to pick 128 pixels that represent all orbits.
        
        # Let's use the condition: i < 16 and j < 16 and i <= j.
        # This is a 16x16 upper triangle (including diagonal).
        # Number of pixels: 16*17/2 = 136.
        # This is slightly more than 128.
        # Why? Because the center of the 32x32 grid is (15.5, 15.5).
        # The pixels are (0..31, 0..31).
        # The symmetry maps (i, j) to (j, i), (i, 31-j), etc.
        # The fundamental domain is 1/8.
        # 32*32/8 = 128.
        
        # Let's just use a random 32x32 grid and enforce symmetry in the evaluation.
        # This is simpler. The GA will operate on the 32x32 grid, but mutations will be symmetry-preserving.
        
        return cell

    # Let's stick to the random 32x32 grid with symmetry enforcement.
    # State: 32x32 array of 0/1.
    # Mutation: Flip a random pixel and its 7 symmetric counterparts.
    
    def enforce_symmetry(cell):
        # cell is 32x32
        # Symmetry operations:
        # 1. cell[i, j]
        # 2. cell[j, i] (transpose)
        # 3. cell[i, 31-j] (flip y)
        # 4. cell[31-i, j] (flip x)
        # 5. cell[31-i, 31-j] (rotate 180)
        # 6. cell[31-j, i] (rotate 90)
        # 7. cell[31-j, 31-i] (rotate 270)
        # 8. cell[j, 31-i] (anti-diagonal)
        
        # We can just set all 8 to the value of the first one?
        # Or average them?
        # Since we are dealing with 0/1, we can just pick one.
        # To be robust, we can average and threshold.
        
        # Let's just copy cell[0,0] to all? No, that's a specific pattern.
        # We want to maintain the pattern.
        # Let's define the orbits.
        
        sym_cell = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                # Get all symmetric pixels
                idxs = [
                    (i, j),
                    (j, i),
                    (i, size-1-j),
                    (size-1-i, j),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (size-1-j, size-1-i),
                    (j, size-1-i)
                ]
                # Average
                val = np.mean([cell[r, c] for r, c in idxs])
                # Threshold
                sym_val = 1 if val >= 0.5 else 0
                # Set all to this value
                for r, c in idxs:
                    sym_cell[r, c] = sym_val
        return sym_cell

    # Initialize population
    pop_size = 50
    population = []
    for _ in range(pop_size):
        cell = np.random.randint(0, 2, (size, size))
        cell = enforce_symmetry(cell)
        population.append(cell)
    
    best_score = -1.0
    best_cell = None
    
    # Evaluate initial population
    for i, cell in enumerate(population):
        if time.time() > deadline:
            break
        score = compute_score(cell.tolist())
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    # Genetic Algorithm
    generations = 50
    for gen in range(generations):
        if time.time() > deadline - 20:
            break
            
        # Evaluate all
        scores = []
        for cell in population:
            score = compute_score(cell.tolist())
            scores.append(score)
            
        # Update best
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_cell = population[max_idx].copy()
            
        # Selection: Tournament
        new_population = []
        for _ in range(pop_size):
            # Pick 3 random, choose best
            idxs = random.sample(range(pop_size), 3)
            best_idx = max(idxs, key=lambda x: scores[x])
            parent = population[best_idx]
            
            # Crossover: Uniform
            # Pick another parent
            idxs2 = random.sample(range(pop_size), 3)
            best_idx2 = max(idxs2, key=lambda x: scores[x])
            parent2 = population[best_idx2]
            
            child = np.zeros_like(parent)
            mask = np.random.randint(0, 2, (size, size))
            child = np.where(mask == 1, parent, parent2)
            
            # Mutation
            # Flip a few pixels and their symmetric counterparts
            n_mutations = random.randint(1, 5)
            for _ in range(n_mutations):
                i = random.randint(0, size-1)
                j = random.randint(0, size-1)
                # Flip (i, j) and all symmetric
                idxs = [
                    (i, j),
                    (j, i),
                    (i, size-1-j),
                    (size-1-i, j),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (size-1-j, size-1-i),
                    (j, size-1-i)
                ]
                for r, c in idxs:
                    child[r, c] = 1 - child[r, c]
            
            # Enforce symmetry (should be symmetric, but just in case)
            child = enforce_symmetry(child)
            
            new_population.append(child)
            
        population = new_population
        
    # Return the best cell
    if best_cell is None:
        # Fallback
        best_cell = np.zeros((size, size), dtype=int)
        best_cell[16:24, 16:24] = 1
        best_cell = enforce_symmetry(best_cell)
        
    return best_cell.tolist()

