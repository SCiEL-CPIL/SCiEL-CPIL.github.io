import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 220.0 # Safety margin

    # --- Solver ---
    
    def get_pw_indices(R):
        """Get plane wave indices for circular cutoff R."""
        indices = []
        for m in range(-R, R+1):
            for n in range(-R, R+1):
                if m*m + n*n <= R*R:
                    indices.append((m, n))
        return indices

    def compute_tm_bands(cell, R, k_points, return_eigvecs=False):
        """
        Compute TM band structure.
        cell: 32x32 binary array.
        R: cutoff radius for PWs.
        k_points: list of (kx, ky).
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        # Fourier coefficients of 1/eps
        # Note: np.fft.fft2 uses e^{-i 2pi ...}
        # We need S[G-G'] = <1/eps>_{G-G'}
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        pw_indices = get_pw_indices(R)
        n_pw = len(pw_indices)
        
        # Precompute S matrix
        # S[i, j] = fft_inv_eps[ (mi-mj) % N, (ni-nj) % N ]
        S = np.zeros((n_pw, n_pw), dtype=complex)
        for i, (mi, ni) in enumerate(pw_indices):
            for j, (mj, nj) in enumerate(pw_indices):
                gm = (mi - mj) % size
                gn = (ni - nj) % size
                S[i, j] = fft_inv_eps[gm, gn]
                
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        
        for kx, ky in k_points:
            # Build H matrix
            # H[i, j] = S[i, j] * |k + G_j|^2
            # Note: The eigenvalue is omega^2.
            # Equation: sum_j S[i,j] (k+G_j)^2 C_j = omega^2 C_i
            # Wait, the standard PWE equation for TM is:
            # sum_{G'} (1/eps)_{G-G'} (k+G')^2 C_{G'} = omega^2 C_G
            # So H[i, j] = S[i, j] * (kx + mj)^2 + (ky + nj)^2
            # where G_j = (mj, nj)
            
            k_sq = np.zeros(n_pw)
            for j, (mj, nj) in enumerate(pw_indices):
                kx_j = kx + mj
                ky_j = ky + nj
                k_sq[j] = kx_j**2 + ky_j**2
            
            H = S * k_sq[None, :] # Broadcasting: S[i, j] * k_sq[j]
            
            try:
                eigvals, eigvecs = np.linalg.eigh(H)
                # Filter out negative eigenvalues (numerical noise)
                eigvals = np.real(eigvals)
                eigvals = np.where(eigvals > 0, eigvals, 0)
                omegas = np.sqrt(eigvals)
                
                if len(omegas) >= 5:
                    w4 = omegas[3]
                    w5 = omegas[4]
                    
                    if w4 > max_omega4:
                        max_omega4 = w4
                    if w5 < min_omega5:
                        min_omega5 = w5
                        best_k = (kx, ky)
                        if return_eigvecs:
                            best_vec4 = eigvecs[:, 3]
                            best_vec5 = eigvecs[:, 4]
            except Exception:
                continue
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            return -1.0, None, None, None
            
        if min_omega5 + max_omega4 <= 0:
            return -1.0, None, None, None
            
        score = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        
        if return_eigvecs:
            return score, best_k, best_vec4, best_vec5
        return score

    # K-points
    # Gamma (0,0) -> X (0.5, 0) -> M (0.5, 0.5) -> Gamma (0,0)
    def get_k_points(n_per_seg):
        k_pts = []
        for t in np.linspace(0, 0.5, n_per_seg + 1):
            k_pts.append((t, 0.0))
        for t in np.linspace(0, 0.5, n_per_seg + 1):
            k_pts.append((0.5, t))
        for t in np.linspace(0.5, 0, n_per_seg + 1):
            k_pts.append((t, t))
        # Remove duplicates
        unique_k = []
        for kp in k_pts:
            if not any(np.allclose(kp, ukp, atol=1e-6) for ukp in unique_k):
                unique_k.append(kp)
        return unique_k

    k_fast = get_k_points(5) # ~15 points
    k_acc = get_k_points(15) # ~45 points

    R_fast = 4 # ~57 PWs
    R_acc = 8 # ~200 PWs (closer to 17x17=289)
    
    # Actually, let's use square cutoff for accuracy to match evaluator?
    # Evaluator uses 289 PWs. 17x17 square.
    # Let's use R_acc such that it's close to 17x17.
    # R=8 circular has ~200.
    # R=9 circular has ~254.
    # R=10 circular has ~314.
    # Let's use R=9 for accuracy.

    def fast_score(cell):
        return compute_tm_bands(cell, R_fast, k_fast)[0]

    def acc_score(cell, return_eigvecs=False):
        return compute_tm_bands(cell, R_acc, k_acc, return_eigvecs=return_eigvecs)

    # --- Symmetry ---
    def symmetrize(cell):
        c = np.array(cell, dtype=float)
        c = (c + c[::-1, :] + c[:, ::-1] + c.T + c[::-1, ::-1] + 
             c[:, ::-1].T + c[::-1, :].T + c[::-1, ::-1].T) / 8.0
        return (c > 0.5).astype(int)

    # --- GA ---
    pop_size = 60
    population = []
    
    # Init 1: 4 rods
    def make_rods(r, exp=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            val = np.abs(DX/r)**exp + np.abs(DY/r)**exp
            cell[val <= 1.0] = 1
        return symmetrize(cell)
    
    # Init 2: 4 holes (Air rods)
    def make_holes(r, exp=2.0):
        cell = np.ones((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            val = np.abs(DX/r)**exp + np.abs(DY/r)**exp
            cell[val <= 1.0] = 0
        return symmetrize(cell)

    # Init 3: Random
    def make_random():
        cell = np.random.randint(0, 2, (size, size))
        return symmetrize(cell)

    # Add initial designs
    population.append(make_rods(0.12, 2.0))
    population.append(make_rods(0.15, 2.0))
    population.append(make_rods(0.10, 2.5))
    population.append(make_holes(0.12, 2.0))
    population.append(make_holes(0.15, 2.0))
    
    while len(population) < pop_size:
        population.append(make_random())
        
    # Evaluate initial
    scores = [fast_score(c) for c in population]
    
    best_cell = population[np.argmax(scores)]
    best_score = max(scores)
    
    # GA Loop
    max_gens = 50
    for gen in range(max_gens):
        if time.time() > deadline - 30:
            break
            
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        new_population = []
        
        # Elitism
        for i in range(5):
            new_population.append(population[i])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                idxs = np.random.choice(len(population), 3, replace=False)
                best_idx = idxs[np.argmax([scores[i] for i in idxs])]
                return population[best_idx]
            
            p1 = select()
            p2 = select()
            
            # Crossover
            mask = np.random.randint(0, 2, (size, size))
            child = np.where(mask == 1, p1, p2)
            
            # Mutation
            # Flip 1-3 pixels
            n_mut = np.random.randint(1, 4)
            for _ in range(n_mut):
                x = np.random.randint(size)
                y = np.random.randint(size)
                child[x, y] = 1 - child[x, y]
                
            child = symmetrize(child)
            new_population.append(child)
            
        population = new_population
        scores = [fast_score(c) for c in population]
        
        # Update best
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_cell = population[max_idx]

    # --- Refinement ---
    # Take top 5 from GA
    sorted_idx = np.argsort(scores)[::-1]
    top_cells = [population[i] for i in sorted_idx[:5]]
    
    for cell in top_cells:
        if time.time() > deadline - 10:
            break
            
        current_cell = cell
        current_score, _, _, _ = acc_score(current_cell, return_eigvecs=True)
        
        # Hill Climbing
        for step in range(20):
            if time.time() > deadline - 5:
                break
                
            score, best_k, vec4, vec5 = acc_score(current_cell, return_eigvecs=True)
            if best_k is None:
                break
                
            kx, ky = best_k
            
            # Compute Field Intensity
            # Use R_acc PWs for field calc? Or just the ones in the solver?
            # The eigenvector is in the PW basis of size R_acc.
            # We need to sum over these PWs.
            
            pw_indices = get_pw_indices(R_acc)
            m_vals = np.array([g[0] for g in pw_indices])
            n_vals = np.array([g[1] for g in pw_indices])
            
            coords = (np.arange(size) + 0.5) / size - 0.5
            xx, yy = np.meshgrid(coords, coords, indexing='ij')
            
            def calc_intensity(vec):
                intensity = np.zeros((size, size), dtype=float)
                # Chunked sum to save memory if needed, but 200x32x32 is small
                m_k = m_vals + kx
                n_k = n_vals + ky
                
                phase = 2.0 * np.pi * (m_k[:, None, None] * xx[None, :, :] + 
                                       n_k[:, None, None] * yy[None, :, :])
                e = np.sum(vec[:, None, None] * np.exp(1j * phase), axis=0)
                return np.abs(e)**2
            
            I4 = calc_intensity(vec4)
            I5 = calc_intensity(vec5)
            
            I4 = I4 / (np.sum(I4) + 1e-12)
            I5 = I5 / (np.sum(I5) + 1e-12)
            
            # Gain
            # Add dielectric (0->1): Gain = I4 - I5
            # Remove dielectric (1->0): Gain = I5 - I4
            
            mask_add = (current_cell == 0)
            mask_rem = (current_cell == 1)
            
            gain_add = np.where(mask_add, I4 - I5, -np.inf)
            gain_rem = np.where(mask_rem, I5 - I4, -np.inf)
            
            # Find best flip
            idx_add = np.unravel_index(np.argmax(gain_add), gain_add.shape)
            best_gain_add = gain_add[idx_add]
            
            idx_rem = np.unravel_index(np.argmax(gain_rem), gain_rem.shape)
            best_gain_rem = gain_rem[idx_rem]
            
            improved = False
            if best_gain_add > best_gain_rem and best_gain_add > 0:
                new_cell = current_cell.copy()
                new_cell[idx_add] = 1
                new_cell = symmetrize(new_cell)
                if not np.all(new_cell == current_cell): # Check if actually changed
                    new_score, _, _, _ = acc_score(new_cell, return_eigvecs=True)
                    if new_score > current_score:
                        current_cell = new_cell
                        current_score = new_score
                        improved = True
            elif best_gain_rem > 0:
                new_cell = current_cell.copy()
                new_cell[idx_rem] = 0
                new_cell = symmetrize(new_cell)
                if not np.all(new_cell == current_cell):
                    new_score, _, _, _ = acc_score(new_cell, return_eigvecs=True)
                    if new_score > current_score:
                        current_cell = new_cell
                        current_score = new_score
                        improved = True
            
            if not improved:
                # Try random flip if stuck? Or break.
                # Let's break to save time for other candidates.
                break
                
        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell

    return best_cell.tolist()
