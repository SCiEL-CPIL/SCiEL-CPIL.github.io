import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry: the cell must be symmetric under the square group about the centre.
    # We enforce this by defining the top-left 16x16 quadrant and mirroring it.
    # Quadrant indices: i in [0,15], j in [0,15]
    # Full cell:
    #   cell[i, j]       = q[i, j]
    #   cell[i, size-1-j] = q[i, j]   (horizontal mirror)
    #   cell[size-1-i, j] = q[i, j]   (vertical mirror)
    #   cell[size-1-i, size-1-j] = q[i, j]
    
    half = size // 2  # 16
    
    # Pre-compute which quadrant pixel maps to which full-cell positions
    # For a given (i, j) in [0, half-1]^2, the orbit in the full cell is:
    #   (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
    # But we need to be careful about the center lines. Since size=32, half=16,
    # there's no center line ambiguity (even size).
    
    def quadrant_to_full(q):
        """q is 16x16, returns 32x32 full cell."""
        full = np.zeros((size, size), dtype=int)
        full[:half, :half] = q
        full[:half, half:] = q[:, ::-1]
        full[half:, :half] = q[::-1, :]
        full[half:, half:] = q[::-1, ::-1]
        return full
    
    def full_to_quadrant(full):
        """Extract the top-left quadrant from a full cell."""
        return full[:half, :half].copy()
    
    # Fast PWE solver using reduced basis
    # Use N_pw such that we have enough modes but stay fast
    N_pw = 7  # -3 to 3, giving 49 modes
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in irreducible BZ (8-fold symmetry)
    # Gamma (0,0), X (0.5,0), M (0.5,0.5), and interior points
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.0),      # Mid-Gamma-X
        (0.25, 0.25),     # Mid-Gamma-M
        (0.5, 0.25),      # Mid-X-M
        (0.125, 0.0),
        (0.125, 0.125),
        (0.375, 0.125),
    ]
    
    def compute_eps_fft(full_cell):
        """Compute Fourier coefficients of epsilon."""
        eps_grid = eps_low + (eps_high - eps_low) * full_cell.astype(float)
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        return eps_fft
    
    def solve_pwe(eps_fft, kx, ky):
        """Solve PWE eigenvalue problem for given k-point. Returns eigenvalues."""
        M = np.zeros((N_modes, N_modes), dtype=float)
        
        for i in range(N_modes):
            gx, gy = G_vectors[i]
            kx_i = kx + gx
            ky_i = ky + gy
            k_sq_i = kx_i**2 + ky_i**2
            
            for j in range(N_modes):
                gx_j, gy_j = G_vectors[j]
                kx_j = kx + gx_j
                ky_j = ky + gy_j
                
                dot = kx_i * kx_j + ky_i * ky_j
                
                dgx = gx - gx_j
                dgy = gy - gy_j
                idx_i = dgx % size
                idx_j = dgy % size
                eps_val = eps_fft[idx_i, idx_j].real
                
                if i == j:
                    M[i, j] = k_sq_i - dot / eps_val
                else:
                    M[i, j] = - dot / eps_val
        
        try:
            eigvals = np.linalg.eigvalsh(M)
        except:
            return None
        
        return eigvals
    
    def evaluate_gap(full_cell):
        """Evaluate the gap-midgap ratio between bands 4 and 5."""
        eps_fft = compute_eps_fft(full_cell)
        
        global_max_4 = 0.0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            eigvals = solve_pwe(eps_fft, kx, ky)
            if eigvals is None or len(eigvals) < 5:
                return -1.0
            
            # eigvals are omega^2, so omega = sqrt(eigval)
            omega_4 = np.sqrt(max(eigvals[3], 0))
            omega_5 = np.sqrt(max(eigvals[4], 0))
            
            if omega_4 > global_max_4:
                global_max_4 = omega_4
            if omega_5 < global_min_5:
                global_min_5 = omega_5
        
        if global_min_5 == float('inf'):
            return -1.0
        
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap == 0:
            return 0.0
        
        return gap / midgap
    
    def get_band_eigenvectors(eps_fft, kx, ky):
        """Get eigenvectors for bands 4 and 5 at a given k-point."""
        M = np.zeros((N_modes, N_modes), dtype=float)
        
        for i in range(N_modes):
            gx, gy = G_vectors[i]
            kx_i = kx + gx
            ky_i = ky + gy
            k_sq_i = kx_i**2 + ky_i**2
            
            for j in range(N_modes):
                gx_j, gy_j = G_vectors[j]
                kx_j = kx + gx_j
                ky_j = ky + gy_j
                
                dot = kx_i * kx_j + ky_i * ky_j
                
                dgx = gx - gx_j
                dgy = gy - gy_j
                idx_i = dgx % size
                idx_j = dgy % size
                eps_val = eps_fft[idx_i, idx_j].real
                
                if i == j:
                    M[i, j] = k_sq_i - dot / eps_val
                else:
                    M[i, j] = - dot / eps_val
        
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
        except:
            return None, None
        
        if len(eigvals) < 5:
            return None, None
        
        # Bands 4 and 5 (0-indexed: 3 and 4)
        vec4 = eigvecs[:, 3]
        vec5 = eigvecs[:, 4]
        return vec4, vec5
    
    def field_energy_density(vec, kx, ky):
        """Compute spatial energy density of a plane-wave mode."""
        # The field in real space: E(r) = sum_G c_G exp(i(k+G)·r)
        # Energy density ~ |E(r)|^2
        # We compute this on a 32x32 grid
        
        xs = (np.arange(size) + 0.5) / size
        ys = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(xs, ys)
        
        field = np.zeros((size, size), dtype=complex)
        
        for idx, (gx, gy) in enumerate(G_vectors):
            if abs(vec[idx]) < 1e-10:
                continue
            kx_i = kx + gx
            ky_i = ky + gy
            field += vec[idx] * np.exp(2j * np.pi * (kx_i * X + ky_i * Y))
        
        return np.abs(field)**2
    
    def spectral_update(q, n_iter=5):
        """
        Spectral-domain inverse design: adjust dielectric distribution based on
        band field energy densities.
        """
        full = quadrant_to_full(q)
        eps_fft = compute_eps_fft(full)
        
        # Find the k-point where the gap is most critical
        # (where band 4 is highest and band 5 is lowest)
        best_k = None
        best_gap = -float('inf')
        max_4_at_k = {}
        min_5_at_k = {}
        
        for kx, ky in k_points:
            eigvals = solve_pwe(eps_fft, kx, ky)
            if eigvals is None or len(eigvals) < 5:
                continue
            omega_4 = np.sqrt(max(eigvals[3], 0))
            omega_5 = np.sqrt(max(eigvals[4], 0))
            max_4_at_k[(kx, ky)] = omega_4
            min_5_at_k[(kx, ky)] = omega_5
        
        if not max_4_at_k:
            return q
        
        # Find k-point with maximum omega_4
        k_max4 = max(max_4_at_k.keys(), key=lambda k: max_4_at_k[k])
        # Find k-point with minimum omega_5
        k_min5 = min(min_5_at_k.keys(), key=lambda k: min_5_at_k[k])
        
        # Get eigenvectors at these k-points
        vec4, _ = get_band_eigenvectors(eps_fft, k_max4[0], k_max4[1])
        _, vec5 = get_band_eigenvectors(eps_fft, k_min5[0], k_min5[1])
        
        if vec4 is None or vec5 is None:
            return q
        
        # Compute field energy densities
        energy4 = field_energy_density(vec4, k_max4[0], k_max4[1])
        energy5 = field_energy_density(vec5, k_min5[0], k_min5[1])
        
        # Normalize
        energy4 = energy4 / (np.sum(energy4) + 1e-10)
        energy5 = energy5 / (np.sum(energy5) + 1e-10)
        
        # The idea: we want dielectric (high eps) where band 4 energy is high
        # and band 5 energy is low, and air (low eps) where band 5 energy is high
        # and band 4 energy is low.
        # Score for making pixel dielectric: energy4 - energy5
        # Score for making pixel air: energy5 - energy4
        
        # Work on the full cell, then extract quadrant
        full_float = full.astype(float)
        
        # Gradient-like update
        # Where energy4 >> energy5, prefer dielectric
        # Where energy5 >> energy4, prefer air
        update = energy4 - energy5
        
        # Apply update to full cell
        for iter_idx in range(n_iter):
            # Binary update: if update > 0, set to dielectric; else air
            # But use a soft threshold to avoid abrupt changes
            threshold = 0.0
            new_full = (update > threshold).astype(int)
            
            # Ensure symmetry by averaging over quadrants
            q_new = full_to_quadrant(new_full)
            
            # Check if this improves the gap
            gap_new = evaluate_gap(quadrant_to_full(q_new))
            gap_old = evaluate_gap(quadrant_to_full(q))
            
            if gap_new > gap_old:
                q = q_new
                # Recompute for next iteration
                full = quadrant_to_full(q)
                eps_fft = compute_eps_fft(full)
                vec4, _ = get_band_eigenvectors(eps_fft, k_max4[0], k_max4[1])
                _, vec5 = get_band_eigenvectors(eps_fft, k_min5[0], k_min5[1])
                if vec4 is not None and vec5 is not None:
                    energy4 = field_energy_density(vec4, k_max4[0], k_max4[1])
                    energy5 = field_energy_density(vec5, k_min5[0], k_min5[1])
                    energy4 = energy4 / (np.sum(energy4) + 1e-10)
                    energy5 = energy5 / (np.sum(energy5) + 1e-10)
                    update = energy4 - energy5
            else:
                break
        
        return q
    
    def local_bit_flip(q, n_steps=200):
        """Local search by flipping individual pixels in the quadrant."""
        best_q = q.copy()
        best_score = evaluate_gap(quadrant_to_full(best_q))
        
        rng = np.random.default_rng(42)
        
        for step in range(n_steps):
            # Pick a random pixel in the quadrant to flip
            i = rng.integers(0, half)
            j = rng.integers(0, half)
            
            # Flip it
            new_q = best_q.copy()
            new_q[i, j] = 1 - new_q[i, j]
            
            new_score = evaluate_gap(quadrant_to_full(new_q))
            
            if new_score > best_score:
                best_q = new_q
                best_score = new_score
        
        return best_q, best_score
    
    # Main optimization loop
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    best_q = None
    best_score = -float('inf')
    
    # Initialize with several random patterns
    rng = np.random.default_rng(123)
    
    # Strategy 1: Random binary patterns
    for init in range(5):
        if time.time() > deadline - 60:
            break
        
        q = (rng.random((half, half)) < 0.5).astype(int)
        
        # Spectral refinement
        q = spectral_update(q, n_iter=3)
        
        # Local bit flip refinement
        q, score = local_bit_flip(q, n_steps=100)
        
        if score > best_score:
            best_score = score
            best_q = q.copy()
    
    # Strategy 2: Start from known good structures
    # Four quadrant rods (sub-periodic)
    if time.time() < deadline - 60:
        q = np.zeros((half, half), dtype=int)
        # Place a rod in each quadrant of the full cell
        # In the 16x16 quadrant, place a small rod
        center_i, center_j = 7, 7  # Center of the 16x16 quadrant
        r = 3
        for i in range(half):
            for j in range(half):
                if (i - center_i)**2 + (j - center_j)**2 < r**2:
                    q[i, j] = 1
        
        q = spectral_update(q, n_iter=5)
        q, score = local_bit_flip(q, n_steps=150)
        
        if score > best_score:
            best_score = score
            best_q = q.copy()
    
    # Strategy 3: Checkerboard-like pattern
    if time.time() < deadline - 60:
        q = np.zeros((half, half), dtype=int)
        for i in range(half):
            for j in range(half):
                if (i // 4 + j // 4) % 2 == 0:
                    q[i, j] = 1
        
        q = spectral_update(q, n_iter=5)
        q, score = local_bit_flip(q, n_steps=150)
        
        if score > best_score:
            best_score = score
            best_q = q.copy()
    
    # If we still have time, do more local refinement on the best
    if best_q is not None and time.time() < deadline - 10:
        remaining_steps = int((deadline - 10 - time.time()) / 0.05)  # ~50ms per step
        remaining_steps = max(remaining_steps, 50)
        best_q, best_score = local_bit_flip(best_q, n_steps=remaining_steps)
    
    # Fallback: simple central rod
    if best_q is None:
        best_q = np.zeros((half, half), dtype=int)
        center_i, center_j = 7, 7
        r = 4
        for i in range(half):
            for j in range(half):
                if (i - center_i)**2 + (j - center_j)**2 < r**2:
                    best_q[i, j] = 1
    
    # Convert to full cell
    full_cell = quadrant_to_full(best_q)
    
    # Final symmetry verification
    # Check all four symmetries
    assert np.array_equal(full_cell, full_cell[::-1, :]), "Vertical symmetry failed"
    assert np.array_equal(full_cell, full_cell[:, ::-1]), "Horizontal symmetry failed"
    assert np.array_equal(full_cell, full_cell.T), "Transpose symmetry failed"
    
    return full_cell.tolist()
