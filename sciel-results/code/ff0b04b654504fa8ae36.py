import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using the standard Galerkin method.
    2. Generate candidate cells using four-quadrant rod configurations with various shapes.
    3. Score each candidate with the internal solver and return the best.
    4. Maintain D4 symmetry and stay within the time budget.
    5. Refine the best candidate using topological optimization guided by field intensities.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 10s margin for overhead
    start_time = time.time()
    deadline = start_time + 220.0

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=10, return_eigvecs=False):
        """
        Compute TM band structure using plane-wave expansion.
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        
        Returns:
        --------
        If return_eigvecs is False:
            score : float, the gap-midgap ratio
        If return_eigvecs is True:
            score, best_k, best_vec4, best_vec5
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        N = size
        
        # Compute Fourier coefficients of 1/epsilon
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Define the set of G vectors to include in the basis.
        # Use a circular cutoff for better accuracy per plane wave count.
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Build the S matrix: S_ij = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n]
        
        # Define k-points in the irreducible Brillouin zone
        # High symmetry points: Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # Path: Gamma -> X -> M -> Gamma
        # We use normalized k coordinates where the BZ is [0, 0.5] x [0, 0.5]
        # Gamma: (0, 0)
        # X: (0.5, 0)
        # M: (0.5, 0.5)
        
        k_points = []
        n_seg = n_kpts // 3
        if n_seg < 2:
            n_seg = 2
            
        # Gamma to X: (t, 0) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        # X to M: (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        # M to Gamma: (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, n_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_gap = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # Build the eigenvalue matrix for this k-point.
            # Equation: sum_{G'} S(G_i - G') * |G' + k|^2 * c_{G'} = omega^2 * c_{G_i}
            # Note: k is in units of 2*pi * k_normalized.
            # G = 2*pi * (m, n).
            # So |G + k|^2 = (2*pi)^2 * |(m, n) + (kx, ky)|^2.
            # The factor (2*pi)^2 cancels out in the ratio, but let's keep it for correctness.
            # Actually, the eigenvalue is (omega a / c)^2.
            # Let's just compute the matrix A where A_ij = S_ij * |(G_j + k)|^2.
            # And solve A c = lambda c.
            # lambda = (omega a / c)^2.
            
            M = np.zeros((n_total, n_total), dtype=complex)
            for i in range(n_total):
                mi, ni = g_indices[i]
                for j in range(n_total):
                    mj, nj = g_indices[j]
                    # G_j = 2*pi * (mj, nj)
                    # k = 2*pi * (kx, ky)
                    # G_j + k = 2*pi * (mj + kx, nj + ky)
                    # |G_j + k|^2 = (2*pi)^2 * ((mj + kx)^2 + (nj + ky)^2)
                    kx_total = mj + kx
                    ky_total = nj + ky
                    k_mag_sq = kx_total**2 + ky_total**2
                    
                    # S(G_i - G_j)
                    gm = mi - mj
                    gn = ni - nj
                    idx_m = gm % N
                    idx_n = gn % N
                    s_val = fft_coeffs[idx_m, idx_n]
                    
                    M[i, j] = s_val * k_mag_sq
            
            # Solve eigenvalue problem
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                # Sort eigenvalues
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                
                # We need bands 4 and 5 (indices 3 and 4)
                # The eigenvalues are (omega * a / c)^2.
                # We want to compare frequencies, so we take sqrt.
                # However, since the ratio is scale-invariant, we can just use the eigenvalues directly?
                # No, the gap is in omega, not omega^2.
                # omega = sqrt(lambda) * c / a.
                # The ratio 2*(w5-w4)/(w5+w4) is invariant to scaling of w.
                # So we can just use sqrt(eigvals).
                
                if len(eigvals) >= 5:
                    w4 = np.sqrt(np.real(eigvals[3]))
                    w5 = np.sqrt(np.real(eigvals[4]))
                    
                    max_omega4 = max(max_omega4, w4)
                    min_omega5 = min(min_omega5, w5)
                    
                    gap = w5 - w4
                    if gap > best_gap:
                        best_gap = gap
                        best_k = (kx, ky)
                        if return_eigvecs:
                            best_vec4 = eigvecs[:, 3]
                            best_vec5 = eigvecs[:, 4]
            except Exception:
                continue
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        if min_omega5 + max_omega4 <= 0:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        score = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        
        if return_eigvecs:
            return score, best_k, best_vec4, best_vec5
        return score

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: cell == cell[::-1] rows, cols, and transpose."""
        cell = np.array(cell)
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        cell = np.maximum(cell, cell[::-1, ::-1])
        return cell.astype(int)

    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # Rod centers in normalized coordinates [-0.5, 0.5]
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        # Vectorized generation for speed
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            # Superellipse distance
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        cell = enforce_d4_symmetry(cell)
        return cell

    def compute_field_intensity(vec, kx, ky):
        """
        Compute |E|^2 on the pixel grid from a plane-wave expansion vector.
        vec : eigenvector in plane-wave basis
        kx, ky : k-point coordinates (normalized)
        """
        half = 6 # Must match n_pw//2 used in solver
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        m_vals = np.array([g[0] for g in g_indices])
        n_vals = np.array([g[1] for g in g_indices])
        
        # Pixel centers in normalized coordinates
        coords = (np.arange(size) + 0.5) / size - 0.5
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        
        intensity = np.zeros((size, size), dtype=float)
        
        # Compute E(r) = sum c_i exp(i (G_i + k) . r)
        # G_i = 2*pi*(m_i, n_i)
        # r = (x, y)
        # Phase = 2*pi * (m_i + kx)*x + 2*pi * (n_i + ky)*y
        
        m_k = m_vals + kx
        n_k = n_vals + ky
        
        # To save memory, compute in chunks
        chunk_size = max(1, 1000000 // (size * size))
        
        for start in range(0, len(g_indices), chunk_size):
            end = min(start + chunk_size, len(g_indices))
            m_chunk = m_k[start:end]
            n_chunk = n_k[start:end]
            c_chunk = vec[start:end]
            
            # Phase shape: (chunk, size, size)
            # phase = 2*pi * (m_chunk[:, None, None] * xx[None, :, :] + n_chunk[:, None, None] * yy[None, :, :])
            # This is heavy. Let's use broadcasting more carefully.
            # xx.ravel() has shape (size*size)
            # m_chunk has shape (chunk)
            # phase = 2*pi * (np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel()))
            # phase has shape (chunk, size*size)
            
            phase = 2.0 * np.pi * (np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel()))
            phase = phase.reshape(end - start, size, size)
            
            e_contrib = np.sum(c_chunk[:, None, None] * np.exp(1j * phase), axis=0)
            intensity += np.abs(e_contrib)**2
        
        return intensity

    # Generate candidate cells
    best_score = -np.inf
    best_cell = None
    
    # Grid search over radius and exponent
    radii = np.linspace(0.08, 0.16, 17)
    exponents = [2.0, 2.5, 3.0, 4.0]
    
    for r in radii:
        if time.time() > deadline - 30:
            break
        for exp in exponents:
            if time.time() > deadline - 20:
                break
            
            cell = make_four_quadrant_cell(r, exp)
            
            # Quick heuristic check: fill fraction
            fill = np.sum(cell) / (size * size)
            
            try:
                score = compute_tm_bands(cell, n_pw=13, n_kpts=10)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell
            except Exception:
                continue
    
    # If no cell was found or all failed, return the default
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
        best_score = compute_tm_bands(best_cell, n_pw=13, n_kpts=10)
    
    # Topological optimization phase
    # Start from the best cell found
    current_cell = best_cell.copy()
    current_score = best_score
    
    # We will try to improve the score by flipping pixels.
    # Strategy:
    # 1. Compute eigenvectors for band 4 and 5 at the k-point with the smallest gap.
    # 2. Compute field intensities |E4|^2 and |E5|^2.
    # 3. Pixels where |E4|^2 is high and |E5|^2 is low should be dielectric (to push band 4 up? or down?).
    #    Actually, adding dielectric increases the local refractive index, which generally lowers the frequency.
    #    So adding dielectric where band 4 has high intensity lowers w4.
    #    Adding dielectric where band 5 has high intensity lowers w5.
    #    We want to increase the gap w5 - w4.
    #    So we want to lower w5 and raise w4? Or lower w4 and raise w5?
    #    Usually, the gap is between a lower band and an upper band.
    #    Lowering the lower band (w4) and raising the upper band (w5) increases the gap.
    #    To raise w4, we should remove dielectric where w4 has high intensity.
    #    To lower w5, we should add dielectric where w5 has high intensity.
    
    # Let's define a metric for pixel flipping:
    # delta_w4 ~ -|E4|^2 * delta_eps
    # delta_w5 ~ -|E5|^2 * delta_eps
    # We want to maximize w5 - w4.
    # delta(w5 - w4) ~ -|E5|^2 delta_eps + |E4|^2 delta_eps = (|E4|^2 - |E5|^2) delta_eps.
    # If we add dielectric (delta_eps > 0), we gain if |E4|^2 > |E5|^2.
    # If we remove dielectric (delta_eps < 0), we gain if |E5|^2 > |E4|^2.
    
    n_iter = 20
    for it in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        # Get eigenvectors
        score, best_k, vec4, vec5 = compute_tm_bands(current_cell, n_pw=13, n_kpts=10, return_eigvecs=True)
        
        if best_k is None:
            break
        
        kx, ky = best_k
        
        # Compute intensities
        I4 = compute_field_intensity(vec4, kx, ky)
        I5 = compute_field_intensity(vec5, kx, ky)
        
        # Normalize intensities
        I4 = I4 / (np.sum(I4) + 1e-12)
        I5 = I5 / (np.sum(I5) + 1e-12)
        
        # Metric for adding dielectric: I4 - I5
        # Metric for removing dielectric: I5 - I4
        # We want to find pixels to flip that maximize the gain.
        
        # Candidate pixels to add (currently 0):
        mask_add = (current_cell == 0)
        # Candidate pixels to remove (currently 1):
        mask_rem = (current_cell == 1)
        
        # Score for adding at pixel p: I4[p] - I5[p]
        # Score for removing at pixel p: I5[p] - I4[p]
        
        # We will try flipping a few pixels at a time.
        # Let's find the best single pixel flip first.
        
        gain_add = np.where(mask_add, I4 - I5, -np.inf)
        gain_rem = np.where(mask_rem, I5 - I4, -np.inf)
        
        # Find the best pixel to add
        idx_add = np.unravel_index(np.argmax(gain_add), gain_add.shape)
        best_gain_add = gain_add[idx_add]
        
        # Find the best pixel to remove
        idx_rem = np.unravel_index(np.argmax(gain_rem), gain_rem.shape)
        best_gain_rem = gain_rem[idx_rem]
        
        # Choose the best flip
        if best_gain_add > best_gain_rem:
            new_cell = current_cell.copy()
            new_cell[idx_add] = 1
            # Enforce symmetry
            new_cell = enforce_d4_symmetry(new_cell)
        else:
            new_cell = current_cell.copy()
            new_cell[idx_rem] = 0
            # Enforce symmetry
            new_cell = enforce_d4_symmetry(new_cell)
        
        # Evaluate the new cell
        new_score = compute_tm_bands(new_cell, n_pw=13, n_kpts=10)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            best_cell = current_cell
            best_score = current_score
        else:
            # If the flip didn't help, maybe try a different one?
            # For simplicity, we stop if the best flip doesn't help.
            # Or we could try the second best.
            # Let's just break to save time.
            break
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
