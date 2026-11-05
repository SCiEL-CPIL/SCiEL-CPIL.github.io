"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using vectorized matrix construction.
    2. Use topological optimization: start from a good geometric seed,
       then iteratively flip pixels based on local field intensity of bands 4 and 5.
    3. Maintain D4 symmetry at each step.
    4. Stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 195.0  # safety margin

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=15, return_eigvecs=False):
        """
        Compute TM band structure using plane-wave expansion.
        Returns min_omega5 and max_omega4 over the BZ path.
        If return_eigvecs, also returns eigenvectors for the 4th and 5th bands
        at the k-point where the gap is smallest.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size
        
        # Get Fourier coefficients of inv_eps via FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Plane wave indices
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Precompute S matrix: S[i,j] = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            mi, ni = g_indices[i]
            for j in range(n_total):
                mj, nj = g_indices[j]
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n]
        
        # k-points along the irreducible BZ boundary
        k_points = []
        n_seg = n_kpts // 3
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0.0, n_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_unique = []
        for kp in k_points:
            if not any(abs(kp[0]-k2[0]) < 1e-10 and abs(kp[1]-k2[1]) < 1e-10 for k2 in k_unique):
                k_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        best_gap = -np.inf
        
        for (kx, ky) in k_unique:
            Gk = np.array([(m + kx, n + ky) for m, n in g_indices])
            Gk_sq = Gk[:, 0]**2 + Gk[:, 1]**2
            
            A = S * Gk_sq[None, :]
            B = S
            
            try:
                from scipy.linalg import eig as sp_eig
                vals, vecs = sp_eig(A, B)
                vals = np.real(vals)
                valid = vals > 1e-6
                vals = vals[valid]
                vecs = vecs[:, valid]
                order = np.argsort(vals)
                vals = vals[order]
                vecs = vecs[:, order]
            except:
                try:
                    B_inv_A = np.linalg.solve(B, A)
                    vals = np.linalg.eigvals(B_inv_A)
                    vals = np.real(vals)
                    valid = vals > 1e-6
                    vals = vals[valid]
                    order = np.argsort(vals)
                    vals = vals[order]
                    vecs = None
                except:
                    continue
            
            if len(vals) < 5:
                continue
            
            omega4 = vals[3]
            omega5 = vals[4]
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
            
            gap = omega5 - omega4
            if gap > best_gap:
                best_gap = gap
                best_k = (kx, ky)
                if vecs is not None:
                    best_vec4 = vecs[:, 3]
                    best_vec5 = vecs[:, 4]
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            return -1.0, None, None, None
        
        if min_omega5 + max_omega4 <= 0:
            return -1.0, None, None, None
        
        score = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        
        if return_eigvecs:
            return score, best_k, best_vec4, best_vec5
        return score, None, None, None
    
    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: cell == cell[::-1] rows, cols, and transpose."""
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        cell = np.maximum(cell, cell[::-1, ::-1])
        return cell
    
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """Create a cell with four rods centered in each quadrant."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        for (cx, cy) in centers:
            dx = xx - cx
            dy = yy - cy
            if rod_radius > 0:
                val = np.abs(dx / rod_radius)**exponent + np.abs(dy / rod_radius)**exponent
            else:
                val = np.zeros_like(xx)
            cell[val <= 1.0 + 1e-9] = 1
        
        cell = enforce_d4_symmetry(cell)
        return cell.tolist()
    
    def field_intensity_from_vec(vec, g_indices, kx, ky, size):
        """Compute |E|^2 on the pixel grid from a plane-wave expansion vector."""
        # vec is the eigenvector in plane-wave basis
        # E(r) = sum_i c_i exp(i (G_i + k) . r)
        # We compute |E|^2 at pixel centers
        half = len(g_indices)
        m_vals = np.array([g[0] for g in g_indices])
        n_vals = np.array([g[1] for g in g_indices])
        
        # Pixel centers in normalized coordinates
        coords = (np.arange(size) + 0.5) / size - 0.5
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        
        # Compute phase for each plane wave at each pixel
        # phase[i, px, py] = (m_i + kx) * xx[px,py] + (n_i + ky) * yy[px,py]
        # This is memory intensive, so do it in chunks
        intensity = np.zeros((size, size), dtype=float)
        chunk_size = max(1, 1000000 // (size * size))
        
        for start in range(0, len(g_indices), chunk_size):
            end = min(start + chunk_size, len(g_indices))
            m_chunk = m_vals[start:end] + kx
            n_chunk = n_vals[start:end] + ky
            c_chunk = vec[start:end]
            
            # phase has shape (chunk, size, size)
            phase = np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel())
            phase = phase.reshape(end - start, size, size)
            
            # E contribution
            e_contrib = np.sum(c_chunk[:, None, None] * np.exp(1j * phase), axis=0)
            intensity += np.abs(e_contrib)**2
        
        return intensity
    
    # Start with a good geometric seed
    best_score = -np.inf
    best_candidate = np.array(make_four_quadrant_cell(0.12, 2.0))
    
    # First, do a quick grid search to find a good starting point
    radii = [0.10, 0.11, 0.12, 0.13, 0.14]
    exponents = [2.0, 2.5, 3.0]
    
    for r in radii:
        if time.time() > deadline - 120:
            break
        for e in exponents:
            if time.time() > deadline - 120:
                break
            cell = np.array(make_four_quadrant_cell(r, e))
            score, _, _, _ = compute_tm_bands(cell, return_eigvecs=False)
            if score > best_score:
                best_score = score
                best_candidate = cell
    
    # Now do topological optimization
    # Iteratively flip pixels based on field intensity of bands 4 and 5
    n_iterations = 0
    max_iterations = 50
    
    while time.time() < deadline - 30 and n_iterations < max_iterations:
        n_iterations += 1
        
        # Get eigenvectors at the k-point with largest gap
        score, best_k, vec4, vec5 = compute_tm_bands(best_candidate, return_eigvecs=True)
        
        if best_k is None or vec4 is None or vec5 is None:
            break
        
        kx, ky = best_k
        g_indices = []
        half = 13 // 2
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        # Compute field intensities
        intensity4 = field_intensity_from_vec(vec4, g_indices, kx, ky, size)
        intensity5 = field_intensity_from_vec(vec5, g_indices, kx, ky, size)
        
        # Normalize intensities
        intensity4 = intensity4 / (np.max(intensity4) + 1e-10)
        intensity5 = intensity5 / (np.max(intensity5) + 1e-10)
        
        # Strategy: We want to push band 4 down and band 5 up.
        # For TM modes, higher eps tends to lower the frequency.
        # If band 4 has high intensity at a pixel, adding dielectric there lowers omega4.
        # If band 5 has high intensity at a pixel, removing dielectric there raises omega5.
        
        # Compute a "desirability" score for each pixel
        # Positive means we want to set it to 1 (dielectric)
        # Negative means we want to set it to 0 (air)
        
        # For pixels currently 0: if intensity4 is high, set to 1 (lowers band 4)
        # For pixels currently 1: if intensity5 is high, set to 0 (raises band 5)
        
        desirability = np.zeros((size, size), dtype=float)
        
        # Pixels currently 0: prefer to set to 1 if band 4 intensity is high
        mask0 = (best_candidate == 0)
        desirability[mask0] = 0.5 * intensity4[mask0]
        
        # Pixels currently 1: prefer to set to 0 if band 5 intensity is high
        mask1 = (best_candidate == 1)
        desirability[mask1] = -0.5 * intensity5[mask1]
        
        # Enforce D4 symmetry on desirability
        desirability = np.maximum(desirability, desirability[::-1, :])
        desirability = np.maximum(desirability, desirability[:, ::-1])
        desirability = np.maximum(desirability, desirability.T)
        desirability = np.maximum(desirability, desirability[::-1, ::-1])
        
        # Determine which pixels to flip
        # Flip pixels where desirability is strongly positive (0->1) or negative (1->0)
        threshold = 0.3
        
        new_candidate = best_candidate.copy()
        
        # 0 -> 1 flips
        flip_to_1 = (best_candidate == 0) & (desirability > threshold)
        new_candidate[flip_to_1] = 1
        
        # 1 -> 0 flips
        flip_to_0 = (best_candidate == 1) & (desirability < -threshold)
        new_candidate[flip_to_0] = 0
        
        # Enforce D4 symmetry
        new_candidate = enforce_d4_symmetry(new_candidate)
        
        # Evaluate the new candidate
        new_score, _, _, _ = compute_tm_bands(new_candidate, return_eigvecs=False)
        
        if new_score > best_score:
            best_score = new_score
            best_candidate = new_candidate
        else:
            # If no improvement, reduce the threshold and try again
            # Or break if we're stuck
            if n_iterations > 10 and new_score < best_score - 0.5:
                break
    
    return best_candidate.tolist()

# EVOLVE-BLOCK-END
