import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using topological optimization driven by
    local field intensities of the 4th and 5th TM bands.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # --- PWE Solver with eigenvectors ---
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        1 -> dielectric (eps_high), 0 -> air (eps_low).
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs, inv_eps_map

    def get_G_vectors(n_pw):
        """
        Returns G vectors (as integer multiples of 2pi/a) sorted by magnitude.
        """
        G_list = []
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
        return G_list[:n_pw * n_pw]

    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g if g >= 0 else N + g

    def solve_band_structure(cell, kx, ky, n_pw=7, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        """
        fft_coeffs, _ = get_fourier_coeffs(cell)
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            for j in range(n):
                gx_j, gy_j = G_vectors[j]
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                idx_x = fft_index(Gx_diff, size)
                idx_y = fft_index(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        K_sq = np.zeros(n)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
        
        # A = diag(K_sq) - M
        A = np.diag(K_sq) - M
        
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            idx_sort = np.argsort(eigvals)
            eigvals = eigvals[idx_sort]
            if return_eigvecs:
                eigvecs = eigvecs[:, idx_sort]
                return eigvals, eigvecs
            return eigvals
        except np.linalg.LinAlgError:
            if return_eigvecs:
                return [0] * n, np.zeros((n, n), dtype=complex)
            return [0] * n

    def compute_field_intensities(cell, k_points, band_indices, n_pw=7):
        """
        Computes the local field intensity at each pixel for given bands
        across multiple k-points, averaged.
        Returns a list of 2D arrays, one per band index.
        """
        total_intensities = [np.zeros((size, size)) for _ in band_indices]
        count = 0
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Precompute pixel positions
        px_range = np.arange(size)
        ry, rx = np.meshgrid(px_range / size, px_range / size, indexing='ij')
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= max(band_indices):
                continue
            
            for idx, band_idx in enumerate(band_indices):
                vec = eigvecs[:, band_idx]
                field = np.zeros((size, size), dtype=complex)
                for i in range(n):
                    gx_i, gy_i = G_vectors[i]
                    Kx = kx + gx_i
                    Ky = ky + gy_i
                    phase = 2 * np.pi * (Kx * rx + Ky * ry)
                    field += vec[i] * np.exp(1j * phase)
                total_intensities[idx] += np.abs(field)**2
            count += 1
        
        if count > 0:
            for idx in range(len(band_indices)):
                total_intensities[idx] /= count
        return total_intensities

    def estimate_gap(cell, k_points, n_pw=7):
        """
        Estimates the gap between band 4 and 5 (0-indexed: 3 and 4).
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky, n_pw)
            if len(eigvals) < 5:
                continue
            w4 = np.sqrt(max(0, eigvals[3]))
            w5 = np.sqrt(max(0, eigvals[4]))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0
        gap = min_omega_5 - max_omega_4
        return gap

    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
                coords = [
                    (i, j),
                    (31-i, j),
                    (i, 31-j),
                    (31-i, 31-j),
                    (j, i),
                    (31-j, i),
                    (j, 31-i),
                    (31-j, 31-i)
                ]
                for r, c in coords:
                    cell[r][c] = val
        return cell

    def get_unique_pixel_indices():
        """
        Returns a mapping from unique pixel (i,j) in [0,15]x[0,15] 
        to all symmetric positions in the full cell.
        """
        mapping = {}
        for i in range(16):
            for j in range(16):
                coords = [
                    (i, j),
                    (31-i, j),
                    (i, 31-j),
                    (31-i, 31-j),
                    (j, i),
                    (31-j, i),
                    (j, 31-i),
                    (31-j, 31-i)
                ]
                unique_coords = list(set(coords))
                mapping[(i, j)] = unique_coords
        return mapping

    # --- Optimization ---
    
    # More comprehensive k-points for better surrogate
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.1),
        (0.4, 0.1),
        (0.1, 0.4),
        (0.3, 0.2),
        (0.2, 0.3),
        (0.15, 0.0),
        (0.35, 0.0),
        (0.0, 0.15),
        (0.0, 0.35),
        (0.45, 0.45),
        (0.55, 0.45),
        (0.45, 0.55),
        (0.55, 0.55),
        (0.2, 0.4),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.2, 0.2),
        (0.4, 0.4),
        (0.1, 0.3),
        (0.3, 0.1),
        (0.25, 0.15),
        (0.15, 0.25),
        (0.35, 0.25),
        (0.25, 0.35),
        (0.4, 0.3),
    ]
    
    pixel_mapping = get_unique_pixel_indices()
    
    # Multi-start optimization with different initializations
    best_gap = -float('inf')
    best_cell = None
    
    # Try several different initialization strategies
    initializations = []
    
    # 1. Circular dielectric inclusion (original)
    mask1 = np.zeros((16, 16), dtype=int)
    center = 7.5
    for r in range(16):
        for c in range(16):
            dist = np.sqrt((r - center)**2 + (c - center)**2)
            if dist < 5:
                mask1[r][c] = 1
    initializations.append(mask1.copy())
    
    # 2. Circular air hole in dielectric
    mask2 = np.ones((16, 16), dtype=int)
    for r in range(16):
        for c in range(16):
            dist = np.sqrt((r - center)**2 + (c - center)**2)
            if dist < 4:
                mask2[r][c] = 0
    initializations.append(mask2.copy())
    
    # 3. Random initialization
    mask3 = np.random.randint(0, 2, (16, 16), dtype=int)
    initializations.append(mask3.copy())
    
    # 4. Checkerboard-like pattern
    mask4 = np.zeros((16, 16), dtype=int)
    for r in range(16):
        for c in range(16):
            if (r // 2 + c // 2) % 2 == 0:
                mask4[r][c] = 1
    initializations.append(mask4.copy())
    
    # 5. Ring pattern
    mask5 = np.zeros((16, 16), dtype=int)
    for r in range(16):
        for c in range(16):
            dist = np.sqrt((r - center)**2 + (c - center)**2)
            if 3 < dist < 6:
                mask5[r][c] = 1
    initializations.append(mask5.copy())
    
    # 6. Cross pattern
    mask6 = np.zeros((16, 16), dtype=int)
    for r in range(16):
        for c in range(16):
            if r in [6, 7, 8, 9] or c in [6, 7, 8, 9]:
                mask6[r][c] = 1
    initializations.append(mask6.copy())
    
    # Time budget per initialization
    num_inits = len(initializations)
    time_per_init = (time_limit - (time.time() - start_time)) / num_inits
    
    for init_idx, init_mask in enumerate(initializations):
        if time.time() - start_time > time_limit:
            break
        
        unique_mask = init_mask.copy()
        cell = fill_symmetric(unique_mask)
        current_gap = estimate_gap(cell, k_points)
        
        if current_gap > best_gap:
            best_gap = current_gap
            best_cell = cell.copy()
        
        # Topological optimization for this initialization
        max_iters = 80
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Compute field intensities for bands 4 and 5
            intensities = compute_field_intensities(cell, k_points[:15], [3, 4])
            intensity_4 = intensities[0]
            intensity_5 = intensities[1]
            
            # Pressure field: positive means add dielectric, negative means remove
            pressure = intensity_4 - intensity_5
            
            # Normalize
            p_max = np.max(np.abs(pressure))
            if p_max > 1e-10:
                pressure /= p_max
            
            # Try flipping multiple pixels per iteration for faster convergence
            num_flips = 3
            accepted_any = False
            
            for _ in range(num_flips):
                if time.time() - start_time > time_limit:
                    break
                
                # Find pixel with highest absolute pressure
                best_pixel = None
                best_pressure = 0
                
                for (i, j), coords in pixel_mapping.items():
                    avg_pressure = np.mean(pressure[coords])
                    if abs(avg_pressure) > abs(best_pressure):
                        best_pressure = avg_pressure
                        best_pixel = (i, j)
                
                if best_pixel is None or abs(best_pressure) < 0.03:
                    i = random.randint(0, 15)
                    j = random.randint(0, 15)
                    best_pixel = (i, j)
                    best_pressure = pressure[i, j]
                
                i, j = best_pixel
                
                # Decide direction
                if best_pressure > 0:
                    new_val = 1
                else:
                    new_val = 0
                
                # Only flip if different
                if unique_mask[i][j] == new_val:
                    new_val = 1 - new_val
                    if unique_mask[i][j] == new_val:
                        continue
                
                # Apply change
                unique_mask[i][j] = new_val
                new_cell = fill_symmetric(unique_mask)
                new_gap = estimate_gap(new_cell, k_points)
                
                # Accept if better
                if new_gap > current_gap:
                    current_gap = new_gap
                    cell = new_cell
                    accepted_any = True
                    if new_gap > best_gap:
                        best_gap = new_gap
                        best_cell = new_cell.copy()
                else:
                    # Revert
                    unique_mask[i][j] = 1 - new_val
                    cell = fill_symmetric(unique_mask)
            
            # If no flips accepted, do a random perturbation
            if not accepted_any:
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
                new_cell = fill_symmetric(unique_mask)
                new_gap = estimate_gap(new_cell, k_points)
                if new_gap > current_gap:
                    current_gap = new_gap
                    cell = new_cell
                    if new_gap > best_gap:
                        best_gap = new_gap
                        best_cell = new_cell.copy()
                else:
                    unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
                    cell = fill_symmetric(unique_mask)
    
    # Final check
    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()

