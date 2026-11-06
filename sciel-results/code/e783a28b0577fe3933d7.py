import numpy as np
import time
import random

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 100  # Leave 20s buffer for final checks and overhead
    
    # Precompute G vectors for n_pw=7 (49 plane waves)
    n_pw = 7
    half = n_pw // 2
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
    G_vectors = G_list[:n_pw * n_pw]
    n_pw_count = len(G_vectors)
    
    # Precompute K^2 offsets for each G vector
    # K = k + G. We need (kx+gx)^2 + (ky+gy)^2
    # We will compute this on the fly for each k, but we can precompute G^2
    G_sq = np.array([g[0]**2 + g[1]**2 for g in G_vectors])
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def fft_index(g, N):
        return g if g >= 0 else N + g

    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        """
        fft_coeffs = get_fourier_coeffs(cell)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        # We can vectorize this slightly or just loop, 49x49 is small
        M = np.zeros((n_pw_count, n_pw_count), dtype=complex)
        for i in range(n_pw_count):
            gx_i, gy_i = G_vectors[i]
            for j in range(n_pw_count):
                gx_j, gy_j = G_vectors[j]
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                idx_x = fft_index(Gx_diff, size)
                idx_y = fft_index(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        K_sq = np.zeros(n_pw_count)
        for i in range(n_pw_count):
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
                return [0] * n_pw_count, np.zeros((n_pw_count, n_pw_count), dtype=complex)
            return [0] * n_pw_count

    def compute_field_intensities(cell, k_points, band_idx):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        Vectorized over pixels.
        """
        total_intensity = np.zeros((size, size))
        count = 0
        
        # Precompute pixel coordinates
        pxs = np.arange(size)
        pys = np.arange(size)
        RX, RY = np.meshgrid(pxs / size, pys / size, indexing='ij')
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
            # Compute field for all pixels at once
            # Field(r) = sum_i vec_i * exp(i (k+G_i) . r)
            # We need to sum over i.
            # Shape of vec: (n_pw_count,)
            # We can create a matrix of phases: (n_pw_count, size, size)
            
            # Kx_i, Ky_i for each i
            Kx_arr = np.array([kx + g[0] for g in G_vectors])
            Ky_arr = np.array([ky + g[1] for g in G_vectors])
            
            # Phase for each i and pixel (px, py)
            # Phase[i, px, py] = 2*pi * (Kx[i]*RX[px,py] + Ky[i]*RY[px,py])
            # This creates a large array (49, 32, 32) which is ~50k floats, manageable.
            Phases = 2 * np.pi * (Kx_arr[:, np.newaxis, np.newaxis] * RX[np.newaxis, :, :] + 
                                   Ky_arr[:, np.newaxis, np.newaxis] * RY[np.newaxis, :, :])
            
            # Complex exponentials
            ExpPhases = np.exp(1j * Phases)
            
            # Field = sum_i vec[i] * ExpPhases[i]
            # vec shape: (49, 1, 1)
            Field = np.sum(vec[:, np.newaxis, np.newaxis] * ExpPhases, axis=0)
            
            total_intensity += np.abs(Field)**2
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity

    def estimate_gap(cell, k_points):
        """
        Estimates the gap between band 4 and 5.
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky)
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
    
    # Define k-points for the surrogate (denser than before)
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 0.5
    # We sample a grid in this triangle
    k_points = []
    for i in range(6):
        for j in range(6):
            kx = i * 0.5 / 5
            ky = j * 0.5 / 5
            if kx <= ky and ky <= 0.5:
                k_points.append((kx, ky))
    # Add critical points explicitly if not covered
    if (0.0, 0.0) not in k_points: k_points.append((0.0, 0.0))
    if (0.5, 0.0) not in k_points: k_points.append((0.5, 0.0))
    if (0.5, 0.5) not in k_points: k_points.append((0.5, 0.5))
    
    # Initialize with a structured pattern
    # Common good starting point for large gaps: Air holes in dielectric
    # Or dielectric rods in air.
    # Let's try a pattern with high contrast features.
    unique_mask = np.zeros((16, 16), dtype=int)
    
    # Try a few random initializations and pick the best
    best_init_mask = None
    best_init_gap = -float('inf')
    
    for _ in range(5):
        # Random initialization with some structure
        mask = np.zeros((16, 16), dtype=int)
        # Create a few random blobs
        for _ in range(3):
            cx = random.randint(2, 13)
            cy = random.randint(2, 13)
            r = random.randint(2, 4)
            for i in range(16):
                for j in range(16):
                    if (i-cx)**2 + (j-cy)**2 < r**2:
                        mask[i][j] = 1
        cell = fill_symmetric(mask)
        gap = estimate_gap(cell, k_points)
        if gap > best_init_gap:
            best_init_gap = gap
            best_init_mask = mask.copy()
            
    unique_mask = best_init_mask
    cell = fill_symmetric(unique_mask)
    current_gap = best_init_gap
    best_gap = current_gap
    best_cell = cell.copy()
    
    pixel_mapping = get_unique_pixel_indices()
    unique_coords_list = list(pixel_mapping.items())
    
    # Topological optimization loop
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        intensity_4 = compute_field_intensities(cell, k_points, 3)
        intensity_5 = compute_field_intensities(cell, k_points, 4)
        
        # Pressure: positive -> add dielectric (1), negative -> remove (0)
        pressure = intensity_4 - intensity_5
        
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Find best pixel to flip
        best_pixel = None
        best_pressure_val = 0
        
        for (i, j), coords in unique_coords_list:
            avg_pressure = np.mean(pressure[coords])
            if abs(avg_pressure) > abs(best_pressure_val):
                best_pressure_val = avg_pressure
                best_pixel = (i, j)
        
        if best_pixel is None or abs(best_pressure_val) < 0.01:
            # Random move
            i, j = random.randint(0, 15), random.randint(0, 15)
            best_pixel = (i, j)
            # Decide direction randomly or based on current value
            if random.random() < 0.5:
                new_val = 1
            else:
                new_val = 0
        else:
            i, j = best_pixel
            if best_pressure_val > 0:
                new_val = 1
            else:
                new_val = 0
        
        # If the target value is the same as current, skip or force flip?
        # If pressure says "add" but it's already 1, maybe we should look for another pixel.
        # But for simplicity, if it's the same, we skip this move (or try opposite).
        if unique_mask[i][j] == new_val:
             # Try the opposite value if the pressure is weak or to explore
             if abs(best_pressure_val) > 0.5:
                 continue # Strong signal, don't flip against it
             else:
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
            if new_gap > best_gap:
                best_gap = new_gap
                best_cell = new_cell.copy()
        else:
            # Revert
            unique_mask[i][j] = 1 - new_val
            cell = fill_symmetric(unique_mask)
            
        # Random perturbation every 10 steps
        if iteration % 10 == 9:
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

    return best_cell.tolist()
