import numpy as np
import time
import random

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for final checks

    # --- PWE Solver ---
    
    # Precompute G vectors for N=7 (49 plane waves)
    # The evaluator uses 289 (17x17), but we can use fewer for optimization.
    # Let's use N=9 (81 plane waves) for better accuracy in surrogate, if time permits.
    # Or stick to N=7 for speed. Let's try N=9.
    n_pw = 9
    half = n_pw // 2
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors)
    # Sort by magnitude to ensure lowest frequencies are found? 
    # Actually, for PWE, the order doesn't strictly matter for the eigenvalue problem 
    # as long as we take the lowest N eigenvalues. But standard practice is to include 
    # low G first. The set is the same.
    
    # Precompute pixel coordinates
    # Pixels are centered in the cell? 
    # The problem says "32x32-pixel unit cell".
    # Usually, pixel (i,j) covers [i/32, (i+1)/32] x [j/32, (j+1)/32].
    # The center of pixel (i,j) is ((i+0.5)/32, (j+0.5)/32).
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, ys)
    pixel_coords = np.column_stack([X.ravel(), Y.ravel()]) # Shape (1024, 2)

    def get_fourier_coeffs(cell):
        """
        Computes Fourier coefficients of 1/epsilon(r).
        cell: 32x32 array of 0/1.
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def fft_index(g, N):
        """Map integer G component to FFT array index (0..N-1)."""
        return g % N

    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves eigenvalue problem for TM bands.
        Returns sorted omega^2 and optionally eigenvectors.
        """
        fft_coeffs = get_fourier_coeffs(cell)
        
        # Build M matrix: M[i,j] = fft_coeffs[G_i - G_j]
        # G_vectors shape (N, 2)
        # We need G_i - G_j for all i, j.
        # G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :] # (N, N, 2)
        # This creates a large array. N=81 -> 81*81*2 = 13k floats. Fine.
        
        G_diff_x = G_vectors[:, 0][:, np.newaxis] - G_vectors[:, 0][np.newaxis, :]
        G_diff_y = G_vectors[:, 1][:, np.newaxis] - G_vectors[:, 1][np.newaxis, :]
        
        # Map to FFT indices
        idx_x = G_diff_x % size
        idx_y = G_diff_y % size
        
        M = fft_coeffs[idx_x, idx_y]
        
        # K^2
        Kx = kx + G_vectors[:, 0]
        Ky = ky + G_vectors[:, 1]
        K_sq = Kx**2 + Ky**2
        
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
                return np.zeros(n_pw*n_pw), np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
            return np.zeros(n_pw*n_pw)

    def compute_field_intensities(cell, k_points, band_idx):
        """
        Computes averaged field intensity for a given band index (0-based).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
            # Compute field at all pixels
            # Phase = 2 * pi * (Kx * X + Ky * Y)
            # Kx = kx + Gx, Ky = ky + Gy
            Kx = kx + G_vectors[:, 0]
            Ky = ky + G_vectors[:, 1]
            
            # pixel_coords shape (1024, 2)
            # K_vectors shape (N, 2)
            # Dot product: (1024, N)
            phases = 2 * np.pi * (pixel_coords[:, 0:1] * Kx[None, :] + pixel_coords[:, 1:2] * Ky[None, :])
            
            # Field = sum(vec[i] * exp(i * phase[i]))
            # vec shape (N,)
            # exp_phases shape (1024, N)
            exp_phases = np.exp(1j * phases)
            field = exp_phases @ vec
            
            intensity = np.abs(field)**2
            total_intensity += intensity.reshape((size, size))
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity

    def estimate_gap(cell, k_points):
        """
        Estimates the gap.
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

    # --- Symmetry Handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        Symmetry: 180 deg rotation and mirror.
        Actually, square symmetry group C4v.
        The unique region is 1/8 of the cell?
        For 32x32, the fundamental domain for C4v is 16x16?
        Let's check.
        (i, j) -> (31-i, j) [Mirror Y]
        (i, j) -> (i, 31-j) [Mirror X]
        (i, j) -> (31-i, 31-j) [Rotation 180]
        (i, j) -> (j, i) [Mirror Diagonal]
        
        If we define unique_mask for i in [0, 15] and j in [0, 15], 
        and enforce cell[i][j] == cell[j][i] == cell[31-i][j] == ...
        This reduces the degrees of freedom to 16x16 = 256 pixels.
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
    
    # K-points for surrogate
    # Include Gamma, X, M and some interior points
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
        (0.45, 0.45),
        (0.05, 0.05),
    ]
    
    # Initialize with a structured pattern
    # Try a few initializations and pick the best
    initializations = []
    
    # 1. Circular hole
    um1 = np.zeros((16, 16), dtype=int)
    center = 7.5
    radius = 5
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                um1[i][j] = 1
    initializations.append(um1)
    
    # 2. Square hole
    um2 = np.zeros((16, 16), dtype=int)
    for i in range(4, 12):
        for j in range(4, 12):
            um2[i][j] = 1
    initializations.append(um2)
    
    # 3. Cross
    um3 = np.zeros((16, 16), dtype=int)
    for i in range(16):
        um3[i][7] = 1
        um3[i][8] = 1
        um3[7][i] = 1
        um3[8][i] = 1
    initializations.append(um3)

    best_gap = -float('inf')
    best_cell = None
    best_mask = None

    for init_mask in initializations:
        cell = fill_symmetric(init_mask)
        gap = estimate_gap(cell, k_points)
        if gap > best_gap:
            best_gap = gap
            best_cell = cell.copy()
            best_mask = init_mask.copy()
            
    # If we didn't find a good one, use the first one
    if best_cell is None:
        best_mask = initializations[0]
        best_cell = fill_symmetric(best_mask)
        best_gap = estimate_gap(best_cell, k_points)

    current_gap = best_gap
    current_mask = best_mask.copy()
    current_cell = best_cell.copy()
    
    pixel_mapping = get_unique_pixel_indices()
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    max_iterations = 1000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities
        intensity_4 = compute_field_intensities(current_cell, k_points, 3)
        intensity_5 = compute_field_intensities(current_cell, k_points, 4)
        
        pressure = intensity_4 - intensity_5
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
            
        # Find best pixel to flip
        best_pixel = None
        best_pressure = 0
        
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            if abs(avg_pressure) > abs(best_pressure):
                best_pressure = avg_pressure
                best_pixel = (i, j)
                
        if best_pixel is None:
            break
            
        i, j = best_pixel
        
        # Determine target value
        if best_pressure > 0:
            new_val = 1
        else:
            new_val = 0
            
        # If current value is already new_val, try the other one? 
        # Or just pick a random pixel?
        if current_mask[i][j] == new_val:
            # Try a random pixel instead
            ri = random.randint(0, 15)
            rj = random.randint(0, 15)
            i, j = ri, rj
            new_val = 1 - current_mask[i][j]
            
        # Apply change
        current_mask[i][j] = new_val
        new_cell = fill_symmetric(current_mask)
        new_gap = estimate_gap(new_cell, k_points)
        
        delta = new_gap - current_gap
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or random.random() < np.exp(delta / T):
            current_gap = new_gap
            current_cell = new_cell
            if new_gap > best_gap:
                best_gap = new_gap
                best_cell = new_cell.copy()
                best_mask = current_mask.copy()
        else:
            # Revert
            current_mask[i][j] = 1 - new_val
            current_cell = fill_symmetric(current_mask)
            
        T *= cooling_rate
        
        # Random restart every 100 iterations?
        if iteration % 100 == 99:
            # Perturb current solution
            for _ in range(5):
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                current_mask[ri][rj] = 1 - current_mask[ri][rj]
            current_cell = fill_symmetric(current_mask)
            current_gap = estimate_gap(current_cell, k_points)

    return best_cell.tolist()
