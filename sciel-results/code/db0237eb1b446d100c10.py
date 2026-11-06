"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    start_time = time.time()
    deadline = start_time + 220.0  # safety margin before 230s limit
    
    size = problem['pixels']  # 32
    eps_low = problem['eps_low']    # 1
    eps_high = problem['eps_high']  # 12
    lower_band = problem['lower_band']  # 4
    
    # We will use a local search over the independent pixels of a square-symmetric cell.
    # The square symmetry group (D4) has 8 elements. For a 32x32 cell, the fundamental domain
    # is roughly 1/8 of the pixels. We define independent pixels as those in the first octant
    # and enforce symmetry.
    
    # Define the mapping from independent pixel indices to all symmetric positions
    # For square symmetry about the center, pixel (i, j) maps to:
    # (i, j), (i, 31-j), (31-i, j), (31-i, 31-j),
    # (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)
    
    # We'll pick a set of independent pixels. A good choice is the upper-left triangle
    # including the diagonal, but we need to be careful about the center.
    # For 32x32, the center is between pixels (15,15), (15,16), (16,15), (16,16).
    # Let's use the region i <= j and i <= 15 - j + 1? No, let's just use a simple set.
    
    # Independent pixels: i in 0..15, j in i..31-i
    # Wait, if i=0, j can be 0..31. But (0, j) and (0, 31-j) are symmetric.
    # So for i=0, j should be 0..15.
    # For i=1, j should be 1..15? No, (1, j) and (1, 31-j) are symmetric. Also (j, 1) and (j, 30) are symmetric.
    # The fundamental domain for D4 symmetry on a square grid is the set of pixels (i, j) 
    # such that i <= j and i <= 31 - j. This is a triangular region in the upper left.
    
    independent_pixels = []
    for i in range(size // 2 + 1):  # 0..16
        for j in range(i, size - i):  # i..31-i
            # Check if (i,j) is in the fundamental domain
            # The condition i <= j and i <= 31 - j ensures we don't double count
            # But we also need to handle the case where i > 15
            # Actually, for i=16, j ranges from 16 to 15, which is empty.
            # So i only goes up to 15.
            pass
    
    # Let's redefine independent pixels more carefully.
    # The D4 group orbits on the 32x32 grid.
    # A pixel (i, j) is equivalent to (j, i), (i, 31-j), (31-i, j), etc.
    # The fundamental domain is the set of pixels where i <= j and i <= 31 - j.
    # This means i <= min(j, 31-j).
    
    independent_pixels = []
    for i in range(size // 2):  # 0..15
        for j in range(i, size - i):  # i..31-i
            independent_pixels.append((i, j))
    
    # The center 4 pixels (15,15), (15,16), (16,15), (16,16) form their own orbit of size 4.
    # We need to handle them separately. Let's add one representative.
    independent_pixels.append((15, 15))
    
    # Now, for each independent pixel, we know all its symmetric equivalents.
    # We'll create a mapping from independent index to all positions.
    sym_map = {}
    for idx, (i, j) in enumerate(independent_pixels):
        positions = set()
        # All 8 symmetric positions
        positions.add((i, j))
        positions.add((i, 31 - j))
        positions.add((31 - i, j))
        positions.add((31 - i, 31 - j))
        positions.add((j, i))
        positions.add((j, 31 - i))
        positions.add((31 - j, i))
        positions.add((31 - j, 31 - i))
        sym_map[idx] = list(positions)
    
    # Initial design: start with a reasonable guess.
    # Let's start with a dielectric rod in the center (air background).
    # Or maybe air holes in dielectric background.
    # For TM polarization, dielectric rods in air often give better gaps.
    # Let's start with a circular dielectric rod of radius ~0.3 in the center.
    
    # Create initial independent pixel values
    init_vals = np.zeros(len(independent_pixels), dtype=np.int8)
    for idx, (i, j) in enumerate(independent_pixels):
        # Distance from center (15.5, 15.5)
        dx = i - 15.5
        dy = j - 15.5
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 0.3 * 16:  # radius ~0.3 in units of half-cell
            init_vals[idx] = 1
    
    # Function to expand independent pixels to full 32x32 cell
    def expand_cell(vals):
        cell = np.zeros((size, size), dtype=np.int8)
        for idx, pos_list in sym_map.items():
            val = vals[idx]
            for (i, j) in pos_list:
                cell[i, j] = val
        return cell
    
    # Function to compute the gap using a fast plane-wave method
    def compute_gap(cell):
        # Use a simplified plane-wave method for speed
        # We'll use a smaller basis set for speed, but this is a proxy.
        # For the actual evaluation, the evaluator uses 289 plane waves.
        # We need a fast proxy that correlates with the actual score.
        
        # Convert cell to epsilon array
        eps = np.where(cell == 1, eps_high, eps_low)
        
        # Compute Fourier coefficients of 1/epsilon for TM mode
        # For TM, we need 1/epsilon in the eigenvalue problem
        inv_eps = 1.0 / eps
        
        # Use FFT to compute Fourier coefficients
        # The plane-wave expansion: 1/eps(r) = sum_G (1/eps)_G exp(iG.r)
        # (1/eps)_G = (1/N^2) sum_n (1/eps)_n exp(-iG.r_n)
        
        # For speed, use a smaller grid for the FFT
        N = 32
        G_max = 4  # Use a small number of G vectors for speed
        
        # Generate G vectors
        G_vectors = []
        for gx in range(-G_max, G_max + 1):
            for gy in range(-G_max, G_max + 1):
                G_vectors.append((gx, gy))
        
        num_G = len(G_vectors)
        
        # Compute Fourier coefficients
        # (1/eps)_G = (1/N^2) sum_{n_x, n_y} (1/eps)_{n_x, n_y} exp(-2pi i (G_x n_x + G_y n_y) / N)
        # This is just the 2D FFT of inv_eps, scaled appropriately
        
        # Use numpy FFT
        fft_inv_eps = np.fft.fft2(inv_eps)
        # The FFT gives sum_n f_n exp(-2pi i (k_x n_x + k_y n_y) / N)
        # So (1/eps)_G = fft_inv_eps[G_x + N, G_y + N] / N^2
        
        # Build the eigenvalue matrix
        # For TM: (G + k)^2 / (1/eps)_{G-G'} |E_{G'}| = omega^2 |E_G|
        # Actually, the matrix is: A_{G,G'} = (G+k)^2 * (1/eps)_{G-G'}
        # Where (1/eps)_{G-G'} is the Fourier coefficient of 1/eps at G-G'
        
        # We need to solve for eigenvalues at various k-points
        # For speed, we'll only check a few k-points
        
        k_points = [
            (0, 0),        # Gamma
            (0.5, 0),      # X
            (0.5, 0.5),    # M
            (0.25, 0),     # Interior
            (0.25, 0.25),  # Interior
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for (kx, ky) in k_points:
            # Build matrix
            A = np.zeros((num_G, num_G), dtype=np.complex128)
            
            for i, (Gx_i, Gy_i) in enumerate(G_vectors):
                for j, (Gx_j, Gy_j) in enumerate(G_vectors):
                    # G_diff = G_i - G_j
                    Gx_diff = Gx_i - Gx_j
                    Gy_diff = Gy_i - Gy_j
                    
                    # Get Fourier coefficient (1/eps)_{G_diff}
                    # Index in FFT array: (Gx_diff + N) % N, (Gy_diff + N) % N
                    idx_x = (Gx_diff + N) % N
                    idx_y = (Gy_diff + N) % N
                    coeff = fft_inv_eps[idx_x, idx_y] / (N * N)
                    
                    # (G + k)^2
                    Gx_k = Gx_i + kx
                    Gy_k = Gy_i + ky
                    G_k_sq = Gx_k * Gx_k + Gy_k * Gy_k
                    
                    A[i, j] = G_k_sq * coeff
            
            # Solve eigenvalue problem
            # For TM, the matrix is not Hermitian in general, but we can use eigh
            # Actually, for real epsilon, the matrix should be Hermitian if we set it up correctly
            # Let's use eigh for speed and stability
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.real(eigvals)
                # Filter out negative or zero eigenvalues
                eigvals = eigvals[eigvals > 0]
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    # omega is proportional to sqrt(eigenvalue)
                    # But we just need the ratio, so we can work with eigenvalues directly
                    # The gap-midgap ratio is 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
                    # Since omega = sqrt(eigval) * constant, the ratio is the same
                    # if we use sqrt(eigval)
                    omega = np.sqrt(eigvals)
                    
                    if len(omega) >= 5:
                        max_omega_4 = max(max_omega_4, omega[3])  # 4th band (index 3)
                        min_omega_5 = min(min_omega_5, omega[4])  # 5th band (index 4)
            except:
                pass
        
        if min_omega_5 == np.inf:
            return -1.0
        
        gap = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap
    
    # Local search
    best_vals = init_vals.copy()
    best_score = compute_gap(expand_cell(best_vals))
    
    # Hill climbing: try flipping individual pixels
    num_independent = len(independent_pixels)
    max_iterations = 5000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try flipping a random pixel
        idx = np.random.randint(num_independent)
        old_val = best_vals[idx]
        best_vals[idx] = 1 - old_val
        
        cell = expand_cell(best_vals)
        score = compute_gap(cell)
        
        if score > best_score:
            best_score = score
        else:
            # Revert
            best_vals[idx] = old_val
    
    # Convert to nested list
    final_cell = expand_cell(best_vals)
    result = final_cell.tolist()
    
    return result

# EVOLVE-BLOCK-END
