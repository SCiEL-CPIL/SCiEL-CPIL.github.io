import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 100.0 # Safety margin
    
    # Initialize with seed: centered rod
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1
                
    # PWM Setup
    # G vectors: -14 to 14 (29x29 = 841? No, 289 plane waves implies 17x17 grid? sqrt(289)=17. -8 to 8)
    # 289 = 17^2. So Gx, Gy in [-8, 8].
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ
    # Irreducible BZ: 0 <= ky <= kx, kx in [0, 1], ky in [0, 1]
    # 30 points for evaluator, we use ~15 for search
    k_points = []
    # Boundary
    for i in range(6):
        k_points.append([i/5.0, 0.0])
        k_points.append([0.5, i/5.0])
        k_points.append([i/5.0, i/5.0])
    # Interior
    k_points.append([0.25, 0.1])
    k_points.append([0.4, 0.2])
    k_points.append([0.3, 0.15])
    k_points = np.array(k_points)
    
    # Precompute G^2
    G_sq = np.sum(G_vectors**2, axis=1)
    
    def get_fourier_eps_inv(cell_arr):
        # cell_arr is 32x32, 0 or 1
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        # FFT gives coefficients for e^{i 2pi n x / N}
        # Our G vectors are integers.
        # fft2(eps_inv) / N^2
        coeffs = np.fft.fft2(eps_inv) / (size * size)
        # coeffs[i, j] corresponds to Gx = i - N/2, Gy = j - N/2
        # We need to map G_vectors to this grid
        # G_vectors are -8 to 8. Grid is 0 to 31.
        # Offset: +16 (size/2)
        offset = size // 2
        Gx_grid = G_vectors[:, 0] + offset
        Gy_grid = G_vectors[:, 1] + offset
        return coeffs[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            # Build H matrix
            # H[i, j] = eps_inv_coeffs[i - j] * (k + G_j)^2
            # Note: eps_inv_coeffs is indexed by G_i - G_j
            # We need to construct the difference matrix
            # G_diff = G_i - G_j
            # This is O(N^3) if done naively. 
            # Better: For each i, we need eps_inv_coeffs for G_i - G_j for all j.
            
            # Optimization: Precompute (k+G_j)^2
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # H is N_pw x N_pw
            # H[i, j] = eps_inv_coeffs[idx(G_i - G_j)] * k_plus_G_sq[j]
            
            # We can use broadcasting if we map the differences correctly.
            # G_i - G_j
            # Create a matrix of differences
            # G_vectors is (N, 2)
            # G_diff = G_vectors[:, None, :] - G_vectors[None, :, :]
            # This is (N, N, 2).
            # We need to index eps_inv_coeffs.
            # eps_inv_coeffs is 1D array of length N_pw, indexed by linear index of G in [-8, 8] grid.
            
            # Let's precompute the index mapping for G differences
            # G_diff components are in [-16, 16].
            # We only care about differences that are within the FFT range?
            # The FFT coefficients are available for all G in the grid.
            # If G_i - G_j is outside [-8, 8], the coefficient is 0?
            # No, the FFT gives coefficients for the periodic extension.
            # The plane wave expansion uses G in the truncation set.
            # The convolution sum is exact if we use the full Fourier series.
            # With truncation, we assume coefficients outside the set are 0.
            # So if G_i - G_j is not in our G set, coeff is 0.
            
            # Efficient construction:
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            # Iterate over i
            for i in range(N_pw):
                Gi = G_vectors[i]
                # Find j such that Gi - Gj is in G_set
                # Or just compute Gi - Gj for all j and check if in set
                diffs = Gi[None, :] - G_vectors
                # Map diffs to index
                # dx = diffs[:, 0], dy = diffs[:, 1]
                # idx = (dx + 8) + (dy + 8) * 17
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                # Mask valid diffs
                mask = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
                if np.any(mask):
                    idxs = (dx[mask] + N_G//2) + (dy[mask] + N_G//2) * N_G
                    coeffs = eps_inv_coeffs[idxs]
                    H[i, np.where(mask)[0]] = coeffs * k_plus_G_sq[np.where(mask)[0]]
            
            # Solve eigenvalue problem
            # H is not Hermitian.
            # eigvals
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            # Get 4th and 5th bands (index 3 and 4)
            # Note: There are 2 bands per k? No, 1 band per polarization per k.
            # We want the 4th and 5th lowest frequencies.
            # Since we have N_pw=289, we have 289 eigenvalues.
            # We take the 4th and 5th smallest.
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Initial evaluation
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Local Search
    def get_orbit(i, j):
        # Returns list of unique (r, c) in the symmetric orbit
        N = size
        points = [
            (i, j),
            (j, N-1-i),
            (N-1-i, N-1-j),
            (N-1-j, i),
            (i, N-1-j),
            (N-1-i, j),
            (j, i),
            (N-1-j, N-1-i)
        ]
        return list(set(points))
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        # Pick random pixel
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        orbit = get_orbit(i, j)
        
        # Flip
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
            
        score = evaluate(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            # Revert
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
                
    # Return best_cell as nested list
    return best_cell.tolist()
