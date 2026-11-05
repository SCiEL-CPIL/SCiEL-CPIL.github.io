import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # 10s margin for safety
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Define Symmetry Orbits for 32x32 grid
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orbit = set()
                coords = [
                    (i, j),
                    (j, size - 1 - i),
                    (size - 1 - i, size - 1 - j),
                    (size - 1 - j, i),
                    (i, size - 1 - j),
                    (size - 1 - i, j),
                    (j, i),
                    (size - 1 - j, size - 1 - i)
                ]
                for r, c in coords:
                    if 0 <= r < size and 0 <= c < size:
                        orbit.add((r, c))
                        visited[r, c] = True
                orbits.append(list(orbit))
    
    # 2. Initialize with the current best known design (4 rods)
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    r = 5.5
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx, cy in centers:
                if (x - cx)**2 + (y - cy)**2 < r**2:
                    cell[i, j] = 1
                    break
    
    # 3. Fast PWE Evaluator using FFT approximation
    N_pw = 4 # Half-width of PW grid. Total (2*N_pw+1)^2 = 81
    k_vectors = []
    for nx in range(-N_pw, N_pw + 1):
        for ny in range(-N_pw, N_pw + 1):
            k_vectors.append((nx, ny))
    num_pw = len(k_vectors)
    
    # Reciprocal lattice vectors for square lattice with a=1: G = (2pi nx, 2pi ny)
    G_vectors = np.array([(2 * math.pi * nx, 2 * math.pi * ny) for nx, ny in k_vectors])
    
    # Map G indices to DFT indices
    # DFT index (kx, ky) corresponds to frequency (2pi kx / size, 2pi ky / size)
    # We want frequency (2pi nx, 2pi ny).
    # So kx = nx * size, ky = ny * size.
    # Since DFT is periodic, kx mod size = 0 if nx is integer?
    # Wait. The DFT of a 32x32 grid gives frequencies 0, 1, ..., 31.
    # These correspond to 2pi * k / 32.
    # We want 2pi * nx.
    # So 2pi * k / 32 = 2pi * nx => k = 32 * nx.
    # k mod 32 = 0.
    # This means all non-zero integer frequencies alias to 0 in the DFT of the discrete grid?
    # No, that's not right. The discrete grid samples the function.
    # The Fourier series coefficients of the *continuous* function sampled on a grid are related to the DFT.
    # Specifically, if we assume the function is band-limited or just approximate,
    # the DFT coefficient at index `k` approximates the integral for frequency `2pi k / N`.
    # If we want frequency `2pi nx`, we need `2pi k / 32 = 2pi nx` => `k = 32 nx`.
    # This implies that for nx=1, k=32, which is 0 mod 32.
    # This suggests that using the DFT of the 32x32 grid directly for integer G vectors is problematic
    # because the grid is too coarse to resolve the high frequencies without aliasing to zero.
    
    # Correct approach for discrete PWE:
    # The matrix element involves the convolution of the pixel values with the plane waves.
    # We must compute the sum explicitly: sum_{i,j} val[i,j] exp(-i G . r_{i,j}).
    # This cannot be replaced by a simple DFT of the 32x32 grid if G is large relative to the grid spacing.
    # However, we can precompute the exponential terms for each G and each pixel.
    # There are 81 Gs and 1024 pixels. 81 * 1024 = 82944 complex numbers.
    # We can store these in a matrix `ExpMatrix` of shape (num_pw, size*size).
    # Then `coeffs = ExpMatrix @ flat_values`.
    # This is a matrix-vector product: (81, 1024) @ (1024, 1).
    # This is very fast in NumPy.
    
    # Precompute ExpMatrix
    i_idx, j_idx = np.mgrid[0:size, 0:size]
    x_coords = j_idx + 0.5
    y_coords = i_idx + 0.5
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    ExpMatrix = np.zeros((num_pw, size * size), dtype=complex)
    for idx, (gx, gy) in enumerate(G_vectors):
        phase = -(gx * x_flat + gy * y_flat)
        ExpMatrix[idx, :] = np.exp(1j * phase)
        
    # Normalize by area
    ExpMatrix /= (size * size)
    
    # Precompute g_map for D matrix
    g_map = {}
    for idx, (nx, ny) in enumerate(k_vectors):
        g_map[(nx, ny)] = idx
        
    # K points
    k_points = [
        (0, 0),       # Gamma
        (math.pi, 0), # X
        (math.pi, math.pi) # M
    ]
    
    def evaluate_cell(cell_grid):
        # Compute inv_eps coefficients
        inv_eps_grid = np.where(cell_grid == 1, 1.0/eps_high, 1.0/eps_low)
        inv_eps_flat = inv_eps_grid.flatten()
        
        # coeffs = ExpMatrix @ inv_eps_flat
        inv_eps_coeffs = ExpMatrix @ inv_eps_flat
        
        # Build D matrix
        D = np.ones((num_pw, num_pw), dtype=complex)
        for m in range(num_pw):
            nx_m, ny_m = k_vectors[m]
            for n in range(num_pw):
                nx_n, ny_n = k_vectors[n]
                dx = nx_m - nx_n
                dy = ny_m - ny_n
                if (dx, dy) in g_map:
                    D[m, n] = inv_eps_coeffs[g_map[(dx, dy)]]
                else:
                    D[m, n] = 1.0
        
        min_ratio = 0.0
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            k_minus_G = k_vec[None, :] - G_vectors
            K_mat = k_minus_G @ k_minus_G.T
            A = K_mat / D
            A = (A + A.conj().T) / 2
            try:
                evals = np.linalg.eigvalsh(A)
                evals = np.sort(evals)
                if len(evals) >= 5:
                    w4 = np.sqrt(evals[3])
                    w5 = np.sqrt(evals[4])
                    ratio = (w5 - w4) / ((w5 + w4) / 2.0)
                    if ratio < min_ratio:
                        min_ratio = ratio
            except:
                pass
                
        return min_ratio

    # 4. Optimization Loop
    best_cell = cell.copy()
    best_score = evaluate_cell(best_cell)
    
    max_iterations = 2000
    iteration = 0
    
    orbit_indices = list(range(len(orbits)))
    np.random.shuffle(orbit_indices)
    
    while iteration < max_iterations:
        if time.time() > deadline - 5:
            break
            
        orbit_idx = orbit_indices[iteration % len(orbit_indices)]
        orbit = orbits[orbit_idx]
        
        new_cell = best_cell.copy()
        for r, c in orbit:
            new_cell[r, c] = 1 - new_cell[r, c]
            
        new_score = evaluate_cell(new_cell)
        
        if new_score > best_score:
            best_cell = new_cell
            best_score = new_score
            
        iteration += 1
        
    return best_cell.tolist()
