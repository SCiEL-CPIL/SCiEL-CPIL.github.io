import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Initialize with seed: centered rod
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1
                
    # PWM Setup
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ (30 points)
    k_points = []
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])
        k_points.append([0.5, t])
        k_points.append([t, t])
    k_points.append([0.25, 0.1])
    k_points.append([0.35, 0.2])
    k_points.append([0.45, 0.3])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
    def get_fourier_eps_inv(cell_arr):
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
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
    
    # Local Search with restarts
    def get_orbit(i, j):
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

    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
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
            no_improve_count = 0
        else:
            # Revert
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
            no_improve_count += 1
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            cell = best_cell.copy()
            for _ in range(10):
                ri = np.random.randint(size)
                rj = np.random.randint(size)
                orbit = get_orbit(ri, rj)
                for (r, c) in orbit:
                    cell[r, c] = 1 - cell[r, c]
            no_improve_count = 0
                
    # Return best_cell as nested list
    return best_cell.tolist()
