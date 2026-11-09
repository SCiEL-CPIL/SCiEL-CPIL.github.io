import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management - explicit deadline guard
    start_time = time.time()
    time_limit = 215.0  # Safety margin before 230s kill
    
    # PWM Setup - use 15x15 plane waves for speed (225 basis functions)
    N_G = 15
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ (30 points covering boundary and interior)
    k_points = []
    # Boundary: X-M edge (kx from 0 to 0.5, ky=0)
    for i in range(7):
        t = i / 6.0 * 0.5
        k_points.append([t, 0.0])
    # Boundary: M-G edge (kx=0.5, ky from 0 to 0.5)
    for i in range(1, 7):
        t = i / 6.0 * 0.5
        k_points.append([0.5, t])
    # Boundary: G-M diagonal (kx=ky from 0 to 0.5)
    for i in range(1, 7):
        t = i / 6.0 * 0.5
        k_points.append([t, t])
    # Interior points
    k_points.append([0.25, 0.1])
    k_points.append([0.35, 0.2])
    k_points.append([0.4, 0.15])
    k_points.append([0.15, 0.3])
    k_points.append([0.2, 0.4])
    k_points.append([0.3, 0.3])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix construction
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
    def get_fourier_eps_inv(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given pixel map."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        # Map G indices to FFT indices
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build H matrix: H[i,j] = eps_inv(G_i - G_j) * |k + G_j|^2
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            # Solve eigenvalue problem
            evals = np.linalg.eigvalsh(H)
            evals = np.sort(np.real(evals))
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return 0.0
        return gap / midgap
    
    def get_orbit(i, j):
        """Get the symmetry orbit of a pixel under D4 group."""
        N = size
        points = set()
        points.add((i, j))
        points.add((j, N-1-i))
        points.add((N-1-i, N-1-j))
        points.add((N-1-j, i))
        points.add((i, N-1-j))
        points.add((N-1-i, j))
        points.add((j, i))
        points.add((N-1-j, N-1-i))
        return list(points)
    
    def generate_symmetric_cell(filled_fraction=0.3):
        """Generate a random symmetric cell with approximately the given fill fraction."""
        cell = np.zeros((size, size), dtype=int)
        # Determine how many orbits to fill
        # Total pixels: size*size, each orbit has 1-8 pixels
        # We'll fill orbits randomly until we reach the target fraction
        orbits = []
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) not in visited:
                    orbit = get_orbit(i, j)
                    orbits.append(orbit)
                    for p in orbit:
                        visited.add(p)
        
        np.random.shuffle(orbits)
        
        target_pixels = int(size * size * filled_fraction)
        current_pixels = 0
        for orbit in orbits:
            if current_pixels + len(orbit) > target_pixels:
                break
            for (r, c) in orbit:
                cell[r, c] = 1
            current_pixels += len(orbit)
        
        return cell
    
    # Initial candidates - try several different designs
    candidates = []
    
    # Candidate 1: Centered square rod
    cell1 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.15 and abs(y) < 0.15:
                cell1[i, j] = 1
    candidates.append(cell1)
    
    # Candidate 2: Centered circular rod
    cell2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.18**2:
                cell2[i, j] = 1
    candidates.append(cell2)
    
    # Candidate 3: Square lattice of small rods (4x4 array)
    cell3 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # 4x4 grid of small squares
            xi = (x + 0.5) % 0.25
            yi = (y + 0.5) % 0.25
            if xi < 0.08 and yi < 0.08:
                cell3[i, j] = 1
    candidates.append(cell3)
    
    # Candidate 4: Random symmetric with ~30% fill
    candidates.append(generate_symmetric_cell(0.3))
    
    # Candidate 5: Random symmetric with ~40% fill
    candidates.append(generate_symmetric_cell(0.4))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = None
    for cell in candidates:
        score = evaluate(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local search with adaptive step size
    def local_search(cell, max_iter=500):
        """Perform local search starting from the given cell."""
        best_local_score = evaluate(cell)
        best_local_cell = cell.copy()
        no_improve = 0
        step_size = 1  # Number of orbits to flip
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick random orbit(s)
            i = np.random.randint(size)
            j = np.random.randint(size)
            orbit = get_orbit(i, j)
            
            # Flip the orbit
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
            
            score = evaluate(cell)
            
            if score > best_local_score:
                best_local_score = score
                best_local_cell = cell.copy()
                no_improve = 0
                # Occasionally increase step size
                if np.random.random() < 0.1:
                    step_size = min(step_size + 1, 3)
            else:
                # Revert
                for (r, c) in orbit:
                    cell[r, c] = 1 - cell[r, c]
                no_improve += 1
                # Decrease step size if stuck
                if no_improve > 50 and step_size > 1:
                    step_size = max(step_size - 1, 1)
            
            if no_improve > 200:
                break
        
        return best_local_cell, best_local_score
    
    # Multi-start local search
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Start from best cell with some perturbation
        cell = best_cell.copy()
        n_perturb = np.random.randint(3, 8)
        for _ in range(n_perturb):
            ri = np.random.randint(size)
            rj = np.random.randint(size)
            orbit = get_orbit(ri, rj)
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
        
        # Run local search
        local_best_cell, local_best_score = local_search(cell)
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_cell = local_best_cell.copy()
    
    # Final refinement on the best cell
    if time.time() - start_time < time_limit - 5:
        final_cell, final_score = local_search(best_cell, max_iter=300)
        if final_score > best_score:
            best_score = final_score
            best_cell = final_cell
    
    # Return best_cell as nested list
    return best_cell.tolist()
