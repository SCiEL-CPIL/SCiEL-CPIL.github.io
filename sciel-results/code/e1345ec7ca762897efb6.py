import numpy as np
import time
import random
import scipy.linalg

def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Symmetry Setup ---
    # C4v symmetry about center (15.5, 15.5)
    # We define canonical representatives for each orbit of 8 pixels.
    # Orbit of (i, j):
    # (i, j), (31-i, 31-j), (j, i), (31-j, 31-i),
    # (31-i, j), (i, 31-j), (31-j, i), (j, 31-i)
    
    canonical_map = {}
    for i in range(size):
        for j in range(size):
            orbit = [
                (i, j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, size - 1 - i),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - j, i),
                (j, size - 1 - i)
            ]
            # Remove duplicates (pixels on symmetry axes)
            unique_orbit = list(set(orbit))
            # Find lexicographically smallest
            rep = min(unique_orbit)
            if rep not in canonical_map:
                canonical_map[rep] = unique_orbit
    
    free_indices = list(canonical_map.keys())
    random.shuffle(free_indices)

    def get_eps_grid(cell):
        """Converts binary cell to epsilon grid."""
        return np.where(cell == 1, eps_high, eps_low).astype(float)

    def evaluate_cell(cell):
        """
        Fast PWE evaluator using 7x7 PWs and 10 k-points.
        Returns the gap-midgap ratio.
        """
        eps_grid = get_eps_grid(cell)
        
        # Compute Fourier coefficients of epsilon
        # fft2 gives coefficients for frequencies 0..N/2, -N/2..-1
        # We need coefficients for G in [-Npw, Npw]
        fft_eps = np.fft.fft2(eps_grid)
        fft_eps /= (size * size)
        
        # Shift to center
        fft_eps = np.fft.fftshift(fft_eps)
        
        N_pw = 3 # 7x7 grid
        # Extract central block
        # Indices for fftshifted array: center is size//2
        center = size // 2
        start = center - N_pw
        end = center + N_pw + 1
        eps_coeffs = fft_eps[start:end, start:end]
        
        # G vectors
        G_vals = np.arange(-N_pw, N_pw + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
        num_pw = len(G_vectors)
        
        # Map G vector to index in eps_coeffs
        # eps_coeffs is indexed by [Gx+Npw, Gy+Npw]
        # G_vectors[i] = [Gx_i, Gy_i]
        # idx_x = Gx_i + N_pw
        # idx_y = Gy_i + N_pw
        
        # Precompute matrix indices for A (epsilon conv)
        # A[i, j] = eps_coeffs[Gi - Gj]
        # We need to access eps_coeffs at (Gi_x - Gj_x + N_pw, Gi_y - Gj_y + N_pw)
        
        # K points in IBZ
        # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
        # Path: Gamma -> M -> X -> Gamma
        k_points = [
            (0.0, 0.0),
            (0.125, 0.0),
            (0.25, 0.0),
            (0.375, 0.0),
            (0.5, 0.0),
            (0.5, 0.125),
            (0.5, 0.25),
            (0.5, 0.375),
            (0.5, 0.5),
            (0.375, 0.375)
        ]
        
        max_omega_4 = -np.inf
        min_omega_5 = np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Construct Matrix A: A[i,j] = eps_coeffs[Gi - Gj]
            # A[i,j] corresponds to epsilon_{Gi-Gj}
            A = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dx = G_vectors[i, 0] - G_vectors[j, 0]
                    dy = G_vectors[i, 1] - G_vectors[j, 1]
                    # Index in eps_coeffs
                    ix = int(dx + N_pw)
                    iy = int(dy + N_pw)
                    A[i, j] = eps_coeffs[ix, iy]
            
            # Construct Matrix B: B[i,i] = |k+Gi|^2
            B = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                k_plus_G = k + G_vectors[i]
                B[i, i] = np.sum(k_plus_G ** 2)
            
            # Solve generalized eigenvalue problem A v = lambda B v
            # lambda = 1/omega^2
            try:
                eigvals, eigvecs = scipy.linalg.eig(A, B)
            except Exception:
                return -1.0
            
            real_eigvals = np.real(eigvals)
            # Filter positive real eigenvalues
            valid_eigvals = real_eigvals[real_eigvals > 1e-9]
            if len(valid_eigvals) < 5:
                continue
            
            # lambda = 1/omega^2 => omega = 1/sqrt(lambda)
            omegas = 1.0 / np.sqrt(valid_eigvals)
            omegas.sort()
            
            if len(omegas) >= 5:
                omega_4 = omegas[3]
                omega_5 = omegas[4]
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        
        ratio = 2 * gap / midgap
        return ratio

    # --- Optimization: Simulated Annealing ---
    
    def symmetrize_cell(cell):
        """Ensures the cell is strictly C4v symmetric."""
        # cell is a 32x32 numpy array
        # We just enforce it by copying from canonical representatives
        # Actually, we maintain symmetry by construction in the mutation step.
        # But for initial guesses, we need to symmetrize.
        sym_cell = np.zeros_like(cell)
        for rep, orbit in canonical_map.items():
            val = cell[rep[0], rep[1]]
            for (i, j) in orbit:
                sym_cell[i, j] = val
        return sym_cell

    # Initial Guesses
    def make_initial_cell(type_id):
        cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        
        if type_id == 0: # All Dielectric
            cell[:] = 1
        elif type_id == 1: # All Air
            cell[:] = 0
        elif type_id == 2: # Central Dielectric Rod
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - (center - 0.5))**2 + (y - (center - 0.5))**2)
            cell[dist < 5] = 1
        elif type_id == 3: # Central Air Hole
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - (center - 0.5))**2 + (y - (center - 0.5))**2)
            cell[:] = 1
            cell[dist < 5] = 0
        elif type_id == 4: # 4 Rods
            y, x = np.ogrid[:size, :size]
            # Centers at (8,8), (8,24), (24,8), (24,24)
            for cx, cy in [(8, 8), (8, 24), (24, 8), (24, 24)]:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                cell[dist < 3] = 1
        elif type_id == 5: # Checkerboard
            y, x = np.ogrid[:size, :size]
            cell[(x + y) % 2 == 0] = 1
        elif type_id == 6: # Random
            cell[:] = np.random.randint(0, 2, (size, size))
        
        return symmetrize_cell(cell)

    best_cell = None
    best_score = -np.inf
    
    # Evaluate initial guesses
    for i in range(7):
        cell = make_initial_cell(i)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    if best_cell is None:
        # Fallback
        return [[0]*size for _ in range(size)]

    current_cell = best_cell.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    iterations = 0
    while time.time() < deadline - 5.0:
        iterations += 1
        
        # Pick a random free index
        idx = random.choice(free_indices)
        orbit = canonical_map[idx]
        
        # Flip the orbit
        for (i, j) in orbit:
            current_cell[i, j] = 1 - current_cell[i, j]
        
        new_score = evaluate_cell(current_cell)
        
        # Metropolis Criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Revert
            for (i, j) in orbit:
                current_cell[i, j] = 1 - current_cell[i, j]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher T if stuck?
            # Simple: just continue at T_min

    # Convert best_cell to list of lists
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
