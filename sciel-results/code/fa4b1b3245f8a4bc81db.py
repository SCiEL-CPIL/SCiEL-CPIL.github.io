"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_pwe_solver(size, eps_high, n_pw=289):
    """Build a vectorized plane-wave eigenvalue solver for TM polarization on a square lattice."""
    gw = int(np.sqrt(n_pw))  # 17
    half = gw // 2
    # Generate G vectors
    G = []
    for ig in range(gw):
        for jg in range(gw):
            Gx = (ig - half)
            Gy = (jg - half)
            G.append((Gx, Gy))
    G = np.array(G, dtype=float)
    nG = len(G)
    
    # Precompute G-G matrix (differences)
    # G_diff[i, j] = G[i] - G[j]
    G_diff = G[:, np.newaxis, :] - G[np.newaxis, :, :]
    
    def compute_epsilon_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon for the pixel cell."""
        eps = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    eps[i, j] = eps_high
        
        inv_eps = 1.0 / eps
        F = np.fft.fft2(inv_eps)
        
        max_freq = 2 * half
        freq_range = np.arange(-max_freq, max_freq + 1)
        
        # We need inv_eps_F[m, n] for m, n in [-max_freq, max_freq]
        # F[k] is the DFT coefficient.
        # The continuous Fourier coefficient c_{mn} = (1/S) * sum_{r} f(r) exp(-i 2pi (m r_x/L + n r_y/L))
        # Here L=1, S=1.
        # DFT F[k, l] = sum_{i,j} f[i,j] exp(-i 2pi (i k / size + j l / size))
        # So c_{mn} = F[k, l] * exp(-i 2pi (m (k-k0)/size + ...)) ?
        # Actually, for a periodic function defined on a grid of size N, the Fourier series coefficients are related to the DFT.
        # c_{mn} = (1/N^2) * F[k, l] * exp(-i 2pi (m * (k - k_shift) / N ...))
        # Let's use the direct definition for safety and correctness with the phase shift to center.
        # The pixel (i,j) is centered at (i+0.5)/size, (j+0.5)/size.
        # f(r) = sum_{i,j} val_{ij} * rect(r - r_{ij})
        # c_{mn} = (1/N^2) * sum_{i,j} val_{ij} * exp(-i 2pi (m (i+0.5)/N + n (j+0.5)/N))
        
        inv_eps_F = np.zeros((2*max_freq+1, 2*max_freq+1), dtype=complex)
        
        # Map integer indices m, n to array indices
        # idx = m + max_freq
        for mi, m in enumerate(freq_range):
            for ni, n in enumerate(freq_range):
                # Compute sum over pixels
                # phase = exp(-i 2pi (m (i+0.5)/N + n (j+0.5)/N))
                # = exp(-i 2pi (m/N) * 0.5) * exp(-i 2pi (n/N) * 0.5) * exp(-i 2pi (m i + n j) / N)
                phase_pix = np.exp(-1j * 2 * np.pi * (m * 0.5 / size + n * 0.5 / size))
                
                # The term exp(-i 2pi (m i + n j) / N) is handled by FFT if we interpret correctly.
                # FFT of inv_eps gives sum_{i,j} inv_eps[i,j] exp(-i 2pi (i k + j l) / N)
                # We need k such that k mod N corresponds to m.
                # Since m is in [-max_freq, max_freq] and max_freq < N/2 usually (17 < 16? No, 17 is half of 34, N=32).
                # Wait, gw=17, half=8. max_freq=16.
                # m is in [-16, 16].
                # k = m % 32.
                k_m = m % size
                k_n = n % size
                
                # The DFT value F[k_m, k_n] is sum_{i,j} inv_eps[i,j] exp(-i 2pi (i k_m + j k_n) / N)
                # This matches the sum part.
                c_val = F[k_m, k_n] * phase_pix / (size * size)
                inv_eps_F[mi, ni] = c_val
                
        return inv_eps_F, max_freq

    def solve_bands_at_k(inv_eps_F, max_freq, kx, ky, n_bands=5):
        """Solve for the first n_bands TM bands at wavevector (kx, ky) using vectorized ops."""
        k = np.array([kx, ky], dtype=float)
        
        # Compute |G_p + k|^2 for all p
        Gk = G + k
        Gk_sq = np.sum(Gk**2, axis=1)
        
        # Construct Matrix A
        # A[p, q] = Gk_sq[q] * inv_eps_F[G_p - G_q]
        
        # We need to map G_p - G_q to indices in inv_eps_F
        # G_diff is shape (nG, nG, 2)
        # diff_m = G_diff[:, :, 0]
        # diff_n = G_diff[:, :, 1]
        
        diff_m = G_diff[:, :, 0].astype(int)
        diff_n = G_diff[:, :, 1].astype(int)
        
        # Map to indices 0..2*max_freq
        idx_m = diff_m + max_freq
        idx_n = diff_n + max_freq
        
        # Check bounds. Since G is in [-8, 8], diff is in [-16, 16].
        # max_freq = 16. So idx is in [0, 32].
        # inv_eps_F size is 33x33.
        
        # Extract the relevant part of inv_eps_F
        # We can just index directly
        eps_inv_matrix = inv_eps_F[idx_m, idx_n]
        
        # A[p, q] = Gk_sq[q] * eps_inv_matrix[p, q]
        # Note: Gk_sq is a vector of length nG. We need to multiply column q by Gk_sq[q].
        A = eps_inv_matrix * Gk_sq[np.newaxis, :]
        
        # Symmetrize to remove numerical noise
        A = (A + A.conj().T) / 2
        
        try:
            eigvals, _ = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            return None
            
        # eigvals are (omega*a)^2
        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0)
        return np.sqrt(eigvals[:n_bands])

    def evaluate_cell(cell, k_points, n_bands=5):
        """Evaluate the gap ratio for a given cell at specific k-points."""
        inv_eps_F, max_freq = compute_epsilon_inv_fourier(cell)
        min_ratio = float('inf')
        
        for kx, ky in k_points:
            eigvals = solve_bands_at_k(inv_eps_F, max_freq, kx, ky, n_bands)
            if eigvals is None:
                return -1.0
            
            w4 = eigvals[3]
            w5 = eigvals[4]
            
            if w4 + w5 < 1e-12:
                ratio = -1.0
            else:
                ratio = 2.0 * (w5 - w4) / (w5 + w4)
                
            if ratio < min_ratio:
                min_ratio = ratio
                
        return min_ratio

    return evaluate_cell

def design(problem):
    """Optimize a symmetric 32x32 photonic crystal cell for maximum TM band gap."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    # Define k-points for fast evaluation (subset)
    # The evaluator uses 30 points. We use a representative subset for optimization.
    k_points_fast = [
        (0.0, 0.0),   # Gamma
        (1.0, 0.0),   # X
        (0.5, 0.5),   # M
        (0.5, 0.0),   # Mid Gamma-X
        (0.0, 0.5),   # Mid Gamma-Y (symmetric to above)
        (0.25, 0.25), # Interior
        (0.75, 0.25), # Interior
        (0.5, 0.25),  # Interior
        (0.25, 0.5),  # Interior
        (0.75, 0.5),  # Interior
        (0.3, 0.4),   # Generic
        (0.6, 0.4),   # Generic
    ]
    
    evaluate_cell = _make_pwe_solver(size, eps_high, n_pw=289)
    
    # Initialize with a symmetric pattern
    # A common starting point for large gaps is a high-contrast disk or cross.
    # Let's start with a central disk of dielectric.
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    radius = 0.35 * size
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) / size - 0.5
            dy = (j + 0.5) / size - 0.5
            if dx*dx + dy*dy < radius*radius:
                cell[i][j] = 1

    def enforce_symmetry(cell):
        n = len(cell)
        new_cell = [[0]*n for _ in range(n)]
        # Symmetry group: D4 (8 elements)
        # (i, j) -> (i, j), (n-1-i, j), (i, n-1-j), (n-1-i, n-1-j)
        #             (j, i), (j, n-1-i), (n-1-j, i), (n-1-j, n-1-i)
        
        # To avoid overwriting issues, we process unique orbits
        visited = set()
        for i in range(n):
            for j in range(n):
                if (i, j) in visited:
                    continue
                
                equivs = [
                    (i, j),
                    (n-1-i, j),
                    (i, n-1-j),
                    (n-1-i, n-1-j),
                    (j, i),
                    (j, n-1-i),
                    (n-1-j, i),
                    (n-1-j, n-1-i)
                ]
                
                # Count ones in this orbit
                ones = sum(1 for (ei, ej) in equivs if cell[ei][ej] == 1)
                # If majority is 1, set all to 1, else 0
                # This enforces the symmetry constraint
                val = 1 if ones > len(equivs)/2 else 0
                
                for (ei, ej) in equivs:
                    new_cell[ei][ej] = val
                    visited.add((ei, ej))
        return new_cell

    cell = enforce_symmetry(cell)
    
    # Get fundamental domain indices
    # We only need to iterate over indices that generate the full cell under symmetry
    # A simple way: iterate all (i,j), but only modify if (i,j) is the "canonical" representative
    # Or just iterate all, but check if we've already processed its orbit.
    
    fd_indices = []
    for i in range(size):
        for j in range(size):
            fd_indices.append((i, j))
            
    # We will use a greedy hill-climbing approach.
    # To speed up, we will precompute the orbits for all indices.
    orbits = {}
    for i in range(size):
        for j in range(size):
            if (i, j) in orbits:
                continue
            equivs = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (j, size-1-i),
                (size-1-j, i),
                (size-1-j, size-1-i)
            ]
            # Unique list
            unique_equivs = list(set(equivs))
            for eq in unique_equivs:
                orbits[eq] = unique_equivs
                
    current_score = evaluate_cell(cell, k_points_fast)
    best_cell = [row[:] for row in cell]
    best_score = current_score
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices_list = list(fd_indices)
        np.random.shuffle(indices_list)
        
        for (i, j) in indices_list:
            if time.time() > deadline:
                break
                
            equivs = orbits[(i, j)]
            
            # Flip the pixel and its symmetric counterparts
            old_vals = []
            for (ei, ej) in equivs:
                old_vals.append(cell[ei][ej])
                cell[ei][ej] = 1 - cell[ei][ej]
                
            new_score = evaluate_cell(cell, k_points_fast)
            
            if new_score > best_score + 1e-8:
                best_score = new_score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                # Revert
                for k, (ei, ej) in enumerate(equivs):
                    cell[ei][ej] = old_vals[k]
                    
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 5:
                break
        else:
            no_improve_count = 0

    return best_cell
# EVOLVE-BLOCK-END
