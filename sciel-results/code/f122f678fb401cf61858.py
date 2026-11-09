import numpy as np
import time
import itertools

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Time management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for returning and cleanup
    
    # 1. Pre-compute G vectors
    # We need a cutoff for G vectors.
    # For 32x32, high resolution is possible.
    # Let's use G vectors with |G| <= G_max.
    # G = (m, n) * 2*pi.
    # We work in units where lattice constant a=1, so G = (m, n).
    # k is also in units of 2*pi/a.
    
    G_max = 6  # This gives approx (2*6+1)^2 ~ 169 vectors, or less if circular cutoff
    G_vectors = []
    for m in range(-G_max, G_max + 1):
        for n in range(-G_max, G_max + 1):
            if m*m + n*n <= G_max*G_max:
                G_vectors.append((m, n))
    G_vectors = np.array(G_vectors, dtype=float)
    N_G = len(G_vectors)
    
    # 2. K-points in the irreducible Brillouin Zone
    # Square lattice, C4v symmetry.
    # Irreducible zone is triangle Gamma-X-M.
    # Gamma: (0,0)
    # X: (1,0)
    # M: (1,1)
    # We sample points along Gamma-X, X-M, Gamma-M.
    
    k_points = []
    # Gamma to X
    for i in range(6):
        k_points.append((i/5.0, 0.0))
    # X to M
    for i in range(1, 6):
        k_points.append((1.0, i/5.0))
    # Gamma to M
    for i in range(1, 6):
        k_points.append((i/5.0, i/5.0))
    # Add some interior points if time permits, but these 15 are good start.
    k_points = np.array(k_points, dtype=float)
    
    # 3. Helper: Symmetrize cell
    def symmetrize(cell):
        """
        Enforce C4v symmetry.
        cell is 32x32.
        """
        # We take the top-left 16x16 quadrant and mirror/transpose
        # Actually, easier to just average or take majority?
        # Since we start with symmetric cells and flip orbits, it should stay symmetric.
        # But to be safe, we can reconstruct from the fundamental domain.
        # Fundamental domain: i in [0, 15], j in [0, i]
        # Let's just return the cell as is, assuming moves preserve symmetry.
        return cell

    # 4. Helper: Compute Fourier Coefficients of 1/epsilon
    def compute_fourier_coeffs(cell):
        # cell is 32x32, values 0 or 1.
        # 1 = dielectric (eps_high), 0 = air (eps_low)
        eps_inv = np.zeros((size, size))
        eps_inv[cell == 1] = 1.0 / eps_high
        eps_inv[cell == 0] = 1.0 / eps_low
        
        # FFT
        # C_mn = 1/N^2 sum f(i,j) exp(-i 2pi (mi+nj)/N)
        fft_vals = np.fft.fft2(eps_inv)
        fft_vals /= (size * size)
        
        # Extract coefficients for our G vectors
        # G_vectors are (m, n).
        # np.fft.fft2 output index (i, j) corresponds to freq (i, j) for i,j < N/2, 
        # and (i-N, j-N) for i,j >= N/2.
        coeffs = np.zeros(N_G, dtype=complex)
        for idx, (m, n) in enumerate(G_vectors):
            # Map m, n to fft indices
            # m is in [-G_max, G_max]. 
            # fft index i = m if m >= 0, else m + N
            i_idx = m if m >= 0 else m + size
            j_idx = n if n >= 0 else n + size
            coeffs[idx] = fft_vals[i_idx, j_idx]
        
        return coeffs

    # 5. Helper: Evaluate Band Gap
    def evaluate_gap(cell):
        coeffs = compute_fourier_coeffs(cell)
        
        # C matrix: C[i, j] = coeffs[G_i - G_j]
        # We need to map difference of G vectors to index in coeffs.
        # Create a dictionary for G vector to index
        g_dict = {tuple(g): i for i, g in enumerate(G_vectors)}
        
        C_mat = np.zeros((N_G, N_G), dtype=complex)
        for i in range(N_G):
            for j in range(N_G):
                g_diff = G_vectors[i] - G_vectors[j]
                key = (int(g_diff[0]), int(g_diff[1]))
                if key in g_dict:
                    C_mat[i, j] = coeffs[g_dict[key]]
                else:
                    # If G_diff is outside our cutoff, coefficient is ~0
                    # Actually, for a finite pixel map, coefficients decay.
                    # If G_diff is large, we assume 0.
                    C_mat[i, j] = 0.0
        
        # For TM polarization:
        # Eigenvalues of D^{-1} C
        # D is diagonal with (k+G)^2
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            # Construct D
            D_vals = np.zeros(N_G)
            for i in range(N_G):
                kg = k + G_vectors[i]
                D_vals[i] = kg[0]**2 + kg[1]**2
            
            # Avoid division by zero (shouldn't happen for k in BZ and G != -k)
            # If D_val is 0, skip or handle.
            # Construct A = D^{-1} C
            A_mat = C_mat / D_vals[:, np.newaxis]
            
            # Solve eigenvalues
            # A_mat is not symmetric, but C is Hermitian and D is real diag.
            # D^{-1} C is similar to D^{-1/2} C D^{-1/2} which is Hermitian.
            # So eigenvalues should be real.
            try:
                eigvals = np.linalg.eigvals(A_mat)
                # Filter real parts
                eigvals = np.real(eigvals)
                eigvals = eigvals[eigvals > 1e-6] # 1/omega^2 must be positive
                if len(eigvals) < 5:
                    continue
                eigvals.sort()
                
                # omega = 1/sqrt(lambda)
                # lambda = 1/omega^2
                # Small lambda -> Large omega.
                # We want lowest 5 bands -> Largest 5 lambdas?
                # No.
                # H A = omega^2 A.
                # D^{-1} C A = (1/omega^2) A.
                # So eigenvalues of D^{-1} C are 1/omega^2.
                # Lowest frequency (omega_1) corresponds to Largest 1/omega^2.
                # Highest frequency corresponds to Smallest 1/omega^2.
                
                # So sort eigvals in DESCENDING order to get omega_1, omega_2...
                eigvals_desc = eigvals[::-1]
                
                if len(eigvals_desc) < 5:
                    continue
                    
                omega_4 = 1.0 / np.sqrt(eigvals_desc[3])
                omega_5 = 1.0 / np.sqrt(eigvals_desc[4])
                
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                    
            except Exception:
                continue
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = gap / midgap
        return ratio

    # 6. Initial Candidates
    def make_rod(radius):
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dist_sq = (i - center)**2 + (j - center)**2
                if dist_sq < radius**2:
                    cell[i, j] = 1
        return cell

    def make_air_rod(radius):
        cell = np.ones((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dist_sq = (i - center)**2 + (j - center)**2
                if dist_sq < radius**2:
                    cell[i, j] = 0
        return cell

    def make_ring(r_inner, r_outer):
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dist_sq = (i - center)**2 + (j - center)**2
                if r_inner**2 < dist_sq < r_outer**2:
                    cell[i, j] = 1
        return cell

    candidates = []
    # Dielectric rod
    candidates.append(make_rod(0.3 * size))
    # Air rod
    candidates.append(make_air_rod(0.3 * size))
    # Ring
    candidates.append(make_ring(0.2 * size, 0.4 * size))
    # Air Ring
    cell_ar = np.ones((size, size), dtype=int)
    # Make hole in middle and hole in corner?
    # Just use a ring of air in dielectric
    candidates.append(1 - make_ring(0.2 * size, 0.4 * size))
    
    # 7. Hill Climbing
    # Pre-compute orbits for moves
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            # Generate orbit
            orbit = set()
            orbit.add((i, j))
            orbit.add((size-1-i, j))
            orbit.add((i, size-1-j))
            orbit.add((size-1-i, size-1-j))
            orbit.add((j, i))
            orbit.add((size-1-j, i))
            orbit.add((j, size-1-i))
            orbit.add((size-1-j, size-1-i))
            
            # Filter to unique indices
            orbit_list = list(orbit)
            for idx in orbit_list:
                visited.add(idx)
            orbits.append(orbit_list)

    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        # Symmetrize just in case
        # (Candidates are already symmetric)
        
        current_cell = cell.copy()
        current_score = evaluate_gap(current_cell)
        
        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell.copy()
            
        # Hill Climb
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_neighbor = current_cell
            best_neighbor_score = current_score
            
            # Shuffle orbits to avoid bias
            np.random.shuffle(orbits)
            
            for orbit in orbits:
                if (time.time() - start_time) > time_limit:
                    break
                
                # Flip orbit
                for idx in orbit:
                    current_cell[idx[0], idx[1]] = 1 - current_cell[idx[0], idx[1]]
                
                neighbor_score = evaluate_gap(current_cell)
                
                # Unflip
                for idx in orbit:
                    current_cell[idx[0], idx[1]] = 1 - current_cell[idx[0], idx[1]]
                
                if neighbor_score > best_neighbor_score:
                    best_neighbor_score = neighbor_score
                    best_neighbor = current_cell.copy()
                    improved = True
            
            if improved:
                current_cell = best_neighbor
                current_score = best_neighbor_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()

    # Convert to list of lists
    if best_cell is None:
        return [[0 for _ in range(size)] for _ in range(size)]
        
    return best_cell.tolist()

# EVOLVE-BLOCK-END
