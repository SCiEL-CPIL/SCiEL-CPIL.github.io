import numpy as np
import time
import sys

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    N = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Define the G-vectors for 289 plane waves
    # 17x17 grid of G-vectors
    # G = (m, n) * 2*pi / a
    # We use indices -8 to 8
    g_indices = np.arange(-8, 9)
    Gx, Gy = np.meshgrid(g_indices, g_indices)
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    G_m = Gx
    G_n = Gy
    num_pw = len(Gx)
    
    # Precompute G^2
    G_sq = Gx**2 + Gy**2
    
    # K-points in the irreducible Brillouin zone
    # Square BZ: Gamma (0,0) to X (pi/a, 0) to M (pi/a, pi/a) to Gamma
    # Irreducible wedge: 0 <= kx <= ky <= pi/a? No, 0 <= kx, 0 <= ky, kx <= ky is 1/4.
    # Symmetry group is C4v. Irreducible wedge is 1/8 of the BZ.
    # 0 <= kx <= ky, kx+ky <= pi/a?
    # Standard path: Gamma(0,0) -> X(1,0) -> M(1,1) -> Gamma(0,0) in units of pi/a.
    # Irreducible part: Triangle (0,0) to (1,0) to (1,1)? No.
    # The wedge is bounded by kx=0, ky=kx, and kx+ky=1?
    # Let's use 30 points covering the high symmetry lines and interior.
    # Points:
    # 1. Gamma (0,0)
    # 2. X (1,0)
    # 3. M (1,1)
    # 4. Points along Gamma-X: (t, 0) for t in 0.1..0.9
    # 5. Points along X-M: (1, t) for t in 0.1..0.9
    # 6. Points along M-Gamma: (t, t) for t in 0.1..0.9
    # 7. Interior points
    
    k_points = []
    # Gamma to X
    for t in np.linspace(0, 1, 10):
        k_points.append((t, 0))
    # X to M
    for t in np.linspace(0, 1, 10):
        k_points.append((1, t))
    # M to Gamma
    for t in np.linspace(0, 1, 10):
        k_points.append((t, t))
    # Remove duplicates or just use unique?
    # The problem says "30 wave vectors". I will generate 30 unique points.
    
    k_points = []
    # 10 points on Gamma-X
    for i in range(10):
        k_points.append((i/9.0, 0.0))
    # 10 points on X-M
    for i in range(1, 10):
        k_points.append((1.0, i/9.0))
    # 10 points on M-Gamma
    for i in range(1, 10):
        k_points.append((i/9.0, i/9.0))
    # Total 28 points. Add 2 interior points?
    k_points.append((0.5, 0.25))
    k_points.append((0.75, 0.5))
    # Note: These are in units of pi/a.
    # The actual k-vector is k * pi / a.
    # In the matrix formula, we use k + G.
    # G is in units of 2*pi/a.
    # So k (pi/a) = 0.5 * G_unit (2*pi/a)?
    # Let's stick to units of 2*pi/a for consistency.
    # G = (m, n).
    # k = (kx, ky) in units of 2*pi/a.
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5).
    
    k_points_scaled = []
    for kx, ky in k_points:
        k_points_scaled.append((kx * 0.5, ky * 0.5)) # Convert pi/a to 2pi/a units
    
    # Function to calculate gap
    def calculate_gap(cell):
        # cell is 32x32 list of 0/1
        # Compute Fourier coefficients epsilon_G
        # epsilon(r) = eps_low + (eps_high - eps_low) * pixel(r)
        # epsilon_G = 1/N^2 sum pixel(i,j) exp(-i G . r_ij)
        # r_ij = ((i+0.5)/N, (j+0.5)/N)
        
        # Precompute exp terms for efficiency?
        # Gx, Gy are integers.
        # exp(-i * 2*pi * (m*(i+0.5)/N + n*(j+0.5)/N))
        
        # We can use FFT?
        # The grid is N x N.
        # We need specific G-vectors, not all.
        # But N=32 is small.
        
        # Let's compute epsilon_G for the 289 G-vectors.
        eps_G = np.zeros(num_pw, dtype=complex)
        
        # Unroll the sum for speed?
        # cell is a list of lists. Convert to numpy array.
        cell_arr = np.array(cell, dtype=float)
        
        # Create meshgrid for pixel centers
        i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        x_pix = (i_idx + 0.5) / N
        y_pix = (j_idx + 0.5) / N
        
        # For each G, compute sum
        # This is a matrix multiplication:
        # epsilon_G[k] = sum_{i,j} cell[i,j] * exp(-i 2pi (Gx[k] x_i + Gy[k] y_j)) / N^2
        
        # Construct the phase matrix
        # Phases[i, j, k] = exp(-i 2pi (Gx[k] x_i + Gy[k] y_j))
        # This is large: 32*32*289 = 296k complex numbers. Fine.
        
        # More efficient:
        # exp(-i 2pi Gx x_i) is a vector of length N for each Gx.
        # We can compute the 2D Fourier transform at specific points.
        
        # Let's just loop over G for simplicity, N is small.
        for k in range(num_pw):
            gx = Gx[k]
            gy = Gy[k]
            # exp(-i 2pi (gx * x_i + gy * y_j))
            phase = np.exp(-1j * 2 * np.pi * (gx * x_pix + gy * y_pix))
            eps_G[k] = np.sum(cell_arr * phase) / (N * N)
            
        # Add background epsilon_low?
        # The formula uses epsilon_{m-n}.
        # epsilon(r) is the full dielectric function.
        # So eps_G[0] should be the average epsilon.
        # The loop above calculates the coefficient of the *pixel* part.
        # epsilon(r) = eps_low + (eps_high - eps_low) * pixel(r)
        # So epsilon_G = (eps_high - eps_low) * pixel_G + eps_low * delta_{G,0}
        
        delta_G0 = (Gx == 0) & (Gy == 0)
        eps_G = eps_G * (eps_high - eps_low)
        eps_G[delta_G0] += eps_low
        
        # Ensure real part for Hermitian matrix?
        # epsilon_G should be real if the structure is symmetric.
        # We will use the real part.
        eps_G = np.real(eps_G)
        
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for kx, ky in k_points_scaled:
            # Construct A and D
            # A_{mn} = delta_{mn} + (G_m - G_n)^2 / (4 pi^2 eps_{m-n})
            # D_{mm} = (kx + G_mx)^2 + (ky + G_my)^2
            
            # Vectorized construction of A
            # G_m - G_n
            dGx = Gx[:, None] - Gx[None, :]
            dGy = Gy[:, None] - Gy[None, :]
            dG_sq = dGx**2 + dGy**2
            
            # eps_{m-n}
            # We need eps_G for indices m-n.
            # G_m - G_n corresponds to G index (m_idx - n_idx)?
            # No, G vectors are not indexed by simple difference in array index if they are not on a grid?
            # They are on a grid.
            # G_m = (m_x, m_y). G_n = (n_x, n_y).
            # G_m - G_n = (m_x - n_x, m_y - n_y).
            # We need to find the index k such that Gx[k] = m_x - n_x and Gy[k] = m_y - n_y.
            
            # This lookup is expensive if done naively.
            # Precompute a mapping from (dx, dy) to index k.
            g_map = {}
            for k in range(num_pw):
                g_map[(Gx[k], Gy[k])] = k
            
            # Build eps_matrix
            eps_mat = np.zeros((num_pw, num_pw))
            for m in range(num_pw):
                for n in range(num_pw):
                    dx = Gx[m] - Gx[n]
                    dy = Gy[m] - Gy[n]
                    idx = g_map.get((dx, dy), -1)
                    if idx != -1:
                        eps_mat[m, n] = eps_G[idx]
                    else:
                        # Should not happen if G set is closed under difference
                        # 289 G-vectors (-8..8) difference is -16..16.
                        # But we only have -8..8.
                        # The plane wave method requires G_m - G_n to be in the set?
                        # Actually, the sum is over all n.
                        # The term is (G_m - G_n)^2 / eps_{m-n}.
                        # If m-n is outside the basis, eps is 0?
                        # No, eps_{G} is defined for all G.
                        # But we only expand in the basis set.
                        # The matrix element A_{mn} involves eps_{m-n}.
                        # If m-n is not in the basis, do we include it?
                        # The standard PWE method sums over all n in the basis.
                        # The term eps_{m-n} is the Fourier coefficient at G_m - G_n.
                        # If G_m - G_n is not in the basis, it is still a valid Fourier coefficient.
                        # We must compute eps_G for ALL possible differences.
                        
                        # My current eps_G array only has 289 entries.
                        # The differences can be from -16 to 16.
                        # I need to compute eps_G for a larger set, or handle it.
                        
                        # Let's compute eps_G for the larger set: -16 to 16.
                        pass
            
            # Rethink: Compute eps_G for the full difference range.
            # Max diff is 16.
            # Let's create a larger G array for eps calculation.
            # Or just compute the sum on the fly for missing indices?
            # Better: Precompute eps_G for all (dx, dy) in [-16, 16].
            
            # I will restructure the calculation to compute eps_G for the extended set.
            
            pass # Placeholder
            
        # This function is getting complex. Let's optimize the G-difference handling.
        
        return 0.0 # Placeholder

    # Refactored calculate_gap
    def calculate_gap_v2(cell):
        cell_arr = np.array(cell, dtype=float)
        i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        x_pix = (i_idx + 0.5) / N
        y_pix = (j_idx + 0.5) / N
        
        # Extended G range for differences
        g_ext = np.arange(-16, 17)
        Gx_ext, Gy_ext = np.meshgrid(g_ext, g_ext, indexing='ij')
        Gx_ext = Gx_ext.flatten()
        Gy_ext = Gy_ext.flatten()
        num_ext = len(Gx_ext)
        
        # Compute eps_G for extended set
        eps_G_ext = np.zeros(num_ext, dtype=complex)
        for k in range(num_ext):
            gx = Gx_ext[k]
            gy = Gy_ext[k]
            phase = np.exp(-1j * 2 * np.pi * (gx * x_pix + gy * y_pix))
            eps_G_ext[k] = np.sum(cell_arr * phase) / (N * N)
            
        eps_G_ext = eps_G_ext * (eps_high - eps_low)
        # Add background
        mask0 = (Gx_ext == 0) & (Gy_ext == 0)
        eps_G_ext[mask0] += eps_low
        eps_G_ext = np.real(eps_G_ext)
        
        # Map from (gx, gy) to index in ext set
        g_map_ext = {}
        for k in range(num_ext):
            g_map_ext[(Gx_ext[k], Gy_ext[k])] = k
            
        # Basis G-vectors
        # Gx, Gy are -8 to 8
        
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for kx, ky in k_points_scaled:
            # Construct A
            A = np.zeros((num_pw, num_pw))
            D = np.zeros((num_pw, num_pw))
            
            for m in range(num_pw):
                # D diagonal
                D[m, m] = (kx + Gx[m])**2 + (ky + Gy[m])**2
                
                for n in range(num_pw):
                    dx = Gx[m] - Gx[n]
                    dy = Gy[m] - Gy[n]
                    idx = g_map_ext.get((dx, dy), -1)
                    if idx != -1:
                        eps_val = eps_G_ext[idx]
                        if abs(eps_val) < 1e-9:
                            A[m, n] = 0 # Avoid division by zero
                        else:
                            A[m, n] = (dx**2 + dy**2) / (4 * np.pi**2 * eps_val)
                    else:
                        A[m, n] = 0
                        
                    if m == n:
                        A[m, n] += 1.0 # Delta term
            
            # Solve A c = lambda D c
            # Use scipy.linalg.eigh
            try:
                from scipy.linalg import eigh
                # eigh(A, D)
                # D is diagonal and positive definite
                vals, vecs = eigh(A, D)
                # Sort eigenvalues
                vals = np.sort(vals)
                
                # Bands are indexed 1..N
                # We want band 4 and 5
                # Index 3 and 4 (0-based)
                if len(vals) >= 5:
                    omega_4 = np.sqrt(vals[3])
                    omega_5 = np.sqrt(vals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except Exception:
                continue
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        mid = (min_omega_5 + max_omega_4) / 2.0
        if mid == 0:
            return 0.0
        return gap / mid

    # Generate Initial Candidates
    def make_cell(bits):
        # bits is 16x16 array (upper triangle)
        cell = [[0]*N for _ in range(N)]
        for i in range(16):
            for j in range(i, 16):
                val = bits[i, j]
                # Top-Left
                cell[i][j] = val
                cell[j][i] = val
                # Top-Right (Vertical Flip)
                cell[i][31-j] = val
                cell[j][31-i] = val
                # Bottom-Left (Horizontal Flip)
                cell[31-i][j] = val
                cell[31-j][i] = val
                # Bottom-Right (Both)
                cell[31-i][31-j] = val
                cell[31-j][31-i] = val
        return cell

    def rod_cell(radius):
        bits = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(i, 16):
                # Map to centered coordinates
                # i, j are 0..15.
                # Center is 15.5.
                # x = (j + 0.5) / 32 - 0.5
                # y = (i + 0.5) / 32 - 0.5
                x = (j + 0.5) / 32.0 - 0.5
                y = (i + 0.5) / 32.0 - 0.5
                if x*x + y*y < radius*radius:
                    bits[i, j] = 1
        return make_cell(bits)

    def plus_cell(width):
        bits = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(i, 16):
                x = (j + 0.5) / 32.0 - 0.5
                y = (i + 0.5) / 32.0 - 0.5
                # Plus shape: |x| < w or |y| < w
                if abs(x) < width or abs(y) < width:
                    bits[i, j] = 1
        return make_cell(bits)

    def square_cell(half_side):
        bits = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(i, 16):
                x = (j + 0.5) / 32.0 - 0.5
                y = (i + 0.5) / 32.0 - 0.5
                if abs(x) < half_side and abs(y) < half_side:
                    bits[i, j] = 1
        return make_cell(bits)

    candidates = []
    candidates.append(rod_cell(0.28))
    candidates.append(rod_cell(0.32))
    candidates.append(plus_cell(0.15))
    candidates.append(plus_cell(0.20))
    candidates.append(square_cell(0.25))
    candidates.append(square_cell(0.30))
    
    best_cell = candidates[0]
    best_score = -10.0
    
    # Evaluate Candidates
    for c in candidates:
        if time.time() > deadline:
            break
        score = calculate_gap_v2(c)
        if score > best_score:
            best_score = score
            best_cell = c
            
    # Local Search
    # Extract bits from best_cell
    def extract_bits(cell):
        bits = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(i, 16):
                bits[i, j] = cell[i][j]
        return bits

    current_bits = extract_bits(best_cell)
    current_score = best_score
    
    # Simple Hill Climbing
    while time.time() < deadline:
        # Pick a random bit to flip
        i = np.random.randint(0, 16)
        j = np.random.randint(i, 16)
        
        old_val = current_bits[i, j]
        current_bits[i, j] = 1 - old_val
        
        new_cell = make_cell(current_bits)
        new_score = calculate_gap_v2(new_cell)
        
        if new_score > current_score:
            current_score = new_score
            best_cell = new_cell
            best_score = current_score
        else:
            current_bits[i, j] = old_val
            
    return best_cell

# EVOLVE-BLOCK-END
