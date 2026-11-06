"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_tm_bands(cell, n_pw=9, k_points=None):
    """Compute TM band structure using plane-wave method.
    
    Args:
        cell: 2D numpy array (0/1) representing dielectric mask
        n_pw: number of plane waves in each direction (total = n_pw^2)
        k_points: list of (kx, ky) in units of 2π/a, default covers irreducible BZ
    
    Returns:
        List of band frequencies (omega/k0*a) for each k-point, sorted
    """
    size = cell.shape[0]
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    # epsilon(x,y) = eps_low + (eps_high - eps_low) * cell
    eps_field = eps_low + (eps_high - eps_low) * eps
    
    # Compute Fourier coefficients of epsilon
    # eps_G = (1/N^2) sum_{i,j} eps_field[i,j] exp(-i G·r_{ij})
    # G = (m*2π, n*2π) for m,n in [-Npw/2, Npw/2)
    N = n_pw
    if N % 2 == 1:
        half = N // 2
        pw_idx = np.arange(-half, half + 1)
    else:
        half = N // 2
        pw_idx = np.arange(-half, half)
    
    # Compute 2D FFT of epsilon field
    # We need eps_G for G values that are differences of plane wave vectors
    # The full set of G needed goes from -2*half to 2*half in each direction
    # Use FFT: compute FFT of eps_field, then extract needed coefficients
    
    # Pad to size 2*N for convolution-like operations
    # Actually, let's just compute the needed Fourier coefficients directly
    # using the discrete sum
    
    # For efficiency, use FFT. The DFT gives us coefficients for all G up to N/2.
    # We need G up to 2*half for the products.
    # Let's compute a larger FFT.
    
    fft_size = 2 * size  # oversample to get more Fourier components
    eps_padded = np.zeros((fft_size, fft_size), dtype=float)
    eps_padded[:size, :size] = eps_field
    
    # Compute 2D FFT
    eps_fft = np.fft.fft2(eps_padded)
    eps_fft /= (fft_size * fft_size)  # normalize
    
    # Map FFT indices to G values
    # FFT index k corresponds to G = k * 2π/fft_size
    # We need G in units of 2π/a, where a is the lattice constant
    # Since our cell is 1 unit, G = (m, n) in units of 2π
    # FFT index i corresponds to frequency i/fft_size cycles per unit
    # So G_i = i / fft_size * 2π = i * 2π / fft_size
    # In units of 2π, G_i = i / fft_size
    
    # For plane wave index m (in units of 2π), we need eps at G = m
    # FFT index for G = m is: i = m * fft_size / 2π * 2π = m * fft_size
    # Wait, let me be more careful.
    
    # The plane wave basis is exp(i * 2π * (m*x + n*y))
    # The Fourier coefficient is eps_{mn} = ∫∫ eps(x,y) exp(-i*2π*(mx+ny)) dxdy
    # Discretely: eps_{mn} ≈ (1/N_pix^2) sum_{i,j} eps[i,j] exp(-i*2π*(m*i/N_pix + n*j/N_pix))
    
    # This is exactly the DFT of eps_field (normalized).
    # np.fft.fft2 gives: F[k1,k2] = sum_{i,j} f[i,j] exp(-i*2π*(k1*i/N + k2*j/N))
    # So eps_{mn} = F[m mod N, n mod N] / N^2
    
    # But we need m,n to range over a wider set (up to 2*half for differences)
    # Let's just compute them directly for the needed range
    
    # Needed G values: for plane waves with indices in pw_idx, differences go from
    # -2*half to 2*half
    g_max = 2 * half
    
    # Compute all needed Fourier coefficients
    # For each (m, n) in [-g_max, g_max]^2
    # eps_{mn} = (1/size^2) sum_{i,j} eps_field[i,j] exp(-i*2π*(m*i/size + n*j/size))
    
    # Use FFT for efficiency: compute FFT of size (2*g_max+1) x (2*g_max+1)?
    # No, the field is on a size x size grid. Let's just use the DFT directly
    # but vectorized.
    
    # Actually, the cleanest way: compute the FFT of the field on its native grid,
    # which gives coefficients for m,n in [-size/2, size/2). Since size=32 and
    # g_max is at most 2*(9//2)=8 for n_pw=9, we're fine.
    
    eps_fft_native = np.fft.fft2(eps_field) / (size * size)
    
    def get_eps_coeff(m, n):
        """Get Fourier coefficient eps_{mn}."""
        # Handle wrapping for the FFT
        m_wrapped = m % size
        n_wrapped = n % size
        return eps_fft_native[m_wrapped, n_wrapped]
    
    # Default k-points: cover the irreducible Brillouin zone
    if k_points is None:
        # Irreducible BZ for square lattice: 0 <= kx <= ky, 0 <= kx + ky <= 1
        # Key points: Gamma(0,0), X(0.5,0), M(0.5,0.5), and points along X-M and Gamma-M
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.25, 0.0),     
            (0.5, 0.0),      # X
            (0.25, 0.25),    
            (0.5, 0.25),     
            (0.5, 0.5),      # M
            (0.125, 0.0),
            (0.375, 0.0),
            (0.125, 0.125),
            (0.375, 0.125),
            (0.375, 0.375),
            (0.25, 0.0),
            (0.5, 0.125),
            (0.5, 0.375),
        ]
    
    all_bands = []
    
    for kx, ky in k_points:
        # Build the Hamiltonian matrix for TM polarization
        # For TM (E_z): the eigenvalue equation in plane-wave form is:
        # sum_{G'} [(k+G)^2 * delta_{G,G'} + k^2 * (eps_{G-G'} - 1)] a_{G'} = 0
        # 
        # More standard form (from Joannopoulos):
        # The TM eigenvalue problem is:
        # sum_{G'} [ delta_{G,G'} * (k+G)^2 + k^2 * eps_{G-G'} ] a_{G'} = lambda * (k+G)^2 * a_G
        # 
        # Actually, the cleanest formulation:
        # For TM: [∇_⊥^2 + k_0^2 ε(r)] E_z = 0
        # In plane waves: sum_{G'} [(k+G)^2 - k_0^2 ε_{G-G'}] a_{G'} = 0
        # This is a generalized eigenvalue problem: A a = λ B a
        # where A_{G,G'} = (k+G)^2 δ_{G,G'} - k_0^2 ε_{G-G'}
        # and B = I (identity)
        # So we just need to find eigenvalues of A.
        # 
        # Wait, that's not right either. Let me use the standard formulation.
        # 
        # The standard plane-wave method for TM polarization:
        # The dielectric function ε(r) is periodic. Expand E_z = sum_G a_G exp(i(k+G)·r)
        # The wave equation: ∇_⊥^2 E_z + k_0^2 ε(r) E_z = 0
        # 
        # ∇_⊥^2 exp(i(k+G)·r) = -(k+G)^2 exp(i(k+G)·r)
        # ε(r) = sum_{G'} ε_{G'} exp(i G'·r)
        # 
        # So: sum_{G'} [ -(k+G')^2 δ_{G,G'} + k_0^2 ε_{G-G'} ] a_{G'} = 0
        # 
        # This gives: sum_{G'} [ (k+G')^2 δ_{G,G'} - k_0^2 ε_{G-G'} ] a_{G'} = 0
        # 
        # For this to have non-trivial solutions: det(...) = 0
        # The eigenvalues are k_0^2 = (ω/c)^2
        # 
        # So we solve: sum_{G'} [ (k+G')^2 δ_{G,G'} - λ ε_{G-G'} ] a_{G'} = 0
        # where λ = k_0^2
        # 
        # This is a standard eigenvalue problem: H a = λ a
        # where H_{G,G'} = (k+G)^2 δ_{G,G'} / ε_{G-G'} ... no.
        # 
        # Let me write it as: M a = λ a where
        # M_{G,G'} = (k+G)^2 δ_{G,G'} - ε_{G-G'}
        # and λ = k_0^2
        # 
        # But this isn't symmetric. Let me use the symmetric form.
        # 
        # Actually, the standard approach is:
        # Multiply by 1/ε or use a different formulation.
        # 
        # The most common formulation for TM:
        # sum_{G'} [ δ_{G,G'} + (k_0^2)/( (k+G)^2 (k+G')^2 ) * (ε_{G-G'} - δ_{G,G'}) ] a_{G'} = 0
        # 
        # No, let me just use the direct formulation and solve the eigenvalue problem.
        # 
        # From the equation: sum_{G'} [ (k+G')^2 δ_{G,G'} - k_0^2 ε_{G-G'} ] a_{G'} = 0
        # 
        # This is: [D - k_0^2 E] a = 0
        # where D_{G,G'} = (k+G)^2 δ_{G,G'} and E_{G,G'} = ε_{G-G'}
        # 
        # For non-trivial solutions: det(D - k_0^2 E) = 0
        # This is a generalized eigenvalue problem: D a = k_0^2 E a
        # 
        # So: E^{-1} D a = k_0^2 a  (if E is invertible)
        # 
        # Let's solve: np.linalg.eigh(E^{-1} D) or use scipy's eig for generalized problems.
        # 
        # Actually, let me just build the matrix and use scipy.linalg.eig.
        
        n_pw_total = len(pw_idx)
        G_vectors = []
        for m in pw_idx:
            for n in pw_idx:
                G_vectors.append((m, n))
        
        # Build matrix M where M_{G,G'} = (k+G)^2 δ_{G,G'} - ε_{G-G'}
        # We want eigenvalues of E^{-1} D, i.e., solve D a = λ E a
        
        D = np.zeros((n_pw_total, n_pw_total))
        E_mat = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        
        for i, (mi, ni) in enumerate(G_vectors):
            for j, (mj, nj) in enumerate(G_vectors):
                # D is diagonal: D_{ii} = (kx + mi)^2 + (ky + ni)^2
                if i == j:
                    D[i, i] = (kx + mi)**2 + (ky + ni)**2
                # E_{ij} = ε_{(mi-mj), (ni-nj)}
                dm = mi - mj
                dn = ni - nj
                E_mat[i, j] = get_eps_coeff(dm, dn)
        
        # Solve generalized eigenvalue problem: D a = λ E a
        # Use scipy
        from scipy.linalg import eig
        try:
            # Make E Hermitian (it should be, since ε_{-G} = ε_G^*)
            E_mat = (E_mat + E_mat.conj().T) / 2
            D = (D + D.T) / 2
            
            # Solve E a = μ D a, then λ = 1/μ
            # Or directly: D a = λ E a
            # Use eig for generalized problem
            vals, vecs = eig(D, E_mat)
            
            # Eigenvalues λ = k_0^2, so ω/k_0 = sqrt(λ)
            # We want the bands sorted by frequency
            real_vals = np.real(vals)
            # Filter out negative or very small values
            real_vals = real_vals[real_vals > 1e-10]
            freqs = np.sqrt(real_vals)
            freqs = np.sort(freqs)
            
            # Normalize: the lattice constant a = 1, so frequencies are in units of 2π/a
            # Actually, k_0 = ω/c, and the wavevector is in units of 2π/a
            # So ω = k_0 * c, and the frequency in units of c/a is k_0 * a / (2π) * 2π = k_0 * a
            # Since a = 1: ω/(c/a) = k_0 * a = sqrt(λ)
            # But we need to be careful with the 2π factors.
            # 
            # In our formulation, k and G are in units of 2π/a (i.e., dimensionless)
            # So k_0^2 = λ, and ω/(c/a) = k_0 * a = sqrt(λ) * a = sqrt(λ) since a=1
            # But wait, the standard normalization: if k is in units of 2π/a, then
            # the actual wavevector is k * 2π/a. So k_0 = sqrt(λ) * 2π/a
            # And ω = c * k_0 = c * sqrt(λ) * 2π/a
            # In units of c/a: ω/(c/a) = 2π * sqrt(λ)
            
            # For the gap ratio, the 2π factor cancels, so we can just use sqrt(λ)
            all_bands.append(freqs)
        except Exception as e:
            # If eigenvalue computation fails, skip this k-point
            all_bands.append(np.array([]))
    
    return all_bands

def _compute_gap_from_bands(all_bands):
    """Compute the 4-5 band gap ratio from computed bands.
    
    Returns: 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
    """
    min_omega5 = np.inf
    max_omega4 = -np.inf
    
    for bands in all_bands:
        if len(bands) < 5:
            continue
        # bands[0], bands[1], bands[2], bands[3], bands[4] are the first 5 bands
        # Band 4 is bands[3] (0-indexed), Band 5 is bands[4]
        omega4 = bands[3]
        omega5 = bands[4]
        max_omega4 = max(max_omega4, omega4)
        min_omega5 = min(min_omega5, omega5)
    
    if max_omega4 == -np.inf or min_omega5 == np.inf:
        return -1.0  # invalid
    
    if min_omega5 <= max_omega4:
        return -1.0  # negative gap
    
    gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap_ratio

def _make_checkerboard_cell(size, variant=0, param=0.5):
    """Generate checkerboard-like patterns known for large photonic band gaps.
    
    variant: different checkerboard patterns
    param: controls the pattern
    """
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    
    if variant == 0:
        # Standard checkerboard with adjustable period
        period = max(2, int(size * param))
        for i in range(size):
            for j in range(size):
                block_i = i // (period // 2)
                block_j = j // (period // 2)
                if (block_i + block_j) % 2 == 0:
                    cell[i, j] = 1
    elif variant == 1:
        # Cross/diamond checkerboard
        for i in range(size):
            for j in range(size):
                # Diamond pattern
                dist = abs(i - half + 0.5) + abs(j - half + 0.5)
                threshold = size * param
                if dist < threshold and ((i + j) % 2 == 0):
                    cell[i, j] = 1
    elif variant == 2:
        # Four-rod with checkerboard modulation
        for i in range(size):
            for j in range(size):
                # Distance to nearest of 4 centers
                best_dist = 1e10
                for cx in [half * 0.5, half * 1.5]:
                    for cy in [half * 0.5, half * 1.5]:
                        dx = i - cx
                        dy = j - cy
                        dist2 = dx*dx + dy*dy
                        if dist2 < best_dist:
                            best_dist = dist2
                radius = size * param * 0.3
                if best_dist < radius**2:
                    # Modulate with checkerboard
                    if ((i // 4) + (j // 4)) % 2 == 0:
                        cell[i, j] = 1
    elif variant == 3:
        # Concentric rings with alternating fill
        for i in range(size):
            for j in range(size):
                dx = i - half + 0.5
                dy = j - half + 0.5
                r = np.sqrt(dx*dx + dy*dy)
                ring = int(r / (size * 0.1))
                if ring % 2 == 0:
                    cell[i, j] = 1
    elif variant == 4:
        # "Fishnet" pattern: dielectric background with air holes in checkerboard
        for i in range(size):
            for j in range(size):
                # Air holes at checkerboard positions
                period = max(4, int(size * 0.25))
                hole_r = period * 0.3
                in_hole = False
                for cx in np.arange(0, size, period):
                    for cy in np.arange(0, size, period):
                        dx = i - cx
                        dy = j - cy
                        if dx*dx + dy*dy < hole_r**2:
                            in_hole = True
                if not in_hole:
                    cell[i, j] = 1
    elif variant == 5:
        # Star/flower pattern
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                theta = np.arctan2(y, x)
                # 4-fold symmetric modulation
                mod = 1.0 + 0.3 * np.cos(4 * theta)
                radius = param * 0.4 * mod
                if r < radius:
                    cell[i, j] = 1
    elif variant == 6:
        # Multi-scale: large dielectric with small air holes
        for i in range(size):
            for j in range(size):
                # Large dielectric region
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                if r > param * 0.45:
                    cell[i, j] = 0
                    continue
                # Small air holes in checkerboard
                period = 4
                in_hole = False
                for cx in np.arange(-0.5, 0.5, period/size):
                    for cy in np.arange(-0.5, 0.5, period/size):
                        dx = x - cx
                        dy = y - cy
                        if dx*dx + dy*dy < (period*0.15/size)**2:
                            in_hole = True
                if in_hole:
                    cell[i, j] = 0
                else:
                    cell[i, j] = 1
    elif variant == 7:
        # Square lattice of dielectric rods with optimized shape
        for i in range(size):
            for j in range(size):
                best_dist = 1e10
                for cx in [-0.25, 0.25]:
                    for cy in [-0.25, 0.25]:
                        dx = (i + 0.5)/size - 0.5 - cx
                        dy = (j + 0.5)/size - 0.5 - cy
                        dist2 = dx*dx + dy*dy
                        if dist2 < best_dist:
                            best_dist = dist2
                # Superellipse shape
                x = (i + 0.5)/size - 0.5
                y = (j + 0.5)/size - 0.5
                # Find nearest center
                min_dx = min(abs(x - (-0.25)), abs(x - 0.25))
                min_dy = min(abs(y - (-0.25)), abs(y - 0.25))
                n_exp = 3.0  # more square-like
                if (min_dx/(param*0.2))**n_exp + (min_dy/(param*0.2))**n_exp < 1.0:
                    cell[i, j] = 1
    
    # Enforce symmetry
    cell = np.minimum(cell, cell[::-1, :])
    cell = np.minimum(cell, cell[:, ::-1])
    cell = np.minimum(cell, cell.T)
    
    return cell

def _make_harmonic_cell(size, lam, thresh, n_rods=1):
    """Generate cell from harmonic polynomial."""
    cell = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            if n_rods == 1:
                x2 = x*x
                y2 = y*y
                harmonic = x2*x2 + y2*y2 - 2*lam*x*y*(x2+y2)
                if harmonic < thresh**4:
                    cell[i, j] = 1
            else:
                # Multiple rods
                best_dist = 1.0
                for cx in [-0.25, 0.25]:
                    for cy in [-0.25, 0.25]:
                        dx = x - cx
                        dy = y - cy
                        dist2 = dx*dx + dy*dy
                        if dist2 < best_dist:
                            best_dist = dist2
                mod = 1.0 + 0.3 * lam * np.cos(4 * np.arctan2(y, x))
                if best_dist < (thresh * mod) ** 2:
                    cell[i, j] = 1
    
    cell = np.minimum(cell, cell[::-1, :])
    cell = np.minimum(cell, cell[:, ::-1])
    cell = np.minimum(cell, cell.T)
    
    return cell

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s safety margin
    
    best_cell = None
    best_score = -np.inf
    
    # Generate candidate cells from different design families
    candidates = []
    
    # Checkerboard variants
    for variant in range(8):
        for param in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 60.0:
                break
            cell = _make_checkerboard_cell(size, variant, param)
            candidates.append((f"checker_{variant}_{param}", cell))
    
    # Harmonic cells
    for lam in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for thresh in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            if time.time() > deadline - 60.0:
                break
            cell = _make_harmonic_cell(size, lam, thresh, n_rods=1)
            candidates.append((f"harmonic_{lam}_{thresh}", cell))
    
    # Multi-rod harmonic cells
    for lam in [0.0, 0.3, 0.6]:
        for thresh in [0.1, 0.15, 0.2, 0.25]:
            if time.time() > deadline - 60.0:
                break
            cell = _make_harmonic_cell(size, lam, thresh, n_rods=4)
            candidates.append((f"multirod_{lam}_{thresh}", cell))
    
    # Evaluate each candidate with the reduced plane-wave solver
    # Use a small basis for speed during search
    n_pw_search = 7  # 49 plane waves total
    
    for name, cell in candidates:
        if time.time() > deadline - 30.0:
            break
        
        cell_arr = np.array(cell)
        bands = _compute_tm_bands(cell_arr, n_pw=n_pw_search)
        score = _compute_gap_from_bands(bands)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            print(f"  New best: {name}, score={score:.4f}")
    
    # If we have time, do a local search around the best cell
    if best_cell is not None and time.time() < deadline - 20.0:
        # Try flipping individual pixels near boundaries to improve the gap
        best_arr = np.array(best_cell)
        improved = True
        max_iterations = 50
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 15.0:
                break
            
            # Find boundary pixels
            boundary_pixels = []
            for i in range(size):
                for j in range(size):
                    is_boundary = False
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if best_arr[ni, nj] != best_arr[i, j]:
                                is_boundary = True
                                break
                    if is_boundary:
                        boundary_pixels.append((i, j))
            
            # Try flipping each boundary pixel
            best_flip_score = best_score
            best_flip = None
            
            for i, j in boundary_pixels[:20]:  # Limit to 20 pixels per iteration
                if time.time() > deadline - 15.0:
                    break
                
                # Try flipping
                test_arr = best_arr.copy()
                test_arr[i, j] = 1 - test_arr[i, j]
                
                # Check symmetry after flip
                if not (np.array_equal(test_arr, test_arr[::-1, :]) and 
                        np.array_equal(test_arr, test_arr[:, ::-1]) and
                        np.array_equal(test_arr, test_arr.T)):
                    continue
                
                bands = _compute_tm_bands(test_arr, n_pw=n_pw_search)
                score = _compute_gap_from_bands(bands)
                
                if score > best_flip_score:
                    best_flip_score = score
                    best_flip = (i, j)
            
            if best_flip is not None:
                best_arr[best_flip[0], best_flip[1]] = 1 - best_arr[best_flip[0], best_flip[1]]
                best_score = best_flip_score
                best_cell = best_arr.tolist()
            else:
                break
    
    if best_cell is None:
        # Fallback: simple checkerboard
        best_cell = _make_checkerboard_cell(size, 0, 0.5)
    
    return best_cell
# EVOLVE-BLOCK-END
