"""Photonic crystal unit cell design using only numpy and standard library."""
import time
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal cell maximizing TM band 4-5 gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210  # 210s budget, 20s margin for safety
    
    # Precompute plane wave vectors (289 = 17x17, indices -8..8)
    # The problem statement says 289 plane waves. 17x17 = 289.
    n_pw = 17
    pw_indices = np.arange(-n_pw // 2, n_pw // 2 + 1)  # -8..8
    G = np.array([(gx, gy) for gy in pw_indices for gx in pw_indices], dtype=float)
    n_G = len(G)
    
    # Precompute |G|^2
    G2 = np.sum(G**2, axis=1)
    
    # Precompute mapping from G vector to index
    G_to_idx = {tuple(g): i for i, g in enumerate(G)}
    
    # Precompute difference indices for B matrix construction
    # B[i, j] = eps_coeffs[G[i] - G[j]]
    # We need to know which eps_coeffs index corresponds to G[i]-G[j]
    # If G[i]-G[j] is not in G_to_idx, the coefficient is 0 (outside the truncation set)
    # However, for a finite truncation, we usually assume periodicity or that coefficients outside are 0.
    # Given the "exact Fourier coefficients of the pixel map" phrasing, 
    # and the use of a specific set of plane waves, we assume the basis is the 289 vectors.
    # Differences G[i]-G[j] can range from -16 to 16.
    # Our G vectors range from -8 to 8.
    # So differences outside [-8, 8] are not in our basis.
    
    # Build index matrix for B construction
    # diff_idx[i, j] = index in eps_coeffs of (G[i] - G[j]), or -1 if not present
    diff_idx = np.full((n_G, n_G), -1, dtype=int)
    for i in range(n_G):
        for j in range(n_G):
            diff = tuple(G[i] - G[j])
            if diff in G_to_idx:
                diff_idx[i, j] = G_to_idx[diff]
    
    # Compute Fourier coefficients of epsilon for the pixel map
    # We need 1/epsilon for TM mode? 
    # Standard PWE for TM:
    # The equation is: sum_G' (k+G)·(k+G') / epsilon_{G-G'} c_{G'} = omega^2 c_G
    # This is a generalized eigenvalue problem A c = omega^2 B c
    # where A is diagonal with (k+G)^2 and B_{G,G'} = (1/epsilon)_{G-G'}
    # Wait, let's verify the formulation.
    # Maxwell's equations for TM (H_z non-zero):
    # (1/epsilon(r)) grad x (grad x H) = (omega/c)^2 H
    # In Fourier space:
    # sum_G' (k+G)·(k+G') (1/epsilon)_{G-G'} c_{G'} = omega^2 c_G
    # So yes, B contains Fourier coefficients of 1/epsilon.
    
    def compute_inv_eps_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon(x,y) for the pixel map."""
        # epsilon(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
        # inv_eps(x,y) = 1 / epsilon(x,y)
        # inv_eps_G = (1/A) sum_pixels inv_eps(pixel) * exp(-i G . r_pixel) * dA
        
        inv_eps_coeffs = np.zeros(n_G, dtype=complex)
        dx = 1.0 / size
        
        # Precompute pixel centers
        x_centers = (np.arange(size) + 0.5 - size / 2) / size
        y_centers = (np.arange(size) + 0.5 - size / 2) / size
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        # Compute 1/epsilon map
        eps_map = eps_low + (eps_high - eps_low) * cell
        inv_eps_map = 1.0 / eps_map
        
        for idx, (gx, gy) in enumerate(G):
            phase = np.exp(-1j * (gx * 2 * np.pi * X + gy * 2 * np.pi * Y))
            inv_eps_coeffs[idx] = np.sum(inv_eps_map * phase) * dx * dx
        
        return inv_eps_coeffs
    
    # Build the B matrix (Hermitian) from inv_eps_coeffs
    def build_B_matrix(inv_eps_coeffs):
        """Build the B matrix where B[i,j] = inv_eps_coeffs[G[i]-G[j]]."""
        B = np.zeros((n_G, n_G), dtype=complex)
        # Use the precomputed diff_idx
        # B[i,j] = inv_eps_coeffs[diff_idx[i,j]] if diff_idx[i,j] != -1 else 0
        valid = diff_idx != -1
        B[valid] = inv_eps_coeffs[diff_idx[valid]]
        return B
    
    # Compute eigenvalues for a single k-point
    def compute_bands_single_k(kx, ky, inv_eps_coeffs):
        """Compute eigenvalues for a single k-point."""
        k_vec = np.array([kx, ky])
        k_plus_G = k_vec + G
        k2 = np.sum(k_plus_G**2, axis=1)
        
        # A is diagonal with k2
        # B is Hermitian matrix of 1/epsilon coefficients
        B = build_B_matrix(inv_eps_coeffs)
        
        # Solve A c = omega^2 B c
        # This is equivalent to B^{-1} A c = omega^2 c
        # Or B^{-1/2} A B^{-1/2} c' = omega^2 c'
        
        # Since B is Hermitian positive definite (1/epsilon > 0), we can use eigh.
        # numpy.linalg.eigh supports complex Hermitian matrices.
        # However, A is real diagonal. 
        # Let's use the generalized eigenvalue solver if available, or transform.
        # numpy.linalg.eigh doesn't do generalized directly in older versions, 
        # but scipy does. Since scipy is broken, we must transform.
        
        # Transform to standard eigenvalue problem:
        # L L^H = B (Cholesky decomposition)
        # Then L^{-H} A L^{-1} c' = omega^2 c'
        
        try:
            L = np.linalg.cholesky(B)
            # A' = L^{-1} A L^{-H}
            # A is diagonal, so A'_{ij} = (L^{-1})_{i, :} dot diag(k2) dot (L^{-H})_{:, j}
            # This is computationally expensive if done naively.
            
            # Alternative: Use scipy.linalg.eigh equivalent via numpy?
            # No direct generalized eigh in numpy.
            
            # Let's use the transformation:
            # M = B^{-1} A
            # M is not Hermitian.
            
            # Better: 
            # C = B^{-1/2} A B^{-1/2}
            # We need B^{-1/2}. 
            # L = chol(B). B = L L^H. B^{-1/2} = L^{-H} (L L^H)^{-1/2} ... no.
            # If B = L L^H, then B^{-1} = L^{-H} L^{-1}.
            # B^{-1/2} is the Hermitian square root inverse.
            
            # Simpler approach for small matrices (289x289):
            # Compute eigenvalues of B.
            # vals_B, vecs_B = np.linalg.eigh(B)
            # B^{-1/2} = vecs_B @ diag(1/sqrt(vals_B)) @ vecs_B^H
            # C = B^{-1/2} @ diag(k2) @ B^{-1/2}
            # vals_C = np.linalg.eigvalsh(C)
            
            vals_B, vecs_B = np.linalg.eigh(B)
            # Ensure positive
            vals_B = np.maximum(vals_B, 1e-12)
            inv_sqrt_vals_B = 1.0 / np.sqrt(vals_B)
            
            # B_inv_sqrt = vecs_B * inv_sqrt_vals_B @ vecs_B^H
            # C = B_inv_sqrt @ diag(k2) @ B_inv_sqrt
            # C = (vecs_B * inv_sqrt_vals_B) @ (vecs_B^H @ diag(k2) @ vecs_B^H) ... no
            # C = V D V^H @ K @ V D V^H where V=vecs_B, D=diag(inv_sqrt)
            # C = V (D V^H K V D) V^H
            # Let's just compute C explicitly.
            
            # V is (n, n), D is (n, n) diagonal
            # V_D = V * inv_sqrt_vals_B (broadcasting along rows)
            # C = V_D @ (V_D^H * k2[None, :]) ? No.
            # K = diag(k2)
            # C = V D V^H K V D V^H
            # Let M = V^H K V. This is a matrix.
            # M_{ij} = sum_k V^*_{k i} k2_k V_{k j}
            # C = V D M D V^H
            # Eigenvalues of C are eigenvalues of D M D (similar matrix)
            # D M D is Hermitian.
            
            # Compute M = V^H @ diag(k2) @ V
            # M = (V.conj().T * k2[None, :]) @ V  -- Broadcasting k2 along rows of V.conj().T
            # V.conj().T is (n, n). k2 is (n,).
            # (V.conj().T * k2) multiplies each column i by k2_i? No.
            # V.conj().T [i, k] * k2[k] -> sum over k.
            # We want M_{ij} = sum_k V^*_{ki} k2_k V_{kj}
            # This is (V.conj().T * k2[None, :]) @ V
            # Let's check dimensions:
            # V.conj().T: (n, n)
            # k2[None, :]: (1, n)
            # Product: (n, n) where element [i, k] is V^*_{ki} * k2_k
            # Multiply by V: (n, n) @ (n, n) -> sum over k.
            # Result [i, j] = sum_k (V^*_{ki} * k2_k) * V_{kj} = M_{ij}. Correct.
            
            M = (vecs_B.conj().T * k2[None, :]) @ vecs_B
            
            # Now compute D M D
            # D is diagonal with inv_sqrt_vals_B
            # D M D is Hermitian.
            # We can compute its eigenvalues.
            # C' = inv_sqrt_vals_B[:, None] * M * inv_sqrt_vals_B[None, :]
            
            C_prime = inv_sqrt_vals_B[:, None] * M * inv_sqrt_vals_B[None, :]
            
            # Ensure Hermitian (numerical errors)
            C_prime = (C_prime + C_prime.conj().T) / 2.0
            
            eigvals = np.linalg.eigvalsh(C_prime)
            
            return eigvals
            
        except np.linalg.LinAlgError:
            return np.array([np.inf] * n_G)

    # Compute band structure for a given cell
    def compute_bands(cell, n_kpoints=30):
        """Compute bands 4 and 5 over the irreducible BZ."""
        inv_eps_coeffs = compute_inv_eps_fourier(cell)
        
        # Define 30 k-points covering the irreducible BZ
        # Irreducible BZ for square lattice: 0 <= ky <= kx, kx, ky in [0, 0.5]
        # Key points: Gamma (0,0), X (0.5,0), M (0.5,0.5)
        # Path: Gamma->X->M->Gamma
        
        k_points = []
        # Gamma to X: (t/2, 0)
        for t in np.linspace(0, 1, 8):
            k_points.append((t / 2, 0))
        # X to M: (0.5, t/2)
        for t in np.linspace(0, 1, 8):
            k_points.append((0.5, t / 2))
        # M to Gamma: (0.5*(1-t), 0.5*(1-t))
        for t in np.linspace(0, 1, 8):
            k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
        # Some interior points
        k_points.append((0.25, 0.1))
        k_points.append((0.3, 0.15))
        k_points.append((0.2, 0.2))
        
        k_points = k_points[:n_kpoints]
        
        band4_max = 0
        band5_min = np.inf
        
        for kx, ky in k_points:
            try:
                eigvals = compute_bands_single_k(kx, ky, inv_eps_coeffs)
                # Take bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigvals) > 4:
                    w4 = eigvals[3]
                    w5 = eigvals[4]
                    band4_max = max(band4_max, w4)
                    band5_min = min(band5_min, w5)
            except Exception:
                continue
        
        if band5_min == np.inf:
            return -1.0
        
        gap = band5_min - band4_max
        midgap = (band5_min + band4_max) / 2
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    def enforce_symmetry(cell):
        """Enforce 4-fold symmetry about the center."""
        size = cell.shape[0]
        new_cell = cell.copy()
        for i in range(size):
            for j in range(size):
                # Orbit of (i,j) under D4 symmetry
                # Generators: flip x, flip y, transpose
                orbit = set()
                ii, jj = i, j
                # Apply all 8 symmetries
                orbit.add((ii, jj))
                orbit.add((size - 1 - ii, jj))
                orbit.add((ii, size - 1 - jj))
                orbit.add((size - 1 - ii, size - 1 - jj))
                orbit.add((jj, ii))
                orbit.add((size - 1 - jj, ii))
                orbit.add((jj, size - 1 - ii))
                orbit.add((size - 1 - jj, size - 1 - ii))
                
                values = [cell[a, b] for a, b in orbit]
                majority = 1 if sum(values) > len(values) // 2 else 0
                for a, b in orbit:
                    new_cell[a, b] = majority
        return new_cell

    # Initialize with a symmetric pattern
    # Start with a rod design
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1
    
    cell = enforce_symmetry(cell)
    
    # Evaluate initial score
    best_score = compute_bands(cell)
    best_cell = cell.copy()
    
    # Optimization loop
    max_iter = 100
    for iteration in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        # Hill climbing: try flipping random orbits
        improved = False
        num_tries = 20
        
        for _ in range(num_tries):
            if time.time() > deadline - 20:
                break
            
            # Pick a random pixel
            i = np.random.randint(size)
            j = np.random.randint(size)
            
            # Get its orbit
            orbit = set()
            ii, jj = i, j
            orbit.add((ii, jj))
            orbit.add((size - 1 - ii, jj))
            orbit.add((ii, size - 1 - jj))
            orbit.add((size - 1 - ii, size - 1 - jj))
            orbit.add((jj, ii))
            orbit.add((size - 1 - jj, ii))
            orbit.add((jj, size - 1 - ii))
            orbit.add((size - 1 - jj, size - 1 - ii))
            
            # Flip the orbit
            for a, b in orbit:
                cell[a, b] = 1 - cell[a, b]
            
            # Evaluate
            new_score = compute_bands(cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = cell.copy()
                improved = True
            else:
                # Revert
                for a, b in orbit:
                    cell[a, b] = 1 - cell[a, b]
                    
        if not improved:
            # Stagnation: try a larger jump or reset to best and try different region
            # For now, just continue.
            pass

    # Convert to list of lists
    result = []
    for i in range(size):
        result.append([int(best_cell[i, j]) for j in range(size)])
    
    return result
