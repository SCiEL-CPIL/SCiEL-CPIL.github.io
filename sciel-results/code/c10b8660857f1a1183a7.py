"""Topological optimization for photonic crystal band gap maximization."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def design(problem):
    """Design a 32x32 photonic crystal cell maximizing TM band 4-5 gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 220  # 220s budget, 10s margin
    
    # Precompute plane wave vectors (289 = 17x17, indices -8..8)
    n_pw = 17
    pw_indices = np.arange(-n_pw // 2, n_pw // 2 + 1)  # -8..8
    G = np.array([(gx, gy) for gy in pw_indices for gx in pw_indices], dtype=float)
    n_G = len(G)
    
    # Precompute |G|^2
    G2 = np.sum(G**2, axis=1)
    
    # Compute Fourier coefficients of epsilon for the pixel map
    def compute_epsilon_fourier(cell):
        """Compute Fourier coefficients of epsilon(x,y) for the pixel map."""
        # cell is size x size binary array (1=dielectric, 0=air)
        # epsilon(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
        # We need epsilon_G = (1/A) ∫∫ epsilon(x,y) exp(-iG·r) dA
        # For a pixel map, this is a discrete sum
        eps_coeffs = np.zeros(n_G, dtype=complex)
        dx = 1.0 / size
        for idx, (gx, gy) in enumerate(G):
            # Sum over all pixels
            # pixel (i,j) has center at ((i+0.5)/size - 0.5, (j+0.5)/size - 0.5)
            # = ((i+0.5-size/2)/size, (j+0.5-size/2)/size)
            # = ((2i+1-size)/(2*size), (2j+1-size)/(2*size))
            x_centers = (np.arange(size) + 0.5 - size / 2) / size
            y_centers = (np.arange(size) + 0.5 - size / 2) / size
            X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
            phase = np.exp(-1j * (gx * 2 * np.pi * X + gy * 2 * np.pi * Y))
            eps_map = eps_low + (eps_high - eps_low) * cell
            eps_coeffs[idx] = np.sum(eps_map * phase) * dx * dx
        return eps_coeffs
    
    # Build TM Hamiltonian for given wavevector and epsilon Fourier coeffs
    def build_tm_hamiltonian(kx, ky, eps_coeffs):
        """Build the TM band Hamiltonian matrix."""
        # For TM polarization, the eigenvalue problem is:
        # sum_G' (|k+G|^2 / epsilon_{G-G'}) |k+G| = omega^2 |k+G>
        # where epsilon_{G-G'} are Fourier coefficients of 1/epsilon
        # But we work with epsilon directly using the formulation:
        # sum_G' (k+G)·(k+G') / epsilon_{G-G'} ... 
        # Actually, the standard PWE formulation for TM:
        # The operator is (k+G)^2 / epsilon(r), which in Fourier space:
        # H_{G,G'} = (k+G)·(k+G') * (1/epsilon)_{G-G'}
        # where (1/epsilon)_{G-G'} = (1/A) ∫∫ (1/epsilon(r)) exp(-i(G-G')·r) dA
        
        # We need 1/epsilon Fourier coefficients, not epsilon
        # Let's compute them from the pixel map
        # Actually, let me reconsider. The standard formulation uses:
        # For TM: omega^2 * sum_G' epsilon_{G-G'} c_{G'} = (k+G)^2 c_G
        # This is a generalized eigenvalue problem: A c = omega^2 B c
        # where A_{G,G'} = delta_{G,G'} (k+G)^2
        # and B_{G,G'} = epsilon_{G-G'}
        
        # So we need epsilon Fourier coefficients (which we have)
        # A is diagonal with (k+G)^2
        # B is the matrix with entries epsilon_{G-G'}
        
        k_vec = np.array([kx, ky])
        k_plus_G = k_vec + G  # shape (n_G, 2)
        k2 = np.sum(k_plus_G**2, axis=1)  # (n_G,)
        
        # A is diagonal
        # B_{G,G'} = epsilon_{G-G'}
        # We need to build B as a matrix where B[i,j] = eps_coeffs[G_i - G_j]
        
        # Precompute mapping from G difference to index
        G_set = set(map(tuple, G))
        G_to_idx = {tuple(g): i for i, g in enumerate(G)}
        
        B = np.zeros((n_G, n_G), dtype=complex)
        for i in range(n_G):
            for j in range(n_G):
                diff = tuple(G[i] - G[j])
                if diff in G_to_idx:
                    B[i, j] = eps_coeffs[G_to_idx[diff]]
        
        # Generalized eigenvalue: A c = omega^2 B c
        # Convert to standard: B^{-1} A c = omega^2 c
        # Or use scipy's eigsh with sigma for generalized problem
        
        # Actually, for efficiency, let's use the standard formulation:
        # D c = omega^2 c where D = B^{-1} A
        # But B may not be well-conditioned. Let's use scipy.linalg.eigh for Hermitian case.
        
        # Since B is Hermitian (epsilon is real and symmetric), we can use:
        # B^{-1/2} A B^{-1/2} c' = omega^2 c'
        
        # For simplicity and robustness, let's just compute all eigenvalues
        # using numpy for the smaller matrix, or use scipy.sparse.linalg
        
        # Let's build the standard eigenvalue problem
        # A is diagonal with k2
        # We solve: A c = omega^2 B c
        # => B^{-1} A c = omega^2 c
        
        # Use scipy.linalg.eigh for the generalized Hermitian problem
        from scipy.linalg import eigh as eigh_dense
        
        # A is real diagonal, B is complex Hermitian
        # Convert to real if possible
        A_mat = np.diag(k2).astype(complex)
        
        # Use eigh for generalized problem
        try:
            eigvals, eigvecs = eigh_dense(A_mat, B, subset_by_index=[0, lower_band + 4])
        except Exception:
            # Fallback: compute all eigenvalues
            eigvals, eigvecs = eigh_dense(A_mat, B)
        
        return eigvals
    
    # Compute band structure for a given cell
    def compute_bands(cell, n_kpoints=30):
        """Compute bands 4 and 5 over the irreducible BZ."""
        eps_coeffs = compute_epsilon_fourier(cell)
        
        # Irreducible BZ for square lattice: 0 <= ky <= kx, kx, ky in [0, 0.5]
        # Key points: Gamma (0,0), X (0.5,0), M (0.5,0.5)
        # Path: Gamma->X->M->Gamma, plus some interior points
        
        # Define 30 k-points covering the irreducible BZ
        k_points = []
        # Gamma to X: (t/2, 0) for t in [0,1]
        for t in np.linspace(0, 1, 8):
            k_points.append((t / 2, 0))
        # X to M: (0.5, t/2) for t in [0,1]
        for t in np.linspace(0, 1, 8):
            k_points.append((0.5, t / 2))
        # M to Gamma: (0.5*(1-t), 0.5*(1-t)) for t in [0,1]
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
                eigvals = compute_bands_single_k(kx, ky, eps_coeffs)
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
    
    def compute_bands_single_k(kx, ky, eps_coeffs):
        """Compute eigenvalues for a single k-point."""
        k_vec = np.array([kx, ky])
        k_plus_G = k_vec + G
        k2 = np.sum(k_plus_G**2, axis=1)
        
        # Build B matrix: B[i,j] = eps_coeffs[G_i - G_j]
        G_to_idx = {tuple(g): i for i, g in enumerate(G)}
        
        B = np.zeros((n_G, n_G), dtype=complex)
        for i in range(n_G):
            for j in range(n_G):
                diff = tuple(G[i] - G[j])
                if diff in G_to_idx:
                    B[i, j] = eps_coeffs[G_to_idx[diff]]
        
        A_mat = np.diag(k2).astype(complex)
        
        from scipy.linalg import eigh as eigh_dense
        try:
            eigvals, _ = eigh_dense(A_mat, B, subset_by_index=[0, 6])
        except Exception:
            eigvals, _ = eigh_dense(A_mat, B)
        
        return eigvals
    
    # Optimization: use field intensities to guide pixel updates
    def get_field_intensities(cell, kx, ky, band_idx):
        """Get field intensity pattern for a specific band."""
        eps_coeffs = compute_epsilon_fourier(cell)
        eigvals, eigvecs = compute_bands_single_k_with_vecs(kx, ky, eps_coeffs)
        if band_idx >= len(eigvals):
            return None
        c = eigvecs[:, band_idx]
        # Field intensity in real space: |sum_G c_G exp(i G·r)|^2
        # Sample on the pixel grid
        intensities = np.zeros((size, size))
        x_centers = (np.arange(size) + 0.5 - size / 2) / size
        for i in range(size):
            for j in range(size):
                x = x_centers[i]
                y = x_centers[j]
                field = 0
                for idx, (gx, gy) in enumerate(G):
                    field += c[idx] * np.exp(1j * (gx * 2 * np.pi * x + gy * 2 * np.pi * y))
                intensities[i, j] = abs(field)**2
        return intensities
    
    def compute_bands_single_k_with_vecs(kx, ky, eps_coeffs):
        """Compute eigenvalues and eigenvectors for a single k-point."""
        k_vec = np.array([kx, ky])
        k_plus_G = k_vec + G
        k2 = np.sum(k_plus_G**2, axis=1)
        
        G_to_idx = {tuple(g): i for i, g in enumerate(G)}
        
        B = np.zeros((n_G, n_G), dtype=complex)
        for i in range(n_G):
            for j in range(n_G):
                diff = tuple(G[i] - G[j])
                if diff in G_to_idx:
                    B[i, j] = eps_coeffs[G_to_idx[diff]]
        
        A_mat = np.diag(k2).astype(complex)
        
        from scipy.linalg import eigh as eigh_dense
        try:
            eigvals, eigvecs = eigh_dense(A_mat, B, subset_by_index=[0, 6])
        except Exception:
            eigvals, eigvecs = eigh_dense(A_mat, B)
        
        return eigvals, eigvecs
    
    # Initialize with a symmetric pattern
    # Start with the rod design as initial guess
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1
    
    # Ensure symmetry: cell == cell[::-1] row-wise and column-wise, and cell == transpose
    # This means 4-fold rotational symmetry and mirror symmetry
    # The cell must be invariant under:
    # 1. Horizontal flip: cell[i,j] == cell[size-1-i, j]
    # 2. Vertical flip: cell[i,j] == cell[i, size-1-j]
    # 3. Transpose: cell[i,j] == cell[j,i]
    
    def enforce_symmetry(cell):
        """Enforce 4-fold symmetry about the center."""
        size = cell.shape[0]
        new_cell = cell.copy()
        for i in range(size):
            for j in range(size):
                # The orbit of (i,j) under the symmetry group
                # Generators: (i,j) -> (size-1-i, j), (i,j) -> (i, size-1-j), (i,j) -> (j,i)
                # The full orbit has up to 8 elements
                orbit = set()
                ii, jj = i, j
                for _ in range(8):
                    orbit.add((ii, jj))
                    # Apply horizontal flip
                    ii2 = size - 1 - ii
                    orbit.add((ii2, jj))
                    # Apply vertical flip
                    orbit.add((ii, size - 1 - jj))
                    # Apply transpose
                    orbit.add((jj, ii))
                    # Update for next iteration
                    ii, jj = size - 1 - ii, size - 1 - jj
                # Average or take majority vote
                values = [cell[a, b] for a, b in orbit]
                majority = 1 if sum(values) > len(values) // 2 else 0
                for a, b in orbit:
                    new_cell[a, b] = majority
        return new_cell
    
    cell = enforce_symmetry(cell)
    
    # Evaluate initial score
    best_score = compute_bands(cell)
    best_cell = cell.copy()
    
    # Optimization loop
    max_iter = 50
    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        # Get field intensities at a few k-points
        k_points_opt = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.25), (0.25, 0.25)]
        
        # Compute intensity difference: band 4 intensity - band 5 intensity
        # Where band 4 has high intensity, add dielectric (if gap is positive, we want to push band 4 down)
        # Actually, to increase the gap:
        # - Add dielectric where band 4 has high intensity (lowers band 4)
        # - Remove dielectric where band 5 has high intensity (raises band 5)
        
        intensity_diff = np.zeros((size, size))
        for kx, ky in k_points_opt:
            try:
                eps_coeffs = compute_epsilon_fourier(cell)
                eigvals, eigvecs = compute_bands_single_k_with_vecs(kx, ky, eps_coeffs)
                if len(eigvals) > 4:
                    c4 = eigvecs[:, 3]
                    c5 = eigvecs[:, 4]
                    
                    x_centers = (np.arange(size) + 0.5 - size / 2) / size
                    for i in range(size):
                        for j in range(size):
                            x = x_centers[i]
                            y = x_centers[j]
                            field4 = 0
                            field5 = 0
                            for idx, (gx, gy) in enumerate(G):
                                phase = np.exp(1j * (gx * 2 * np.pi * x + gy * 2 * np.pi * y))
                                field4 += c4[idx] * phase
                                field5 += c5[idx] * phase
                            intensity_diff[i, j] += abs(field4)**2 - abs(field5)**2
            except Exception:
                continue
        
        # Propose pixel flips
        # Where intensity_diff > 0: band 4 has more intensity, so add dielectric
        # Where intensity_diff < 0: band 5 has more intensity, so remove dielectric
        
        # Get unique orbits to avoid redundant work
        orbits = []
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) not in visited:
                    orbit = set()
                    ii, jj = i, j
                    for _ in range(8):
                        orbit.add((ii, jj))
                        orbit.add((size - 1 - ii, jj))
                        orbit.add((ii, size - 1 - jj))
                        orbit.add((jj, ii))
                        ii, jj = size - 1 - ii, size - 1 - jj
                    orbits.append(orbit)
                    visited.update(orbit)
        
        # For each orbit, decide whether to flip
        improved = False
        for orbit in orbits:
            if time.time() > deadline - 30:
                break
            
            # Average intensity_diff over the orbit
            avg_diff = np.mean([intensity_diff[i, j] for i, j in orbit])
            
            # Current value at this orbit
            current_val = cell[list(orbit)[0][0], list(orbit)[0][1]]
            
            # Decide flip
            if avg_diff > 0 and current_val == 0:
                # Add dielectric
                for i, j in orbit:
                    cell[i, j] = 1
                improved = True
            elif avg_diff < 0 and current_val == 1:
                # Remove dielectric
                for i, j in orbit:
                    cell[i, j] = 0
                improved = True
        
        if not improved:
            # Try random flips
            for _ in range(5):
                if time.time() > deadline - 30:
                    break
                orbit = orbits[np.random.randint(len(orbits))]
                for i, j in orbit:
                    cell[i, j] = 1 - cell[i, j]
                improved = True
        
        # Evaluate new score
        new_score = compute_bands(cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = cell.copy()
        else:
            # Revert
            cell = best_cell.copy()
    
    # Convert to list of lists
    result = []
    for i in range(size):
        result.append([int(cell[i, j]) for j in range(size)])
    
    return result
