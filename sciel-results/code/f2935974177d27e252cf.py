import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using vectorized matrix construction.
    2. Structural Change: Use Single Central Rod and Single Central Hole geometries 
       for initialization, rather than just 4 rods.
    3. Screen candidates using a fast solver.
    4. Refine the best candidates using topological optimization guided by field intensities.
    5. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 10s margin for overhead
    start_time = time.time()
    deadline = start_time + 220.0

    # Precompute G-vector indices for different n_pw values
    def get_g_indices(n_pw):
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        return np.array(g_indices, dtype=int)
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=10, return_eigvecs=False):
        """
        Compute TM band structure using plane-wave expansion.
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        
        Returns:
        --------
        If return_eigvecs is False:
            score : float, the gap-midgap ratio
        If return_eigvecs is True:
            score, best_k, best_vec4, best_vec5
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        N = size
        
        # Compute Fourier coefficients of 1/epsilon
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Define the set of G vectors to include in the basis.
        # Use a circular cutoff for better accuracy per plane wave count.
        g_indices = get_g_indices(n_pw)
        n_total = len(g_indices)
        
        # Build the S matrix: S_ij = inv_eps(G_i - G_j)
        # Vectorized construction
        m_diff = g_indices[:, 0][:, None] - g_indices[:, 0][None, :]
        n_diff = g_indices[:, 1][:, None] - g_indices[:, 1][None, :]
        
        # Handle negative indices modulo N
        m_idx = np.mod(m_diff, N)
        n_idx = np.mod(n_diff, N)
        
        S = fft_coeffs[m_idx, n_idx]
        
        # Precompute |G|^2 for each G vector
        G_sq = g_indices[:, 0]**2 + g_indices[:, 1]**2
        
        # Define k-points in the irreducible Brillouin zone
        # High symmetry points: Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # Path: Gamma -> X -> M -> Gamma
        # We use normalized k coordinates where the BZ is [0, 0.5] x [0, 0.5]
        
        k_points = []
        n_seg = max(2, n_kpts // 3)
        
        # Gamma to X: (t, 0) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        # X to M: (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        # M to Gamma: (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, n_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_gap = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        
        # Precompute k-dependent terms
        m_vals = g_indices[:, 0]
        n_vals = g_indices[:, 1]
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # |G_j + k|^2 = (m_j + kx)^2 + (n_j + ky)^2
            k_mag_sq = (m_vals + kx)**2 + (n_vals + ky)**2
            
            # M = S * diag(k_mag_sq)
            M = S * k_mag_sq[None, :]
            
            # Solve eigenvalue problem
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                # Sort eigenvalues
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                
                if len(eigvals) >= 5:
                    w4 = np.sqrt(np.real(eigvals[3]))
                    w5 = np.sqrt(np.real(eigvals[4]))
                    
                    max_omega4 = max(max_omega4, w4)
                    min_omega5 = min(min_omega5, w5)
                    
                    gap = w5 - w4
                    if gap > best_gap:
                        best_gap = gap
                        best_k = (kx, ky)
                        if return_eigvecs:
                            best_vec4 = eigvecs[:, 3]
                            best_vec5 = eigvecs[:, 4]
            except Exception:
                continue
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        if min_omega5 + max_omega4 <= 0:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        score = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        
        if return_eigvecs:
            return score, best_k, best_vec4, best_vec5
        return score

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: cell == cell[::-1] rows, cols, and transpose."""
        cell = np.array(cell)
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        cell = np.maximum(cell, cell[::-1, ::-1])
        return cell.astype(int)

    def make_single_rod_cell(rod_radius, exponent=2.0, is_hole=False):
        """
        Create a cell with a single rod or hole centered in the cell.
        is_hole=False: Dielectric rod in air (1 in center, 0 elsewhere)
        is_hole=True: Air hole in dielectric (0 in center, 1 elsewhere)
        """
        cell = np.zeros((size, size), dtype=int)
        if is_hole:
            cell[:] = 1  # Background is dielectric
        
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        # Center is (0,0)
        DX = X
        DY = Y
        
        if rod_radius > 0:
            val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            mask = val <= 1.0 + 1e-9
        else:
            mask = np.zeros_like(X, dtype=bool)
        
        if is_hole:
            cell[mask] = 0 # Make hole
        else:
            cell[mask] = 1 # Make rod
            
        cell = enforce_d4_symmetry(cell)
        return cell

    def make_four_quadrant_cell(rod_radius, exponent=2.0, is_hole=False):
        """
        Create a cell with four rods/holes centered in each quadrant.
        """
        cell = np.zeros((size, size), dtype=int)
        if is_hole:
            cell[:] = 1
            
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
                mask = val <= 1.0 + 1e-9
            else:
                mask = np.zeros_like(X, dtype=bool)
                
            if is_hole:
                cell[mask] = 0
            else:
                cell[mask] = 1
                
        cell = enforce_d4_symmetry(cell)
        return cell

    def compute_field_intensity(vec, kx, ky, g_indices):
        """
        Compute |E|^2 on the pixel grid from a plane-wave expansion vector.
        """
        m_vals = g_indices[:, 0].astype(float)
        n_vals = g_indices[:, 1].astype(float)
        
        coords = (np.arange(size) + 0.5) / size - 0.5
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        
        intensity = np.zeros((size, size), dtype=float)
        
        m_k = m_vals + kx
        n_k = n_vals + ky
        
        chunk_size = max(1, 1000000 // (size * size))
        
        for start in range(0, len(g_indices), chunk_size):
            end = min(start + chunk_size, len(g_indices))
            m_chunk = m_k[start:end]
            n_chunk = n_k[start:end]
            c_chunk = vec[start:end]
            
            phase = 2.0 * np.pi * (np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel()))
            phase = phase.reshape(end - start, size, size)
            
            e_contrib = np.sum(c_chunk[:, None, None] * np.exp(1j * phase), axis=0)
            intensity += np.abs(e_contrib)**2
        
        return intensity

    # Generate candidate cells
    best_score = -np.inf
    best_cell = None
    candidates = []
    
    # Structural Change: Include Single Rod/Hole and 4 Rod/Hole geometries
    # We will screen these with a fast solver
    
    # Parameters for Single Feature
    single_radii = [0.15, 0.20, 0.25, 0.30]
    # Parameters for Four Features
    four_radii = [0.08, 0.10, 0.12, 0.15]
    exponents = [2.0, 3.0, 4.0]
    
    # Generate Single Rods
    for r in single_radii:
        for exp in exponents:
            cell = make_single_rod_cell(r, exp, is_hole=False)
            candidates.append(cell)
            
    # Generate Single Holes
    for r in single_radii:
        for exp in exponents:
            cell = make_single_rod_cell(r, exp, is_hole=True)
            candidates.append(cell)
            
    # Generate 4 Rods (from original strategy)
    for r in four_radii:
        for exp in exponents:
            cell = make_four_quadrant_cell(r, exp, is_hole=False)
            candidates.append(cell)
            
    # Generate 4 Holes
    for r in four_radii:
        for exp in exponents:
            cell = make_four_quadrant_cell(r, exp, is_hole=True)
            candidates.append(cell)

    # Screen candidates
    # Use n_pw=9 and n_kpts=6 for speed during screening
    for cell in candidates:
        if time.time() > deadline - 80:
            break
        try:
            score = compute_tm_bands(cell, n_pw=9, n_kpts=6)
            if score > best_score:
                best_score = score
                best_cell = cell
        except Exception:
            continue
    
    # If no cell was found or all failed, return a default
    if best_cell is None:
        best_cell = make_single_rod_cell(0.20, 2.0, is_hole=False)
        best_score = compute_tm_bands(best_cell, n_pw=9, n_kpts=6)
    
    # Re-evaluate best candidate with higher accuracy
    best_score = compute_tm_bands(best_cell, n_pw=13, n_kpts=12)
    
    # Topological optimization phase
    current_cell = best_cell.copy()
    current_score = best_score
    g_indices_13 = get_g_indices(13)
    
    n_iter = 40
    stall_count = 0
    max_stall = 5
    
    for it in range(n_iter):
        if time.time() > deadline - 15:
            break
        
        # Get eigenvectors
        score, best_k, vec4, vec5 = compute_tm_bands(current_cell, n_pw=13, n_kpts=12, return_eigvecs=True)
        
        if best_k is None:
            break
        
        kx, ky = best_k
        
        # Compute intensities
        I4 = compute_field_intensity(vec4, kx, ky, g_indices_13)
        I5 = compute_field_intensity(vec5, kx, ky, g_indices_13)
        
        # Normalize intensities
        I4 = I4 / (np.sum(I4) + 1e-12)
        I5 = I5 / (np.sum(I5) + 1e-12)
        
        # Metric for adding dielectric: I4 - I5
        # Metric for removing dielectric: I5 - I4
        
        mask_add = (current_cell == 0)
        mask_rem = (current_cell == 1)
        
        gain_add = np.where(mask_add, I4 - I5, -np.inf)
        gain_rem = np.where(mask_rem, I5 - I4, -np.inf)
        
        # Try top 3 candidates for each type
        improved = False
        
        # Get top candidates for adding
        add_candidates = np.argpartition(gain_add.ravel(), -3)[-3:]
        add_candidates = np.array([np.unravel_index(c, gain_add.shape) for c in add_candidates])
        
        # Get top candidates for removing
        rem_candidates = np.argpartition(gain_rem.ravel(), -3)[-3:]
        rem_candidates = np.array([np.unravel_index(c, gain_rem.shape) for c in rem_candidates])
        
        # Try each candidate
        for idx in add_candidates:
            if time.time() > deadline - 15:
                break
            new_cell = current_cell.copy()
            new_cell[idx] = 1
            new_cell = enforce_d4_symmetry(new_cell)
            
            new_score = compute_tm_bands(new_cell, n_pw=13, n_kpts=12)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                best_cell = current_cell
                best_score = current_score
                improved = True
                break
        
        if not improved:
            for idx in rem_candidates:
                if time.time() > deadline - 15:
                    break
                new_cell = current_cell.copy()
                new_cell[idx] = 0
                new_cell = enforce_d4_symmetry(new_cell)
                
                new_score = compute_tm_bands(new_cell, n_pw=13, n_kpts=12)
                
                if new_score > current_score:
                    current_cell = new_cell
                    current_score = new_score
                    best_cell = current_cell
                    best_score = current_score
                    improved = True
                    break
        
        if improved:
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= max_stall:
                break
    
    # Final polish with more k-points
    if time.time() < deadline - 20:
        final_score = compute_tm_bands(best_cell, n_pw=15, n_kpts=15)
        if final_score > best_score:
            best_score = final_score
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
