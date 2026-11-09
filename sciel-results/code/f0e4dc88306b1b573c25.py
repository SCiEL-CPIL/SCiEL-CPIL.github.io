    def evaluate_cell(cell):
        """Compute the gap-midgap ratio for the given cell."""
        cell = np.array(cell, dtype=int)
        # Ensure symmetry
        cell = symmetrize(cell)
        
        # Compute Fourier coefficients of 1/eps
        eps_real = eps_low + (eps_high - eps_low) * cell
        inv_eps_real = 1.0 / eps_real
        
        # Compute inv_eps_coeff on the grid of G_vectors
        inv_eps_coeff = np.zeros(len(G_vectors), dtype=complex)
        
        # Vectorized direct sum for Fourier coefficients
        Gx_arr = G_vectors[:, 0]
        Gy_arr = G_vectors[:, 1]
        
        r_x = (np.arange(size) + 0.5) / size
        r_y = (np.arange(size) + 0.5) / size
        
        # Phase shape (N_G, N_pix, N_pix)
        phase = 2 * np.pi * (np.outer(Gx_arr, r_x)[:, :, np.newaxis] + 
                             np.outer(Gy_arr, r_y)[:, np.newaxis, :])
                             
        term = inv_eps_real[np.newaxis, :, :] * np.exp(-1j * phase)
        inv_eps_coeff = np.sum(term, axis=(1, 2)) / (size * size)
        
        # Precompute coefficient grid for differences
        coeff_grid = np.zeros((4*N_PW + 1, 4*N_PW + 1), dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            row = int(gx) + 2 * N_PW
            col = int(gy) + 2 * N_PW
            coeff_grid[row, col] = inv_eps_coeff[idx]
            
        # Precompute kG_sq for all k_points
        kG_sq_all = np.zeros((len(k_points), len(G_vectors)))
        for k_idx, (kx, ky) in enumerate(k_points):
            kG_sq_all[k_idx, :] = (kx + G_vectors[:, 0])**2 + (ky + G_vectors[:, 1])**2
            
        # Build the difference matrix D
        Gx_diff = G_vectors[:, 0][:, np.newaxis] - G_vectors[:, 0][np.newaxis, :]
        Gy_diff = G_vectors[:, 1][:, np.newaxis] - G_vectors[:, 1][np.newaxis, :]
        
        D = coeff_grid[Gx_diff + 2 * N_PW, Gy_diff + 2 * N_PW]
        
        # Solve for each k-point
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for k_idx in range(len(k_points)):
            kG_sq = kG_sq_all[k_idx]
            M = -D
            np.fill_diagonal(M, kG_sq)
            
            # Make Hermitian
            M = (M + M.conj().T) / 2
            
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(max(eigvals[3], 0))
                    omega5 = np.sqrt(max(eigvals[4], 0))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                continue
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return -1.0
        
        return 2 * gap / midgap
