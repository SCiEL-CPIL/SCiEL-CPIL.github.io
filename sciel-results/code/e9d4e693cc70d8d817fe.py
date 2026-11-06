import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics that directly couple to the 4th and 5th TM bands.
    Optimizes amplitudes and phases to maximize the spectral separation.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute basis functions for C4v-symmetric combinations
    # (2,0) + (0,2): cos(4πX) + cos(4πY)
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    
    # (2,2) + (2,-2): cos(4π(X+Y)) + cos(4π(X-Y))
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # (4,0) + (0,4): cos(8πX) + cos(8πY)
    basis_40 = np.cos(8 * np.pi * X) + np.cos(8 * np.pi * Y)
    
    # (4,4) + (4,-4): cos(8π(X+Y)) + cos(8π(X-Y))
    basis_44 = np.cos(8 * np.pi * (X + Y)) + np.cos(8 * np.pi * (X - Y))
    
    # (1,0) + (0,1): cos(2πX) + cos(2πY)
    basis_10 = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)
    
    # (1,1) + (1,-1): cos(2π(X+Y)) + cos(2π(X-Y))
    basis_11 = np.cos(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X - Y))
    
    # (3,0) + (0,3): cos(6πX) + cos(6πY)
    basis_30 = np.cos(6 * np.pi * X) + np.cos(6 * np.pi * Y)
    
    # (3,3) + (3,-3): cos(6π(X+Y)) + cos(6π(X-Y))
    basis_33 = np.cos(6 * np.pi * (X + Y)) + np.cos(6 * np.pi * (X - Y))
    
    # Fast PWE Surrogate Parameters
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # Focus on points where bands 4 and 5 are close
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.1, 0.1),
        (0.45, 0.05),
        (0.5, 0.25),
        (0.2, 0.4),
        (0.15, 0.35),
        (0.3, 0.1),
        (0.4, 0.4),
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        def get_eps(Gx_val, Gy_val):
            ix = int(Gx_val) % size
            iy = int(Gy_val) % size
            return fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            
            # Vectorized computation of epsilon matrix
            dx_matrix = Gx_flat[:, None] - Gx_flat[None, :]
            dy_matrix = Gy_flat[:, None] - Gy_flat[None, :]
            
            E = np.zeros((N_PW, N_PW), dtype=complex)
            for i in range(N_PW):
                for j in range(N_PW):
                    E[i, j] = get_eps(Gx_flat[i] - Gx_flat[j], Gy_flat[i] - Gy_flat[j])
            
            A = D * E
            A = (A + A.conj().T) / 2.0
            A = np.real(A)
            
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                return -np.inf
                
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                
                if w4 > 0 and w5 > 0:
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell(params):
        """Generate a binary cell from Fourier parameters.
        params: (A10, A11, A20, A22, A30, A33, A40, A44, offset, p10, p11, p20, p22, p30, p33, p40, p44)
        """
        A10, A11, A20, A22, A30, A33, A40, A44, offset = params[0:9]
        p10, p11, p20, p22, p30, p33, p40, p44 = params[9:17]
        
        field = np.full_like(X, offset, dtype=float)
        if abs(A10) > 1e-10:
            field += A10 * (np.cos(2 * np.pi * X + p10) + np.cos(2 * np.pi * Y + p10))
        if abs(A11) > 1e-10:
            field += A11 * (np.cos(2 * np.pi * (X + Y) + p11) + np.cos(2 * np.pi * (X - Y) + p11))
        if abs(A20) > 1e-10:
            field += A20 * (np.cos(4 * np.pi * X + p20) + np.cos(4 * np.pi * Y + p20))
        if abs(A22) > 1e-10:
            field += A22 * (np.cos(4 * np.pi * (X + Y) + p22) + np.cos(4 * np.pi * (X - Y) + p22))
        if abs(A30) > 1e-10:
            field += A30 * (np.cos(6 * np.pi * X + p30) + np.cos(6 * np.pi * Y + p30))
        if abs(A33) > 1e-10:
            field += A33 * (np.cos(6 * np.pi * (X + Y) + p33) + np.cos(6 * np.pi * (X - Y) + p33))
        if abs(A40) > 1e-10:
            field += A40 * (np.cos(8 * np.pi * X + p40) + np.cos(8 * np.pi * Y + p40))
        if abs(A44) > 1e-10:
            field += A44 * (np.cos(8 * np.pi * (X + Y) + p44) + np.cos(8 * np.pi * (X - Y) + p44))
        
        cell_raw = (field > 0.5).astype(int)
        
        # Enforce C4v symmetry by averaging over all 8 symmetry operations
        sym_cells = [
            cell_raw,
            cell_raw[:, ::-1],
            cell_raw[::-1, :],
            cell_raw.T,
            cell_raw[::-1, ::-1],
            cell_raw.T[::-1, :],
            cell_raw[:, ::-1].T,
            cell_raw[::-1, :].T
        ]
        
        avg_cell = np.mean(sym_cells, axis=0)
        cell_sym = (avg_cell >= 0.5).astype(int)
        
        return cell_sym

    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Multi-stage optimization:
    # Stage 1: Coarse grid search over key amplitudes and offset
    # Focus on (2,0), (2,2) as primary, with smaller contributions from others
    
    A20_coarse = np.linspace(0.1, 1.2, 7)
    A22_coarse = np.linspace(0.0, 1.0, 6)
    A40_coarse = np.linspace(0.0, 0.5, 4)
    A44_coarse = np.linspace(0.0, 0.4, 4)
    offset_coarse = np.linspace(0.30, 0.70, 7)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for A40 in A40_coarse:
                for A44 in A44_coarse:
                    for offset in offset_coarse:
                        # Set other modes to 0 for coarse search
                        params = (0.0, 0.0, A20, A22, 0.0, 0.0, A40, A44, offset,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        cell = generate_cell(params)
                        
                        fill = np.sum(cell) / (size * size)
                        if fill < 0.10 or fill > 0.90:
                            continue
                        
                        score = calculate_surrogate_gap(cell)
                        
                        if score > best_score:
                            best_score = score
                            best_cell = cell.tolist()
                            best_params = params
    
    # Stage 2: Fine-tune around best candidate with phase optimization
    if best_cell is not None:
        A20_best = best_params[2]
        A22_best = best_params[3]
        A40_best = best_params[6]
        A44_best = best_params[7]
        offset_best = best_params[8]
        
        # Fine grid around best values
        A20_fine = np.linspace(max(0.05, A20_best - 0.15), min(1.5, A20_best + 0.15), 4)
        A22_fine = np.linspace(max(0.0, A22_best - 0.15), min(1.5, A22_best + 0.15), 4)
        A40_fine = np.linspace(max(0.0, A40_best - 0.1), min(0.8, A40_best + 0.1), 3)
        A44_fine = np.linspace(max(0.0, A44_best - 0.1), min(0.6, A44_best + 0.1), 3)
        offset_fine = np.linspace(max(0.25, offset_best - 0.06), min(0.75, offset_best + 0.06), 4)
        phase20_fine = np.linspace(-0.5, 0.5, 3)
        phase22_fine = np.linspace(-0.5, 0.5, 3)
        phase40_fine = np.linspace(-0.5, 0.5, 3)
        phase44_fine = np.linspace(-0.5, 0.5, 3)
        
        for A20 in A20_fine:
            for A22 in A22_fine:
                for A40 in A40_fine:
                    for A44 in A44_fine:
                        for offset in offset_fine:
                            for p20 in phase20_fine:
                                for p22 in phase22_fine:
                                    for p40 in phase40_fine:
                                        for p44 in phase44_fine:
                                            params = (0.0, 0.0, A20, A22, 0.0, 0.0, A40, A44, offset,
                                                      0.0, 0.0, p20, p22, 0.0, 0.0, p40, p44)
                                            cell = generate_cell(params)
                                            
                                            fill = np.sum(cell) / (size * size)
                                            if fill < 0.10 or fill > 0.90:
                                                continue
                                            
                                            score = calculate_surrogate_gap(cell)
                                            
                                            if score > best_score:
                                                best_score = score
                                                best_cell = cell.tolist()
                                                best_params = params
    
    # Stage 3: Add lower-order harmonics for fine structure
    if best_cell is not None:
        A20_best = best_params[2]
        A22_best = best_params[3]
        A40_best = best_params[6]
        A44_best = best_params[7]
        offset_best = best_params[8]
        p20_best = best_params[11]
        p22_best = best_params[12]
        p40_best = best_params[15]
        p44_best = best_params[16]
        
        A10_fine = np.linspace(-0.2, 0.2, 3)
        A11_fine = np.linspace(-0.2, 0.2, 3)
        A30_fine = np.linspace(-0.15, 0.15, 3)
        A33_fine = np.linspace(-0.15, 0.15, 3)
        
        for A10 in A10_fine:
            for A11 in A11_fine:
                for A30 in A30_fine:
                    for A33 in A33_fine:
                        params = (A10, A11, A20_best, A22_best, A30, A33, A40_best, A44_best, offset_best,
                                  0.0, 0.0, p20_best, p22_best, 0.0, 0.0, p40_best, p44_best)
                        cell = generate_cell(params)
                        
                        fill = np.sum(cell) / (size * size)
                        if fill < 0.10 or fill > 0.90:
                            continue
                        
                        score = calculate_surrogate_gap(cell)
                        
                        if score > best_score:
                            best_score = score
                            best_cell = cell.tolist()
                            best_params = params
    
    # Stage 4: Try known good configurations
    if best_cell is not None:
        test_configs = [
            # (A10, A11, A20, A22, A30, A33, A40, A44, offset, p10, p11, p20, p22, p30, p33, p40, p44)
            (0.0, 0.0, 0.8, 0.5, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.6, 0.7, 0.0, 0.0, 0.2, 0.3, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.9, 0.3, 0.0, 0.0, 0.4, 0.1, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.7, 0.6, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.7, 0.6, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.8, 0.0, 0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.8, 0.0, 0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.5, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.1, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.0, 0.7, 0.5, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.1, 0.7, 0.5, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-0.1, 0.0, 0.7, 0.5, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.6, 0.6, 0.1, 0.0, 0.3, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.6, 0.6, 0.0, 0.1, 0.3, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.75, 0.55, 0.0, 0.0, 0.35, 0.25, 0.52, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1),
        ]
        
        for params in test_configs:
            cell = generate_cell(params)
            
            fill = np.sum(cell) / (size * size)
            if fill < 0.10 or fill > 0.90:
                continue
            
            score = calculate_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
                best_params = params
    
    # Stage 5: Local refinement with small perturbations around best
    if best_cell is not None and best_params is not None:
        # Try small variations around the best parameters
        for dA20 in [-0.05, 0.0, 0.05]:
            for dA22 in [-0.05, 0.0, 0.05]:
                for doffset in [-0.02, 0.0, 0.02]:
                    for dp20 in [-0.1, 0.0, 0.1]:
                        for dp22 in [-0.1, 0.0, 0.1]:
                            new_params = list(best_params)
                            new_params[2] = max(0.0, min(1.5, new_params[2] + dA20))
                            new_params[3] = max(0.0, min(1.5, new_params[3] + dA22))
                            new_params[8] = max(0.25, min(0.75, new_params[8] + doffset))
                            new_params[11] = new_params[11] + dp20
                            new_params[12] = new_params[12] + dp22
                            new_params = tuple(new_params)
                            
                            cell = generate_cell(new_params)
                            
                            fill = np.sum(cell) / (size * size)
                            if fill < 0.10 or fill > 0.90:
                                continue
                            
                            score = calculate_surrogate_gap(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.tolist()
                                best_params = new_params
    
    if best_cell is None:
        # Fallback: Standard rod
        best_cell = []
        for i in range(size):
            xi = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                yj = (j + 0.5) / size - 0.5
                if xi*xi + yj*yj < 0.2**2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell
