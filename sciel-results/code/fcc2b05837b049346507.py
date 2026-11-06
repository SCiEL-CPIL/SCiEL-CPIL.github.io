import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (2,0) and (2,2) modes. 
    Optimizes signed amplitudes and offset to maximize the TM band gap 
    between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Basis functions (C4v symmetric)
    # (2,0) + (0,2)
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    # (2,2) + (2,-2)
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Precompute FFT of the identity? No, we need FFT of the cell.
    # But we can precompute the index mapping for the E-matrix construction.
    
    M_PWE = 4  # Indices -4 to 4, so 9x9 = 81 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute index differences for E-matrix lookup
    dx_matrix = Gx_flat[:, None] - Gx_flat[None, :]
    dy_matrix = Gy_flat[:, None] - Gy_flat[None, :]
    # Map to 0..size-1
    ix_indices = (dx_matrix % size + size) % size
    iy_indices = (dy_matrix % size + size) % size
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # The irreducible BZ for square lattice is 0 <= kx <= ky <= 0.5
    # We sample points along the boundary and interior.
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.1, 0.0),
        (0.1, 0.1),
        (0.4, 0.2),
        (0.45, 0.45)
    ]

    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        # Vectorized lookup of epsilon Fourier coefficients
        # E[i, j] = eps(Gi - Gj)
        E = fft_vals[ix_indices, iy_indices]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            # D is diagonal with (k+G)^2
            D_diag = kGx**2 + kGy**2
            
            # A = D * E  (Hadamard product of diagonal matrix and E)
            # A[i, j] = D_diag[i] * E[i, j]
            A = D_diag[:, None] * E
            
            # Ensure Hermitian (should be, but numerical noise might creep in)
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

    def generate_cell(A20, A22, offset):
        """Generate a binary cell from Fourier parameters."""
        field = offset + A20 * basis_20 + A22 * basis_22
        cell_raw = (field > 0.5).astype(int)
        
        # The field is already C4v symmetric, so cell_raw should be too.
        # However, thresholding can break symmetry if the field is exactly 0.5.
        # To be safe, we can enforce symmetry by averaging, but since the field
        # is constructed from symmetric terms, it should be fine.
        # Let's verify symmetry by checking against its transpose and reverses.
        # If not symmetric, we enforce it.
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
    best_params = (0.0, 0.0, 0.5)
    
    # Stage 1: Coarse grid search
    # A20 and A22 can be negative.
    A20_coarse = np.linspace(-1.5, 1.5, 11)
    A22_coarse = np.linspace(-1.5, 1.5, 11)
    offset_coarse = np.linspace(0.3, 0.7, 9)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                cell = generate_cell(A20, A22, offset)
                
                fill = np.sum(cell) / (size * size)
                if fill < 0.10 or fill > 0.90:
                    continue
                
                score = calculate_surrogate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (A20, A22, offset)
    
    # Stage 2: Fine-tune around best candidate
    if best_cell is not None:
        A20_best, A22_best, offset_best = best_params
        
        # Fine grid
        step_A = 0.2
        step_off = 0.05
        
        A20_fine = np.linspace(A20_best - step_A, A20_best + step_A, 5)
        A22_fine = np.linspace(A22_best - step_A, A22_best + step_A, 5)
        offset_fine = np.linspace(offset_best - step_off, offset_best + step_off, 5)
        
        for A20 in A20_fine:
            for A22 in A22_fine:
                for offset in offset_fine:
                    cell = generate_cell(A20, A22, offset)
                    
                    fill = np.sum(cell) / (size * size)
                    if fill < 0.10 or fill > 0.90:
                        continue
                    
                    score = calculate_surrogate_gap(cell)
                    
                    if score > best_score:
                        best_score = score
                        best_cell = cell.tolist()
                        best_params = (A20, A22, offset)
                        
        # Stage 2.1: Even finer local search
        A20_best, A22_best, offset_best = best_params
        step_A = 0.05
        step_off = 0.01
        
        A20_fine2 = np.linspace(A20_best - step_A, A20_best + step_A, 5)
        A22_fine2 = np.linspace(A22_best - step_A, A22_best + step_A, 5)
        offset_fine2 = np.linspace(offset_best - step_off, offset_best + step_off, 5)
        
        for A20 in A20_fine2:
            for A22 in A22_fine2:
                for offset in offset_fine2:
                    cell = generate_cell(A20, A22, offset)
                    
                    fill = np.sum(cell) / (size * size)
                    if fill < 0.10 or fill > 0.90:
                        continue
                    
                    score = calculate_surrogate_gap(cell)
                    
                    if score > best_score:
                        best_score = score
                        best_cell = cell.tolist()
    
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
