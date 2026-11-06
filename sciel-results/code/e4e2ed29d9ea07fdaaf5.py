import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (2,0), (0,2), and (2,2) modes that directly 
    couple to the 4th and 5th TM bands. Optimizes amplitudes and phases 
    to maximize the spectral separation between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # C4v-symmetric basis functions
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    basis_30 = np.cos(6 * np.pi * X) + np.cos(6 * np.pi * Y)
    basis_33 = np.cos(6 * np.pi * (X + Y)) + np.cos(6 * np.pi * (X - Y))
    basis_10 = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)
    
    # Fast PWE Surrogate Parameters
    M_PWE = 5
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute pairwise differences for epsilon matrix
    dx_matrix = Gx_flat[:, None] - Gx_flat[None, :]
    dy_matrix = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Key k-points in the irreducible BZ
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.15, 0.0),
        (0.15, 0.15),
        (0.45, 0.0),
        (0.45, 0.45),
        (0.3, 0.0),
        (0.3, 0.3),
        (0.2, 0.1),
        (0.4, 0.1),
        (0.35, 0.35),
        (0.25, 0.1),
        (0.4, 0.3),
        (0.1, 0.0),
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        fft_vals = np.fft.fft2(cell.astype(float))
        fft_vals /= (size * size)
        
        E = fft_vals[np.mod(dx_matrix, size), np.mod(dy_matrix, size)]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = kGx**2 + kGy**2
            
            A = D[:, None] * E
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

    def generate_cell(A10, A20, A22, A30, A33, offset, phase20=0.0, phase22=0.0):
        """Generate a binary cell from Fourier parameters with C4v symmetry."""
        field = np.full((size, size), offset, dtype=float)
        
        if abs(A10) > 1e-10:
            field += A10 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))
        
        if abs(A20) > 1e-10:
            field += A20 * (np.cos(4 * np.pi * X + phase20) + np.cos(4 * np.pi * Y + phase20))
        
        if abs(A22) > 1e-10:
            field += A22 * (np.cos(4 * np.pi * (X + Y) + phase22) + np.cos(4 * np.pi * (X - Y) + phase22))
        
        if abs(A30) > 1e-10:
            field += A30 * (np.cos(6 * np.pi * X) + np.cos(6 * np.pi * Y))
        
        if abs(A33) > 1e-10:
            field += A33 * (np.cos(6 * np.pi * (X + Y)) + np.cos(6 * np.pi * (X - Y)))
        
        cell_raw = (field > 0.5).astype(int)
        
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
    
    # Stage 1: Coarse grid search
    A20_coarse = np.linspace(0.0, 1.5, 10)
    A22_coarse = np.linspace(0.0, 1.2, 8)
    offset_coarse = np.linspace(0.25, 0.75, 11)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                cell = generate_cell(0.0, A20, A22, 0.0, 0.0, offset)
                
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                
                score = calculate_surrogate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (0.0, A20, A22, 0.0, 0.0, offset, 0.0, 0.0)
    
    # Stage 2: Add A10 and A30
    if best_cell is not None:
        A10_coarse = np.linspace(-0.5, 0.5, 5)
        A30_coarse = np.linspace(-0.5, 0.5, 5)
        
        A20_best = best_params[1]
        A22_best = best_params[2]
        offset_best = best_params[5]
        
        A20_fine = np.linspace(max(0.0, A20_best - 0.3), min(2.0, A20_best + 0.3), 5)
        A22_fine = np.linspace(max(0.0, A22_best - 0.3), min(1.5, A22_best + 0.3), 5)
        offset_fine = np.linspace(max(0.2, offset_best - 0.1), min(0.8, offset_best + 0.1), 5)
        
        for A10 in A10_coarse:
            for A30 in A30_coarse:
                for A20 in A20_fine:
                    for A22 in A22_fine:
                        for offset in offset_fine:
                            cell = generate_cell(A10, A20, A22, A30, 0.0, offset)
                            
                            fill = np.sum(cell) / (size * size)
                            if fill < 0.15 or fill > 0.85:
                                continue
                            
                            score = calculate_surrogate_gap(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.tolist()
                                best_params = (A10, A20, A22, A30, 0.0, offset, 0.0, 0.0)
    
    # Stage 3: Phase optimization
    if best_cell is not None:
        A10_best = best_params[0]
        A20_best = best_params[1]
        A22_best = best_params[2]
        A30_best = best_params[3]
        offset_best = best_params[5]
        
        phase20_fine = np.linspace(-1.0, 1.0, 5)
        phase22_fine = np.linspace(-1.0, 1.0, 5)
        
        for p20 in phase20_fine:
            for p22 in phase22_fine:
                cell = generate_cell(A10_best, A20_best, A22_best, A30_best, 0.0, offset_best, p20, p22)
                
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                
                score = calculate_surrogate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (A10_best, A20_best, A22_best, A30_best, 0.0, offset_best, p20, p22)
    
    # Stage 4: Local refinement
    if best_cell is not None:
        A10_best, A20_best, A22_best, A30_best, A33_best, offset_best, p20_best, p22_best = best_params
        
        np.random.seed(42)
        for _ in range(50):
            dA10 = np.random.uniform(-0.15, 0.15)
            dA20 = np.random.uniform(-0.15, 0.15)
            dA22 = np.random.uniform(-0.15, 0.15)
            dA30 = np.random.uniform(-0.15, 0.15)
            doffset = np.random.uniform(-0.05, 0.05)
            dp20 = np.random.uniform(-0.2, 0.2)
            dp22 = np.random.uniform(-0.2, 0.2)
            
            A10 = A10_best + dA10
            A20 = max(0.0, A20_best + dA20)
            A22 = max(0.0, A22_best + dA22)
            A30 = A30_best + dA30
            offset = min(0.85, max(0.15, offset_best + doffset))
            p20 = p20_best + dp20
            p22 = p22_best + dp22
            
            cell = generate_cell(A10, A20, A22, A30, 0.0, offset, p20, p22)
            
            fill = np.sum(cell) / (size * size)
            if fill < 0.15 or fill > 0.85:
                continue
            
            score = calculate_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
                best_params = (A10, A20, A22, A30, 0.0, offset, p20, p22)
    
    # Stage 5: Known good configurations
    if best_cell is not None:
        test_configs = [
            (0.0, 0.8, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.0, 0.6, 0.7, 0.0, 0.0, 0.45, 0.0, 0.0),
            (0.0, 0.9, 0.3, 0.0, 0.0, 0.55, 0.0, 0.0),
            (0.0, 0.7, 0.6, 0.0, 0.0, 0.5, 0.2, 0.0),
            (0.0, 0.7, 0.6, 0.0, 0.0, 0.5, -0.2, 0.0),
            (0.0, 0.5, 0.8, 0.0, 0.0, 0.4, 0.0, 0.3),
            (0.0, 0.5, 0.8, 0.0, 0.0, 0.4, 0.0, -0.3),
            (0.0, 1.0, 0.4, 0.0, 0.0, 0.6, 0.0, 0.0),
            (0.0, 0.4, 1.0, 0.0, 0.0, 0.4, 0.0, 0.0),
            (0.2, 0.8, 0.5, 0.2, 0.0, 0.5, 0.0, 0.0),
            (-0.2, 0.8, 0.5, -0.2, 0.0, 0.5, 0.0, 0.0),
            (0.3, 0.7, 0.6, 0.1, 0.0, 0.45, 0.1, 0.1),
            (-0.3, 0.7, 0.6, -0.1, 0.0, 0.45, -0.1, -0.1),
        ]
        
        for A10, A20, A22, A30, A33, offset, p20, p22 in test_configs:
            cell = generate_cell(A10, A20, A22, A30, A33, offset, p20, p22)
            
            fill = np.sum(cell) / (size * size)
            if fill < 0.15 or fill > 0.85:
                continue
            
            score = calculate_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
    
    if best_cell is None:
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
