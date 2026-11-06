import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order C4v-symmetric Fourier harmonics. Optimizes amplitudes to 
    maximize the gap between the 4th and 5th TM bands.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Define C4v symmetric basis functions
    # We use cos terms to ensure real-valued and symmetric distributions.
    # For C4v symmetry, we need invariance under:
    # 1. x -> -x, y -> y (mirror)
    # 2. x -> y, y -> x (rotation 90)
    # 3. x -> -x, y -> -y (rotation 180)
    # etc.
    
    # Basis functions:
    # (1,0) + (0,1): cos(2πx) + cos(2πy)
    # (2,0) + (0,2): cos(4πx) + cos(4πy)
    # (2,2) + (2,-2): cos(4π(x+y)) + cos(4π(x-y))
    # (3,0) + (0,3): cos(6πx) + cos(6πy)
    # (3,3) + (3,-3): cos(6π(x+y)) + cos(6π(x-y))
    # (4,0) + (0,4): cos(8πx) + cos(8πy)
    # (4,4) + (4,-4): cos(8π(x+y)) + cos(8π(x-y))
    
    # We will parameterize the field as:
    # F(x,y) = a0 + a1*(cos(2πx)+cos(2πy)) + a2*(cos(4πx)+cos(4πy)) + 
    #          a3*(cos(4π(x+y))+cos(4π(x-y))) + a4*(cos(6πx)+cos(6πy)) +
    #          a5*(cos(6π(x+y))+cos(6π(x-y))) + a6*(cos(8πx)+cos(8πy)) +
    #          a7*(cos(8π(x+y))+cos(8π(x-y)))
    
    # Precompute basis matrices for fast evaluation
    # Note: For binary output, we threshold F(x,y) > 0.5 (or some offset)
    # To allow for fill fraction control, we include a DC term a0.
    
    def generate_cell(params):
        """
        params: [a0, a1, a2, a3, a4, a5, a6, a7]
        Returns a 32x32 binary cell.
        """
        a0, a1, a2, a3, a4, a5, a6, a7 = params
        
        field = np.full((size, size), a0)
        
        if a1 != 0:
            field += a1 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))
        if a2 != 0:
            field += a2 * (np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y))
        if a3 != 0:
            field += a3 * (np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y)))
        if a4 != 0:
            field += a4 * (np.cos(6 * np.pi * X) + np.cos(6 * np.pi * Y))
        if a5 != 0:
            field += a5 * (np.cos(6 * np.pi * (X + Y)) + np.cos(6 * np.pi * (X - Y)))
        if a6 != 0:
            field += a6 * (np.cos(8 * np.pi * X) + np.cos(8 * np.pi * Y))
        if a7 != 0:
            field += a7 * (np.cos(8 * np.pi * (X + Y)) + np.cos(8 * np.pi * (X - Y)))
        
        # Threshold to binary
        # We want to enforce exact symmetry. Since the basis functions are exactly C4v symmetric
        # and the grid is symmetric, the resulting field is symmetric.
        # However, floating point errors might break exact integer symmetry if not careful.
        # We generate the raw binary cell and then enforce symmetry by averaging over group operations.
        
        cell_raw = (field > 0.5).astype(int)
        
        # Enforce C4v symmetry strictly
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

    # Fast Surrogate PWE Solver
    # Use a smaller number of plane waves for speed during optimization.
    # M_PWE = 4 gives 9x9 = 81 PWs. This is a good balance.
    M_PWE = 4
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # The irreducible BZ for a square lattice is the triangle (0,0)-(0.5,0)-(0.5,0.5).
    # We sample points along the high-symmetry lines and interior.
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.0),      # Midpoint Gamma-X
        (0.25, 0.25),     # Midpoint Gamma-M (diagonal)
        (0.5, 0.25),      # Midpoint X-M
        (0.15, 0.15),     # Interior near Gamma
        (0.35, 0.35),     # Interior near M
        (0.4, 0.1),       # Interior
        (0.45, 0.45),     # Near M
        (0.1, 0.4),       # Interior
        (0.3, 0.4),       # Interior
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        # Note: The evaluator uses exact coefficients. We use FFT for speed.
        # The FFT of a real symmetric grid is real and symmetric.
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        # Precompute difference indices for the E matrix
        # E[i,j] = eps(G_i - G_j)
        # We can precompute the indices for all i,j pairs
        # But since N_PW is small (81), we can just do it in a loop or vectorize.
        
        # Vectorized E matrix construction
        # Gx_diff[i,j] = Gx_flat[i] - Gx_flat[j]
        # We need to map these differences to the FFT grid indices.
        # The FFT grid indices go from 0 to size-1.
        # The difference d can be negative. We use modulo arithmetic.
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        # Precompute E matrix once? No, it depends on cell (fft_vals).
        # But the structure of indices is fixed.
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = (kGx**2 + kGy**2)
            
            # Construct E matrix
            # E[i,j] = fft_vals[ (Gx_flat[i] - Gx_flat[j]) % size, (Gy_flat[i] - Gy_flat[j]) % size ]
            
            # Create index matrices
            ix_idx = (Gx_flat[:, None] - Gx_flat[None, :]) % size
            iy_idx = (Gy_flat[:, None] - Gy_flat[None, :]) % size
            
            E = fft_vals[ix_idx, iy_idx]
            
            # TM mode equation:
            # For TM, the eigenvalue equation is:
            # sum_j (k_x_i * k_x_j + k_y_i * k_y_j) / (k_i * k_j) * eps(G_i - G_j) * c_j = omega^2 * c_i
            # Wait, the standard PWE form for TM is:
            # H * c = omega^2 * c
            # H_ij = (k_i x k_j) * eps(G_i - G_j) / (k_i * k_j) ? No.
            
            # Let's use the standard formulation:
            # For TM:
            # (k + G_i) x (k + G_j) * eps(G_i - G_j) * c_j = omega^2 * c_i
            # Where (k+G_i) x (k+G_j) is the dot product of the vectors (k+G_i) and (k+G_j).
            # Actually, the operator is:
            # 1/eps(k) * (k+G) x (k+G) ... no.
            
            # Correct TM PWE:
            # The field is H_z.
            # Maxwell: k x (k x H) = omega^2 / c^2 * eps * H
            # (k^2 H - k(k.H)) = omega^2/eps H
            # In PWE basis:
            # sum_j [ (k+G_i) . (k+G_j) / ( (k+G_i) . (k+G_j) ) ? No.
            
            # Let's use the matrix form:
            # A_ij = ( (k+G_i) . (k+G_j) ) * eps(G_i - G_j) / ( (k+G_i) . (k+G_i) ) ? No.
            
            # Standard form from Joannopoulos:
            # For TM:
            # ( (k+G_i) . (k+G_j) ) * eps(G_i - G_j) = omega^2 * delta_ij ? No.
            
            # Let's derive quickly.
            # (k+G_i)^2 H_i - (k+G_i) . sum_j (k+G_j) H_j = omega^2 / eps_i H_i
            # This is not a simple eigenvalue problem directly on H.
            
            # Alternative form used in many codes:
            # Eigenvalue problem for omega^2:
            # sum_j [ (k+G_i) . (k+G_j) * eps(G_i - G_j) ] c_j = omega^2 * ( (k+G_i) . (k+G_i) ) c_i ? No.
            
            # Let's use the form:
            # M_ij = ( (k+G_i) . (k+G_j) ) * eps(G_i - G_j)
            # And solve M c = omega^2 * I c ? No, the RHS is not identity.
            
            # The correct generalized eigenvalue problem is:
            # sum_j ( (k+G_i) . (k+G_j) / ( (k+G_i) . (k+G_i) ) ) * eps(G_i - G_j) * c_j = omega^2 c_i ? No.
            
            # Let's stick to the most common formulation:
            # For TM, the eigenvalue equation is:
            # sum_j [ ( (k+G_i) . (k+G_j) ) * eps(G_i - G_j) ] c_j = omega^2 * ( (k+G_i) . (k+G_i) ) c_i
            # This is a generalized eigenvalue problem A c = omega^2 B c.
            # A_ij = ( (k+G_i) . (k+G_j) ) * eps(G_i - G_j)
            # B_ij = ( (k+G_i) . (k+G_i) ) * delta_ij
            
            # We can convert to standard eigenvalue problem by B^{-1/2} A B^{-1/2}.
            # Or use scipy.linalg.eigh(A, B).
            
            # Let's construct A and B.
            # (k+G_i) . (k+G_j) = (kx_i + Gx_i)(kx_j + Gx_j) + (ky_i + Gy_i)(ky_j + Gy_j)
            
            # Vectorized dot products
            # Kx = kx + Gx_flat
            # Ky = ky + Gy_flat
            # Dot_ij = Kx_i Kx_j + Ky_i Ky_j
            
            Kx = kx + Gx_flat
            Ky = ky + Gy_flat
            
            Dot = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            
            A = Dot * E
            # A is symmetric?
            # Dot is symmetric. E is symmetric (eps(G_i-G_j) = eps(G_j-G_i) for real eps? 
            # eps is real and even in G, so yes).
            # So A is symmetric.
            
            # B is diagonal with elements (k+G_i)^2 = Kx_i^2 + Ky_i^2
            B_diag = Kx**2 + Ky**2
            
            # Solve A c = omega^2 B c
            # Using scipy.linalg.eigh(A, B)
            try:
                eigvals, eigvecs = np.linalg.eigh(A, B)
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

    # Optimization Loop
    # We will use a simple local search / hill climbing on the parameters.
    # Parameters: [a0, a1, a2, a3, a4, a5, a6, a7]
    # Initial guess: A typical "cross" or "ring" shape.
    # For a square lattice TM gap, a "cross" shape (high fill in center, low in corners) or "ring" is often good.
    # Let's start with a few diverse initial points.
    
    initial_params_list = [
        [0.5, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],  # Basic cross
        [0.5, -0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0], # Inverted cross?
        [0.4, 0.3, 0.5, 0.4, 0.1, 0.1, 0.0, 0.0],  # Complex
        [0.6, 0.0, 0.8, 0.5, 0.2, 0.2, 0.1, 0.1],  # High order
        [0.5, 0.2, -0.3, 0.4, -0.2, 0.3, 0.1, -0.1], # Mixed signs
    ]
    
    best_score = -np.inf
    best_params = None
    best_cell = None
    
    # Time budget check
    start_time = time.time()
    time_limit = 100  # Leave 20s for final processing
    
    # Perform local optimization for each initial point
    for init_params in initial_params_list:
        if time.time() - start_time > time_limit:
            break
            
        current_params = np.array(init_params)
        current_cell = generate_cell(current_params)
        current_score = calculate_surrogate_gap(current_cell)
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_cell = current_cell.tolist()
        
        # Local search: Perturb parameters
        step_size = 0.1
        num_iterations = 20
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(len(current_params)):
                # Try increasing
                new_params = current_params.copy()
                new_params[i] += step_size
                new_cell = generate_cell(new_params)
                # Check fill fraction
                fill = np.sum(new_cell) / (size * size)
                if 0.1 < fill < 0.9:
                    new_score = calculate_surrogate_gap(new_cell)
                    if new_score > current_score:
                        current_params = new_params
                        current_cell = new_cell
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
                            best_cell = current_cell.tolist()
                
                # Try decreasing
                new_params = current_params.copy()
                new_params[i] -= step_size
                new_cell = generate_cell(new_params)
                fill = np.sum(new_cell) / (size * size)
                if 0.1 < fill < 0.9:
                    new_score = calculate_surrogate_gap(new_cell)
                    if new_score > current_score:
                        current_params = new_params
                        current_cell = new_cell
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
                            best_cell = current_cell.tolist()
            
            # Reduce step size if no improvement or to refine
            if not improved:
                step_size *= 0.5
                if step_size < 0.01:
                    break

    if best_cell is None:
        # Fallback
        best_cell = []
        for i in range(size):
            xi = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                yj = (j + 0.5) / size - 0.5
                if abs(xi) < 0.2 or abs(yj) < 0.2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell
