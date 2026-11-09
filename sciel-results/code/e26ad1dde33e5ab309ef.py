import numpy as np
import time

def design(problem):
    """
    Optimizes a symmetric 32x32 photonic crystal unit cell to maximize
    the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Precomputation ---
    # G vectors for 289 plane waves (square cutoff N=8)
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)

    # K-points in the irreducible Brillouin zone (1/8 of the square BZ)
    # The irreducible BZ is defined by 0 <= kx <= ky <= 0.5
    # We sample points along the boundaries and interior.
    k_points = []
    # Gamma (0,0) to M (0, 0.5) -- wait, standard notation:
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Irreducible BZ is 0 <= kx <= 0.5, 0 <= ky <= kx? No.
    # For square lattice, high symmetry points: Gamma(0,0), M(pi/a, 0) -> (0.5, 0) in units of 2pi/a?
    # Let's use units where the reciprocal lattice vectors are (2pi, 0) and (0, 2pi).
    # Then Gamma = (0,0), X = (0.5, 0) (midpoint of BZ edge), M = (0.5, 0.5) (corner).
    # The irreducible BZ is the triangle with vertices Gamma(0,0), X(0.5,0), M(0.5,0.5).
    # Wait, the symmetry group is C4v. The fundamental domain is 1/8 of the square.
    # Vertices: (0,0), (0.5, 0), (0.5, 0.5).
    
    # Let's generate 30 points covering this triangle.
    # We can use a grid or specific paths.
    # Paths: Gamma-X, X-M, Gamma-M.
    # Gamma-X: (t, 0) for t in [0, 0.5]
    # X-M: (0.5, t) for t in [0, 0.5]
    # Gamma-M: (t, t) for t in [0, 0.5]
    
    num_k_path = 10
    for t in np.linspace(0, 0.5, num_k_path):
        k_points.append((t, 0.0))
    for t in np.linspace(0, 0.5, num_k_path):
        k_points.append((0.5, t))
    for t in np.linspace(0, 0.5, num_k_path):
        k_points.append((t, t))
    
    # Remove duplicates
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
            
    # Ensure we have 30 points. If we have more, subsample. If less, add random interior.
    if len(unique_k) > 30:
        # Stratified sampling
        idx = np.linspace(0, len(unique_k)-1, 30).astype(int)
        unique_k = [unique_k[i] for i in idx]
    elif len(unique_k) < 30:
        # Add random points in the triangle
        while len(unique_k) < 30:
            t1 = np.random.random()
            t2 = np.random.random()
            if t1 + t2 > 1:
                t1, t2 = 1 - t1, 1 - t2
            kx = t1 * 0.5
            ky = t2 * 0.5
            unique_k.append((kx, ky))

    # Precompute (k+G)^2 for each k-point and G-vector
    # K_G_SQ[k_idx, G_idx] = |k + G|^2
    K_G_SQ = np.zeros((len(unique_k), num_pw))
    for ki, (kx, ky) in enumerate(unique_k):
        k_vec = np.array([kx, ky])
        KG = k_vec + G_vectors.T
        K_G_SQ[ki] = np.sum(KG**2, axis=1)

    # --- Evaluation Function ---
    def evaluate_cell(cell_binary):
        """
        Computes the gap-midgap ratio for a 32x32 binary cell.
        """
        # Build epsilon grid
        eps_grid = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if cell_binary[i][j] == 1:
                    eps_grid[i, j] = eps_high
                else:
                    eps_grid[i, j] = eps_low

        # For TM mode, we need Fourier coefficients of 1/epsilon
        inv_eps_grid = 1.0 / eps_grid
        
        # Compute FFT of 1/epsilon
        # The grid is 32x32.
        # We need coefficients for G in [-8, 8].
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)

        # Extract coefficients for G in [-N_pw, N_pw]
        # The FFT output is ordered [0, 1, ..., 15, -16, ..., -1]
        # We need a helper to get coefficient for integer G
        def get_fft_coeff(fft_arr, gx, gy):
            ix = gx % size
            iy = gy % size
            return fft_arr[ix, iy]

        # Precompute the convolution matrix C where C[i, j] = (1/eps)_{G_i - G_j}
        # This matrix is independent of k.
        C = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            Gi = G_vectors[i]
            for j in range(num_pw):
                Gj = G_vectors[j]
                dx = int(Gi[0] - Gj[0])
                dy = int(Gi[1] - Gj[1])
                C[i, j] = get_fft_coeff(fft_inv_eps, dx, dy)

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for ki in range(len(unique_k)):
            # M[i, j] = C[i, j] * K_G_SQ[ki, i]
            # Note: The operator is H = (1/eps) * (k+G)^2.
            # In plane wave basis, H_{G, G'} = (1/eps)_{G-G'} * (k+G)^2.
            # So M = C * diag(K_G_SQ[ki])
            M = C * K_G_SQ[ki]
            
            # Solve eigenvalue problem
            # M is not Hermitian, but eigenvalues should be real.
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Bands are indexed 1..N. We want bands 4 and 5.
                # Indices 3 and 4 (0-based).
                omega_4_sq = real_eigvals[3]
                omega_5_sq = real_eigvals[4]
                
                if omega_4_sq < 0 or omega_5_sq < 0:
                    continue
                    
                omega_4 = np.sqrt(omega_4_sq)
                omega_5 = np.sqrt(omega_5_sq)
                
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5

        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # --- Optimization ---
    # We will use a random search with local refinement.
    # The search space is the set of C4-symmetric binary 32x32 grids.
    # We can represent the grid by its 1/8th (the triangle).
    # Number of pixels in the triangle:
    # Total 32x32 = 1024.
    # 1/8th is 128 pixels.
    # However, the symmetry is about the center.
    # Let's just generate random C4-symmetric grids and evaluate them.
    
    def generate_random_cell():
        # Generate a 16x16 top-left quadrant? No, symmetry is about center.
        # Let's generate the full 32x32 and enforce symmetry.
        # We can generate the top-left 16x16 and mirror.
        # Wait, the center is between pixels (15,15) and (16,16)?
        # No, 32 is even. Center is at (15.5, 15.5).
        # Symmetry: cell[i][j] == cell[31-i][31-j]
        # Also cell[i][j] == cell[j][i]
        
        # Let's generate a 16x16 grid and reflect it.
        # The fundamental domain is the upper-right triangle of the 16x16?
        # Actually, it's easier to generate the 32x32 and enforce constraints.
        # Or generate the 1/8th part.
        
        # Let's just generate a random 16x16 and mirror to 32x32.
        # This ensures 180-degree rotational symmetry and reflection across x=15.5, y=15.5.
        # But we also need reflection across y=x.
        
        # Let's generate the 16x16 top-left quadrant.
        # Then:
        # Top-Right: mirror Top-Left horizontally? No.
        # Center is at 15.5.
        # Row i, Col j (i,j in 0..31)
        # Symmetry 1: (i, j) -> (31-i, 31-j)
        # Symmetry 2: (i, j) -> (j, i)
        
        # Let's generate the 16x16 block where i in 0..15, j in 0..15.
        # This is the top-left quadrant.
        # From this, we can get:
        # Top-Right (i in 0..15, j in 16..31): j' = 31-j. So cell[i][j] = cell[i][31-j] = cell[i][j'] where j' in 0..15.
        # Bottom-Left (i in 16..31, j in 0..15): i' = 31-i. cell[i][j] = cell[31-i][j] = cell[i'][j].
        # Bottom-Right (i in 16..31, j in 16..31): cell[i][j] = cell[31-i][31-j] = cell[i'][j'].
        
        # This satisfies 180-degree rotation and reflections across the midlines.
        # Does it satisfy y=x symmetry?
        # cell[i][j] = cell[j][i].
        # If we just mirror the 16x16 block, we don't enforce y=x symmetry within the block.
        # We need cell[i][j] = cell[j][i] for all i,j.
        # So we need to enforce that the 16x16 block is symmetric? No, that's not enough.
        # cell[0][1] must equal cell[1][0].
        # cell[0][20] must equal cell[20][0].
        
        # Let's generate the 1/8th part: the triangle i <= j in the 16x16?
        # No, the fundamental domain is the triangle in the 32x32.
        
        # Easiest way: Generate a random 32x32, then enforce symmetry.
        cell = np.random.randint(0, 2, (size, size))
        # Enforce 180-degree rotation
        cell = (cell + cell[::-1, ::-1]) // 2
        # Enforce transpose symmetry
        cell = (cell + cell.T) // 2
        # Re-enforce 180-degree rotation (since transpose might break it? No, (A+A^T) is symmetric, and (A+A^T)^T = A+A^T. (A+A^T)[::-1, ::-1] = A[::-1, ::-1] + A^T[::-1, ::-1] = A[::-1, ::-1] + (A[::-1, ::-1])^T. So it's still symmetric under both.)
        # Let's just do:
        # cell[i][j] = 1 if any of (i,j), (31-i,31-j), (j,i), (31-j,31-i) is 1?
        # Or AND?
        # Usually we want to preserve the "intensity". OR is safer for keeping features.
        # Let's use OR.
        sym_cell = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                sym_cell[i, j] = cell[i, j] | cell[31-i, 31-j] | cell[j, i] | cell[31-j, 31-i]
        return sym_cell.tolist()

    best_cell = None
    best_score = -np.inf
    
    # Initial candidates
    candidates = []
    
    # 1. Random
    candidates.append(generate_random_cell())
    
    # 2. Rod
    rod = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.15:
                rod[i][j] = 1
    candidates.append(rod)
    
    # 3. Ring
    ring = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.1 < r < 0.25:
                ring[i][j] = 1
    candidates.append(ring)
    
    # 4. Square rod
    sq_rod = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.15 and abs(y) < 0.15:
                sq_rod[i][j] = 1
    candidates.append(sq_rod)
    
    # Evaluate initial candidates
    for cell in candidates:
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # Random search with local refinement
    num_iterations = 100
    for it in range(num_iterations):
        if time.time() > deadline - 20:
            break
            
        # Generate a new random cell
        cell = generate_random_cell()
        
        # Local refinement: flip random pixels
        for _ in range(5):
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                
            # Flip a random pixel
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            cell[i][j] = 1 - cell[i][j]
            # Enforce symmetry
            sym_cell = np.zeros_like(cell)
            for ii in range(size):
                for jj in range(size):
                    sym_cell[ii, jj] = cell[ii, jj] | cell[31-ii, 31-jj] | cell[jj, ii] | cell[31-jj, 31-ii]
            cell = sym_cell.tolist()
            
            if time.time() > deadline - 20:
                break

    # Return the best cell
    if best_cell is None:
        # Fallback
        best_cell = [[0 for _ in range(size)] for _ in range(size)]
        
    return best_cell
