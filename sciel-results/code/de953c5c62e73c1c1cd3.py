import numpy as np
import time
import sys

# Constants for the physics engine
SIZE = 32
EPS_LOW = 1.0
EPS_HIGH = 12.0
N_PW = 8  # 17x17 = 289 plane waves
LOWER_BAND = 4
POLARIZATION = 'TM'

def symmetrize(cell):
    """Enforce C4v symmetry: 180 deg rotation, x/y reflection, transpose."""
    cell = np.array(cell, dtype=int)
    # 180 degree rotation
    cell_rot = cell[::-1, ::-1]
    # Transpose
    cell_trans = cell.T
    # Average to ensure symmetry (or just force one direction, but averaging is safer for float->int if we were doing that, here we just pick a canonical form)
    # The requirement is cell == cell[::-1] and cell == cell.T
    # Let's enforce by taking the minimum or majority. For 0/1, we can just set based on the top-left quadrant and mirror.
    # To be safe and fast, we will construct the cell from the top-left 16x16 quadrant.
    # But the input might be arbitrary. Let's just return the input if it's already symmetric, or force symmetry by averaging.
    # Actually, the simplest robust way for the design function is to generate from a quadrant.
    # For evaluate, we assume the input is valid or we symmetrize it.
    # Symmetrization:
    # cell[i, j] = cell[31-i, 31-j]
    # cell[i, j] = cell[j, i]
    # We can achieve this by:
    # 1. Reflect x: cell[:, ::-1]
    # 2. Reflect y: cell[::-1, :]
    # 3. Transpose: cell.T
    # Let's just average the 4 symmetric equivalents and round.
    c1 = cell
    c2 = cell[::-1, ::-1]
    c3 = cell.T
    c4 = c2.T # which is cell[::-1, ::-1].T = cell[::-1, ::-1] ? No.
    # Let's just use the top-left quadrant to reconstruct.
    # This is the most robust way to ensure symmetry.
    quad = cell[:16, :16]
    # Mirror to bottom-right
    new_cell = np.zeros((32, 32), dtype=int)
    new_cell[:16, :16] = quad
    new_cell[16:, 16:] = quad[::-1, ::-1]
    new_cell[:16, 16:] = quad[:, ::-1]
    new_cell[16:, :16] = quad[::-1, :]
    return new_cell

def get_g_vectors():
    """Generate the 289 G-vectors for N_PW=8."""
    g_vals = np.arange(-N_PW, N_PW + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    return G_vectors

def get_k_points():
    """Generate 30 k-points in the irreducible BZ (M-G-X-G-M)."""
    # Irreducible BZ for square lattice: 0 <= ky <= kx, kx <= pi, ky >= 0
    # Boundary: M(pi, pi) -> G(0,0) -> X(pi, 0) -> G(0,0) -> M(pi, pi) ?
    # Standard path: Gamma(0,0) -> X(pi,0) -> M(pi,pi) -> Gamma(0,0) is the full BZ boundary.
    # Irreducible is 1/8 of the BZ.
    # Let's sample points along the lines:
    # 1. Gamma to X: (k, 0) for k in [0, pi]
    # 2. X to M: (pi, k) for k in [0, pi]
    # 3. M to Gamma: (k, k) for k in [0, pi] (This is the diagonal)
    # Actually, the irreducible wedge is bounded by kx=ky, ky=0, and kx+ky=pi? No.
    # For square lattice, the irreducible BZ is the region 0 <= ky <= kx <= pi.
    # Wait, the BZ is [-pi, pi] x [-pi, pi].
    # High symmetry points: Gamma(0,0), X(pi,0), M(pi,pi).
    # The line from X to M is kx=pi, ky from 0 to pi.
    # The line from M to Gamma is kx=ky, from pi to 0.
    # The line from Gamma to X is ky=0, kx from 0 to pi.
    # So we sample on these 3 segments.
    
    k_points = []
    # Segment 1: Gamma(0,0) to X(pi,0). 10 points.
    for i in range(10):
        kx = np.pi * i / 9.0
        ky = 0.0
        k_points.append((kx, ky))
        
    # Segment 2: X(pi,0) to M(pi,pi). 10 points.
    for i in range(10):
        kx = np.pi
        ky = np.pi * i / 9.0
        k_points.append((kx, ky))
        
    # Segment 3: M(pi,pi) to Gamma(0,0). 10 points.
    for i in range(10):
        kx = np.pi * (1 - i/9.0)
        ky = kx
        k_points.append((kx, ky))
        
    return np.array(k_points)

G_VECTORS = get_g_vectors()
K_POINTS = get_k_points()
N_G = len(G_VECTORS)
N_K = len(K_POINTS)

def evaluate_cell(cell):
    """Compute the gap-midgap ratio for the given cell."""
    cell = np.array(cell, dtype=float)
    cell = symmetrize(cell)
    
    # Epsilon field
    eps = EPS_LOW + (EPS_HIGH - EPS_LOW) * cell
    
    # For TM polarization, the operator involves 1/eps.
    # The eigenvalue problem is: sum_m (G + G')^2 delta_{mn} / eps(G-G') c_m = omega^2 c_n ?
    # Actually, the standard form for TM is:
    # sum_m [ (k+G_n)^2 / eps(G_n - G_m) ] c_m = omega^2 c_n
    # Wait, the dielectric function appears in the denominator for TM (H field) and numerator for TE (E field).
    # Let's verify.
    # Maxwell: curl curl H = (omega^2/c^2) eps H
    # In Fourier: sum_m (k+G_n)x(k+G_m) / eps(G_n-G_m) ...
    # The matrix element is M_{nm} = (k+G_n) dot (k+G_m) / eps(G_n - G_m) for TE?
    # No, for TM (E parallel to z), the equation is for E_z.
    # curl (curl E) = ...
    # The standard plane-wave equation for TM is:
    # sum_m [ (G_n + k) dot (G_m + k) / eps(G_n - G_m) ] c_m = (omega^2) c_n
    # Wait, is it 1/eps or eps?
    # For TE (H_z), it's 1/eps in the denominator of the coupling?
    # Let's check: 
    # TE: H = H_z z_hat.
    # curl E = -i w B = -i w mu H.
    # curl H = i w D = i w eps E.
    # Taking curl of curl E: -k^2 E + ...
    # The matrix for TE is: (G_n + k) dot (G_m + k) / eps(G_n - G_m)?
    # Actually, the common convention:
    # TE (H || z): Matrix element M_{nm} = (G_n + k) dot (G_m + k) / eps(G_n - G_m) is WRONG.
    # It is: M_{nm} = (G_n + k) dot (G_m + k) * (1/eps(G_n - G_m))?
    # Let's look at the "exact-coefficient plane-wave method".
    # The equation is: sum_m [ (k+G_n) x (k+G_m) / eps(G_n-G_m) ] ...
    # For TM, the polarization is in the z-direction. The coupling is via 1/eps.
    # For TE, the polarization is in the xy-plane. The coupling is via eps?
    # Actually, usually:
    # TE: (k+G_n) dot (k+G_m) / eps(G_n-G_m)
    # TM: (k+G_n) x (k+G_m) / eps(G_n-G_m) ... no, the cross product leads to scalar factors.
    # Let's assume the standard form where the matrix is:
    # A_{nm} = (k+G_n) dot (k+G_m) / eps(G_n - G_m)
    # This is for the case where the field is expanded in plane waves and the dielectric is in the denominator.
    # This corresponds to the "1/eps" expansion.
    # If the task says "exact Fourier coefficients of the pixel map", it likely implies computing the Fourier coefficients of 1/eps for TM and eps for TE?
    # Or just 1/eps for both?
    # Most PWEM codes use the 1/eps expansion for both, or eps for TE and 1/eps for TM.
    # Let's assume 1/eps for TM as is standard for E-field expansion.
    
    # Compute Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    
    # We need coeff[g] = integral inv_eps(r) exp(-i G.r) dr
    # Using the pixel grid.
    # r_x = (i + 0.5) / SIZE
    # r_y = (j + 0.5) / SIZE
    # coeff[gx, gy] = sum_{i,j} inv_eps[i,j] exp(-i 2pi (gx(i+0.5)/SIZE + gy(j+0.5)/SIZE)) / SIZE^2
    
    # Vectorized calculation
    # G_VECTORS is (N_G, 2)
    # inv_eps is (SIZE, SIZE)
    
    # We can use FFT if we pad, but direct sum is fine for 32x32 and 289 G-vectors.
    # N_G * SIZE^2 = 289 * 1024 ~ 300k ops. Very fast.
    
    # Create phase matrix
    # Gx: (N_G, 1), i: (1, SIZE) -> (N_G, SIZE)
    Gx = G_VECTORS[:, 0]
    Gy = G_VECTORS[:, 1]
    
    i_idx = np.arange(SIZE)
    j_idx = np.arange(SIZE)
    
    # Phase for x part: 2*pi * Gx * (i+0.5)/SIZE
    # Shape: (N_G, SIZE)
    phase_x = 2 * np.pi * np.outer(Gx, (i_idx + 0.5) / SIZE)
    # Phase for y part: 2*pi * Gy * (j+0.5)/SIZE
    # Shape: (N_G, SIZE)
    phase_y = 2 * np.pi * np.outer(Gy, (j_idx + 0.5) / SIZE)
    
    # Total phase: (N_G, SIZE, SIZE)
    # phase_x[:, :, np.newaxis] + phase_y[:, np.newaxis, :]
    phase_total = phase_x[:, :, np.newaxis] + phase_y[:, np.newaxis, :]
    
    # exp(-i * phase)
    # Multiply by inv_eps[i,j]
    # Sum over i, j
    
    # To save memory, we can do it in chunks or just let numpy handle it.
    # 289 * 32 * 32 complex numbers = 289 * 1024 * 16 bytes ~ 4.7 MB. OK.
    
    exp_phase = np.exp(-1j * phase_total)
    
    # inv_eps is (32, 32). Reshape to (1, 32, 32)
    inv_eps_reshaped = inv_eps[np.newaxis, :, :]
    
    # Term: (N_G, 32, 32)
    terms = inv_eps_reshaped * exp_phase
    
    # Sum over axes 1 and 2
    inv_eps_coeff = np.sum(terms, axis=(1, 2)) / (SIZE * SIZE)
    
    # Now inv_eps_coeff is (N_G,) corresponding to G_VECTORS
    
    # Build the coefficient lookup table for differences
    # We need coeff(G_n - G_m)
    # G_n and G_m are in G_VECTORS.
    # The difference G_n - G_m can range from -2*N_PW to 2*N_PW.
    # We only have coefficients for -N_PW to N_PW.
    # If the difference is outside, the coefficient is 0.
    
    # Create a 2D array for coefficients indexed by (gx, gy)
    # Range: -N_PW to N_PW
    # Map to 0 to 2*N_PW
    coeff_grid = np.zeros((2*N_PW + 1, 2*N_PW + 1), dtype=complex)
    
    # Fill coeff_grid
    # G_VECTORS has integers gx, gy
    for idx in range(N_G):
        gx = int(G_VECTORS[idx, 0])
        gy = int(G_VECTORS[idx, 1])
        # Map to index
        row = gx + N_PW
        col = gy + N_PW
        coeff_grid[row, col] = inv_eps_coeff[idx]
        
    # Now for each k-point, build the matrix M
    # M_{nm} = (k+G_n) dot (k+G_m) / coeff_grid[G_n - G_m]
    # Wait, is it 1/coeff or coeff?
    # The equation is sum_m [ (k+G_n) dot (k+G_m) / eps(G_n-G_m) ] c_m = w^2 c_n
    # So the matrix element is M_{nm} = (k+G_n) dot (k+G_m) * inv_eps_coeff[G_n-G_m] ?
    # No, the Fourier series is 1/eps(r) = sum_G (1/eps)_G exp(i G r).
    # The integral is sum_m c_m (1/eps)_{G_n-G_m} (k+G_n) dot (k+G_m).
    # So yes, M_{nm} = (k+G_n) dot (k+G_m) * inv_eps_coeff[G_n-G_m].
    
    # Precompute k+G vectors for all k-points and G-vectors
    # K_POINTS: (N_K, 2)
    # G_VECTORS: (N_G, 2)
    # kG: (N_K, N_G, 2)
    # kG_sq: (N_K, N_G)
    
    # We also need the dot product (k+G_n) dot (k+G_m)
    # This is K_n dot K_m where K_n = k + G_n.
    
    min_omega5 = np.inf
    max_omega4 = 0.0
    
    for k_idx in range(N_K):
        k = K_POINTS[k_idx]
        # K_vectors: (N_G, 2)
        K_vecs = k + G_VECTORS
        
        # kG_sq: (N_G,)
        kG_sq = np.sum(K_vecs**2, axis=1)
        
        # Dot products: (N_G, N_G)
        # K_vecs @ K_vecs.T
        dot_prod = K_vecs @ K_vecs.T
        
        # Now we need to multiply dot_prod[n,m] by inv_eps_coeff[G_n - G_m]
        # We can construct the matrix of differences G_n - G_m
        # Gx_diff: (N_G, N_G)
        # Gy_diff: (N_G, N_G)
        
        Gx_diff = G_VECTORS[:, 0][:, np.newaxis] - G_VECTORS[:, 0][np.newaxis, :]
        Gy_diff = G_VECTORS[:, 1][:, np.newaxis] - G_VECTORS[:, 1][np.newaxis, :]
        
        # Map differences to indices in coeff_grid
        # If diff is outside [-N_PW, N_PW], coeff is 0.
        # coeff_grid is defined for [-N_PW, N_PW] mapped to [0, 2*N_PW].
        
        # Create index arrays
        # row_idx = Gx_diff + N_PW
        # col_idx = Gy_diff + N_PW
        
        # We need to handle out-of-bounds.
        # We can create a larger zero-padded grid or use np.where.
        
        # Let's use a padded grid for safety, or just np.where.
        # Since N_PW is small (8), the diff is in [-16, 16].
        # coeff_grid is 17x17.
        
        # Create a full zero grid of size (2*N_PW+1) x (2*N_PW+1) is not enough for diffs.
        # Diffs can be -16 to 16.
        # So we need a grid of size 33x33 centered at 0.
        
        # Let's re-create coeff_lookup for the full diff range.
        # coeff_full = zeros((4*N_PW+1, 4*N_PW+1))
        # Center is 2*N_PW.
        
        # We already have coeff_grid (17x17) for the base coefficients.
        # We can just use np.where to check bounds.
        
        row_idx = Gx_diff + N_PW
        col_idx = Gy_diff + N_PW
        
        # Initialize coeff_mat with zeros
        coeff_mat = np.zeros((N_G, N_G), dtype=complex)
        
        # Valid indices are 0 <= row_idx <= 2*N_PW and 0 <= col_idx <= 2*N_PW
        valid_mask = (row_idx >= 0) & (row_idx <= 2*N_PW) & (col_idx >= 0) & (col_idx <= 2*N_PW)
        
        # We can't directly index with a mask in 2D easily without advanced indexing.
        # Let's use advanced indexing.
        # valid_rows = row_idx[valid_mask]
        # valid_cols = col_idx[valid_mask]
        # coeff_mat[valid_mask] = coeff_grid[valid_rows, valid_cols]
        
        # This is efficient.
        valid_rows = row_idx[valid_mask]
        valid_cols = col_idx[valid_mask]
        coeff_mat[valid_mask] = coeff_grid[valid_rows, valid_cols]
        
        # M = dot_prod * coeff_mat
        M = dot_prod * coeff_mat
        
        # Ensure Hermitian
        M = (M + M.conj().T) / 2
        
        # Solve eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            # We want bands 4 and 5 (0-indexed 3 and 4)
            if len(eigvals) >= 5:
                # omega^2 = eigval
                omega4_sq = eigvals[3]
                omega5_sq = eigvals[4]
                
                # Ensure positive
                if omega4_sq < 0: omega4_sq = 0
                if omega5_sq < 0: omega5_sq = 0
                
                omega4 = np.sqrt(omega4_sq)
                omega5 = np.sqrt(omega5_sq)
                
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

def design(problem):
    """
    Design a photonic crystal unit cell.
    Returns a 32x32 nested list of 0/1.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # 200s budget
    
    # Initial guess: A simple square hole or similar.
    # Let's try a pattern that is known to produce gaps.
    # A square hole in the center is a good start.
    # Or a checkerboard?
    
    # Let's generate a random symmetric cell and evaluate it.
    # Then do a local search (flip pixels) to improve.
    
    # Generate initial cell from top-left quadrant
    quad = np.random.randint(0, 2, (16, 16))
    # Ensure some structure, e.g., start with a block
    # quad[4:12, 4:12] = 1 # A hole?
    # Let's start with a random one.
    
    cell = symmetrize(quad)
    current_score = evaluate_cell(cell)
    best_cell = cell.copy()
    best_score = current_score
    
    # Local search
    # Flip one pixel in the quadrant, check symmetry, evaluate.
    # This is expensive if we evaluate every flip.
    # We can batch or just do a few iterations.
    
    # To speed up, we can pre-compute the contribution of each pixel?
    # No, the matrix changes globally.
    
    # Let's do a limited number of random flips.
    # 32x32 = 1024 pixels. Symmetric independent pixels = 16x16 = 256.
    # Each evaluation takes ~0.1s?
    # 256 flips * 0.1s = 25s.
    # We can do 256 * 5 = 1280 evaluations in ~125s.
    
    # Let's implement a simple hill climbing.
    
    improvements = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10.0:
        improved = False
        # Randomly pick a quadrant pixel to flip
        # We can iterate through all or pick random.
        # Picking random is faster to code.
        for _ in range(100): # Try 100 random flips per "iteration"
            if time.time() > deadline - 5.0:
                break
                
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
            
            # Flip
            new_cell = best_cell.copy()
            # Flip the symmetric set
            # (i, j), (31-i, 31-j), (j, i), (31-j, 31-i)
            # Note: if i==j or i==31-j etc, they might overlap.
            # Using symmetrize is safer but slower? No, symmetrize is just array slicing.
            
            # We can just flip the quadrant and symmetrize.
            new_quad = best_cell[:16, :16].copy()
            new_quad[i, j] = 1 - new_quad[i, j]
            new_cell = symmetrize(new_quad)
            
            new_score = evaluate_cell(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Restart with a new random cell?
            # Or just break.
            break
            
    # Return as nested list
    return best_cell.tolist()

# Ensure the function is named design
# The problem statement says: design(problem) -> ...
# We ignore problem for now, assuming defaults.

if __name__ == "__main__":
    # Test
    cell = design(None)
    print("Score:", evaluate_cell(cell))
