"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_fourier_coeffs(cell, size, n_pw):
    """Compute Fourier coefficients of 1/eps for the cell using FFT."""
    eps = np.where(np.array(cell, dtype=float) > 0.5, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    # Compute 2D FFT of inv_eps
    # Note: Standard FFT gives coefficients for k in 0..N-1.
    # We need coefficients for G in -N//2 .. N//2.
    # Using fftshift allows us to index by negative/positive integers directly.
    fft_coeffs = np.fft.fftshift(np.fft.fft2(inv_eps)) / (size * size)
    
    coeffs = {}
    half = n_pw // 2
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            # Map negative indices to positive array indices
            # fftshift moves index 0 to N/2.
            # So index for Gx is (gx + N/2) % N ?
            # Actually, fftshift output has index 0 corresponding to -N/2 (for even N).
            # Let's just extract the central block.
            # The array fft_coeffs has shape (N, N).
            # Index i corresponds to Gx = i - N//2 (if i < N//2) or i - N (if i >= N//2)?
            # No, fftshift(A) moves the zero-frequency component to the center.
            # For N=32, index 16 is G=0. Index 15 is G=-1. Index 17 is G=1.
            # Index 0 is G=-16.
            # So G = i - N//2 works for i in [0, N//2] ?
            # If i=0, G=-16. If i=16, G=0. If i=31, G=15.
            # We need G in [-half, half].
            # So we can just take the slice [N//2 - half : N//2 + half + 1]
            pass
            
    # More efficient: Just slice the array
    start = size // 2 - half
    end = size // 2 + half + 1
    block = fft_coeffs[start:end, start:end]
    
    # Now block[i, j] corresponds to Gx = i - half, Gy = j - half
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            gx = i - half
            gy = j - half
            coeffs[(gx, gy)] = block[i, j]
            
    return coeffs

def _band_structure(cell, size, n_pw=17, n_bands=6):
    """Compute TM band structure at a few key points."""
    coeffs = _compute_fourier_coeffs(cell, size, n_pw)
    
    # Key k-points in the irreducible BZ
    k_points = [
        (0, 0),      # Gamma
        (1, 0),      # X
        (0, 1),      # Y  
        (0.5, 0.5),  # M
        (0.3, 0.1),  # interior
        (0.1, 0.3),  # interior
    ]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        # Build the plane-wave matrix
        Gx = np.arange(-n_pw//2, n_pw//2 + 1)
        Gy = np.arange(-n_pw//2, n_pw//2 + 1)
        GX, GY = np.meshgrid(Gx, Gy, indexing='ij')
        GX_flat = GX.flatten()
        GY_flat = GY.flatten()
        N = len(GX_flat)
        
        # For TM: H field, equation involves 1/eps
        # (G+k)^2 * A = omega^2 * sum_G' (1/eps)_{G-G'} A_{G'}
        # In 2D: ((Gx+kx)^2 + (Gy+ky)^2) A = omega^2 * C @ A
        
        diag = ((GX_flat + kx)**2 + (GY_flat + ky)**2)
        
        # Build coupling matrix C where C[i,j] = (1/eps)_{G_i - G_j}
        C = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                dgx = int(GX_flat[i] - GX_flat[j])
                dgy = int(GY_flat[i] - GY_flat[j])
                C[i, j] = coeffs.get((dgx, dgy), 0.0)
        
        # Solve eigenvalue problem: D @ A = omega^2 * C @ A
        # => D @ C^{-1} @ A = omega^2 * A
        try:
            C_inv = np.linalg.inv(C)
            D_C_inv = np.diag(diag) @ C_inv
            # Make it Hermitian
            D_C_inv = (D_C_inv + D_C_inv.conj().T) / 2
            eigvals = np.linalg.eigvalsh(D_C_inv)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= n_bands:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf'):
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap <= 0:
        return 0.0
    return gap / midgap

def _symmetrize(cell, size):
    """Enforce square symmetry: cell[i][j] = cell[size-1-i][size-1-j] = cell[j][i] = cell[size-1-j][size-1-i]"""
    sym_cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # All four symmetric positions must agree
            positions = [
                (i, j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, size-1-i)
            ]
            # Use majority vote or just take the first
            val = cell[i][j]
            for pi, pj in positions:
                sym_cell[pi][pj] = val
    return sym_cell

def design(problem):
    """Local search to maximize TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    deadline = time.time() + 100  # 100 second budget
    
    # Start with a rod design
    best_cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        best_cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    
    best_score = _band_structure(best_cell, size, n_pw=13, n_bands=6)
    
    # Local search: try flipping pixels
    cell = [row[:] for row in best_cell]
    
    iteration = 0
    # Use a tighter deadline to account for overhead
    search_deadline = deadline - 10 
    
    while time.time() < search_deadline:
        iteration += 1
        improved = False
        
        # Try flipping a random pixel
        # Reduce number of trials per iteration to allow more iterations
        for _ in range(20):
            if time.time() > search_deadline:
                break
            
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            
            # Get symmetric positions
            positions = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
            
            # Try flipping all symmetric positions
            old_vals = [cell[pi][pj] for pi, pj in positions]
            new_val = 1 - cell[i][j]
            for pi, pj in positions:
                cell[pi][pj] = new_val
            
            score = _band_structure(cell, size, n_pw=13, n_bands=6)
            
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                # Revert
                for (pi, pj), ov in zip(positions, old_vals):
                    cell[pi][pj] = ov
        
        # If no improvement in this round, try a different strategy
        if not improved and iteration % 10 == 0:
            # Try larger radius rod
            for r in [0.15, 0.25, 0.3, 0.35]:
                if time.time() > search_deadline:
                    break
                test_cell = []
                for i in range(size):
                    x = (i + 0.5) / size - 0.5
                    test_cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < r ** 2 else 0 for j in range(size)])
                score = _band_structure(test_cell, size, n_pw=13, n_bands=6)
                if score > best_score:
                    best_score = score
                    best_cell = [row[:] for row in test_cell]
    
    return best_cell
# EVOLVE-BLOCK-END
