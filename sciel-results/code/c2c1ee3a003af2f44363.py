import numpy as np
import time
import scipy.linalg as sla

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell maximizing the TM band gap 
    between the 4th and 5th bands.
    
    Strategy:
    1. Use a surrogate solver with 17x17 plane waves (289 PWs) to closely match the evaluator.
    2. Use a parameterized shape family: R(theta) = R0 + a4*cos(4*theta) + a8*cos(8*theta) + a12*cos(12*theta)
    3. Perform a coarse grid search followed by local hill-climbing refinement.
    4. Strict time management to stay within the 230s budget.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    timeout = 220.0  # Safety margin before 230s kill
    
    # Pre-compute grid coordinates
    # Coordinates centered at 0.5, 0.5 for the 32x32 grid
    # x, y in [0, 1)
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Distance from center (0.5, 0.5)
    dx = X - 0.5
    dy = Y - 0.5
    R = np.sqrt(dx**2 + dy**2)
    Theta = np.arctan2(dy, dx)
    
    # Surrogate Solver Setup
    # We use 17x17 plane waves (289 PWs) to match the evaluator closely.
    N_pw = 17
    # G vectors for square lattice
    G = []
    half = N_pw // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            G.append((i, j))
    G = np.array(G)
    N_G = len(G)
    
    # Key K-points in the irreducible BZ
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    # And some interior points to catch the band edges which might not be at high symmetry
    # for 4th/5th bands.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.4),
        (0.4, 0.1),
        (0.3, 0.3),
        (0.1, 0.1),
        (0.4, 0.4),
    ]
    
    # Pre-compute FFT indices for G vectors
    # The FFT of a 32x32 grid gives frequencies -16..15.
    # With fftshift, index i corresponds to freq i-16.
    # So freq f is at index f+16.
    
    def _make_cell_from_params(R0, a4, a8, a12):
        """
        Generates a 32x32 mask from shape parameters.
        R(theta) = R0 + a4*cos(4*theta) + a8*cos(8*theta) + a12*cos(12*theta)
        """
        R_target = R0 + a4 * np.cos(4 * Theta) + a8 * np.cos(8 * Theta) + a12 * np.cos(12 * Theta)
        mask = (R < R_target).astype(int)
        
        # Enforce 4-fold symmetry and reflections
        # The formula is inherently 4-fold symmetric, but pixelization might break exact symmetry.
        # Enforce: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
        
        # Reflection X: mask[i, j] == mask[i, size-1-j]
        # Reflection Y: mask[i, j] == mask[size-1-i, j]
        # Transpose: mask[i, j] == mask[j, i]
        
        # Since we generated from a radial function with 4-fold symmetry, 
        # it should satisfy these. But let's enforce to be safe.
        # We'll take the majority vote or just ensure consistency.
        
        # Actually, if the continuous function is symmetric, the pixel centers 
        # are symmetric, so the mask should be symmetric.
        
        return mask
    
    def _surrogate_gap(mask):
        """
        Calculates the gap ratio using the surrogate solver.
        mask: 32x32 array of 0/1.
        Returns: gap_ratio
        """
        # Compute Fourier coefficients of 1/epsilon
        # 1/eps(r) = 1/eps_low * (1 - mask) + 1/eps_high * mask
        # Let's compute the FT of the binary mask and 1/eps mask.
        
        # Compute 1/eps mask
        inv_eps_mask = np.zeros_like(mask, dtype=float)
        inv_eps_mask[mask == 1] = 1.0 / eps_high
        inv_eps_mask[mask == 0] = 1.0 / eps_low
        
        fft_inv_eps = np.fft.fft2(inv_eps_mask)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        
        # Normalize FFT to get Fourier coefficients
        # c_n = (1/N^2) * FFT_n
        
        # Build M
        M = np.zeros((N_G, N_G), dtype=complex)
        for idx1, G1 in enumerate(G):
            for idx2, G2 in enumerate(G):
                dG = G1 - G2
                # Get coefficient c_{dG}
                # dG is (dx, dy).
                # FFT index: 16 + dx, 16 + dy
                # Note: fftshift moves 0 to center.
                # Original FFT index for freq f is (f + 16) % 32.
                # With fftshift, index i corresponds to freq i-16.
                # So freq f is at index f+16.
                # Since dG is small (-8..8), index is 8..24.
                c = fft_inv_eps[16 + int(dG[0]), 16 + int(dG[1])] / (size * size)
                M[idx1, idx2] = c
        
        # Now solve for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            kx, ky = k
            # Build D matrix (diagonal)
            D_vals = np.array([ (kx + gx)**2 + (ky + gy)**2 for gx, gy in G ])
            
            # Solve generalized eigenvalue problem:
            # D a = omega^2 M a
            # This is a symmetric generalized eigenvalue problem if D and M are symmetric.
            # We can use scipy.linalg.eigh(D, M).
            
            try:
                eigvals, eigvecs = sla.eigh(D_vals, M)
                
                # eigvals are lambda = omega^2
                # Sort them
                eigvals = np.sort(eigvals)
                
                # We need the 4th and 5th bands.
                # Note: The bands are indexed 1-based in the problem description?
                # "fourth and fifth TM bands".
                # Usually, band 1 is the lowest.
                # So we take eigvals[3] and eigvals[4] (0-indexed).
                
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    
                    # Ensure positive
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = np.sqrt(w4_sq)
                        w5 = np.sqrt(w5_sq)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
        
        if max_omega_4 == 0.0 or min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    best_score = -1.0
    best_mask = None
    best_params = None
    
    # Coarse grid search parameters
    # R0: base radius
    # a4, a8, a12: harmonic amplitudes
    
    R0_range = np.linspace(0.15, 0.35, 8)
    a4_range = np.linspace(-0.12, 0.12, 7)
    a8_range = np.linspace(-0.08, 0.08, 5)
    a12_range = np.linspace(-0.05, 0.05, 4)
    
    # Initial guess: The baseline circle
    # Let's start with a few points
    
    for R0 in R0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                for a12 in a12_range:
                    if time.time() - start_time > timeout - 30:
                        break
                        
                    # Generate mask
                    mask = _make_cell_from_params(R0, a4, a8, a12)
                    
                    score = _surrogate_gap(mask)
                    
                    if score > best_score:
                        best_score = score
                        best_mask = mask.copy()
                        best_params = (R0, a4, a8, a12)
                        
                    if time.time() - start_time > timeout - 30:
                        break
                if time.time() - start_time > timeout - 30:
                    break
            if time.time() - start_time > timeout - 30:
                break
        if time.time() - start_time > timeout - 30:
            break
    
    # Local hill-climbing refinement
    if best_params is not None:
        current_params = list(best_params)
        current_score = best_score
        current_mask = best_mask.copy()
        
        # Step sizes for local search
        step = [0.02, 0.02, 0.01, 0.01]
        
        # Try to improve
        for iteration in range(50):
            if time.time() - start_time > timeout - 10:
                break
                
            improved = False
            for param_idx in range(4):
                for sign in [-1, 1]:
                    new_params = list(current_params)
                    new_params[param_idx] += sign * step[param_idx]
                    
                    # Keep parameters in reasonable bounds
                    if new_params[0] < 0.1 or new_params[0] > 0.4:
                        continue
                    if abs(new_params[1]) > 0.15:
                        continue
                    if abs(new_params[2]) > 0.1:
                        continue
                    if abs(new_params[3]) > 0.06:
                        continue
                    
                    new_mask = _make_cell_from_params(new_params[0], new_params[1], new_params[2], new_params[3])
                    new_score = _surrogate_gap(new_mask)
                    
                    if new_score > current_score:
                        current_score = new_score
                        current_mask = new_mask.copy()
                        current_params = new_params
                        improved = True
                        # Update best if this is also the global best
                        if new_score > best_score:
                            best_score = new_score
                            best_mask = new_mask.copy()
                            best_params = tuple(new_params)
                
                if time.time() - start_time > timeout - 10:
                    break
            
            if not improved:
                # No improvement, reduce step size
                step = [s * 0.7 for s in step]
                if step[0] < 0.001:
                    break
    
    # If no valid mask found, return baseline
    if best_mask is None:
        best_mask = (R < 0.2).astype(int)
    
    # Convert to list of lists
    return best_mask.tolist()
