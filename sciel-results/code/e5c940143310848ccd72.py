import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the unit cell as a thresholded sum of C4v-symmetric basis functions.
    2. Optimize the coefficients of these basis functions using a hybrid 
       random-restart and coordinate-descent (hill-climbing) algorithm.
    3. Use a fast plane-wave evaluator (9x9 modes) to score candidate designs.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin

    # --- 1. Define C4v-Symmetric Basis Functions ---
    # We use a set of cosine products centered at the cell center (15.5, 15.5)
    # to naturally respect the inversion symmetry, then we enforce full C4v by averaging.
    # However, for speed, we can define basis functions that are inherently C4v symmetric.
    # A function f(x,y) is C4v symmetric if f(x,y) = f(31-x, y) = f(x, 31-y) = ...
    
    # Let's generate a library of basis functions on the grid.
    # x, y in [0, 31]
    x_coords = np.arange(size)
    y_coords = np.arange(size)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Center coordinates for symmetry (half-integer)
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    
    # Generate basis functions: cos(u * (x-cx)) * cos(v * (y-cy))
    # To ensure C4v, we symmetrize each basis function.
    # Symmetrization: Average over all 8 symmetric points.
    
    def symmetrize(field):
        """
        Enforce C4v symmetry by averaging the field with its 8 symmetric counterparts.
        Points (i,j) map to:
        (i,j), (31-i, j), (i, 31-j), (31-i, 31-j),
        (j, i), (31-j, j), (j, 31-i), (31-j, 31-i)
        """
        sym_field = np.zeros_like(field)
        # We can do this efficiently by slicing and adding, but for 32x32, explicit loops are fast enough
        # or we can use np.roll/flip, but explicit indices are safer for correctness.
        # Actually, vectorized approach:
        # 1. Identity
        sym_field += field
        # 2. Flip X
        sym_field += field[::-1, :]
        # 3. Flip Y
        sym_field += field[:, ::-1]
        # 4. Flip X and Y
        sym_field += field[::-1, ::-1]
        # 5. Transpose
        sym_field += field.T
        # 6. Transpose + Flip X (which is Flip Y of Transpose)
        # (j, i) -> (31-j, i) is Flip X of Transpose?
        # Transpose: (i,j) -> (j,i). Flip X of Transpose: (31-j, i). Correct.
        sym_field += field.T[::-1, :]
        # 7. Transpose + Flip Y
        # (j, i) -> (j, 31-i). Flip Y of Transpose.
        sym_field += field.T[:, ::-1]
        # 8. Transpose + Flip X + Flip Y
        # (j, i) -> (31-j, 31-i). Flip X and Y of Transpose.
        sym_field += field.T[::-1, ::-1]
        
        sym_field /= 8.0
        return sym_field

    # Create a set of basis functions
    # We pick frequencies (u, v) that are integers.
    # Low frequencies capture large features, high frequencies capture details.
    freqs = []
    for u in range(1, 5):
        for v in range(1, 5):
            freqs.append((u, v))
    # Add some mixed frequencies
    for u in [2, 3, 4]:
        for v in [1, 2, 3]:
            if (u, v) not in freqs:
                freqs.append((u, v))
                
    num_basis = len(freqs)
    
    # Precompute the symmetrized basis functions
    basis_funcs = np.zeros((num_basis, size, size))
    for idx, (u, v) in enumerate(freqs):
        # Use phase shift to center the waves
        # cos(2*pi*u*(x-cx)/size)
        # Note: using /size ensures integer periods over the domain length
        bx = np.cos(2 * np.pi * u * (X - cx) / size)
        by = np.cos(2 * np.pi * v * (Y - cy) / size)
        raw_func = bx * by
        basis_funcs[idx] = symmetrize(raw_func)
        
    # Normalize basis functions to unit norm for stable optimization
    for idx in range(num_basis):
        norm = np.linalg.norm(basis_funcs[idx])
        if norm > 1e-6:
            basis_funcs[idx] /= norm

    # --- 2. Fast Evaluator ---
    # Plane Wave Method
    pw_half = 4  # 9x9 = 81 modes
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences indices for FFT lookup
    # We need coeffs[G_i - G_j]
    # G_diffs[i, j] = G_i - G_j
    # We can compute this on the fly or precompute indices.
    # For 81 modes, 81x81 = 6561 diffs. Precomputing indices is fast.
    
    # Create a 2D array of G vectors for broadcasting
    Gx_2d = Gx_arr[:, np.newaxis]
    Gy_2d = Gy_arr[:, np.newaxis]
    
    # Differences
    dGx = Gx_2d - Gx_2d.T
    dGy = Gy_2d - Gy_2d.T
    
    # Map to FFT indices (0 to size-1)
    # FFT of 1/eps gives coefficients for G = (m, n) where m, n are integers.
    # In numpy fft2, index [i, j] corresponds to frequency (i, j) if i < size/2, etc.
    # Specifically, fft2(A)[i, j] corresponds to exp(-2*pi*i*x/N) exp(-2*pi*j*y/N)
    # So the frequency is (i, j) or (i-size, j) etc.
    # We need the coefficient for G_diff = (dx, dy).
    # The index in the FFT array is (dx % size, dy % size).
    
    idx_dGx = dGx % size
    idx_dGy = dGy % size
    
    # K-points in irreducible BZ
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Path: Gamma -> M -> X
    k_points = []
    # Gamma to M (0, 0) to (0.5, 0)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0.0))
    # M to X (0.5, 0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5, t))
    # Remove duplicates (M is shared)
    k_points = list(set(k_points))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1
        """
        # Build inverse epsilon grid
        inv_eps = np.where(cell_grid == 1, 1.0 / eps_high, 1.0 / eps_low)
        
        # FFT for Fourier coefficients
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # Extract coefficients for all G_diffs
        # coeffs[i, j] = fft_inv_eps[idx_dGx[i, j], idx_dGy[i, j]]
        coeffs = fft_inv_eps[idx_dGx, idx_dGy] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i . K_j) * coeffs_{G_i - G_j}
            # K_i . K_j = Kx_i Kx_j + Ky_i Ky_j
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Ensure symmetry (should be real symmetric, but numerical errors might creep in)
            H_real = (H_real + H_real.T) / 2.0
            
            # Get eigenvalues
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Optimization ---
    
    def coeffs_to_cell(coeffs_vec):
        """
        Convert coefficient vector to binary cell.
        """
        # Field = sum c_i * basis_i
        # coeffs_vec shape (N, 1, 1) or (N,)
        # basis_funcs shape (N, size, size)
        field = np.tensordot(coeffs_vec, basis_funcs, axes=([0], [0]))
        
        # Threshold at the mean of the field to maintain roughly 50% fill factor
        # Or a fixed threshold? Mean is adaptive.
        threshold = np.mean(field)
        cell = (field > threshold).astype(int)
        return cell

    # Initial best
    best_score = -1.0
    best_cell = None
    
    # Random restarts
    num_restarts = 20
    steps_per_restart = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initial coefficients
        # Scale coefficients to have unit variance
        c = np.random.normal(0, 1, num_basis)
        
        # Local Optimization (Coordinate Descent)
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            current_cell = coeffs_to_cell(c)
            current_score = evaluate_gap(current_cell)
            
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
                
            # Try to improve by perturbing one coefficient at a time
            improved = False
            # Randomly shuffle order of coefficients to avoid bias
            order = np.random.permutation(num_basis)
            
            for idx in order:
                if time.time() > deadline:
                    break
                    
                # Try increasing and decreasing
                delta = 0.1
                # Adaptive step size? Keep it simple for now.
                
                c_old = c[idx]
                
                # Try +
                c[idx] = c_old + delta
                cell_plus = coeffs_to_cell(c)
                score_plus = evaluate_gap(cell_plus)
                
                # Try -
                c[idx] = c_old - delta
                cell_minus = coeffs_to_cell(c)
                score_minus = evaluate_gap(cell_minus)
                
                # Restore
                c[idx] = c_old
                
                # Choose best
                if score_plus > current_score:
                    c[idx] = c_old + delta
                    current_score = score_plus
                    improved = True
                elif score_minus > current_score:
                    c[idx] = c_old - delta
                    current_score = score_minus
                    improved = True
                else:
                    # If no improvement, try smaller step?
                    # For simplicity, we just move to next coefficient.
                    pass
                    
            if not improved:
                # Stagnation: perturb the whole vector slightly to escape local minima
                c += np.random.normal(0, 0.2, num_basis)

    # --- 4. Finalize ---
    if best_cell is None:
        # Fallback: Return a simple disk configuration if optimization failed
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        radius = 4.0
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        cell[i, j] = 1
        best_cell = cell

    # Ensure exact symmetry on the final result (in case thresholding broke it slightly due to floating point)
    # Although coeffs_to_cell uses symmetrized basis, the thresholding might be sensitive.
    # Re-symmetrize the final binary cell.
    def enforce_binary_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    final_cell = enforce_binary_symmetry(best_cell)
    return final_cell.tolist()

