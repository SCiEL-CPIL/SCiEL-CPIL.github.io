import numpy as np
import time

def design(problem):
    start_time = time.time()
    size = problem['pixels'] # 32
    eps_low = problem['eps_low'] # 1
    eps_high = problem['eps_high'] # 12
    lower_band = problem['lower_band'] # 4
    
    # Define independent pixels
    # i in 0..15, j in i..31-i
    indep_indices = []
    for i in range(16):
        for j in range(i, 32 - i):
            indep_indices.append((i, j))
            
    # Initial design: Circle radius 0.25
    # Center is 16, 16 (indices 0..31, center is 15.5)
    # Pixel (i,j) center is (i+0.5, j+0.5)
    # Distance from center (16, 16)
    # r = sqrt((i+0.5-16)^2 + (j+0.5-16)^2)
    # If r < 0.25 * 32 = 8.0, then 1
    
    current_indep = [0] * len(indep_indices)
    for idx, (i, j) in enumerate(indep_indices):
        dx = i + 0.5 - 16.0
        dy = j + 0.5 - 16.0
        if dx*dx + dy*dy < 8.0**2:
            current_indep[idx] = 1
            
    def expand_cell(indep_vals):
        cell = np.zeros((size, size), dtype=np.uint8)
        for idx, val in enumerate(indep_vals):
            i, j = indep_indices[idx]
            cell[i, j] = val
            # Symmetries:
            # (N-1-i, j)
            cell[size-1-i, j] = val
            # (i, N-1-j)
            cell[i, size-1-j] = val
            # (N-1-i, N-1-j)
            cell[size-1-i, size-1-j] = val
            # (j, i)
            cell[j, i] = val
            # (N-1-j, i)
            cell[size-1-j, i] = val
            # (j, N-1-i)
            cell[j, size-1-i] = val
            # (N-1-j, N-1-i)
            cell[size-1-j, size-1-i] = val
        return cell

    def compute_score(cell, n_pw=11, n_kpts=8):
        # cell is 32x32 numpy array
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        # FFT
        fft_inv = np.fft.fft2(inv_eps)
        
        # Plane wave indices
        # n_pw is odd, so range is -n_pw//2 to n_pw//2
        half = n_pw // 2
        g_vals = np.arange(-half, half + 1)
        gx, gy = np.meshgrid(g_vals, g_vals, indexing='ij')
        g_vecs = np.stack([gx.flatten(), gy.flatten()], axis=1)
        n_pw_actual = len(g_vecs)
        
        # Precompute G differences?
        # We need fft_inv at (G_alpha - G_beta)
        # The FFT array fft_inv has indices 0..N-1.
        # Frequency index k in FFT corresponds to k if k < N/2, else k-N.
        # We need to map G_diff to FFT index.
        
        # K-points
        # Irreducible BZ: (0,0) to (1,0) to (1,1) to (0,0)
        # Actually, just sample along the boundary.
        k_points = []
        # Gamma to X
        for t in np.linspace(0, 1, n_kpts//2):
            k_points.append((t, 0))
        # X to M
        for t in np.linspace(0, 1, n_kpts//2 - 1):
            k_points.append((1, t))
        
        max_omega_4 = 0
        min_omega_5 = np.inf
        
        for k in k_points:
            k_vec = np.array(k)
            # Build matrix
            # M[alpha, beta] = fft_inv[G_alpha - G_beta] * (k + G_beta)^2
            # Note: (k + G_beta)^2 is scalar for TM?
            # For TM, the operator is epsilon^-1 * (k+G)^2.
            # Yes, scalar.
            
            matrix = np.zeros((n_pw_actual, n_pw_actual), dtype=np.complex128)
            
            for alpha in range(n_pw_actual):
                ga = g_vecs[alpha]
                for beta in range(n_pw_actual):
                    gb = g_vecs[beta]
                    g_diff = ga - gb
                    # Map g_diff to FFT index
                    # g_diff components are integers in [-n_pw, n_pw]
                    # FFT index: if val >= 0, idx = val, else idx = N + val
                    idx_x = g_diff[0] % size
                    idx_y = g_diff[1] % size
                    coeff = fft_inv[idx_x, idx_y]
                    
                    k_plus_g = k_vec + gb
                    val = coeff * (k_plus_g[0]**2 + k_plus_g[1]**2)
                    matrix[alpha, beta] = val
            
            # Solve eigenvalues
            # Matrix is Hermitian
            eigs = np.linalg.eigvalsh(matrix)
            # eigs are omega^2
            # We need 4th and 5th bands (indices 3 and 4)
            if len(eigs) > 4:
                w4 = np.sqrt(eigs[3])
                w5 = np.sqrt(eigs[4])
                if w4 > max_omega_4: max_omega_4 = w4
                if w5 < min_omega_5: min_omega_5 = w5
                
        if min_omega_5 == np.inf: return -1
        gap = min_omega_5 - max_omega_4
        if gap < 0: return -1
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    # Initial score
    best_cell = expand_cell(current_indep)
    best_score = compute_score(best_cell)
    
    # Hill Climbing
    while time.time() - start_time < 110:
        # Pick random index
        idx = np.random.randint(len(indep_indices))
        old_val = current_indep[idx]
        current_indep[idx] = 1 - old_val
        
        new_cell = expand_cell(current_indep)
        new_score = compute_score(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell
        else:
            current_indep[idx] = old_val
            
    # Final check with higher precision if time allows
    if time.time() - start_time < 115:
        final_score = compute_score(best_cell, n_pw=17, n_kpts=30)
        if final_score > best_score:
             # It might be better, but we return the cell, not the score
             pass
             
    # Convert best_cell to list of lists
    return best_cell.tolist()
