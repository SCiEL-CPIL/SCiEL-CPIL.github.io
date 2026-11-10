import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 215.0
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Parameterization via Fourier coefficients compatible with C4v symmetry about center.
    # Symmetry requirements:
    # 1. f(x,y) = f(1-x, y)
    # 2. f(x,y) = f(x, 1-y)
    # 3. f(x,y) = f(y,x)
    # Using basis: C_m(x) = cos(2*pi*m*(x-0.5))
    # f(x,y) = sum_{m,n} A_{mn} C_m(x) C_n(y)
    # Symmetry 1 & 2 satisfied by cosine basis.
    # Symmetry 3 requires A_{mn} = A_{nm}.
    
    max_h = 4  # Harmonic order up to 4
    params_idx = []
    for m in range(max_h + 1):
        for n in range(m, max_h + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    # Precompute grid points
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    # Precompute basis functions for speed
    # Basis[m, n] = cos(2*pi*m*Xc) * cos(2*pi*n*Yc)
    # We can just compute on the fly inside get_field if needed, but precomputing vectors is faster.
    cos_terms = {}
    for m in range(max_h + 1):
        cos_terms[m] = np.cos(2 * np.pi * m * Xc)
    cos_terms_y = {}
    for n in range(max_h + 1):
        cos_terms_y[n] = np.cos(2 * np.pi * n * Yc)
        
    def get_field(params):
        # params is a flat list of length num_params
        # Reconstruct symmetric matrix A
        # A[m,n] = params[idx]
        field = np.zeros((size, size))
        for i, (m, n) in enumerate(params_idx):
            a_val = params[i]
            if a_val == 0: continue
            if m == n:
                # term is A_mm * C_m(x) * C_m(y)
                field += a_val * cos_terms[m] * cos_terms_y[m]
            else:
                # term is A_mn * (C_m(x)C_n(y) + C_n(x)C_m(y))
                # Since A_mn = A_nm
                field += a_val * (cos_terms[m] * cos_terms_y[n] + cos_terms[n] * cos_terms_y[m])
        
        # Map to epsilon
        # Use tanh to keep in range
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # 2. Fast Evaluator
    # Use 289 Plane Waves (N=8) to match evaluator closely.
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    G_indices = np.arange(num_pw)
    
    # Precompute G differences for matrix construction
    # diff_idx[i, j] = index of G_i - G_j in a lookup table?
    # Since G_i - G_j ranges from -16 to 16.
    # We can precompute the matrix of differences.
    G_diffs = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :] # shape (num_pw, num_pw, 2)
    G_diffs_flat = G_diffs.reshape(num_pw * num_pw, 2)
    
    # K-points
    # Irreducible BZ: k_x >= k_y >= 0, k_x + k_y <= 0.5
    # We sample 30 points as per task description.
    k_points = []
    # Gamma-M: (t, 0)
    # M-X: (0.5, t) with t in [0, 0.5] -> but M-X is (0.5, t) where t goes 0 to 0.5?
    # Wait, M is (0.5, 0), X is (0.5, 0.5).
    # Gamma is (0,0).
    # Path Gamma-M: (t, 0) t in [0, 0.5]
    # Path M-X: (0.5, t) t in [0, 0.5]
    # Path Gamma-X: (t, t) t in [0, 0.5]
    
    # The task says "30 wave vectors covering the boundary and interior".
    # Let's use 10 on each path.
    t_vals = np.linspace(0, 0.5, 10)
    for t in t_vals:
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    
    # Remove duplicates
    unique_k = list(set(k_points))
    
    def evaluate_cell(eps_grid):
        inv_eps = 1.0 / eps_grid
        # FFT of 1/eps
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # We need coefficients for all differences G_i - G_j
        # G_diffs_flat has shape (num_pw*num_pw, 2)
        # We need to map each difference (dx, dy) to fft_inv_eps index.
        # fft_inv_eps index for (dx, dy) is (dx % size, dy % size)
        
        # Vectorized lookup
        idx_x = G_diffs_flat[:, 0] % size
        idx_y = G_diffs_flat[:, 1] % size
        coeffs = fft_inv_eps[idx_x, idx_y] # shape (num_pw*num_pw,)
        coeff_matrix = coeffs.reshape(num_pw, num_pw)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            # (k + G)^2 for each G
            k_plus_G = G_vectors + k
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Matrix M = diag(omega_sq) @ coeff_matrix?
            # No, M_{ij} = (1/eps)_{G_i - G_j} * |k + G_i|^2
            # This is diag(omega_sq) * coeff_matrix?
            # Let's check: (diag(omega_sq) * coeff_matrix)_{ij} = omega_sq[i] * coeff_matrix[i, j]
            # Yes.
            
            M = omega_sq[:, np.newaxis] * coeff_matrix
            
            # Solve eigenvalue problem
            # M is not Hermitian generally?
            # Wait. (1/eps) is real symmetric?
            # coeff_matrix[i, j] = (1/eps)_{G_i - G_j}
            # coeff_matrix[j, i] = (1/eps)_{G_j - G_i} = (1/eps)_{-(G_i - G_j)}
            # For real eps, (1/eps)_G = (1/eps)_{-G}^*. So coeff_matrix is Hermitian?
            # (1/eps) is real, so (1/eps)_G = (1/eps)_{-G}.
            # So coeff_matrix is real symmetric?
            # coeff_matrix[i,j] = c_{G_i-G_j}. coeff_matrix[j,i] = c_{G_j-G_i} = c_{-(G_i-G_j)} = c_{G_i-G_j}.
            # Yes, coeff_matrix is symmetric.
            # But M = diag(omega_sq) * coeff_matrix is NOT symmetric unless omega_sq is constant.
            # So we must use general eigenvalues.
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            # Ensure positive
            real_eigvals = np.maximum(real_eigvals, 0)
            
            if len(real_eigvals) >= 5:
                # Bands 4 and 5 (1-indexed) -> indices 3 and 4
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # 3. Optimization
    # Use a combination of random restarts and Nelder-Mead or simple hill climbing.
    # Given the small parameter space (~15 params), we can do a reasonable search.
    
    best_params = None
    best_score = -np.inf
    
    # Initial guess: A "blob" or ring.
    # A ring has high values at intermediate radius.
    # In Fourier space, a ring is a Bessel function J0, which has peaks at certain harmonics.
    # Let's just start with random points and keep the best.
    
    num_restarts = 20
    steps_per_restart = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random start
        # Scale of parameters: tanh saturates at ~2-3. So range [-3, 3] is good.
        current_params = np.random.uniform(-2.5, 2.5, num_params)
        current_score = evaluate_cell(get_field(current_params))
        
        # Hill climbing
        step_size = 0.5
        no_improve = 0
        for s in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
                
            # Random perturbation
            # Focus on lower harmonics? Or all?
            # Let's perturb all.
            noise = np.random.normal(0, step_size, num_params)
            # Dampen higher harmonics?
            # for i in range(num_params):
            #   m,n = params_idx[i]
            #   noise[i] /= (1 + m + n)
            
            new_params = current_params + noise
            new_score = evaluate_cell(get_field(new_params))
            
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                no_improve = 0
                # Increase step size slightly?
                # step_size *= 1.05
            else:
                no_improve += 1
                step_size *= 0.9
                if no_improve > 10:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
    
    # Fallback if no params found
    if best_params is None:
        # Return a simple circle
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                xi = (i + 0.5) / size - 0.5
                yj = (j + 0.5) / size - 0.5
                if xi*xi + yj*yj < 0.15:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell

    final_grid = get_field(best_params)
    
    # Binarize
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if final_grid[i, j] > threshold:
                row.append(1)
            else:
                row.append(0)
        binary_cell.append(row)
        
    # Enforce symmetry on binary grid just in case
    # The construction should guarantee it, but binarization might flip ties.
    # Symmetry: cell[i][j] == cell[size-1-i][size-1-j] and cell[i][j] == cell[j][i]
    # We can average over symmetry group and re-threshold?
    # Or just force it.
    # Since the field is symmetric, the only issue is if field value is extremely close to threshold.
    # Let's enforce by taking majority vote or just forcing.
    # Forcing: if we break symmetry, we might drop score.
    # But invalidity is worse.
    # Let's check if it's already symmetric.
    # If not, we can symmetrize the field before binarization.
    
    # Let's re-evaluate the field with symmetrization to be safe.
    sym_grid = np.zeros_like(final_grid)
    for i in range(size):
        for j in range(size):
            # 4 symmetric points
            p1 = final_grid[i, j]
            p2 = final_grid[size-1-i, size-1-j]
            p3 = final_grid[j, i]
            p4 = final_grid[size-1-j, size-1-i]
            sym_grid[i, j] = (p1 + p2 + p3 + p4) / 4.0
            
    # Re-binarize
    for i in range(size):
        for j in range(size):
            if sym_grid[i, j] > threshold:
                binary_cell[i][j] = 1
            else:
                binary_cell[i][j] = 0
                
    return binary_cell
