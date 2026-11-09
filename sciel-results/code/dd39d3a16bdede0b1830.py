import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """
    Optimizes discrete Fourier coefficients of a C4v-symmetric dielectric map
    to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- 1. Define C4v symmetric Fourier basis ---
    # We represent epsilon(x,y) on the grid [0, size-1] x [0, size-1].
    # Symmetry: cell[i][j] == cell[size-1-i][j] (x-reflection), 
    #           cell[i][j] == cell[i][size-1-j] (y-reflection),
    #           cell[i][j] == cell[j][i] (transpose).
    # This implies symmetry about x=0.5*size and y=0.5*size and y=x.
    
    # Use a basis of cosines centered at (size/2, size/2).
    # f(x,y) = sum c_{mn} cos(pi*m*(x - size/2 + 0.5)/size) cos(pi*n*(y - size/2 + 0.5)/size)
    # Actually, simpler: use DCT-II like basis on the grid indices.
    # Let's define basis vectors on the 32x32 grid.
    
    x = np.arange(size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # Center coordinates
    xc = X - (size - 1) / 2.0
    yc = Y - (size - 1) / 2.0
    
    # Basis functions: cos(pi * m * xc / size) * cos(pi * n * yc / size)
    # To ensure C4v symmetry:
    # 1. Reflection x -> -x: cos is even, so any m works.
    # 2. Reflection y -> -y: cos is even, so any n works.
    # 3. Transpose (x,y) -> (y,x): requires c_{mn} = c_{nm}.
    # So we only need to optimize c_{mn} for m <= n, and set c_{nm} = c_{mn}.
    
    max_harmonic = 6  # Go higher for more detail
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    
    num_params = len(params_idx)
    
    # Precompute basis functions
    # basis[k] is a 32x32 array
    basis_funcs = np.zeros((num_params, size, size))
    for idx, (m, n) in enumerate(params_idx):
        cm = np.cos(np.pi * m * xc / size)
        cn = np.cos(np.pi * n * yc / size)
        basis_funcs[idx] = cm * cn
    
    # --- 2. Plane Wave Solver ---
    # Use a moderate number of plane waves for speed and accuracy.
    # The evaluator uses 289 (17x17). We will use 121 (11x11) for optimization speed,
    # and maybe 289 for final check if time permits.
    N_pw = 11
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute G-G' differences
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    
    # K points: Irreducible BZ
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5) are high symmetry.
    # Path: Gamma -> M -> X -> Gamma? No, square BZ.
    # Irreducible triangle: Gamma(0,0), M(0.5,0), X(0.5,0.5).
    # Points:
    # 1. Along Gamma-M: (t, 0) for t in [0, 0.5]
    # 2. Along M-X: (0.5, t) for t in [0, 0.5]
    # 3. Along Gamma-X: (t, t) for t in [0, 0.5]
    
    k_points = []
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    def get_eps_grid(params):
        """
        Reconstructs the epsilon grid from Fourier parameters.
        params: array of length num_params
        Returns: 32x32 array of epsilon values.
        """
        # field = sum params[i] * basis_funcs[i]
        field = np.tensordot(params, basis_funcs, axes=1)
        # Map field to [eps_low, eps_high]
        # We want a binary-like output eventually, but for optimization,
        # we can optimize a continuous field and then binarize.
        # Or, we can optimize the binary mask directly? 
        # Binary optimization is hard. Let's optimize a continuous field 
        # with a strong sigmoid to push it towards binary, or just 
        # optimize the continuous field and binarize at the end.
        # The evaluator takes a binary 0/1 cell.
        # So the "design" is the binary cell.
        # We will optimize continuous params, compute the best binary cell,
        # and evaluate that binary cell's gap.
        
        # Normalize field to [0, 1]
        # We need to handle the scaling.
        # Let's assume the params are scaled such that field is in [-1, 1] roughly.
        # We'll use a sigmoid to create a probability, then threshold?
        # No, let's just use the field directly to determine the binary state.
        # We want to find the optimal binary pattern.
        # Strategy: 
        # 1. Optimize continuous field f.
        # 2. Binarize: eps = eps_high if f > 0 else eps_low.
        # 3. Evaluate gap of this binary eps.
        # This is a non-differentiable objective.
        # Alternative: Optimize the continuous field, and use the continuous
        # field in the PW solver? No, the task requires a binary 0/1 cell.
        # So the search space is discrete.
        
        # Let's stick to the previous approach: 
        # Optimize continuous params -> get continuous field -> binarize -> evaluate.
        
        # To make the binarization effective, we should ensure the field
        # has a clear separation.
        
        # For now, return the continuous field, and we will binarize in the evaluator.
        return field

    def evaluate_binary_cell(binary_cell):
        """
        binary_cell: 32x32 array of 0/1.
        Returns the gap-midgap ratio.
        """
        # Construct epsilon grid
        eps_grid = np.where(binary_cell == 1, eps_high, eps_low)
        
        # Compute FFT of 1/epsilon
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Build coefficient matrix for PW method
        # coeff[i,j] = fft_inv_eps[(G_i - G_j) mod size]
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Hermitian matrix for TM modes:
            # H[i,j] = (k+G_i)^2 * sum_{G''} epsilon_{G''} delta_{G_i+G'', G_j}
            # Wait, the standard PW formulation for TM is:
            # sum_j [ (k+G_i)^2 / epsilon_{G_i-G_j} ] c_j = omega^2 c_i
            # This is not Hermitian.
            # However, we can transform it to a Hermitian form by defining d_i = c_i / sqrt(epsilon_{0,0})? No.
            # The matrix M_{ij} = (k+G_i)^2 * (1/epsilon_{G_i-G_j}) is not Hermitian.
            # But the eigenvalues are real.
            # Using np.linalg.eigvals is fine.
            
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            # Ensure Hermitian part for stability? 
            # M is not Hermitian because 1/eps is real but the indices are shifted.
            # Actually, 1/eps_{G} is real. 
            # M_{ij} = omega_sq_i * inv_eps_{G_i - G_j}.
            # M_{ji} = omega_sq_j * inv_eps_{G_j - G_i}.
            # Since inv_eps is real and even (if eps is real and symmetric), inv_eps_{-G} = inv_eps_G.
            # So M_{ji} = omega_sq_j * inv_eps_{G_i - G_j}.
            # M is not symmetric unless omega_sq_i = omega_sq_j.
            # So we must use general eigvals.
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Ensure positive eigenvalues
                real_eigvals = np.maximum(real_eigvals, 0)
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

    def objective(params):
        """
        Objective function for minimization (negative gap).
        """
        field = get_eps_grid(params)
        # Binarize: threshold at 0
        binary_cell = (field > 0).astype(int)
        
        # Enforce symmetry on the binary cell
        # The basis is symmetric, but binarization might break it slightly due to noise?
        # No, if params are symmetric, field is symmetric.
        # But we need to ensure the final output is strictly symmetric.
        # The basis construction ensures field[i,j] == field[j,i] etc.
        # So binary_cell should be symmetric.
        
        score = evaluate_binary_cell(binary_cell)
        return -score  # Minimize negative score

    # --- 3. Optimization ---
    
    # Initial guess: A dielectric rod in the center.
    # This corresponds to low frequency components.
    init_params = np.zeros(num_params)
    # (0,0) term is the average. Set it to 0.
    # (1,1) term creates a bump?
    # Let's check the basis.
    # m=0, n=0: constant.
    # m=1, n=0: cos(pi*x/size). Alternating stripes.
    # m=1, n=1: cos(pi*x/size)*cos(pi*y/size). Bump in center?
    # At center (size/2), xc=0, cos(0)=1. At edges, cos(pi)=-1.
    # So (1,1) is positive in center, negative at edges.
    # We want dielectric in center. So set (1,1) positive.
    
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 1.0
    
    # Also try a hole in the center (air in center).
    # This would require (1,1) negative.
    
    best_score = -np.inf
    best_params = None
    
    # Multi-start random search + Nelder-Mead
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        if r == 0:
            p0 = init_params.copy()
        else:
            p0 = np.random.normal(0, 0.5, num_params)
            # Bias towards rod-like or hole-like
            if r % 2 == 0:
                p0[idx_11] = abs(p0[idx_11]) + 0.5
            else:
                p0[idx_11] = -abs(p0[idx_11]) - 0.5
        
        # Use Nelder-Mead for local optimization
        # Limit function evaluations to save time
        maxfev = 50
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead', 
                           options={'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 1e-4})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
            
        # Also do a few random steps around the best found so far
        if best_params is not None:
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                p_try = best_params + np.random.normal(0, 0.2, num_params)
                score_try = -objective(p_try)
                if score_try > best_score:
                    best_score = score_try
                    best_params = p_try.copy()

    # --- 4. Finalize ---
    
    if best_params is None:
        # Fallback: simple rod
        binary_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                xc = (i - (size-1)/2.0) / size
                yc = (j - (size-1)/2.0) / size
                if xc*xc + yc*yc < 0.2:
                    binary_cell[i, j] = 1
        best_params = np.zeros(num_params) # Placeholder
    
    # Reconstruct final field and binarize
    final_field = get_eps_grid(best_params)
    final_binary = (final_field > 0).astype(int)
    
    # Enforce strict symmetry on the final binary cell
    # The basis should guarantee it, but let's be safe.
    sym_cell = np.zeros((size, size), dtype=int)
    counts = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            val = final_binary[i, j]
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            for (pi, pj) in positions:
                sym_cell[pi, pj] += val
                counts[pi, pj] += 1
                
    final_cell_list = []
    for i in range(size):
        row = []
        for j in range(size):
            if counts[i, j] > 0:
                avg_val = sym_cell[i, j] / counts[i, j]
                row.append(1 if avg_val >= 0.5 else 0)
            else:
                row.append(0)
        final_cell_list.append(row)
        
    return final_cell_list
