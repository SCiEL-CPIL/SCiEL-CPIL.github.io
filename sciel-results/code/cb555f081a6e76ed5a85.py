"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap."""
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    time_budget = 100.0  # Leave 20s margin for 120s limit

    def is_symmetric(cell):
        cell = np.array(cell)
        # Check all D4 symmetries
        if not np.array_equal(cell, cell[::-1, :]): return False
        if not np.array_equal(cell, cell[:, ::-1]): return False
        if not np.array_equal(cell, cell.T): return False
        if not np.array_equal(cell, cell[::-1, ::-1]): return False
        return True

    def compute_gap(cell, n_pw=15, k_points=None):
        """
        Compute the gap between band 4 and band 5.
        Returns (max_omega_4, min_omega_5).
        """
        cell = np.array(cell)
        N = len(cell)
        
        # Calculate Fourier coefficients of epsilon
        # G vectors: -M to M in steps of 1, where M = (n_pw - 1) // 2
        M = (n_pw - 1) // 2
        G_vals = np.arange(-M, M + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.vstack([Gx.ravel(), Gy.ravel()])
        
        # Positions of pixels
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        # Pixel centers
        x = (i + 0.5) / N - 0.5
        y = (j + 0.5) / N - 0.5
        r = np.vstack([x.ravel(), y.ravel()])
        
        # Epsilon values
        eps_vals = np.where(cell.ravel() == 1, eps_high, eps_low)
        
        # Fourier coefficients: eps_G = 1/N^2 sum eps(r) exp(-i G dot r)
        # Phase matrix: G dot r
        phase = np.exp(-1j * (G_vectors[0, :] @ r[0, :] + G_vectors[1, :] @ r[1, :]))
        eps_G = np.sum(eps_vals * phase, axis=1) / (N * N)
        
        # Map G index to value
        G_dict = {}
        for idx in range(len(G_vals)):
            G_dict[(G_vectors[0, idx], G_vectors[1, idx])] = idx
        
        # Define k points
        if k_points is None:
            # Irreducible BZ points
            # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
            # And some intermediate points
            k_points = [
                (0.0, 0.0),
                (0.25, 0.0),
                (0.5, 0.0),
                (0.5, 0.25),
                (0.5, 0.5),
                (0.25, 0.25),
                (0.1, 0.1),
                (0.4, 0.1),
                (0.4, 0.4)
            ]
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kx, ky in k_points:
            # Build Hamiltonian matrix
            # H_{G', G} = ( (k+G') . (k+G) ) / eps_{G-G'}
            # Note: For TM, the dot product is scalar in 2D.
            
            # k + G vectors
            kx_G = kx + G_vectors[0, :]
            ky_G = ky + G_vectors[1, :]
            kG_sq = kx_G**2 + ky_G**2
            
            H = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
            
            for i_idx in range(n_pw*n_pw):
                Gi = G_vectors[:, i_idx]
                for j_idx in range(n_pw*n_pw):
                    Gj = G_vectors[:, j_idx]
                    G_diff = Gi - Gj
                    # Find eps_G_diff
                    # If G_diff is not in the basis, we need to handle it.
                    # Usually, we assume the basis is large enough to contain all differences.
                    # With M=7 (15x15), max diff is 14.
                    # If G_diff is outside, eps_G_diff is approx 0 or mean?
                    # For a finite basis, we should project.
                    # However, 15x15 is small. Let's check if G_diff is in G_dict.
                    if tuple(G_diff) in G_dict:
                        eps_diff = eps_G[G_dict[tuple(G_diff)]]
                    else:
                        # If not in basis, we can't compute exactly.
                        # For high G, eps_G is small. Approximate as 0?
                        # Or use the average epsilon?
                        # Better: ensure basis is large enough.
                        # 15x15 basis has G range -7 to 7.
                        # Diff range -14 to 14.
                        # This is a problem. The basis must be larger than the difference range.
                        # If we use 15x15, we can only handle differences up to 7.
                        # This means we need a larger basis or a different approach.
                        # Let's use a larger basis for the solver, e.g., 21x21 (M=10).
                        # 21x21 = 441 matrix. 441^3 ~ 8.5e7. Still fast.
                        pass
                    
            # The above loop is too slow in Python.
            # We need to vectorize or use a smarter matrix construction.
            # Let's restart the solver logic with vectorization.
            
        return max_omega_4, min_omega_5

    # The above solver draft is inefficient. Let's implement a vectorized version.
    
    def solve_bands_vectorized(cell, n_pw=21, k_points=None):
        cell = np.array(cell)
        N = len(cell)
        
        # Fourier coefficients
        M = (n_pw - 1) // 2
        G_vals = np.arange(-M, M + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.vstack([Gx.ravel(), Gy.ravel()])
        
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        x = (i + 0.5) / N - 0.5
        y = (j + 0.5) / N - 0.5
        r = np.vstack([x.ravel(), y.ravel()])
        
        eps_vals = np.where(cell.ravel() == 1, eps_high, eps_low)
        
        # Phase: (N_G, N_pixels)
        phase = np.exp(-1j * (G_vectors[0, :] @ r[0, :] + G_vectors[1, :] @ r[1, :]))
        eps_G = np.sum(eps_vals * phase, axis=1) / (N * N)
        
        # Map G to index
        G_set = set(zip(G_vectors[0], G_vectors[1]))
        G_to_idx = {g: idx for idx, g in enumerate(zip(G_vectors[0], G_vectors[1]))}
        
        # Precompute inverse epsilon for all G differences that fall within basis
        # We need H_{ij} = (k+Gi).(k+Gj) / eps_{Gi-Gj}
        
        if k_points is None:
            k_points = [
                (0.0, 0.0), (0.25, 0.0), (0.5, 0.0),
                (0.5, 0.25), (0.5, 0.5), (0.25, 0.25),
                (0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.3, 0.3)
            ]
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kx, ky in k_points:
            kx_G = kx + G_vectors[0, :]
            ky_G = ky + G_vectors[1, :]
            kG_sq = kx_G**2 + ky_G**2
            
            H = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
            
            # Vectorized construction of H
            # We need to iterate over pairs of G vectors.
            # This is O(N_G^2).
            # N_G = 441. 441^2 ~ 200k.
            # This loop might be slow in Python.
            
            # Optimization:
            # H_{ij} depends on Gi - Gj.
            # We can precompute a matrix of eps_inv for all differences.
            
            # Create a matrix of G differences
            # Gi is (N_G, 2), Gj is (N_G, 2)
            # Diff = Gi[:, np.newaxis, :] - Gj[np.newaxis, :, :]
            # This is (N_G, N_G, 2).
            
            # Instead, let's just do the loop but in a tighter way.
            # Or use scipy.sparse? No, dense is fine.
            
            # Let's try to speed up the inner loop by precomputing eps_inv for all possible differences
            # that are within the basis.
            
            # Actually, 200k operations in Python is ~0.1s.
            # 10 k-points * 0.1s = 1s per candidate.
            # If we have 50 candidates, that's 50s. Acceptable.
            
            for i_idx in range(n_pw*n_pw):
                Gi = G_vectors[:, i_idx]
                kx_Gi = kx + Gi[0]
                ky_Gi = ky + Gi[1]
                for j_idx in range(n_pw*n_pw):
                    Gj = G_vectors[:, j_idx]
                    G_diff = Gi - Gj
                    if tuple(G_diff) in G_to_idx:
                        idx_diff = G_to_idx[tuple(G_diff)]
                        eps_diff = eps_G[idx_diff]
                        if abs(eps_diff) > 1e-10:
                            H[i_idx, j_idx] = (kx_Gi * (kx + Gj[0]) + ky_Gi * (ky + Gj[1])) / eps_diff
                        else:
                            H[i_idx, j_idx] = 0
                    else:
                        H[i_idx, j_idx] = 0
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(eigenvalues)
            
            # Bands 4 and 5 (1-indexed in problem, 0-indexed in array: 3 and 4)
            # Problem says "between fourth and fifth TM bands".
            # So we look at eigenvalues[3] and eigenvalues[4].
            
            if len(eigenvalues) >= 5:
                omega_4 = eigenvalues[3]
                omega_5 = eigenvalues[4]
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        return max_omega_4, min_omega_5

    def score_cell(cell):
        try:
            max_4, min_5 = solve_bands_vectorized(cell, n_pw=15, k_points=[(0,0), (0.5,0), (0.5,0.5)])
            if min_5 <= max_4:
                return -1.0 # Overlap
            return 2 * (min_5 - max_4) / (min_5 + max_4)
        except:
            return -1.0

    # Generate candidates
    candidates = []
    
    def make_cell_from_params(shape_type, params):
        cell = np.zeros((size, size), dtype=int)
        i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        
        if shape_type == 'circle':
            R = params['R']
            mask = x**2 + y**2 < R**2
        elif shape_type == 'square':
            R = params['R']
            mask = np.maximum(np.abs(x), np.abs(y)) < R
        elif shape_type == 'diamond':
            R = params['R']
            mask = np.abs(x) + np.abs(y) < R
        elif shape_type == 'superellipse':
            R = params['R']
            n = params['n']
            mask = np.abs(x)**n + np.abs(y)**n < R**n
        elif shape_type == 'ring':
            R_in = params['R_in']
            R_out = params['R_out']
            r = np.sqrt(x**2 + y**2)
            mask = (r > R_in) & (r < R_out)
        elif shape_type == 'cross':
            w = params['w']
            mask = (np.abs(x) < w) | (np.abs(y) < w)
        else:
            return None
            
        cell[mask] = 1
        return cell.tolist()

    # Sweep parameters
    # Diamond is often best for square lattices
    for R in np.arange(0.1, 0.5, 0.02):
        candidates.append(make_cell_from_params('diamond', {'R': R}))
        
    for R in np.arange(0.1, 0.5, 0.02):
        candidates.append(make_cell_from_params('square', {'R': R}))
        
    for R in np.arange(0.1, 0.5, 0.02):
        candidates.append(make_cell_from_params('circle', {'R': R}))
        
    for n in [2, 3, 4, 6, 8, 12]:
        for R in np.arange(0.1, 0.5, 0.05):
            candidates.append(make_cell_from_params('superellipse', {'R': R, 'n': n}))
            
    for R_in in np.arange(0.05, 0.3, 0.05):
        for R_out in np.arange(R_in + 0.05, 0.5, 0.05):
            candidates.append(make_cell_from_params('ring', {'R_in': R_in, 'R_out': R_out}))
            
    for w in np.arange(0.05, 0.3, 0.02):
        candidates.append(make_cell_from_params('cross', {'w': w}))
        
    # Filter valid candidates
    valid_candidates = []
    for c in candidates:
        if c is None:
            continue
        if is_symmetric(c):
            valid_candidates.append(c)
        else:
            # Fix symmetry if needed? Or just skip.
            # Parametric shapes should be symmetric.
            pass

    # Score all candidates
    best_score = -np.inf
    best_cell = valid_candidates[0] if valid_candidates else None
    
    for c in valid_candidates:
        s = score_cell(c)
        if s > best_score:
            best_score = s
            best_cell = c
        if time.time() - start_time > time_budget * 0.8:
            break

    # Local search on best cell
    if best_cell is not None and time.time() - start_time < time_budget * 0.9:
        best_cell_arr = np.array(best_cell)
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            # Try flipping pixels in the fundamental domain
            # Fundamental domain: i <= j, i < 16, j < 16
            for i in range(16):
                for j in range(i, 16):
                    # Flip pixel (i,j) and its symmetric counterparts
                    # (i,j), (j,i), (31-i, j), (31-i, 31-j), etc.
                    # Actually, just flip the whole symmetric group.
                    orig_val = best_cell_arr[i, j]
                    new_val = 1 - orig_val
                    
                    # Apply flip to all symmetric positions
                    positions = [
                        (i, j), (j, i),
                        (31-i, j), (31-j, i),
                        (31-i, 31-j), (31-j, 31-i)
                    ]
                    # Remove duplicates
                    positions = list(set(positions))
                    
                    for p in positions:
                        best_cell_arr[p] = new_val
                    
                    new_score = score_cell(best_cell_arr.tolist())
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        # Revert
                        for p in positions:
                            best_cell_arr[p] = orig_val
        
        best_cell = best_cell_arr.tolist()
        
    return best_cell
# EVOLVE-BLOCK-END
