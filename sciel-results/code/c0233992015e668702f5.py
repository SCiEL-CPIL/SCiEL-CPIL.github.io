import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry enforcement:
    # The cell must be symmetric under D4.
    # We can work on the top-left 16x16 quadrant and mirror.
    
    def expand_quadrant(Q):
        # Q is 16x16
        # Top-left: Q
        # Top-right: Q[:, ::-1]
        # Bottom-left: Q[::-1, :]
        # Bottom-right: Q[::-1, ::-1]
        top = np.hstack((Q, Q[:, ::-1]))
        bottom = np.hstack((Q[::-1, :], Q[::-1, ::-1]))
        return np.vstack((top, bottom))

    # PWE Solver
    n_g = 8
    G_vecs = []
    for nx in range(-n_g, n_g+1):
        for ny in range(-n_g, n_g+1):
            G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
    N_G = len(G_vecs)
    
    # Pre-compute G indices for epsilon coupling
    E_indices = np.zeros((N_G, N_G), dtype=int)
    for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
        for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
            dx = nx - nxp
            dy = ny - nyp
            if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                E_indices[idx_g, idx_gp] = (dx + n_g) * (2*n_g+1) + (dy + n_g)
            else:
                E_indices[idx_g, idx_gp] = -1
    valid_mask = E_indices != -1
    eps_indices_flat = E_indices[valid_mask]

    def get_fourier_coefficients(cell):
        """
        Compute Fourier coefficients of epsilon.
        cell is 32x32 binary or continuous.
        """
        coeffs = np.zeros((2*n_g+1, 2*n_g+1), dtype=complex)
        M = np.array(cell, dtype=float)
        
        term_x = np.zeros((size, 2*n_g+1), dtype=complex)
        term_y = np.zeros((size, 2*n_g+1), dtype=complex)
        
        for n in range(-n_g, n_g+1):
            idx = n + n_g
            G = 2 * np.pi * n
            if G == 0:
                term_x[:, idx] = 1.0 / size
                term_y[:, idx] = 1.0 / size
            else:
                sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                base_exp = np.exp(-1j * G * 0.5 / size)
                i_vals = np.arange(size)
                exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                term_x[:, idx] = sinc_factor * exp_terms
                term_y[:, idx] = sinc_factor * exp_terms
        
        f_coeffs = term_x.T @ (M @ term_y)
        
        delta_eps = eps_high - eps_low
        eps_coeffs = f_coeffs * delta_eps
        eps_coeffs[n_g, n_g] += eps_low
        
        return eps_coeffs

    def solve_bands(eps_coeffs, k_points):
        """
        Solve for bands at given k points.
        Returns list of arrays of omega values.
        """
        all_omegas = []
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            A = np.zeros((N_G, N_G), dtype=complex)
            A[valid_mask] = eps_flat[eps_indices_flat]
            
            kG_x = np.array([kx + G[0] for G in G_vecs])
            kG_y = np.array([ky + G[1] for G in G_vecs])
            
            T = np.outer(kG_x, kG_x) + np.outer(kG_y, kG_y)
            A_mat = A * T
            
            # Hermitian symmetrize
            A_mat = (A_mat + A_mat.conj().T) / 2.0
            
            try:
                w2, v = np.linalg.eigh(A_mat)
                w2 = np.real(w2)
                
                valid_idx = w2 > 1e-10
                w2_valid = w2[valid_idx]
                
                omegas = 1.0 / np.sqrt(w2_valid)
                omegas_sorted = np.sort(omegas)
                
                all_omegas.append(omegas_sorted)
            except Exception:
                continue
        
        return all_omegas

    def compute_gap(eps_coeffs, k_points):
        """
        Compute gap and midgap ratio.
        """
        all_omegas = solve_bands(eps_coeffs, k_points)
        
        if not all_omegas:
            return 0.0, 0.0, 0.0
            
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for omegas in all_omegas:
            if len(omegas) >= 5:
                w4 = omegas[3]
                w5 = omegas[4]
                if w4 > max_omega4:
                    max_omega4 = w4
                if w5 < min_omega5:
                    min_omega5 = w5
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return 0.0, 0.0, 0.0
            
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        ratio = gap / midgap if midgap > 0 else 0.0
        
        return ratio, max_omega4, min_omega5

    # Optimization Loop
    # 1. Generate diverse candidates
    candidates = []
    
    # Symmetric shapes
    # Central Square
    for s in range(4, 16, 2):
        Q = np.zeros((16, 16), dtype=int)
        Q[8-s//2:8+s//2, 8-s//2:8+s//2] = 1
        candidates.append(expand_quadrant(Q))
        
    # Central Circle
    for r in range(4, 16, 2):
        Q = np.zeros((16, 16), dtype=int)
        yy, xx = np.mgrid[0:16, 0:16]
        mask = (xx - 7.5)**2 + (yy - 7.5)**2 <= r**2
        Q[mask] = 1
        candidates.append(expand_quadrant(Q))
        
    # Checkerboard variations
    Q = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                Q[i, j] = 1
    candidates.append(expand_quadrant(Q))
    
    # Rotated Checkerboard (diagonals)
    Q = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            if (i - j) % 2 == 0:
                Q[i, j] = 1
    candidates.append(expand_quadrant(Q))
    
    # "Plus" shapes
    for w in range(2, 8, 2):
        Q = np.zeros((16, 16), dtype=int)
        Q[8-w//2:8+w//2, :] = 1
        Q[:, 8-w//2:8+w//2] = 1
        candidates.append(expand_quadrant(Q))
        
    # "X" shapes
    for w in range(2, 8, 2):
        Q = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                if abs(i - 8) <= abs(j - 8) - w//2 or abs(j - 8) <= abs(i - 8) - w//2:
                     # This logic is tricky, let's just do a simple diagonal band
                     pass
        # Simpler X:
        Q = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                if abs(i - 7.5) <= w/2 and abs(j - 7.5) <= w/2:
                    Q[i, j] = 1
                if abs((i - 7.5) - (j - 7.5)) <= w/2 or abs((i - 7.5) + (j - 7.5)) <= w/2:
                    Q[i, j] = 1
        candidates.append(expand_quadrant(Q))

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c.flatten())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    # K-points for optimization
    # Use a subset of the evaluator's k-points for speed
    k_points_opt = []
    # Gamma
    k_points_opt.append((0, 0))
    # M
    k_points_opt.append((np.pi, 0))
    k_points_opt.append((0, np.pi))
    k_points_opt.append((np.pi, np.pi))
    # X
    k_points_opt.append((np.pi/2, 0))
    k_points_opt.append((0, np.pi/2))
    k_points_opt.append((np.pi/2, np.pi))
    k_points_opt.append((np.pi, np.pi/2))
    # Interior
    k_points_opt.append((np.pi/4, np.pi/4))
    k_points_opt.append((3*np.pi/4, np.pi/4))
    k_points_opt.append((np.pi/4, 3*np.pi/4))
    k_points_opt.append((3*np.pi/4, 3*np.pi/4))
    k_points_opt.append((np.pi/2, np.pi/2))
    k_points_opt.append((np.pi/6, np.pi/6))
    k_points_opt.append((5*np.pi/6, np.pi/6))
    k_points_opt.append((np.pi/6, 5*np.pi/6))
    k_points_opt.append((5*np.pi/6, 5*np.pi/6))
    k_points_opt.append((np.pi/3, 0))
    k_points_opt.append((2*np.pi/3, 0))
    k_points_opt.append((0, np.pi/3))
    k_points_opt.append((0, 2*np.pi/3))

    best_ratio = -1.0
    best_cell = candidates[0]
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline - 30:
            break
        eps_coeffs = get_fourier_coefficients(c)
        ratio, _, _ = compute_gap(eps_coeffs, k_points_opt)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = c

    # Local Refinement: Hill Climbing on best_cell
    # We try flipping individual pixels in the quadrant
    Q_best = best_cell[:16, :16].copy()
    
    # Shuffle pixels to avoid bias
    pixel_indices = [(i, j) for i in range(16) for j in range(16)]
    np.random.seed(42)
    np.random.shuffle(pixel_indices)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i, j in pixel_indices:
            if time.time() > deadline - 10:
                break
            
            # Flip pixel
            old_val = Q_best[i, j]
            Q_best[i, j] = 1 - old_val
            
            cell_test = expand_quadrant(Q_best)
            eps_coeffs = get_fourier_coefficients(cell_test)
            ratio, _, _ = compute_gap(eps_coeffs, k_points_opt)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_cell = cell_test
                improved = True
            else:
                # Revert
                Q_best[i, j] = old_val
                
    return best_cell.tolist()
