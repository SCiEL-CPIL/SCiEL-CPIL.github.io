# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 200s limit

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: row/col reversal and transpose invariance."""
        final_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[size-1-i][j],
                    cell[i][size-1-j],
                    cell[size-1-i][size-1-j],
                    cell[j][i],
                    cell[size-1-j][size-1-i],
                    cell[j][size-1-i],
                    cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                final_cell[i][j] = 1 if avg >= 0.5 else 0
        return final_cell

    def compute_fourier_coeffs_matrix(cell, max_order):
        """Compute Fourier coefficients as a numpy matrix indexed by (m+max_order, n+max_order)."""
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        mat = np.zeros((2 * max_order + 1, 2 * max_order + 1), dtype=complex)
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                mat[m + max_order, n + max_order] = fft_2d[m % size, n % size]
        return mat

    def approximate_gap_ratio(cell, max_order=5, g_range=4):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        """
        coeffs_mat = compute_fourier_coeffs_matrix(cell, max_order)
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # Key k-points in irreducible BZ
        k_points = []
        for kx in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            k_points.append((kx, 0.0))
        for kx in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            k_points.append((kx, kx))
        for kx in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            for ky in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44]:
                if ky <= kx:
                    k_points.append((kx, ky))
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        gx = np.array([g[0] for g in g_vectors], dtype=float)
        gy = np.array([g[1] for g in g_vectors], dtype=float)
        
        for kx, ky in k_points:
            kx_total = kx + gx
            ky_total = ky + gy
            k_sq = kx_total**2 + ky_total**2
            
            H = np.zeros((N, N), dtype=complex)
            for i in range(N):
                for j in range(N):
                    m_diff = g_vectors[i][0] - g_vectors[j][0]
                    n_diff = g_vectors[i][1] - g_vectors[j][1]
                    if -max_order <= m_diff <= max_order and -max_order <= n_diff <= max_order:
                        eps_inv_G = coeffs_mat[m_diff + max_order, n_diff + max_order]
                    else:
                        eps_inv_G = 0.0
                    H[i, j] = k_sq[i] * eps_inv_G
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def generate_2x2_supercell_params(params):
        """
        Generate a 2x2 supercell of rods with optional central feature.
        params: [r_rod, r_center, type_center (0=none, 1=rod, 2=ring_inner, 3=ring_outer)]
        Simplified: Just 2 rods radius and 1 central radius.
        """
        r_rod = params[0]
        r_center = params[1]
        
        cell = [[0] * size for _ in range(size)]
        cx_center = size / 2.0
        cy_center = size / 2.0
        offset = size / 4.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        
        r_rod_px = r_rod * size
        r_center_px = r_center * size
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                is_dielectric = 0
                # Check 4 rods
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_rod_px * r_rod_px:
                        is_dielectric = 1
                        break
                # Check center
                if is_dielectric == 0:
                    dx = x - cx_center
                    dy = y - cy_center
                    if dx * dx + dy * dy < r_center_px * r_center_px:
                        is_dielectric = 1
                
                cell[i][j] = is_dielectric
        return cell

    # Initial best from discrete search (quick pass)
    best_cell = None
    best_score = -1.0
    
    # Quick discrete search to find a good starting point
    # We'll use a few representative candidates
    def generate_single_rod(radius_frac):
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    cell[i][j] = 1
        return cell

    def generate_2x2_supercell(rod_radius_frac):
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    candidates = []
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_2x2_supercell(r))
    for r in [0.20, 0.25, 0.30]:
        candidates.append(generate_single_rod(r))
        
    for cell in candidates:
        if time.time() - start_time > deadline - 60:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym, max_order=4, g_range=3)
        if score > best_score:
            best_score = score
            best_cell = cell_sym

    if best_cell is None:
        best_cell = generate_2x2_supercell(0.15)
        best_cell = enforce_d4_symmetry(best_cell)
        best_score = approximate_gap_ratio(best_cell, max_order=4, g_range=3)

    # Continuous optimization of the 2x2 supercell parameters
    # We assume the best structure is of the 2x2 supercell type.
    # We optimize [r_rod, r_center]
    # We need to map the current best_cell to initial parameters.
    # Heuristic: Estimate r_rod from the 2x2 supercell candidates.
    # Let's just start with a reasonable guess and let the optimizer work.
    
    # Define the objective function for the optimizer
    # We want to maximize the gap ratio, so we minimize the negative.
    # To handle the non-smooth min/max, we use a soft approximation.
    # However, approximate_gap_ratio already returns a scalar.
    # The non-smoothness is in the min/max over k-points.
    # We can just use the discrete value, but Powell's method is robust to non-smoothness.
    
    def objective(params):
        r_rod = params[0]
        r_center = params[1]
        if r_rod < 0 or r_rod > 0.5 or r_center < 0 or r_center > 0.5:
            return 1.0 # Penalty
        
        cell = generate_2x2_supercell_params([r_rod, r_center])
        cell_sym = enforce_d4_symmetry(cell)
        
        # Use fast approximation
        score = approximate_gap_ratio(cell_sym, max_order=4, g_range=3)
        return -score

    # Initial guess based on best discrete candidate
    # If best_cell was a 2x2 supercell, we can try to extract r_rod.
    # For simplicity, we start with r_rod=0.15, r_center=0.0
    x0 = [0.15, 0.0]
    
    # Try a few starting points
    starts = [[0.15, 0.0], [0.12, 0.05], [0.18, 0.0], [0.10, 0.10]]
    
    opt_best_score = best_score
    opt_best_params = x0
    
    for start in starts:
        if time.time() - start_time > deadline - 30:
            break
            
        try:
            res = minimize(objective, start, method='Powell', 
                           options={'maxiter': 50, 'maxfev': 200, 'disp': False})
            
            if res.success or res.fun < objective(opt_best_params):
                # Check if the result is valid
                r_rod = res.x[0]
                r_center = res.x[1]
                if 0 <= r_rod <= 0.5 and 0 <= r_center <= 0.5:
                    cell_opt = generate_2x2_supercell_params(res.x)
                    cell_opt_sym = enforce_d4_symmetry(cell_opt)
                    score_opt = approximate_gap_ratio(cell_opt_sym, max_order=4, g_range=3)
                    
                    if score_opt > opt_best_score:
                        opt_best_score = score_opt
                        opt_best_params = res.x
                        best_cell = cell_opt_sym
                        best_score = score_opt
                        
        except Exception:
            pass

    # Final high-precision evaluation and local discrete polish
    if best_cell is not None and time.time() - start_time < deadline - 10:
        # Re-evaluate with higher precision
        final_score = approximate_gap_ratio(best_cell, max_order=6, g_range=4)
        if final_score > best_score:
            best_score = final_score
            
        # Local discrete polish
        base = best_cell
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            best_local = best_score
            best_local_cell = base
            
            for flip_i in range(0, size, 2): # Step by 2 to save time
                for flip_j in range(0, size, 2):
                    if time.time() - start_time > deadline - 2:
                        break
                    
                    cell_trial = [row[:] for row in base]
                    cell_trial[flip_i][flip_j] = 1 - base[flip_i][flip_j]
                    cell_sym = enforce_d4_symmetry(cell_trial)
                    if cell_sym != base:
                        score = approximate_gap_ratio(cell_sym, max_order=4, g_range=3)
                        if score > best_local + 1e-6:
                            best_local = score
                            best_local_cell = cell_sym
                            improved = True
                    if time.time() - start_time > deadline - 2:
                        break
                if time.time() - start_time > deadline - 2:
                    break
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base = best_cell
                # Update best_cell to the new local best

    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
