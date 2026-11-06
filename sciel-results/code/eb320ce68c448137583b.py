"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 220.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE
        
    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points as specified)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        
    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs
        
    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        # 4-fold rotation: 90, 180, 270 degrees
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        # Reflections
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        # Transpose
        trans = cell.T.copy()
        # All 8 symmetries
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    def generate_from_radial(r0, a4, a8, phase4, phase8):
        """
        Generate a cell with a central dielectric region defined by a radial function
        r(theta) = r0 + a4*cos(4*theta + phase4) + a8*cos(8*theta + phase8)
        This maintains 4-fold symmetry (only m=4,8 harmonics).
        """
        cell = np.zeros((size, size), dtype=float)
        # Pixel centers relative to cell center
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Radial boundary
        r_boundary = r0 + a4 * np.cos(4 * Theta + phase4) + a8 * np.cos(8 * Theta + phase8)
        
        # Fill pixels where R <= r_boundary
        cell[R <= r_boundary] = 1.0
        
        cell = cell.astype(int)
        return enforce_symmetry(cell)

    def evaluate_params(params):
        """Evaluate a parameter set, returning negative ratio (for minimization)."""
        if not time_ok():
            return 0.0
        r0, a4, a8, phase4, phase8 = params
        cell = generate_from_radial(r0, a4, a8, phase4, phase8)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        return -ratio

    # Phase 1: Differential evolution over radial parameters
    # Parameters: r0 (base radius), a4 (m=4 amplitude), a8 (m=8 amplitude), phase4, phase8
    # r0 in [0.15, 0.45], a4 in [0, 0.15], a8 in [0, 0.10], phases in [0, pi/2]
    bounds = [
        (0.15, 0.45),   # r0
        (0.0, 0.15),    # a4
        (0.0, 0.10),    # a8
        (0.0, np.pi/2), # phase4
        (0.0, np.pi/2), # phase8
    ]
    
    best_ratio = -np.inf
    best_params = None
    best_cell = None
    
    # Run differential evolution
    try:
        result = differential_evolution(
            evaluate_params,
            bounds,
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,
            callback=lambda x, convergence: None
        )
        if result.success or result.fun < 0:
            best_params = result.x
            best_ratio = -result.fun
            best_cell = generate_from_radial(*best_params)
    except Exception:
        pass
    
    # If DE didn't find anything good, try a grid search
    if best_ratio < 0.01 or best_cell is None:
        r0_vals = [0.20, 0.25, 0.30, 0.35]
        a4_vals = [0.0, 0.05, 0.10]
        a8_vals = [0.0, 0.03, 0.06]
        phase_vals = [0.0, np.pi/4, np.pi/2]
        
        for r0 in r0_vals:
            for a4 in a4_vals:
                for a8 in a8_vals:
                    for p4 in phase_vals:
                        for p8 in phase_vals:
                            if not time_ok():
                                break
                            cell = generate_from_radial(r0, a4, a8, p4, p8)
                            coeffs = get_fourier_coeffs(cell)
                            ratio = compute_band_ratio(coeffs)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_params = (r0, a4, a8, p4, p8)
                                best_cell = cell.copy()
                            if not time_ok():
                                break
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    if best_cell is None:
        best_cell = generate_from_radial(0.25, 0.05, 0.03, 0.0, 0.0)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (0.25, 0.05, 0.03, 0.0, 0.0)

    # Phase 2: Local refinement with Nelder-Mead
    current_params = np.array(best_params)
    current_ratio = best_ratio
    current_cell = best_cell.copy()
    
    search_start = time.time()
    search_time_limit = 100.0
    
    try:
        from scipy.optimize import minimize
        # Use Nelder-Mead for local refinement
        def objective(params):
            if not time_ok():
                return 0.0
            r0, a4, a8, phase4, phase8 = params
            cell = generate_from_radial(r0, a4, a8, phase4, phase8)
            coeffs = get_fourier_coeffs(cell)
            return -compute_band_ratio(coeffs)
        
        res = minimize(
            objective,
            current_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': 0.02}
        )
        if res.fun < -current_ratio:
            current_params = res.x
            current_ratio = -res.fun
            current_cell = generate_from_radial(*current_params)
    except Exception:
        pass
    
    # Phase 3: Pixel-level refinement
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 20:
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 15.0
        
        no_improve_refine = 0
        for (i, j) in indep_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
                
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1
                if no_improve_refine > 30:
                    break

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
        
    return current_cell.tolist()
# EVOLVE-BLOCK-END
