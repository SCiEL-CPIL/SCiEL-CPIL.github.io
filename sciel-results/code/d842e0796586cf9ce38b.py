import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell with superellipse-based dielectric shapes
    optimized to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 205.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    def make_cell_superellipse(radius, n_order, offset, size=32):
        """Create a symmetric 4-shape cell using superellipse boundaries."""
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (size * 0.25 + offset, size * 0.25 + offset),
            (size * 0.75 - offset, size * 0.25 + offset),
            (size * 0.25 + offset, size * 0.75 - offset),
            (size * 0.75 - offset, size * 0.75 - offset),
        ]
        for i in range(size):
            for j in range(size):
                y_pix = i + 0.5
                x_pix = j + 0.5
                for cx, cy in centers:
                    dx = x_pix - cx
                    dy = y_pix - cy
                    # Superellipse: |dx/r|^n + |dy/r|^n < 1
                    if radius > 0:
                        val = abs(dx / radius) ** n_order + abs(dy / radius) ** n_order
                        if val < 1.0:
                            cell[i, j] = 1
                            break
        return cell
    
    def compute_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d
    
    def solve_bands(cell, n_pw=12, k_pts=None):
        """
        Fast TM band solver using plane-wave expansion.
        Returns gap-midgap ratio for bands 4 and 5.
        """
        fft_2d = compute_fft_inv_eps(cell)
        
        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]
        
        if k_pts is None:
            k_pts = [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.25, 0.0),
                (0.25, 0.25),
                (0.5, 0.25),
            ]
        
        # Build list of plane waves within circular cutoff
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))
        
        n_pw_actual = len(g_pairs)
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in k_pts:
            # Build Hermitian matrix for TM polarization
            # H[G, G'] = eps_inv[G - G'] * |k - G'|^2
            # To make it Hermitian, we symmetrize: H = (M + M^H) / 2
            # where M[i,j] = eps_inv[Gi-Gj] * |k-Gj|^2
            M = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            
            for i, (mi, ni) in enumerate(g_pairs):
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    M[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)
            
            # Symmetrize to get Hermitian matrix
            H = (M + M.conj().T) / 2.0
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # Search parameters: radius, superellipse order, offset
    best_score = -1e10
    best_params = (5.5, 2.0, 0.0)
    
    # Coarse search over superellipse parameters
    radii = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]
    orders = [2.0, 3.0, 4.0, 6.0]  # 2=circle, 4=square, higher=more square-like
    offsets = [-1.0, 0.0, 1.0]
    
    for r in radii:
        for n_ord in orders:
            for off in offsets:
                if time.time() > deadline:
                    break
                cell = make_cell_superellipse(r, n_ord, off)
                score = solve_bands(cell, n_pw=10)
                if score > best_score:
                    best_score = score
                    best_params = (r, n_ord, off)
        if time.time() > deadline:
            break
    
    # Fine search around best
    if time.time() < deadline - 40:
        r_best, n_best, off_best = best_params
        for _ in range(2):  # 2 refinement iterations
            step_r = 0.5
            step_n = 0.5
            step_off = 0.5
            improved = False
            for dr in [-step_r, 0, step_r]:
                for dn in [-step_n, 0, step_n]:
                    for do in [-step_off, 0, step_off]:
                        if time.time() > deadline:
                            break
                        r = r_best + dr
                        n_ord = n_best + dn
                        off = off_best + do
                        if r < 2 or r > 12 or n_ord < 1.5 or n_ord > 8:
                            continue
                        cell = make_cell_superellipse(r, n_ord, off)
                        score = solve_bands(cell, n_pw=12)
                        if score > best_score:
                            best_score = score
                            best_params = (r, n_ord, off)
                            r_best, n_best, off_best = r, n_ord, off
                            improved = True
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if not improved:
                break
    
    # Create final cell
    r_final, n_final, off_final = best_params
    final_cell = make_cell_superellipse(r_final, n_final, off_final)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
