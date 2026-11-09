import numpy as np
import time
from math import cos, sin, pi

def design(problem):
    """
    Designs a 32x32 unit cell with a single C4v-symmetric dielectric shape
    at the center, using complex algebraic curve families optimized for
    maximum TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    def make_cell_c4v(curve_type, param1, param2, param3, size=32):
        """
        Create a C4v-symmetric cell with a single centered shape.
        curve_type: 0=rotated ellipse, 1=lemniscate, 2=astroid-like, 3=superellipse
        param1: main size parameter
        param2: aspect ratio or rotation parameter
        param3: additional shape parameter
        """
        cell = np.zeros((size, size), dtype=int)
        cx = cy = size / 2.0
        
        for i in range(size):
            for j in range(size):
                y_pix = i + 0.5
                x_pix = j + 0.5
                dx = x_pix - cx
                dy = y_pix - cy
                r = sqrt(dx*dx + dy*dy)
                
                inside = False
                if curve_type == 0:  # Rotated ellipse
                    angle = pi / 4.0  # 45 degrees for C4v
                    x_rot = dx * cos(angle) + dy * sin(angle)
                    y_rot = -dx * sin(angle) + dy * cos(angle)
                    a = param1
                    b = param1 * param2
                    if a > 0 and b > 0:
                        inside = (x_rot/a)**2 + (y_rot/b)**2 < 1.0
                elif curve_type == 1:  # Lemniscate of Bernoulli
                    a = param1
                    if a > 0:
                        # (x^2 + y^2)^2 = a^2 * (x^2 - y^2)
                        left = (dx*dx + dy*dy)**2
                        right = a*a * abs(dx*dx - dy*dy)
                        inside = left < right
                elif curve_type == 2:  # Astroid-like: |x|^n + |y|^n = r^n
                    n = param1
                    r_param = param2
                    if r_param > 0 and n > 0:
                        inside = abs(dx/r_param)**n + abs(dy/r_param)**n < 1.0
                elif curve_type == 3:  # Superellipse with rotation
                    n = param1
                    r_param = param2
                    angle = pi/4.0
                    x_rot = dx * cos(angle) + dy * sin(angle)
                    y_rot = -dx * sin(angle) + dy * cos(angle)
                    if r_param > 0 and n > 0:
                        inside = abs(x_rot/r_param)**n + abs(y_rot/r_param)**n < 1.0
                elif curve_type == 4:  # Modified lemniscate with parameter
                    a = param1
                    b = param2
                    if a > 0 and b > 0:
                        # Generalized: (x^2+y^2)^2 = a^2*x^2 + b^2*y^2
                        left = (dx*dx + dy*dy)**2
                        right = a*a*dx*dx + b*b*dy*dy
                        inside = left < right
                elif curve_type == 5:  # Circle with sinusoidal modulation
                    r_base = param1
                    amp = param2
                    freq = param3
                    if r_base > 0:
                        theta = atan2(dy, dx)
                        r_mod = r_base * (1.0 + amp * cos(freq * theta))
                        inside = r < r_mod
        
        # Enforce C4v symmetry: average over 8 symmetric positions
        # For a single centered shape, just ensure 180-degree and 90-degree symmetry
        for i in range(size):
            for j in range(size):
                val = cell[i, j]
                # Check 90-degree rotation
                cell[j, size-1-i] = max(cell[j, size-1-i], val)
                # Check 180-degree rotation
                cell[size-1-i, size-1-j] = max(cell[size-1-i, size-1-j], val)
                # Check reflection
                cell[i, size-1-j] = max(cell[i, size-1-j], val)
        
        return cell
    
    def compute_fourier_coefficients(cell, n_max=16):
        """Compute Fourier coefficients of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d
    
    def solve_bands(cell, n_pw=10, k_pts=None):
        """
        Fast TM band solver using plane-wave expansion.
        Returns gap-midgap ratio for bands 4 and 5.
        """
        fft_2d = compute_fourier_coefficients(cell)
        
        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]
        
        if k_pts is None:
            # Key points in irreducible BZ: (0,0), (0.5,0), (0.5,0.5), (0.25,0.25)
            k_pts = [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.25, 0.25),
                (0.25, 0.0),
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
    
    def sqrt(x):
        return x ** 0.5 if x >= 0 else 0
    
    def atan2(y, x):
        return np.arctan2(y, x)
    
    # Search over curve types and parameters
    best_score = -1e10
    best_params = (3, 8.0, 1.0, 0.0)  # Default: superellipse
    
    curve_types = [0, 1, 2, 3, 4, 5]
    
    # Coarse search
    for ct in curve_types:
        if time.time() > deadline:
            break
        
        if ct == 0:  # Rotated ellipse
            params = [(6.0, 1.0, 0), (7.0, 0.8, 0), (8.0, 0.7, 0), (7.0, 1.0, 0), (6.5, 0.9, 0)]
        elif ct == 1:  # Lemniscate
            params = [(8.0, 0, 0), (10.0, 0, 0), (12.0, 0, 0), (9.0, 0, 0), (11.0, 0, 0)]
        elif ct == 2:  # Astroid
            params = [(3.0, 8.0, 0), (4.0, 9.0, 0), (3.5, 8.5, 0), (4.5, 9.5, 0), (3.0, 9.0, 0)]
        elif ct == 3:  # Superellipse rotated
            params = [(3.0, 8.0, 0), (4.0, 9.0, 0), (3.5, 8.5, 0), (5.0, 9.5, 0), (4.0, 8.0, 0)]
        elif ct == 4:  # Modified lemniscate
            params = [(8.0, 8.0, 0), (10.0, 10.0, 0), (12.0, 12.0, 0), (9.0, 9.0, 0), (11.0, 11.0, 0)]
        elif ct == 5:  # Sinusoidal circle
            params = [(7.0, 0.1, 4), (8.0, 0.15, 4), (7.5, 0.2, 4), (8.5, 0.1, 4), (7.0, 0.1, 8)]
        
        for p1, p2, p3 in params:
            if time.time() > deadline:
                break
            cell = make_cell_c4v(ct, p1, p2, p3)
            score = solve_bands(cell, n_pw=10)
            if score > best_score:
                best_score = score
                best_params = (ct, p1, p2, p3)
    
    # Fine search around best
    if time.time() < deadline - 50:
        ct_best, p1_best, p2_best, p3_best = best_params
        
        for _ in range(3):  # 3 refinement iterations
            step_p1 = 0.5
            step_p2 = 0.1
            improved = False
            
            for dp1 in [-step_p1, 0, step_p1]:
                for dp2 in [-step_p2, 0, step_p2]:
                    if time.time() > deadline:
                        break
                    p1 = p1_best + dp1
                    p2 = p2_best + dp2
                    p3 = p3_best
                    
                    if p1 < 2 or p1 > 15:
                        continue
                    if p2 < 0.3 or p2 > 1.5:
                        continue
                    
                    cell = make_cell_c4v(ct_best, p1, p2, p3)
                    score = solve_bands(cell, n_pw=12)
                    if score > best_score:
                        best_score = score
                        best_params = (ct_best, p1, p2, p3)
                        p1_best, p2_best = p1, p2
                        improved = True
                if time.time() > deadline:
                    break
            if not improved:
                break
    
    # Create final cell
    ct_final, p1_final, p2_final, p3_final = best_params
    final_cell = make_cell_c4v(ct_final, p1_final, p2_final, p3_final)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
