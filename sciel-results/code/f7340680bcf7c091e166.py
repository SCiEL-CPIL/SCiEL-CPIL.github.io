import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def make_cell(shape_type, param1, param2=None):
        """Generate a symmetric 32x32 cell."""
        cell = [[0] * size for _ in range(size)]
        center = (size - 1) / 2.0
        
        for i in range(size):
            for j in range(size):
                # Normalized coordinates in [-0.5, 0.5]
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                
                is_dielectric = False
                if shape_type == 'circle':
                    r = param1
                    if x * x + y * y < r * r:
                        is_dielectric = True
                elif shape_type == 'square':
                    side = param1
                    if abs(x) < side / 2 and abs(y) < side / 2:
                        is_dielectric = True
                elif shape_type == 'super_ellipse':
                    r = param1
                    n = param2
                    if r > 0:
                        val = (abs(x / r) ** n + abs(y / r) ** n)
                        if val < 1.0:
                            is_dielectric = True
                        
                if is_dielectric:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
                    
        # Enforce strict symmetry (though generation should be symmetric)
        # For 32x32, center is 15.5. Symmetry maps (i,j) -> (31-i, j), (i, 31-j), (31-i, 31-j)
        # We just re-assign to be safe, though the math above is exact for floats.
        for i in range(size):
            for j in range(size):
                # If any of the symmetric partners is 1, and we want strict "OR" or "AND"?
                # The shapes are geometrically symmetric. 
                # x^2 + y^2 is symmetric. |x|^n + |y|^n is symmetric.
                # So cell[i][j] should equal cell[31-i][j] etc.
                # We can just leave it.
                pass
        
        return cell

    def compute_gap(cell):
        """
        Compute the gap-midgap ratio for TM bands 4 and 5.
        Uses a plane-wave method with 289 G-vectors (17x17).
        """
        # 1. Compute Fourier coefficients of epsilon(r)
        # epsilon(r) = eps_low + (eps_high - eps_low) * f(r)
        # f(r) is 1 inside dielectric, 0 outside.
        # f_G = (1/A) * integral f(r) exp(-i G.r) dr
        # For pixel map, integral is sum over pixels.
        
        def get_fourier_coefficients(cell):
            coeffs = np.zeros((17, 17), dtype=complex)
            # Map index (0..16) to n (-8..8)
            # G = 2*pi*n
            
            # Precompute pixel contributions?
            # For each G, sum over pixels where cell=1.
            # This is O(17*17 * 32*32) = 300k ops. Acceptable.
            
            # Optimization: Precompute the "sinc" terms for each pixel and each G component.
            # sinc_term(G, n) = (2/G) * exp(-i G (n+0.5)/size) * sin(G/(2size)) if G!=0 else 1/size
            
            # Let's just do the direct sum, it's fast enough in NumPy if vectorized or even pure Python for 300k.
            # But we need to do this for multiple candidates. Let's optimize.
            
            # Precompute terms for each pixel (i,j) and each Gx, Gy component.
            # term_x[i, nx] = integral over x for pixel i, Gx = 2*pi*nx
            # term_y[j, ny] = integral over y for pixel j, Gy = 2*pi*ny
            
            term_x = np.zeros((size, 17), dtype=complex)
            term_y = np.zeros((size, 17), dtype=complex)
            
            for n in range(-8, 9):
                idx = n + 8
                G = 2 * np.pi * n
                if G == 0:
                    term_x[:, idx] = 1.0 / size
                    term_y[:, idx] = 1.0 / size
                else:
                    # exp(-i G (pixel_center)) * sinc_factor
                    # pixel center for index i is (i+0.5)/size
                    # integral = (2/G) * sin(G/(2size)) * exp(-i G (i+0.5)/size)
                    sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                    # exp term varies with i
                    # exp(-i G (i+0.5)/size) = exp(-i G/(2size)) * exp(-i G i/size)
                    base_exp = np.exp(-1j * G * 0.5 / size)
                    angle_step = -1j * G / size
                    # Use np.exp with array of i
                    i_vals = np.arange(size)
                    exp_terms = np.exp(i_vals * angle_step) * base_exp
                    term_x[:, idx] = sinc_factor * exp_terms
                    term_y[:, idx] = sinc_factor * exp_terms
            
            # Now sum over pixels where cell[i][j] == 1
            # We can use matrix multiplication?
            # Sum_{i,j where cell=1} term_x[i, nx] * term_y[j, ny]
            # This is like (term_x.T * mask * term_y) ? No.
            # It's sum_i sum_j mask[i,j] term_x[i,nx] term_y[j,ny]
            # = sum_i (term_x[i,nx] * sum_j mask[i,j] term_y[j,ny])
            # Let M be mask matrix (size x size)
            # For each nx, ny:
            #   coeff[nx, ny] = sum_i sum_j M[i,j] tx[i,nx] ty[j,ny]
            
            # Vectorized approach:
            # tx is (size, 17), ty is (size, 17), M is (size, size)
            # We want C[nx, ny] = sum_i sum_j M[i,j] tx[i,nx] ty[j,ny]
            # This is equivalent to:
            # temp = M.T @ ty  -> (size, 17) where temp[i, ny] = sum_j M[j,i] ty[j,ny] ... wait indices.
            # M[i,j] is row i, col j.
            # sum_j M[i,j] ty[j,ny] = (M @ ty)[i, ny]
            # Then sum_i (M @ ty)[i, ny] tx[i, nx] = tx.T @ (M @ ty) -> (17, 17)
            
            M = np.array(cell, dtype=float)
            # M @ ty: (size, size) @ (size, 17) -> (size, 17)
            # tx.T: (17, size)
            # Result: (17, 17)
            coeffs = tx.T @ (M @ ty)
            
            # Add eps_low to the DC term?
            # epsilon_G = eps_low * delta_{G,0} + (eps_high - eps_low) * f_G
            # f_G is what we just computed.
            # For G=0 (nx=0, ny=0 -> idx 8,8), f_0 is the average.
            # eps_0 = eps_low + (eps_high - eps_low) * f_0
            # For G!=0, eps_G = (eps_high - eps_low) * f_G
            
            delta_eps = eps_high - eps_low
            coeffs[8, 8] += eps_low  # Only DC component gets eps_low
            
            return coeffs * delta_eps # Wait, DC term is eps_low + delta_eps * f_0.
                                      # So coeffs[8,8] should be eps_low + delta_eps*f_0.
                                      # My code: coeffs[8,8] = f_0. Then + eps_low. Then * delta_eps?
                                      # No.
                                      # Let's redo.
                                      # f_G computed above.
                                      # eps_G = eps_low * (1 if G=0 else 0) + delta_eps * f_G
                                      # So:
                                      # coeffs[8,8] = eps_low + delta_eps * f_0
                                      # coeffs[others] = delta_eps * f_G
                                      #
                                      # Current code:
                                      # coeffs = tx.T @ (M @ ty)  -> this is f_G
                                      # coeffs[8,8] += eps_low   -> f_0 + eps_low
                                      # return coeffs * delta_eps -> (f_0 + eps_low)*delta_eps ... WRONG for DC.
                                      #
                                      # Correction:
                                      # f_coeffs = tx.T @ (M @ ty)
                                      # eps_coeffs = f_coeffs * delta_eps
                                      # eps_coeffs[8,8] += eps_low
                                      # return eps_coeffs

            # Let's re-implement the return part correctly in the actual function body below
            pass

        # Re-doing get_fourier_coefficients correctly
        def get_fourier_coefficients_correct(cell):
            coeffs = np.zeros((17, 17), dtype=complex)
            term_x = np.zeros((size, 17), dtype=complex)
            term_y = np.zeros((size, 17), dtype=complex)
            
            for n in range(-8, 9):
                idx = n + 8
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
            
            M = np.array(cell, dtype=float)
            f_coeffs = term_x.T @ (M @ term_y)
            
            delta_eps = eps_high - eps_low
            eps_coeffs = f_coeffs * delta_eps
            eps_coeffs[8, 8] += eps_low
            
            return eps_coeffs

        eps_coeffs = get_fourier_coefficients_correct(cell)
        
        # 2. Solve eigenvalue problem for each k-point
        # G vectors
        G_vecs = []
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
        
        # K points in IBZ (0 <= kx <= ky <= pi)
        # Use a grid + high symmetry points
        k_points = []
        # Grid
        for i in range(5):
            for j in range(5):
                kx = i / 4.0 * np.pi
                ky = j / 4.0 * np.pi
                if kx <= ky:
                    k_points.append((kx, ky))
        # High symmetry
        k_points.append((0, 0))
        k_points.append((np.pi, 0))
        k_points.append((np.pi, np.pi))
        k_points.append((np.pi/2, np.pi/2))
        k_points.append((np.pi/3, 2*np.pi/3)) # Some interior point
        
        all_omegas = []
        
        # Pre-build E matrix structure?
        # E[idx_g, idx_gp] = eps_{G - G'}
        # G - G' = (nx - nxp, ny - nyp)
        # We can precompute the indices into eps_coeffs.
        
        N_G = 289
        E_indices = np.zeros((N_G, N_G), dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                # Map dx, dy (-16..16) to index?
                # Our eps_coeffs is 17x17 for -8..8.
                # If dx is outside -8..8, coeff is 0.
                if -8 <= dx <= 8 and -8 <= dy <= 8:
                    E_indices[idx_g, idx_gp] = (dx + 8) * 17 + (dy + 8)
                else:
                    E_indices[idx_g, idx_gp] = -1 # Sentinel for 0
        
        # Flatten eps_coeffs
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            if time.time() > deadline:
                break
                
            # Build D vector
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            # Build E matrix
            # E[idx_g, idx_gp] = eps_flat[E_indices[idx_g, idx_gp]] if != -1 else 0
            # Use advanced indexing
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            # Solve D^{-1} E c = mu c
            # Equivalent to E c = mu D c
            # Use scipy.linalg.eigh? No, non-Hermitian.
            # Use np.linalg.eigvals.
            # Or better: transform to Hermitian.
            # D^{1/2} E D^{-1/2} c' = mu c'
            # H = D^{1/2} E D^{-1/2}
            # H is Hermitian if E is Hermitian.
            # E is Hermitian? eps_G is Hermitian (eps_{-G} = conj(eps_G)).
            # Yes.
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                # H = sqrt_D[:, None] * E * inv_sqrt_D[None, :]
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                
                # Ensure Hermitian
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                # Filter valid
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio

    # Search
    # Based on inspirations:
    # 1. Circle r=0.2 (Score 1.0)
    # 2. Super-ellipse n=4, area-matched to r=0.2 circle (Score 0.9)
    # 3. Super-ellipse n=4, r=0.28 (Score 0.02 - bad)
    # 4. Current program tried various shapes.
    
    # Let's test a few promising candidates.
    # The seed (circle r=0.2) got 1.0.
    # Maybe a slightly larger circle or a squircle with same area but different shape?
    
    candidates = [
        ('circle', 0.2, None),
        ('circle', 0.22, None),
        ('circle', 0.18, None),
        ('super_ellipse', 0.2, 4.0), # Same radius, squircle
        ('super_ellipse', 0.22, 4.0),
        ('super_ellipse', 0.18, 4.0),
        ('super_ellipse', 0.25, 4.0),
        ('square', 0.28, None), # Square with side 0.28
        ('square', 0.32, None),
    ]
    
    best_score = -1.0
    best_cell = None
    
    for shape, p1, p2 in candidates:
        if time.time() > deadline:
            break
            
        cell = make_cell(shape, p1, p2)
        score = compute_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    if best_cell is None:
        # Fallback to seed
        best_cell = make_cell('circle', 0.2, None)
        
    return best_cell
