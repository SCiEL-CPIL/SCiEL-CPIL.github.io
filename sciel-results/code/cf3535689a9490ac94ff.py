import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell with a 4-shape superlattice pattern.
    Uses polygons with rounded corners (octagons) to optimize the TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 205.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    def make_cell_octagon(side_len, corner_radius, offset, size=32):
        """
        Create a symmetric 4-shape cell using octagonal boundaries with rounded corners.
        side_len: length of the straight edge of the octagon
        corner_radius: radius of the circular arcs replacing the corners
        offset: shift from the quarter-point center
        """
        cell = np.zeros((size, size), dtype=int)
        # Centers at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
        centers = [
            (size * 0.25 + offset, size * 0.25 + offset),
            (size * 0.75 - offset, size * 0.25 + offset),
            (size * 0.25 + offset, size * 0.75 - offset),
            (size * 0.75 - offset, size * 0.75 - offset),
        ]
        
        # Precompute pixel coordinates
        y_coords, x_coords = np.mgrid[:size, :size]
        y_coords = y_coords + 0.5
        x_coords = x_coords + 0.5
        
        # For each center, check if pixel is inside the octagon
        # An octagon can be defined as the intersection of a square and a circle? 
        # No, let's define it via distance to the "core" square and the "corner" circles.
        # A regular octagon approximated by a square of side S with corners cut off by arcs of radius R.
        # Actually, simpler: define by max(|x|, |y|) <= L - R and (|x|-R)^2 + (|y|-R)^2 <= R^2 for corners?
        # Let's use the standard octagon equation:
        # For a symmetric octagon centered at 0,0:
        # The shape is defined by |x| + |y| <= K (diamond) intersected with x^2+y^2 <= R^2? No.
        
        # Let's stick to a robust definition:
        # An octagon with horizontal/vertical symmetry can be defined by:
        # 1. If inside the central square |x|<=s/2, |y|<=s/2 -> Inside
        # 2. If in the corner region |x|>s/2 or |y|>s/2:
        #    Check distance to the corner centers (s/2, s/2), (-s/2, s/2), etc.
        #    If dist <= r_corner -> Inside
        
        s = side_len
        r_c = corner_radius
        
        for cx, cy in centers:
            dx = np.abs(x_coords - cx)
            dy = np.abs(y_coords - cy)
            
            # Inside the central square part
            mask_center = (dx <= s / 2.0) & (dy <= s / 2.0)
            
            # Inside the corner circles
            # The corners of the square are at (s/2, s/2) relative to center.
            # We place a circle of radius r_c at these corners to smooth them.
            # Wait, if we just union the square and the circles, we get a shape with bumps.
            # We want a shape with *rounded* corners.
            # So we take the square and "cut" the corners, but replace them with arcs.
            # Effectively, the boundary is:
            # If dx < s/2 and dy < s/2: inside.
            # Else if (dx >= s/2 and dy >= s/2):
            #    Let dx_c = dx - s/2, dy_c = dy - s/2.
            #    If dx_c^2 + dy_c^2 <= r_c^2: inside.
            # This creates a shape that is square in the middle and circular at the corners.
            
            corner_mask = (dx >= s / 2.0) & (dy >= s / 2.0)
            dx_c = dx - s / 2.0
            dy_c = dy - s / 2.0
            mask_corner = corner_mask & (dx_c * dx_c + dy_c * dy_c <= r_c * r_c)
            
            inside = mask_center | mask_corner
            
            cell[inside] = 1

        return cell

    def solve_bands(cell, n_pw=12):
        """
        Fast approximate band solver for TM polarization.
        """
        size = len(cell)
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute 2D FFT of inv_eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]
        
        # Plane wave indices
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))
        
        n_pw_actual = len(g_pairs)
        
        # K-points in the irreducible BZ
        # Gamma, X, M, and some interior points
        k_pts = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.0),
            (0.25, 0.25),
            (0.5, 0.25),
            (0.125, 0.125),
            (0.375, 0.125),
        ]
        
        all_omega4 = []
        all_omega5 = []
        
        for kx, ky in k_pts:
            H = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            
            for i, (mi, ni) in enumerate(g_pairs):
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    # H[i, j] = eps_inv[Gi-Gj] * (k-Gj)^2
                    # Note: This matrix form assumes a specific ordering. 
                    # For TM, the operator is (1/eps) (k-G)^2.
                    # The matrix element is sum_G'' (1/eps)_{G-G''} (k-G'')^2 delta_{G'' G'}?
                    # Actually, the plane wave expansion of 1/eps(r) is sum_G (1/eps)_G e^{i G r}.
                    # The term (k-G)^2 c_G is diagonal in G.
                    # The multiplication by 1/eps is a convolution.
                    # So H[G, G'] = (1/eps)_{G-G'} (k-G')^2.
                    # This is what is implemented above.
                    H[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)
            
            # The matrix H is not Hermitian in the standard basis if (1/eps) is real but the 
            # weights (k-G')^2 are real. Wait.
            # H[i,j] = c_{Gi-Gj} * w_j
            # H[j,i] = c_{Gj-Gi} * w_i
            # c_{-k} = c_k^*. So H[j,i] = H[i,j]^* * (w_i/w_j)? No.
            # w_i = (k-Gi)^2, w_j = (k-Gj)^2.
            # H[j,i] = c_{Gj-Gi} w_i = c_{-(Gi-Gj)} w_i = c_{Gi-Gj}^* w_i.
            # H[i,j]^* = (c_{Gi-Gj} w_j)^* = c_{Gi-Gj}^* w_j.
            # So H[j,i] = H[i,j]^* * (w_i/w_j).
            # This means H is similar to a Hermitian matrix via diagonal scaling D.
            # D^{-1} H D is Hermitian if D_j = w_j?
            # Let M = D^{-1} H D. M[i,j] = H[i,j] D_j / D_i.
            # M[i,j] = c_{Gi-Gj} w_j * D_j / D_i.
            # If we choose D_j = 1/sqrt(w_j), then M[i,j] = c_{Gi-Gj} sqrt(w_j/w_i).
            # M[j,i] = c_{Gj-Gi} sqrt(w_i/w_j) = c_{Gi-Gj}^* sqrt(w_i/w_j).
            # M[i,j]^* = c_{Gi-Gj}^* sqrt(w_j/w_i).
            # So M[j,i] = M[i,j]^* * (w_i/w_j) * (w_j/w_i) ... wait.
            # M[j,i] = c_{Gi-Gj}^* sqrt(w_i/w_j).
            # M[i,j]^* = c_{Gi-Gj}^* sqrt(w_j/w_i).
            # They are not equal unless w_i = w_j.
            
            # Actually, the standard formulation uses a Hermitian matrix for TM.
            # The equation is: sum_{G'} (1/eps)_{G-G'} (k-G')^2 c_{G'} = omega^2 c_G
            # This is a generalized eigenvalue problem? No, it's a standard one if we define the vector properly.
            # Let c_G be the coefficients.
            # Often, one uses the expansion of E(r) = sum c_G e^{i(k+G)r}.
            # The operator is - (1/eps) div (grad E).
            # In 2D TM: (1/eps) (k-G)^2 c_G.
            # The matrix A where A_{G, G'} = (1/eps)_{G-G'} (k-G')^2 is NOT Hermitian.
            # However, A is similar to a Hermitian matrix.
            # Let B_{G, G'} = (1/eps)_{G-G'} sqrt((k-G)^2 (k-G')^2).
            # B is Hermitian because (1/eps)_{-k} = (1/eps)_k^*.
            # The eigenvalues of A are the same as B.
            # So we should solve for eigenvalues of B.
            
            H_herm = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            for i, (mi, ni) in enumerate(g_pairs):
                kx_i = kx - mi
                ky_i = ky - ni
                w_i = np.sqrt(kx_i * kx_i + ky_i * ky_i)
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_j = kx - mj
                    ky_j = ky - nj
                    w_j = np.sqrt(kx_j * kx_j + ky_j * ky_j)
                    H_herm[i, j] = eps_val * w_i * w_j
            
            try:
                eigenvalues = np.linalg.eigvalsh(H_herm)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    all_omega4.append(eigenvalues[3])
                    all_omega5.append(eigenvalues[4])
            except:
                pass
        
        if not all_omega4:
            return -1.0
        
        max_omega4 = np.max(all_omega4)
        min_omega5 = np.min(all_omega5)
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # Initial guess
    # Based on previous findings, side_len around 6-8, corner_radius around 1-3, offset 0
    best_score = -1e10
    best_params = (6.0, 2.0, 0.0)
    
    # Coarse search
    side_lens = [5.0, 6.0, 7.0, 8.0]
    corner_radii = [1.0, 1.5, 2.0, 2.5, 3.0]
    offsets = [-1.0, 0.0, 1.0]
    
    for s in side_lens:
        for r_c in corner_radii:
            for off in offsets:
                if time.time() > deadline:
                    break
                cell = make_cell_octagon(s, r_c, off)
                score = solve_bands(cell, n_pw=10)
                if score > best_score:
                    best_score = score
                    best_params = (s, r_c, off)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    s_best, r_c_best, off_best = best_params
    
    # Local refinement
    if time.time() < deadline - 20:
        step_s = 0.5
        step_r = 0.5
        step_off = 0.5
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for ds in [-step_s, 0, step_s]:
                for dr in [-step_r, 0, step_r]:
                    for do in [-step_off, 0, step_off]:
                        if time.time() > deadline:
                            break
                        s_new = s_best + ds
                        r_c_new = r_c_best + dr
                        off_new = off_best + do
                        
                        if s_new < 3 or s_new > 12 or r_c_new < 0 or r_c_new > s_new/2:
                            continue
                        
                        cell = make_cell_octagon(s_new, r_c_new, off_new)
                        score = solve_bands(cell, n_pw=12)
                        if score > best_score:
                            best_score = score
                            best_params = (s_new, r_c_new, off_new)
                            s_best, r_c_best, off_best = s_new, r_c_new, off_new
                            improved = True
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            
            # Reduce step size if we are still improving but want to be finer
            step_s *= 0.8
            step_r *= 0.8
            step_off *= 0.8

    # Create final cell
    s_final, r_c_final, off_final = best_params
    final_cell = make_cell_octagon(s_final, r_c_final, off_final)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
