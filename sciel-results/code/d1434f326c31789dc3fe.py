import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s budget

    def build_cell(quad):
        """
        Builds a 32x32 cell from a 16x16 quadrant.
        quad[i][j] maps to cell[16+i][16+j].
        Symmetries:
        cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
        cell[i][j] = cell[j][i] (Transpose symmetry for C4v)
        """
        cell = np.zeros((size, size), dtype=int)
        # Fill top-right quadrant (relative to center)
        # Indices 16..31
        for i in range(16):
            for j in range(16):
                val = quad[i, j]
                i1, j1 = 16 + i, 16 + j
                cell[i1, j1] = val
                # Reflect across vertical axis (x -> 31-x)
                cell[31 - i1, j1] = val
                # Reflect across horizontal axis (y -> 31-y)
                cell[i1, 31 - j1] = val
                # Reflect across both
                cell[31 - i1, 31 - j1] = val
        
        # Enforce C4v (Transpose symmetry)
        # The above construction ensures C2v.
        # To ensure C4v, we must ensure cell == cell.T.
        # This implies quad must be symmetric in a specific way or we just symmetrize the result.
        # Symmetrizing is safer.
        cell = (cell + cell.T) // 2
        # Re-apply strict C2v to be sure
        cell = (cell + cell[:, ::-1]) // 2
        cell = (cell + cell[::-1, :]) // 2
        # Final binary
        cell[cell > 0.5] = 1
        cell[cell <= 0.5] = 0
        return cell

    def get_eps_coeffs(cell, n_max):
        """
        Compute exact Fourier coefficients for G in [-n_max, n_max]^2.
        """
        gx_vals = np.arange(-n_max, n_max + 1)
        n_g = len(gx_vals)
        
        # Pixel centers
        pos = (np.arange(size) + 0.5) / size
        
        # Sinc terms: sin(pi*g/size) / (pi*g/size)
        def sinc(g):
            if g == 0:
                return 1.0
            return np.sin(np.pi * g / size) / (np.pi * g / size)
        
        Sinc = np.array([sinc(g) for g in gx_vals])
        
        # Exponential terms: exp(-i 2 pi g * pos)
        # Shape: (n_g, size)
        Exp = np.exp(-1j * 2 * np.pi * np.outer(gx_vals, pos))
        
        # cell is (size, size)
        # Sum over j for each i and gy
        # RowSum[i, gy_idx] = sum_j cell[i, j] Exp[gy_idx, j]
        # cell is real, Exp is complex
        RowSum = cell.astype(complex) @ Exp.T
        
        # Sum over i for each gx and gy
        # FinalSum[gx_idx, gy_idx] = sum_i Exp[gx_idx, i] RowSum[i, gy_idx]
        FinalSum = Exp @ RowSum
        
        # Apply Sinc and scaling
        SincMat = np.outer(Sinc, Sinc)
        eps_coeffs = SincMat * FinalSum / (size * size)
        
        return eps_coeffs, gx_vals

    def calc_gap(cell, n_pw=5):
        """
        Calculate gap between band 4 and 5.
        """
        # We need eps_coeffs for differences up to 2*n_pw
        eps_coeffs, gx_vals = get_eps_coeffs(cell, 2 * n_pw)
        # Map (gx, gy) to index in eps_coeffs
        # gx_vals is [-n, ..., n]
        def get_eps(gx, gy):
            if abs(gx) <= 2 * n_pw and abs(gy) <= 2 * n_pw:
                idx_gx = gx + 2 * n_pw
                idx_gy = gy + 2 * n_pw
                return eps_coeffs[idx_gx, idx_gy]
            return 0.0

        # G vectors for the basis
        G_list = [(gx, gy) for gx in range(-n_pw, n_pw + 1) for gy in range(-n_pw, n_pw + 1)]
        N_pw = len(G_list)
        
        # K points in irreducible BZ
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.25),
            (0.333, 0.333)
        ]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            # Build PWE matrix
            # M is 2xN_pw x 2xN_pw
            # We can vectorize the construction
            kx_gs = np.array([kx + gx for gx, gy in G_list])
            ky_gs = np.array([ky + gy for gx, gy in G_list])
            k_sqs = kx_gs**2 + ky_gs**2
            
            # Precompute eps matrix
            # Eps[i, j] = eps(G_i - G_j)
            EpsMat = np.zeros((N_pw, N_pw), dtype=complex)
            for i in range(N_pw):
                for j in range(N_pw):
                    dgx = G_list[i][0] - G_list[j][0]
                    dgy = G_list[i][1] - G_list[j][1]
                    EpsMat[i, j] = get_eps(dgx, dgy)
            
            # Construct M
            # M[2i, 2j] = (k_sq_i - kx_i kx_j) * Eps[i, j]
            # M[2i, 2j+1] = -kx_i ky_j * Eps[i, j]
            # M[2i+1, 2j] = -ky_i kx_j * Eps[i, j]
            # M[2i+1, 2j+1] = (k_sq_i - ky_i ky_j) * Eps[i, j]
            
            # Vectorized construction
            kx_i = kx_gs[:, None]
            ky_i = ky_gs[:, None]
            kx_j = kx_gs[None, :]
            ky_j = ky_gs[None, :]
            k_sq_i = k_sqs[:, None]
            
            M_00 = (k_sq_i - kx_i * kx_j) * EpsMat
            M_01 = -kx_i * ky_j * EpsMat
            M_10 = -ky_i * kx_j * EpsMat
            M_11 = (k_sq_i - ky_i * ky_j) * EpsMat
            
            # Interleave rows and columns
            M = np.zeros((2 * N_pw, 2 * N_pw), dtype=complex)
            M[0::2, 0::2] = M_00
            M[0::2, 1::2] = M_01
            M[1::2, 0::2] = M_10
            M[1::2, 1::2] = M_11
            
            # Solve
            eigvals = np.linalg.eigvalsh(M)
            eigvals = eigvals[eigvals > 1e-12]
            
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Search Strategy
    best_score = -1.0
    best_quad = None
    
    # Generate candidates
    candidates = []
    
    # 1. Circular Rods
    for r in np.linspace(0.05, 0.45, 15):
        quad = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                # Map to cell coords centered at 0
                x = (16 + i + 0.5) / 32 - 0.5
                y = (16 + j + 0.5) / 32 - 0.5
                if x*x + y*y < r*r:
                    quad[i, j] = 1
        candidates.append(quad)
        
    # 2. Air Holes (Inverse Rod)
    for r in np.linspace(0.05, 0.45, 15):
        quad = np.ones((16, 16))
        for i in range(16):
            for j in range(16):
                x = (16 + i + 0.5) / 32 - 0.5
                y = (16 + j + 0.5) / 32 - 0.5
                if x*x + y*y < r*r:
                    quad[i, j] = 0
        candidates.append(quad)
        
    # 3. Super-ellipses
    for n in [1.5, 2.0, 3.0, 4.0]:
        for r in np.linspace(0.1, 0.4, 8):
            quad = np.zeros((16, 16))
            for i in range(16):
                for j in range(16):
                    x = (16 + i + 0.5) / 32 - 0.5
                    y = (16 + j + 0.5) / 32 - 0.5
                    if abs(x)**n + abs(y)**n < r**n:
                        quad[i, j] = 1
            candidates.append(quad)
            
    # Evaluate candidates
    for quad in candidates:
        if time.time() > deadline:
            break
        cell = build_cell(quad)
        score = calc_gap(cell, n_pw=5)
        if score > best_score:
            best_score = score
            best_quad = quad.copy()
            
    # Local Refinement
    if best_quad is not None and time.time() < deadline - 30:
        current_quad = best_quad.copy()
        current_score = best_score
        # Try flipping pixels in a few random locations
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            i, j = np.random.randint(0, 16, 2)
            original_val = current_quad[i, j]
            current_quad[i, j] = 1 - original_val
            cell = build_cell(current_quad)
            score = calc_gap(cell, n_pw=5)
            if score > current_score:
                current_score = score
                best_score = score
                best_quad = current_quad.copy()
            else:
                current_quad[i, j] = original_val
                
    # If we didn't find a positive gap, return a known good simple design (e.g. medium rod)
    if best_score <= 0:
        # Fallback: Moderate rod
        quad = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                x = (16 + i + 0.5) / 32 - 0.5
                y = (16 + j + 0.5) / 32 - 0.5
                if x*x + y*y < 0.25*0.25: # r=0.25
                    quad[i, j] = 1
        best_quad = quad
        best_score = 0.0 # It might still be negative, but it's a valid structure
        
    final_cell = build_cell(best_quad)
    return final_cell.tolist()
