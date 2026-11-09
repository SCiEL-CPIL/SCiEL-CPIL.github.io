"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Invert material assignment: Dielectric background with Air holes (based on inspiration 2).
    2. Use a superellipse-based geometric parameterization for the holes (based on inspiration 1).
    3. Optimize parameters (radius, order, offset) using a fast approximate plane-wave evaluator.
    4. The fast evaluator uses a smaller basis size (e.g., 10x10 PWs) and a subset of k-points.
    5. Enforce C4v symmetry by construction (4-fold symmetric placement of holes).
    6. Time management ensures the function returns within the 200s limit.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Parameterization ---
    # We use a superellipse-based parameterization for 4 air holes.
    # The holes are centered at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    # with an optional offset to break symmetry slightly (but we enforce C4v later).
    # Actually, to strictly enforce C4v about the center, we can just place 4 identical holes
    # symmetrically. The "offset" in inspiration 1 was shifting the centers, which might break
    # strict C4v if not done carefully. Let's stick to the symmetric centers for now,
    # but allow the radius and superellipse order to vary.
    
    # Superellipse equation: |dx/r|^n + |dy/r|^n < 1
    # dx, dy are distances from the center.
    
    def make_cell(radius, n_order, offset=0.0):
        """
        Creates a 32x32 cell with 4 air holes (0) in a dielectric background (1).
        radius: radius of the holes.
        n_order: superellipse order (2=circle, 4=square, higher=more square-like).
        offset: optional offset for the centers (must be handled carefully for symmetry).
                 For strict C4v, offset should be 0 or applied symmetrically.
                 Let's assume offset shifts the centers away from the ideal 0.25/0.75 positions
                 in a symmetric way? Or maybe just use 0 for now to be safe.
                 Actually, inspiration 1 used offset to shift the centers.
                 Let's keep offset=0 for the first pass to ensure symmetry, then refine.
        """
        cell = np.ones((size, size), dtype=int)  # Background is dielectric (1)
        
        # Centers of the 4 holes
        # Ideal centers are at (0.25, 0.25), etc.
        # With offset, we can shift them. But to maintain C4v, the shift must be symmetric.
        # Let's just use fixed centers for now: (0.25, 0.25), etc.
        cx1, cy1 = 0.25, 0.25
        cx2, cy2 = 0.75, 0.25
        cx3, cy3 = 0.25, 0.75
        cx4, cy4 = 0.75, 0.75
        
        # If offset is non-zero, we might shift them, but let's stick to 0 for symmetry.
        # The superellipse shape itself is symmetric.
        
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                is_air = False
                for cx, cy in [(cx1, cy1), (cx2, cy2), (cx3, cy3), (cx4, cy4)]:
                    dx = x - cx
                    dy = y - cy
                    if radius > 0:
                        val = abs(dx / radius) ** n_order + abs(dy / radius) ** n_order
                        if val < 1.0:
                            is_air = True
                            break
                if is_air:
                    cell[i, j] = 0  # Air hole
        
        return cell

    # --- 2. Fast Evaluator ---
    # We need a fast proxy for the band gap.
    # The evaluator uses 289 plane waves (17x17 grid in reciprocal space).
    # We will use a smaller set, e.g., 10x10 = 100 plane waves (circular cutoff).
    
    def compute_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands(cell, n_pw=10, k_pts=None):
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
            # Key points in the irreducible Brillouin zone
            k_pts = [
                (0.0, 0.0),   # Gamma
                (0.5, 0.0),   # M
                (0.5, 0.5),   # X
                (0.25, 0.0),  # Midpoint Gamma-M
                (0.25, 0.25), # Interior point
                (0.5, 0.25),  # Midpoint M-X
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

    # --- 3. Optimization Loop ---
    # Search parameters: radius, superellipse order
    best_score = -1e10
    best_params = (5.5, 2.0)
    
    # Coarse search over superellipse parameters
    radii = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]
    orders = [2.0, 3.0, 4.0, 6.0]
    
    for r in radii:
        for n_ord in orders:
            if time.time() > deadline:
                break
            cell = make_cell(r, n_ord)
            score = solve_bands(cell, n_pw=10)
            if score > best_score:
                best_score = score
                best_params = (r, n_ord)
        if time.time() > deadline:
            break
    
    # Fine search around best
    if time.time() < deadline - 40:
        r_best, n_best = best_params
        for _ in range(2):  # 2 refinement iterations
            step_r = 0.5
            step_n = 0.5
            improved = False
            for dr in [-step_r, 0, step_r]:
                for dn in [-step_n, 0, step_n]:
                    if time.time() > deadline:
                        break
                    r = r_best + dr
                    n_ord = n_best + dn
                    if r < 2 or r > 12 or n_ord < 1.5 or n_ord > 8:
                        continue
                    cell = make_cell(r, n_ord)
                    score = solve_bands(cell, n_pw=12)
                    if score > best_score:
                        best_score = score
                        best_params = (r, n_ord)
                        r_best, n_best = r, n_ord
                        improved = True
                if time.time() > deadline:
                    break
            if not improved:
                break
    
    # --- 4. Finalize Design ---
    r_final, n_final = best_params
    final_cell = make_cell(r_final, n_final)
    
    # Enforce symmetry exactly (just in case of floating point errors in thresholding)
    # The construction ensures symmetry, but let's verify/force it.
    # Symmetry: cell[i][j] == cell[size-1-i][size-1-j] == cell[j][i]
    # Our basis functions are symmetric. The thresholding might break it if values are close to threshold.
    # We can average the 4 symmetric pixels and threshold the average.
    
    sym_cell = np.zeros_like(final_cell)
    for i in range(size):
        for j in range(size):
            # Symmetric equivalents:
            # (i, j)
            # (size-1-i, j)  -> reflection x
            # (i, size-1-j)  -> reflection y
            # (size-1-i, size-1-j) -> rotation 180
            # Also transpose: (j, i), etc.
            # C4v group has 8 elements.
            # Let's just average the 4 unique ones under the group?
            # Actually, the group is generated by reflection x, reflection y, and transpose.
            # If we enforce reflection x and y, and transpose, we get C4v.
            
            # For now, let's just take the majority vote or average of the 4 main symmetric points:
            # (i,j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
            # And their transposes.
            # It's safer to just re-compute the average of the 8 symmetric points.
            
            points = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            # Unique points?
            # For a 32x32 grid, these might not all be unique.
            # Let's just sum them and divide by 8.
            s = 0
            for pi, pj in points:
                s += final_cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0
    
    # Convert to list of lists
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
