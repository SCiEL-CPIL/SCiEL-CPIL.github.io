"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

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

    def compute_fourier_coefficients(cell, max_order):
        """Compute Fourier coefficients of 1/epsilon using FFT for speed."""
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        # Use 2D FFT to get all coefficients at once
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                # Map negative indices to positive using FFT convention
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def compute_fourier_coeffs_matrix(cell, max_order):
        """Compute Fourier coefficients as a numpy matrix indexed by (m+max_order, n+max_order)."""
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        # Build matrix where index (m+max_order, n+max_order) = coeffs[(m,n)]
        mat = np.zeros((2 * max_order + 1, 2 * max_order + 1), dtype=complex)
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                mat[m + max_order, n + max_order] = fft_2d[m % size, n % size]
        return mat

    def approximate_gap_ratio(cell, max_order=6, g_range=4):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        Uses a larger basis and more k-points for better accuracy.
        """
        coeffs_mat = compute_fourier_coeffs_matrix(cell, max_order)
        
        # G vectors in the basis - use larger range for better accuracy
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # Key k-points in irreducible BZ (triangle: 0<=ky<=kx<=0.5)
        # Include high-symmetry points and interior points
        k_points = []
        # Along kx axis (ky=0)
        for kx in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            k_points.append((kx, 0.0))
        # Along ky=kx diagonal
        for kx in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            k_points.append((kx, kx))
        # Interior points of the triangle
        for kx in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            for ky in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44]:
                if ky <= kx:
                    k_points.append((kx, ky))
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        # Precompute G vector arrays for speed
        gx = np.array([g[0] for g in g_vectors], dtype=float)
        gy = np.array([g[1] for g in g_vectors], dtype=float)
        
        for kx, ky in k_points:
            kx_total = kx + gx
            ky_total = ky + gy
            k_sq = kx_total**2 + ky_total**2
            
            # Build Hamiltonian matrix
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

    def generate_single_rod(radius_frac):
        """Central circular rod."""
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

    def generate_ring(inner_frac, outer_frac):
        """Circular ring (dielectric annulus)."""
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                if r_in * r_in <= r2 < r_out * r_out:
                    cell[i][j] = 1
        return cell

    def generate_four_rods(radius_frac):
        """Four circular rods at corners (D4 symmetric)."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        # Rod centers at (size/4, size/4) and symmetric positions
        centers = [(size/4.0, size/4.0), (size*3/4.0, size/4.0),
                   (size/4.0, size*3/4.0), (size*3/4.0, size*3/4.0)]
        for i in range(size):
            for j in range(size):
                for cx, cy in centers:
                    x = (j + 0.5) - cx
                    y = (i + 0.5) - cy
                    if x * x + y * y < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_cross(arm_width_frac, arm_length_frac):
        """Cross shape (plus sign)."""
        cell = [[0] * size for _ in range(size)]
        w = int(arm_width_frac * size)
        l = int(arm_length_frac * size)
        cx = size // 2
        cy = size // 2
        half_w = w // 2
        for i in range(size):
            for j in range(size):
                dx = abs(j - cx)
                dy = abs(i - cy)
                if (dx <= half_w and dy <= l) or (dy <= half_w and dx <= l):
                    cell[i][j] = 1
        return cell

    def generate_2x2_supercell(rod_radius_frac):
        """
        Generate a 2x2 supercell arrangement of rods within the 32x32 unit cell.
        This creates four smaller rods arranged in a 2x2 pattern, which can open
        higher-order band gaps.
        """
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        # Centers of the four sub-rods in a 2x2 arrangement
        # Offset from center by size/4
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

    def generate_2x2_supercell_ring(inner_frac, outer_frac):
        """
        Generate a 2x2 supercell arrangement of rings within the 32x32 unit cell.
        """
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
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
                    r2 = dx * dx + dy * dy
                    if r_in * r_in <= r2 < r_out * r_out:
                        cell[i][j] = 1
                        break
        return cell

    def generate_hexagonal_like(rod_radius_frac, spacing_frac):
        """
        Generate a pattern with rods arranged in a hexagonal-like D4 symmetric pattern.
        """
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        s = spacing_frac * size
        
        # Center rod
        centers = [(cx, cy)]
        # Six surrounding rods in hexagonal pattern, but adjust for D4 symmetry
        # Use 8 positions for D4 symmetry
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = math.radians(angle)
            cx_r = cx + s * math.cos(rad)
            cy_r = cy + s * math.sin(rad)
            centers.append((cx_r, cy_r))
            
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx_r, cy_r in centers:
                    dx = x - cx_r
                    dy = y - cy_r
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_diamond_pattern(arm_width_frac, size_frac):
        """Diamond (rotated square) pattern with D4 symmetry."""
        cell = [[0] * size for _ in range(size)]
        w = int(arm_width_frac * size)
        s = int(size_frac * size)
        cx = size / 2.0
        cy = size / 2.0
        half_w = w / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                # Diamond: |x|/a + |y|/b <= 1
                if abs(x) / s + abs(y) / s <= 1.0:
                    # Check if within the arm width from the diamond edge
                    dist_from_edge = 1.0 - (abs(x) / s + abs(y) / s)
                    if dist_from_edge * s >= half_w or abs(x) / s + abs(y) / s <= 0.3:
                        cell[i][j] = 1
        return cell

    def generate_checkerboard_block(block_size):
        """Block pattern with D4 symmetry."""
        cell = [[0] * size for _ in range(size)]
        bs = block_size
        for i in range(size):
            for j in range(size):
                # Map to block coordinates
                bi = i // bs
                bj = j // bs
                # D4 symmetric pattern: fill blocks where (bi+bj) is even
                if (bi + bj) % 2 == 0:
                    cell[i][j] = 1
        return cell

    def generate_concentric_rings(radii_fracs):
        """Multiple concentric rings."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                for k in range(len(radii_fracs) - 1):
                    r_in = radii_fracs[k] * size
                    r_out = radii_fracs[k + 1] * size
                    if r_in * r_in <= r2 < r_out * r_out:
                        cell[i][j] = 1
                        break
        return cell

    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.10, 0.25), (0.20, 0.30)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    # Crosses
    for w, l in [(0.15, 0.40), (0.20, 0.35), (0.25, 0.30), (0.15, 0.35)]:
        candidates.append(generate_cross(w, l))
    
    # Checkerboard blocks
    for bs in [4, 6, 8]:
        candidates.append(generate_checkerboard_block(bs))
    
    # Concentric rings
    for radii in [(0.0, 0.20), (0.0, 0.25, 0.40), (0.15, 0.30, 0.45)]:
        candidates.append(generate_concentric_rings(radii))
    
    # 2x2 supercell arrangements (key improvement based on knowledge)
    for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
        candidates.append(generate_2x2_supercell(r))
    
    # 2x2 supercell with rings
    for inner, outer in [(0.05, 0.15), (0.08, 0.18), (0.10, 0.20), (0.06, 0.16)]:
        candidates.append(generate_2x2_supercell_ring(inner, outer))
    
    # Hexagonal-like patterns
    for r, s in [(0.08, 0.25), (0.10, 0.25), (0.12, 0.30), (0.08, 0.30), (0.10, 0.30)]:
        candidates.append(generate_hexagonal_like(r, s))
    
    # Diamond patterns
    for w, s in [(0.15, 0.40), (0.20, 0.35), (0.15, 0.30), (0.25, 0.40)]:
        candidates.append(generate_diamond_pattern(w, s))
    
    # Add more 2x2 supercell variants with finer radii
    for r in [0.09, 0.11, 0.13, 0.15, 0.17, 0.19]:
        candidates.append(generate_2x2_supercell(r))
    
    # Add more 2x2 supercell ring variants
    for inner, outer in [(0.07, 0.17), (0.09, 0.19), (0.11, 0.21), (0.08, 0.16)]:
        candidates.append(generate_2x2_supercell_ring(inner, outer))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # If we have time, do a second pass with the best candidate and nearby variants
    if time.time() - start_time < deadline - 20 and best_score > 0:
        base = best_cell
        # Try small perturbations around the best design
        for flip_i in range(0, size, 1):
            for flip_j in range(0, size, 1):
                if time.time() - start_time > deadline - 15:
                    break
                
                # Try flipping single pixel
                cell_trial = [row[:] for row in base]
                cell_trial[flip_i][flip_j] = 1 - base[flip_i][flip_j]
                cell_sym = enforce_d4_symmetry(cell_trial)
                if cell_sym != base:
                    score = approximate_gap_ratio(cell_sym)
                    if score > best_score + 1e-6:
                        best_score = score
                        best_cell = cell_sym
                        base = cell_sym
                
            if time.time() - start_time > deadline - 15:
                break
    
    # If time remains, try local perturbations around best design
    if time.time() - start_time < deadline - 15:
        base = best_cell
        improved = True
        # Use a smaller basis for local search to speed it up
        def approximate_gap_ratio_fast(cell):
            return approximate_gap_ratio(cell, max_order=4, g_range=3)
        
        while improved and time.time() - start_time < deadline - 10:
            improved = False
            best_local = best_score
            best_local_cell = base
            
            # Try flipping individual pixels and small patches
            for flip_i in range(0, size, 1):
                for flip_j in range(0, size, 1):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    # Try flipping single pixel
                    cell_trial = [row[:] for row in base]
                    cell_trial[flip_i][flip_j] = 1 - base[flip_i][flip_j]
                    cell_sym = enforce_d4_symmetry(cell_trial)
                    if cell_sym != base:
                        score = approximate_gap_ratio_fast(cell_sym)
                        if score > best_local + 1e-6:
                            best_local = score
                            best_local_cell = cell_sym
                            improved = True
                    
                    # Try flipping 3x3 patch
                    cell_trial2 = [row[:] for row in base]
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni = (flip_i + di) % size
                            nj = (flip_j + dj) % size
                            cell_trial2[ni][nj] = 1 - base[ni][nj]
                    cell_sym2 = enforce_d4_symmetry(cell_trial2)
                    if cell_sym2 != base and cell_sym2 != cell_sym:
                        score2 = approximate_gap_ratio_fast(cell_sym2)
                        if score2 > best_local + 1e-6:
                            best_local = score2
                            best_local_cell = cell_sym2
                            improved = True
                
                if time.time() - start_time > deadline - 5:
                    break
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base = best_cell
    
    # Final high-precision polish: use larger basis for the best candidate
    # and its immediate neighbors to get a more accurate final answer
    if time.time() - start_time < deadline - 5:
        final_base = best_cell
        final_best = best_score
        final_best_cell = final_base
        
        # Evaluate the best cell with higher accuracy
        def approximate_gap_ratio_high(cell):
            return approximate_gap_ratio(cell, max_order=8, g_range=6)
        
        # Re-evaluate best cell with high precision
        score_hp = approximate_gap_ratio_high(final_base)
        if score_hp > final_best:
            final_best = score_hp
        
        # Try single-pixel flips with high-precision evaluation
        for flip_i in range(0, size, 1):
            for flip_j in range(0, size, 1):
                if time.time() - start_time > deadline - 2:
                    break
                cell_trial = [row[:] for row in final_base]
                cell_trial[flip_i][flip_j] = 1 - final_base[flip_i][flip_j]
                cell_sym = enforce_d4_symmetry(cell_trial)
                if cell_sym != final_base:
                    score_hp = approximate_gap_ratio_high(cell_sym)
                    if score_hp > final_best + 1e-6:
                        final_best = score_hp
                        final_best_cell = cell_sym
                        final_base = cell_sym
            
            if time.time() - start_time > deadline - 2:
                break
        
        best_cell = final_best_cell
        best_score = final_best
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
