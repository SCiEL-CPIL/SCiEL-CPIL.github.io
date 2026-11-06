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
    # Evaluator kills at 230s. We need to return well before.
    # Let's target 120s for safety, leaving buffer for overhead.
    deadline = start_time + 115 

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
                # Use majority vote or average. Average is smoother for optimization.
                # But for binary, > 0.5 is standard.
                avg = sum(vals) / len(vals)
                final_cell[i][j] = 1 if avg >= 0.5 else 0
        return final_cell

    def compute_fourier_coefficients_fast(cell, g_range):
        """
        Compute Fourier coefficients of 1/epsilon.
        cell: 2D list of 0/1
        g_range: max order for G vectors
        Returns: dict mapping (m, n) to coeff
        """
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        # Use 2D FFT. The coefficients are C_{mn} = (1/N^2) sum_{x,y} f(x,y) exp(-i 2 pi (mx+ny)/N)
        # np.fft.fft2 computes sum_{x,y} f(x,y) exp(-i 2 pi (mx+ny)/N)
        # So C_{mn} = fft2[ (m mod N), (n mod N) ] / N^2
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-g_range, g_range + 1):
            for n in range(-g_range, g_range + 1):
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def evaluate_gap(cell, g_range, k_points):
        """
        Calculate gap-midgap ratio for bands 4 and 5.
        """
        coeffs = compute_fourier_coefficients_fast(cell, g_range)
        
        # G vectors
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    eps_inv_G = coeffs.get((m_diff, n_diff), 0.0)
                    H[i, j] = k_sq * eps_inv_G
            
            # Solve eigenvalue problem
            # H is Hermitian
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

    # K-points for high fidelity (matching evaluator's 30 points roughly)
    k_points_high = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1),
        (0.1, 0.2), (0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2),
        (0.1, 0.3), (0.2, 0.3), (0.3, 0.3), (0.4, 0.3), (0.5, 0.3),
        (0.1, 0.4), (0.2, 0.4), (0.3, 0.4), (0.4, 0.4), (0.5, 0.4),
        (0.5, 0.5)
    ]
    
    # Lower fidelity for local search
    # Use same k-points but smaller G range? Or fewer k-points?
    # Let's use same k-points but G_range=6 (13x13=169 vectors) vs High G_range=8 (17x17=289 vectors)
    # Actually, 289 is the evaluator's size. Let's use 289 for high, 169 for low.
    # Wait, 289 is 17x17, so g_range=8.
    # 169 is 13x13, so g_range=6.
    
    def generate_candidate_single_rod(radius_frac):
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

    def generate_candidate_ring(inner_frac, outer_frac):
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

    def generate_candidate_2x2_rods(radius_frac):
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
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

    def generate_candidate_2x2_rings(inner_frac, outer_frac):
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

    def generate_candidate_cross(arm_width_frac, arm_length_frac):
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

    def generate_candidate_diamond(size_frac, width_frac):
        cell = [[0] * size for _ in range(size)]
        s = size_frac * size
        w = width_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                # Diamond boundary
                if abs(x) / s + abs(y) / s <= 1.0:
                    # Inside diamond
                    # Check if near edge or center? Let's just fill the diamond.
                    cell[i][j] = 1
        return cell

    def generate_candidate_multi_scale(inner_r, mid_r, outer_r, pattern_type):
        """
        Multi-scale candidate:
        pattern_type 0: Ring at inner, Rod at mid, Ring at outer
        pattern_type 1: Rod at inner, Ring at mid, Rod at outer
        """
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        r1 = inner_r * size
        r2 = mid_r * size
        r3 = outer_r * size
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2_val = x * x + y * y
                r = math.sqrt(r2_val)
                
                is_dielectric = False
                if pattern_type == 0:
                    # Inner rod
                    if r < r1: is_dielectric = True
                    # Mid ring
                    elif r1 <= r < r2: is_dielectric = False
                    elif r2 <= r < r3: is_dielectric = True
                else:
                    # Inner rod
                    if r < r1: is_dielectric = True
                    # Mid air
                    elif r1 <= r < r2: is_dielectric = False
                    # Outer rod
                    elif r2 <= r < r3: is_dielectric = True
                    
                if is_dielectric:
                    cell[i][j] = 1
        return cell

    # Generate diverse candidates
    candidates = []
    
    # Single rods
    for r in np.linspace(0.1, 0.4, 10):
        candidates.append(generate_candidate_single_rod(r))
        
    # Rings
    for inner in np.linspace(0.1, 0.3, 5):
        for outer in np.linspace(inner + 0.05, 0.5, 5):
            if outer < 0.5:
                candidates.append(generate_candidate_ring(inner, outer))

    # 2x2 Rods
    for r in np.linspace(0.05, 0.2, 10):
        candidates.append(generate_candidate_2x2_rods(r))
        
    # 2x2 Rings
    for inner in np.linspace(0.02, 0.1, 5):
        for outer in np.linspace(inner + 0.02, 0.25, 5):
            if outer < 0.25:
                candidates.append(generate_candidate_2x2_rings(inner, outer))

    # Crosses
    for w in [0.1, 0.2, 0.3]:
        for l in [0.3, 0.4, 0.5]:
            candidates.append(generate_candidate_cross(w, l))
            
    # Diamonds
    for s in [0.3, 0.4, 0.5]:
        candidates.append(generate_candidate_diamond(s, 0.0))

    # Multi-scale (Wavelet-like superpositions)
    # These capture features at different radial scales
    for i_r in [0.05, 0.1, 0.15]:
        for m_r in [0.2, 0.25, 0.3]:
            for o_r in [0.35, 0.4, 0.45]:
                if i_r < m_r < o_r:
                    candidates.append(generate_candidate_multi_scale(i_r, m_r, o_r, 0))
                    candidates.append(generate_candidate_multi_scale(i_r, m_r, o_r, 1))

    # Enforce symmetry and deduplicate
    unique_candidates = []
    seen_hashes = set()
    for cell in candidates:
        sym_cell = enforce_d4_symmetry(cell)
        h = hash(str(sym_cell))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_candidates.append(sym_cell)
            
    # Evaluate candidates with high fidelity
    # Use g_range=8 (17x17=289 planes)
    high_g_range = 8
    best_score = -10.0
    best_cell = unique_candidates[0]
    
    # Sort candidates by a quick low-fidelity score first?
    # Or just evaluate all?
    # 100-200 candidates. 289x289 eig is ~50ms? 200 * 50ms = 10s.
    # We have 115s. This is feasible.
    
    for cell in unique_candidates:
        if time.time() > deadline - 30: # Leave 30s for local search
            break
        score = evaluate_gap(cell, high_g_range, k_points_high)
        if score > best_score:
            best_score = score
            best_cell = cell

    # Local search on the best candidate
    # Use lower fidelity for speed: g_range=6 (13x13=169 planes)
    low_g_range = 6
    current_cell = best_cell
    current_score = best_score
    
    # Hill Climbing with random restarts if stuck?
    # Just simple hill climbing with larger patches.
    improved = True
    max_local_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_local_iters and time.time() < deadline - 10:
        improved = False
        iter_count += 1
        best_local_score = current_score
        best_local_cell = current_cell
        
        # Try flipping pixels
        # To speed up, we can try flipping only pixels in the "active" region or random subset
        # But 32x32 is small. Let's try all pixels but with a larger patch size (3x3 or 5x5)
        # Flipping 1 pixel is very local. Flipping 3x3 is better.
        
        for i in range(0, size, 2): # Step 2 to reduce iterations
            for j in range(0, size, 2):
                if time.time() > deadline - 5:
                    break
                
                # Try flipping 3x3 patch centered at i,j
                cell_trial = [row[:] for row in current_cell]
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            cell_trial[ni][nj] = 1 - current_cell[ni][nj]
                            
                cell_sym = enforce_d4_symmetry(cell_trial)
                
                # Quick check if changed
                if cell_sym != current_cell:
                    score = evaluate_gap(cell_sym, low_g_range, k_points_high)
                    if score > best_local_score + 1e-6:
                        best_local_score = score
                        best_local_cell = cell_sym
                        improved = True
                        
                # Try flipping 5x5 patch
                cell_trial2 = [row[:] for row in current_cell]
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            cell_trial2[ni][nj] = 1 - current_cell[ni][nj]
                            
                cell_sym2 = enforce_d4_symmetry(cell_trial2)
                if cell_sym2 != current_cell and cell_sym2 != cell_sym:
                    score2 = evaluate_gap(cell_sym2, low_g_range, k_points_high)
                    if score2 > best_local_score + 1e-6:
                        best_local_score = score2
                        best_local_cell = cell_sym2
                        improved = True
                        
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
                
        if improved:
            current_cell = best_local_cell
            current_score = best_local_score

    # Final high-fidelity evaluation of the locally optimized cell
    final_score = evaluate_gap(current_cell, high_g_range, k_points_high)
    
    # If the local search improved the low-fidelity score, verify if it also improved high-fidelity
    # If not, keep the original best_cell
    if final_score > best_score:
        best_cell = current_cell
        best_score = final_score

    # One last check: if we have time, try a few random perturbations of the best cell
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            cell_trial = [row[:] for row in best_cell]
            # Flip 10 random pixels
            for _ in range(10):
                ri = np.random.randint(0, size)
                rj = np.random.randint(0, size)
                cell_trial[ri][rj] = 1 - cell_trial[ri][rj]
                
            cell_sym = enforce_d4_symmetry(cell_trial)
            if cell_sym != best_cell:
                score = evaluate_gap(cell_sym, high_g_range, k_points_high)
                if score > best_score:
                    best_score = score
                    best_cell = cell_sym

    return enforce_d4_symmetry(best_cell)

# EVOLVE-BLOCK-END
