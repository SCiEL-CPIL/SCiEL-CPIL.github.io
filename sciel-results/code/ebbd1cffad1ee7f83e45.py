"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill
    
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
    n_valid = len(valid_rows)
    
    # Independent pixels: Upper triangle of top-left 16x16 quadrant
    # i, j in [0, 15], i >= j
    indep_coords = [(i, j) for i in range(16) for j in range(i + 1)]
    n_indep = len(indep_coords)
    
    def build_cell_from_indep(indep_vals):
        """Construct full 32x32 cell from independent triangle values."""
        cell = np.zeros((size, size), dtype=int)
        for k, (i, j) in enumerate(indep_coords):
            val = indep_vals[k]
            # Symmetries:
            # (i, j)
            cell[i, j] = val
            # (j, i) - Transpose
            if i != j:
                cell[j, i] = val
            # (i, 31-j) - Horizontal Ref
            cell[i, size - 1 - j] = val
            if i != j:
                cell[j, size - 1 - i] = val
            # (31-i, j) - Vertical Ref
            cell[size - 1 - i, j] = val
            if i != j:
                cell[size - 1 - j, i] = val
            # (31-i, 31-j) - 180 Rot
            cell[size - 1 - i, size - 1 - j] = val
            if i != j:
                cell[size - 1 - j, size - 1 - i] = val
        return cell

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

    # Phase 1: Generate diverse initial candidates
    # We will define a few patterns and convert them to indep_vals
    
    def cell_to_indep(cell):
        """Extract independent values from a symmetric cell."""
        vals = []
        for i, j in indep_coords:
            vals.append(cell[i, j])
        return np.array(vals, dtype=int)

    initial_candidates = []
    
    # Pattern 1: Four rods
    def make_four_rods(radius, offset):
        cell = np.zeros((size, size), dtype=int)
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy <= radius * radius:
                        cell[i, j] = 1
                        break
        return cell

    # Pattern 2: Single rod
    def make_single_rod(radius):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y <= radius * radius:
                    cell[i, j] = 1
        return cell

    # Pattern 3: Cross
    def make_cross(width):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < width/2 or abs(y) < width/2:
                    cell[i, j] = 1
        return cell

    # Generate candidates
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.20, 0.25, 0.30]:
            c = make_four_rods(r, off)
            initial_candidates.append(cell_to_indep(c))
            
    for r in [0.20, 0.25, 0.30]:
        c = make_single_rod(r)
        initial_candidates.append(cell_to_indep(c))
        
    for w in [0.1, 0.2, 0.3]:
        c = make_cross(w)
        initial_candidates.append(cell_to_indep(c))
        
    # Also add random candidates
    for _ in range(10):
        initial_candidates.append(np.random.randint(0, 2, n_indep))

    # Evaluate initial candidates
    best_ratio = -np.inf
    best_indep = None
    
    for indep_vals in initial_candidates:
        if not time_ok():
            break
        cell = build_cell_from_indep(indep_vals)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_indep = indep_vals.copy()
            
    if best_indep is None:
        best_indep = np.random.randint(0, 2, n_indep)
        best_ratio = compute_band_ratio(get_fourier_coeffs(build_cell_from_indep(best_indep)))

    # Phase 2: Simulated Annealing
    current_indep = best_indep.copy()
    current_ratio = best_ratio
    best_indep_sa = current_indep.copy()
    best_ratio_sa = current_ratio
    
    T = 0.1
    T_min = 0.001
    alpha = 0.99
    max_iter = 1000
    search_start = time.time()
    search_time_limit = 190.0
    
    iter_count = 0
    while time.time() - search_start < search_time_limit and time_ok() and T > T_min:
        # Pick a random independent pixel to flip
        idx = np.random.randint(0, n_indep)
        new_indep = current_indep.copy()
        new_indep[idx] = 1 - new_indep[idx]
        
        new_cell = build_cell_from_indep(new_indep)
        new_coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(new_coeffs)
        
        delta = new_ratio - current_ratio
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_indep = new_indep
            current_ratio = new_ratio
            if current_ratio > best_ratio_sa:
                best_ratio_sa = current_ratio
                best_indep_sa = current_indep.copy()
        
        T *= alpha
        iter_count += 1
        
        # If we have plenty of time left, we can restart with higher T if stuck
        # But for now, just continue cooling
        
    # Final check: if SA didn't improve, use the best initial
    if best_ratio_sa > best_ratio:
        best_indep = best_indep_sa
        best_ratio = best_ratio_sa

    final_cell = build_cell_from_indep(best_indep)
    return final_cell.tolist()
# EVOLVE-BLOCK-END
