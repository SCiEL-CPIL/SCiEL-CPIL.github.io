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
    DEADLINE = 225.0  # safety margin before 230s kill
    
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
    n_kpts = len(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((n_kpts, n_pw_total))
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
        
    def compute_band_ratio(coeffs, use_full_bz=True):
        """Compute gap-midgap ratio."""
        if not use_full_bz:
            # Use a subset of k-points for speed during search
            subset_indices = [0, 9, 18, 27, 4, 14, 23, 28] 
            k_indices = subset_indices
        else:
            k_indices = range(n_kpts)
            
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
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
        """Enforce 4-fold rotational + reflection symmetry (D4)."""
        cell = np.array(cell, dtype=int)
        # Generate all 8 D4 transforms
        transforms = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1].copy(),
            cell[::-1, :].copy(),
            cell.T.copy(),
            np.rot90(cell.T.copy(), 1)
        ]
        # Average and threshold
        avg = np.sum(transforms, axis=0)
        result = (avg >= 4).astype(int)
        return result

    def generate_superellipses(n_exp, a_size, offset):
        """Generate a cell with 4 superellipses at (+/-offset, +/-offset)."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
            
        return enforce_symmetry(cell)

    def generate_ring_pattern(inner_r, outer_r):
        """Generate a ring (annulus) pattern centered at cell center."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        dist = np.sqrt(X**2 + Y**2)
        cell[(dist >= inner_r) & (dist <= outer_r)] = 1
        return enforce_symmetry(cell)

    def generate_cross_pattern(arm_width, arm_length):
        """Generate a cross/plus pattern."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        # Horizontal bar
        cell[(np.abs(Y) < arm_width/2) & (np.abs(X) < arm_length/2)] = 1
        # Vertical bar
        cell[(np.abs(X) < arm_width/2) & (np.abs(Y) < arm_length/2)] = 1
        return enforce_symmetry(cell)

    def generate_diamond_pattern(size_diag):
        """Generate a diamond (rotated square) pattern."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        dist = np.abs(X) + np.abs(Y)
        cell[dist <= size_diag/2] = 1
        return enforce_symmetry(cell)

    def generate_dual_pattern(inner_shape, outer_shape):
        """Generate a pattern with inner hole in outer shape."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        # Outer shape
        if outer_shape == 'circle':
            outer_mask = np.sqrt(X**2 + Y**2) <= 0.45
        elif outer_shape == 'square':
            outer_mask = (np.abs(X) <= 0.4) & (np.abs(Y) <= 0.4)
        elif outer_shape == 'diamond':
            outer_mask = np.abs(X) + np.abs(Y) <= 0.5
        else:
            outer_mask = np.sqrt(X**2 + Y**2) <= 0.45
        cell[outer_mask] = 1
        
        # Inner hole
        if inner_shape == 'circle':
            inner_mask = np.sqrt(X**2 + Y**2) <= inner_r
        elif inner_shape == 'square':
            inner_mask = (np.abs(X) <= inner_r) & (np.abs(Y) <= inner_r)
        elif inner_shape == 'diamond':
            inner_mask = np.abs(X) + np.abs(Y) <= inner_r
        else:
            inner_mask = np.sqrt(X**2 + Y**2) <= inner_r
        cell[inner_mask] = 0
        
        return enforce_symmetry(cell)

    # Phase 1: Fast Global Search with diverse candidates
    best_ratio = -np.inf
    best_cell = None
    
    candidates = []
    
    # Superellipse variations (proven effective)
    for n in [3.0, 4.0, 6.0, 8.0, 12.0]:
        for a in [0.12, 0.15, 0.18, 0.20, 0.22]:
            for off in [0.22, 0.25, 0.28]:
                cell = generate_superellipses(n, a, off)
                candidates.append(cell)
                
    # Ring patterns
    for inner in [0.05, 0.10, 0.15, 0.20]:
        for outer in [0.30, 0.35, 0.40, 0.45]:
            if outer > inner:
                cell = generate_ring_pattern(inner, outer)
                candidates.append(cell)
                
    # Cross patterns
    for w in [0.05, 0.08, 0.10, 0.12]:
        for l in [0.30, 0.40, 0.50]:
            cell = generate_cross_pattern(w, l)
            candidates.append(cell)
            
    # Diamond patterns
    for d in [0.30, 0.40, 0.50, 0.60]:
        cell = generate_diamond_pattern(d)
        candidates.append(cell)
        
    # Dual patterns (ring-like with different shapes)
    for inner_r in [0.05, 0.10, 0.15]:
        for outer in ['circle', 'square', 'diamond']:
            cell = generate_dual_pattern(inner_r, outer)
            candidates.append(cell)

    # Evaluate candidates with coarse BZ
    for cell in candidates:
        if not time_ok():
            break
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs, use_full_bz=False)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
            
    if best_cell is None:
        best_cell = generate_superellipses(4.0, 0.18, 0.25)
        
    # Phase 2: Refine best candidate with full BZ and local pixel optimization
    current_cell = best_cell.copy()
    coeffs = get_fourier_coeffs(current_cell)
    current_ratio = compute_band_ratio(coeffs, use_full_bz=True)
    
    # Local optimization: try flipping pixels in the fundamental domain
    # D4 symmetry: fundamental domain is 1/8 of the cell
    # We'll try flipping random pixels and enforcing symmetry
    np.random.seed(42)
    max_steps = 50
    step = 0
    no_improve_count = 0
    
    while step < max_steps and time_ok():
        step += 1
        improved = False
        
        # Try multiple random pixel flips per step
        for _ in range(3):
            if not time_ok():
                break
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            new_cell = enforce_symmetry(new_cell)
            
            new_coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(new_coeffs, use_full_bz=True)
            
            if new_ratio > current_ratio:
                current_cell = new_cell
                current_ratio = new_ratio
                improved = True
                no_improve_count = 0
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break

    return current_cell.tolist()
# EVOLVE-BLOCK-END
