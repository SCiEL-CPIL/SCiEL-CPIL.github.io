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
    DEADLINE = 218.0  # safety margin before 230s kill
    
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

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        # 4-fold rotation: 90, 180, 270 degrees
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        # Reflections
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        # Transpose
        trans = cell.T.copy()
        # All 8 symmetries
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    def generate_superellipses(n_exp, a_size, offset):
        """
        Generate a cell with 4 superellipses at (+/-offset, +/-offset).
        Equation: |x/a|^n + |y/a|^n <= 1
        """
        cell = np.zeros((size, size), dtype=int)
        # Coordinates relative to center, scaled to [-0.5, 0.5]
        # Pixel centers are at (i+0.5)/size - 0.5
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            # Superellipse condition
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
            
        return enforce_symmetry(cell)

    def generate_fractal_disk(r0, a4, a8, phase4=0.0, phase8=0.0):
        """
        Generate a centrally located dielectric disk with fractal boundary.
        r(theta) = r0 * (1 + a4*cos(4*theta + phase4) + a8*cos(8*theta + phase8))
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        r_func = r0 * (1.0 + a4 * np.cos(4 * Theta + phase4) + a8 * np.cos(8 * Theta + phase8))
        
        cell[R <= r_func] = 1
        
        return enforce_symmetry(cell)

    def generate_ca_pattern(seed, n_steps, threshold=0.5):
        """
        Generate a cell using a cellular automaton approach.
        Uses a symmetric 2D CA rule to generate complex patterns.
        The CA evolves on a toroidal grid to maintain symmetry.
        """
        # Initialize with a small symmetric seed pattern
        state = np.zeros((size, size), dtype=float)
        
        # Place seed bits in a symmetric pattern
        # Use the seed to determine which of the 8 symmetric positions get initial 1s
        seed_int = int(seed)
        # The fundamental domain is 16x16 (top-left quadrant including center line)
        # We pick some positions in the fundamental domain
        for i in range(16):
            for j in range(16):
                # Use bits of the seed to determine if this position is active
                bit_index = i * 16 + j
                if (seed_int >> bit_index) & 1:
                    state[i, j] = 1.0
        
        # Evolve the CA for n_steps
        for step in range(n_steps):
            # Compute neighbors (Moore neighborhood)
            new_state = np.zeros_like(state)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    # Toroidal shift
                    shifted = np.roll(np.roll(state, di, axis=0), dj, axis=1)
                    new_state += shifted * 0.1  # Weighted sum of neighbors
            
            # Apply a nonlinear threshold rule
            # This creates complex patterns
            state = (new_state > threshold).astype(float) * 0.5 + state * 0.5
            
        # Convert to binary and enforce symmetry
        cell = (state > 0.3).astype(int)
        return enforce_symmetry(cell)

    def generate_concentric_rings(n_rings, radii, widths, phase_offset=0.0):
        """
        Generate a cell with concentric rings of dielectric material.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        R = np.sqrt(X**2 + Y**2)
        
        for i in range(n_rings):
            r_inner = radii[i] - widths[i] / 2.0
            r_outer = radii[i] + widths[i] / 2.0
            mask = (R >= r_inner) & (R <= r_outer)
            cell[mask] = 1
            
        return enforce_symmetry(cell)

    # Phase 1: Global Search over diverse design families
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    best_family = None
    
    # Family 1: Superellipses (proven approach)
    n_values = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
    a_values = [0.08, 0.10, 0.12, 0.14]
    off_values = [0.20, 0.22, 0.25, 0.28]
    
    for n in n_values:
        for a in a_values:
            for off in off_values:
                if not time_ok():
                    break
                cell = generate_superellipses(n, a, off)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (n, a, off)
                    best_cell = cell.copy()
                    best_family = 'superellipse'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 2: Fractal disks
    r0_values = [0.20, 0.22, 0.25, 0.28, 0.30]
    a4_values = [0.05, 0.10, 0.15, 0.20]
    a8_values = [0.02, 0.05, 0.10, 0.15]
    
    for r0 in r0_values:
        for a4 in a4_values:
            for a8 in a8_values:
                if not time_ok():
                    break
                cell = generate_fractal_disk(r0, a4, a8)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (r0, a4, a8)
                    best_cell = cell.copy()
                    best_family = 'fractal_disk'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 3: Cellular Automaton patterns
    ca_seeds = [1, 3, 7, 15, 31, 63, 127, 255]
    ca_steps = [3, 5, 8, 12, 16]
    ca_thresholds = [0.4, 0.5, 0.6]
    
    for seed in ca_seeds:
        for steps in ca_steps:
            for thresh in ca_thresholds:
                if not time_ok():
                    break
                cell = generate_ca_pattern(seed, steps, thresh)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (seed, steps, thresh)
                    best_cell = cell.copy()
                    best_family = 'ca'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 4: Concentric rings
    ring_configs = [
        (3, [0.15, 0.25, 0.35], [0.05, 0.05, 0.05]),
        (4, [0.10, 0.20, 0.30, 0.40], [0.04, 0.04, 0.04, 0.04]),
        (2, [0.20, 0.35], [0.08, 0.08]),
        (5, [0.10, 0.18, 0.26, 0.34, 0.42], [0.03, 0.03, 0.03, 0.03, 0.03]),
    ]
    
    for n_rings, radii, widths in ring_configs:
        if not time_ok():
            break
        cell = generate_concentric_rings(n_rings, radii, widths)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_params = (n_rings, radii, widths)
            best_cell = cell.copy()
            best_family = 'rings'
    
    if best_cell is None:
        # Fallback
        best_cell = generate_superellipses(2.0, 0.12, 0.25)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (2.0, 0.12, 0.25)
        best_family = 'superellipse'

    # Phase 2: Local Optimization
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 140.0  # Leave ~50s for safety and potential pixel refinement
    
    no_improve_count = 0
    max_no_improve = 30
    
    # Determine which family we're in and use appropriate perturbation
    if best_family == 'superellipse':
        current_n, current_a, current_off = best_params
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            improved = False
            
            new_n = current_n * (1.0 + np.random.normal(0, 0.1))
            new_a = current_a * (1.0 + np.random.normal(0, 0.1))
            new_off = current_off + np.random.normal(0, 0.02)
            
            new_n = np.clip(new_n, 1.0, 20.0)
            new_a = np.clip(new_a, 0.05, 0.20)
            new_off = np.clip(new_off, 0.15, 0.35)
            
            new_cell = generate_superellipses(new_n, new_a, new_off)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_n, current_a, current_off = new_n, new_a, new_off
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    elif best_family == 'fractal_disk':
        current_r0, current_a4, current_a8 = best_params
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            improved = False
            
            new_r0 = current_r0 * (1.0 + np.random.normal(0, 0.05))
            new_a4 = current_a4 * (1.0 + np.random.normal(0, 0.1))
            new_a8 = current_a8 * (1.0 + np.random.normal(0, 0.1))
            
            new_r0 = np.clip(new_r0, 0.15, 0.35)
            new_a4 = np.clip(new_a4, 0.01, 0.30)
            new_a8 = np.clip(new_a8, 0.005, 0.25)
            
            new_cell = generate_fractal_disk(new_r0, new_a4, new_a8)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_r0, current_a4, current_a8 = new_r0, new_a4, new_a8
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    elif best_family == 'ca':
        current_seed, current_steps, current_thresh = best_params
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            improved = False
            
            # Perturb the CA parameters
            new_seed = current_seed + np.random.choice([-2, -1, 1, 2])
            new_seed = max(1, new_seed)
            new_steps = current_steps + np.random.choice([-2, -1, 1, 2])
            new_steps = max(1, min(20, new_steps))
            new_thresh = current_thresh + np.random.normal(0, 0.05)
            new_thresh = np.clip(new_thresh, 0.3, 0.7)
            
            new_cell = generate_ca_pattern(new_seed, new_steps, new_thresh)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_seed, current_steps, current_thresh = new_seed, new_steps, new_thresh
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    elif best_family == 'rings':
        current_n_rings, current_radii, current_widths = best_params
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            improved = False
            
            # Perturb ring parameters
            new_radii = [r * (1.0 + np.random.normal(0, 0.1)) for r in current_radii]
            new_widths = [w * (1.0 + np.random.normal(0, 0.1)) for w in current_widths]
            new_radii = [np.clip(r, 0.05, 0.45) for r in new_radii]
            new_widths = [np.clip(w, 0.01, 0.15) for w in new_widths]
            
            new_cell = generate_concentric_rings(current_n_rings, new_radii, new_widths)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_radii, current_widths = new_radii, new_widths
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
    else:
        # Generic fallback: just use pixel refinement
        pass

    # Phase 3: Pixel Refinement
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 30:
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 25.0
        
        no_improve_refine = 0
        for (i, j) in indep_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
                
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1
                if no_improve_refine > 50:
                    break

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
        
    return current_cell.tolist()
# EVOLVE-BLOCK-END
