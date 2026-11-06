import numpy as np
import time

def design(problem):
    """
    Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5.
    Uses Simulated Annealing on the binary pixel representation.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 215.0  # Safety margin before 230s kill

    def time_ok():
        return time.time() - start_time < deadline

    # --- Plane Wave Setup ---
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points as per spec)
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
    
    # --- Symmetry Handling ---
    # The cell is invariant under D4 symmetry (8 elements).
    # We define independent pixels as those in the top-left 16x16 quadrant,
    # specifically the upper triangle (i <= j) because of transpose symmetry C[i,j] = C[j,i].
    # Wait, C[i,j] = C[j,i] is transpose.
    # C[i,j] = C[31-i, 31-j] is 180 rot.
    # C[i,j] = C[i, 31-j] is h-flip.
    # C[i,j] = C[31-i, j] is v-flip.
    
    # Let's identify the fundamental domain.
    # The center is between 15 and 16.
    # Indices 0..15 are the "first half".
    # Due to h-flip: C[i,j] = C[i, 31-j].
    # Due to v-flip: C[i,j] = C[31-i, j].
    # Due to transpose: C[i,j] = C[j,i].
    
    # The independent region is i in [0, 15], j in [0, 15] with i <= j?
    # Let's check:
    # If we know C[i,j] for i<=15, j<=15, i<=j:
    # 1. C[j,i] is known (transpose).
    # 2. C[i, 31-j]: Since j<=15, 31-j >= 16.
    #    We need to map 31-j back to the independent set.
    #    Let j' = 31-j. j' is in [16, 31].
    #    This is getting complicated to map manually.
    
    # Simpler approach:
    # Just use the top-left 16x16 quadrant as independent variables (256 vars).
    # Enforce symmetry by constructing the full 32x32 from this 16x16 block.
    # This reduces variables from 1024 to 256.
    # We can further reduce to 136 by enforcing transpose in the block,
    # but 256 is still a good reduction and easier to manage.
    
    # Let's use the 16x16 top-left quadrant.
    quad_size = size // 2  # 16
    indep_pixels = np.arange(quad_size * quad_size)
    
    def reconstruct_full_cell(indep_block):
        """
        indep_block: 1D array of 0/1, length 256 (16x16)
        Returns: 32x32 array
        """
        block = indep_block.reshape((quad_size, quad_size))
        full = np.zeros((size, size), dtype=int)
        
        # Top-Left: block
        full[:quad_size, :quad_size] = block
        
        # Top-Right: h-flip of block
        # C[i, j] = C[i, 31-j]
        # For j in [16, 31], 31-j is in [0, 15].
        full[:quad_size, quad_size:] = block[:, ::-1]
        
        # Bottom-Left: v-flip of block
        # C[i, j] = C[31-i, j]
        # For i in [16, 31], 31-i is in [0, 15].
        full[quad_size:, :quad_size] = block[::-1, :]
        
        # Bottom-Right: 180 rot of block (or h-flip of bottom-left)
        # C[i, j] = C[31-i, 31-j]
        full[quad_size:, quad_size:] = block[::-1, ::-1]
        
        # Enforce transpose symmetry: C[i,j] = C[j,i]
        # The above construction ensures C[i,j] = C[31-i, 31-j] etc.
        # Does it ensure C[i,j] = C[j,i]?
        # Let's check.
        # full[i,j] for i<16, j<16 is block[i,j].
        # full[j,i] for j<16, i<16 is block[j,i].
        # If block is not symmetric, full is not transpose symmetric.
        # The problem statement says: "cell == its transpose".
        # So we MUST enforce C[i,j] = C[j,i].
        # This means block MUST be symmetric: block[i,j] == block[j,i].
        
        # To handle this in search, we can either:
        # 1. Only search over the upper triangle of the block (136 vars) and mirror it.
        # 2. Search over the full block (256 vars) and symmetrize after every step.
        
        # Option 2 is easier to code but wastes 50% of search space.
        # Option 1 is better.
        
        # Let's implement Option 1:
        # Independent pixels are (i,j) in 16x16 block with i <= j.
        # Count: 16*17/2 = 136.
        
        return full

    # Let's switch to Option 1 for efficiency.
    indep_coords = []
    for i in range(quad_size):
        for j in range(i, quad_size):
            indep_coords.append((i, j))
    
    n_indep = len(indep_coords)
    
    def reconstruct_full_cell_optimized(indep_vals):
        """
        indep_vals: 1D array of 0/1, length 136
        """
        # Create a symmetric 16x16 block
        block = np.zeros((quad_size, quad_size), dtype=int)
        for k, (i, j) in enumerate(indep_coords):
            val = indep_vals[k]
            block[i, j] = val
            block[j, i] = val  # Enforce transpose symmetry in the block
        
        full = np.zeros((size, size), dtype=int)
        full[:quad_size, :quad_size] = block
        full[:quad_size, quad_size:] = block[:, ::-1]
        full[quad_size:, :quad_size] = block[::-1, :]
        full[quad_size:, quad_size:] = block[::-1, ::-1]
        
        return full

    # --- Evaluation ---
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
        """Compute gap-midgap ratio using all 30 k-points."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def evaluate(indep_vals):
        cell = reconstruct_full_cell_optimized(indep_vals)
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio(coeffs)

    # --- Initial Candidates ---
    def make_initial_cell_four_rods(radius, offset):
        """
        Generate independent values for 4 rods at (±offset, ±offset) in normalized coords.
        """
        indep_vals = np.zeros(n_indep, dtype=int)
        # We need to map normalized coords to pixel coords.
        # Center of cell is (15.5, 15.5).
        # Rod centers:
        # (15.5 - offset*size, 15.5 - offset*size) -> Top-Left
        # (15.5 + offset*size, 15.5 + offset*size) -> Bottom-Right
        # etc.
        
        # Since we only define the top-left 16x16 block (indices 0-15),
        # and the rod is at (15.5 - d, 15.5 - d),
        # if d is small, the rod is near the center of the quadrant?
        # No, the center of the cell is the corner of the 4 quadrants.
        # The top-left quadrant covers x in [0, 15.5], y in [0, 15.5].
        # The rod center in global coords: (15.5 - d, 15.5 - d).
        # In local quadrant coords (0-based index i,j corresponds to x=i+0.5, y=j+0.5):
        # We want pixels where distance to (15.5-d, 15.5-d) is < r.
        
        # Let's just construct the full cell and extract independent values.
        full_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Check distance to 4 centers
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        cx = sx * offset
                        cy = sy * offset
                        if (x - cx)**2 + (y - cy)**2 < radius**2:
                            full_cell[i, j] = 1
                            break
                    else:
                        continue
                    break
        
        # Extract independent values
        for k, (i, j) in enumerate(indep_coords):
            indep_vals[k] = full_cell[i, j]
            
        return indep_vals

    # Generate initial candidates
    candidates = []
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(make_initial_cell_four_rods(r, off))
    
    # Single centered rod (mapped to independent vars)
    for r in [0.15, 0.20, 0.25]:
        full_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < r*r:
                    full_cell[i, j] = 1
        indep_vals = np.zeros(n_indep, dtype=int)
        for k, (i, j) in enumerate(indep_coords):
            indep_vals[k] = full_cell[i, j]
        candidates.append(indep_vals)

    # Evaluate candidates to find best start
    best_indep = None
    best_score = -np.inf
    for cand in candidates:
        if not time_ok(): break
        score = evaluate(cand)
        if score > best_score:
            best_score = score
            best_indep = cand.copy()

    if best_indep is None:
        best_indep = make_initial_cell_four_rods(0.10, 0.25)
        best_score = evaluate(best_indep)

    # --- Simulated Annealing ---
    current_indep = best_indep.copy()
    current_score = best_score
    best_indep_sa = current_indep.copy()
    best_score_sa = current_score
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    
    # Estimate time per eval
    # We have ~115s left.
    # Let's run SA until deadline.
    
    rng = np.random.default_rng()
    
    while time_ok() and T > T_min:
        # Pick a random independent pixel
        idx = rng.integers(0, n_indep)
        
        # Flip it
        old_val = current_indep[idx]
        current_indep[idx] = 1 - old_val
        
        new_score = evaluate(current_indep)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            current_score = new_score
            if new_score > best_score_sa:
                best_score_sa = new_score
                best_indep_sa = current_indep.copy()
        else:
            # Reject
            current_indep[idx] = old_val
            
        T *= cooling_rate
        
        # Check time
        if not time_ok():
            break

    # Final check with best found
    final_cell = reconstruct_full_cell_optimized(best_indep_sa)
    
    return final_cell.tolist()
