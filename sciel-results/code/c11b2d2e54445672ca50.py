"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a superellipse-based parameterization for 4 dielectric rods (or holes)
       to cover a broad, physically relevant design space (circles to squares).
    2. Optimize geometric parameters (radius, order, offset) using a coarse-to-fine
       grid search followed by local hill-climbing.
    3. Use a fast, corrected plane-wave solver (Hermitian matrix) as the objective.
    4. Enforce C4v symmetry exactly.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 230s limit

    # --- 1. Parameterization ---
    # We use a "4-rod" superellipse model.
    # Background is air (0). 4 dielectric rods (1) are placed at (0.25, 0.25), (0.75, 0.25), etc.
    # Wait, usually for high-contrast gaps, we want high eps in center or specific patterns.
    # Inspiration 1 (score 7.25) used 4 shapes.
    # Inspiration 2 (score 8.79) used 4 air holes in dielectric background.
    # Let's stick to the 4-dielectric-rods-in-air model as it's standard for TM gaps in square lattices
    # or 4-air-holes-in-dielectric.
    # Let's implement a flexible "blob" generator that can be either rod or hole.
    # Actually, let's follow Inspiration 1's "4-shape" but make it robust.
    # Centers: (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75).
    # These are symmetric under C4v about (0.5, 0.5).
    
    def make_cell(radius, n_order, offset, is_hole=False):
        """
        Create a symmetric 4-shape cell using superellipse boundaries.
        radius: effective radius in pixel units.
        n_order: superellipse exponent (2=circle, 4=square, etc).
        offset: shift the centers slightly? Inspiration 1 used offset.
                If offset=0, centers are at 0.25, 0.75.
                If offset != 0, centers move. 
                Wait, moving centers breaks the symmetry about 0.5 unless they move in pairs?
                (0.25+off, 0.25+off) and (0.75-off, 0.75-off) are symmetric.
                (0.25+off, 0.75-off) and (0.75-off, 0.25+off) are symmetric.
                So yes, this preserves C4v.
        is_hole: if True, background is 1, shapes are 0. If False, background is 0, shapes are 1.
        """
        cell = np.zeros((size, size), dtype=int)
        if is_hole:
            cell[:] = 1
            
        # Centers in pixel coordinates (center of pixel is i+0.5)
        # Fractional coords: 0.25, 0.75
        c1 = size * 0.25 + offset
        c2 = size * 0.75 - offset
        
        centers = [
            (c1, c1),
            (c2, c1),
            (c1, c2),
            (c2, c2)
        ]
        
        # Precompute pixel coords
        y_coords, x_coords = np.mgrid[0:size, 0:size]
        # Shift to center of pixel
        x_coords = x_coords + 0.5
        y_coords = y_coords + 0.5
        
        mask = np.zeros((size, size), dtype=bool)
        
        if radius <= 0:
            return cell
            
        for cx, cy in centers:
            dx = x_coords - cx
            dy = y_coords - cy
            
            # Superellipse: |dx/r|^n + |dy/r|^n < 1
            # Handle n=0 or small n carefully
            if n_order < 0.1:
                # Circle limit
                val = dx*dx + dy*dy
                inside = val < radius*radius
            else:
                # Avoid division by zero or negative base issues with fractional powers
                # Since dx, dy can be negative, we use abs
                # |dx/r|^n + |dy/r|^n
                # If n is large, this approximates a box
                dx_abs = np.abs(dx) / radius
                dy_abs = np.abs(dy) / radius
                
                # Clip to avoid overflow for large n?
                # For n=10, 1.0^10 = 1. 1.1^10 ~ 2.59.
                # If dx/r > 1, term > 1, sum > 1, outside.
                # We can just compute.
                val = np.power(dx_abs, n_order) + np.power(dy_abs, n_order)
                inside = val < 1.0
                
            mask = mask | inside
            
        if is_hole:
            cell[mask] = 0
        else:
            cell[mask] = 1
            
        return cell

    # --- 2. Fast Evaluator ---
    # Corrected plane-wave solver.
    # Key fix: Ensure the matrix is Hermitian.
    # H_{ij} = < (G_i+k) dot (G_j+k) / eps >
    # For TM: The operator is (k+G) x (k+G) 1/eps (k+G) ... wait.
    # Standard PW expansion for TM (scalar field Ez):
    # sum_j (G_i + k) dot (G_j + k) * (1/eps)_{G_i - G_j} * E_j = (omega^2/c^2) E_i
    # Let K_i = G_i + k.
    # H_{ij} = K_i dot K_j * eps_inv[G_i - G_j]
    # Is this Hermitian?
    # H_{ji} = K_j dot K_i * eps_inv[G_j - G_i]
    # Since dot product is commutative, K_i dot K_j = K_j dot K_i.
    # Since eps_inv is real and symmetric (eps(x) = eps(-x) for symmetric cell),
    # eps_inv[G] = eps_inv[-G].
    # So H_{ji} = K_i dot K_j * eps_inv[G_i - G_j] = H_{ij}.
    # So the matrix should be real symmetric (Hermitian) if eps is symmetric.
    # Inspiration 1 explicitly symmetrized: H = (M + M^H)/2.
    # Let's do that to be safe against numerical noise or asymmetry in FFT extraction.
    
    pw_radius = 10  # Circular cutoff for G vectors
    G_vectors = []
    for h in range(-pw_radius, pw_radius + 1):
        for k in range(-pw_radius, pw_radius + 1):
            if h * h + k * k <= pw_radius * pw_radius:
                G_vectors.append((h, k))
    N_pw = len(G_vectors)
    
    # K-points for the irreducible BZ
    # Gamma (0,0) to M (0.5, 0) to X (0.5, 0.5)
    # We need enough points to find the max of band 4 and min of band 5.
    # Let's use a denser set.
    k_points = []
    # Gamma-M
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.0, t))
    # M-X
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.5 - t))
    # Gamma-X (diagonal)
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
        
    # Remove duplicates
    unique_k_points = []
    for kp in k_points:
        if kp not in unique_k_points:
            unique_k_points.append(kp)
    k_points = unique_k_points

    def evaluate_gap(cell):
        """
        Computes the approximate band gap between bands 4 and 5.
        """
        # 1. Compute FFT of 1/eps
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(h, k):
            idx_h = h % size
            idx_k = k % size
            return fft_1eps[idx_h, idx_k]
        
        max_omega_4_sq = 0.0
        min_omega_5_sq = float('inf')
        
        for kx, ky in k_points:
            # Build H matrix
            # H is N_pw x N_pw
            # We can vectorize the dot product calculation if needed, but N_pw ~ 300, 300x300=90k ops per k-point.
            # 30 k-points * 90k = 2.7M ops. Very fast in Python/NumPy.
            
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i, (hi, ki) in enumerate(G_vectors):
                Kx_i = hi + kx
                Ky_i = ki + ky
                
                for j, (hj, kj) in enumerate(G_vectors):
                    Kx_j = hj + kx
                    Ky_j = kj + ky
                    
                    dot = Kx_i * Kx_j + Ky_i * Ky_j
                    
                    dh = hi - hj
                    dk = ki - kj
                    coeff = get_coeff(dh, dk)
                    
                    H[i, j] = dot * coeff
            
            # Symmetrize
            H = (H + H.conj().T) / 2.0
            
            # Solve
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                
                if len(eigenvalues) >= 5:
                    w4_sq = eigenvalues[3]
                    w5_sq = eigenvalues[4]
                    
                    # Ensure positive
                    if w4_sq < 0: w4_sq = 0
                    if w5_sq < 0: w5_sq = 0
                    
                    if w4_sq > max_omega_4_sq:
                        max_omega_4_sq = w4_sq
                    if w5_sq < min_omega_5_sq:
                        min_omega_5_sq = w5_sq
                        
            except Exception:
                pass
        
        if min_omega_5_sq == float('inf') or max_omega_4_sq == 0:
            return -1.0
            
        # Convert to omega for ratio
        w4 = np.sqrt(max_omega_4_sq)
        w5 = np.sqrt(min_omega_5_sq)
        
        gap = w5 - w4
        midgap = (w5 + w4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Optimization Loop ---
    
    # Coarse Grid Search
    best_score = -1e10
    best_params = None
    
    # Parameters: radius, n_order, offset, is_hole
    # Let's search both "rods" and "holes"
    
    radii = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    orders = [2.0, 3.0, 4.0, 5.0, 6.0]
    offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
    hole_options = [False, True]
    
    for is_hole in hole_options:
        for r in radii:
            for n_ord in orders:
                for off in offsets:
                    if time.time() > deadline - 30:
                        break
                    cell = make_cell(r, n_ord, off, is_hole)
                    score = evaluate_gap(cell)
                    if score > best_score:
                        best_score = score
                        best_params = (r, n_ord, off, is_hole)
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # If no params found, return a default
    if best_params is None:
        # Default: 4 circles
        best_params = (5.0, 2.0, 0.0, False)
        final_cell = make_cell(*best_params)
        return final_cell.tolist()

    # Fine-tuning (Hill Climbing)
    r_best, n_best, off_best, hole_best = best_params
    
    step_sizes = {
        'r': 0.5,
        'n': 0.5,
        'off': 0.5
    }
    
    # Local search
    for _ in range(20): # Max 20 refinement steps
        if time.time() > deadline - 10:
            break
            
        improved = False
        
        # Try perturbations
        candidates = []
        # Radius
        candidates.append((r_best + step_sizes['r'], n_best, off_best, hole_best))
        candidates.append((r_best - step_sizes['r'], n_best, off_best, hole_best))
        # Order
        candidates.append((r_best, n_best + step_sizes['n'], off_best, hole_best))
        candidates.append((r_best, n_best - step_sizes['n'], off_best, hole_best))
        # Offset
        candidates.append((r_best, n_best, off_best + step_sizes['off'], hole_best))
        candidates.append((r_best, n_best, off_best - step_sizes['off'], hole_best))
        
        # Filter valid params
        valid_candidates = []
        for c in candidates:
            r, n, off, h = c
            if r < 1.0 or r > 10.0: continue
            if n < 1.0 or n > 10.0: continue
            if abs(off) > 4.0: continue
            valid_candidates.append(c)
            
        for c in valid_candidates:
            if time.time() > deadline - 5:
                break
            cell = make_cell(*c)
            score = evaluate_gap(cell)
            if score > best_score:
                best_score = score
                best_params = c
                r_best, n_best, off_best, hole_best = c
                improved = True
        
        if not improved:
            # Reduce step sizes
            step_sizes['r'] *= 0.7
            step_sizes['n'] *= 0.7
            step_sizes['off'] *= 0.7
            if step_sizes['r'] < 0.1:
                break

    # --- 4. Finalize Design ---
    r_final, n_final, off_final, hole_final = best_params
    final_cell = make_cell(r_final, n_final, off_final, hole_final)
    
    # Enforce symmetry exactly (just in case)
    # The construction is symmetric, but let's verify/force it.
    sym_cell = np.zeros_like(final_cell)
    for i in range(size):
        for j in range(size):
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
            s = 0
            for pi, pj in points:
                s += final_cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0
    
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
