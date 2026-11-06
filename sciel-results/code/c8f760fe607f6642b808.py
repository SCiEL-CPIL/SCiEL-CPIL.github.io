"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np

_DEADLINE = 218.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return _DEADLINE - _elapsed()

def _enforce_d4_strict(cell):
    """
    Enforce D4 symmetry by mirroring the top-left quadrant.
    """
    size = 32
    half = size // 2
    result = [[0] * size for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = cell[i][j]
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            for pi, pj in positions:
                result[pi][pj] = val
    return result

def _compute_tm_gap_fast(cell, N_pw=4, n_kpts=8):
    """
    Fast TM band gap computation using plane-wave method.
    Optimized for speed: vectorized matrix construction and reduced k-points.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    # Convert cell to array for vectorized operations
    cell_arr = np.array(cell)
    inv_eps = np.where(cell_arr == 1, 1.0 / eps_high, 1.0 / eps_low)
    
    # FFT of inverse epsilon
    fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
    
    # Precompute G indices
    g_indices = []
    for m in range(-N_pw, N_pw + 1):
        for n in range(-N_pw, N_pw + 1):
            g_indices.append((m, n))
    npw = len(g_indices)
    
    # Precompute S matrix (Fourier coefficients of 1/eps)
    # S[i,j] = G_{G_i - G_j}
    S = np.zeros((npw, npw), dtype=float)
    for i_idx, (mi, ni) in enumerate(g_indices):
        for j_idx, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i_idx, j_idx] = fft_coeffs[gm % size, gn % size].real

    # Define k-points in irreducible BZ
    # Triangle: (0,0) -> (0.5, 0) -> (0.5, 0.5)
    kpts = []
    # Gamma to X
    for i in range(3):
        kx = i / 2.0 * 0.5
        kpts.append((kx, 0.0))
    # X to M
    for i in range(1, 3):
        ky = i / 2.0 * 0.5
        kpts.append((0.5, ky))
    # M to Gamma
    for i in range(1, 3):
        t = i / 2.0
        kpts.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Add center of triangle
    kpts.append((1/6, 1/6))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        # (k+G)^2 vector
        km2 = np.array([(2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2 for m, n in g_indices])
        
        # A = S * diag(km2)
        # Eigenvalues of A give omega^2
        A = S * km2[np.newaxis, :]
        
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues.sort()
            
            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]
                
                if omega4_sq > 1e-6 and omega5_sq > 1e-6:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)
                    
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 < 1e-6:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap

def _make_cell_parametric(half_width, half_height, is_dielectric_bg=True):
    """
    Create a cell with a central square hole (if dielectric bg) or square rod (if air bg).
    half_width, half_height in pixels (fraction of 16, so 0.5 is 8 pixels).
    """
    size = 32
    cell = [[1] * size for _ in range(size)] if is_dielectric_bg else [[0] * size for _ in range(size)]
    
    val = 0 if is_dielectric_bg else 1
    
    # Center is at 15.5, 15.5
    # We define the shape in the top-left quadrant [0:16, 0:16]
    # Mirror symmetry handles the rest.
    
    # Convert half_width/height to pixel indices in the 16x16 quadrant
    # The quadrant covers x in [0, 0.5) of the cell, which is 16 pixels.
    # Center of cell is at 15.5.
    # In quadrant coords (0..15), the center is at 15.5? No.
    # Let's just fill the full 32x32 based on distance from center (15.5, 15.5).
    
    for i in range(size):
        for j in range(size):
            # Distance from center
            dx = abs(i - 15.5)
            dy = abs(j - 15.5)
            # Scale to "fraction of half-cell" (0 to 0.5)
            # half_width is given as fraction of the full cell width? 
            # Let's define half_width as fraction of the 16-pixel quadrant.
            # So half_width=1.0 means the shape extends to the edge of the quadrant (center of cell).
            # half_width=0.5 means it extends to 1/4 of the way.
            
            # Actually, let's use pixels directly for simplicity.
            # Let's say 'size_param' is the half-width in pixels.
            pass
            
    # Simpler approach: define by pixel indices
    # Let's redefine: make_cell_square(half_size_px, is_dielectric_bg)
    # half_size_px is how many pixels from the center line (15.5) are included.
    # If half_size_px = 4, pixels 12..15 and 16..19 are affected.
    
    cell = [[0] * size for _ in range(size)]
    val = 1 if is_dielectric_bg else 0 # This is confusing. Let's stick to:
    # If is_dielectric_bg=True, background is 1, hole is 0.
    # If is_dielectric_bg=False, background is 0, rod is 1.
    
    cell = [[1] * size for _ in range(size)] if is_dielectric_bg else [[0] * size for _ in range(size)]
    hole_val = 0 if is_dielectric_bg else 1
    
    # half_w_px, half_h_px: number of pixels from the center (15.5)
    # e.g., 4.0 means pixels 12,13,14,15 (and 16,17,18,19)
    
    for i in range(size):
        for j in range(size):
            # Check if inside the central square
            if abs(i - 15.5) < half_w_px and abs(j - 15.5) < half_h_px:
                cell[i][j] = hole_val
                
    return _enforce_d4_strict(cell)

def design(problem):
    global _start_time
    _start_time = time.time()
    
    # Parametric search over square holes/rods
    # Dimensions: half-width in pixels (0.5 to 7.5)
    # We test both Dielectric Background (Air Hole) and Air Background (Dielectric Rod)
    
    best_score = 0.0
    best_cell = None
    
    # Grid of sizes
    sizes = np.arange(1.0, 7.5, 0.5)
    
    # 1. Air holes in Dielectric
    for hw in sizes:
        for hh in sizes:
            if _remaining() < 10:
                return best_cell if best_cell else _make_cell_parametric(4.0, 4.0, True)
            
            cell = _make_cell_parametric(hw, hh, True)
            score = _compute_tm_gap_fast(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
                
    # 2. Dielectric rods in Air
    for hw in sizes:
        for hh in sizes:
            if _remaining() < 10:
                return best_cell if best_cell else _make_cell_parametric(4.0, 4.0, False)
            
            cell = _make_cell_parametric(hw, hh, False)
            score = _compute_tm_gap_fast(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
                
    # 3. Circular holes/rods (approximate with square or just check a few radii)
    # Let's add circular shapes
    def _make_cell_circle(radius_px, is_dielectric_bg=True):
        size = 32
        cell = [[1] * size for _ in range(size)] if is_dielectric_bg else [[0] * size for _ in range(size)]
        hole_val = 0 if is_dielectric_bg else 1
        for i in range(size):
            for j in range(size):
                dx = i - 15.5
                dy = j - 15.5
                if dx*dx + dy*dy < radius_px*radius_px:
                    cell[i][j] = hole_val
        return _enforce_d4_strict(cell)
        
    radii = np.arange(1.0, 7.5, 0.5)
    for r in radii:
        if _remaining() < 10:
            break
        cell = _make_cell_circle(r, True)
        score = _compute_tm_gap_fast(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            
    for r in radii:
        if _remaining() < 10:
            break
        cell = _make_cell_circle(r, False)
        score = _compute_tm_gap_fast(cell)
        if score > best_score:
            best_score = score
            best_cell = cell

    if best_cell is None:
        best_cell = _make_cell_parametric(4.0, 4.0, True)
        
    return best_cell
