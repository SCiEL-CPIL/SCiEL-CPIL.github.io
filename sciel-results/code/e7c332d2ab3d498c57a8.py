"""Rewritten: hole-in-dielectric photonic crystal with parameter search."""
import time
import numpy as np

def _compute_eps_grid(cell, eps_low, eps_high):
    """Convert 0/1 pixel cell to epsilon grid."""
    arr = np.array(cell, dtype=np.float64)
    return np.where(arr > 0.5, eps_high, eps_low)

def _eps_fourier(eps_grid, N, kmax):
    """Compute Fourier coefficients of epsilon on a grid of size N.
    Returns dict mapping (mx, my) -> coeff for |mx|,|my| <= kmax."""
    N = eps_grid.shape[0]
    coeffs = {}
    for mx in range(-kmax, kmax + 1):
        for my in range(-kmax, kmax + 1):
            # Fourier coefficient: (1/N^2) sum_{i,j} eps[i,j] exp(-2pi i (mx*i + my*j)/N)
            # Use numpy FFT for speed
            pass
    # More efficient: use 2D FFT
    eps_ext = np.zeros((2 * N, 2 * N), dtype=np.float64)
    eps_ext[:N, :N] = eps_grid
    F = np.fft.fft2(eps_ext)
    F /= (2 * N) ** 2
    for mx in range(-kmax, kmax + 1):
        for my in range(-kmax, kmax + 1):
            # Index in FFT: positive frequencies in [0, N), negative in [N, 2N)
            ix = mx % (2 * N)
            iy = my % (2 * N)
            coeffs[(mx, my)] = F[iy, ix].real  # epsilon is real and symmetric
    return coeffs

def _band_structure(cell, eps_low, eps_high, n_pw=13, n_kpts=8):
    """Quick band structure calculation for TM mode.
    Returns (max_band4, min_band5) in units of a/lambda * c."""
    N = len(cell)
    eps_grid = _compute_eps_grid(cell, eps_low, eps_high)
    kmax = (n_pw - 1) // 2
    coeffs = _eps_fourier(eps_grid, N, kmax)
    
    # Plane wave vectors: G = (2pi/a) * (mx, my)
    # We work in units where a = 1, so G = 2pi*(mx, my)
    gw = []
    for mx in range(-kmax, kmax + 1):
        for my in range(-kmax, kmax + 1):
            gw.append((mx, my))
    n_pw_actual = len(gw)
    
    # Epsilon inverse matrix: (eps^{-1})_{G,G'} = delta_{G,G'} / <eps>_{G-G'}
    # For TM: (eps^{-1})_{G,G'} = delta_{G,G'} / eps_{G-G'}
    # Actually for TM, the eigenvalue problem is:
    # sum_{G'} (1/eps_{G-G'}) * |G+K|^2 * C_{G'} = omega^2/a^2 * C_G
    # Wait, let me be more careful.
    
    # For TM polarization, the equation is:
    # (1/eps(r)) * (-nabla^2) * H_z = (omega/c)^2 * H_z
    # In plane waves: H_z = sum_C C_G exp(i(G+K).r)
    # -nabla^2 exp(i(G+K).r) = |G+K|^2 exp(i(G+K).r)
    # So: sum_{G'} C_{G'} |G'+K|^2 / eps_{G-G'} = (omega a / 2pi c)^2 * C_G
    # (working in units where the eigenvalue is (omega a / 2pi)^2)
    
    # Build the matrix M where M_{G,G'} = |G'+K|^2 / eps_{G-G'}
    # Actually the standard form: the operator is (1/eps) * (-nabla^2)
    # In Fourier space, 1/eps becomes a convolution with 1/eps_Fourier
    # And -nabla^2 becomes multiplication by |G+K|^2
    # So the matrix element <G| (1/eps)(-nabla^2) |G'> = |G'+K|^2 * (1/eps)_{G-G'}
    
    # Where (1/eps)_{G-G'} is the Fourier coefficient of 1/eps(r), NOT 1/eps_Fourier_coeff
    # This is a critical distinction. We need Fourier coefficients of 1/eps(r).
    
    # Let me redo: compute 1/eps grid first
    inv_eps_grid = 1.0 / eps_grid
    inv_eps_coeffs = _eps_fourier(inv_eps_grid, N, kmax)
    
    # K points along the symmetry line M-X-G (irreducible BZ)
    # For square lattice: G = (0,0), X = (pi/a, 0), M = (pi/a, pi/a)
    # In units of 2pi/a: G=(0,0), X=(0.5, 0), M=(0.5, 0.5)
    # Irreducible BZ path: M -> X -> G -> (0, 0.5) -> M
    # Let's use a few key points
    
    k_points = [
        (0.5, 0.5),  # M
        (0.5, 0.0),  # X
        (0.0, 0.0),  # G
        (0.0, 0.5),  # midpoint of G-X
        (0.25, 0.25), # interior
        (0.5, 0.25), # interior
        (0.1, 0.1),  # near G
        (0.4, 0.3),  # interior
    ]
    
    max_band4 = 0.0
    min_band5 = float('inf')
    
    for Kx, Ky in k_points:
        # K in units of 2pi/a
        K = np.array([Kx, Ky]) * 2 * np.pi
        
        # Build matrix
        M = np.zeros((n_pw_actual, n_pw_actual), dtype=np.float64)
        for i, (mx, my) in enumerate(gw):
            Gi = np.array([mx, my]) * 2 * np.pi
            Ki = Gi + K
            ki2 = np.dot(Ki, Ki)
            for j, (mx2, my2) in enumerate(gw):
                dGx = mx - mx2
                dGy = my - my2
                inv_e = inv_eps_coeffs.get((dGx, dGy), 0.0)
                M[i, j] = ki2 * inv_e
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            # The eigenvalues are (omega * a / 2pi)^2
            # We want bands 4 and 5 (1-indexed), so indices 3 and 4
            if len(eigvals) >= 5:
                band4 = np.sqrt(eigvals[3])
                band5 = np.sqrt(eigvals[4])
                max_band4 = max(max_band4, band4)
                min_band5 = min(min_band5, band5)
        except np.linalg.LinAlgError:
            continue
    
    if min_band5 == float('inf'):
        return 0.0
    
    gap_ratio = 2.0 * (min_band5 - max_band4) / (min_band5 + max_band4)
    return gap_ratio

def _make_cell_hole(size, radius, shape='circle'):
    """Create a hole-in-dielectric cell: 1 everywhere except a central hole."""
    cell = []
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    for i in range(size):
        row = []
        for j in range(size):
            dx = (j - cx) / size
            dy = (i - cy) / size
            if shape == 'circle':
                is_hole = (dx * dx + dy * dy) < radius * radius
            elif shape == 'square':
                is_hole = (abs(dx) < radius and abs(dy) < radius)
            elif shape == 'superellipse':
                # |dx|^n + |dy|^n < r^n with n=4
                n = 4.0
                is_hole = (abs(dx) ** n + abs(dy) ** n) < radius ** n
            else:
                is_hole = (dx * dx + dy * dy) < radius * radius
            row.append(0 if is_hole else 1)
        cell.append(row)
    return cell

def _make_cell_cross(size, arm_width, length):
    """Dielectric cross shape (air background)."""
    cell = []
    half = size // 2
    for i in range(size):
        row = []
        for j in range(size):
            dist_from_center_x = abs(j - half)
            dist_from_center_y = abs(i - half)
            if dist_from_center_x < arm_width // 2 and dist_from_center_y < length // 2:
                row.append(1)
            elif dist_from_center_y < arm_width // 2 and dist_from_center_x < length // 2:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell

def _make_cell_ring(size, outer_r, inner_r):
    """Dielectric ring (air background with central air hole and outer air)."""
    cell = []
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    for i in range(size):
        row = []
        for j in range(size):
            dx = (j - cx) / size
            dy = (i - cy) / size
            r2 = dx * dx + dy * dy
            if inner_r * inner_r < r2 < outer_r * outer_r:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = start_time + 210  # 210s safety margin
    
    best_score = -1.0
    best_cell = None
    
    # Strategy: search over hole-in-dielectric parameters
    # The hole-in-dielectric is the standard large-gap TM configuration
    
    # Phase 1: Coarse search over hole radius and shape
    shapes = ['circle', 'square', 'superellipse']
    radii = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33]
    
    for shape in shapes:
        for radius in radii:
            if time.time() > deadline - 60:
                break
            cell = _make_cell_hole(size, radius, shape)
            score = _band_structure(cell, eps_low, eps_high, n_pw=11, n_kpts=8)
            if score > best_score:
                best_score = score
                best_cell = cell
        if time.time() > deadline - 60:
            break
    
    # Phase 2: Fine search around best parameters
    if best_cell is not None:
        # Determine which shape and radius were best, then refine
        # Re-search with finer radii
        for shape in shapes:
            for radius in np.arange(0.12, 0.38, 0.02):
                if time.time() > deadline - 30:
                    break
                cell = _make_cell_hole(size, radius, shape)
                score = _band_structure(cell, eps_low, eps_high, n_pw=13, n_kpts=8)
                if score > best_score:
                    best_score = score
                    best_cell = cell
            if time.time() > deadline - 30:
                break
    
    # Phase 3: Try cross and ring shapes
    for arm_w in [4, 6, 8, 10]:
        for length in [8, 12, 16, 20, 24]:
            if time.time() > deadline - 20:
                break
            cell = _make_cell_cross(size, arm_w, length)
            score = _band_structure(cell, eps_low, eps_high, n_pw=11, n_kpts=8)
            if score > best_score:
                best_score = score
                best_cell = cell
        if time.time() > deadline - 20:
            break
    
    for outer_r in [0.3, 0.35, 0.4, 0.45]:
        for inner_r in [0.1, 0.15, 0.2, 0.25]:
            if time.time() > deadline - 10:
                break
            cell = _make_cell_ring(size, outer_r, inner_r)
            score = _band_structure(cell, eps_low, eps_high, n_pw=11, n_kpts=8)
            if score > best_score:
                best_score = score
                best_cell = cell
        if time.time() > deadline - 10:
            break
    
    # Fallback: if nothing found, use the seed (centered rod)
    if best_cell is None:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    return best_cell
