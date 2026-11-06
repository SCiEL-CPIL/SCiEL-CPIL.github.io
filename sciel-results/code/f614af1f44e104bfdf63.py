"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit
    
    # Precompute Fourier coefficients for the pixel grid
    # We'll use a plane-wave expansion with N=17 modes (17x17=289 total)
    N = 17  # odd number for symmetry
    n_max = (N - 1) // 2  # 8
    
    # Generate wave vector grid for irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi/a
    # We'll sample points along the boundary and interior
    k_points_full = _generate_k_points()
    # Coarse k-points for screening
    k_points_coarse = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.25, 0.25), (0.1, 0.3)]
    
    best_cell = None
    best_score = -1.0
    
    # Strategy: Search over various geometric patterns with full square symmetry (D4)
    
    # 1. Single central circle (rod)
    for r_frac in np.linspace(0.15, 0.45, 10):
        if time.time() > deadline:
            break
        cell = _make_circle_cell(size, r_frac)
        score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
            # Refine if promising
            if best_score > 0.01:
                for r_ref in np.linspace(r_frac - 0.02, r_frac + 0.02, 5):
                    if r_ref <= 0 or r_ref >= 0.5: continue
                    cell_ref = _make_circle_cell(size, r_ref)
                    score_ref = _evaluate_gap(cell_ref, size, N, k_points_full)
                    if score_ref > best_score:
                        best_score = score_ref
                        best_cell = cell_ref

    # 2. Four circles at diagonal positions (2x2 lattice)
    for r_frac in np.linspace(0.08, 0.22, 8):
        if time.time() > deadline:
            break
        for offset in [0.22, 0.25, 0.28]:
            cell = _make_four_circles_cell(size, r_frac, offset)
            score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
            if score > best_score:
                best_score = score
                best_cell = cell
                if best_score > 0.01:
                    score_ref = _evaluate_gap(cell, size, N, k_points_full)
                    if score_ref > best_score:
                        best_score = score_ref
                        best_cell = cell

    # 3. Square holes in dielectric background (1x1 lattice of holes)
    for hole_frac in np.linspace(0.15, 0.35, 10):
        if time.time() > deadline:
            break
        cell = _make_square_hole_cell(size, hole_frac)
        score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
            if best_score > 0.01:
                score_ref = _evaluate_gap(cell, size, N, k_points_full)
                if score_ref > best_score:
                    best_score = score_ref
                    best_cell = cell

    # 4. 4x4 Grid of Rods
    for r_frac in np.linspace(0.02, 0.12, 10):
        if time.time() > deadline:
            break
        cell = _make_grid_rods_cell(size, r_frac, 4)
        score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
            if best_score > 0.01:
                score_ref = _evaluate_gap(cell, size, N, k_points_full)
                if score_ref > best_score:
                    best_score = score_ref
                    best_cell = cell

    # 5. 4x4 Grid of Holes (Dielectric background, air holes)
    for r_frac in np.linspace(0.02, 0.12, 10):
        if time.time() > deadline:
            break
        cell = _make_grid_holes_cell(size, r_frac, 4)
        score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
            if best_score > 0.01:
                score_ref = _evaluate_gap(cell, size, N, k_points_full)
                if score_ref > best_score:
                    best_score = score_ref
                    best_cell = cell
    
    # 6. 2x2 Grid of Rods (Larger rods)
    for r_frac in np.linspace(0.08, 0.20, 10):
        if time.time() > deadline:
            break
        cell = _make_grid_rods_cell(size, r_frac, 2)
        score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
            if best_score > 0.01:
                score_ref = _evaluate_gap(cell, size, N, k_points_full)
                if score_ref > best_score:
                    best_score = score_ref
                    best_cell = cell

    # 7. 2x2 Grid of Holes
    for r_frac in np.linspace(0.08, 0.20, 10):
        if time.time() > deadline:
            break
        cell = _make_grid_holes_cell(size, r_frac, 2)
        score = _evaluate_gap_fast(cell, size, N, k_points_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
            if best_score > 0.01:
                score_ref = _evaluate_gap(cell, size, N, k_points_full)
                if score_ref > best_score:
                    best_score = score_ref
                    best_cell = cell

    # If no valid cell found or score is still negative, create a simple symmetric pattern
    if best_cell is None or best_score < 0:
        # Fallback to a known good structure if possible, or just return the best found
        if best_cell is None:
            best_cell = _make_circle_cell(size, 0.3)
        else:
            # If best score is negative, we might want to try a few more random symmetric patterns?
            # But for now, return the best found. The evaluator will score it (likely 0 or negative).
            # To ensure we don't return a *worse* negative, we stick to best.
            pass

    return best_cell

def _make_circle_cell(size, r_frac):
    """Create a cell with a single central circle of radius r_frac * size."""
    cell = [[0] * size for _ in range(size)]
    r = r_frac * size
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < r * r:
                cell[i][j] = 1
    return cell

def _make_four_circles_cell(size, r_frac, offset):
    """Create a cell with four circles at diagonal positions."""
    cell = [[0] * size for _ in range(size)]
    r = r_frac * size
    centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    cell[i][j] = 1
                    break
    return cell

def _make_square_hole_cell(size, hole_frac):
    """Create a dielectric background with a central square hole."""
    cell = [[1] * size for _ in range(size)]
    half_hole = hole_frac / 2
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if abs(x) < half_hole and abs(y) < half_hole:
                cell[i][j] = 0
    return cell

def _make_grid_rods_cell(size, r_frac, grid_size):
    """Create a grid of circular rods."""
    cell = [[0] * size for _ in range(size)]
    r = r_frac * size
    step = size / grid_size
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            # Find nearest grid center
            # Grid centers are at -0.5 + (k+0.5)*step for k in 0..grid_size-1
            # Or simply: distance to nearest multiple of step (adjusted for center)
            # Let's map x,y to grid indices
            # x ranges from -0.5 to 0.5
            # Grid lines at -0.5 + k*step
            # Nearest center x_c = round((x + 0.5) / step - 0.5) * step - 0.5 ?
            # Simpler:
            gx = np.round((x + 0.5) / step) * step - 0.5
            gy = np.round((y + 0.5) / step) * step - 0.5
            # Check if inside any rod
            dx = x - gx
            dy = y - gy
            if dx * dx + dy * dy < r * r:
                cell[i][j] = 1
    return cell

def _make_grid_holes_cell(size, r_frac, grid_size):
    """Create a dielectric background with a grid of circular holes."""
    cell = [[1] * size for _ in range(size)]
    r = r_frac * size
    step = size / grid_size
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            gx = np.round((x + 0.5) / step) * step - 0.5
            gy = np.round((y + 0.5) / step) * step - 0.5
            dx = x - gx
            dy = y - gy
            if dx * dx + dy * dy < r * r:
                cell[i][j] = 0
    return cell

def _generate_k_points():
    """Generate wave vectors for the irreducible Brillouin zone.
    Returns a list of (kx, ky) in units of pi/a."""
    k_points = []
    # Along Gamma-M edge: kx = 0, ky from 0 to 0.5
    for ky in np.linspace(0, 0.5, 8):
        k_points.append((0.0, ky))
    # Along M-X edge: kx from 0 to 0.5, ky = 0.5
    for kx in np.linspace(0, 0.5, 8):
        k_points.append((kx, 0.5))
    # Along X-Gamma edge: kx from 0.5 to 0, ky = 0
    for kx in np.linspace(0.5, 0, 8):
        k_points.append((kx, 0.0))
    # Interior points
    for kx in [0.1, 0.2, 0.3]:
        for ky in [0.1, 0.2, 0.3]:
            if kx <= ky:  # irreducible wedge
                k_points.append((kx, ky))
    return k_points

def _compute_fourier_coefficients(cell, size, N):
    """Compute exact Fourier coefficients of the pixel map.
    Returns a 2D array of shape (N, N) with coefficients for e^{i(2pi/a)(nx*x + ny*y)}."""
    n_max = (N - 1) // 2
    coeffs = np.zeros((N, N), dtype=complex)
    eps_low = 1.0
    eps_high = 12.0
    
    # For each pixel, compute its contribution to each Fourier mode
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            eps = eps_high if cell[i][j] == 1 else eps_low
            for ni in range(N):
                nx = ni - n_max
                for nj in range(N):
                    ny = nj - n_max
                    # Integral of eps * exp(-i(2pi/a)(nx*x + ny*y)) over pixel area
                    # Pixel extends from x-0.5/size to x+0.5/size, etc.
                    phase_x = 2 * np.pi * nx * x
                    phase_y = 2 * np.pi * ny * y
                    # For small pixels, approximate as point sample
                    coeffs[ni, nj] += eps * np.exp(-1j * (phase_x + phase_y))
    
    coeffs /= (size * size)
    return coeffs

def _solve_tm_bands(coeffs, kx, ky, N, n_bands=5):
    """Solve for TM bands using plane-wave method.
    Returns sorted list of omega values (in units of 2pi*c/a)."""
    n_max = (N - 1) // 2
    kx_norm = kx  # in units of pi/a
    ky_norm = ky
    
    # Precompute G vectors
    G_vectors = []
    for ni in range(N):
        nx = ni - n_max
        for nj in range(N):
            ny = nj - n_max
            G_vectors.append((nx, ny))
            
    num_G = len(G_vectors)
    
    # Build inverse epsilon matrix
    eps_inv = np.zeros(num_G)
    for idx, (nx, ny) in enumerate(G_vectors):
        ni_idx = nx + n_max
        nj_idx = ny + n_max
        eps_val = coeffs[ni_idx, nj_idx]
        if abs(eps_val) < 1e-10:
            eps_val = 1.0  # fallback
        eps_inv[idx] = 1.0 / eps_val
    
    # Build the matrix H
    # H[m, n] = (k+G_m)·(k+G_n) / eps(k+G_n)
    # Note: (k+G_m)·(k+G_n) is NOT simply dot product of vectors in G-space.
    # k is a vector. G_m, G_n are vectors.
    # k + G_m = (kx + G_m_x, ky + G_m_y)
    # k + G_n = (kx + G_n_x, ky + G_n_y)
    # Dot product = (kx + G_m_x)(kx + G_n_x) + (ky + G_m_y)(ky + G_n_y)
    
    H = np.zeros((num_G, num_G), dtype=complex)
    for m in range(num_G):
        nx_m, ny_m = G_vectors[m]
        kx_m = kx_norm + nx_m
        ky_m = ky_norm + ny_m
        for n in range(num_G):
            nx_n, ny_n = G_vectors[n]
            kx_n = kx_norm + nx_n
            ky_n = ky_norm + ny_n
            dot = kx_m * kx_n + ky_m * ky_n
            H[m, n] = dot * eps_inv[n]
    
    # Solve eigenvalue problem
    # H is Hermitian?
    # H[m,n] = dot(m,n) / eps(n)
    # H[n,m] = dot(n,m) / eps(m)
    # dot(m,n) is real. eps(n) is complex? No, eps is real and positive for lossless.
    # But coeffs are complex. 1/eps is complex.
    # So H is NOT Hermitian in general?
    # Wait, the operator for TM is (k+G)·(k+G) / eps(k+G).
    # The matrix element is <G_m | (k+G)·(k+G)/eps(k+G) | G_n >.
    # In the basis of plane waves, the operator is diagonal in G-space? No.
    # The equation is:
    # sum_n (k+G_m)·(k+G_n) / eps(k+G_n) c_n = omega^2 c_m
    # This is a generalized eigenvalue problem if we don't symmetrize.
    # However, we can define a new basis vector phi_m = c_m / sqrt(eps_m) ?
    # Let's stick to the direct matrix construction.
    # Is it Hermitian?
    # H[m,n] = D[m,n] * E_inv[n]
    # H[n,m] = D[n,m] * E_inv[m]
    # D[m,n] = D[n,m] (real)
    # If E_inv[n] != E_inv[m], H is not Hermitian.
    # We should use np.linalg.eig instead of eigvalsh if it's not Hermitian.
    # But physically the eigenvalues must be real.
    # Let's use np.linalg.eigvalsh only if we symmetrize.
    # Symmetrized form: A[m,n] = D[m,n] / sqrt(eps_m eps_n)
    # Then A is Hermitian if eps is real.
    # Let's build A.
    
    A = np.zeros((num_G, num_G), dtype=complex)
    sqrt_eps_inv = np.sqrt(eps_inv)
    for m in range(num_G):
        nx_m, ny_m = G_vectors[m]
        kx_m = kx_norm + nx_m
        ky_m = ky_norm + ny_m
        for n in range(num_G):
            nx_n, ny_n = G_vectors[n]
            kx_n = kx_norm + nx_n
            ky_n = ky_norm + ny_n
            dot = kx_m * kx_n + ky_m * ky_n
            A[m, n] = dot * sqrt_eps_inv[m] * sqrt_eps_inv[n]
            
    # A is Hermitian
    eigenvalues = np.linalg.eigvalsh(A)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)
    
    # omega^2 = eigenvalue * (2pi/a)^2
    # In units of 2pi*c/a, omega = sqrt(eigenvalue)
    omega_values = np.sqrt(np.maximum(eigenvalues, 0))
    
    return omega_values[:n_bands]

def _evaluate_gap(cell, size, N, k_points):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    coeffs = _compute_fourier_coefficients(cell, size, N)
    
    max_band4 = 0.0
    min_band5 = float('inf')
    
    for kx, ky in k_points:
        try:
            omega = _solve_tm_bands(coeffs, kx, ky, N, n_bands=5)
            if len(omega) >= 5:
                band4 = omega[3]  # 0-indexed, so band 4 is index 3
                band5 = omega[4]  # band 5 is index 4
                max_band4 = max(max_band4, band4)
                min_band5 = min(min_band5, band5)
        except Exception:
            continue
    
    if max_band4 == 0.0 or min_band5 == float('inf'):
        return -1.0
    
    if min_band5 <= max_band4:
        return -1.0  # bands overlap
    
    gap = min_band5 - max_band4
    midgap = (min_band5 + max_band4) / 2.0
    
    if midgap < 1e-10:
        return -1.0
    
    return gap / midgap

def _evaluate_gap_fast(cell, size, N, k_points):
    """Fast evaluation using fewer k-points."""
    return _evaluate_gap(cell, size, N, k_points)

def _check_symmetry(cell, size):
    """Check if cell has square symmetry: 180-degree rotation, transpose, and reflections."""
    # Check 180-degree rotation symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                return False
    
    # Check transpose symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                return False
    
    # Check horizontal reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                return False
    
    # Check vertical reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                return False
    
    return True
# EVOLVE-BLOCK-END
