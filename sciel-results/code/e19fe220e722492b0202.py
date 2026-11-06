"""Design a 32x32 symmetric photonic-crystal unit cell maximizing the TM 4-5 band gap."""
import time
import numpy as np
from scipy.fft import fft2, ifft2

_DEADLINE_MARGIN = 15.0  # seconds before the 230s wall-clock kill

def _build_symmetric_cell(size, quadrant):
    """Expand a top-left quadrant (size//2 x size//2) to a full size x size cell
    with 180° rotational and reflection symmetry about the cell centre."""
    half = size // 2
    # quadrant is rows 0..half-1, cols 0..half-1
    # Full cell:
    #   top-left = quadrant
    #   top-right = quadrant flipped horizontally
    #   bottom-left = quadrant flipped vertically
    #   bottom-right = quadrant flipped both
    qr = quadrant[:, ::-1]   # flip cols
    qb = quadrant[::-1, :]   # flip rows
    qbr = quadrant[::-1, ::-1]
    top = np.hstack([quadrant, qr])
    bottom = np.hstack([qb, qbr])
    cell = np.vstack([top, bottom])
    return cell.tolist()


def _fourier_eps(cell_array, size):
    """Compute the inverse-permittivity Fourier coefficients for the PWE solver.
    For TM: the matrix uses 1/eps(k-G, k-G') = delta_{G,G'}/eps_G.
    We need the Fourier series of 1/eps(x)."""
    # cell_array is size x size, values 0 (air) or 1 (dielectric)
    eps = np.where(cell_array == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps
    # FFT to get Fourier coefficients
    # The pixel grid represents the unit cell [0,1) x [0,1)
    # Fourier coefficient c_{mn} = sum_{i,j} f(i,j) exp(-2pi i (m i/N + n j/N)) / N^2
    # numpy fft2 gives sum f(i,j) exp(-2pi i (m i/N + n j/N))
    # So c_{mn} = fft2(f) / N^2
    fft_inv = fft2(inv_eps)
    # fft_inv[m, n] corresponds to frequency (m, n) but with wrap-around
    # For m in 0..N-1, the actual frequency index is m if m <= N/2, else m-N
    # We'll index by the raw FFT index and handle the mapping
    return fft_inv / (size * size)


def _solve_pwe(kx, ky, cell_array, size, num_pw=17):
    """Solve the TM PWE eigenvalue problem at wavevector (kx, ky) in units of 2pi/a.
    Returns the list of eigenvalues (omega^2 in units of (2pi/a)^2 / c^2... actually
    we work in dimensionless units where a=1, so omega is in units of 2pi/a).
    
    For TM polarization:
    (|G+k|^2) * a_{G'} - sum_{G''} (1/eps_{G-G''}) * a_{G''} = omega^2 * a_{G'}
    
    Actually the standard form is:
    sum_{G'} [ |k+G|^2 delta_{G,G'} - (1/eps_{G-G'}) ] a_{G'} = omega^2 a_{G}
    
    where eps_{G-G'} are the Fourier coefficients of eps (not 1/eps).
    Wait, let me reconsider. For TM:
    The equation is:
    (1/eps(k)) * |k|^2 * E(k) = omega^2 E(k)
    
    In plane waves:
    sum_{G'} (1/eps_{G-G'}) |k+G'|^2 a_{G'} = omega^2 a_{G}
    
    So the matrix element M[G, G'] = (1/eps_{G-G'}) * |k+G'|^2
    and we solve M a = omega^2 a.
    
    But 1/eps_{G-G'} is the Fourier coefficient of 1/eps.
    """
    # Build the list of G vectors
    # Use a square grid of G vectors from -num_pw//2 to num_pw//2
    half = num_pw // 2
    G_vals = np.arange(-half, half + 1)  # e.g., -8..8 for num_pw=17
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_list = np.column_stack([Gx.ravel(), Gy.ravel()])
    N = len(G_list)
    
    # k in units of 2pi/a (dimensionless)
    k_vec = np.array([kx, ky])
    
    # k+G for each G
    KG = k_vec + G_list  # N x 2
    
    # |k+G|^2 for each G
    KG2 = np.sum(KG**2, axis=1)  # N
    
    # Fourier coefficients of 1/eps
    fft_inv = _fourier_eps(cell_array, size)
    
    # For each pair (G_i, G_j), we need 1/eps_{G_i - G_j}
    # G_i - G_j has indices (G_i_x - G_j_x, G_i_y - G_j_y)
    # The FFT index for frequency (m, n) where m can be negative:
    #   if m >= 0: fft_index = m
    #   if m < 0: fft_index = m + size
    # Similarly for n.
    
    # Build the matrix
    # M[i, j] = (1/eps_{G_i - G_j}) * |k + G_j|^2
    # We can vectorize: for each i, compute G_i - G_j for all j
    
    # Precompute: for each possible difference (dx, dy) in range [-half*2, half*2]
    # Actually G_i - G_j can range from -2*half to 2*half in each component
    
    # Let's just loop over i for clarity, but vectorize over j
    M = np.zeros((N, N), dtype=complex)
    
    # Map from (dx, dy) to FFT index
    # dx = G_i_x - G_j_x, dy = G_i_y - G_j_y
    # FFT index: (dx mod size, dy mod size)
    
    for i in range(N):
        dx = G_list[i, 0] - G_list[:, 0]  # N values
        dy = G_list[i, 1] - G_list[:, 1]  # N values
        # Convert to FFT indices
        idx_x = np.mod(dx, size)
        idx_y = np.mod(dy, size)
        eps_coeff = fft_inv[idx_x, idx_y]  # 1/eps_{G_i - G_j} for each j
        M[i, :] = eps_coeff * KG2  # M[i,j] = (1/eps_{G_i-G_j}) * |k+G_j|^2
    
    # Solve eigenvalue problem
    # M is Hermitian (should be, up to numerical error)
    # Use scipy's eigh for Hermitian matrices
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(M)
    # eigvals are omega^2
    omega2 = eigvals.real  # should be real
    # Sort (eigh already sorts in ascending order)
    # omega = sqrt(omega2)
    omega = np.sqrt(np.maximum(omega2, 0))
    return omega


def _compute_gap(omega_all):
    """Given a list of omega arrays (one per k-point), compute the gap-midgap ratio
    between bands 4 and 5 (0-indexed: bands 3 and 4)."""
    # omega_all: list of arrays, each of length num_bands
    # max of band 4 (index 3) across all k-points
    max_band4 = max(omega[3] for omega in omega_all)
    # min of band 5 (index 4) across all k-points
    min_band5 = min(omega[4] for omega in omega_all)
    gap = min_band5 - max_band4
    midgap = (min_band5 + max_band4) / 2.0
    if midgap <= 0:
        return -1.0
    return gap / midgap


def design(problem):
    """Design a symmetric 32x32 photonic-crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 230.0 - _DEADLINE_MARGIN
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    
    half = size // 2  # 16
    
    # k-points in the irreducible Brillouin zone
    # The BZ is [0, 1] x [0, 1] in units of 2pi/a
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 1, and kx + ky <= 1
    # Actually the standard irreducible BZ for a square lattice is the triangle
    # with vertices (0,0), (0.5, 0), (0.5, 0.5) -- no, that's not right.
    # 
    # The full BZ is [-0.5, 0.5] x [-0.5, 0.5] or equivalently [0,1] x [0,1].
    # The irreducible BZ (1/8 of the BZ) is: 0 <= kx <= ky, kx >= 0, ky >= 0,
    # and we use the symmetry kx -> 1-kx, ky -> 1-ky, swap kx<->ky.
    # The irreducible region is: 0 <= kx <= 0.5, kx <= ky <= 0.5, and kx+ky <= 0.5? 
    # No. Let me think again.
    # 
    # For a square lattice, the BZ is a square. The irreducible BZ is 1/8 of it.
    # Using the symmetries: (kx,ky) -> (ky,kx), (kx,ky) -> (1-kx,ky), etc.
    # The irreducible BZ is the triangle with vertices (0,0), (0.5,0), (0.5,0.5).
    # Wait, that's 1/4. Let me reconsider.
    # 
    # Actually the standard irreducible BZ for a square lattice (with the full
    # point group D4) is the triangle with vertices (0,0), (1/2, 0), (1/2, 1/2).
    # This is 1/8 of the full BZ area.
    # 
    # Let me use k-points along the boundaries and interior of this triangle.
    
    # Generate k-points
    k_points = []
    # Along kx axis: (kx, 0) for kx in [0, 0.5]
    for kx in np.linspace(0, 0.5, 6):
        k_points.append((kx, 0.0))
    # Along ky = kx: (kx, kx) for kx in [0, 0.5]
    for kx in np.linspace(0, 0.5, 6):
        k_points.append((kx, kx))
    # Along ky = 0.5: (kx, 0.5) for kx in [0, 0.5]
    for kx in np.linspace(0, 0.5, 6):
        k_points.append((kx, 0.5))
    # Interior points
    k_points.append((0.2, 0.3))
    k_points.append((0.3, 0.4))
    k_points.append((0.1, 0.4))
    k_points.append((0.25, 0.25))
    
    # Remove duplicates
    k_points = list(set(k_points))
    
    # PWE parameters
    num_pw = 13  # 13x13 = 169 plane waves, manageable
    
    def _evaluate_cell(quadrant):
        """Evaluate the gap-midgap ratio for a given quadrant."""
        cell_array = np.array(_build_symmetric_cell(size, quadrant))
        omega_all = []
        for kx, ky in k_points:
            omega = _solve_pwe(kx, ky, cell_array, size, num_pw)
            omega_all.append(omega)
        return _compute_gap(omega_all)
    
    # Start with the seed design: centered rod of radius 0.2
    def _make_quadrant_rod(radius_frac):
        """Create a quadrant with a circular rod centered at (0.5, 0.5) in full-cell coords.
        In quadrant coords (0..15, 0..15), the center is at (15.5, 15.5) in full-cell
        but in the quadrant, the center of the full cell is at the corner (15.5, 15.5).
        Wait, the quadrant is the top-left 16x16. The center of the full 32x32 cell
        is at (15.5, 15.5) in 0-indexed pixel centers... actually pixel centers are at
        (i+0.5)/32 - 0.5 for i in 0..31.
        
        For the quadrant (top-left 16x16), pixel (i,j) in quadrant corresponds to
        full-cell pixel (i, j). The center of the full cell is at (16, 16) in pixel
        index (between pixels 15 and 16). So in the quadrant, the center is at the
        bottom-right corner.
        
        For a rod centered at the cell center, in the quadrant we want pixels close
        to (15.5, 15.5) (the corner) to be dielectric.
        """
        quad = np.zeros((half, half), dtype=int)
        for i in range(half):
            for j in range(half):
                # Full cell pixel center: ((i+0.5)/size - 0.5, (j+0.5)/size - 0.5)
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius_frac ** 2:
                    quad[i, j] = 1
        return quad
    
    # Evaluate seed
    seed_quad = _make_quadrant_rod(0.2)
    best_score = _evaluate_cell(seed_quad)
    best_quad = seed_quad.copy()
    
    # Try a few different rod radii and positions
    radii = [0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3]
    for r in radii:
        if time.time() > deadline:
            break
        quad = _make_quadrant_rod(r)
        score = _evaluate_cell(quad)
        if score > best_score:
            best_score = score
            best_quad = quad.copy()
    
    # Try shifting the rod center within the quadrant
    # The rod center in full-cell coords can be varied
    for dx in [-0.05, 0, 0.05]:
        for dy in [-0.05, 0, 0.05]:
            if time.time() > deadline:
                break
            quad = np.zeros((half, half), dtype=int)
            for i in range(half):
                for j in range(half):
                    x = (i + 0.5) / size - 0.5 + dx
                    y = (j + 0.5) / size - 0.5 + dy
                    if x * x + y * y < 0.22 ** 2:
                        quad[i, j] = 1
            score = _evaluate_cell(quad)
            if score > best_score:
                best_score = score
                best_quad = quad.copy()
    
    # Try a square rod (different shape)
    for half_size in [2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
        quad = np.zeros((half, half), dtype=int)
        # Square centered at the corner of the quadrant (which is the cell center)
        # In quadrant coords, the cell center is at (half-0.5, half-0.5) approximately
        # Actually the center is at the corner, so the square should be in the bottom-right
        for i in range(half - half_size, half):
            for j in range(half - half_size, half):
                quad[i, j] = 1
        score = _evaluate_cell(quad)
        if score > best_score:
            best_score = score
            best_quad = quad.copy()
    
    # Try a ring shape (annulus)
    for r_inner in [0.05, 0.1, 0.15]:
        for r_outer in [0.25, 0.3, 0.35]:
            if time.time() > deadline:
                break
            if r_inner >= r_outer:
                continue
            quad = np.zeros((half, half), dtype=int)
            for i in range(half):
                for j in range(half):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    r2 = x * x + y * y
                    if r_inner ** 2 < r2 < r_outer ** 2:
                        quad[i, j] = 1
            score = _evaluate_cell(quad)
            if score > best_score:
                best_score = score
                best_quad = quad.copy()
    
    # Local optimization: flip individual pixels in the quadrant
    # Use a simple hill-climbing approach
    current_quad = best_quad.copy()
    current_score = best_score
    
    # Sort pixels by some heuristic (e.g., distance from cell center)
    pixels = []
    for i in range(half):
        for j in range(half):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            pixels.append((i, j, x*x + y*y))
    pixels.sort(key=lambda p: p[2])
    
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations and time.time() < deadline:
        improved = False
        iteration += 1
        for i, j, _ in pixels:
            if time.time() > deadline:
                break
            # Try flipping pixel (i, j)
            old_val = current_quad[i, j]
            current_quad[i, j] = 1 - old_val
            new_score = _evaluate_cell(current_quad)
            if new_score > current_score:
                current_score = new_score
                best_score = new_score
                best_quad = current_quad.copy()
                improved = True
            else:
                current_quad[i, j] = old_val
    
    # Return the best cell found
    return _build_symmetric_cell(size, best_quad)
