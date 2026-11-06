"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
from scipy import linalg

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell maximizing the TM 4-5 band gap.
    Uses a plane-wave method with exact Fourier coefficients of the pixel map.
    """
    t_start = time.time()
    DEADLINE = 200.0  # safety margin before 230s kill

    size = problem["pixels"]  # 32
    eps_high = problem["eps_high"]  # 12
    eps_low = problem["eps_low"]  # 1
    N_pw = 17  # 17x17 = 289 plane waves

    def time_ok():
        return (time.time() - t_start) < DEADLINE

    def make_symmetric_cell(binary_32x32):
        """
        Enforce full D4 symmetry: row-reverse, col-reverse, transpose.
        Input: 32x32 numpy array of 0/1
        Output: 32x32 numpy array with full symmetry
        """
        cell = binary_32x32.astype(np.float64).copy()
        # Row reverse symmetry
        cell = (cell + cell[::-1, :]) / 2.0
        # Col reverse symmetry
        cell = (cell + cell[:, ::-1]) / 2.0
        # Transpose symmetry
        cell = (cell + cell.T) / 2.0
        # Round to binary
        cell = (cell > 0.5).astype(np.float64)
        # Re-apply all symmetries after rounding
        cell = (cell + cell[::-1, :]) / 2.0
        cell = (cell + cell[:, ::-1]) / 2.0
        cell = (cell + cell.T) / 2.0
        cell = (cell > 0.5).astype(np.float64)
        return cell

    def get_fourier_coeffs(cell, n_max):
        """
        Compute Fourier coefficients of the dielectric function.
        cell: 32x32 binary array (1 = dielectric, 0 = air)
        n_max: maximum harmonic order
        Returns dict mapping (nx, ny) -> coefficient for eps(x,y)
        
        eps(x,y) = eps_low + (eps_high - eps_low) * f(x,y)
        where f is 0/1 indicator.
        
        We need 1/eps for TM mode.
        """
        eps_map = eps_low + (eps_high - eps_low) * cell
        inv_eps = 1.0 / eps_map
        
        # Compute DFT of inv_eps
        F = np.fft.fft2(inv_eps)
        F /= (size * size)
        
        # Also compute for eps (for completeness, though TM uses 1/eps)
        F_eps = np.fft.fft2(eps_map)
        F_eps /= (size * size)
        
        # Extract coefficients for |nx|, |ny| <= n_max
        coeffs_inv_eps = {}
        coeffs_eps = {}
        for nx in range(-n_max, n_max + 1):
            for ny in range(-n_max, n_max + 1):
                # DFT index: positive freqs at [0..N/2], negative at [N/2+1..N-1]
                ix = nx % size
                iy = ny % size
                coeffs_inv_eps[(nx, ny)] = F[ix, iy]
                coeffs_eps[(nx, ny)] = F_eps[ix, iy]
        
        return coeffs_inv_eps, coeffs_eps

    def compute_band_ratio_fast(coeffs_inv_eps, coeffs_eps, kx, ky, n_bands=5):
        """
        Compute the ratio 2*(min_omega_5 - max_omega_4)/(min_omega_5 + max_omega_4)
        for TM bands at given wavevector (kx, ky) in units of 2*pi/a.
        
        For TM: (k - G)^2 * (1/eps) * E = omega^2 * E
        Matrix: M[i,j] = (k - G_i) * (k - G_j) * sum_{G'} (1/eps)_{G'} * delta_{G_i - G_j, G'}
        
        Actually for TM:
        The eigenvalue equation is:
        sum_{G'} (k - G_i)·(k - G_j) * (1/eps)_{G_i - G_j} * E_{G'} = omega^2 * E_{G_i}
        
        Wait, let me be more careful. For TM polarization in 2D:
        The Maxwell equation gives:
        (k - G)^2 / eps(r) * E(k-G) = omega^2 E(k)
        
        In Fourier space:
        sum_{G'} (k - G_i)^2 * (1/eps)_{G_i - G'} * E_{G'} = omega^2 * E_{G_i}
        
        So the matrix is:
        M[i,j] = (k - G_i)^2 * (1/eps)_{G_i - G_j}
        
        where (1/eps)_{G_i - G_j} is the Fourier coefficient of 1/eps at frequency G_i - G_j.
        """
        # Build list of G vectors
        G_list = []
        for nx in range(-N_pw // 2, N_pw // 2 + 1):
            for ny in range(-N_pw // 2, N_pw // 2 + 1):
                G_list.append((2 * np.pi * nx, 2 * np.pi * ny))
        
        N = len(G_list)
        M = np.zeros((N, N), dtype=np.complex128)
        
        for i in range(N):
            kx_i = kx - G_list[i][0]
            ky_i = ky - G_list[i][1]
            k_sq_i = kx_i**2 + ky_i**2
            for j in range(N):
                dnx = G_list[i][0] / (2 * np.pi) - G_list[j][0] / (2 * np.pi)
                dny = G_list[i][1] / (2 * np.pi) - G_list[j][1] / (2 * np.pi)
                dnx = int(round(dnx))
                dny = int(round(dny))
                coeff = coeffs_inv_eps.get((dnx, dny), 0.0)
                M[i, j] = k_sq_i * coeff
        
        # Solve eigenvalue problem
        # M is Hermitian if the system is properly set up
        # Use eigh for Hermitian matrices
        M = (M + M.conj().T) / 2.0  # Enforce Hermiticity
        try:
            eigvals = linalg.eigvalsh(M)
        except:
            eigvals = np.sort(np.real(linalg.eigvals(M)))
        
        eigvals = np.sort(np.real(eigvals))
        
        if len(eigvals) < 5:
            return -1.0
        
        omega_4_max = eigvals[3]  # 4th band (0-indexed: 3)
        omega_5_min = eigvals[4]  # 5th band (0-indexed: 4)
        
        if omega_5_min + omega_4_max < 1e-10:
            return -1.0
        
        return 2.0 * (omega_5_min - omega_4_max) / (omega_5_min + omega_4_max)

    def compute_band_ratio_full(coeffs_inv_eps, coeffs_eps, kx, ky, n_bands=5):
        """Same as fast but with more careful handling."""
        return compute_band_ratio_fast(coeffs_inv_eps, coeffs_eps, kx, ky, n_bands)

    def evaluate_cell(cell):
        """
        Evaluate the gap-midgap ratio for a given cell.
        Returns the minimum ratio over 30 wave vectors in the irreducible BZ.
        """
        coeffs_inv_eps, coeffs_eps = get_fourier_coeffs(cell, N_pw // 2)
        
        # 30 wave vectors in irreducible BZ
        # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi/a)
        # Actually: 0 <= kx <= ky <= 1 - kx for the first BZ
        # Let me use a standard set of 30 points
        
        k_points = []
        # Along M-X edge: kx = ky, from 0 to 0.5
        for t in np.linspace(0, 0.5, 6):
            k_points.append((t, t))
        # Along Gamma-X: ky = 0, kx from 0 to 0.5
        for t in np.linspace(0, 0.5, 6):
            k_points.append((t, 0))
        # Along Gamma-M: kx = ky, from 0 to 0.5 (already covered)
        # Interior points
        for t1 in np.linspace(0.1, 0.4, 4):
            for t2 in np.linspace(0.1, 0.4, 4):
                if t1 <= t2 and t1 + t2 <= 0.8:
                    k_points.append((t1, t2))
        
        # Deduplicate and ensure we have about 30
        unique_kp = []
        for kp in k_points:
            if not any(abs(kp[0] - u[0]) < 1e-6 and abs(kp[1] - u[1]) < 1e-6 for u in unique_kp):
                unique_kp.append(kp)
        
        # Take first 30
        k_points = unique_kp[:30]
        
        min_ratio = np.inf
        for kx, ky in k_points:
            if not time_ok():
                break
            ratio = compute_band_ratio_fast(coeffs_inv_eps, coeffs_eps, kx, ky)
            if ratio < min_ratio:
                min_ratio = ratio
        
        return min_ratio

    def generate_harmonic_cell(lam, threshold, size=32):
        """
        Generate a symmetric cell from a harmonic polynomial field.
        f(x,y) = x^4 + y^4 - 2*lam*x*y*(x^2+y^2)
        Cell is 1 where f > threshold.
        """
        cell = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                val = x**4 + y**4 - 2 * lam * x * y * (x**2 + y**2)
                if val > threshold:
                    cell[i, j] = 1.0
        return cell

    def generate_rod_cell(radius, offset, size=32):
        """Generate cell with four rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for dx, dy in [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]:
                    if (x - dx)**2 + (y - dy)**2 < radius**2:
                        cell[i, j] = 1.0
                        break
        return cell

    def generate_cross_cell(width, length, size=32):
        """Generate a cross/plus shape."""
        cell = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if (abs(x) < width/2 and abs(y) < length/2) or (abs(y) < width/2 and abs(x) < length/2):
                    cell[i, j] = 1.0
        return cell

    def generate_flower_cell(n_petals, r_inner, r_outer, size=32):
        """Generate a flower-like shape with n petals."""
        cell = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x**2 + y**2)
                if r == 0:
                    cell[i, j] = 1.0
                    continue
                theta = np.arctan2(y, x)
                # Modulate radius
                r_mod = r_inner + (r_outer - r_inner) * (1 + np.cos(n_petals * theta)) / 2
                if r < r_mod:
                    cell[i, j] = 1.0
        return cell

    best_cell = None
    best_score = -np.inf

    # Strategy 1: Search over harmonic polynomial parameters
    for lam in np.linspace(-1.0, 1.0, 15):
        if not time_ok():
            break
        for threshold in np.linspace(-0.05, 0.05, 15):
            if not time_ok():
                break
            cell = generate_harmonic_cell(lam, threshold)
            cell = make_symmetric_cell(cell)
            fill = np.mean(cell)
            if fill < 0.05 or fill > 0.95:
                continue
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()

    # Strategy 2: Search over rod parameters
    for radius in np.linspace(0.05, 0.25, 10):
        if not time_ok():
            break
        for offset in np.linspace(0.05, 0.35, 10):
            if not time_ok():
                break
            cell = generate_rod_cell(radius, offset)
            cell = make_symmetric_cell(cell)
            fill = np.mean(cell)
            if fill < 0.05 or fill > 0.95:
                continue
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()

    # Strategy 3: Search over cross parameters
    for width in np.linspace(0.05, 0.3, 8):
        if not time_ok():
            break
        for length in np.linspace(0.2, 0.5, 8):
            if not time_ok():
                break
            cell = generate_cross_cell(width, length)
            cell = make_symmetric_cell(cell)
            fill = np.mean(cell)
            if fill < 0.05 or fill > 0.95:
                continue
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()

    # Strategy 4: Search over flower parameters
    for n_petals in [4, 6, 8]:
        if not time_ok():
            break
        for r_inner in np.linspace(0.05, 0.15, 5):
            if not time_ok():
                break
            for r_outer in np.linspace(0.15, 0.35, 5):
                if not time_ok():
                    break
                cell = generate_flower_cell(n_petals, r_inner, r_outer)
                cell = make_symmetric_cell(cell)
                fill = np.mean(cell)
                if fill < 0.05 or fill > 0.95:
                    continue
                score = evaluate_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()

    # Fallback
    if best_cell is None:
        best_cell = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    best_cell[i, j] = 1.0
        best_cell = make_symmetric_cell(best_cell)

    # Final symmetry enforcement
    best_cell = make_symmetric_cell(best_cell)
    best_cell = (best_cell > 0.5).astype(int)

    # Convert to nested list
    result = best_cell.tolist()

    return result
# EVOLVE-BLOCK-END
