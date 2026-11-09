"""Photonic crystal band-gap optimizer using a fast TM plane-wave solver + hill-climbing."""
import time
import math
import random
import numpy as np
from scipy.fft import fft2, ifft2

# ---- Fast TM band solver (plane-wave, Galerkin) ----
# We precompute the inverse-epsilon matrix in reciprocal space for a 32x32 cell.
# The method: for TM polarization, the eigenvalue problem is:
#   sum_G'  [ (k+G)^2 / (2*pi)^2 ] * delta_{G,G'} * eps_inv(G-G') * c_{G'} = omega^2 * c_G
# where eps_inv is the Fourier transform of 1/eps(r).

def _build_eps_inv_matrix(cell, n_pw):
    """
    Build the 2D Fourier coefficients of 1/eps(r) on a grid of n_pw x n_pw G-vectors.
    cell: 32x32 nested list, 1=eps_high, 0=eps_low
    n_pw: number of G-vectors along each direction (odd, e.g. 17)
    Returns: eps_inv as a 2D numpy array indexed by (m_idx, n_idx) where
             G = (m, n) * 2*pi/L, m,n in [-n_pw//2, n_pw//2]
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    eps_grid = np.array(cell, dtype=float)
    eps_grid = np.where(eps_grid > 0.5, eps_high, eps_low)
    inv_eps = 1.0 / eps_grid

    # FFT of inv_eps to get Fourier coefficients
    # We need coefficients for G = (m, n) * 2*pi/L where m, n are integers
    # Using FFT: F[m,n] = sum_{x,y} f[x,y] * exp(-2*pi*i*(m*x/N + n*y/N))
    # The physical coefficient for G=(m,n) is F[m,n] / N^2 (since our grid is N x N)
    
    # Use fft2 with the convention that matches our G-vector indexing
    # numpy fft2: F[k,l] = sum_{m,n} f[m,n] * exp(-2*pi*i*(k*m/N + l*n/N))
    # We want coefficients for G = (p, q) * 2*pi/N where p, q are integers
    # F[p mod N, q mod N] gives us the coefficient for G=(p,q)*2*pi/N
    
    F = fft2(inv_eps)
    # F is N x N. The coefficient for G=(p,q) is F[p mod N, q mod N] / N^2
    # But we need to be careful with the sign convention.
    # Actually, the standard Fourier series: f(x,y) = sum_{p,q} c_{p,q} exp(2*pi*i*(p*x/L + q*y/L))
    # where c_{p,q} = (1/L^2) ∫∫ f(x,y) exp(-2*pi*i*(p*x/L + q*y/L)) dxdy
    # Discrete: c_{p,q} ≈ (1/N^2) sum_{x,y} f[x,y] exp(-2*pi*i*(p*x/N + q*y/N))
    # This is exactly (1/N^2) * F[p mod N, q mod N] where F = fft2(f)
    
    # Build the eps_inv array for G-vectors in range [-n_pw//2, n_pw//2]
    half = n_pw // 2
    eps_inv = np.zeros((n_pw, n_pw), dtype=complex)
    for i in range(n_pw):
        p = i - half
        for j in range(n_pw):
            q = j - half
            eps_inv[i, j] = F[p % size, q % size] / (size * size)
    
    return eps_inv, half

def _compute_band_omega(kx, ky, eps_inv, half, n_pw, n_bands=6):
    """
    Compute the first n_bands TM frequencies for a given k-point (kx, ky) in units of 2*pi/L.
    kx, ky are in units of 2*pi/L (i.e., k = kx * 2*pi/L).
    Returns sorted list of omega values (in units of c/L * 2*pi... actually omega^2 eigenvalues).
    """
    # Build the matrix A where A_{G,G'} = (k+G)^2 * delta_{G,G'} * eps_inv(G-G')
    # Actually the eigenvalue problem for TM is:
    # sum_{G'} (k+G)^2 * eps_inv(G-G') * c_{G'} = omega^2 * c_G
    # Wait, let me be more careful. The standard formulation:
    # For TM: [ (k+G)^2 / eps_inv(G-G') ] ... no.
    # 
    # The correct TM eigenvalue problem in plane-wave method:
    # sum_{G'} (k+G)^2 * delta_{G,G'} * (1/eps)(G-G') * c_{G'} = (omega/c)^2 * c_G
    # 
    # So the matrix is: M[G, G'] = (k+G)^2 * delta_{G,G'} * eps_inv[G-G']
    # where eps_inv[G-G'] is the Fourier coefficient of 1/eps.
    #
    # Note: (k+G)^2 = (kx+Gx)^2 + (ky+Gy)^2 where G = (Gx, Gy) = (m, n) * 2*pi/L
    # and k = (kx, ky) * 2*pi/L
    # So (k+G)^2 / (2*pi/L)^2 = (kx+m)^2 + (ky+n)^2
    
    n = n_pw
    A = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        mi = i - half
        for j in range(n):
            mj = j - half
            # eps_inv for G_i - G_j = (mi-mj, ni-nj)
            # Index in eps_inv array: (mi-mj+half, ni-nj+half)
            di = mi - mj
            dj = (i - half) - (j - half)  # wait, this is wrong. Let me redo.
            # G_i = (mi, ni) where ni = i - half
            # G_j = (mj, nj) where nj = j - half
            ni = i - half
            nj = j - half
            di = mi - mj
            dj = ni - nj
            # Check if di, dj are in range
            idx_di = di + half
            idx_dj = dj + half
            if 0 <= idx_di < n and 0 <= idx_dj < n:
                eps_inv_val = eps_inv[idx_di, idx_dj]
            else:
                eps_inv_val = 0.0
            
            # (k + G_i)^2 in units of (2*pi/L)^2
            ki_x = kx + mi
            ki_y = ky + ni
            k_sq = ki_x * ki_x + ki_y * ki_y
            
            A[i, j] = k_sq * eps_inv_val
    
    # Solve eigenvalue problem
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Sort eigenvalues
    idx = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[idx].real
    
    # omega^2 = eigval, so omega = sqrt(eigval)
    # But we need to be careful: the eigenvalues should be positive
    eigvals_sorted = np.maximum(eigvals_sorted, 0.0)
    omegas = np.sqrt(eigvals_sorted)
    
    return omegas[:n_bands]

def _evaluate_gap(cell, n_pw=17, n_bands=6):
    """
    Evaluate the gap-midgap ratio for bands 4 and 5 (1-indexed, so indices 3 and 4).
    Returns the score: 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    """
    size = len(cell)
    eps_inv, half = _build_eps_inv_matrix(cell, n_pw)
    
    # Irreducible Brillouin zone for square lattice:
    # kx >= 0, ky >= 0, kx >= ky (by symmetry)
    # We sample 30 k-points along the boundary and interior
    
    # Key k-points for square lattice irreducible BZ:
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5), and points along Gamma-M, Gamma-X, M-X
    # Let's define a set of k-points
    
    k_points = []
    # Along Gamma to M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0.0))
    # Along Gamma to X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    # Along M to X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.3, 0.15))
    k_points.append((0.2, 0.2))
    
    # Remove duplicates and ensure we have ~30 points
    seen = set()
    unique_k = []
    for kp in k_points:
        key = (round(kp[0], 4), round(kp[1], 4))
        if key not in seen:
            seen.add(key)
            unique_k.append(kp)
    
    k_points = unique_k[:30]
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for (kx, ky) in k_points:
        omegas = _compute_band_omega(kx, ky, eps_inv, half, n_pw, n_bands)
        if len(omegas) >= 5:
            max_omega_4 = max(max_omega_4, omegas[3])  # 4th band (0-indexed: 3)
            min_omega_5 = min(min_omega_5, omegas[4])  # 5th band (0-indexed: 4)
    
    if min_omega_5 == float('inf') or max_omega_4 == 0.0:
        return 0.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return 2.0 * gap / midgap

def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry: cell == cell[::-1] row-wise, col-wise, and transpose."""
    size = len(cell)
    cell = np.array(cell, dtype=float)
    # Average over the symmetry group
    # The symmetry group of the square: identity, 90-deg rotation, 180-deg rotation, 270-deg rotation,
    # and reflections. For a 32x32 cell centered at (15.5, 15.5):
    # - 180-deg rotation: cell[i][j] -> cell[size-1-i][size-1-j]
    # - Horizontal flip: cell[i][j] -> cell[i][size-1-j]
    # - Vertical flip: cell[i][j] -> cell[size-1-i][j]
    # - Transpose: cell[i][j] -> cell[j][i]
    
    sym_sum = np.zeros_like(cell)
    sym_count = 0
    
    # Identity
    sym_sum += cell
    sym_count += 1
    
    # 180-degree rotation
    rot180 = cell[::-1, ::-1]
    sym_sum += rot180
    sym_count += 1
    
    # 90-degree rotation
    rot90 = np.rot90(cell, 1)
    sym_sum += rot90
    sym_count += 1
    
    # 270-degree rotation
    rot270 = np.rot90(cell, 3)
    sym_sum += rot270
    sym_count += 1
    
    # Horizontal flip
    hflip = cell[:, ::-1]
    sym_sum += hflip
    sym_count += 1
    
    # Vertical flip
    vflip = cell[::-1, :]
    sym_sum += vflip
    sym_count += 1
    
    # Transpose
    trans = cell.T
    sym_sum += trans
    sym_count += 1
    
    # Anti-transpose (transpose + 180 rot)
    anti_trans = rot180.T
    sym_sum += anti_trans
    sym_count += 1
    
    cell_avg = sym_sum / sym_count
    
    # Threshold to binary
    cell_bin = (cell_avg >= 0.5).astype(int)
    
    return cell_bin.tolist()

def design(problem):
    """Design a symmetric 32x32 photonic crystal cell to maximize the TM band gap between bands 4 and 5."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s limit
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Start with a good initial guess: a dielectric rod
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            row.append(1 if r2 < 0.2 ** 2 else 0)
        cell.append(row)
    
    # Enforce symmetry
    cell = _enforce_symmetry(cell)
    
    # Evaluate initial score
    current_score = _evaluate_gap(cell, n_pw=13, n_bands=6)
    best_cell = [row[:] for row in cell]
    best_score = current_score
    
    # Hill climbing with random perturbations
    rng = random.Random(42)
    
    # Use smaller n_pw for speed during optimization
    n_pw_opt = 13
    
    iter_count = 0
    while time.time() < deadline - 30:  # Leave 30s margin
        iter_count += 1
        
        # Generate a perturbation
        # Randomly flip a few pixels
        n_flips = rng.randint(1, 8)
        new_cell = [row[:] for row in best_cell]
        for _ in range(n_flips):
            i = rng.randint(0, size - 1)
            j = rng.randint(0, size - 1)
            new_cell[i][j] = 1 - new_cell[i][j]
        
        # Enforce symmetry
        new_cell = _enforce_symmetry(new_cell)
        
        # Evaluate
        new_score = _evaluate_gap(new_cell, n_pw=n_pw_opt, n_bands=6)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = [row[:] for row in new_cell]
        
        # Occasionally try a larger perturbation
        if iter_count % 20 == 0:
            n_flips = rng.randint(10, 30)
            new_cell = [row[:] for row in best_cell]
            for _ in range(n_flips):
                i = rng.randint(0, size - 1)
                j = rng.randint(0, size - 1)
                new_cell[i][j] = 1 - new_cell[i][j]
            new_cell = _enforce_symmetry(new_cell)
            new_score = _evaluate_gap(new_cell, n_pw=n_pw_opt, n_bands=6)
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in new_cell]
    
    # Final evaluation with higher precision
    final_score = _evaluate_gap(best_cell, n_pw=17, n_bands=6)
    
    return best_cell
