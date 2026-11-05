"""Photonic crystal design with TM band gap optimization."""
import numpy as np
from scipy.linalg import eigh
import time
import math

def _compute_fourier_coeffs(cell, N=17):
    """Compute exact Fourier coefficients of 1/eps for the cell.
    N is half the number of plane waves (total = (2N-1)^2)"""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Fourier coefficients: c_{mn} = (1/A) integral inv_eps(x,y) exp(-i(m*2pi*x + n*2pi*y)) dx dy
    # For pixelated: sum over pixels
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            for m_idx in range(2*N-1):
                m = m_idx - (N-1)
                for n_idx in range(2*N-1):
                    n = n_idx - (N-1)
                    coeffs[m_idx, n_idx] += inv_eps[i, j] * np.exp(-2j * np.pi * (m*x + n*y)) / (size*size)
    return coeffs

def _compute_fourier_coeffs_fast(cell, N=17):
    """Fast computation using 2D FFT for Fourier coefficients."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Upsample to get better resolution, then use FFT
    # Actually, for pixelated structure, we can use direct summation with vectorization
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    
    # Vectorized computation
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    m_vals = np.arange(-(N-1), N)
    n_vals = np.arange(-(N-1), N)
    
    # For each Fourier mode (m,n), compute the coefficient
    for m_idx, m in enumerate(m_vals):
        for n_idx, n in enumerate(n_vals):
            phase = np.exp(-2j * np.pi * (m * X + n * Y))
            coeffs[m_idx, n_idx] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _pwe_tm_bands(cell, kx, ky, N=17, n_bands=6):
    """Solve TM band structure at wave vector (kx, ky) using PWE method.
    Returns the first n_bands eigenvalues (omega^2)."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Get Fourier coefficients of 1/eps
    coeffs = _compute_fourier_coeffs_fast(cell, N)
    
    # Plane wave grid
    G = 2 * np.pi
    m_vals = np.arange(-(N-1), N)
    Gx = G * m_vals
    Gy = G * m_vals
    
    # Build the eigenvalue matrix for TM polarization
    # TM: sum_{G'} (1/eps)_{G-G'} * |k+G|^2 * c_{G'} = omega^2 * c_G
    # In matrix form: A * c = omega^2 * c where A[G,G'] = (1/eps)_{G-G'} * |k+G|^2
    
    # Precompute |k+G|^2 for each plane wave
    pw_indices = []
    for i in range(len(m_vals)):
        for j in range(len(m_vals)):
            pw_indices.append((i, j))
    n_pw = len(pw_indices)
    
    kG2 = np.zeros(n_pw)
    for idx, (i, j) in enumerate(pw_indices):
        kx_total = kx + Gx[i]
        ky_total = ky + Gy[j]
        kG2[idx] = kx_total**2 + ky_total**2
    
    # Build matrix A
    A = np.zeros((n_pw, n_pw), dtype=complex)
    for idx1, (i1, j1) in enumerate(pw_indices):
        for idx2, (i2, j2) in enumerate(pw_indices):
            # G1 - G2 has components (Gx[i1]-Gx[i2], Gy[j1]-Gy[j2])
            # Fourier index: m_diff = i1-i2 + (N-1), n_diff = j1-j2 + (N-1)
            m_diff = i1 - i2 + (N-1)
            n_diff = j1 - j2 + (N-1)
            if 0 <= m_diff < 2*N-1 and 0 <= n_diff < 2*N-1:
                A[idx1, idx2] = coeffs[m_diff, n_diff] * kG2[idx1]
    
    # Solve eigenvalue problem
    vals, vecs = eigh(A)
    # Return first n_bands eigenvalues
    return vals[:n_bands].real

def _irreducible_bz_points(n_kx=6, n_ky=6):
    """Generate wave vectors in the irreducible Brillouin zone (triangle from Gamma to X to M)."""
    points = []
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= pi (in units where G=2pi)
    # Actually for square lattice, irreducible BZ is the triangle: 0<=kx<=ky, kx+ky<=pi
    # Let's use a grid in the full BZ and filter
    for i in range(n_kx):
        for j in range(n_ky):
            kx = (i / (n_kx - 1)) * np.pi
            ky = (j / (n_ky - 1)) * np.pi
            # Check if in irreducible BZ: 0 <= kx <= ky and kx + ky <= pi
            if kx <= ky and (kx + ky) <= np.pi + 1e-10:
                points.append((kx, ky))
    # Add special high-symmetry points
    points.append((0, 0))  # Gamma
    points.append((np.pi, 0))  # X
    points.append((np.pi/2, np.pi/2))  # M
    # Remove duplicates
    unique_points = []
    for p in points:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in unique_points):
            unique_points.append(p)
    return unique_points

def _evaluate_gap(cell, k_points=None, n_bands=6):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    if k_points is None:
        k_points = _irreducible_bz_points()
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        try:
            vals = _pwe_tm_bands(cell, kx, ky, N=17, n_bands=n_bands)
            if len(vals) >= 5:
                omega4 = vals[3]  # 4th band (0-indexed: 3)
                omega5 = vals[4]  # 5th band (0-indexed: 4)
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 <= max_omega4:
        return -1.0  # Negative indicates overlap
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _apply_symmetry(cell):
    """Ensure the cell has 4-fold rotational symmetry about the center."""
    size = len(cell)
    result = np.array(cell, dtype=float)
    
    # Average over the 4 symmetry operations: identity, 90°, 180°, 270°
    # 90° rotation: new[i][j] = old[size-1-j][i]
    # 180° rotation: new[i][j] = old[size-1-i][size-1-j]
    # 270° rotation: new[i][j] = old[j][size-1-i]
    
    rot90 = np.zeros_like(result)
    rot180 = np.zeros_like(result)
    rot270 = np.zeros_like(result)
    
    for i in range(size):
        for j in range(size):
            rot90[i, j] = result[size-1-j, i]
            rot180[i, j] = result[size-1-i, size-1-j]
            rot270[i, j] = result[j, size-1-i]
    
    averaged = (result + rot90 + rot180 + rot270) / 4.0
    
    # Threshold at 0.5
    result_sym = (averaged >= 0.5).astype(int)
    
    # Verify symmetry
    assert np.array_equal(result_sym, result_sym[::-1, :]), "Row reversal symmetry failed"
    assert np.array_equal(result_sym, result_sym[:, ::-1]), "Column reversal symmetry failed"
    assert np.array_equal(result_sym, result_sym.T), "Transpose symmetry failed"
    
    return result_sym.tolist()

def _flip_pixel(cell, i, j, size):
    """Flip a pixel and apply symmetry."""
    cell = np.array(cell, dtype=int)
    old_val = cell[i, j]
    cell[i, j] = 1 - old_val
    
    # Apply symmetry: also flip the symmetric positions
    # 90°: (size-1-j, i)
    # 180°: (size-1-i, size-1-j)
    # 270°: (j, size-1-i)
    sym_positions = [(i, j), (size-1-j, i), (size-1-i, size-1-j), (j, size-1-i)]
    
    for si, sj in sym_positions:
        if cell[si, sj] == old_val:
            cell[si, sj] = 1 - old_val
    
    return cell

def _local_field_intensity(cell, kx, ky, band_idx, N=17):
    """Compute local field intensity for a given band at wave vector."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    coeffs = _compute_fourier_coeffs_fast(cell, N)
    
    G = 2 * np.pi
    m_vals = np.arange(-(N-1), N)
    Gx = G * m_vals
    Gy = G * m_vals
    
    pw_indices = []
    for i in range(len(m_vals)):
        for j in range(len(m_vals)):
            pw_indices.append((i, j))
    n_pw = len(pw_indices)
    
    kG2 = np.zeros(n_pw)
    for idx, (i, j) in enumerate(pw_indices):
        kx_total = kx + Gx[i]
        ky_total = ky + Gy[j]
        kG2[idx] = kx_total**2 + ky_total**2
    
    A = np.zeros((n_pw, n_pw), dtype=complex)
    for idx1, (i1, j1) in enumerate(pw_indices):
        for idx2, (i2, j2) in enumerate(pw_indices):
            m_diff = i1 - i2 + (N-1)
            n_diff = j1 - j2 + (N-1)
            if 0 <= m_diff < 2*N-1 and 0 <= n_diff < 2*N-1:
                A[idx1, idx2] = coeffs[m_diff, n_diff] * kG2[idx1]
    
    vals, vecs = eigh(A)
    
    if band_idx >= len(vals):
        return None
    
    eigvec = vecs[:, band_idx]
    
    # Reconstruct field in real space
    # E(x,y) = sum_G c_G exp(i(k+G)·r)
    intensity = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            field = 0.0
            for idx, (mi, ni) in enumerate(pw_indices):
                kx_total = kx + Gx[mi]
                ky_total = ky + Gy[ni]
                field += eigvec[idx] * np.exp(1j * (kx_total * x + ky_total * y))
            intensity[i, j] = np.abs(field)**2
    
    return intensity

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s kill
    
    # Initial design: centered dielectric rod
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < 0.2**2 else 0)
        cell.append(row)
    
    cell = _apply_symmetry(cell)
    
    # Evaluate initial gap
    k_points = _irreducible_bz_points(n_kx=5, n_ky=5)
    best_gap = _evaluate_gap(cell, k_points)
    best_cell = [row[:] for row in cell]
    
    # Optimization: try flipping pixels guided by field intensities
    # Use a subset of k-points for speed
    opt_k_points = k_points[:10]
    
    max_iter = 50
    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        # Compute field intensities for bands 4 and 5 at a few k-points
        intensity_4 = np.zeros((size, size))
        intensity_5 = np.zeros((size, size))
        n_valid = 0
        
        for kx, ky in opt_k_points[:5]:
            int4 = _local_field_intensity(cell, kx, ky, 3)  # Band 4 (0-indexed: 3)
            int5 = _local_field_intensity(cell, kx, ky, 4)  # Band 5 (0-indexed: 4)
            if int4 is not None and int5 is not None:
                intensity_4 += int4
                intensity_5 += int5
                n_valid += 1
        
        if n_valid == 0:
            break
        
        intensity_4 /= n_valid
        intensity_5 /= n_valid
        
        # Score each pixel flip: add dielectric where band 4 has high intensity,
        # remove where band 5 has high intensity
        cell_np = np.array(cell, dtype=int)
        best_score = best_gap
        best_flip = None
        
        # Only consider pixels in the fundamental domain for efficiency
        for i in range(size // 2):
            for j in range(size // 2):
                if time.time() > deadline - 30:
                    break
                
                # Try flipping pixel (i, j)
                new_cell = _flip_pixel(cell_np, i, j, size)
                
                # Quick evaluation
                gap = _evaluate_gap(new_cell.tolist(), opt_k_points)
                
                if gap > best_score:
                    best_score = gap
                    best_flip = (i, j)
            
            if time.time() > deadline - 30:
                break
        
        if best_flip is not None:
            cell = _flip_pixel(cell_np, best_flip[0], best_flip[1], size).tolist()
            cell = _apply_symmetry(cell)
            best_cell = [row[:] for row in cell]
            best_gap = best_score
        else:
            # No improvement, try random flips
            improved = False
            for _ in range(10):
                if time.time() > deadline - 30:
                    break
                ri = np.random.randint(0, size // 2)
                rj = np.random.randint(0, size // 2)
                new_cell = _flip_pixel(cell_np, ri, rj, size).tolist()
                gap = _evaluate_gap(new_cell, opt_k_points)
                if gap > best_gap:
                    cell = new_cell
                    cell = _apply_symmetry(cell)
                    best_cell = [row[:] for row in cell]
                    best_gap = gap
                    improved = True
                    break
            
            if not improved:
                break
    
    # Final verification of symmetry
    final_cell = _apply_symmetry(best_cell)
    
    return final_cell
