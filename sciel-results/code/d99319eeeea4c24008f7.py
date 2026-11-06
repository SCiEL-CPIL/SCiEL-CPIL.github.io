"""Photonic crystal design with TM band gap optimization using field-intensity guided search."""
import numpy as np
import time
import math

def _compute_fourier_coeffs(cell, N=17):
    """Compute exact Fourier coefficients of 1/eps for the pixelated cell."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    xs = (np.arange(size) + 0.5) / size
    m_vals = np.arange(-(N-1), N)
    n_vals = np.arange(-(N-1), N)
    
    # Build 2D grid of coordinates
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    for mi, m in enumerate(m_vals):
        for ni, n in enumerate(n_vals):
            phase = np.exp(-2j * np.pi * (m * X + n * Y))
            coeffs[mi, ni] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _pwe_tm_eigh(cell, kx, ky, N=17, n_bands=6, coeffs_cache=None):
    """Solve TM band structure at wave vector (kx, ky). Returns eigenvalues and eigenvectors."""
    size = len(cell)
    if coeffs_cache is None:
        coeffs = _compute_fourier_coeffs(cell, N)
    else:
        coeffs = coeffs_cache
    
    m_vals = np.arange(-(N-1), N)
    Gx = 2 * np.pi * m_vals
    Gy = 2 * np.pi * m_vals
    
    n_pw = len(m_vals) ** 2
    pw_i = np.repeat(np.arange(len(m_vals)), len(m_vals))
    pw_j = np.tile(np.arange(len(m_vals)), len(m_vals))
    
    # |k+G|^2 for each plane wave
    kG2 = (kx + Gx[pw_i])**2 + (ky + Gy[pw_j])**2
    
    # Build the matrix A where A[idx1, idx2] = coeffs[pw_i[idx1]-pw_i[idx2]+N-1, pw_j[idx1]-pw_j[idx2]+N-1] * kG2[idx1]
    # We can exploit the structure to build this faster.
    
    # For a given idx1, the row is kG2[idx1] * (row of the Toeplitz-like coefficient matrix)
    A = np.zeros((n_pw, n_pw), dtype=complex)
    
    for idx1 in range(n_pw):
        i1 = pw_i[idx1]
        j1 = pw_j[idx1]
        di = i1 - pw_i
        dj = j1 - pw_j
        ci = di + (N - 1)
        cj = dj + (N - 1)
        
        mask = (ci >= 0) & (ci < 2*N-1) & (cj >= 0) & (cj < 2*N-1)
        
        row = np.zeros(n_pw, dtype=complex)
        row[mask] = coeffs[ci[mask], cj[mask]]
        A[idx1, :] = row * kG2[idx1]
    
    # A is Hermitian
    vals, vecs = np.linalg.eigh(A)
    return vals[:n_bands].real, vecs[:, :n_bands]

def _irreducible_bz_points(n_per_side=5):
    """Generate wave vectors in the irreducible Brillouin zone."""
    points = []
    # Irreducible BZ for square lattice: triangle with vertices (0,0), (pi,0), (pi/2, pi/2)
    # Actually: 0 <= kx <= ky, kx + ky <= pi
    for i in range(n_per_side):
        for j in range(n_per_side):
            kx = (i / max(n_per_side - 1, 1)) * np.pi
            ky = (j / max(n_per_side - 1, 1)) * np.pi
            if kx <= ky + 1e-10 and (kx + ky) <= np.pi + 1e-10:
                points.append((kx, ky))
    
    # Ensure high-symmetry points are included
    points.append((0.0, 0.0))       # Gamma
    points.append((np.pi, 0.0))     # X
    points.append((np.pi/2, np.pi/2))  # M
    
    # Remove duplicates
    unique = []
    for p in points:
        if not any(abs(p[0]-q[0]) < 1e-8 and abs(p[1]-q[1]) < 1e-8 for q in unique):
            unique.append(p)
    return unique

def _evaluate_gap(cell, k_points, n_bands=6):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        try:
            vals, _ = _pwe_tm_eigh(cell, kx, ky, N=17, n_bands=n_bands)
            if len(vals) >= 5:
                omega4 = vals[3]
                omega5 = vals[4]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 <= max_omega4:
        return -1.0
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _compute_field_intensities(cell, kx, ky, N=17, n_bands=6, coeffs_cache=None):
    """Compute |E|^2 on the pixel grid for bands 4 and 5 at wave vector (kx,ky)."""
    size = len(cell)
    vals, vecs = _pwe_tm_eigh(cell, kx, ky, N=N, n_bands=n_bands, coeffs_cache=coeffs_cache)
    
    if len(vals) < 5:
        return None, None, None, None
    
    m_vals = np.arange(-(N-1), N)
    Gx = 2 * np.pi * m_vals
    Gy = 2 * np.pi * m_vals
    
    xs = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    
    # Precompute phases for all plane waves
    # phases[mi, ni, i_pix, j_pix] would be too big. Do it per pixel in a loop.
    intensity4 = np.zeros((size, size))
    intensity5 = np.zeros((size, size))
    
    vec4 = vecs[:, 3]  # 4th band (0-indexed 3)
    vec5 = vecs[:, 4]  # 5th band (0-indexed 4)
    
    pw_i = np.repeat(np.arange(len(m_vals)), len(m_vals))
    pw_j = np.tile(np.arange(len(m_vals)), len(m_vals))
    
    # For efficiency, precompute the exponential factors for each pixel
    # E(x,y) = sum_G c_G exp(i(k+G).r)
    # = sum_G c_G exp(i(kx+Gx[mi])*x) * exp(i(ky+Gy[ni])*y)
    
    # Precompute exp(i*(kx+Gx)*xs) and exp(i*(ky+Gy)*xs) for each pixel position
    exp_x4 = np.exp(1j * (kx + Gx[pw_i])[:, None] * xs[None, :])  # shape (n_pw, size)
    exp_y4 = np.exp(1j * (ky + Gy[pw_j])[:, None] * xs[None, :])  # shape (n_pw, size)
    
    # Actually the grid is 2D: pixel (i,j) has x=xs[i], y=xs[j]
    # E(i,j) = sum_{mi,ni} c_{mi,ni} exp(i(kx+Gx[mi])*xs[i]) * exp(i(ky+Gy[ni])*xs[j])
    
    # Vectorize:
    # field4[i,j] = sum_p c4[p] * exp(i(kx+Gx[pw_i[p]])*xs[i]) * exp(i(ky+Gy[pw_j[p]])*xs[j])
    
    # Build matrices:
    # Ex4[i, p] = exp(i(kx+Gx[pw_i[p]])*xs[i])
    # Ey4[j, p] = exp(i(ky+Gy[pw_j[p]])*xs[j])
    # field4 = Ex4 @ (c4 * Ey4)  -- not quite, need to be careful with broadcasting.
    
    # field4[i,j] = sum_p c4[p] * Ex4[i,p] * Ey4[j,p]
    # = sum_p (c4[p] * Ex4[i,p]) * Ey4[j,p]
    # = (c4 * Ex4^T)[p] contracted with Ey4 ... 
    # Actually: field4 = Ex4 @ diag(c4) @ Ey4^T? No.
    # field4[i,j] = sum_p c4[p] * Ex4[i,p] * Ey4[j,p]
    # Let M[i,p] = c4[p] * Ex4[i,p], then field4[i,j] = sum_p M[i,p] * Ey4[j,p] = (M @ Ey4^T)[i,j]
    
    Ex4 = np.exp(1j * (kx + Gx[pw_i]) * xs[:, None])  # (size, n_pw)
    Ey4 = np.exp(1j * (ky + Gy[pw_j]) * xs[:, None])  # (size, n_pw)
    
    M4 = vec4[None, :] * Ex4  # (size, n_pw)
    field4 = M4 @ Ey4.T  # (size, size)
    
    Ex5 = Ex4  # same as above
    Ey5 = Ey4
    
    M5 = vec5[None, :] * Ex5
    field5 = M5 @ Ey5.T
    
    intensity4 = np.abs(field4)**2
    intensity5 = np.abs(field5)**2
    
    # Normalize
    intensity4 /= (np.sum(intensity4) + 1e-12)
    intensity5 /= (np.sum(intensity5) + 1e-12)
    
    return intensity4, intensity5, vals, (vecs)

def _apply_symmetry(cell):
    """Ensure the cell has 4-fold rotational symmetry about the center."""
    size = len(cell)
    result = np.array(cell, dtype=float)
    
    rot90 = np.zeros_like(result)
    rot180 = np.zeros_like(result)
    rot270 = np.zeros_like(result)
    
    for i in range(size):
        for j in range(size):
            rot90[i, j] = result[size-1-j, i]
            rot180[i, j] = result[size-1-i, size-1-j]
            rot270[i, j] = result[j, size-1-i]
    
    averaged = (result + rot90 + rot180 + rot270) / 4.0
    result_sym = (averaged >= 0.5).astype(int)
    
    return result_sym.tolist()

def _flip_pixel_symmetric(cell_np, i, j, size):
    """Flip a pixel and all its symmetric counterparts."""
    cell = cell_np.copy()
    old_val = cell[i, j]
    
    # Symmetric positions under C4v about center
    # Identity: (i, j)
    # 90° rot: (size-1-j, i)
    # 180° rot: (size-1-i, size-1-j)
    # 270° rot: (j, size-1-i)
    # Reflections are already included if we have these 4 for a generic pixel.
    # Actually for a pixel on the diagonal, some of these may coincide.
    
    sym_positions = [(i, j), (size-1-j, i), (size-1-i, size-1-j), (j, size-1-i)]
    
    new_val = 1 - old_val
    for si, sj in sym_positions:
        cell[si, sj] = new_val
    
    return cell

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 215  # Safety margin before 230s kill
    
    # Initial design: four small dielectric dots in the quadrants
    cell = np.zeros((size, size), dtype=int)
    # Place dots at (8,8), (8,24), (24,8), (24,24) approximately
    for center in [(8, 8), (8, size-1-8), (size-1-8, 8), (size-1-8, size-1-8)]:
        ci, cj = center
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii, jj = ci + di, cj + dj
                if 0 <= ii < size and 0 <= jj < size:
                    if di*di + dj*dj <= 3:
                        cell[ii, jj] = 1
    
    cell = _apply_symmetry(cell.tolist())
    
    # Use a small set of k-points for search, larger for final
    k_points_search = _irreducible_bz_points(n_per_side=4)
    k_points_final = _irreducible_bz_points(n_per_side=6)
    
    # Precompute Fourier coefficients for initial cell
    coeffs = _compute_fourier_coeffs(cell, N=17)
    
    best_gap = _evaluate_gap(cell, k_points_search)
    best_cell = np.array(cell, dtype=int)
    
    print(f"Initial gap: {best_gap:.4f}", flush=True)
    
    # Field-intensity guided search
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() < deadline - 30:
        if no_improve_count >= max_no_improve:
            break
        
        # Compute field intensities at a few k-points to guide the search
        # Use the k-point where the gap is most sensitive (where omega4 is max or omega5 is min)
        # For simplicity, use Gamma point and X point
        
        guide_k_points = [(0.0, 0.0), (np.pi, 0.0)]
        
        total_int4 = np.zeros((size, size))
        total_int5 = np.zeros((size, size))
        n_valid = 0
        
        for kx, ky in guide_k_points:
            int4, int5, vals, vecs = _compute_field_intensities(
                best_cell.tolist(), kx, ky, N=17, n_bands=6, coeffs_cache=coeffs)
            if int4 is not None:
                total_int4 += int4
                total_int5 += int5
                n_valid += 1
        
        if n_valid == 0:
            break
        
        total_int4 /= n_valid
        total_int5 /= n_valid
        
        # Score each pixel flip
        # If pixel is air (0), adding dielectric should help if band 4 intensity is high there
        # If pixel is dielectric (1), removing it should help if band 5 intensity is high there
        # Score for adding: int4[i,j]
        # Score for removing: int5[i,j]
        
        best_score = -1.0
        best_flip = None
        
        # Only consider the fundamental domain (one quadrant) to respect symmetry
        # Actually, flipping one pixel and its symmetric counterparts is one move
        # We iterate over pixels in the first quadrant
        half = size // 2
        for i in range(half):
            for j in range(half):
                if best_cell[i, j] == 0:
                    # Consider adding dielectric
                    score = total_int4[i, j] - total_int5[i, j]
                else:
                    # Consider removing dielectric
                    score = total_int5[i, j] - total_int4[i, j]
                
                if score > best_score:
                    best_score = score
                    best_flip = (i, j)
        
        if best_flip is None or best_score < 0.001:
            no_improve_count += 1
            continue
        
        # Apply the flip
        new_cell = _flip_pixel_symmetric(best_cell, best_flip[0], best_flip[1], size)
        
        # Update Fourier coefficients
        coeffs = _compute_fourier_coeffs(new_cell.tolist(), N=17)
        
        # Evaluate the new gap
        new_gap = _evaluate_gap(new_cell.tolist(), k_points_search)
        
        if new_gap > best_gap:
            best_gap = new_gap
            best_cell = new_cell
            no_improve_count = 0
            print(f"Iteration: gap improved to {best_gap:.4f}", flush=True)
        else:
            no_improve_count += 1
    
    # Final evaluation with more k-points
    final_gap = _evaluate_gap(best_cell.tolist(), k_points_final)
    
    final_cell = _apply_symmetry(best_cell.tolist())
    
    return final_cell
