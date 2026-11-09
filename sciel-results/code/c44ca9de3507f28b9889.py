"""Design a 32x32 symmetric photonic crystal unit cell maximizing the TM band-4/5 gap."""
import time
import numpy as np
from itertools import product

# Deadline guard: return best result well before the 200s wall-clock limit
_DEADLINE = 185.0  # seconds from start of design()
_start_time = None

def _elapsed():
    return time.time() - _start_time

def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry: cell == cell[::-1] row-wise, col-wise, and transpose."""
    cell = np.array(cell, dtype=np.int8)
    n = cell.shape[0]
    # Symmetrize: average over all 8 symmetry operations, then threshold
    # For 0/1, we can just take the AND or OR; let's use the average and threshold at 0.5
    # Actually for exact symmetry, we need to pick one representative and mirror it.
    # The symmetry group is D4 (8 elements). We'll enforce by taking the top-left quadrant
    # and mirroring.
    
    # For a 32x32 cell with center at (15.5, 15.5), the symmetry is about the center.
    # cell[i][j] = cell[n-1-i][j] = cell[i][n-1-j] = cell[n-1-i][n-1-j]
    # and cell[i][j] = cell[j][i] (transpose symmetry)
    
    # We'll enforce by: for each (i,j), set cell[i][j] = cell[j][i] = cell[n-1-i][j] = ...
    # Simplest: iterate over the fundamental domain and copy.
    
    result = np.zeros_like(cell)
    for i in range(n):
        for j in range(n):
            # The 8 symmetric positions
            positions = set()
            for (ii, jj) in [(i,j), (n-1-i,j), (i,n-1-j), (n-1-i,n-1-j),
                             (j,i), (n-1-j,i), (j,n-1-i), (n-1-j,n-1-i)]:
                positions.add((ii, jj))
            # Take the value from the first position (or majority)
            val = 0
            for (ii, jj) in positions:
                if cell[ii, jj] == 1:
                    val = 1
                    break
            for (ii, jj) in positions:
                result[ii, jj] = val
    return result.tolist()

def _fourier_coeffs(cell, n_pw=17):
    """Compute exact Fourier coefficients of the dielectric function.
    
    For TM polarization, we need 1/epsilon in the plane-wave expansion.
    epsilon(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
    We need the Fourier coefficients of 1/epsilon.
    
    Since epsilon is piecewise constant (1 or 12), 1/epsilon is also piecewise constant.
    The Fourier coefficient for (nx, ny) is:
      c_{nx,ny} = (1/A) * integral (1/eps(x,y)) exp(-i 2pi (nx x + ny y)) dA
    
    For a pixelated cell, this is a sum over pixels.
    """
    n = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    inv_eps_low = 1.0 / eps_low
    inv_eps_high = 1.0 / eps_high
    
    # Precompute the pixel values as 0/1
    pix = np.array(cell, dtype=np.float64)
    
    # Fourier coefficients of the pixel function (0 or 1)
    # c_pixel_{nx,ny} = (1/N^2) sum_{i,j} pix[i,j] exp(-i 2pi (nx (i+0.5)/N + ny (j+0.5)/N))
    
    # For efficiency, use the fact that this is a 2D DFT of the pixel array
    # with a half-pixel phase shift.
    
    max_n = n_pw  # we need up to +/- n_pw in each direction
    
    coeffs = {}
    for nx in range(-max_n, max_n + 1):
        for ny in range(-max_n, max_n + 1):
            # Sum over all pixels
            # exp(-i 2pi (nx (i+0.5)/N + ny (j+0.5)/N))
            # = exp(-i 2pi nx (i+0.5)/N) * exp(-i 2pi ny (j+0.5)/N)
            
            # This is separable
            # sum_i pix_col_sum[i] * exp(-i 2pi nx (i+0.5)/N)
            # where pix_col_sum[i] = sum_j pix[i,j] * exp(-i 2pi ny (j+0.5)/N)
            
            # For small n_pw and N=32, we can just compute directly
            total = 0.0
            for i in range(n):
                for j in range(n):
                    if pix[i, j] == 1:
                        angle_x = 2 * np.pi * nx * (i + 0.5) / n
                        angle_y = 2 * np.pi * ny * (j + 0.5) / n
                        total += np.exp(-1j * (angle_x + angle_y))
            c_pixel = total / (n * n)
            
            # 1/eps = inv_eps_low + (inv_eps_high - inv_eps_low) * pix
            # So c_{1/eps}_{nx,ny} = inv_eps_low * delta_{nx,ny,0} + (inv_eps_high - inv_eps_low) * c_pixel
            if nx == 0 and ny == 0:
                c_inv_eps = inv_eps_low + (inv_eps_high - inv_eps_low) * c_pixel
            else:
                c_inv_eps = (inv_eps_high - inv_eps_low) * c_pixel
            
            coeffs[(nx, ny)] = c_inv_eps
    
    return coeffs

def _band_structure(cell, n_pw=17, n_k=30):
    """Compute TM band structure using plane-wave method.
    
    Returns the gap-midgap ratio for bands 4 and 5.
    """
    n = len(cell)
    coeffs = _fourier_coeffs(cell, n_pw)
    
    # Build the plane-wave matrix
    # For TM polarization, the eigenvalue problem is:
    # sum_{G'} (K+G) dot (K+G') * c_{1/eps}_{G-G'} * A_{G'} = omega^2 * A_G
    # where K is the wavevector, G are reciprocal lattice vectors
    
    # Reciprocal lattice vectors for square lattice with a=1:
    # G = (m, n) * 2pi, for integers m, n
    
    # Plane waves: index them by (m, n) where -n_pw <= m, n <= n_pw
    pw_indices = [(m, nn) for m in range(-n_pw, n_pw+1) for nn in range(-n_pw, n_pw+1)]
    n_pw_total = len(pw_indices)
    
    # Precompute K+G for each plane wave
    # We'll evaluate at several k-points in the irreducible BZ
    
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= pi (in units of 2pi/a)
    # Actually, the irreducible BZ is a triangle: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi)
    
    # Generate k-points
    k_points = []
    # Boundary points
    k_points.append((0, 0))
    k_points.append((0.5, 0))
    k_points.append((0.5, 0.5))
    k_points.append((0.25, 0.25))
    k_points.append((0.1, 0))
    k_points.append((0.3, 0))
    k_points.append((0.4, 0))
    k_points.append((0.05, 0.05))
    k_points.append((0.15, 0.15))
    k_points.append((0.25, 0.25))
    k_points.append((0.35, 0.35))
    k_points.append((0.45, 0.45))
    k_points.append((0.0, 0.25))
    k_points.append((0.0, 0.5))
    k_points.append((0.25, 0.0))
    k_points.append((0.1, 0.2))
    k_points.append((0.2, 0.3))
    k_points.append((0.15, 0.25))
    k_points.append((0.05, 0.15))
    k_points.append((0.3, 0.1))
    k_points.append((0.2, 0.1))
    k_points.append((0.4, 0.2))
    k_points.append((0.3, 0.3))
    k_points.append((0.1, 0.3))
    k_points.append((0.2, 0.2))
    k_points.append((0.15, 0.1))
    k_points.append((0.25, 0.1))
    k_points.append((0.35, 0.15))
    k_points.append((0.05, 0.25))
    k_points.append((0.45, 0.2))
    
    # Trim to n_k points
    k_points = k_points[:n_k]
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for (kx, ky) in k_points:
        # K in units of 2pi
        Kx = kx
        Ky = ky
        
        # Build the matrix
        # M_{G,G'} = (K+G) dot (K+G') * c_{1/eps}_{G-G'}
        
        # For efficiency, precompute (K+G) for each G
        KG = []
        for (m, nn) in pw_indices:
            KGx = Kx + m
            KGy = Ky + nn
            KG.append((KGx, KGy))
        
        # Build matrix
        M = np.zeros((n_pw_total, n_pw_total), dtype=np.float64)
        for i, (Gi) in enumerate(pw_indices):
            for j, (Gj) in enumerate(pw_indices):
                Gdiff = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                if Gdiff[0] > n_pw or Gdiff[0] < -n_pw or Gdiff[1] > n_pw or Gdiff[1] < -n_pw:
                    c = 0.0
                else:
                    c = coeffs.get(Gdiff, 0.0)
                    if isinstance(c, complex):
                        c = c.real  # Should be real for symmetric cell
                
                dot = KG[i][0] * KG[j][0] + KG[i][1] * KG[j][1]
                M[i, j] = dot * c
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(M)
        except:
            continue
        
        # Take the first 5 eigenvalues (bands 1-5)
        if len(eigvals) >= 5:
            omega4 = np.sqrt(max(eigvals[3], 0))
            omega5 = np.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    
    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0
    
    gap = 2 * (min_omega5 - max_omega4)
    midgap = min_omega5 + max_omega4
    return gap / midgap

def _initial_designs(size=32):
    """Generate several initial design candidates."""
    designs = []
    
    # 1. Simple dielectric rod
    cell = np.zeros((size, size), dtype=np.int8)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < 6.5:  # radius ~2.5 pixels
                cell[i, j] = 1
    designs.append(cell.tolist())
    
    # 2. Air hole in dielectric
    cell = np.ones((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < 6.5:
                cell[i, j] = 0
    designs.append(cell.tolist())
    
    # 3. Ring (annulus)
    cell = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            r2 = dx*dx + dy*dy
            if 9 < r2 < 16:  # radius 3 to 4
                cell[i, j] = 1
    designs.append(cell.tolist())
    
    # 4. Cross pattern
    cell = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            if abs(i - center) < 2 or abs(j - center) < 2:
                cell[i, j] = 1
    designs.append(cell.tolist())
    
    # 5. Checkerboard-like
    cell = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            if (i // 4 + j // 4) % 2 == 0:
                cell[i, j] = 1
    designs.append(cell.tolist())
    
    # 6. Random symmetric
    np.random.seed(42)
    cell = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            if np.random.rand() < 0.4:
                cell[i, j] = 1
    cell = _enforce_symmetry(cell.tolist())
    designs.append(cell)
    
    return designs

def design(problem):
    global _start_time
    _start_time = time.time()
    
    size = problem["pixels"]
    
    # Generate initial designs
    candidates = _initial_designs(size)
    
    # Evaluate each candidate
    best_score = -np.inf
    best_cell = candidates[0]
    
    for cell in candidates:
        if _elapsed() > _DEADLINE - 30:
            break
        score = _band_structure(cell, n_pw=17, n_k=30)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search: flip pixels in the fundamental domain
    # The fundamental domain for D4 symmetry on a 32x32 grid
    # We'll work with the top-left 8x8 quadrant (since 32/4 = 8)
    # Actually, the symmetry is about the center, so the fundamental domain is more complex.
    
    # Let's do a simple local search: randomly flip pixels and check symmetry
    n_iter = 500
    current_cell = best_cell
    current_score = best_score
    
    for it in range(n_iter):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Create a perturbation
        new_cell = [row[:] for row in current_cell]
        
        # Flip a random pixel (and its symmetric counterparts)
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Get all symmetric positions
        positions = set()
        for (ii, jj) in [(i,j), (size-1-i,j), (i,size-1-j), (size-1-i,size-1-j),
                         (j,i), (size-1-j,i), (j,size-1-i), (size-1-j,size-1-i)]:
            positions.add((ii, jj))
        
        # Flip all of them
        for (ii, jj) in positions:
            new_cell[ii][jj] = 1 - new_cell[ii][jj]
        
        # Evaluate
        score = _band_structure(new_cell, n_pw=17, n_k=30)
        
        if score > current_score:
            current_cell = new_cell
            current_score = score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell
    
    return best_cell
