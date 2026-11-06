import numpy as np
import time
import sys

def compute_epsilon_fourier(cell, N):
    """
    Compute Fourier coefficients of epsilon for a 32x32 cell.
    G ranges from -N to N in both directions.
    cell: 32x32 list of lists (0 or 1).
    N: max G index.
    Returns dict or 2D array of epsilon_G.
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # Precompute pixel centers and areas
    # Pixel (i,j) covers [i/size, (i+1)/size] x [j/size, (j+1)/size]
    # We need integral of exp(-i G . r) over pixel.
    # Integral over x: [i/size, (i+1)/size] exp(-i Gx x) dx
    # = exp(-i Gx (i+1)/size) - exp(-i Gx i/size) divided by (-i Gx)
    # If Gx == 0, it's just width.
    
    Gs = np.arange(-N, N+1)
    epsilon_G = np.zeros((len(Gs), len(Gs)), dtype=complex)
    
    for gx_idx, gx in enumerate(Gs):
        for gy_idx, gy in enumerate(Gs):
            sum_val = 0.0
            for i in range(size):
                x_start = i / size
                x_end = (i + 1) / size
                # Integrate exp(-i gx x)
                if gx == 0:
                    int_x = x_end - x_start
                else:
                    int_x = (np.exp(-1j * gx * x_end) - np.exp(-1j * gx * x_start)) / (-1j * gx)
                
                for j in range(size):
                    y_start = j / size
                    y_end = (j + 1) / size
                    if gy == 0:
                        int_y = y_end - y_start
                    else:
                        int_y = (np.exp(-1j * gy * y_end) - np.exp(-1j * gy * y_start)) / (-1j * gy)
                    
                    # cell[i][j] is 1 (dielectric) or 0 (air)
                    val = eps_high if cell[i][j] == 1 else eps_low
                    sum_val += val * int_x * int_y
            
            # Normalize by cell area (which is 1x1)
            epsilon_G[gx_idx, gy_idx] = sum_val
            
    return epsilon_G, Gs

def solve_tm_bands(epsilon_G, Gs, kx, ky, num_bands=5):
    """
    Solve TM band structure for a given k.
    Returns list of eigenvalues (omega^2).
    """
    N = len(Gs) // 2
    M = np.zeros((len(Gs), len(Gs)), dtype=complex)
    
    for i, gx in enumerate(Gs):
        for j, gy in enumerate(Gs):
            # Vector k + G
            kx_g = kx + 2*np.pi*gx
            ky_g = ky + 2*np.pi*gy
            k_sq = kx_g**2 + ky_g**2
            
            for ii, gpx in enumerate(Gs):
                for jj, gpy in enumerate(Gs):
                    # G - G'
                    dgx = gx - gpx
                    dgy = gy - gpy
                    # Find index of dgx, dgy in Gs
                    # Gs is sorted -N..N
                    idx_x = dgx + N
                    idx_y = dgy + N
                    if 0 <= idx_x < len(Gs) and 0 <= idx_y < len(Gs):
                        eps_inv = 1.0 / epsilon_G[idx_x, idx_y]
                    else:
                        # If G-G' is outside the set, we approximate epsilon as average? 
                        # No, in PW method, we only sum over G' in the set.
                        # The matrix element M_{GG'} involves epsilon_{G-G'}.
                        # If G-G' is not in the set, its coefficient is 0 in the truncated basis?
                        # Actually, the convolution is exact if we have infinite G.
                        # With truncated set, we assume epsilon_{G-G'} = 0 if |G-G'| > N?
                        # No, that's not right. The Fourier series is truncated.
                        # The matrix element is sum over G'' of epsilon_{G-G''} ... 
                        # Wait, the standard PW matrix for TM is:
                        # H_{GG'} = ( (k+G) . (k+G') ) / epsilon_{G-G'}
                        # This assumes that 1/epsilon is expanded in Fourier series.
                        # If we truncate 1/epsilon to N, then epsilon_{G-G'} is 0 if |G-G'| > N.
                        # So we just check if the index is valid.
                        continue
                    
                    # Dot product (k+G) . (k+G')
                    kx_gp = kx + 2*np.pi*gpx
                    ky_gp = ky + 2*np.pi*gpy
                    dot = kx_g * kx_gp + ky_g * ky_gp
                    
                    M[i, j] += dot * eps_inv # Wait, the loop structure is wrong.
                    # M[i,j] is for G_i, G_j.
                    # The term is (k+G_i) . (k+G_j) / epsilon_{G_i - G_j}
                    # So for fixed i, j, we just compute this value.
                    # We don't sum over ii, jj inside the i,j loop.
                    # The matrix is diagonal in the "convolution" sense? No.
                    # H_{G G'} = (k+G) . (k+G') * (1/epsilon)_{G-G'}
                    # So M[i, j] = (k+G_i) . (k+G_j) * eps_inv[G_i - G_j]
                    
    # Re-do matrix construction correctly
    M = np.zeros((len(Gs), len(Gs)), dtype=complex)
    for i, gx in enumerate(Gs):
        kx_g = kx + 2*np.pi*gx
        ky_g = ky + 2*np.pi*gy
        for j, gpx in enumerate(Gs):
            kx_gp = kx + 2*np.pi*gpx
            ky_gp = ky + 2*np.pi*gpy
            dot = kx_g * kx_gp + ky_g * ky_gp
            
            dgx = gx - gpx
            dgy = gy - gpy
            idx_x = dgx + N
            idx_y = dgy + N
            if 0 <= idx_x < len(Gs) and 0 <= idx_y < len(Gs):
                eps_inv = 1.0 / epsilon_G[idx_x, idx_y]
                M[i, j] = dot * eps_inv
            else:
                M[i, j] = 0.0
                
    # Solve eigenvalue problem
    # M is Hermitian?
    # M_{G G'} = (k+G).(k+G') / eps_{G-G'}
    # M_{G' G} = (k+G').(k+G) / eps_{G'-G}
    # eps_{G'-G} = conj(eps_{G-G'}) if eps is real?
    # eps(r) is real, so eps_G = conj(eps_{-G}).
    # So eps_{G'-G} = conj(eps_{G-G'}).
    # Thus M_{G' G} = dot / conj(eps) = conj( dot / eps ) = conj(M_{G G'}).
    # So M is Hermitian.
    
    try:
        vals, vecs = np.linalg.eigh(M)
        # Filter out small negative eigenvalues due to numerical error
        vals = np.real(vals)
        vals = vals[vals > 1e-10]
        if len(vals) < num_bands:
            return []
        return vals[:num_bands]
    except:
        return []

def design_square_hole(t):
    """Dielectric background, air square hole of size t."""
    size = 32
    cell = [[1 for _ in range(size)] for _ in range(size)]
    # Center is (15.5, 15.5)
    # Hole from (16 - t*16) to (16 + t*16)?
    # Normalized coords: x in [0,1]. Center 0.5.
    # Hole width t.
    # x in [0.5 - t/2, 0.5 + t/2]
    start = int((0.5 - t/2) * size)
    end = int((0.5 + t/2) * size)
    for i in range(start, end):
        for j in range(start, end):
            cell[i][j] = 0
    return cell

def design_square_rod(t):
    """Air background, dielectric square rod of size t."""
    size = 32
    cell = [[0 for _ in range(size)] for _ in range(size)]
    start = int((0.5 - t/2) * size)
    end = int((0.5 + t/2) * size)
    for i in range(start, end):
        for j in range(start, end):
            cell[i][j] = 1
    return cell

def design_cross_rod(t):
    """Air background, dielectric cross. Width t."""
    size = 32
    cell = [[0 for _ in range(size)] for _ in range(size)]
    # Cross defined by |x-0.5| < t/2 OR |y-0.5| < t/2
    half_t = t / 2
    for i in range(size):
        x = (i + 0.5) / size
        for j in range(size):
            y = (j + 0.5) / size
            if abs(x - 0.5) < half_t or abs(y - 0.5) < half_t:
                cell[i][j] = 1
    return cell

def design_cross_hole(t):
    """Dielectric background, air cross. Width t."""
    size = 32
    cell = [[1 for _ in range(size)] for _ in range(size)]
    half_t = t / 2
    for i in range(size):
        x = (i + 0.5) / size
        for j in range(size):
            y = (j + 0.5) / size
            if abs(x - 0.5) < half_t or abs(y - 0.5) < half_t:
                cell[i][j] = 0
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Fast solver parameters
    N_fast = 4 # (2*4+1)^2 = 81 PWs
    # K points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5), K (0.25, 0.5)
    # Irreducible BZ for square: Triangle Gamma-M-X
    # Gamma: (0,0)
    # M: (0.5, 0)
    # X: (0.5, 0.5)
    # Midpoint Gamma-M: (0.25, 0)
    # Midpoint Gamma-X: (0.25, 0.25)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25)
    ]
    
    candidates = []
    # Generate candidate designs
    for t in np.arange(0.2, 0.9, 0.05):
        candidates.append(design_square_hole(t))
        candidates.append(design_square_rod(t))
        candidates.append(design_cross_rod(t))
        candidates.append(design_cross_hole(t))
        
    best_score = -1
    best_cell = None
    
    for cell in candidates:
        if time.time() > deadline:
            break
            
        # Compute epsilon Fourier for this cell
        try:
            epsilon_G, Gs = compute_epsilon_fourier(cell, N_fast)
        except:
            continue
            
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            try:
                bands = solve_tm_bands(epsilon_G, Gs, kx, ky, num_bands=5)
                if len(bands) < 5:
                    continue
                # bands are omega^2
                # omega = sqrt(band)
                w4 = np.sqrt(bands[3])
                w5 = np.sqrt(bands[4])
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
            except:
                continue
                
        if min_omega_5 > max_omega_4:
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2
            score = 2 * gap / midgap if midgap > 0 else 0
            if score > best_score:
                best_score = score
                best_cell = cell
                
    if best_cell is not None:
        return best_cell
    else:
        # Fallback to seed
        size = 32
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
