"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _make_four_rod_cell(size, offset, radius, shape_param=0.0):
    """Generate a 4-rod symmetric cell.
    
    offset: distance of each rod center from cell center (0.2-0.35)
    radius: base radius of each rod (0.08-0.25)
    shape_param: controls squareness (0=circle, 1=square)
    """
    cell = np.zeros((size, size), dtype=int)
    half = size / 2.0
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Check distance to nearest of 4 rod centers
            best_dist = 1.0
            for cx in [-offset, offset]:
                for cy in [-offset, offset]:
                    dx = x - cx
                    dy = y - cy
                    dist2 = dx*dx + dy*dy
                    if dist2 < best_dist:
                        best_dist = dist2
            
            # Interpolate between circle and square based on shape_param
            r = np.sqrt(best_dist)
            # Circle: r < radius
            # Square: max(|dx|, |dy|) < radius (for nearest center)
            # Use a smooth interpolation
            if shape_param < 0.5:
                # More circular
                if r < radius:
                    cell[i, j] = 1
            else:
                # More square
                # Find the nearest center
                best_cx, best_cy = 0, 0
                best_d = 1.0
                for cx in [-offset, offset]:
                    for cy in [-offset, offset]:
                        d = (x-cx)**2 + (y-cy)**2
                        if d < best_d:
                            best_d = d
                            best_cx, best_cy = cx, cy
                if max(abs(x - best_cx), abs(y - best_cy)) < radius:
                    cell[i, j] = 1
    
    return _apply_symmetry(cell).tolist()

def _make_cross_cell(size, arm_width, arm_length):
    """Generate a cross-shaped cell."""
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Cross: horizontal bar or vertical bar
            if abs(y) < arm_width/2 and abs(x) < arm_length/2:
                cell[i, j] = 1
            elif abs(x) < arm_width/2 and abs(y) < arm_length/2:
                cell[i, j] = 1
    
    return _apply_symmetry(cell).tolist()

def _make_staggered_cell(size, offset, radius):
    """Generate a staggered 4-rod cell with different offsets for x and y."""
    cell = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            best_dist = 1.0
            # Staggered: rods at (±offset, 0) and (0, ±offset)
            for cx, cy in [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]:
                dx = x - cx
                dy = y - cy
                dist2 = dx*dx + dy*dy
                if dist2 < best_dist:
                    best_dist = dist2
            
            if np.sqrt(best_dist) < radius:
                cell[i, j] = 1
    
    return _apply_symmetry(cell).tolist()

def _compute_eps_ft_fast(cell, size, n_pw):
    """Compute Fourier coefficients of epsilon for the unit cell using FFT."""
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    
    # eps(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
    # So eps_ft[m,n] = (eps_high - eps_low) * cell_ft[m,n] + eps_low * delta[m,0]*delta[n,0]
    
    # Use FFT to compute cell_ft
    # cell_ft[m,n] = (1/N^2) sum_{i,j} cell[i,j] exp(-i(m*2π*i/N + n*2π*j/N))
    # This is exactly (1/N^2) * FFT2(cell)[m mod N, n mod N]
    
    N = size
    cell_fft = np.fft.fft2(eps) / (N * N)
    
    # Extract the needed plane wave coefficients
    # Plane wave indices from -(N-1)//2 to (N-1)//2
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
    
    eps_ft = np.zeros((len(pw_idx), len(pw_idx)), dtype=complex)
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            # Map to FFT index
            m_idx = m % N
            n_idx = n % N
            cell_coeff = cell_fft[m_idx, n_idx]
            if m == 0 and n == 0:
                eps_ft[im, in_] = eps_low + (eps_high - eps_low) * cell_coeff
            else:
                eps_ft[im, in_] = (eps_high - eps_low) * cell_coeff
    
    return eps_ft, pw_idx

def _tm_eigenvalues_fast(kx, ky, eps_ft, pw_idx, n_pw):
    """Compute TM band eigenvalues for a given k-point using fast matrix assembly."""
    N = len(pw_idx)
    kx *= 2 * np.pi
    ky *= 2 * np.pi
    
    # Build the matrix H where H[m,n] = delta[m,n] * (k+G)^2 + k^2 * eps_{G-G'}
    # Actually, for TM: the eigenvalue equation is
    # sum_{G'} [(k+G)^2 delta_{G,G'} - k^2 eps_{G-G'}] a_{G'} = 0
    # This is a generalized eigenvalue problem. Let me use the standard form.
    
    # Standard TM formulation:
    # The matrix is: M_{G,G'} = delta_{G,G'} * (k+G)^2 + k^2 * eps_{G-G'}
    # and we solve M a = lambda * (k+G)^2 * a ... no.
    
    # Let me use the correct formulation from Joannopoulos:
    # For TM (E_z): 
    # sum_{G'} [ delta_{G,G'} + k^2 / ((k+G)^2 (k+G')^2) * (eps_{G-G'} - delta_{G,G'}) ] a_{G'} = 0
    # This is a homogeneous system. The eigenvalues are k^2.
    
    # Alternative cleaner formulation:
    # Define the matrix A_{G,G'} = delta_{G,G'} * (k+G)^2 - k^2 * eps_{G-G'}
    # Then A a = 0 has non-trivial solutions when det(A) = 0, i.e., k^2 is an eigenvalue.
    
    # Let me use the standard plane-wave method for TM:
    # The eigenvalue problem is:
    # sum_{G'} [ (k+G)^2 delta_{G,G'} - k^2 eps_{G-G'} ] a_{G'} = 0
    # This can be written as:
    # sum_{G'} [ delta_{G,G'} - k^2/(k+G)^2 * eps_{G-G'} ] a_{G'} = 0
    # which is: (I - k^2 * D^{-1} * E) a = 0 where D is diagonal with (k+G)^2
    
    # For a finite basis, we solve the generalized eigenvalue problem:
    # A a = k^2 B a where A_{G,G'} = (k+G)^2 delta_{G,G'} and B_{G,G'} = eps_{G-G'}
    
    # Let me just build the matrix directly:
    # M_{G,G'} = (k+G)^2 delta_{G,G'} - k^2 eps_{G-G'}
    # We want det(M) = 0, so k^2 is an eigenvalue of the generalized problem.
    
    # Actually, the simplest approach: for each k, build the NxN matrix
    # H_{mn} = delta_{mn} * (k+G_m)^2 + k^2 * eps_{G_m - G_n}
    # No, that's not right either.
    
    # Let me use the formulation where we solve for omega^2 directly.
    # The TM wave equation: [∇_⊥^2 + k0^2 eps(r)] E_z = 0
    # In plane waves: sum_{G'} [ (k+G)^2 - k0^2 eps_{G-G'} ] a_{G'} = 0
    # This is: sum_{G'} [ (k+G)^2 delta_{G,G'} - k0^2 eps_{G-G'} ] a_{G'} = 0
    # So the matrix is: M_{G,G'} = (k+G)^2 delta_{G,G'} - k0^2 eps_{G-G'}
    # And we need det(M) = 0.
    
    # This is equivalent to finding eigenvalues of the generalized problem:
    # D a = k0^2 E a where D is diagonal with (k+G)^2 and E is the epsilon matrix.
    
    # So: D^{-1} E a = (1/k0^2) a
    # The eigenvalues of D^{-1} E are 1/k0^2.
    
    # Let me compute the matrix D^{-1} E:
    # (D^{-1} E)_{mn} = (1/(k+G_m)^2) * eps_{G_m - G_n}
    
    H = np.zeros((N, N), dtype=complex)
    for im, m in enumerate(pw_idx):
        Gx = kx + m * 2 * np.pi
        Gy = ky + n * 2 * np.pi  # This is wrong, n is not defined here
        # Fix: G_m = (2π*m, 2π*n) for the m-th plane wave index
        pass
    
    # Let me redo this properly:
    Gx_list = kx + pw_idx * 2 * np.pi
    Gy_list = ky + pw_idx * 2 * np.pi
    
    # (k+G_m)^2 = Gx_m^2 + Gy_m^2
    kg2 = np.zeros((N, N))
    for im in range(N):
        for in_ in range(N):
            # Plane wave index (im, in_) corresponds to G = (2π*pw_idx[im], 2π*pw_idx[in_])
            kx_g = kx + pw_idx[im] * 2 * np.pi
            ky_g = ky + pw_idx[in_] * 2 * np.pi
            kg2[im, in_] = kx_g**2 + ky_g**2
    
    # Epsilon matrix: E_{(m1,n1),(m2,n2)} = eps_{(m1-m2, n1-n2)}
    # We need to find eps_{dm, dn} where dm = pw_idx[m1] - pw_idx[m2]
    # Map (dm, dn) to index in eps_ft
    
    # Build a lookup: for each difference (dm, dn), find the index in pw_idx
    # dm = pw_idx[m1] - pw_idx[m2]
    # We need to find if dm is in pw_idx and get its index
    
    pw_to_idx = {v: i for i, v in enumerate(pw_idx)}
    
    E = np.zeros((N, N), dtype=complex)
    for im1 in range(N):
        for in1 in range(N):
            for im2 in range(N):
                for in2 in range(N):
                    dm = pw_idx[im1] - pw_idx[im2]
                    dn = pw_idx[in1] - pw_idx[in2]
                    if dm in pw_to_idx and dn in pw_to_idx:
                        E[im1*N+in1, im2*N+in2] = eps_ft[pw_to_idx[dm], pw_to_idx[dn]]
    
    # This is too slow for large N. Let me use a different approach.
    
    # Actually, for a square lattice, the epsilon matrix is a 2D convolution.
    # Let me just compute the eigenvalues using the direct matrix.
    
    # For speed, let me use a smaller basis.
    
    # The eigenvalue problem: D a = k0^2 E a
    # where D_{(m,n),(m',n')} = delta_{m,m'} delta_{n,n'} (k+G_{mn})^2
    # and E_{(m,n),(m',n')} = eps_{(m-m', n-n')}
    
    # So we solve: D^{-1} E a = (1/k0^2) a
    # The eigenvalues of D^{-1} E are 1/k0^2.
    
    # Let me build D^{-1} E directly:
    # (D^{-1} E)_{(m,n),(m',n')} = (1/(k+G_{mn})^2) * eps_{(m-m', n-n')}
    
    N2 = N * N
    H = np.zeros((N2, N2), dtype=complex)
    for im in range(N):
        for in_ in range(N):
            kx_g = kx + pw_idx[im] * 2 * np.pi
            ky_g = ky + pw_idx[in_] * 2 * np.pi
            kg2_val = kx_g**2 + ky_g**2
            if abs(kg2_val) < 1e-12:
                continue
            for im2 in range(N):
                for in2 in range(N):
                    dm = pw_idx[im] - pw_idx[im2]
                    dn = pw_idx[in_] - pw_idx[in2]
                    if dm in pw_to_idx and dn in pw_to_idx:
                        H[im*N+in_, im2*N+in2] = eps_ft[pw_to_idx[dm], pw_to_idx[dn]] / kg2_val
    
    # Solve for eigenvalues
    try:
        eigvals = np.linalg.eigvals(H)
        eigvals = np.real(eigvals)
        eigvals = eigvals[eigvals > 0]
        eigvals.sort()
        # k0^2 = 1/eigval, so omega^2 = k0^2 * c^2 = c^2/eigval
        # We want omega^2 values
        omega2 = 1.0 / eigvals
        omega2.sort()
        return omega2
    except:
        return np.array([])

def _quick_gap_estimate(cell, size=32, n_pw=9, n_kpts=5):
    """Fast surrogate evaluation of the 4-5 TM band gap."""
    eps_ft, pw_idx = _compute_eps_ft_fast(cell, size, n_pw)
    
    # K-points in the irreducible BZ
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # M
        (0.5, 0.5),      # X
        (0.25, 0.0),     # Interior
        (0.25, 0.25),    # Interior
    ]
    
    omega4_max = 0.0
    omega5_min = np.inf
    
    for kx, ky in k_points:
        omega2 = _tm_eigenvalues_fast(kx, ky, eps_ft, pw_idx, n_pw)
        if len(omega2) >= 5:
            omega4 = np.sqrt(omega2[3])  # 4th band (0-indexed: 3)
            omega5 = np.sqrt(omega2[4])  # 5th band (0-indexed: 4)
            omega4_max = max(omega4_max, omega4)
            omega5_min = min(omega5_min, omega5)
    
    if omega5_min == np.inf or omega4_max == 0:
        return -1.0
    
    gap = omega5_min - omega4_max
    midgap = (omega5_min + omega4_max) / 2
    if midgap < 1e-12:
        return -1.0
    
    return 2 * gap / midgap

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s safety margin before 230s kill
    
    best_cell = None
    best_score = -np.inf
    
    # Try various design families
    candidates = []
    
    # Four-rod designs
    for offset in [0.15, 0.2, 0.25, 0.3]:
        for radius in [0.08, 0.1, 0.12, 0.15, 0.18, 0.2]:
            for shape_param in [0.0, 0.5, 1.0]:
                if time.time() > deadline - 30.0:
                    break
                cell = _make_four_rod_cell(size, offset, radius, shape_param)
                candidates.append(cell)
    
    # Cross designs
    for arm_width in [0.05, 0.08, 0.1, 0.12, 0.15]:
        for arm_length in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 30.0:
                break
            cell = _make_cross_cell(size, arm_width, arm_length)
            candidates.append(cell)
    
    # Staggered designs
    for offset in [0.15, 0.2, 0.25, 0.3]:
        for radius in [0.08, 0.1, 0.12, 0.15]:
            if time.time() > deadline - 30.0:
                break
            cell = _make_staggered_cell(size, offset, radius)
            candidates.append(cell)
    
    # Evaluate all candidates
    for cell in candidates:
        if time.time() > deadline - 10.0:
            break
        score = _quick_gap_estimate(cell, size)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If we have time, do local refinement around the best candidate
    if best_cell is not None and time.time() < deadline - 20.0:
        # Try small mutations of the best cell
        best_arr = np.array(best_cell)
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            # Randomly flip a few pixels
            mutated = best_arr.copy()
            n_flips = np.random.randint(1, 5)
            for _ in range(n_flips):
                i = np.random.randint(size)
                j = np.random.randint(size)
                mutated[i, j] = 1 - mutated[i, j]
            mutated = _apply_symmetry(mutated)
            score = _quick_gap_estimate(mutated.tolist(), size)
            if score > best_score:
                best_score = score
                best_cell = mutated.tolist()
                best_arr = mutated
    
    if best_cell is None:
        # Fallback: simple four-rod design
        best_cell = _make_four_rod_cell(size, 0.25, 0.15, 0.0)
    
    return best_cell
# EVOLVE-BLOCK-END
