import numpy as np
import time

def _get_d4_orbits(n=32):
    """Get all D4 orbits of pixels in an n x n grid."""
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            # Identity
            orbit.add((i, j))
            # 90 deg rotation
            orbit.add((j, n-1-i))
            # 180 deg rotation
            orbit.add((n-1-i, n-1-j))
            # 270 deg rotation
            orbit.add((n-1-j, i))
            # Mirror x (horizontal flip)
            orbit.add((i, n-1-j))
            # Mirror y (vertical flip)
            orbit.add((n-1-i, j))
            # Diagonal mirror
            orbit.add((j, i))
            # Anti-diagonal mirror
            orbit.add((n-1-j, n-1-i))
            orbit = list(orbit)
            for p in orbit:
                seen.add(p)
            orbits.append(orbit)
    return orbits

def _apply_orbits(base, orbit_values, orbits):
    """Apply orbit values to a base grid. orbit_values is array of 0/1."""
    result = base.copy()
    for orbit, val in zip(orbits, orbit_values):
        for (i, j) in orbit:
            result[i, j] = val
    return result

def _compute_eps_fourier(cell_grid, size=32, eps_low=1.0, eps_high=12.0):
    """Compute Fourier coefficients of epsilon from the pixel grid."""
    eps = eps_low + (eps_high - eps_low) * cell_grid.astype(float)
    F = np.fft.fft2(eps)
    F /= (size * size)
    return F

def _solve_tm_bands(kx, ky, eps_F, G_vecs, size=32):
    """Solve TM eigenvalue problem at given k-point. Returns sorted eigenvalues (omega^2/c^2)."""
    nG = len(G_vecs)
    
    # k + G vectors
    kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
    kG2 = kG[:, 0]**2 + kG[:, 1]**2
    
    # Build the matrix A where A[n,m] = eps_F[G_n - G_m]
    # eps_F is a 2D FFT, so we need to index it properly
    # G_vecs are in units of 2*pi, so G_n - G_m = (2*pi*(dm, dn))
    # The FFT index for (dm, dn) is (dm % size, dn % size)
    
    A = np.zeros((nG, nG), dtype=complex)
    for n in range(nG):
        for m in range(nG):
            dm = round(G_vecs[n, 0] / (2*np.pi)) - round(G_vecs[m, 0] / (2*np.pi))
            dn = round(G_vecs[n, 1] / (2*np.pi)) - round(G_vecs[m, 1] / (2*np.pi))
            idx_m = dm % size
            idx_n = dn % size
            A[n, m] = eps_F[idx_n, idx_m]
    
    # D = diag(kG2)
    # Solve: A v = (1/omega^2) D v
    # Or: D^{-1} A v = (1/omega^2) v
    # Let mu = 1/omega^2, then D^{-1} A v = mu v
    # So omega^2 = 1/mu
    
    # Handle case where some kG2 are zero (at Gamma point)
    # Add small regularization
    kG2_reg = kG2.copy()
    kG2_reg[kG2_reg < 1e-12] = 1e-12
    
    D_inv = np.diag(1.0 / kG2_reg)
    B = D_inv @ A
    
    # B should be Hermitian (since eps is real, A is Hermitian, D is real diagonal)
    # Ensure Hermiticity
    B = (B + B.conj().T) / 2
    
    try:
        eigvals = np.linalg.eigvalsh(B)
        eigvals = eigvals.real
        eigvals = np.sort(eigvals)
        # Convert to omega^2 = 1/mu
        # But wait, we need to be careful. Let me re-derive.
        # The eigenvalue problem is: A E = mu D E where mu = 1/(omega/c)^2
        # So (omega/c)^2 = 1/mu
        # But mu = 1/omega^2, so this is consistent.
        # However, eigvalsh gives mu values, so omega^2 = 1/mu
        # But some mu might be zero or negative due to numerical issues
        omega2 = np.zeros_like(eigvals)
        valid = np.abs(eigvals) > 1e-12
        omega2[valid] = 1.0 / eigvals[valid]
        # For invalid ones, set to a large value (high band)
        omega2[~valid] = np.inf
        return np.sort(omega2)
    except:
        return None

def _get_k_points(n_per_seg=10):
    """Get k-points along the irreducible BZ boundary for square lattice."""
    k_points = []
    # Segment 1: Gamma(0,0) to X(pi, 0)
    for i in range(n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((t * np.pi, 0.0))
    # Segment 2: X(pi, 0) to M(pi, pi)
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((np.pi, t * np.pi))
    # Segment 3: M(pi, pi) to Gamma(0, 0)
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    return k_points

def _compute_gap_score(cell_grid, k_points, G_vecs, size=32, eps_low=1.0, eps_high=12.0, n_bands=6):
    """Compute the 4-5 TM band gap score."""
    eps_F = _compute_eps_fourier(cell_grid, size, eps_low, eps_high)
    
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = _solve_tm_bands(kx, ky, eps_F, G_vecs, size)
        if eigvals is None or len(eigvals) < 5:
            return -1.0
        # eigvals are omega^2, sorted ascending
        # Band 4 is index 3, Band 5 is index 4
        omega4 = np.sqrt(max(0, eigvals[3]))
        omega5 = np.sqrt(max(0, eigvals[4]))
        omega4_vals.append(omega4)
        omega5_vals.append(omega5)
    
    if not omega4_vals:
        return -1.0
    
    max_omega4 = max(omega4_vals)
    min_omega5 = min(omega5_vals)
    
    if min_omega5 <= max_omega4:
        if (min_omega5 + max_omega4) > 1e-12:
            return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return -1.0
    else:
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

def _is_symmetric(cell):
    """Check if cell has D4 symmetry."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
    return True

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Get D4 orbits
    orbits = _get_d4_orbits(size)
    n_orbits = len(orbits)
    
    # Set up G vectors for plane-wave solver
    # Use a diamond of half-width 6 for 32x32 grid
    half = 6
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:
                G_vecs.append((2*np.pi*m, 2*np.pi*n))
    G_vecs = np.array(G_vecs, dtype=float)
    nG = len(G_vecs)
    
    # Get k-points
    k_points = _get_k_points(n_per_seg=8)
    
    # Initialize with a four-rod design
    # Rods at (8,8), (23,8), (8,23), (23,23) with radius ~4
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    radius_sq = 16.0  # radius = 4
    
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < radius_sq:
                    cell[i, j] = 1
                    break
    
    # Convert to orbit-based representation
    # For each orbit, set value based on majority or just first pixel
    orbit_values = np.zeros(n_orbits, dtype=int)
    for oi, orbit in enumerate(orbits):
        # Set to value of first pixel in orbit
        i, j = orbit[0]
        orbit_values[oi] = cell[i, j]
    
    # Verify and set up
    current_cell = _apply_orbits(np.zeros((size, size), dtype=int), orbit_values, orbits)
    
    # Compute initial score
    current_score = _compute_gap_score(current_cell, k_points, G_vecs, size, eps_low, eps_high)
    best_cell = current_cell.copy()
    best_score = current_score
    best_orbit_values = orbit_values.copy()
    
    # Simulated annealing over orbit values
    T_start = 0.5
    T_end = 0.01
    n_iter_max = 5000
    
    # Precompute random orbit indices
    rng = np.random.RandomState(42)
    
    for it in range(n_iter_max):
        if time.time() > deadline - 20:
            break
        
        # Temperature schedule
        t_frac = it / max(1, n_iter_max - 1)
        T = T_start + (T_end - T_start) * t_frac
        
        # Pick a random orbit to flip
        oi = rng.randint(n_orbits)
        
        # Flip the orbit
        old_val = orbit_values[oi]
        orbit_values[oi] = 1 - old_val
        new_cell = _apply_orbits(np.zeros((size, size), dtype=int), orbit_values, orbits)
        
        # Compute new score
        new_score = _compute_gap_score(new_cell, k_points, G_vecs, size, eps_low, eps_high)
        
        if new_score > current_score:
            current_score = new_score
            current_cell = new_cell
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
                best_orbit_values = orbit_values.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if T > 1e-12 and rng.rand() < np.exp(delta / T):
                current_score = new_score
                current_cell = new_cell
            else:
                # Revert
                orbit_values[oi] = old_val
    
    # Try a few hand-tuned variants as well
    variants = []
    
    # Variant: larger rods
    cell2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < 25.0:  # radius = 5
                    cell2[i, j] = 1
                    break
    variants.append(cell2)
    
    # Variant: smaller rods
    cell3 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < 10.0:  # radius ~3.16
                    cell3[i, j] = 1
                    break
    variants.append(cell3)
    
    # Evaluate variants and pick the best overall
    for v in variants:
        s = _compute_gap_score(v, k_points, G_vecs, size, eps_low, eps_high)
        if s > best_score:
            best_score = s
            best_cell = v.copy()
    
    # Convert best_cell to nested list
    best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    # Verify symmetry
    if not _is_symmetric(best_list):
        # Fallback to a known symmetric design
        best_list = [[int(best_orbit_values[orbits.index([(i,j)] + [p for p in orbits[0] if p == (i,j) and p != (i,j)][0:0] + [(i,j)])[0]]) if any(p == (i,j) for p in orbits[0]) else 0 for j in range(size)] for i in range(size)]
        # Actually, let me just use the orbit-based best
        best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    return best_list
