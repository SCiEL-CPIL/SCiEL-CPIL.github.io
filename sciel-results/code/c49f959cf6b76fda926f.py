# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_bands(cell, size=32, num_pw=7, num_k=10, band_max=6):
    """
    Compute the first 'band_max' TM bands for a given cell using plane-wave method.
    Returns (max_omega_4, min_omega_5) in units of a/(2*pi*c).
    """
    # Dielectric function
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    
    # Inverse dielectric function
    inv_eps = 1.0 / eps
    
    # Plane wave grid
    # G vectors such that |k+G| is covered.
    # Square grid of G vectors centered at 0.
    # G = (nx, ny) * 2*pi/a.
    # k is in units of 2*pi/a.
    
    g_vals = np.arange(-num_pw, num_pw + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    G_flat = np.stack([Gx.ravel(), Gy.ravel()], axis=1)
    num_G = len(G_flat)
    
    # Fourier coefficients of inv_eps
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_fft = inv_eps_fft / (size * size)
    
    # Map G_flat to FFT indices
    # G_flat contains integer coordinates m, n.
    # We need to wrap them to [0, size).
    fft_idx_m = G_flat[:, 0] % size
    fft_idx_n = G_flat[:, 1] % size
    
    inv_eps_G = inv_eps_fft[fft_idx_m, fft_idx_n]
    
    # Precompute difference indices for all pairs (i, j)
    # We need inv_eps_{G_i - G_j}
    # Let's build a lookup table for the coefficients
    # Since num_G is small (e.g. 225), we can precompute the matrix of coefficients
    # S[i, j] = inv_eps_{G_i - G_j}
    
    S = np.zeros((num_G, num_G), dtype=complex)
    for i in range(num_G):
        for j in range(num_G):
            G_diff = G_flat[i] - G_flat[j]
            m_diff = G_diff[0] % size
            n_diff = G_diff[1] % size
            S[i, j] = inv_eps_fft[m_diff, n_diff]
            
    def get_band_structure(kx, ky):
        # k is in units of 2*pi/a
        k_vec = np.array([kx, ky])
        
        # |k+G|^2
        k_plus_G = G_flat + k_vec
        k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
        # Ensure positive
        k_plus_G_sq = np.maximum(k_plus_G_sq, 1e-12)
        sqrt_k_plus_G = np.sqrt(k_plus_G_sq)
        
        # Construct Hermitian matrix B
        # B[i, j] = sqrt(|k+G_i|^2) * S[i, j] * sqrt(|k+G_j|^2)
        # B = diag(sqrt_k_plus_G) @ S @ diag(sqrt_k_plus_G)
        
        # Efficient multiplication
        # B = (sqrt_k_plus_G[:, None] * S) * sqrt_k_plus_G[None, :]
        B = (sqrt_k_plus_G[:, None] * S) * sqrt_k_plus_G[None, :]
        
        # Solve B y = omega^2 y
        # B is Hermitian
        # Use numpy.linalg.eigh for speed and availability
        vals, vecs = np.linalg.eigh(B)
        
        # Sort eigenvalues (eigh returns sorted)
        # Take the first band_max
        return vals[:band_max]
    
    # Sample k-points in irreducible BZ
    # Square BZ: kx, ky in [0, 1] (units 2*pi/a).
    # Irreducible zone for C4v: Triangle (0,0) - (0.5,0) - (0.5,0.5)
    # Condition: 0 <= ky <= kx <= 0.5
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    k_points = []
    # High symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    
    # Grid points in the triangle
    # We want to cover the area well.
    # Let's use a simple grid and filter
    grid_size = num_k
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            kx = 0.5 * i / grid_size
            ky = 0.5 * j / grid_size
            # Check if in triangle: 0 <= ky <= kx <= 0.5
            if 0 <= ky <= kx <= 0.5:
                k_points.append((kx, ky))
                
    # Remove duplicates
    k_points = list(set(k_points))
    
    for kx, ky in k_points:
        vals = get_band_structure(kx, ky)
        # vals are omega^2
        # We want omega
        omegas = np.sqrt(vals)
        if len(omegas) >= 5:
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
    if min_omega_5 == float('inf'):
        return 0.0, 0.0
        
    return max_omega_4, min_omega_5

def _score_cell(cell):
    """
    Calculate the gap-midgap ratio.
    """
    max_4, min_5 = _compute_bands(cell, num_pw=7, num_k=10, band_max=6)
    if min_5 == 0 or max_4 == 0:
        return 0.0
    gap = min_5 - max_4
    midgap = min_5 + max_4
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _make_cell_from_rings(size, ring_bits):
    cell = [[0] * size for _ in range(size)]
    half = size / 2.0
    for i in range(size):
        dx = abs(i - (size - 1) / 2.0)
        for j in range(size):
            dy = abs(j - (size - 1) / 2.0)
            r = int(max(dx, dy))
            if r < len(ring_bits):
                cell[i][j] = ring_bits[r]
    return cell

def design(problem):
    size = problem["pixels"]
    t0 = time.time()
    deadline = t0 + 110.0

    # 1. Generate Seed (Circle)
    seed_cell = [[0] * size for _ in range(size)]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                seed_cell[i][j] = 1

    best_cell = seed_cell
    best_score = _score_cell(seed_cell)
    
    # 2. Generate Candidates
    candidates = [seed_cell]
    
    # Square rods
    for r in range(3, 8):
        rings = [1 if i < r else 0 for i in range(size//2)]
        candidates.append(_make_cell_from_rings(size, rings))
        
    # Checkerboard
    checker = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i//2 + j//2) % 2 == 0:
                checker[i][j] = 1
    candidates.append(checker)
    
    # 3. Evaluate Candidates
    for cell in candidates:
        if time.time() > deadline - 10:
            break
        score = _score_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
            
    # 4. Local Search (Hill Climbing)
    # Flip pixels in symmetric orbits to improve score
    # We need to identify orbits under C4v symmetry.
    # An orbit is a set of pixels that map to each other under symmetry.
    # If we flip all pixels in an orbit, symmetry is preserved.
    
    # Precompute orbits
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            orbit = []
            # Generate symmetry equivalents
            # (i, j)
            # (size-1-i, j) ? No, reflection about center.
            # Center is (size-1)/2.
            # i' = size-1-i.
            # j' = size-1-j.
            # Rotations:
            # 90 deg: (i, j) -> (j, size-1-i)
            # 180 deg: (i, j) -> (size-1-i, size-1-j)
            # 270 deg: (i, j) -> (size-1-j, i)
            
            pts = [
                (i, j),
                (j, size-1-i),
                (size-1-i, size-1-j),
                (size-1-j, i),
                (size-1-i, j),
                (j, i),
                (size-1-i, size-1-j), # Duplicate of 180
                (size-1-j, size-1-i)
            ]
            # Unique points
            orbit = list(set(pts))
            # Mark visited
            for p in orbit:
                visited.add(p)
            orbits.append(orbit)
            
    # Shuffle orbits to avoid bias
    np.random.seed(42)
    np.random.shuffle(orbits)
    
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Hill climbing
    # Try flipping each orbit
    # If score improves, accept.
    # Limit number of flips to fit in time.
    
    max_flips = 500
    flips_done = 0
    
    while flips_done < max_flips and time.time() < deadline - 5:
        improved = False
        # Try a random subset of orbits? Or all?
        # All orbits is too many (32*32 / orbit_size).
        # Orbit size is usually 8. 1024/8 = 128 orbits.
        # 128 evaluations per iteration.
        # Each evaluation takes ~0.1s?
        # 128 * 0.1 = 12.8s per iteration.
        # We have 100s. So ~7 iterations.
        
        # Let's try all orbits in one pass.
        for orbit in orbits:
            if time.time() > deadline - 5:
                break
            # Flip orbit
            for (r, c) in orbit:
                current_cell[r][c] = 1 - current_cell[r][c]
                
            new_score = _score_cell(current_cell)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_cell = [row[:] for row in current_cell]
                improved = True
            else:
                # Revert
                for (r, c) in orbit:
                    current_cell[r][c] = 1 - current_cell[r][c]
                    
        flips_done += 1
        if not improved:
            # Stagnation? Try random restarts or break.
            # Let's break to save time for other candidates?
            # Or try a few random flips?
            break
            
    return best_cell

# EVOLVE-BLOCK-END
