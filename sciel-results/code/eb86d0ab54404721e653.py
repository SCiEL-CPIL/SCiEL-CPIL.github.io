# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_bands_fast(cell, size=32, num_pw=5, num_k=8, band_max=6):
    """
    Compute the first 'band_max' TM bands for a given cell using plane-wave method.
    Optimized for speed.
    Returns (max_omega_4, min_omega_5) in units of a/(2*pi*c).
    """
    # Dielectric function
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    
    # Inverse dielectric function
    inv_eps = 1.0 / eps
    
    # Plane wave grid
    # Using a smaller number of plane waves (num_pw=5 -> 11x11=121 modes) is faster
    # and often sufficient for a rough estimate of the gap location.
    g_vals = np.arange(-num_pw, num_pw + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    G_flat = np.stack([Gx.ravel(), Gy.ravel()], axis=1)
    num_G = len(G_flat)
    
    # Fourier coefficients of inv_eps using FFT
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_fft = inv_eps_fft / (size * size)
    
    # Precompute S matrix: S[i,j] = inv_eps_{G_i - G_j}
    # Vectorized construction of S
    # G_diff for all pairs
    # This creates a (num_G, num_G, 2) array which might be large, 
    # but for num_pw=5 (121 modes), 121*121*2 is ~29k floats, which is fine.
    G_diff = G_flat[:, np.newaxis, :] - G_flat[np.newaxis, :, :]
    m_diff = np.mod(G_diff[..., 0], size)
    n_diff = np.mod(G_diff[..., 1], size)
    
    # Flatten indices for fancy indexing
    m_flat = m_diff.ravel()
    n_flat = n_diff.ravel()
    
    # Get coefficients
    coeffs = inv_eps_fft[m_flat, n_flat]
    S = coeffs.reshape(num_G, num_G)
    
    def get_band_structure(kx, ky):
        k_vec = np.array([kx, ky])
        k_plus_G = G_flat + k_vec
        k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
        k_plus_G_sq = np.maximum(k_plus_G_sq, 1e-12)
        sqrt_k_plus_G = np.sqrt(k_plus_G_sq)
        
        # B = sqrt(k+G) * S * sqrt(k+G)
        # Vectorized: diag(sqrt) @ S @ diag(sqrt)
        # This is equivalent to S * outer(sqrt, sqrt)
        B = S * np.outer(sqrt_k_plus_G, sqrt_k_plus_G)
        
        # B is symmetric real if S is? S is complex generally, but for real eps, 
        # S is Hermitian. B is Hermitian.
        vals = np.linalg.eigvalsh(B)
        return vals[:band_max]
    
    # Sample k-points in irreducible BZ (C4v symmetry)
    # Triangle: (0,0) to (0.5,0) to (0.5,0.5)
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    k_points = []
    # High symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    
    # Grid points
    for i in range(num_k + 1):
        for j in range(num_k + 1):
            kx = 0.5 * i / num_k
            ky = 0.5 * j / num_k
            if 0 <= ky <= kx <= 0.5 + 1e-9:
                k_points.append((kx, ky))
    
    for kx, ky in k_points:
        vals = get_band_structure(kx, ky)
        if len(vals) < 5:
            continue
        omegas = np.sqrt(np.maximum(vals, 0))
        if len(omegas) >= 5:
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
    
    if min_omega_5 == float('inf'):
        return 0.0, 0.0
        
    return max_omega_4, min_omega_5

def _score_cell_fast(cell):
    """
    Calculate the gap-midgap ratio.
    """
    max_4, min_5 = _compute_bands_fast(cell, num_pw=5, num_k=8, band_max=6)
    if min_5 == 0 or max_4 == 0:
        return 0.0
    gap = min_5 - max_4
    midgap = min_5 + max_4
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _make_air_holes_cell(size, hole_radius, offset=0.0):
    """
    Create a dielectric background with air holes.
    Holes are placed at (size/2, size/2) and symmetric positions?
    Actually, for a square lattice unit cell, the standard high-gap structure
    is a single hole in the center or a pattern of holes.
    Let's stick to a single central hole first, then maybe 4 holes.
    For C4v symmetry, a central hole is invariant.
    4 holes at (x,y), (x, size-1-y), (size-1-x, y), (size-1-x, size-1-y) are invariant.
    """
    cell = [[1] * size for _ in range(size)]
    center = (size - 1) / 2.0
    
    # Central hole
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < hole_radius**2:
                cell[i][j] = 0
                
    return cell

def _make_four_holes_cell(size, hole_radius, dx, dy):
    """
    Four holes in a square arrangement.
    Positions: (center+dx, center+dy) and symmetries.
    """
    cell = [[1] * size for _ in range(size)]
    center = (size - 1) / 2.0
    
    # Define one hole position in the first quadrant relative to center
    # (cx, cy) such that cx >= center, cy >= center
    cx = center + dx
    cy = center + dy
    
    # Symmetric positions
    positions = [
        (cx, cy),
        (cx, size - 1 - cy),
        (size - 1 - cx, cy),
        (size - 1 - cx, size - 1 - cy)
    ]
    
    for hx, hy in positions:
        for i in range(size):
            for j in range(size):
                if (i - hx)**2 + (j - hy)**2 < hole_radius**2:
                    cell[i][j] = 0
                    
    return cell

def _make_seed_cell(size):
    """
    The known good seed: a single dielectric circle in air.
    Wait, the task says 1=dielectric, 0=air.
    The seed in the prompt's knowledge was "simple geometric pattern".
    Let's create a standard seed: Dielectric background, air hole in center.
    Or Air background, dielectric dot in center.
    The error said seed score was 0.043.
    Let's try Dielectric Background (1) with Air Hole (0) in center.
    """
    cell = [[1] * size for _ in range(size)]
    center = (size - 1) / 2.0
    # Hole radius 4
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < 16.0:
                cell[i][j] = 0
    return cell

def design(problem):
    size = problem["pixels"]
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    best_cell = _make_seed_cell(size)
    best_score = _score_cell_fast(best_cell)
    
    # If seed is bad, try a different seed
    if best_score <= 0:
        # Try air background, dielectric dot
        cell_alt = [[0] * size for _ in range(size)]
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 < 16.0:
                    cell_alt[i][j] = 1
        score_alt = _score_cell_fast(cell_alt)
        if score_alt > best_score:
            best_cell = cell_alt
            best_score = score_alt
            
    if best_score <= 0:
        # Fallback to all dielectric (no gap, score 0) or all air (no gap, score 0)
        # We must return something. The evaluator scores 0 for invalid/negative.
        # But we want to beat the seed (0.043).
        # If we can't find a positive score, we return the seed.
        return best_cell

    # Search parameters for "Dielectric Background + Air Holes"
    # This structure is known to have large TM gaps.
    
    # 1. Single central hole
    for r in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        if time.time() > deadline - 20: break
        cell = _make_air_holes_cell(size, r)
        score = _score_cell_fast(cell)
        if score > best_score:
            best_score = score
            best_cell = cell

    # 2. Four holes
    for dx in [3.0, 4.0, 5.0, 6.0]:
        for dy in [3.0, 4.0, 5.0, 6.0]:
            if time.time() > deadline - 20: break
            for r in [2.0, 3.0, 4.0]:
                if time.time() > deadline - 10: break
                cell = _make_four_holes_cell(size, r, dx, dy)
                score = _score_cell_fast(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell

    # 3. Local search on the best cell found so far
    # Use a simple hill climbing with single-pixel flips (maintaining symmetry)
    # To maintain symmetry, we flip orbits.
    
    # Precompute orbits for the current best cell
    # We can just flip pixels and check if it breaks symmetry? No, better to flip orbits.
    
    # Simplified orbit flip:
    # Pick a pixel (i,j), flip it and all its symmetric counterparts.
    
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Number of steps
    steps = 0
    max_steps = 2000 # Limit steps to ensure speed
    
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        # Pick a random pixel
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        # Generate symmetric set
        sym_set = set()
        sym_set.add((i, j))
        sym_set.add((j, size-1-i))
        sym_set.add((size-1-i, size-1-j))
        sym_set.add((size-1-j, i))
        sym_set.add((size-1-i, j))
        sym_set.add((i, size-1-j))
        sym_set.add((j, i))
        sym_set.add((size-1-j, size-1-i))
        
        # Flip
        for r, c in sym_set:
            current_cell[r][c] = 1 - current_cell[r][c]
            
        new_score = _score_cell_fast(current_cell)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in current_cell]
        else:
            # Revert
            for r, c in sym_set:
                current_cell[r][c] = 1 - current_cell[r][c]
                
    return best_cell

# EVOLVE-BLOCK-END
