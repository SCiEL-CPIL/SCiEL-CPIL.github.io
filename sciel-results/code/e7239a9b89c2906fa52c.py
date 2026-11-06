"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_gap_fast(cell, eps_high=12.0, N_pw=15, k_points=None):
    """
    Fast proxy for the 4th-5th TM band gap.
    Uses a truncated plane-wave basis.
    """
    N = len(cell)
    cell_arr = np.array(cell, dtype=float)
    
    # Compute Fourier coefficients of 1/epsilon (inverse dielectric)
    # For TM mode, the operator involves 1/eps
    inv_eps = 1.0 / (1.0 + (eps_high - 1.0) * cell_arr)
    
    # FFT to get Fourier coefficients
    # We need G vectors in the unit cell.
    # The lattice constant is 1.
    # G = (h, k) where h, k are integers.
    # The plane waves are exp(i G . r).
    # The matrix element H_{G, G'} = |G-G'|^2 * (1/eps)_{G-G'}
    
    # We select a set of G vectors
    # Symmetric selection around 0
    # Let's pick G vectors such that |G| <= G_max
    # For N_pw=15, we can pick a diamond or circle of G vectors.
    # Simple approach: pick h, k in range [-M, M] and filter by radius.
    # Or just pick the first N_pw/2 unique symmetric vectors.
    
    # Let's generate G vectors
    max_g = 4
    g_vectors = []
    for h in range(-max_g, max_g + 1):
        for k in range(-max_g, max_g + 1):
            if h == 0 and k == 0:
                continue
            g_vectors.append((h, k))
    
    # Sort by magnitude
    g_vectors.sort(key=lambda x: x[0]**2 + x[1]**2)
    
    # We need to include 0 vector
    all_g = [(0,0)] + g_vectors
    # Limit to N_pw vectors? No, N_pw is the basis size.
    # Let's take the smallest N_pw G vectors.
    # But we need symmetry.
    # Actually, for a quick proxy, we can just use a fixed set of G vectors.
    # Let's use G vectors with |G| <= 2 (i.e., h,k in {-2,-1,0,1,2})
    # That is 5x5 = 25 vectors.
    # 25x25 matrix is small.
    
    g_set = []
    for h in range(-2, 3):
        for k in range(-2, 3):
            g_set.append((h, k))
    N_g = len(g_set)
    
    # Compute (1/eps)_G for these G vectors
    # (1/eps)_G = (1/N^2) * sum_{x,y} (1/eps)(x,y) * exp(-i G . (x,y))
    # Using FFT:
    # fft2 gives coefficients for G = (h, k) where h, k are indices.
    # We need to map our integer G vectors to FFT indices.
    # FFT indices: 0, 1, ..., N-1.
    # G = (h, k) corresponds to index (h % N, k % N).
    
    fft_inv_eps = np.fft.fft2(inv_eps)
    fft_inv_eps = fft_inv_eps / (N * N)
    
    # Build H matrix
    H = np.zeros((N_g, N_g), dtype=complex)
    
    # Precompute k-point?
    # We need to evaluate at several k-points in the irreducible BZ.
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx+ky <= pi (in units of 2pi/a? No, BZ boundary is pi/a).
    # Let's use k in [0, 1] x [0, 1] representing fractions of 2pi/a?
    # Standard BZ for square lattice a=1: [-pi, pi] x [-pi, pi].
    # Irreducible sector: 0 <= kx <= ky, kx + ky <= pi.
    # Let's sample a few k-points:
    # Gamma: (0, 0)
    # M: (pi, 0) -> (1, 0) in units of pi? No, let's use radians.
    # X: (pi, pi) -> (1, 1) in units of pi?
    # Let's use k_x, k_y in radians.
    
    k_pts = [
        (0.0, 0.0),      # Gamma
        (np.pi/2, 0.0),  # Midpoint Gamma-X
        (np.pi, 0.0),    # X (M in some conventions, but corner is (pi,pi))
        (np.pi/2, np.pi/2), # Center of edge? No, this is inside.
        (np.pi, np.pi)   # Corner
    ]
    
    # Wait, the irreducible BZ of a square lattice is the triangle with vertices (0,0), (pi,0), (pi,pi).
    # Symmetry operations: kx<->ky, kx->-kx, etc.
    # So we can restrict to 0 <= kx <= ky and kx+ky <= pi?
    # Let's check: (pi, 0) is on the boundary. (pi, pi) is the corner.
    # The line kx=ky is a symmetry axis.
    # The line kx+ky=pi is the boundary of the BZ?
    # The BZ is [-pi, pi] x [-pi, pi].
    # The irreducible sector is 0 <= kx <= ky, kx + ky <= pi.
    # Vertices: (0,0), (pi/2, pi/2), (pi, 0).
    
    k_pts = [
        (0.0, 0.0),
        (np.pi/4, np.pi/4),
        (np.pi/2, np.pi/2),
        (np.pi, 0.0),
        (np.pi/3, np.pi/3)
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_pts:
        # Construct H matrix for this k
        # H_{ij} = |K_i - K_j|^2 * (1/eps)_{K_i - K_j}
        # Where K_i = k + G_i
        
        K = np.array([(kx + h, ky + k) for h, k in g_set])
        
        for i in range(N_g):
            for j in range(N_g):
                G_diff = (g_set[i][0] - g_set[j][0], g_set[i][1] - g_set[j][1])
                # Get (1/eps)_{G_diff}
                # FFT index for G_diff
                idx_h = G_diff[0] % N
                idx_k = G_diff[1] % N
                coeff = fft_inv_eps[idx_h, idx_k]
                
                # |K_i - K_j|^2
                dK = K[i] - K[j]
                dist_sq = np.dot(dK, dK)
                
                H[i, j] = dist_sq * coeff
        
        # Solve generalized eigenvalue problem?
        # No, H is the operator. We solve det(H - omega^2 I) = 0.
        # So eigenvalues of H are omega^2.
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.real(eigvals)
        eigvals = np.sort(eigvals)
        
        # We want bands 4 and 5.
        # In a 2D square lattice, the number of bands per k-point is N_g.
        # We need the 4th and 5th smallest positive eigenvalues.
        # Note: The zero eigenvalue might appear if there are degeneracies or if the basis is not full?
        # Usually, the lowest band is non-zero.
        # Let's take eigvals[3] and eigvals[4] (0-indexed).
        
        if len(eigvals) >= 5:
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            
            if w4_sq > 0 and w5_sq > 0:
                w4 = np.sqrt(w4_sq)
                w5 = np.sqrt(w5_sq)
                
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
    
    if min_omega_5 == np.inf or max_omega_4 == -np.inf:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    return gap / midgap

def design(problem):
    """
    Searches for a 4-fold symmetric pattern maximizing the 4-5 TM band gap.
    Uses a fast proxy solver.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 100  # 100s budget for search, leaving time for safety
    
    best_score = -1.0
    best_cell = None
    
    # Define a set of candidate patterns
    # 1. Dielectric Cross (plus sign)
    # 2. Air Cross (minus sign)
    # 3. Ring
    # 4. 2x2 Rods
    
    candidates = []
    
    # Parameter space
    # Widths from 2 to 16 pixels
    widths = [2, 4, 6, 8, 10, 12, 14]
    
    # 1. Dielectric Cross
    for w in widths:
        cell = np.zeros((size, size), dtype=int)
        # Horizontal bar
        cell[size//2 - w//2 : size//2 + w//2, :] = 1
        # Vertical bar
        cell[:, size//2 - w//2 : size//2 + w//2] = 1
        candidates.append(cell.tolist())
        
    # 2. Air Cross (Dielectric background with air cross)
    for w in widths:
        cell = np.ones((size, size), dtype=int)
        # Air horizontal bar
        cell[size//2 - w//2 : size//2 + w//2, :] = 0
        # Air vertical bar
        cell[:, size//2 - w//2 : size//2 + w//2] = 0
        candidates.append(cell.tolist())
        
    # 3. Ring
    # Inner radius r_in, outer radius r_out
    for r_in in range(2, 10, 2):
        for r_out in range(r_in + 2, 16, 2):
            cell = np.zeros((size, size), dtype=int)
            x, y = np.meshgrid(np.arange(size), np.arange(size))
            x = x - size/2.0
            y = y - size/2.0
            r = np.sqrt(x**2 + y**2)
            cell[(r >= r_in) & (r <= r_out)] = 1
            candidates.append(cell.tolist())
            
    # 4. 2x2 Rods
    # Centered at (8,8), (23,8), (8,23), (23,23) approx
    # Let's place them at 1/4 and 3/4 positions
    pos = size // 4
    pos2 = 3 * size // 4
    for r in range(2, 8, 2):
        cell = np.zeros((size, size), dtype=int)
        for cx in [pos, pos2]:
            for cy in [pos, pos2]:
                x, y = np.meshgrid(np.arange(size), np.arange(size))
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                cell[dist < r] = 1
        candidates.append(cell.tolist())
        
    # 5. Checkerboard-like or other high symmetry
    # Maybe a 4x4 supercell pattern repeated?
    # Let's stick to the above.
    
    # Evaluate candidates
    for cell in candidates:
        if time.time() > deadline:
            break
            
        score = _compute_gap_fast(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # If no good candidate found, fallback to seed
    if best_cell is None:
        # Return seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell
# EVOLVE-BLOCK-END
