"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _make_cell_from_map(mask):
    """
    Create a 32x32 cell from a boolean mask.
    1 = dielectric, 0 = air.
    """
    size = 32
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 if mask[i, j] else 0)
        cell.append(row)
    return cell

def _generate_candidate(center_r, ring_r, ring_width, n_holes=0, hole_r=0.0, hole_dist=0.0):
    """
    Generate a symmetric cell.
    center_r: radius of central dielectric disk (0 for none)
    ring_r: radius of dielectric ring (0 for none)
    ring_width: width of the ring
    n_holes: number of air holes in the dielectric background (if background is dielectric)
             or dielectric rods in air background.
             This logic is complex, so we stick to:
             Background is Air (0).
             We place Dielectric (1) features.
             Features:
             1. Central disk
             2. Ring
             3. 4 or 8 surrounding disks (holes in a superlattice sense, but here just added dielectric)
             
             Actually, standard high-gap TM crystals often have:
             - Dielectric background with air holes (rods of air)
             - Air background with dielectric rods
             
             Let's try both by swapping 0 and 1 in the mask logic if needed, 
             but the prompt says 1=dielectric, 0=air.
             
             Strategy:
             1. Air background (0), Dielectric rods (1).
             2. Dielectric background (1), Air holes (0).
             
             We will generate the mask for "Dielectric features in Air" and then
             also consider "Air features in Dielectric" by inverting the mask 
             and checking which yields a better score.
    """
    size = 32
    # Coordinates centered at 0, in units of lattice constant
    # Pixel centers are at (i+0.5)/32 - 0.5
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    mask = np.zeros((size, size), dtype=bool)
    
    # 1. Central Disk
    if center_r > 0:
        mask |= (X**2 + Y**2) < center_r**2
        
    # 2. Ring
    if ring_r > 0 and ring_width > 0:
        r2 = X**2 + Y**2
        mask |= (r2 >= (ring_r - ring_width/2)**2) & (r2 <= (ring_r + ring_width/2)**2)
        
    # 3. Surrounding Disks (e.g., 4 or 8)
    if n_holes > 0 and hole_r > 0 and hole_dist > 0:
        # 4 disks at (d, 0), (-d, 0), (0, d), (0, -d)
        if n_holes == 4:
            mask |= ((X - hole_dist)**2 + Y**2) < hole_r**2
            mask |= ((X + hole_dist)**2 + Y**2) < hole_r**2
            mask |= (X**2 + (Y - hole_dist)**2) < hole_r**2
            mask |= (X**2 + (Y + hole_dist)**2) < hole_r**2
        elif n_holes == 8:
            # 4 on axes, 4 on diagonals
            mask |= ((X - hole_dist)**2 + Y**2) < hole_r**2
            mask |= ((X + hole_dist)**2 + Y**2) < hole_r**2
            mask |= (X**2 + (Y - hole_dist)**2) < hole_r**2
            mask |= (X**2 + (Y + hole_dist)**2) < hole_r**2
            d_diag = hole_dist / np.sqrt(2)
            mask |= ((X - d_diag)**2 + (Y - d_diag)**2) < hole_r**2
            mask |= ((X + d_diag)**2 + (Y + d_diag)**2) < hole_r**2
            mask |= ((X - d_diag)**2 + (Y + d_diag)**2) < hole_r**2
            mask |= ((X + d_diag)**2 + (Y - d_diag)**2) < hole_r**2
            
    return mask

def _estimate_band_gap_fast(cell, eps_high=12):
    """
    Fast vectorized estimate of the band gap.
    Uses a small set of plane waves (N=7 or 9) for speed.
    """
    size = 32
    N = 9  # Number of G-vectors in each direction
    
    # Convert cell to epsilon map
    eps_map = np.array(cell, dtype=float)
    eps_map[eps_map == 0] = 1.0
    eps_map[eps_map == 1] = eps_high
    
    inv_eps = 1.0 / eps_map
    
    # Compute Fourier coefficients of 1/epsilon
    # We need inv_eps_{G} for G = (gx, gy) in the range used for the matrix
    # The matrix indices i, j correspond to G_i, G_j.
    # The term A[i,j] involves (k+G_i)^2 * (1/eps)_{G_i - G_j}
    
    # Generate G-vectors
    # G = (2pi/a) * (gx, gy). Since a=1 in our units, G = 2pi * (gx, gy).
    # The eigenvalue is (wa)^2.
    # Matrix element: (k+G_i)^2 * sum_G (1/eps)_G exp(i G . (r_i - r_j)) ... 
    # Actually, the standard PWE matrix for TM is:
    # sum_j (k+G_i)^2 (1/eps)_{G_i - G_j} E_j = (wa)^2 E_i
    # (1/eps)_G is the Fourier coefficient.
    
    # We need (1/eps)_{G} for G in the set {G_i - G_j}.
    # If G_i, G_j are in [-N/2, N/2], then G_i - G_j is in [-N, N].
    # So we need Fourier coeffs for G in [-N, N].
    
    G_max = N  # Max G index needed for differences
    
    # Compute Fourier coefficients for all G in [-G_max, G_max]
    # F[Gx, Gy] = sum_{x,y} inv_eps[x,y] exp(-i 2pi (Gx x + Gy y) / size) / size^2
    # Note: The pixel grid is discrete.
    
    # Vectorized FFT is faster, but we only need a subset.
    # Let's just compute the full 2D FFT of inv_eps.
    # The FFT gives coefficients for frequencies -size/2 to size/2-1.
    # Our G indices are integers.
    # F[kx, ky] from FFT corresponds to G = (kx, ky) if we shift properly.
    
    # inv_eps is 32x32.
    # fft2(inv_eps) gives F.
    # F[m, n] = sum_{x,y} inv_eps[x,y] exp(-i 2pi (m x + n y) / 32)
    # This matches the definition of Fourier coeff for G=(m,n) if we consider the domain 0..32.
    # However, our coordinates are centered. The FFT assumes periodicity 0..L.
    # Since the cell is periodic, this is fine.
    
    F = np.fft.fft2(inv_eps) / (size * size)
    
    # Map G indices to FFT indices
    # FFT index m corresponds to frequency m if m <= 16, and m-32 if m > 16.
    # We need G in [-9, 9].
    
    def get_F(gx, gy):
        # Convert G to FFT index
        ix = gx % size
        iy = gy % size
        return F[ix, iy]
    
    # Precompute F for all needed G differences
    # G_i, G_j in [-4, 4] (since N=9, range is -4 to 4)
    # G_diff in [-8, 8]
    
    G_list = []
    for gx in range(-N//2, N//2 + 1):
        for gy in range(-N//2, N//2 + 1):
            G_list.append((gx, gy))
            
    N_G = len(G_list)
    
    # Precompute (1/eps)_{G_i - G_j} for all i, j
    # This is a matrix of Fourier coefficients
    inv_eps_G = np.zeros((N_G, N_G), dtype=complex)
    for i, (gx_i, gy_i) in enumerate(G_list):
        for j, (gx_j, gy_j) in enumerate(G_list):
            dgx = gx_i - gx_j
            dgy = gy_i - gy_j
            inv_eps_G[i, j] = get_F(dgx, dgy)
            
    # K-points in irreducible BZ
    # 0 <= kx <= ky, kx + ky <= 0.5
    k_points = []
    # Gamma-X (0,0) to (0.5, 0)
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
    # X-M (0.5, 0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.5, t))
    # Gamma-M (0,0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, t))
        
    band_4_max = 0.0
    band_5_min = float('inf')
    
    for kx, ky in k_points:
        # Build Matrix A
        # A[i, j] = |k + G_i|^2 * (1/eps)_{G_i - G_j}
        # Note: The standard equation is sum_j (k+G_i)^2 (1/eps)_{G_i-G_j} E_j = w^2 E_i
        # So A[i,j] = (k+G_i)^2 * inv_eps_G[i,j]
        
        # Vectorized construction
        # k+G_i for all i
        k_vecs = np.array([(kx + gx, ky + gy) for gx, gy in G_list])
        k2 = np.sum(k_vecs**2, axis=1) # |k+G_i|^2
        
        # A = diag(k2) @ inv_eps_G
        A = np.diag(k2) * inv_eps_G
        
        # Solve Eigenvalue problem
        # A is Hermitian? 
        # (1/eps)_{-G} = conj((1/eps)_G)
        # A[j,i] = (k+G_j)^2 (1/eps)_{G_j-G_i}
        # A[i,j]^* = (k+G_i)^2 (1/eps)_{G_i-G_j}^* = (k+G_i)^2 (1/eps)_{G_j-G_i}
        # They are not equal unless (k+G_i)^2 == (k+G_j)^2, which is false.
        # Wait, the PWE matrix is NOT Hermitian in this form?
        # Actually, the operator is (k+G)^2 / eps.
        # The matrix is M_{ij} = (k+G_i)^2 (1/eps)_{G_i-G_j}.
        # This matrix is NOT Hermitian generally.
        # However, the eigenvalues are real.
        # We should use np.linalg.eigvals or eigvalsh if we transform it.
        # Standard practice: Use np.linalg.eigvals and take real parts, or use a generalized eigenvalue solver if we reformulate.
        # For speed, np.linalg.eigvals is fine for small N.
        
        try:
            eigvals = np.linalg.eigvals(A)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w4 > band_4_max:
                    band_4_max = w4
                if w5 < band_5_min:
                    band_5_min = w5
        except:
            continue
            
    if band_5_min == float('inf'):
        return 0.0
        
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    
    if midgap <= 0:
        return 0.0
        
    return gap / midgap

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    deadline = 220.0
    
    best_cell = None
    best_score = -1.0
    
    # Strategy:
    # 1. Try Air Background (0) with Dielectric Rods (1)
    # 2. Try Dielectric Background (1) with Air Holes (0)
    
    # We will iterate over a grid of parameters for the geometric features.
    # Parameters:
    # - center_r
    # - ring_r, ring_width
    # - n_holes (0, 4, 8), hole_r, hole_dist
    
    # To keep it fast, we will test a coarse grid first, then refine.
    
    # Grid for center_r
    center_rs = [0.0, 0.1, 0.2, 0.3]
    
    # Grid for ring
    ring_rs = [0.0, 0.3, 0.4, 0.5]
    ring_widths = [0.0, 0.1, 0.2]
    
    # Grid for holes
    hole_configs = [
        (0, 0.0, 0.0),
        (4, 0.1, 0.3),
        (4, 0.15, 0.35),
        (4, 0.2, 0.4),
        (8, 0.1, 0.3),
        (8, 0.15, 0.35),
    ]
    
    candidates = []
    
    # Generate candidates for Air Background (Dielectric features)
    for cr in center_rs:
        for rr in ring_rs:
            for rw in ring_widths:
                for nh, hr, hd in hole_configs:
                    # Skip invalid combos (e.g., ring overlaps center too much? 
                    # The boolean OR handles it, but physically it might be messy. 
                    # We'll just let the estimator judge.)
                    mask = _generate_candidate(cr, rr, rw, nh, hr, hd)
                    cell = _make_cell_from_map(mask)
                    candidates.append(cell)
                    
    # Generate candidates for Dielectric Background (Air holes)
    # Invert the mask
    for cell in candidates:
        mask_inv = np.array(cell)
        mask_inv = 1 - mask_inv
        cell_inv = _make_cell_from_map(mask_inv.astype(bool))
        candidates.append(cell_inv)
        
    # Deduplicate candidates to save time
    unique_candidates = []
    seen = set()
    for cell in candidates:
        # Simple hash: tuple of tuples
        key = tuple(tuple(row) for row in cell)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cell)
            
    # Evaluate candidates
    # Shuffle to ensure we don't bias towards first ones if time runs out?
    # No, better to evaluate most promising first. 
    # But we don't know which are promising. 
    # Let's just evaluate all.
    
    # Time check
    start_eval_time = time.time()
    
    for cell in unique_candidates:
        if _elapsed() > deadline - 10:
            break
            
        score = _estimate_band_gap_fast(cell, eps_high)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # If we still have time, refine the best region
    # Find the parameters of the best cell and do a local search
    # This is complex to reverse-engineer parameters from the cell.
    # Instead, if time permits, we can run a finer grid around the best score's parameters.
    
    # For now, the coarse grid should find a good candidate.
    # Let's add a second pass with finer parameters if time > 30s remains.
    
    if _elapsed() < deadline - 30 and best_cell is not None:
        # Refine: Try small perturbations?
        # Or just a finer grid of the same parameters.
        # Let's just do a finer grid of center_r and hole_r.
        
        fine_center_rs = np.linspace(0.1, 0.4, 7)
        fine_hole_rs = np.linspace(0.1, 0.25, 7)
        fine_hole_dists = np.linspace(0.3, 0.45, 7)
        
        # We need to know which configuration type (4 holes, 8 holes, etc.) was best.
        # We can't easily know from the cell. 
        # So we will just evaluate a dense set of specific "types".
        
        # Type 1: Center + 4 Holes
        for cr in fine_center_rs:
            for hr in fine_hole_rs:
                for hd in fine_hole_dists:
                    if _elapsed() > deadline - 10:
                        break
                    mask = _generate_candidate(cr, 0.0, 0.0, 4, hr, hd)
                    cell = _make_cell_from_map(mask)
                    score = _estimate_band_gap_fast(cell, eps_high)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
                        
                    # Also try inverted
                    mask_inv = 1 - mask
                    cell_inv = _make_cell_from_map(mask_inv.astype(bool))
                    score_inv = _estimate_band_gap_fast(cell_inv, eps_high)
                    if score_inv > best_score:
                        best_score = score_inv
                        best_cell = cell_inv
                        
        # Type 2: Center + 8 Holes
        if _elapsed() < deadline - 10:
            for cr in fine_center_rs:
                for hr in fine_hole_rs:
                    for hd in fine_hole_dists:
                        if _elapsed() > deadline - 10:
                            break
                        mask = _generate_candidate(cr, 0.0, 0.0, 8, hr, hd)
                        cell = _make_cell_from_map(mask)
                        score = _estimate_band_gap_fast(cell, eps_high)
                        if score > best_score:
                            best_score = score
                            best_cell = cell
                            
                        mask_inv = 1 - mask
                        cell_inv = _make_cell_from_map(mask_inv.astype(bool))
                        score_inv = _estimate_band_gap_fast(cell_inv, eps_high)
                        if score_inv > best_score:
                            best_score = score_inv
                            best_cell = cell_inv
                            
        # Type 3: Ring + 4 Holes
        if _elapsed() < deadline - 10:
            for rr in [0.3, 0.4]:
                for rw in [0.1, 0.2]:
                    for hr in fine_hole_rs:
                        for hd in fine_hole_dists:
                            if _elapsed() > deadline - 10:
                                break
                            mask = _generate_candidate(0.0, rr, rw, 4, hr, hd)
                            cell = _make_cell_from_map(mask)
                            score = _estimate_band_gap_fast(cell, eps_high)
                            if score > best_score:
                                best_score = score
                                best_cell = cell
                                
                            mask_inv = 1 - mask
                            cell_inv = _make_cell_from_map(mask_inv.astype(bool))
                            score_inv = _estimate_band_gap_fast(cell_inv, eps_high)
                            if score_inv > best_score:
                                best_score = score_inv
                                best_cell = cell_inv

    # Fallback to seed if no cell found
    if best_cell is None:
        best_cell = _make_cell_from_map(_generate_candidate(0.2, 0.0, 0.0, 0, 0.0, 0.0))
        
    return best_cell
# EVOLVE-BLOCK-END
