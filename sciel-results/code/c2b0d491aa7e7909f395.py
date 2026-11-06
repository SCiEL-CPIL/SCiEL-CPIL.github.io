"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_gap_fast(cell, eps_high=12.0, G_max=3, k_points=None):
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
    
    # Generate G vectors
    # Use a diamond/circle cutoff for better accuracy
    g_set = []
    for h in range(-G_max, G_max + 1):
        for k in range(-G_max, G_max + 1):
            # Keep vectors within a radius for isotropy
            if h*h + k*k <= G_max*G_max:
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
    
    # Precompute k-points in the irreducible BZ
    # Irreducible BZ for square lattice: triangle (0,0), (pi,0), (pi,pi)
    # Actually, the standard irreducible BZ for a square lattice is the triangle with vertices (0,0), (pi,0), (pi,pi).
    # Wait, the BZ is [-pi, pi] x [-pi, pi].
    # The irreducible sector is 0 <= kx <= ky and kx + ky <= pi? No.
    # Symmetries: kx <-> ky, kx -> -kx, etc.
    # The irreducible wedge is 0 <= kx <= ky, kx <= pi, ky <= pi.
    # Actually, the high symmetry points are Gamma (0,0), X (pi,0), M (pi,pi).
    # The line from Gamma to X is (k, 0).
    # The line from X to M is (pi, k).
    # The line from M to Gamma is (k, k).
    # So we sample along Gamma-X, X-M, and M-Gamma.
    
    if k_points is None:
        # Sample 5 points along each edge of the irreducible triangle
        k_pts = []
        # Gamma to X: (t*pi, 0) for t in [0, 1]
        for t in np.linspace(0, 1, 5):
            k_pts.append((t * np.pi, 0.0))
        # X to M: (pi, t*pi) for t in [0, 1]
        for t in np.linspace(0, 1, 5):
            k_pts.append((np.pi, t * np.pi))
        # M to Gamma: (t*pi, t*pi) for t in [1, 0]
        for t in np.linspace(1, 0, 5):
            k_pts.append((t * np.pi, t * np.pi))
        # Remove duplicates
        unique_k_pts = []
        for kp in k_pts:
            if kp not in unique_k_pts:
                unique_k_pts.append(kp)
        k_pts = unique_k_pts
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_pts:
        # Construct H matrix for this k
        # H_{ij} = |K_i - K_j|^2 * (1/eps)_{K_i - K_j}
        # Where K_i = k + G_i
        
        K = np.array([(kx + h, ky + k) for h, k in g_set])
        
        # Build H matrix
        H = np.zeros((N_g, N_g), dtype=complex)
        
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
        
        # Solve eigenvalue problem
        # H is Hermitian
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

def _coeffs_to_cell(coeffs, size=32):
    """
    Convert 15 polynomial coefficients to a 32x32 binary cell.
    P(x,y) = sum c_ij x^(2i) y^(2j) for i,j in 0..4 (but we have 15 coeffs, so maybe i,j in 0..2? 3x3=9? No, 15 coeffs implies a specific set).
    Let's use a 4x4 grid of coefficients for i,j in 0..3? 4x4=16. We have 15.
    Let's use i,j in 0..2 (3x3=9) plus some higher terms?
    Actually, the prompt says "sum c_ij x^(2i) y^(2j) where i,j in {0,1,2,3,4}". That's 5x5=25 coeffs.
    But it says "15 coefficients". Maybe a subset?
    Let's assume a 3x5 or 5x3 or a specific triangular set.
    Let's just use the first 15 of a 5x5 grid in row-major order, or a specific pattern.
    To be safe and symmetric, let's use a 3x5 grid: i in 0,1,2 and j in 0..4?
    Or maybe i,j in 0..2 (9 coeffs) and then 6 more?
    Let's define the 15 coefficients as corresponding to (i,j) pairs:
    (0,0), (0,1), (0,2), (0,3), (0,4)
    (1,0), (1,1), (1,2), (1,3), (1,4)
    (2,0), (2,1), (2,2), (2,3), (2,4)
    This is a 3x5 grid.
    
    x, y are normalized to [-0.5, 0.5].
    """
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, size), np.linspace(-0.5, 0.5, size))
    # Normalize to [-1, 1] for better polynomial behavior?
    x = 2 * x
    y = 2 * y
    
    P = np.zeros((size, size))
    idx = 0
    for i in range(3):
        for j in range(5):
            if idx < len(coeffs):
                P += coeffs[idx] * (x ** (2 * i)) * (y ** (2 * j))
                idx += 1
    
    # Threshold at 0
    cell = (P > 0).astype(int)
    return cell

def design(problem):
    """
    Searches for a 4-fold symmetric pattern maximizing the 4-5 TM band gap.
    Uses a polynomial parameterization and Nelder-Mead optimization.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 180  # 180s budget for search, leaving time for safety
    
    # Initial guess for coefficients
    # Try a few different initial points
    initial_guesses = [
        np.zeros(15),
        np.ones(15) * 0.1,
        np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),  # Ring-like
        np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]),  # Cross-like
    ]
    
    best_score = -1.0
    best_cell = None
    best_coeffs = None
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        # Define objective function for minimization (we want to maximize gap)
        def objective(coeffs):
            cell = _coeffs_to_cell(coeffs, size)
            score = _compute_gap_fast(cell)
            return -score  # Negate for minimization
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3})
            
            if time.time() > deadline:
                break
                
            final_coeffs = res.x
            final_cell = _coeffs_to_cell(final_coeffs, size)
            final_score = _compute_gap_fast(final_cell)
            
            if final_score > best_score:
                best_score = final_score
                best_cell = final_cell
                best_coeffs = final_coeffs
                
        except Exception as e:
            pass
    
    # If no good candidate found, fallback to a simple ring
    if best_cell is None:
        cell = np.zeros((size, size), dtype=int)
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x - size/2.0
        y = y - size/2.0
        r = np.sqrt(x**2 + y**2)
        cell[(r >= 5) & (r <= 10)] = 1
        return cell.tolist()
        
    return best_cell
# EVOLVE-BLOCK-END
