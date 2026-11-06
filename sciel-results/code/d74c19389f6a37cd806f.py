"""Design a symmetric 32x32 photonic-crystal unit cell to maximize the TM band-4/5 gap."""
import time
import numpy as np
from scipy.optimize import minimize

# Global deadline guard
_DEADLINE = time.time() + 210.0  # 210s safety margin within 230s limit


def _enforce_symmetry(cell):
    """Enforce C4v symmetry: 180° rotation and transpose invariance."""
    cell = np.array(cell, dtype=np.float64)
    size = cell.shape[0]
    # 180° rotation invariance: cell[i,j] == cell[size-1-i, size-1-j]
    rot180 = cell[::-1, ::-1]
    cell = 0.5 * (cell + rot180)
    # Transpose invariance: cell[i,j] == cell[j,i]
    cell = 0.5 * (cell + cell.T)
    return cell


def _compute_fourier_coefficients(cell, n_pw=17):
    """
    Compute exact Fourier coefficients of the dielectric function.
    For TM polarization, we need 1/epsilon in the plane-wave method.
    
    cell: 32x32 array with values in [0,1] (fraction of dielectric)
    Returns: array of shape (n_pw, n_pw) with Fourier coefficients of 1/eps.
    """
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    
    # Interpolate to get continuous dielectric function
    # Use bilinear interpolation on the pixel grid
    x = np.linspace(0, 1, size + 1)
    # Cell values represent pixel centers
    cell_float = cell.astype(np.float64)
    eps = eps_low + (eps_high - eps_low) * cell_float
    
    # Compute 1/eps
    inv_eps = 1.0 / eps
    
    # Fourier coefficients: c_{m,n} = integral of f(x,y) * exp(-i 2pi (m x + n y)) dx dy
    # For a 32x32 pixel map, we can compute this numerically
    # Using the pixel values as a Riemann sum approximation
    dx = 1.0 / size
    dy = 1.0 / size
    
    # Create coordinate arrays
    i_idx = np.arange(size)
    j_idx = np.arange(size)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    xi = (I + 0.5) / size
    yi = (J + 0.5) / size
    
    n_half = n_pw // 2  # 8
    m_vals = np.arange(-n_half, n_half + 1)  # -8 to 8
    n_vals = np.arange(-n_half, n_half + 1)
    
    coeffs = np.zeros((n_pw, n_pw), dtype=np.complex128)
    for mi, m in enumerate(m_vals):
        for ni, n in enumerate(n_vals):
            # exp(-i 2pi (m*x + n*y))
            phase = np.exp(-1j * 2 * np.pi * (m * xi + n * yi))
            coeffs[mi, ni] = np.sum(inv_eps * phase) * dx * dy
    
    return coeffs


def _solve_tm_bands(coeffs, kx, ky, n_pw=17):
    """
    Solve the TM band structure at wave vector (kx, ky) using plane-wave method.
    Returns the first 6 eigenfrequencies (in units of 2pi/a).
    
    For TM: (k + G)^2 * c_{G} = omega^2 * sum_{G'} c_{G-G'} * c_{G'}
    where c_G are Fourier coefficients of 1/eps.
    """
    n_half = n_pw // 2
    G_vals = np.arange(-n_half, n_half + 1)
    
    # Build G vector list
    G_vectors = []
    for m in G_vals:
        for n in G_vals:
            G_vectors.append((m, n))
    G_vectors = np.array(G_vectors, dtype=np.float64)
    n_G = len(G_vectors)
    
    # k vector in units of 2pi/a
    k = np.array([kx, ky], dtype=np.float64)
    
    # (k + G)^2 for each G
    kg = G_vectors + k
    kg_sq = np.sum(kg ** 2, axis=1)
    
    # Build the matrix A where A[i,j] = c_{G_i - G_j} * (k + G_i)^2
    # Actually for TM: sum_j (k+G_i)^2 * delta_{ij} - omega^2 * sum_j c_{G_i-G_j} = 0
    # This is a generalized eigenvalue problem: A c = omega^2 B c
    # where A_{ij} = (k+G_i)^2 delta_{ij}, B_{ij} = c_{G_i - G_j}
    
    # Better formulation: 
    # omega^2 = (k+G)^2 / <1/eps> type thing, but more precisely:
    # The eigenvalue equation is: sum_{G'} (k+G)^2 delta_{G,G'} c_{G-G'} a_{G'} = omega^2 a_G
    # i.e., M a = omega^2 a where M_{G,G'} = (k+G)^2 * delta_{G,G'} * c_{G-G'}... 
    # 
    # Actually the standard PW method for TM:
    # (k+G)^2 a_G = omega^2 sum_{G'} c_{G-G'} a_{G'}
    # This is: sum_{G'} [(k+G)^2 delta_{G,G'} - omega^2 c_{G-G'}] a_{G'} = 0
    # Generalized eigenvalue: A a = omega^2 B a
    # A = diag((k+G)^2), B_{G,G'} = c_{G-G'}
    
    # Build B matrix
    B = np.zeros((n_G, n_G), dtype=np.complex128)
    for i in range(n_G):
        for j in range(n_G):
            dG = G_vectors[i] - G_vectors[j]
            mi = int(dG[0]) + n_half
            ni = int(dG[1]) + n_half
            if 0 <= mi < n_pw and 0 <= ni < n_pw:
                B[i, j] = coeffs[mi, ni]
            else:
                B[i, j] = 0.0
    
    # A is diagonal
    A = np.diag(kg_sq)
    
    # Solve generalized eigenvalue problem A a = omega^2 B a
    # => B^{-1} A a = omega^2 a
    # Or: A a - omega^2 B a = 0
    # Use scipy to solve
    
    try:
        from scipy.linalg import eigh
        # Convert to standard eigenvalue problem
        # B^{-1} A a = omega^2 a
        B_inv_A = np.linalg.solve(B, A)
        # Symmetrize for numerical stability
        B_inv_A = 0.5 * (B_inv_A + B_inv_A.T.conj())
        eigenvalues, _ = eigh(B_inv_A)
        eigenvalues = np.real(eigenvalues)
        # Sort and take first 6
        eigenvalues = np.sort(eigenvalues)
        return eigenvalues[:6]
    except Exception:
        return np.array([np.inf] * 6)


def _compute_score(cell, n_k=30, n_pw=17):
    """
    Compute the gap-midgap ratio for TM bands 4 and 5.
    Returns the score (higher is better).
    """
    coeffs = _compute_fourier_coefficients(cell, n_pw)
    
    # Irreducible Brillouin zone for square lattice
    # BZ: -pi < kx,ky < pi (in units of 2pi/a, so -0.5 < kx,ky < 0.5)
    # Irreducible: 0 <= ky <= kx <= 0.5 (for the triangle)
    # Actually for square lattice C4v symmetry, irreducible BZ is:
    # 0 <= ky <= kx, with kx from 0 to 0.5, and also the line kx=0.5, ky from 0 to 0.5
    # Let's use a set of 30 wave vectors covering the irreducible BZ
    
    k_points = []
    # Along Gamma-X: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0.0))
    # Along X-M: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Along M-Gamma: (0.5-t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5 - t, t))
    # Some interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.3, 0.15))
    k_points.append((0.4, 0.2))
    k_points.append((0.15, 0.05))
    
    # Take first 30
    k_points = k_points[:30]
    
    max_omega4 = -np.inf
    min_omega5 = np.inf
    
    for kx, ky in k_points:
        omegas = _solve_tm_bands(coeffs, kx, ky, n_pw)
        if len(omegas) >= 5:
            max_omega4 = max(max_omega4, omegas[3])  # 4th band (0-indexed: 3)
            min_omega5 = min(min_omega5, omegas[4])  # 5th band (0-indexed: 4)
    
    if max_omega4 <= 0 or min_omega5 <= 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap


def _initial_design_square(size=32, frac=0.35):
    """Create an initial square pillar design."""
    cell = np.zeros((size, size), dtype=np.float64)
    # Square pillar centered
    half = int(frac * size / 2)
    start = size // 2 - half
    end = size // 2 + half
    cell[start:end, start:end] = 1.0
    return _enforce_symmetry(cell)


def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell to maximize the TM band-4/5 gap.
    Uses a square pillar initial guess with local optimization.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Start with a square pillar design
    best_cell = _initial_design_square(size, frac=0.35)
    best_score = -np.inf
    
    # Try a few different initial designs
    initial_fracs = [0.25, 0.30, 0.35, 0.40, 0.45]
    
    for frac in initial_fracs:
        if time.time() > _DEADLINE - 30:
            break
        
        cell = _initial_design_square(size, frac=frac)
        
        # Evaluate this design
        try:
            score = _compute_score(cell, n_k=30, n_pw=17)
        except Exception:
            score = -1.0
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Try to improve with a simple local search on the square size
    if time.time() < _DEADLINE - 60:
        # Binary search / grid search on the square fraction
        for frac in np.arange(0.20, 0.55, 0.025):
            if time.time() > _DEADLINE - 30:
                break
            
            cell = _initial_design_square(size, frac=frac)
            try:
                score = _compute_score(cell, n_k=30, n_pw=17)
            except Exception:
                score = -1.0
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Try a ring/square-ring design (hollow square)
    if time.time() < _DEADLINE - 40:
        for outer_frac in [0.45, 0.50, 0.55]:
            for inner_frac in [0.15, 0.20, 0.25, 0.30]:
                if time.time() > _DEADLINE - 30:
                    break
                
                cell = np.zeros((size, size), dtype=np.float64)
                outer_half = int(outer_frac * size / 2)
                inner_half = int(inner_frac * size / 2)
                o_start = size // 2 - outer_half
                o_end = size // 2 + outer_half
                i_start = size // 2 - inner_half
                i_end = size // 2 + inner_half
                
                cell[o_start:o_end, o_start:o_end] = 1.0
                cell[i_start:i_end, i_start:i_end] = 0.0
                
                cell = _enforce_symmetry(cell)
                
                try:
                    score = _compute_score(cell, n_k=30, n_pw=17)
                except Exception:
                    score = -1.0
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
    
    # Convert to 0/1 integer list
    best_cell_binary = (best_cell > 0.5).astype(int)
    
    # Ensure exact symmetry
    best_cell_binary = _enforce_symmetry(best_cell_binary).astype(int)
    best_cell_binary = (best_cell_binary > 0.5).astype(int)
    
    # Return as nested list
    return best_cell_binary.tolist()
