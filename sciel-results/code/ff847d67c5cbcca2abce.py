"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_cell(size, radius, shape_param=1.0):
    """Build a symmetric 32x32 cell with a centered dielectric rod.
    shape_param > 1 makes it more square-like, < 1 more circular."""
    cell = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Superellipse-like shape: |x|^p + |y|^p < r^p
            p = 2.0 + shape_param  # shape_param 0 -> circle, higher -> more square
            if abs(x)**p + abs(y)**p < radius**p:
                cell[i, j] = 1
    return cell

def _pwe_bands(cell, n_pw=17, n_bands=6, k_points=None):
    """Solve TM band structure using plane-wave expansion.
    Returns array of shape (n_k, n_bands) with frequencies in units of a/lambda.
    """
    size = cell.shape[0]
    eps = 1.0 + 11.0 * cell.astype(float)
    
    # Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    # Compute Fourier coefficients up to n_pw
    # G = 2*pi*(m, n) for |m|,|n| <= n_pw
    gs = []
    for m in range(-n_pw, n_pw + 1):
        for n in range(-n_pw, n_pw + 1):
            gs.append((m, n))
    n_g = len(gs)
    
    # Fourier coefficients of inv_eps
    fourier_inv_eps = np.zeros(n_g, dtype=complex)
    for idx, (m, n) in enumerate(gs):
        # F[m,n] = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-2*pi*i*(m*i/N + n*j/N))
        phase = np.exp(-2j * np.pi * (m * np.arange(size) / size + n * np.arange(size) / size))
        fourier_inv_eps[idx] = np.sum(inv_eps * np.outer(phase, phase)) / (size * size)
    
    # Build G matrix: G[k] = (k + G_i) . (k + G_j) * inv_eps[G_i - G_j]
    if k_points is None:
        # Use a few key k-points in irreducible BZ
        # For square lattice: G=(0,0), X=(0.5,0), M=(0.5,0.5)
        k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.0), (0.25, 0.25)]
    
    bands = []
    for kx, ky in k_points:
        k = np.array([kx, ky]) * 2 * np.pi  # in units of 2pi/a
        
        # Build eigenvalue matrix
        A = np.zeros((n_g, n_g), dtype=complex)
        for i, (mi, ni) in enumerate(gs):
            Gi = np.array([mi, ni]) * 2 * np.pi
            ki_Gi = k + Gi
            for j, (mj, nj) in enumerate(gs):
                Gj = np.array([mj, nj]) * 2 * np.pi
                kj_Gj = k + Gj
                # Find index of G_i - G_j
                diff_m = mi - mj
                diff_n = ni - nj
                # Find index in gs list
                diff_idx = None
                for idx2, (dm, dn) in enumerate(gs):
                    if dm == diff_m and dn == diff_n:
                        diff_idx = idx2
                        break
                if diff_idx is not None:
                    A[i, j] = np.dot(ki_Gi, kj_Gj) * fourier_inv_eps[diff_idx]
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(np.real(eigvals))
            # Take first n_bands
            bands.append(eigvals[:n_bands])
        except:
            bands.append(np.zeros(n_bands))
    
    return np.array(bands)

def _evaluate_gap(bands):
    """Compute gap-midgap ratio between band 4 and band 5 (0-indexed: 3 and 4)."""
    # bands shape: (n_k, n_bands)
    # max of band 4 (index 3) over all k-points
    max_band4 = np.max(bands[:, 3])
    # min of band 5 (index 4) over all k-points
    min_band5 = np.min(bands[:, 4])
    
    if min_band5 + max_band4 < 1e-10:
        return -1.0
    
    gap_ratio = 2 * (min_band5 - max_band4) / (min_band5 + max_band4)
    return gap_ratio

def design(problem):
    """Search for optimal rod geometry using internal PWE solver."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 120.0  # 120s budget with margin
    
    best_score = -np.inf
    best_cell = None
    
    # Try a grid of radius and shape parameters
    radii = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    shapes = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Start with seed value as fallback
    seed_cell = _build_cell(size, 0.2, 0.0)
    best_cell = seed_cell.tolist()
    
    for radius in radii:
        for shape in shapes:
            if time.time() > deadline - 30.0:
                break
            
            cell = _build_cell(size, radius, shape)
            bands = _pwe_bands(cell, n_pw=10, n_bands=6)
            score = _evaluate_gap(bands)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
    
    return best_cell
# EVOLVE-BLOCK-END
