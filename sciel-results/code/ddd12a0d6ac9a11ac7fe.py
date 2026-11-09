"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _compute_eps_inv_fourier(cell, size, Npw):
    """Compute the inverse-epsilon Fourier matrix using FFT for speed."""
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / 12.0
    eps_inv[eps <= 0.5] = 1.0
    
    half = (Npw - 1) // 2
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Use FFT to compute Fourier coefficients of eps_inv
    # Pad to size x size if needed, but size is 32 which is power of 2
    eps_inv_fft = np.fft.fft2(eps_inv)
    
    eps_F = np.zeros((N, N), dtype=complex)
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            # Map to FFT indices
            fx = dgx % size
            fy = dgy % size
            eps_F[idx_g][idx_h] = eps_inv_fft[fy, fx] / (size * size)
    
    return eps_F

def _quick_score(cell, size, eps_high, Npw=17):
    """Fast approximate score using reduced plane-wave basis for local optimization."""
    half = (Npw - 1) // 2
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    eps_F = _compute_eps_inv_fourier(cell, size, Npw)
    
    # Define wave vectors in irreducible BZ
    # Square lattice: BZ is [-pi, pi] x [-pi, pi], irreducible: 0 <= ky <= kx, kx <= pi
    k_points = []
    # Gamma point
    k_points.append((0.0, 0.0))
    # X point
    k_points.append((1.0, 0.0))
    # M point
    k_points.append((1.0, 1.0))
    # Midpoint Gamma-X
    k_points.append((0.5, 0.0))
    # Midpoint Gamma-M
    k_points.append((0.5, 0.5))
    # Midpoint X-M
    k_points.append((1.0, 0.5))
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        Kx = 2 * np.pi * kx
        Ky = 2 * np.pi * ky
        
        A = np.zeros((N, N), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h, (hx, hy) in enumerate(G_vectors):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            # Take the 5th and 4th eigenvalues (0-indexed: index 3 and 4)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    return -1.0

def design(problem):
    """Local search optimization for photonic crystal unit cell."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin before 120s limit
    
    # Initial design: circle with radius ~0.2
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i][j] = 1
    
    # Get unique symmetric positions
    unique_positions = []
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            unique_positions.append((i, j))
    
    best_cell = cell.copy()
    best_score = _quick_score(best_cell, size, 12, Npw=17)
    
    # Local search: flip individual symmetric pixels
    random.seed(42)
    pos_idx = list(range(len(unique_positions)))
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Pick a random unique position
        idx = random.choice(pos_idx)
        i, j = unique_positions[idx]
        
        # Try flipping this pixel
        old_val = cell[i][j]
        cell[i][j] = 1 - old_val
        
        # Enforce symmetry
        cell = _enforce_symmetry(cell, size)
        
        # Score
        score = _quick_score(cell, size, 12, Npw=17)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            # Revert
            cell[i][j] = old_val
            cell = _enforce_symmetry(cell, size)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Convert back to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
