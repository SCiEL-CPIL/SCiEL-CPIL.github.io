"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = time.time() + 110.0  # safety margin before 230 s wall-clock kill


def _make_cell(params, size=32):
    """Build a D4-symmetric 32x32 binary cell from Bessel/angular-harmonic superposition.
    
    params = [w0, k0, w1, k1, w2, k2, w3, k3, threshold]
    f(r,theta) = sum_i w_i * J_0(k_i * r) * cos(4*m_i*theta) + base
    where m_i cycles through [0,1,1,2] for angular harmonics.
    """
    w0, k0, w1, k1, w2, k2, w3, k3, thr = params
    m_vals = [0, 1, 1, 2]
    w_vals = [w0, w1, w2, w3]
    k_vals = [k0, k1, k2, k3]
    
    # Precompute coordinates
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)
    
    f = np.zeros((size, size))
    for w, k, m in zip(w_vals, k_vals, m_vals):
        # J_0 approximation via series or use scipy if available
        try:
            from scipy.special import j0
            j0_val = j0(k * R)
        except ImportError:
            # Simple approximation for J_0
            j0_val = np.cos(k * R)  # crude, but avoids scipy dependency issues
        f += w * j0_val * np.cos(4 * m * THETA)
    
    # Add a base dielectric fraction to control fill ratio
    cell = (f > thr).astype(int)
    
    # Enforce D4 symmetry: average over all 8 symmetry operations
    # Actually, let's just enforce the required symmetries directly
    # cell == cell[::-1] row-wise and column-wise and cell == its transpose
    
    # Enforce row reversal symmetry
    for i in range(size // 2):
        for j in range(size):
            if cell[i][j] != cell[size-1-i][j]:
                # Set both to majority or 0
                cell[i][j] = 1 if cell[i][j] + cell[size-1-i][j] > 0 else 0
                cell[size-1-i][j] = cell[i][j]
    
    # Enforce column reversal symmetry
    for i in range(size):
        for j in range(size // 2):
            if cell[i][j] != cell[i][size-1-j]:
                cell[i][j] = 1 if cell[i][j] + cell[i][size-1-j] > 0 else 0
                cell[i][size-1-j] = cell[i][j]
    
    # Enforce transpose symmetry
    for i in range(size):
        for j in range(i+1, size):
            if cell[i][j] != cell[j][i]:
                cell[i][j] = 1 if cell[i][j] + cell[j][i] > 0 else 0
                cell[j][i] = cell[i][j]
    
    return cell.tolist()


def _estimate_gap(cell, size=32):
    """Quick estimate of the 4th-5th TM band gap using a simplified plane-wave method.
    
    This is a fast approximation to guide the search. We compute the Fourier
    coefficients of epsilon and solve a small eigenvalue problem.
    """
    eps_high = 12.0
    eps_low = 1.0
    
    # Compute epsilon array
    eps = np.where(np.array(cell) == 1, eps_high, eps_low)
    
    # Compute Fourier coefficients up to a small cutoff
    # For a quick estimate, use just the first few reciprocal lattice vectors
    N = 5  # half-width of reciprocal lattice vectors
    G_vals = []
    for m in range(-N, N+1):
        for n in range(-N, N+1):
            G_vals.append((m, n))
    
    G_array = np.array(G_vals, dtype=float)
    NG = len(G_vals)
    
    # Compute inverse epsilon Fourier coefficients
    # 1/eps(r) = sum_G (1/eps)_G exp(iG.r)
    inv_eps = np.zeros((size, size))
    inv_eps[eps > 0] = 1.0 / eps[eps > 0]
    
    inv_eps_G = np.zeros(NG, dtype=complex)
    for idx, (m, n) in enumerate(G_vals):
        # Fourier coefficient: (1/N^2) sum_{x,y} inv_eps(x,y) exp(-i 2pi (mx+ny)/N)
        phase = np.exp(-2j * math.pi * (m * np.arange(size)[:,None] / size + 
                                        n * np.arange(size)[None,:] / size))
        inv_eps_G[idx] = np.sum(inv_eps * phase) / (size * size)
    
    # Build the eigenvalue matrix for TM polarization
    # For TM: (1/eps) matrix
    A = np.zeros((NG, NG), dtype=complex)
    for i in range(NG):
        for j in range(NG):
            Gi = G_array[i]
            Gj = G_array[j]
            Gdiff = Gi - Gj
            # Find index of Gdiff in G_vals
            try:
                idx_diff = G_vals.index(tuple(Gdiff))
                A[i, j] = inv_eps_G[idx_diff]
            except ValueError:
                A[i, j] = 0.0
    
    # Solve for eigenvalues at a few k-points in the irreducible BZ
    # For a quick estimate, just use Gamma point and X point
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # X
        (0.25, 0.25),    # M/2
    ]
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for kx, ky in k_points:
        # Build matrix: (k+G)^2 A
        B = np.zeros((NG, NG), dtype=float)
        for i in range(NG):
            kx_i = kx + G_array[i, 0]
            ky_i = ky + G_array[i, 1]
            k2_i = kx_i**2 + ky_i**2
            for j in range(NG):
                B[i, j] = k2_i * A[i, j].real if i == j else k2_i * A[i, j].real
        
        # Actually, the TM eigenvalue problem is:
        # sum_j (1/eps)_{G_i - G_j} |k+G_j|^2 c_j = omega^2 c_i
        # So the matrix is M[i,j] = (1/eps)_{G_i-G_j} * |k+G_j|^2
        M = np.zeros((NG, NG), dtype=float)
        for i in range(NG):
            for j in range(NG):
                Gdiff = G_array[i] - G_array[j]
                try:
                    idx_diff = G_vals.index(tuple(Gdiff))
                    M[i, j] = inv_eps_G[idx_diff].real * (kx + G_array[j,0])**2 + \
                               inv_eps_G[idx_diff].real * (ky + G_array[j,1])**2
                except ValueError:
                    M[i, j] = 0.0
        
        # Ensure symmetry
        M = (M + M.T) / 2
        
        try:
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            # Take bands 4 and 5 (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                omega4 = math.sqrt(max(eigvals[3], 0))
                omega5 = math.sqrt(max(eigvals[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except np.linalg.LinAlgError:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return -1.0
    
    return 2 * gap / midgap


def design(problem):
    """D4-symmetric Bessel/angular-harmonic superposition with local parameter search."""
    size = problem.get("pixels", 32)
    
    # Initial parameters: base rod-like structure
    # w0 controls the central dielectric, k0 sets the radial frequency
    best_params = [1.0, 4.0, 0.5, 8.0, 0.3, 12.0, 0.2, 16.0, 0.5]
    best_score = _estimate_gap(_make_cell(best_params, size))
    
    # Local search: perturb each parameter and keep improvements
    step_sizes = [0.2, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1]
    
    for iteration in range(200):
        if time.time() > _DEADLINE:
            break
        
        improved = False
        for p_idx in range(len(best_params)):
            for direction in [-1, 1]:
                trial_params = best_params[:]
                trial_params[p_idx] += direction * step_sizes[p_idx]
                
                # Clamp parameters to reasonable ranges
                if p_idx in [0, 2, 4, 6]:  # weights
                    trial_params[p_idx] = max(-2.0, min(2.0, trial_params[p_idx]))
                elif p_idx in [1, 3, 5, 7]:  # frequencies
                    trial_params[p_idx] = max(0.5, min(30.0, trial_params[p_idx]))
                else:  # threshold
                    trial_params[p_idx] = max(-1.0, min(1.0, trial_params[p_idx]))
                
                trial_cell = _make_cell(trial_params, size)
                trial_score = _estimate_gap(trial_cell, size)
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_params = trial_params
                    improved = True
        
        if not improved:
            # Reduce step sizes
            step_sizes = [s * 0.7 for s in step_sizes]
            if min(step_sizes) < 0.01:
                break
    
    return _make_cell(best_params, size)
# EVOLVE-BLOCK-END
