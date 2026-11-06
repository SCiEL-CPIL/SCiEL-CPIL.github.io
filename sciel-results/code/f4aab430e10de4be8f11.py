"""Photonic crystal design with TM band gap optimization."""
import numpy as np
import time
import math

def _enforce_c4v_symmetry(cell):
    """Ensure the cell has C4v symmetry by averaging over the symmetry group."""
    size = len(cell)
    result = np.array(cell, dtype=float)
    
    # C4v group: identity, 90°, 180°, 270° rotation, and 4 reflections
    # For a pixel grid centered at (size/2, size/2):
    # Rotation 90°: (i,j) -> (j, size-1-i)
    # Rotation 180°: (i,j) -> (size-1-i, size-1-j)  
    # Rotation 270°: (i,j) -> (size-1-j, i)
    # Reflection x: (i,j) -> (size-1-i, j)
    # Reflection y: (i,j) -> (i, size-1-j)
    # Reflection xy: (i,j) -> (size-1-i, size-1-j)
    # Reflection x=y: (i,j) -> (j, i)
    
    # Average over all 8 symmetry operations
    sym_ops = [
        lambda i, j: (i, j),
        lambda i, j: (j, size-1-i),
        lambda i, j: (size-1-i, size-1-j),
        lambda i, j: (size-1-j, i),
        lambda i, j: (size-1-i, j),
        lambda i, j: (i, size-1-j),
        lambda i, j: (size-1-j, size-1-i),
        lambda i, j: (j, i),
    ]
    
    averaged = np.zeros_like(result)
    for op in sym_ops:
        for i in range(size):
            for j in range(size):
                si, sj = op(i, j)
                averaged[i, j] += result[si, sj]
    averaged /= len(sym_ops)
    
    # Threshold at 0.5
    result_sym = (averaged >= 0.5).astype(int)
    
    return result_sym.tolist()

def _fourier_coeffs_from_cell(cell, N):
    """Compute Fourier coefficients of 1/eps from a pixel cell."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    coeffs = np.zeros((2*N-1, 2*N-1), dtype=complex)
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    m_vals = np.arange(-(N-1), N)
    n_vals = np.arange(-(N-1), N)
    
    for m_idx, m in enumerate(m_vals):
        for n_idx, n in enumerate(n_vals):
            phase = np.exp(-2j * np.pi * (m * X + n * Y))
            coeffs[m_idx, n_idx] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _tm_bands_at_k(cell, kx, ky, N, n_bands=6):
    """Solve TM band structure at wave vector (kx, ky)."""
    size = len(cell)
    coeffs = _fourier_coeffs_from_cell(cell, N)
    
    m_vals = np.arange(-(N-1), N)
    Gx = 2 * np.pi * m_vals
    Gy = 2 * np.pi * m_vals
    
    n_pw = len(m_vals) ** 2
    pw_i = np.repeat(np.arange(len(m_vals)), len(m_vals))
    pw_j = np.tile(np.arange(len(m_vals)), len(m_vals))
    
    kG2 = np.zeros(n_pw)
    for idx in range(n_pw):
        kx_t = kx + Gx[pw_i[idx]]
        ky_t = ky + Gy[pw_j[idx]]
        kG2[idx] = kx_t**2 + ky_t**2
    
    A = np.zeros((n_pw, n_pw), dtype=complex)
    for idx1 in range(n_pw):
        i1, j1 = pw_i[idx1], pw_j[idx1]
        di = i1 - pw_i
        dj = j1 - pw_j
        ci = di + (N - 1)
        cj = dj + (N - 1)
        mask = (ci >= 0) & (ci < 2*N-1) & (cj >= 0) & (cj < 2*N-1)
        row_coeffs = np.zeros(n_pw, dtype=complex)
        row_coeffs[mask] = coeffs[ci[mask], cj[mask]]
        A[idx1, :] = row_coeffs * kG2[idx1]
    
    try:
        vals, _ = np.linalg.eigh(A)
        return vals[:n_bands].real
    except:
        return np.zeros(n_bands)

def _irreducible_bz_points():
    """Generate wave vectors in the irreducible Brillouin zone."""
    points = []
    n = 8
    for i in range(n):
        for j in range(n):
            kx = (i / (n - 1)) * np.pi
            ky = (j / (n - 1)) * np.pi
            if kx <= ky + 1e-10 and kx + ky <= np.pi + 1e-10:
                points.append((kx, ky))
    
    # Add high-symmetry points
    for p in [(0, 0), (np.pi, 0), (np.pi/2, np.pi/2)]:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in points):
            points.append(p)
    return points

def _evaluate_gap(cell, k_points=None, N=9):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    if k_points is None:
        k_points = _irreducible_bz_points()
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        try:
            vals = _tm_bands_at_k(cell, kx, ky, N, n_bands=6)
            if len(vals) >= 5:
                omega4 = vals[3]
                omega5 = vals[4]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 <= max_omega4:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _param_to_cell(params, size=32):
    """Convert C4v-symmetric Fourier parameters to a pixel cell.
    
    params: array of coefficients for cos(2πmx)cos(2πny) terms where m >= n >= 0
    The function is: f(x,y) = sum_{m>=n>=0} a_{mn} cos(2πmx)cos(2πny)
    We map f to [0,1] and threshold at 0.5 to get 0/1 pixels.
    """
    # Define the set of (m,n) pairs to use
    # For efficiency, use low-order harmonics
    harmonics = []
    max_order = 4
    for m in range(max_order + 1):
        for n in range(m + 1):
            harmonics.append((m, n))
    
    n_params = len(harmonics)
    if len(params) < n_params:
        params = np.zeros(n_params)
        params[:len(params)] = np.array(params[:n_params]) if hasattr(params, '__len__') else 0
    
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    f = np.zeros((size, size))
    for idx, (m, n) in enumerate(harmonics):
        a = params[idx]
        f += a * np.cos(2 * np.pi * m * X) * np.cos(2 * np.pi * n * Y)
    
    # Normalize f to [0, 1]
    f_min, f_max = f.min(), f.max()
    if f_max - f_min > 1e-10:
        f_norm = (f - f_min) / (f_max - f_min)
    else:
        f_norm = np.zeros_like(f)
    
    # Threshold at 0.5
    cell = (f_norm >= 0.5).astype(int)
    
    return cell.tolist()

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200
    
    # Use N=9 to match the evaluator's 289 plane waves (17x17)
    N_eval = 9
    
    # Define harmonics for the parametric representation
    harmonics = []
    max_order = 5
    for m in range(max_order + 1):
        for n in range(m + 1):
            harmonics.append((m, n))
    n_params = len(harmonics)
    
    # Initial guess: a disk-like structure
    # Use a combination of low-order harmonics
    params = np.zeros(n_params)
    params[0] = 1.0  # DC term
    params[1] = -0.5  # cos(2πx)
    params[2] = -0.3  # cos(2πy)
    params[3] = 0.2   # cos(2πx)cos(2πy)
    
    cell = _param_to_cell(params, size)
    cell = _enforce_c4v_symmetry(cell)
    
    # Evaluate initial gap
    k_points = _irreducible_bz_points()
    best_gap = _evaluate_gap(cell, k_points, N_eval)
    best_cell = [row[:] for row in cell]
    best_params = params.copy()
    
    # Gradient-free optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 30:
            return 1e6
        cell = _param_to_cell(params, size)
        cell = _enforce_c4v_symmetry(cell)
        # Use fewer k-points for speed during optimization
        k_sub = k_points[:5]
        return -_evaluate_gap(cell, k_sub, N_eval)
    
    # Use Nelder-Mead for gradient-free optimization
    try:
        result = minimize(objective, best_params, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if result.success or result.fun < -best_gap:
            best_params = result.x
            cell = _param_to_cell(best_params, size)
            cell = _enforce_c4v_symmetry(cell)
            best_gap = -result.fun
            best_cell = [row[:] for row in cell]
    except:
        pass
    
    # If we have time, do a few local refinements
    if time.time() < deadline - 60:
        # Try some random perturbations of the best params
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            perturbed = best_params + np.random.randn(n_params) * 0.1
            cell = _param_to_cell(perturbed, size)
            cell = _enforce_c4v_symmetry(cell)
            gap = _evaluate_gap(cell, k_points[:5], N_eval)
            if gap > best_gap:
                best_gap = gap
                best_cell = [row[:] for row in cell]
                best_params = perturbed.copy()
    
    # Final symmetry enforcement
    final_cell = _enforce_c4v_symmetry(best_cell)
    
    return final_cell
