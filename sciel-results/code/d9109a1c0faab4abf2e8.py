import numpy as np
import time
import math

_DEADLINE = 225.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

# ---- fast plane-wave TM surrogate (vectorized) ------------------------------
def _build_pw_evaluator(size, eps_low, eps_high, pw_half=2):
    """
    Precompute everything needed for fast TM band-structure evaluation.
    Uses (2*pw_half+1)^2 plane waves.
    """
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1)
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=np.float64)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=np.float64)
    
    # G difference indices for FFT lookup
    # We compute these on the fly in the evaluator to save memory if N_pw is small,
    # but for 5x5=25 or 7x7=49 it's fine.
    # Actually, let's precompute the index grid for speed.
    G_diff_idx_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_idx_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_idx_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diff_idx_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size
            
    # K-points in irreducible BZ
    # Irreducible BZ for square lattice: 0 <= ky <= kx, kx + ky <= 0.5
    # Key points: Gamma(0,0), M(0.5,0), X(0.5,0.5)
    # We sample a few points along the edges and interior.
    k_points = []
    k_points.append((0.0, 0.0)) # Gamma
    k_points.append((0.5, 0.0)) # M
    k_points.append((0.5, 0.5)) # X
    # Along Gamma-M
    k_points.append((0.25, 0.0))
    k_points.append((0.125, 0.0))
    # Along M-X
    k_points.append((0.5, 0.25))
    k_points.append((0.5, 0.125))
    # Along Gamma-X
    k_points.append((0.25, 0.25))
    k_points.append((0.125, 0.125))
    # Interior
    k_points.append((0.3, 0.1))
    k_points.append((0.4, 0.2))
    k_points.append((0.45, 0.05))
    
    k_points = np.array(k_points, dtype=np.float64)
    
    return Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw

def _evaluate_gap_fast(cell_grid, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low=1.0, eps_high=12.0):
    """
    Fast evaluation of the gap-midgap ratio between bands 4 and 5.
    cell_grid: 32x32 array of 0/1
    """
    size = cell_grid.shape[0]
    eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
    inv_eps = 1.0 / eps_grid
    
    # FFT of 1/epsilon
    fft_1eps = np.fft.fft2(inv_eps)
    # coeffs[i,j] = (1/eps)_{G_i - G_j}
    coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        Kx = Gx_arr + kx
        Ky = Gy_arr + ky
        
        # H[i,j] = (K_i dot K_j) * coeffs_{G_i - G_j}
        dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
        H = dots * coeffs
        H_real = H.real
        # Symmetrize for numerical stability
        H_real = (H_real + H_real.T) * 0.5
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(H_real)
        
        if len(eigenvalues) >= 5:
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            
            if w4_sq > 0:
                w4 = math.sqrt(w4_sq)
                if w4 > max_omega_4:
                    max_omega_4 = w4
            if w5_sq > 0:
                w5 = math.sqrt(w5_sq)
                if w5 < min_omega_5:
                    min_omega_5 = w5
    
    if min_omega_5 == float('inf') or max_omega_4 <= 0:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) * 0.5
    
    if midgap <= 0:
        return -1.0
        
    return 2.0 * gap / midgap

def _enforce_c4v(cell, size):
    """Enforce C4v symmetry by averaging over all 8 symmetry operations."""
    sym = np.zeros_like(cell, dtype=float)
    for i in range(size):
        for j in range(size):
            points = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            s = sum(cell[pi, pj] for pi, pj in points)
            sym[i, j] = 1.0 if s >= 4 else 0.0
    return sym.astype(int)

def _generate_cell_from_params(params, size=32):
    """
    Generate a C4v-symmetric binary mask from harmonic parameters.
    params: [a0, a10, a11, a20, a21, a22, ...]
    """
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size
    y = (yy + 0.5) / size
    
    field = np.zeros((size, size), dtype=np.float64)
    
    # C4v symmetric harmonics
    # f(x,y) = sum a_lm * (cos(2pi l x) cos(2pi m y) + cos(2pi m x) cos(2pi l y))
    # Note: For l=m, we only use one term to avoid double counting, but the sum is symmetric anyway.
    # Modes: (0,0) is constant. (1,0), (0,1) are same. (1,1), (2,0), (2,1), (2,2)
    
    # Map params to modes
    # params[0]: DC (bias)
    # params[1]: (1,0)/(0,1)
    # params[2]: (1,1)
    # params[3]: (2,0)/(0,2)
    # params[4]: (2,1)/(1,2)
    # params[5]: (2,2)
    # params[6]: (3,0)/(0,3)
    # params[7]: (3,1)/(1,3)
    # params[8]: (3,2)/(2,3)
    # params[9]: (3,3)
    
    modes = [
        (0, 0),
        (1, 0), # implies (0,1)
        (1, 1),
        (2, 0), # implies (0,2)
        (2, 1), # implies (1,2)
        (2, 2),
        (3, 0), # implies (0,3)
        (3, 1), # implies (1,3)
        (3, 2), # implies (2,3)
        (3, 3)
    ]
    
    for idx, (l, m) in enumerate(modes):
        if idx >= len(params):
            break
        amp = params[idx]
        if l == 0 and m == 0:
            field += amp
        elif l == m:
            field += amp * np.cos(2 * np.pi * l * x) * np.cos(2 * np.pi * m * y)
        else:
            field += amp * (np.cos(2 * np.pi * l * x) * np.cos(2 * np.pi * m * y) +
                            np.cos(2 * np.pi * m * x) * np.cos(2 * np.pi * l * y))
            
    # Threshold
    cell = (field > 0).astype(int)
    
    # Enforce C4v symmetry
    cell = _enforce_c4v(cell, size)
    
    return cell

def _search_harmonic(size=32, eps_low=1.0, eps_high=12.0):
    """
    Coarse-to-fine grid search over harmonic parameters.
    """
    Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw = _build_pw_evaluator(
        size, eps_low, eps_high, pw_half=2)
    
    best_cell = None
    best_score = -1e30
    
    # Number of parameters to optimize
    n_params = 6 # DC, (1,0), (1,1), (2,0), (2,1), (2,2)
    
    # Define ranges for each parameter
    # We want to explore structures that look like rods, holes, or complex patterns.
    ranges = [
        (-1.0, 1.0),  # DC
        (-1.0, 1.0),  # (1,0)
        (-1.0, 1.0),  # (1,1)
        (-1.0, 1.0),  # (2,0)
        (-1.0, 1.0),  # (2,1)
        (-1.0, 1.0)   # (2,2)
    ]
    
    # Coarse grid: 3 points per dimension? 3^6 = 729 evaluations.
    # Each eval is fast (25x25 matrix eig, 12 k-points).
    # Let's try 4 points per dimension for first 3 params, 3 for rest?
    # Or just a random sampling / Latin hypercube to cover the space better?
    # Grid search is safer for "structural change" directive.
    
    # Let's use a small grid: 3 levels for all 6 params.
    n_levels = 3
    grid = np.meshgrid(*[np.linspace(lo, hi, n_levels) for lo, hi in ranges])
    grid_flat = [tuple(c) for c in zip(*[g.ravel() for g in grid])]
    
    # If time is tight, reduce levels or params
    if _remaining() < 120.0:
        n_levels = 2
        grid = np.meshgrid(*[np.linspace(lo, hi, n_levels) for lo, hi in ranges])
        grid_flat = [tuple(c) for c in zip(*[g.ravel() for g in grid])]
        
    for params in grid_flat:
        if _remaining() < 10.0:
            break
        cell = _generate_cell_from_params(params, size)
        score = _evaluate_gap_fast(cell, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low, eps_high)
        if score > best_score:
            best_score = score
            best_cell = cell
            best_params = params
    
    if best_cell is None:
        return None, -1e30
        
    # Fine-tuning around the best found parameters
    if _remaining() > 20.0:
        # Local refinement: small grid around best_params
        step = 0.2
        local_grid = []
        # Generate offsets
        offsets = [(-step, 0, step)] * n_params
        local_grid = np.meshgrid(*offsets)
        local_flat = [tuple(c) for c in zip(*[g.ravel() for g in local_grid])]
        
        for offsets in local_flat:
            if _remaining() < 10.0:
                break
            new_params = [best_params[i] + offsets[i] for i in range(n_params)]
            cell = _generate_cell_from_params(new_params, size)
            score = _evaluate_gap_fast(cell, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low, eps_high)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = new_params
                
    return best_cell, best_score

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Seed: centred dielectric rod radius 0.2 (guaranteed valid, C4v symmetric)
    seed = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        seed.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    
    if _remaining() < 30.0:
        return seed
    
    best_cell = None
    best_score = -1e30
    
    # Strategy: Harmonic Grid Search
    try:
        cell, score = _search_harmonic(size, eps_low, eps_high)
        if cell is not None and score > best_score:
            best_score = score
            best_cell = cell
    except Exception:
        pass
    
    # Fallback to seed if nothing found or score is very bad
    if best_cell is None or best_score < 0.0:
        return seed
    
    # Final C4v enforcement
    c = np.asarray(best_cell, dtype=int)
    c = _enforce_c4v(c, size)
    
    # Verify symmetry
    if not np.array_equal(c, c.T):
        c = (c + c.T) // 2
    if not np.array_equal(c, c[::-1, :]):
        c = (c + c[::-1, :]) // 2
    if not np.array_equal(c, c[:, ::-1]):
        c = (c + c[:, ::-1]) // 2
        
    return c.tolist()
