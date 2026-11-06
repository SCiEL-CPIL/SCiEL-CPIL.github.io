"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _make_superellipse_cell(n, a, p, center=(0.5, 0.5)):
    """Create an n×n binary cell with a superellipse centered at (0.5, 0.5).
    
    Superellipse: |x/a|^p + |y/a|^p <= 1 (isotropic with 4-fold symmetry)
    """
    cell = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            px = (i + 0.5) / n - center[0]
            py = (j + 0.5) / n - center[1]
            val = (abs(px / a) ** p + abs(py / a) ** p)
            if val <= 1.0:
                cell[i, j] = 1
    # Enforce 4-fold symmetry
    sym = cell.copy()
    sym = (sym + sym[::-1, :]) // 2
    sym = (sym + sym[:, ::-1]) // 2
    sym = (sym + sym.T) // 2
    return sym.tolist()

def _compute_tm_bands_fast(cell, n, N=7, n_kpts=8):
    """Fast computation of TM bands using plane-wave method.
    
    cell: n×n list of lists (0/1)
    N: number of plane waves
    Returns: gap-midgap ratio for bands 4-5
    """
    eps_low = 1.0
    eps_high = 12.0
    
    # Precompute 1/eps at pixel centers
    inv_eps = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            inv_eps[i, j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
    
    # Compute Fourier coefficients of 1/eps
    # inv_eps_{mx,my} = (1/n^2) sum_{i,j} inv_eps[i,j] exp(-i*2*pi*(mx*(i+0.5)/n + my*(j+0.5)/n))
    max_G = N // 2
    inv_eps_coeffs = {}
    for mx in range(-max_G, max_G + 1):
        for my in range(-max_G, max_G + 1):
            total = 0.0
            for i in range(n):
                x = (i + 0.5) / n
                for j in range(n):
                    y = (j + 0.5) / n
                    total += inv_eps[i, j] * np.exp(-1j * 2 * np.pi * (mx * x + my * y))
            inv_eps_coeffs[(mx, my)] = total / (n * n)
    
    # G vectors in units of 2*pi
    G_vectors = []
    for mx in range(-max_G, max_G + 1):
        for my in range(-max_G, max_G + 1):
            G_vectors.append((mx, my))
    
    n_pw = len(G_vectors)
    
    # K-points in irreducible BZ (in units of 2*pi)
    # Irreducible sector: 0 <= kx <= ky, kx + ky <= 0.5
    k_points = []
    for ik in range(n_kpts):
        t = ik / max(1, n_kpts - 1)
        k_points.append((t * 0.25, 0.0))  # X to M direction
        k_points.append((t * 0.125, t * 0.125))  # G to M direction
    
    # Deduplicate and limit
    seen = set()
    unique_kpts = []
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_kpts.append(k)
    k_points = unique_kpts[:n_kpts]
    
    band4_vals = []
    band5_vals = []
    
    for kx, ky in k_points:
        # Build matrices
        A = np.zeros((n_pw, n_pw), dtype=complex)
        B = np.zeros((n_pw, n_pw), dtype=complex)
        
        for gi, (Gx, Gy) in enumerate(G_vectors):
            kpx = kx + Gx
            kpy = ky + Gy
            A[gi, gi] = (2 * np.pi * kpx) ** 2 + (2 * np.pi * kpy) ** 2
        
        for gi, (Gx, Gy) in enumerate(G_vectors):
            for gj, (Gx2, Gy2) in enumerate(G_vectors):
                dGx = Gx - Gx2
                dGy = Gy - Gy2
                if (dGx, dGy) in inv_eps_coeffs:
                    B[gi, gj] = inv_eps_coeffs[(dGx, dGy)]
        
        try:
            B_inv_A = np.linalg.solve(B, A)
            eigvals = np.linalg.eigvals(B_inv_A)
            eigvals = np.real(eigvals)
            eigvals = eigvals[eigvals > 0]
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                band4_vals.append(eigvals[3])
                band5_vals.append(eigvals[4])
        except:
            pass
    
    if not band4_vals or not band5_vals:
        return 0.0
    
    max_omega4 = max(band4_vals)
    min_omega5 = min(band5_vals)
    
    denom = min_omega5 + max_omega4
    if denom == 0:
        return 0.0
    
    return 2 * (min_omega5 - max_omega4) / denom

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 215  # 15s safety margin
    
    best_cell = None
    best_score = -1.0
    
    # Phase 1: Coarse grid over superellipse parameters
    a_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    p_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
    
    for a in a_values:
        if time.time() > deadline:
            break
        for p in p_values:
            if time.time() > deadline:
                break
            cell = _make_superellipse_cell(size, a, p)
            score = _compute_tm_bands_fast(cell, size, N=7, n_kpts=6)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    if time.time() > deadline:
        if best_cell is None:
            best_cell = _make_superellipse_cell(size, 0.3, 2.0)
        return best_cell
    
    # Phase 2: Nelder-Mead refinement
    def objective(params):
        a, p = params
        if a < 0.1 or a > 0.6 or p < 0.5 or p > 12:
            return 0.0
        cell = _make_superellipse_cell(size, a, p)
        return -_compute_tm_bands_fast(cell, size, N=7, n_kpts=6)
    
    # Find best initial point
    best_a, best_p = 0.3, 2.0
    for a in a_values:
        for p in p_values:
            cell = _make_superellipse_cell(size, a, p)
            score = _compute_tm_bands_fast(cell, size, N=7, n_kpts=6)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_a, best_p = a, p
    
    x0 = np.array([best_a, best_p])
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if result.fun < 0:
            a_opt, p_opt = result.x
            if 0.1 < a_opt < 0.6 and 0.5 < p_opt < 12:
                cell = _make_superellipse_cell(size, a_opt, p_opt)
                score = _compute_tm_bands_fast(cell, size, N=9, n_kpts=8)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    except:
        pass
    
    if time.time() > deadline:
        return best_cell if best_cell else _make_superellipse_cell(size, 0.3, 2.0)
    
    # Phase 3: Try asymmetric superellipses (break 4-fold temporarily, then symmetrize)
    # Actually, let's try a different shape family: ring/annulus
    for inner_r in [0.1, 0.15, 0.2]:
        for outer_r in [0.3, 0.35, 0.4, 0.45]:
            if time.time() > deadline:
                break
            cell = np.zeros((size, size), dtype=np.int32)
            for i in range(size):
                for j in range(size):
                    px = (i + 0.5) / size - 0.5
                    py = (j + 0.5) / size - 0.5
                    r = np.sqrt(px * px + py * py)
                    if inner_r <= r <= outer_r:
                        cell[i, j] = 1
            # Enforce 4-fold symmetry
            sym = cell.copy()
            sym = (sym + sym[::-1, :]) // 2
            sym = (sym + sym[:, ::-1]) // 2
            sym = (sym + sym.T) // 2
            cell = sym
            score = _compute_tm_bands_fast(cell.tolist(), size, N=7, n_kpts=6)
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
    
    if time.time() > deadline:
        return best_cell if best_cell else _make_superellipse_cell(size, 0.3, 2.0)
    
    # Phase 4: Fine-tune with higher resolution
    if best_cell is not None:
        for da in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            for dp in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                if time.time() > deadline:
                    break
                a_try = best_a + da
                p_try = best_p + dp
                if a_try < 0.1 or a_try > 0.6 or p_try < 0.5 or p_try > 12:
                    continue
                cell = _make_superellipse_cell(size, a_try, p_try)
                score = _compute_tm_bands_fast(cell, size, N=9, n_kpts=8)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    if best_cell is None:
        best_cell = _make_superellipse_cell(size, 0.3, 2.0)
    
    return best_cell
# EVOLVE-BLOCK-END
