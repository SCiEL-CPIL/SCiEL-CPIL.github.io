"""Design a symmetric 32x32 photonic-crystal unit cell to maximize the TM band-4/5 gap."""
import time
import math
import numpy as np

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0
_EPS_INV_MEAN = 0.5 * (1.0 / _EPS_HIGH + 1.0 / _EPS_LOW)
_EPS_INV_AMP = 0.5 * (1.0 / _EPS_HIGH - 1.0 / _EPS_LOW)

# Plane-wave basis: 289 = 17x17 grid of G vectors
_G_RANGE = 8
_GV = [(m, n) for m in range(-_G_RANGE, _G_RANGE+1) for n in range(-_G_RANGE, _G_RANGE+1)]
_NPW = len(_GV)  # 289

# Precompute phase factors
_PIXEL_X = np.array([(i + 0.5) / _SIZE - 0.5 for i in range(_SIZE)])
_PIXEL_Y = np.array([(j + 0.5) / _SIZE - 0.5 for j in range(_SIZE)])
_XY = np.meshgrid(_PIXEL_X, _PIXEL_Y, indexing='ij')
_X_FLAT = _XY[0].flatten()
_Y_FLAT = _XY[1].flatten()

_PHASE = np.zeros((_NPW, _SIZE * _SIZE), dtype=np.complex128)
for ig, (gm, gn) in enumerate(_GV):
    phase = 2.0 * np.pi * (gm * _X_FLAT + gn * _Y_FLAT)
    _PHASE[ig, :] = np.exp(1j * phase)

# Precompute G index lookup: (dm, dn) -> index in _GV
_G_INDEX = {}
for idx, (gm, gn) in enumerate(_GV):
    _G_INDEX[(gm, gn)] = idx

# Precompute difference indices for S matrix construction
_DIFF_IDX = np.zeros((_NPW, _NPW), dtype=np.int32)
for i in range(_NPW):
    gi, gj_ = _GV[i]
    for j in range(_NPW):
        gj, gk = _GV[j]
        dm = gi - gj
        dn = gj_ - gk
        if -_G_RANGE <= dm <= _G_RANGE and -_G_RANGE <= dn <= _G_RANGE:
            _DIFF_IDX[i, j] = _G_INDEX[(dm, dn)]
        else:
            _DIFF_IDX[i, j] = -1

# K-points: 30 points in the irreducible BZ
_KPOINTS = []
_KPOINTS.append((0.0, 0.0))
_KPOINTS.append((0.25, 0.25))
_KPOINTS.append((0.0, 0.5))
_KPOINTS.append((0.125, 0.125))
_KPOINTS.append((0.0, 0.25))
_KPOINTS.append((0.125, 0.25))
for i in range(5):
    for j in range(5):
        u = i / 4.0
        v = j / 4.0
        if u + v <= 1.0:
            kx = u * 0.25
            ky = u * 0.25 + v * 0.5
            _KPOINTS.append((kx, ky))

_seen = set()
_unique_kpts = []
for p in _KPOINTS:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in _seen:
        _seen.add(key)
        _unique_kpts.append(p)
_KPOINTS = _unique_kpts[:30]

# Precompute D_diag for each k-point
_D_DIAGS = []
for kx, ky in _KPOINTS:
    D_diag = np.zeros(_NPW, dtype=np.float64)
    for i in range(_NPW):
        gm, gn = _GV[i]
        kx_tot = kx + gm
        ky_tot = ky + gn
        D_diag[i] = kx_tot * kx_tot + ky_tot * ky_tot
    _D_DIAGS.append(D_diag)

# Precompute valid mask for _DIFF_IDX
_VALID_MASK = _DIFF_IDX >= 0

def _compute_fourier_coeffs(cell_flat):
    """Compute Fourier coefficients of 1/eps for a given cell (flat array)."""
    g = 2.0 * cell_flat - 1.0
    # coeffs[0] = _EPS_INV_MEAN
    # coeffs[ig] = _EPS_INV_AMP * (1/N) * sum_pixels g[p] * conj(phase[ig, p])
    # vdot does conj(first) * second, so vdot(_PHASE[ig,:], g) = sum(conj(phase)*g)
    f_coeffs = np.vdot(_PHASE[1:, :], g) / (_SIZE * _SIZE)
    coeffs = np.empty(_NPW, dtype=np.complex128)
    coeffs[0] = _EPS_INV_MEAN
    coeffs[1:] = _EPS_INV_AMP * f_coeffs
    return coeffs

def _build_S_matrix(coeffs):
    """Build the (1/eps) matrix S from Fourier coefficients."""
    S = np.zeros((_NPW, _NPW), dtype=np.complex128)
    S[_VALID_MASK] = coeffs[_DIFF_IDX[_VALID_MASK]]
    return S

def _tm_bands_fast(cell):
    """Compute TM band-4/5 gap ratio using vectorized NumPy solver."""
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    coeffs = _compute_fourier_coeffs(cell_flat)
    S = _build_S_matrix(coeffs)
    S = 0.5 * (S + S.conj().T)
    
    # Precompute S inverse once (it's the same for all k-points)
    S_reg = S + 1e-10 * np.eye(_NPW)
    
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for ik in range(len(_KPOINTS)):
        D_diag = _D_DIAGS[ik]
        
        try:
            # Solve generalized eigenvalue: D E = omega^2 S E
            # Convert: S^{-1} D E = omega^2 E
            # Use Cholesky: S = L L^H
            L = np.linalg.cholesky(S_reg)
            L_inv = np.linalg.inv(L)
            # M = L^{-H} D L^{-1}
            M = L_inv.conj().T @ (D_diag[:, None] * L_inv)
            M = 0.5 * (M + M.conj().T)
            eigvals = np.linalg.eigvalsh(M)
            if len(eigvals) >= 5:
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                if w4_sq > 0 and w5_sq > 0:
                    w4 = math.sqrt(w4_sq)
                    w5 = math.sqrt(w5_sq)
                    band_4_max = max(band_4_max, w4)
                    band_5_min = min(band_5_min, w5)
        except Exception:
            try:
                A = np.linalg.solve(S_reg, D_diag[:, None])
                A = 0.5 * (A + A.conj().T)
                eigvals = np.linalg.eigvalsh(A)
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = math.sqrt(w4_sq)
                        w5 = math.sqrt(w5_sq)
                        band_4_max = max(band_4_max, w4)
                        band_5_min = min(band_5_min, w5)
            except Exception:
                continue
    
    if band_4_max <= 0 or band_5_min <= 0:
        return -1.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / (band_5_min + band_4_max)

def _enforce_symmetry(cell):
    """Enforce all symmetry constraints on a 32x32 cell."""
    cell = np.array(cell, dtype=np.int8)
    # Vertical mirror: cell[i][j] = cell[31-i][j]
    for i in range(_SIZE // 2):
        for j in range(_SIZE):
            cell[_SIZE-1-i, j] = cell[i, j]
    # Horizontal mirror: cell[i][j] = cell[i][31-j]
    for i in range(_SIZE):
        for j in range(_SIZE // 2):
            cell[i, _SIZE-1-j] = cell[i, j]
    # Transpose: cell[i][j] = cell[j][i]
    for i in range(_SIZE):
        for j in range(i+1, _SIZE):
            if cell[i, j] != cell[j, i]:
                val = max(cell[i, j], cell[j, i])
                cell[i, j] = val
                cell[j, i] = val
    return cell.tolist()

# High-symmetry center points for RBF representation
# C4v symmetric positions: center, edges, corners
_RBF_CENTERS = [
    (0.0, 0.0),       # center
    (0.25, 0.0),      # edge midpoints (4 equivalent)
    (0.5, 0.0),       # corner-adjacent edge
    (0.25, 0.25),     # diagonal midpoints
    (0.5, 0.5),       # corner
    (0.125, 0.0),     # off-center edge
    (0.125, 0.125),   # off-center diagonal
]

def _make_cell_from_rbf(params):
    """Create a 32x32 cell from RBF parameters.
    
    params: [threshold, w0, s0, w1, s1, w2, s2, w3, s3, w4, s4, w5, s5, w6, s6]
    where w_i are weights and s_i are sigmas for each center type.
    """
    threshold = params[0] if len(params) > 0 else 0.0
    cell = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    
    for i in range(_SIZE):
        for j in range(_SIZE):
            x = (i + 0.5) / _SIZE - 0.5
            y = (j + 0.5) / _SIZE - 0.5
            val = 0.0
            
            for c_idx, (cx, cy) in enumerate(_RBF_CENTERS):
                w = params[1 + 2*c_idx] if len(params) > 1 + 2*c_idx else 0.0
                s = params[2 + 2*c_idx] if len(params) > 2 + 2*c_idx else 0.1
                if s <= 0.001:
                    continue
                # Generate all C4v equivalent points
                dx = x - cx
                dy = y - cy
                r2 = dx*dx + dy*dy
                val += w * math.exp(-r2 / (2.0 * s * s))
                
                # C4v: rotate by 90, 180, 270 and reflect
                # Rotation 90: (x,y) -> (-y, x), so center (cx,cy) -> (-cy, cx)
                dx2 = x + cy
                dy2 = y - cx
                r2_90 = dx2*dx2 + dy2*dy2
                val += w * math.exp(-r2_90 / (2.0 * s * s))
                
                # Rotation 180: (x,y) -> (-x, -y), center (cx,cy) -> (-cx, -cy)
                dx3 = x + cx
                dy3 = y + cy
                r2_180 = dx3*dx3 + dy3*dy3
                val += w * math.exp(-r2_180 / (2.0 * s * s))
                
                # Rotation 270: (x,y) -> (y, -x), center (cx,cy) -> (cy, -cx)
                dx4 = x - cy
                dy4 = y + cx
                r2_270 = dx4*dx4 + dy4*dy4
                val += w * math.exp(-r2_270 / (2.0 * s * s))
                
                # Reflection x -> -x: center (cx,cy) -> (-cx, cy)
                dx5 = x + cx
                dy5 = y - cy
                r2_ref_x = dx5*dx5 + dy5*dy5
                val += w * math.exp(-r2_ref_x / (2.0 * s * s))
                
                # Reflection y -> -y: center (cx,cy) -> (cx, -cy)
                dx6 = x - cx
                dy6 = y + cy
                r2_ref_y = dx6*dx6 + dy6*dy6
                val += w * math.exp(-r2_ref_y / (2.0 * s * s))
                
                # Reflection x->-x, y->-y: center (cx,cy) -> (-cx, -cy) - same as 180
                # Already covered
                
            cell[i, j] = val
    
    # Threshold to binary
    binary_cell = (cell > threshold).astype(np.int8)
    return _enforce_symmetry(binary_cell)

def _optimize_cell_rbf(initial_params, time_limit, rng=None):
    """Optimize RBF parameters using simulated annealing with local refinement."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    best_params = list(initial_params)
    best_score = _tm_bands_fast(_make_cell_from_rbf(best_params))
    current_params = list(best_params)
    current_score = best_score
    
    n_params = len(current_params)
    deadline = time.time() + time_limit
    
    # Temperature schedule
    T0 = 0.05
    T_min = 0.001
    cooling = 0.97
    
    T = T0
    no_improve = 0
    
    while time.time() < deadline - 3.0:
        # Generate perturbation
        trial_params = list(current_params)
        p = rng.integers(0, n_params)
        if p == 0:
            trial_params[p] += rng.normal(0, T)
            trial_params[p] = max(-0.5, min(0.5, trial_params[p]))
        else:
            if p % 2 == 1:  # weight
                trial_params[p] += rng.normal(0, T)
                trial_params[p] = max(-2.0, min(2.0, trial_params[p]))
            else:  # sigma
                trial_params[p] += rng.normal(0, T * 0.5)
                trial_params[p] = max(0.02, min(0.3, trial_params[p]))
        
        trial_cell = _make_cell_from_rbf(trial_params)
        trial_score = _tm_bands_fast(trial_cell)
        
        # Accept/reject
        if trial_score > current_score or rng.random() < math.exp((trial_score - current_score) / max(T, 1e-10)):
            current_params = trial_params
            current_score = trial_score
            no_improve = 0
            if trial_score > best_score:
                best_score = trial_score
                best_params = list(current_params)
        else:
            no_improve += 1
        
        T = max(T_min, T * cooling)
        
        # If stuck for too long, restart with larger perturbation
        if no_improve > 200:
            T = T0 * 0.5
            no_improve = 0
    
    # Local refinement around best
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0:
        best_params, best_score = _local_refine_rbf(best_params, remaining, rng)
    
    return best_params, best_score

def _local_refine_rbf(params, time_limit, rng=None):
    """Local refinement of RBF parameters."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    best_params = list(params)
    best_score = _tm_bands_fast(_make_cell_from_rbf(best_params))
    n_params = len(best_params)
    
    deadline = time.time() + time_limit
    step_sizes = [0.02] + [0.05] * ((n_params - 1) // 2) + [0.02] * ((n_params - 1) // 2)
    
    while time.time() < deadline - 2.0:
        improved = False
        for p in range(n_params):
            for direction in [-1, 1]:
                trial_params = list(best_params)
                trial_params[p] += direction * step_sizes[p]
                
                if p == 0:
                    trial_params[p] = max(-0.5, min(0.5, trial_params[p]))
                elif p % 2 == 1:
                    trial_params[p] = max(-2.0, min(2.0, trial_params[p]))
                else:
                    trial_params[p] = max(0.02, min(0.3, trial_params[p]))
                
                trial_cell = _make_cell_from_rbf(trial_params)
                trial_score = _tm_bands_fast(trial_cell)
                
                if trial_score > best_score:
                    best_params = trial_params
                    best_score = trial_score
                    improved = True
                    step_sizes[p] *= 1.3
                else:
                    step_sizes[p] *= 0.7
        
        if not improved:
            all_small = all(s < 0.002 for s in step_sizes)
            if all_small:
                break
    
    return best_params, best_score

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 220.0
    
    size = problem.get("pixels", 32)
    if size != _SIZE:
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if x*x + y*y < 0.04 else 0
        return cell
    
    rng = np.random.default_rng(42)
    
    # Initial RBF parameter candidates
    # [threshold, w0, s0, w1, s1, w2, s2, w3, s3, w4, s4, w5, s5, w6, s6]
    candidates = [
        [0.0, 1.0, 0.15, 0.5, 0.1, 0.3, 0.08, 0.8, 0.12, 0.2, 0.2, 0.4, 0.06, 0.3, 0.05],
        [0.0, 1.5, 0.12, 0.8, 0.08, 0.5, 0.06, 1.0, 0.1, 0.3, 0.15, 0.6, 0.05, 0.4, 0.04],
        [0.0, 1.0, 0.1, 1.0, 0.06, 0.5, 0.05, 1.2, 0.08, 0.2, 0.12, 0.8, 0.04, 0.5, 0.03],
        [0.0, 0.8, 0.18, 0.6, 0.12, 0.4, 0.1, 0.7, 0.15, 0.15, 0.22, 0.3, 0.08, 0.25, 0.06],
        [0.0, 1.2, 0.14, 0.9, 0.09, 0.6, 0.07, 1.1, 0.11, 0.25, 0.18, 0.7, 0.06, 0.45, 0.05],
        [0.0, 1.0, 0.16, 0.7, 0.11, 0.5, 0.09, 0.9, 0.13, 0.2, 0.2, 0.5, 0.07, 0.35, 0.06],
    ]
    
    best_params = candidates[0]
    best_score = -np.inf
    
    for init in candidates:
        if time.time() > deadline - 25.0:
            break
        remaining = deadline - time.time() - 8.0
        if remaining < 5.0:
            break
        params, score = _optimize_cell_rbf(init, min(25.0, remaining), rng)
        if score > best_score:
            best_score = score
            best_params = params
    
    # Final refinement
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        params, score = _local_refine_rbf(best_params, remaining, rng)
        if score > best_score:
            best_score = score
            best_params = params
    
    cell = _make_cell_from_rbf(best_params)
    cell = _enforce_symmetry(cell)
    
    return cell
