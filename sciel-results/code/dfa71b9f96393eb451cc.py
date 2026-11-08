"""
Combined model-and-optimize approach for the second autocorrelation inequality.

Key structural facts:
  * h >= 0, so g = conv(h,h) >= 0.
  * norm_1 = sum(g)/(m+1) = sum(h)^2/(m+1).
  * norm_inf = max(g).
  * The objective is scale-invariant: value depends only on shape of h.
  * We want to maximize l2 / (norm_1 * norm_inf).

Strategy:
  1. DCT-based low-rank superposition as the primary representation.
     This enforces global smoothness and reduces dimensionality, preventing
     high-frequency spikes that degrade the L2-to-L1 ratio.
  2. Multi-stage optimization:
     a. Coarse DCT optimization (few modes, L-BFGS-B)
     b. Fine DCT optimization (more modes, L-BFGS-B)
     c. Nelder-Mead polish on exact objective
     d. Final high-precision DCT polish
  3. Multi-start with diverse initializations in DCT coefficient space.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _neg_value(h):
    """Negative exact value for minimization."""
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _dct_basis(n_modes):
    """Return DCT-II basis matrix of shape (N, n_modes)."""
    k = np.arange(N)
    n = np.arange(n_modes)
    # DCT-II: cos(pi * (2k+1) * n / (2N))
    basis = np.cos(np.pi * np.outer(2 * k + 1, n) / (2 * N))
    return basis


def _dct_to_h(coeffs, basis):
    """Reconstruct h from DCT coefficients, enforcing nonnegativity."""
    h = basis @ coeffs
    return np.maximum(0.0, h)


def _h_to_dct(h, basis):
    """Project h onto DCT basis."""
    return basis.T @ h


def _dct_objective(coeffs, basis, deadline):
    """Objective function for DCT coefficient optimization."""
    if time.time() > deadline:
        return 1e10
    h = _dct_to_h(coeffs, basis)
    return -compute_value(h)


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 0: Quick baselines ---
    # All-ones
    track(np.ones(N))
    
    # Several plateaus
    for k in range(10, N + 1, 5):
        h = np.zeros(N)
        center = N // 2
        start = max(0, center - k // 2)
        end = min(N, start + k)
        h[start:end] = 1.0
        track(h)

    # --- Phase 1: Coarse DCT optimization (3-5 modes) ---
    coarse_modes_list = [3, 5]
    
    for n_modes in coarse_modes_list:
        if time.time() > deadline - 40:
            break
        
        basis = _dct_basis(n_modes)
        
        # Multiple diverse starts in coefficient space
        starts = []
        
        # Start 1: Project current best
        starts.append(_h_to_dct(best_h, basis))
        
        # Start 2: Uniform-ish (mostly DC component)
        s2 = np.zeros(n_modes)
        s2[0] = 1.0
        starts.append(s2)
        
        # Start 3: Low-frequency cosine bump
        s3 = np.zeros(n_modes)
        s3[0] = 0.5
        if n_modes > 1:
            s3[1] = 0.3
        starts.append(s3)
        
        # Start 4: Random
        starts.append(RNG.normal(0.5, 0.3, n_modes))
        
        # Start 5: Another random
        starts.append(RNG.normal(0.3, 0.5, n_modes))
        
        for start in starts:
            if time.time() > deadline - 20:
                break
            
            try:
                res = opt.minimize(
                    lambda c: _dct_objective(c, basis, deadline),
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_dct = _dct_to_h(res.x, basis)
                track(h_dct)
            except Exception:
                pass

    # --- Phase 2: Fine DCT optimization (8-12 modes) ---
    fine_modes_list = [8, 12]
    
    for n_modes in fine_modes_list:
        if time.time() > deadline - 30:
            break
        
        basis = _dct_basis(n_modes)
        
        # Start from projection of best_h
        init_coeffs = _h_to_dct(best_h, basis)
        
        # Multiple starts
        starts = [
            init_coeffs.copy(),
            init_coeffs + RNG.normal(0, 0.05, n_modes),
            init_coeffs * 0.8 + RNG.normal(0, 0.2, n_modes),
            init_coeffs * 0.5 + RNG.normal(0, 0.3, n_modes),
        ]
        
        for start in starts:
            if time.time() > deadline - 15:
                break
            
            try:
                res = opt.minimize(
                    lambda c: _dct_objective(c, basis, deadline),
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                h_dct = _dct_to_h(res.x, basis)
                track(h_dct)
            except Exception:
                pass

    # --- Phase 3: Nelder-Mead polish on exact objective ---
    for _ in range(15):
        if time.time() > deadline - 15:
            break
        scale = 0.02 + 0.03 * RNG.rand()
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            track(res.x)
        except Exception:
            pass

    # --- Phase 4: High-precision DCT polish (15-20 modes) ---
    hp_modes_list = [15, 20]
    
    for n_modes in hp_modes_list:
        if time.time() > deadline - 10:
            break
        
        basis = _dct_basis(n_modes)
        init_coeffs = _h_to_dct(best_h, basis)
        
        try:
            res = opt.minimize(
                lambda c: _dct_objective(c, basis, deadline),
                init_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            h_dct = _dct_to_h(res.x, basis)
            track(h_dct)
        except Exception:
            pass

    # --- Phase 5: Final Nelder-Mead polish ---
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _neg_value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
