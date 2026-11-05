"""
Flattest plus-minus-one polynomials with 24 coefficients.
Strategy:
  1. LP-based constraint satisfaction: encode |P(e^{iθ_j})| < T as linear constraints,
     use scipy.optimize.linprog to find ±1 assignments satisfying the bound.
  2. Binary search on T to find the tightest feasible sup-norm bound.
  3. Local search polishing from LP-found solutions and spectral synthesis starts.
  4. Deadline guard: return best found before 140 s wall-clock.
"""
import numpy as np
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150 s kill

N = 24
SQRTN = math.sqrt(N)

# Precompute powers of z for evaluation at a given theta (vectorized)
_KS = np.arange(N)

def _time_left():
    return _DEADLINE - time.time()

def _eval_poly(coeffs, theta):
    """Evaluate |P(e^{i theta})| where P(z) = sum c_k z^k."""
    z = np.exp(1j * theta)
    return float(np.abs(np.dot(coeffs, z ** _KS)))

def _refine_peak(coeffs, ang, iters=50):
    """
    Golden-section refine the maximum of |P(e^{i theta})| near ang.
    """
    half_w = math.pi / 4096.0
    a = ang - half_w
    b = ang + half_w
    
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _eval_poly(coeffs, c)
    fd = _eval_poly(coeffs, d)
    
    for _ in range(iters):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _eval_poly(coeffs, d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _eval_poly(coeffs, c)
    
    mid = (a + b) / 2.0
    return _eval_poly(coeffs, mid)

def _supnorm_precise(coeffs, n_grid=8192):
    """
    Compute sup-norm: FFT screening on dense grid, then refine top peaks.
    """
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    abs_vals = np.abs(vals)
    
    top_idx = np.argsort(abs_vals)[-10:][::-1]
    
    best = 0.0
    for idx in top_idx:
        ang = 2.0 * math.pi * idx / n_grid
        val = _refine_peak(coeffs, ang, iters=40)
        if val > best:
            best = val
    
    return best

def _supnorm_fast(coeffs, n_grid=4096):
    """Fast sup-norm estimate via FFT only (no refinement)."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _local_search(coeffs, max_iters=2000):
    """
    2-bit-flip local search with simulated annealing.
    Returns best coeffs and best sup-norm found.
    """
    coeffs = list(coeffs)
    best_coeffs = list(coeffs)
    best_val = _supnorm_fast(coeffs)
    
    temp = 0.5
    cool_rate = 0.999
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        i, j = random.sample(range(N), 2)
        coeffs[i] = -coeffs[i]
        coeffs[j] = -coeffs[j]
        
        new_val = _supnorm_fast(coeffs)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = list(coeffs)
        elif new_val < best_val + temp:
            if random.random() < math.exp(-(new_val - best_val) / (temp + 1e-12)):
                pass
            else:
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
        else:
            coeffs[i] = -coeffs[i]
            coeffs[j] = -coeffs[j]
        
        temp *= cool_rate
    
    final_val = _supnorm_precise(best_coeffs)
    return best_coeffs, final_val

def _try_golay_rudin_shapiro():
    """Golay-Rudin-Shapiro recursion, truncated to 24 coefficients."""
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    return p[:24]

def _spectral_synthesis_init(seed_idx=0):
    """
    Generate a coherent ±1 sequence via inverse FFT of a flattened target waveform.
    """
    rng = np.random.RandomState(seed_idx * 1000 + 42)
    
    alpha = 0.3 + 0.5 * (seed_idx % 5) / 5.0
    
    n_spec = 64
    freqs = np.arange(n_spec)
    taper = 0.5 * (1.0 + np.cos(np.pi * freqs / (n_spec - 1)))
    
    target = (1 - alpha) * np.ones(n_spec) + alpha * taper
    target = target / np.max(target)
    
    phases = rng.uniform(0, 2 * np.pi, n_spec)
    target_complex = target * np.exp(1j * phases)
    
    time_signal = np.fft.ifft(target_complex)
    
    real_part = np.real(time_signal[:N])
    
    coeffs = np.sign(real_part)
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    
    return [int(c) for c in coeffs]

def _lp_find_assignment(T, n_grid=256, max_iter=50):
    """
    Use LP relaxation + rounding to find a ±1 assignment satisfying |P(e^{iθ_j})| < T.
    
    We encode the constraints:
      For each grid point θ_j:
        Re(P(e^{iθ_j})) ≤ T  and  Re(P(e^{iθ_j})) ≥ -T
        Im(P(e^{iθ_j})) ≤ T  and  Im(P(e^{iθ_j})) ≥ -T
    
    Since c_k ∈ {±1}, we can relax to c_k ∈ [-1, 1] and solve LP,
    then round to nearest ±1 and check feasibility.
    
    Returns: (coeffs, feasible) where feasible is True if the rounded solution satisfies all constraints.
    """
    from scipy.optimize import linprog
    
    # Precompute the constraint matrix
    # For each θ_j and each k: contribution to Re and Im
    thetas = 2.0 * np.pi * np.arange(n_grid) / n_grid
    cos_mat = np.cos(np.outer(thetas, _KS))  # shape (n_grid, N)
    sin_mat = np.sin(np.outer(thetas, _KS))  # shape (n_grid, N)
    
    # Constraints: for each θ_j:
    #   sum_k c_k cos(kθ_j) ≤ T    (Re ≤ T)
    #   -sum_k c_k cos(kθ_j) ≤ T   (Re ≥ -T)
    #   sum_k c_k sin(kθ_j) ≤ T    (Im ≤ T)
    #   -sum_k c_k sin(kθ_j) ≤ T   (Im ≥ -T)
    #   c_k ≤ 1, -c_k ≤ 1 for all k (bounds)
    
    # A matrix: shape (4*n_grid + 2*N, N)
    A_rows = []
    b_vals = []
    
    for j in range(n_grid):
        A_rows.append(cos_mat[j])
        b_vals.append(T)
        A_rows.append(-cos_mat[j])
        b_vals.append(T)
        A_rows.append(sin_mat[j])
        b_vals.append(T)
        A_rows.append(-sin_mat[j])
        b_vals.append(T)
    
    # Bounds: -1 ≤ c_k ≤ 1
    bounds = [(-1.0, 1.0)] * N
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Objective: minimize 0 (feasibility problem)
    c_obj = np.zeros(N)
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    except Exception:
        return None, False
    
    if not res.success:
        return None, False
    
    # Round to ±1
    x = res.x
    coeffs = np.sign(x)
    coeffs[coeffs == 0] = 1.0  # handle zeros
    coeffs = [int(c) for c in coeffs]
    
    # Check if the rounded solution actually satisfies all constraints
    # Compute P(e^{iθ_j}) for all grid points
    z_vals = np.exp(1j * thetas)
    P_vals = np.array([np.dot(coeffs, z_vals[j] ** _KS) for j in range(n_grid)])
    
    max_abs = np.max(np.abs(P_vals))
    
    if max_abs < T:
        return coeffs, True
    else:
        return coeffs, False

def _lp_binary_search(T_low=1.0, T_high=5.0, n_grid=256, max_iters=30):
    """
    Binary search on T to find the smallest feasible sup-norm bound.
    Returns the best coefficients found.
    """
    best_coeffs = None
    best_T = T_high
    
    for _ in range(max_iters):
        if _time_left() < 10.0:
            break
        
        T_mid = (T_low + T_high) / 2.0
        coeffs, feasible = _lp_find_assignment(T_mid, n_grid=n_grid)
        
        if feasible:
            best_T = T_mid
            best_coeffs = coeffs
            T_high = T_mid
        else:
            T_low = T_mid
        
        # If we've converged or time is running out
        if T_high - T_low < 0.01:
            break
    
    return best_coeffs, best_T

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    np.random.seed(42)
    
    # Start with Golay-Rudin-Shapiro as initial best
    best_coeffs = _try_golay_rudin_shapiro()
    best_val = _supnorm_precise(best_coeffs)
    
    # Phase 1: LP-based constraint satisfaction to find good starting points
    lp_candidates = []
    
    # Try different grid resolutions for LP
    for n_grid in [128, 256, 512]:
        if _time_left() < 20.0:
            break
        
        # Binary search on T
        coeffs, T_found = _lp_binary_search(T_low=1.0, T_high=5.0, n_grid=n_grid, max_iters=20)
        if coeffs is not None:
            lp_candidates.append(coeffs)
            val = _supnorm_precise(coeffs)
            if val < best_val:
                best_val = val
                best_coeffs = coeffs
    
    # Phase 2: Spectral synthesis initial points
    spectral_inits = []
    for s in range(15):
        spectral_inits.append(_spectral_synthesis_init(s))
    
    # Phase 3: Multi-start local search
    # First try LP candidates, then spectral synthesis, then random
    all_starts = lp_candidates + spectral_inits
    
    n_restarts = 400
    for r in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        if r < len(all_starts):
            coeffs = all_starts[r]
        else:
            coeffs = [random.choice([1, -1]) for _ in range(N)]
        
        new_coeffs, new_val = _local_search(coeffs, max_iters=1500)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = new_coeffs
    
    # Final precise evaluation
    final_val = _supnorm_precise(best_coeffs)
    
    return best_coeffs

if __name__ == "__main__":
    result = construct()
    print(result)
