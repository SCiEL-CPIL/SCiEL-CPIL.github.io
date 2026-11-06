"""
Improved sparse atom-based step height construction for third autocorrelation inequality.

Strategy:
- Model h as a sparse linear combination of shifted, scaled, and sign-flipped copies
  of a single medium-length "atom" (length-20 Hamming window).
- Parameterize by amplitudes and continuous offsets for 8-12 atoms.
- Use differential evolution to find good atom configurations.
- Then perform full-vector local search using gradient information.
- Key: normalize sum(h) during optimization to decouple scale from shape.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 400
_t0 = time.time()
_DEADLINE = 135.0

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _value_normalized(h):
    """Value with h normalized to sum=1, then scaled back.
    This decouples shape optimization from scale."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    h_norm = h / s  # sum = 1
    conv = _fft_convolve(h_norm)
    m = float(np.max(conv))
    # Original value = 2*N*max(conv(h))/sum(h)^2 = 2*N*max(conv(h/s * s))/s^2 = 2*N*s*max(conv(h/s))/s^2 = 2*N*max(conv(h/s))/s
    # Wait: conv(h) = conv(s*h_norm) = s^2 * conv(h_norm)
    # So value = 2*N*s^2*max(conv(h_norm))/s^2 = 2*N*max(conv(h_norm))
    # The s cancels! So value depends only on the shape (normalized h).
    return abs(2.0 * N * m)

def _smooth_value(h, beta=50.0):
    """Smooth approximation for gradient-based methods."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    h_norm = h / s
    conv = _fft_convolve(h_norm)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp)

# Atom definition: length-20 Hamming window
_ATOM_LEN = 20
_atom = np.hamming(_ATOM_LEN)

def _build_h(params):
    """
    Build h from sparse atom parameters.
    params: [amp_0, offset_0, sign_0, amp_1, offset_1, sign_1, ..., base]
    """
    K = (len(params) - 1) // 3
    h = np.zeros(N)
    for k in range(K):
        amp = params[3 * k]
        offset = params[3 * k + 1]
        sign = 1.0 if params[3 * k + 2] >= 0 else -1.0
        # Place atom at continuous offset, interpolate
        start = offset
        end = offset + _ATOM_LEN
        # Clip to valid range
        if start >= N or end <= 0:
            continue
        # For each position in the atom, determine which h index it maps to
        for j in range(_ATOM_LEN):
            pos = start + j
            if 0 <= pos < N:
                h[int(pos)] += sign * amp * _atom[j]
    base = params[-1]
    h += base
    return h

def _obj(params):
    """Exact objective using normalized value."""
    h = _build_h(params)
    return _value_normalized(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: Differential evolution with 8 atoms
    if deadline_ok():
        try:
            K = 8
            n_params = 3 * K + 1
            bounds = []
            for k in range(K):
                bounds.append((-2.0, 2.0))   # amplitude
                bounds.append((0.0, float(N - _ATOM_LEN)))  # offset
                bounds.append((-1.0, 1.0))   # sign
            bounds.append((-1.0, 1.0))       # base
            
            res = differential_evolution(
                _obj, bounds, seed=42, maxiter=100, tol=1e-8,
                popsize=20, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: Differential evolution with 10 atoms
    if deadline_ok():
        try:
            K = 10
            n_params = 3 * K + 1
            bounds = []
            for k in range(K):
                bounds.append((-2.0, 2.0))
                bounds.append((0.0, float(N - _ATOM_LEN)))
                bounds.append((-1.0, 1.0))
            bounds.append((-1.0, 1.0))
            
            res = differential_evolution(
                _obj, bounds, seed=7, maxiter=80, tol=1e-8,
                popsize=15, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Try symmetric structures - atoms in pairs with opposite signs
    if deadline_ok():
        try:
            # Symmetric: for each pair, one atom at offset d, one at N-ATOM_LEN-d with opposite sign
            K_pairs = 5
            def _obj_sym(params):
                h = np.zeros(N)
                for k in range(K_pairs):
                    amp = params[2 * k]
                    offset = params[2 * k + 1]
                    # Positive atom at offset
                    start1 = int(round(offset))
                    end1 = start1 + _ATOM_LEN
                    if 0 <= start1 < N:
                        s = max(0, start1)
                        e = min(N, end1)
                        if s < e:
                            h[s:e] += amp * _atom[s - start1:e - start1]
                    # Negative atom at symmetric position
                    start2 = N - _ATOM_LEN - int(round(offset))
                    end2 = start2 + _ATOM_LEN
                    if 0 <= start2 < N:
                        s = max(0, start2)
                        e = min(N, end2)
                        if s < e:
                            h[s:e] -= amp * _atom[s - start2:e - start2]
                return _value_normalized(h)
            
            bounds_sym = [(-2.0, 2.0), (0.0, float(N - _ATOM_LEN))] * K_pairs
            res = differential_evolution(
                _obj_sym, bounds_sym, seed=13, maxiter=120, tol=1e-8,
                popsize=25, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = np.zeros(N)
                for k in range(K_pairs):
                    amp = res.x[2 * k]
                    offset = res.x[2 * k + 1]
                    start1 = int(round(offset))
                    end1 = start1 + _ATOM_LEN
                    if 0 <= start1 < N:
                        s = max(0, start1)
                        e = min(N, end1)
                        if s < e:
                            h[s:e] += amp * _atom[s - start1:e - start1]
                    start2 = N - _ATOM_LEN - int(round(offset))
                    end2 = start2 + _ATOM_LEN
                    if 0 <= start2 < N:
                        s = max(0, start2)
                        e = min(N, end2)
                        if s < e:
                            h[s:e] -= amp * _atom[s - start2:e - start2]
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Full-vector local search using gradient of smooth objective
    # Since value = 2*N*max(conv(h_norm)), we can use the fact that the peak is at some lag
    # to compute approximate gradients
    if deadline_ok():
        try:
            h_local = best_h.copy()
            # Normalize for optimization
            s_init = np.sum(h_local)
            if abs(s_init) > 1e-12:
                h_work = h_local / s_init
            else:
                h_work = h_local.copy()
            
            for outer_it in range(50):
                if not deadline_ok():
                    break
                # Find peak lag
                conv = _fft_convolve(h_work)
                peak_lag = int(np.argmax(conv))
                
                # Gradient of max(conv) w.r.t. h at the peak lag
                # conv[k] = sum_i h[i] * h[i+k]
                # d(conv[k])/d(h[j]) = h[j+k] + h[j-k] (with appropriate bounds)
                grad = np.zeros(N)
                for j in range(N):
                    # Contribution from conv[peak_lag]
                    if 0 <= j + peak_lag < N:
                        grad[j] += h_work[j + peak_lag]
                    if 0 <= j - peak_lag < N:
                        grad[j] += h_work[j - peak_lag]
                
                # Also add small regularization to keep sum nonzero
                # Gradient of smooth max for other lags (approximate)
                for lag in range(max(0, peak_lag - 3), min(N, peak_lag + 4)):
                    if lag == peak_lag:
                        continue
                    weight = np.exp(-abs(lag - peak_lag) * 2.0) * 0.1
                    for j in range(N):
                        if 0 <= j + lag < N:
                            grad[j] += weight * h_work[j + lag]
                        if 0 <= j - lag < N:
                            grad[j] += weight * h_work[j - lag]
                
                # Line search
                best_step = 0.01
                best_v_trial = _value_normalized(h_work)
                for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                    h_trial = h_work - step * grad
                    v_trial = _value_normalized(h_trial)
                    if v_trial < best_v_trial:
                        best_v_trial = v_trial
                        best_step = step
                
                if best_step > 0:
                    h_work = h_work - best_step * grad
                
                # Update best
                h_candidate = h_work * s_init  # Scale back
                update_best(h_candidate)
                
                # Check convergence
                if best_step < 1e-6:
                    break
        except Exception:
            pass
    
    # Phase 5: Coordinate descent on full vector with FFT-based evaluation
    if deadline_ok():
        try:
            h_cd = best_h.copy()
            s_cd = np.sum(h_cd)
            if abs(s_cd) < 1e-12:
                h_cd = np.ones(N)
                s_cd = N
            h_work = h_cd / s_cd
            
            step_size = 0.05
            for it in range(200):
                if not deadline_ok():
                    break
                improved = False
                # Try random subset of coordinates for speed
                idxs = np.random.choice(N, size=min(50, N), replace=False)
                for i in idxs:
                    for delta in [step_size, -step_size]:
                        h_trial = h_work.copy()
                        h_trial[i] += delta
                        v_trial = _value_normalized(h_trial)
                        if v_trial < _value_normalized(h_work):
                            h_work = h_trial
                            improved = True
                            break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                update_best(h_work * s_cd)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
