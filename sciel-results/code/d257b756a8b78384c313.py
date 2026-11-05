"""Combined program: sparse Fourier basis + smooth max surrogate + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _value(h):
    """Compute the evaluator's value."""
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    ac = np.convolve(h, h)
    return abs(2 * N * np.max(ac) / s ** 2)


def _smooth_max(x, beta=100.0):
    """Log-sum-exp smooth approximation of max(x)."""
    m = np.max(x)
    if m < -700:
        return m
    return m + np.log(np.sum(np.exp(x - m))) / beta


def _fourier_to_h(coeffs):
    """Convert complex Fourier coefficients to real step heights.
    
    coeffs[0] is the DC term (real), coeffs[1:M] are complex amplitudes for
    harmonics 1..M-1. We use h[k] = sum_{m=0}^{M-1} Re(c[m] * exp(2*pi*i*m*k/N)).
    To ensure h is real, we enforce conjugate symmetry: c[m] for m > N/2 is
    determined by c[N-m]. But since we only use first M harmonics (M << N/2),
    we just take the real part of the sum.
    """
    M = len(coeffs)
    k = np.arange(N)
    h = np.zeros(N)
    # DC term
    if M > 0:
        h += np.real(coeffs[0])
    # Higher harmonics
    for m in range(1, M):
        c = coeffs[m]
        h += np.real(c * np.exp(2j * np.pi * m * k / N))
    return h


def _h_to_fourier(h, M):
    """Extract first M Fourier coefficients from h."""
    k = np.arange(N)
    coeffs = np.zeros(M, dtype=complex)
    coeffs[0] = np.sum(h) / N
    for m in range(1, M):
        coeffs[m] = np.sum(h * np.exp(-2j * np.pi * m * k / N)) / N
    return coeffs


def _objective_fourier(coeffs, beta=100.0, M=50):
    """Smooth surrogate objective in Fourier coefficient space."""
    h = _fourier_to_h(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    ac = np.convolve(h, h)
    sm = _smooth_max(ac, beta)
    val = 2.0 * N * sm / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val


def _objective_direct(h, beta=100.0):
    """Smooth surrogate objective in direct space."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    ac = np.convolve(h, h)
    sm = _smooth_max(ac, beta)
    val = 2.0 * N * sm / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val


def _make_sine_init(freq_idx, phase=0.0):
    """Create a sine-wave initialization with given frequency index."""
    k = np.arange(N)
    h = np.sin(2 * np.pi * freq_idx * k / N + phase)
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = h + 0.01
        s = np.sum(h)
    return h / s * np.sqrt(N)


def _spike_candidate():
    """Two-spike + small negative tail candidate."""
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h


def construct():
    t_start = time.time()
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # Should be 2.0
    
    # Phase 1: Quick evaluation of diverse initializations
    candidates = []
    
    # Sine-based initializations
    freq_candidates = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
    phases = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for freq in freq_candidates:
        for phase in phases:
            h = _make_sine_init(freq, phase)
            candidates.append(h)
    
    # Spike-based candidates
    candidates.append(_spike_candidate())
    for K in (50, 100, 200, 300, 399):
        for eps in (0.0005, 0.001, 0.002, 0.005):
            if K - 2 <= 0:
                continue
            h = np.zeros(N)
            h[0] = 0.5
            h[1] = 0.5
            h[2:K] = -eps
            candidates.append(h)
    
    # Evaluate all candidates
    init_list = []
    for h in candidates:
        v = _value(h)
        if np.isfinite(v):
            init_list.append((v, h))
    
    init_list.sort(key=lambda x: x[0])
    
    # Update best from candidates
    if init_list:
        best_val = init_list[0][0]
        best_h = init_list[0][1].copy()
    
    # Phase 2: L-BFGS optimization from top initializations in direct space
    for i, (v0, h0) in enumerate(init_list[:6]):
        if time.time() - t_start > DEADLINE - 30:
            break
        
        beta = 80.0 + i * 40.0
        try:
            res = minimize(
                _objective_direct, h0.copy(), args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            v_opt = _value(res.x)
            if v_opt < best_val and np.isfinite(v_opt):
                best_val = v_opt
                best_h = res.x.copy()
        except Exception:
            continue
    
    # Phase 3: Fourier-space optimization
    # Use the best candidate found so far, project to Fourier space, optimize
    if time.time() - t_start < DEADLINE - 20:
        M = 30  # Number of Fourier harmonics to optimize
        coeffs = _h_to_fourier(best_h, M)
        
        for beta in (100.0, 200.0, 400.0):
            if time.time() - t_start > DEADLINE - 15:
                break
            try:
                res = minimize(
                    _objective_fourier, coeffs.copy(), args=(beta, M),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                h_opt = _fourier_to_h(res.x)
                v_opt = _value(h_opt)
                if v_opt < best_val and np.isfinite(v_opt):
                    best_val = v_opt
                    best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 4: High-precision polish in direct space
    if time.time() - t_start < DEADLINE - 10:
        try:
            beta_high = 600.0
            res = minimize(
                _objective_direct, best_h.copy(), args=(beta_high,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            v_polish = _value(res.x)
            if v_polish < best_val and np.isfinite(v_polish):
                best_val = v_polish
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Local random refinement
    if time.time() - t_start < DEADLINE - 3:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() - t_start > DEADLINE - 3:
                break
            h = best_h + rng.normal(0.0, 0.005, N)
            v = _value(h)
            if v < best_val and np.isfinite(v):
                best_val = v
                best_h = h.copy()
    
    # Final check
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return best_h


# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
