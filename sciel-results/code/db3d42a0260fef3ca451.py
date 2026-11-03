"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Strategy: Spectral design with flat-magnitude constraint.
1. Work in frequency domain: enforce constant-magnitude DFT (flat spectrum).
2. This simultaneously suppresses all off-zero-lag autoconvolution coefficients.
3. Optimize phase via L-BFGS to minimize max|autoconv| / (sum)^2.
4. Use multiple random phase initializations and keep the best result.
5. Deadline guard to return before timeout.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2 * n - 1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _phase_to_sequence(magnitude, phase):
    """
    Convert magnitude and phase in frequency domain to a real sequence.
    magnitude: array of length n//2+1 (rfft output length)
    phase: array of length n//2+1
    Returns real sequence of length n.
    """
    n = 2 * (len(magnitude) - 1)
    # Construct full complex spectrum
    F = np.zeros(n, dtype=complex)
    F[:len(magnitude)] = magnitude * np.exp(1j * phase)
    # Ensure conjugate symmetry for real output
    # For rfft: F[k] = conj(F[n-k]) for k=1..n/2-1
    # We already have F[0..n/2], need F[n/2+1..n-1]
    if n > 1:
        F[n//2 + 1:] = np.conj(F[1:n//2][::-1])
    # Inverse FFT
    f = np.fft.irfft(F, n)
    return f


def _sequence_to_phase(f):
    """Extract phase from a real sequence's DFT."""
    n = len(f)
    F = np.fft.rfft(f)
    magnitude = np.abs(F)
    phase = np.angle(F)
    return magnitude, phase


def _objective_and_grad(x, target_magnitude):
    """
    Objective: max|autoconv| / (sum)^2 for sequence derived from phases x.
    x: phase array of length n//2+1
    target_magnitude: fixed magnitude spectrum
    """
    f = _phase_to_sequence(target_magnitude, x)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return 1e10, None
    ac = _autoconv_fft(f)
    score = np.max(np.abs(ac)) / (s * s)
    return score, None


def _spectral_optimize(magnitude, initial_phase, max_iter=2000, tol=1e-8):
    """
    Optimize phase to minimize max|autoconv|/(sum)^2 with fixed magnitude.
    Uses L-BFGS-B with numerical gradient (since analytical is complex).
    """
    n_phase = len(initial_phase)
    
    def obj(x):
        val, _ = _objective_and_grad(x, magnitude)
        return val
    
    # L-BFGS-B with numerical gradient
    res = minimize(
        obj, initial_phase, method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6, 'maxls': 50}
    )
    return res.x, res.fun


def _random_phase_init(n_freq, rng):
    """Generate random phase initialization with conjugate symmetry constraints."""
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    # For real sequence, phase[0] and phase[n/2] (if n even) must be 0 or pi
    phase[0] = 0.0
    if n_freq > 1:
        n = 2 * (n_freq - 1)
        if n % 2 == 0:  # n/2 is an integer frequency
            phase[n//2] = 0.0
    return phase


def _try_several_phases(magnitude, n_trials=5, max_iter=500):
    """Try multiple random phase initializations and return best."""
    n_freq = len(magnitude)
    rng = np.random.RandomState(42)
    
    best_phase = None
    best_score = np.inf
    
    for trial in range(n_trials):
        if _remaining() < 10.0:
            break
        init_phase = _random_phase_init(n_freq, rng)
        opt_phase, opt_score = _spectral_optimize(magnitude, init_phase, max_iter=max_iter)
        if opt_score < best_score:
            best_score = opt_score
            best_phase = opt_phase.copy()
    
    if best_phase is not None:
        f = _phase_to_sequence(magnitude, best_phase)
        # Round to nearest ±1
        f_rounded = np.sign(f)
        f_rounded[f_rounded == 0] = 1.0
        return f_rounded, _score(f_rounded)
    return None, np.inf


def _refine_time_domain(f, max_iters=5000):
    """Local search: single-bit flips to improve score."""
    cur_score = _score(f)
    best_f = f.copy()
    best_score = cur_score
    
    for _ in range(max_iters):
        if _remaining() < 5.0:
            break
        idx = np.random.randint(N)
        old_val = f[idx]
        f[idx] = -f[idx]
        new_score = _score(f)
        if new_score < cur_score:
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
        else:
            f[idx] = old_val
    
    return best_f, best_score


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    # Start with a good initial sequence
    rng = np.random.RandomState(12345)
    f_init = rng.choice([-1.0, 1.0], size=N)
    
    # Get its magnitude spectrum
    magnitude, _ = _sequence_to_phase(f_init)
    
    # Try spectral optimization with multiple phase initializations
    f_spectral, score_spectral = _try_several_phases(magnitude, n_trials=5, max_iter=300)
    
    # Also try with a few different initial magnitudes
    best_f = f_init.copy()
    best_score = _score(f_init)
    
    if f_spectral is not None and score_spectral < best_score:
        best_f = f_spectral
        best_score = score_spectral
    
    # Try with a few different random initial sequences for magnitude
    for trial in range(3):
        if _remaining() < 15.0:
            break
        f_rand = rng.choice([-1.0, 1.0], size=N)
        mag_rand, _ = _sequence_to_phase(f_rand)
        f_try, score_try = _try_several_phases(mag_rand, n_trials=3, max_iter=200)
        if f_try is not None and score_try < best_score:
            best_f = f_try
            best_score = score_try
    
    # Final time-domain refinement
    if _remaining() > 10.0:
        best_f, best_score = _refine_time_domain(best_f, max_iters=3000)
    
    return best_f
