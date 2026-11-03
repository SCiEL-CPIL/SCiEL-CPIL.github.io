"""
Spectral-domain optimization for signed step functions minimizing
max_k |c_k| / (sum x_i)^2 where c = x * x (linear autoconvolution).

Key insight: In frequency domain, the autoconvolution's DFT is X(k)^2.
If we make |X(k)| flat (constant magnitude), the off-diagonal autoconvolution
terms are minimized. We optimize the phase spectrum to minimize the max
absolute autoconvolution while keeping the magnitude spectrum flat.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N - 1
    X = np.fft.fft(x, n=2*N)
    C = np.fft.ifft(X * X).real[:n]
    return C

def _evaluate(x):
    """Compute the raw objective value: max|c_k| / (sum x_i)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s**2)

def _spectral_design(mag_target, phases, N):
    """
    Construct a real signal from a target magnitude spectrum and phases.
    
    mag_target: array of length N//2+1 with desired magnitudes for k=0..N//2
    phases: array of length N//2+1 with desired phases for k=0..N//2
    Returns real signal of length N.
    """
    X = np.zeros(N, dtype=complex)
    half = N // 2
    # DC component
    X[0] = mag_target[0] * np.exp(1j * phases[0])
    # Nyquist component (if N is even)
    if N % 2 == 0:
        X[half] = mag_target[half] * np.exp(1j * phases[half])
    # Other positive frequencies
    for k in range(1, half):
        X[k] = mag_target[k] * np.exp(1j * phases[k])
        X[N-k] = np.conj(X[k])
    x = np.fft.ifft(X).real
    return x

def _phase_optimize(mag_target, phase_init, max_iters=200, deadline=None):
    """
    Optimize phases to minimize max|autoconv| given a fixed magnitude spectrum.
    Uses L-BFGS-B on the phase variables.
    """
    half = N // 2
    n_params = half + 1  # phases for k=0..N//2
    
    def objective(params):
        if deadline is not None and time.time() > deadline:
            return 1e10
        phases = params
        x = _spectral_design(mag_target, phases, N)
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        c = _autoconv(x)
        return np.max(np.abs(c)) / (s**2)
    
    def gradient(params):
        # Numerical gradient for phase optimization
        eps = 1e-5
        grad = np.zeros(n_params)
        f0 = objective(params)
        for i in range(n_params):
            params_p = params.copy()
            params_p[i] += eps
            f_p = objective(params_p)
            grad[i] = (f_p - f0) / eps
        return grad
    
    result = minimize(
        objective,
        phase_init,
        method='L-BFGS-B',
        jac=gradient,
        bounds=[(-np.pi, np.pi)] * n_params,
        options={'maxiter': max_iters, 'ftol': 1e-8, 'gtol': 1e-5}
    )
    
    return result.x, result.fun

def _flat_magnitude_optimize(phase_init, deadline=None):
    """
    Optimize with flat magnitude spectrum (all magnitudes equal).
    This enforces orthogonality in frequency domain.
    """
    half = N // 2
    mag_target = np.ones(half + 1)
    mag_target[0] = 0.0  # DC = 0 for zero-mean signal
    
    phases, obj = _phase_optimize(mag_target, phase_init, max_iters=300, deadline=deadline)
    
    x = _spectral_design(mag_target, phases, N)
    return x, obj

def _random_phases():
    """Generate random initial phases."""
    half = N // 2
    return np.random.uniform(-np.pi, np.pi, size=half + 1)

def _structured_phases(mode):
    """Generate structured initial phases."""
    half = N // 2
    phases = np.zeros(half + 1)
    if mode == 0:  # Linear phase
        phases = np.linspace(0, 2*np.pi, half + 1)
    elif mode == 1:  # Quadratic phase
        k = np.arange(half + 1)
        phases = (k * k * 0.1) % (2 * np.pi)
    elif mode == 2:  # Alternating
        phases = np.array([0 if i % 2 == 0 else np.pi for i in range(half + 1)])
    else:  # Random with structure
        base = np.random.uniform(-np.pi, np.pi, size=half + 1)
        phases = base + 0.1 * np.arange(half + 1)
    return phases

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    Uses spectral-domain optimization with flat magnitude constraint.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    np.random.seed(42)
    
    # Baseline: constant sequence
    baseline = np.ones(N)
    best_result = baseline.copy()
    best_val = _evaluate(baseline)
    
    # Try multiple phase initializations with flat magnitude spectrum
    phase_init_modes = [
        _random_phases,
        lambda: _structured_phases(0),
        lambda: _structured_phases(1),
        lambda: _structured_phases(2),
        lambda: _structured_phases(3),
    ]
    
    for mode_idx, init_fn in enumerate(phase_init_modes):
        if time.time() > deadline:
            break
        
        phase_init = init_fn()
        x_candidate, obj_val = _flat_magnitude_optimize(phase_init, deadline=deadline)
        
        if time.time() > deadline:
            break
        
        raw_val = _evaluate(x_candidate)
        if raw_val < best_val:
            best_result = x_candidate.copy()
            best_val = raw_val
    
    # Additional restarts with different random seeds
    for seed in [123, 456, 789]:
        if time.time() > deadline - 10:
            break
        
        np.random.seed(seed)
        phase_init = _random_phases()
        x_candidate, obj_val = _flat_magnitude_optimize(phase_init, deadline=deadline)
        
        if time.time() > deadline:
            break
        
        raw_val = _evaluate(x_candidate)
        if raw_val < best_val:
            best_result = x_candidate.copy()
            best_val = raw_val
    
    # Try non-flat magnitude: optimize magnitudes as well
    if time.time() < deadline - 20:
        # Start from best result, extract its spectrum, then optimize
        x_best = best_result.copy()
        s = np.sum(x_best)
        if abs(s) > 1e-12:
            x_norm = x_best / s
            X = np.fft.fft(x_norm)
            half = N // 2
            mag_current = np.abs(X[:half+1])
            phase_current = np.angle(X[:half+1])
            
            # Optimize phases with current magnitudes
            def _obj_with_mag(params):
                if deadline is not None and time.time() > deadline:
                    return 1e10
                phases = params
                x = _spectral_design(mag_current, phases, N)
                s = np.sum(x)
                if abs(s) < 1e-12:
                    return 1e10
                c = _autoconv(x)
                return np.max(np.abs(c)) / (s**2)
            
            def _grad_with_mag(params):
                eps = 1e-5
                grad = np.zeros(len(params))
                f0 = _obj_with_mag(params)
                for i in range(len(params)):
                    p = params.copy()
                    p[i] += eps
                    f_p = _obj_with_mag(p)
                    grad[i] = (f_p - f0) / eps
                return grad
            
            result = minimize(
                _obj_with_mag,
                phase_current,
                method='L-BFGS-B',
                jac=_grad_with_mag,
                bounds=[(-np.pi, np.pi)] * (half + 1),
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            x_opt = _spectral_design(mag_current, result.x, N)
            raw_val = _evaluate(x_opt)
            if raw_val < best_val:
                best_result = x_opt.copy()
                best_val = raw_val
    
    # Ensure valid output
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result

# EVOLVE-BLOCK-END
