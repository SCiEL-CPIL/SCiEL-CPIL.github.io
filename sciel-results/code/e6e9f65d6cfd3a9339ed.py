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

def _soft_max_abs(c, beta=50.0):
    """
    Smooth approximation of max|c_k| using log-sum-exp.
    As beta -> inf, this approaches the true max|c_k|.
    """
    ac = np.abs(c)
    m = np.max(ac)
    if m < 1e-15:
        return 0.0
    # log-sum-exp trick for numerical stability
    shifted = ac / m
    log_sum = np.log(np.sum(np.exp(beta * (shifted - 1.0))))
    return m * (1.0 + log_sum / beta)

def _soft_objective(x, beta=50.0):
    """
    Smooth surrogate for max|c_k| / (sum x_i)^2.
    Uses soft-max of |c_k| instead of hard max.
    """
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    smax = _soft_max_abs(c, beta)
    return smax / (s**2)

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
    Optimize phases to minimize soft-max|autoconv| given a fixed magnitude spectrum.
    Uses L-BFGS-B on the phase variables with a smooth surrogate objective.
    """
    half = N // 2
    n_params = half + 1  # phases for k=0..N//2
    
    # Adaptive beta: start moderate, increase toward hard max
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    def objective(params, beta=50.0):
        if deadline is not None and time.time() > deadline:
            return 1e10
        phases = params
        x = _spectral_design(mag_target, phases, N)
        return _soft_objective(x, beta)
    
    def gradient(params, beta=50.0):
        # Numerical gradient for phase optimization
        eps = 1e-5
        grad = np.zeros(n_params)
        f0 = objective(params, beta)
        for i in range(n_params):
            params_p = params.copy()
            params_p[i] += eps
            f_p = objective(params_p, beta)
            grad[i] = (f_p - f0) / eps
        return grad
    
    best_phases = phase_init.copy()
    best_val = _evaluate(_spectral_design(mag_target, phase_init, N))
    
    # Multi-stage optimization with increasing beta
    for beta in beta_schedule:
        if deadline is not None and time.time() > deadline:
            break
        
        iters_for_stage = max(30, max_iters // len(beta_schedule))
        
        result = minimize(
            lambda p: objective(p, beta),
            best_phases,
            method='L-BFGS-B',
            jac=lambda p: gradient(p, beta),
            bounds=[(-np.pi, np.pi)] * n_params,
            options={'maxiter': iters_for_stage, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        x_candidate = _spectral_design(mag_target, result.x, N)
        raw_val = _evaluate(x_candidate)
        if raw_val < best_val:
            best_phases = result.x.copy()
            best_val = raw_val
    
    return best_phases, best_val

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
    for seed in [123, 456, 789, 1011, 2024]:
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
    
    # Try non-flat magnitude: optimize magnitudes as well with soft surrogate
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
            
            # Optimize phases with current magnitudes using soft surrogate
            def _obj_with_mag(params, beta=50.0):
                if deadline is not None and time.time() > deadline:
                    return 1e10
                phases = params
                x = _spectral_design(mag_current, phases, N)
                return _soft_objective(x, beta)
            
            def _grad_with_mag(params, beta=50.0):
                eps = 1e-5
                grad = np.zeros(len(params))
                f0 = _obj_with_mag(params, beta)
                for i in range(len(params)):
                    p = params.copy()
                    p[i] += eps
                    f_p = _obj_with_mag(p, beta)
                    grad[i] = (f_p - f0) / eps
                return grad
            
            # Multi-stage with increasing beta
            for beta in [20.0, 50.0, 100.0]:
                if time.time() > deadline - 5:
                    break
                
                result = minimize(
                    lambda p, b=beta: _obj_with_mag(p, b),
                    phase_current,
                    method='L-BFGS-B',
                    jac=lambda p, b=beta: _grad_with_mag(p, b),
                    bounds=[(-np.pi, np.pi)] * (half + 1),
                    options={'maxiter': 80, 'ftol': 1e-10}
                )
                
                x_opt = _spectral_design(mag_current, result.x, N)
                raw_val = _evaluate(x_opt)
                if raw_val < best_val:
                    best_result = x_opt.copy()
                    best_val = raw_val
                    phase_current = result.x.copy()
    
    # Ensure valid output
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result

# EVOLVE-BLOCK-END
