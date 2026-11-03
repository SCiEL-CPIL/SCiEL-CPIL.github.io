"""
Spectral-flatness optimization: minimize max|c_k|/(sum x)^2 by flattening
the magnitude spectrum |X(omega)|^2, since autoconvolution c = ifft(|X|^2).

Key structural insight: the autoconvolution depends ONLY on the magnitude
spectrum |X[k]|^2, not on the phase. So we can optimize the magnitudes
directly in frequency domain, then reconstruct a real signal.

Strategy:
1. Parameterize by magnitude spectrum m[k] = |X[k]| with conjugate symmetry.
2. Objective: max|ifft(m^2)_k| / m[0]^2 (normalized).
3. Gradient descent on log-magnitudes for better conditioning.
4. Multiple restarts with different spectral initializations.
5. Final: reconstruct real x, optionally threshold to signs.
"""
import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s * s)

def _spectral_objective(m, temperature):
    """
    Given magnitude spectrum m (length N, with m[N-k]=m[k]),
    compute c = ifft(m^2) and return max|c_k| with soft-max surrogate.
    
    m[0] is the DC component = sum(x) for real x.
    We normalize by m[0]^2 to get the ratio.
    
    Returns: (objective_value, gradient_wrt_log_m)
    """
    # m^2 is the power spectrum
    power = m * m  # |X|^2
    
    # c = ifft(power) - this gives the circular autoconvolution
    # For linear autoconvolution of length N, we need zero-padding
    # But for the spectral approach, circular autoconv is what ifft gives
    # The linear autoconv c_lin has c_lin[k] for k=-(N-1)..(N-1)
    # The circular autoconv c_circ[k] = sum_{j=0}^{N-1} x_j * x_{(j+k) mod N}
    # These differ. Let's use the circular version for spectral optimization
    # and check the linear version at the end.
    
    c = np.fft.ifft(power).real  # length N, circular autoconv
    
    # c[0] is the "zero lag" = sum x_j^2
    # We want to minimize max|c[k]| / c[0]... no, the ratio is max|c_lin| / (sum x)^2
    # For circular: max|c_circ| / (sum x)^2 where sum x = m[0] (DC)
    
    # Actually, let's think about this more carefully.
    # For the linear autoconvolution, c_lin[k] for k=0..N-1 is:
    # c_lin[k] = sum_{j=0}^{N-1-k} x_j * x_{j+k}
    # The full linear autoconv has 2N-1 values.
    # The circular autoconv c_circ[k] = sum_{j=0}^{N-1} x_j * x_{(j+k) mod N}
    # c_circ[k] = c_lin[k] + c_lin[k-N] for k=1..N-1 (with c_lin negative lags)
    
    # For our purpose, let's optimize the circular autoconv and verify with linear.
    # The max of circular autoconv is >= max of linear (since it adds terms).
    # Actually no, circular can be smaller in some cases.
    
    # Let's use: objective = max|c_circ| / m[0]^2
    # where m[0] = X[0] = sum(x) (for real x, X[0] is real)
    
    abs_c = np.abs(c)
    m0_sq = m[0] * m[0]
    if m0_sq < 1e-15:
        return np.inf, np.zeros(N)
    
    # Soft-max of abs_c / m0_sq
    # Normalize first
    norm_c = abs_c / m0_sq
    max_val = np.max(norm_c)
    if max_val < 1e-15:
        return max_val, np.zeros(N)
    
    exp_vals = np.exp((norm_c - max_val) / temperature)
    sum_exp = np.sum(exp_vals)
    soft_max = max_val + temperature * np.log(sum_exp)
    
    # Gradient: d(soft_max)/d(c_k) = sign(c_k) * exp_vals[k] / sum_exp / m0_sq
    # d(c_k)/d(m_j): c = ifft(m^2), so c_k = (1/N) * sum_j m[j]^2 * exp(2*pi*i*j*k/N)
    # d(c_k)/d(m_j) = (2*m[j]/N) * exp(2*pi*i*j*k/N)
    # This is complex. Let's use FFT-based gradient.
    
    # c_k = (1/N) * sum_j P[j] * exp(2*pi*i*j*k/N) where P[j] = m[j]^2
    # d(c_k)/d(P[j]) = (1/N) * exp(2*pi*i*j*k/N)
    # d(P[j])/d(m[j]) = 2*m[j]
    # d(soft_max)/d(P[j]) = sum_k d(soft_max)/d(c_k) * d(c_k)/d(P[j])
    # = sum_k [sign(c_k) * exp_vals[k] / sum_exp / m0_sq] * (1/N) * exp(2*pi*i*j*k/N)
    
    # Let w_k = sign(c_k) * exp_vals[k] / sum_exp / m0_sq
    # d(soft_max)/d(P[j]) = (1/N) * sum_k w_k * exp(2*pi*i*j*k/N) = (1/N) * FFT(w)[j]
    
    w = np.sign(c) * exp_vals / sum_exp / m0_sq
    grad_P = np.fft.fft(w) / N  # d(soft_max)/d(P[j])
    
    # d(soft_max)/d(m[j]) = grad_P[j] * 2 * m[j]
    grad_m = grad_P * 2.0 * m
    
    # We want gradient w.r.t. log(m) for better conditioning
    # d(soft_max)/d(log m[j]) = grad_m[j] * m[j]
    grad_log_m = grad_m * m
    
    return soft_max, grad_log_m.real  # Should be real by symmetry

def _spectral_descent(m0, max_iters=5000, lr_init=0.1, lr_min=1e-5,
                      temp_init=1.0, temp_min=0.05, deadline=None):
    """
    Gradient descent on log-magnitude spectrum.
    m0: initial magnitude spectrum (length N, with m[N-k]=m[k], m[0]>0)
    """
    # Work with log magnitudes for positivity
    # m[0] is fixed (DC component = sum x). We'll normalize so m[0]=1.
    log_m = np.log(np.maximum(m0, 1e-10)).copy()
    # Fix DC: log_m[0] = 0 (m[0] = 1)
    log_m[0] = 0.0
    
    best_obj = np.inf
    best_log_m = log_m.copy()
    
    temperature = temp_init
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        m = np.exp(log_m)
        # Enforce conjugate symmetry: m[N-k] = m[k]
        # For real signals, |X[N-k]| = |X[k]|
        # Average to enforce
        m_sym = np.zeros(N)
        m_sym[0] = m[0]
        if N % 2 == 0:
            m_sym[N//2] = m[N//2]
        for k in range(1, N//2 + (1 if N % 2 == 1 else 0)):
            m_sym[k] = 0.5 * (m[k] + m[N-k])
            m_sym[N-k] = m_sym[k]
        m = m_sym
        log_m = np.log(np.maximum(m, 1e-10))
        log_m[0] = 0.0
        
        obj, grad_log_m = _spectral_objective(m, temperature)
        
        if obj < best_obj:
            best_obj = obj
            best_log_m = log_m.copy()
        
        # Gradient step
        # Zero out the DC gradient (m[0] is fixed)
        grad_log_m[0] = 0.0
        # Enforce symmetry in gradient
        g_sym = np.zeros(N)
        g_sym[0] = 0
        if N % 2 == 0:
            g_sym[N//2] = grad_log_m[N//2]
        for k in range(1, N//2 + (1 if N % 2 == 1 else 0)):
            g_sym[k] = 0.5 * (grad_log_m[k] + grad_log_m[N-k])
            g_sym[N-k] = g_sym[k]
        
        log_m = log_m - lr * g_sym
        log_m[0] = 0.0  # Keep DC fixed
        
        # Anneal
        if it > 0 and it % 50 == 0:
            temperature = max(temp_min, temperature * 0.97)
            lr = max(lr_min, lr * 0.995)
    
    return best_log_m, best_obj

def _reconstruct_real_x(log_m, phase_seed=None):
    """
    Reconstruct a real signal from log-magnitude spectrum.
    Choose phases to make x real.
    """
    m = np.exp(log_m)
    # Enforce symmetry
    m_sym = np.zeros(N)
    m_sym[0] = m[0]
    if N % 2 == 0:
        m_sym[N//2] = m[N//2]
    for k in range(1, N//2 + (1 if N % 2 == 1 else 0)):
        m_sym[k] = 0.5 * (m[k] + m[N-k])
        m_sym[N-k] = m_sym[k]
    m = m_sym
    
    # For real x, we need X[N-k] = conj(X[k])
    # So X[k] = m[k] * exp(i*phi[k]) and X[N-k] = m[k] * exp(-i*phi[k])
    # We can choose arbitrary phases phi[1..N/2-1]
    # To get a "nice" signal, use random phases or zero phases
    
    if phase_seed is not None:
        np.random.seed(phase_seed)
    
    X = np.zeros(N, dtype=complex)
    X[0] = m[0]  # DC, real and positive
    if N % 2 == 0:
        X[N//2] = m[N//2]  # Nyquist, real
    for k in range(1, N//2 + (1 if N % 2 == 1 else 0)):
        if k == N//2 and N % 2 == 0:
            continue
        if phase_seed is not None:
            phi = np.random.uniform(-np.pi, np.pi)
        else:
            phi = 0.0
        X[k] = m[k] * np.exp(1j * phi)
        X[N-k] = m[k] * np.exp(-1j * phi)
    
    x = np.fft.ifft(X).real
    return x

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Baseline: constant sequence
    x_baseline = np.ones(N)
    best_result = x_baseline.copy()
    best_val = _eval_ratio(x_baseline)  # = 1/N
    
    np.random.seed(42)
    
    # Multiple spectral restarts
    num_restarts = 20
    max_iters_per_restart = 3000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different spectral initializations
        if r == 0:
            # Flat spectrum (all magnitudes equal) -> delta autoconv
            m0 = np.ones(N)
        elif r == 1:
            # Slightly perturbed flat
            m0 = np.ones(N) + 0.1 * np.random.randn(N)
            m0 = np.abs(m0)
        elif r == 2:
            # Low-pass emphasis
            freqs = np.fft.fftfreq(N)
            m0 = 1.0 / (1.0 + 5.0 * np.abs(freqs))
        elif r == 3:
            # High-pass emphasis  
            freqs = np.fft.fftfreq(N)
            m0 = 1.0 + 2.0 * np.abs(freqs)
        elif r == 4:
            # Band-pass
            freqs = np.fft.fftfreq(N)
            m0 = np.exp(-10.0 * (np.abs(freqs) - 0.2)**2)
        elif r == 5:
            # Random magnitudes
            m0 = np.abs(np.random.randn(N)) + 0.1
        else:
            # Mix of strategies
            strategy = r % 5
            if strategy == 0:
                m0 = np.ones(N) + 0.2 * np.random.randn(N)
                m0 = np.abs(m0)
            elif strategy == 1:
                freqs = np.fft.fftfreq(N)
                m0 = 1.0 / (1.0 + 3.0 * np.abs(freqs))
            elif strategy == 2:
                m0 = np.abs(np.random.randn(N)) + 0.5
            elif strategy == 3:
                freqs = np.fft.fftfreq(N)
                m0 = np.exp(-5.0 * np.abs(freqs)**2)
            else:
                m0 = np.ones(N)
                # Random dips
                for _ in range(5):
                    idx = np.random.randint(1, N-1)
                    m0[idx] *= 0.5
                    m0[N-idx] *= 0.5
        
        # Enforce conjugate symmetry in initial m
        m0_sym = np.zeros(N)
        m0_sym[0] = m0[0]
        if N % 2 == 0:
            m0_sym[N//2] = m0[N//2]
        for k in range(1, N//2 + (1 if N % 2 == 1 else 0)):
            m0_sym[k] = 0.5 * (m0[k] + m0[N-k])
            m0_sym[N-k] = m0_sym[k]
        m0 = np.maximum(m0_sym, 1e-10)
        
        # Run spectral descent
        best_log_m, spec_obj = _spectral_descent(
            m0,
            max_iters=max_iters_per_restart,
            lr_init=0.1,
            lr_min=1e-5,
            temp_init=1.0,
            temp_min=0.05,
            deadline=deadline
        )
        
        # Reconstruct real signal with different phase choices
        for phase_seed in [None, r, r + 100, r + 200]:
            if time.time() > deadline:
                break
            
            x = _reconstruct_real_x(best_log_m, phase_seed=phase_seed)
            
            # Normalize so sum(x) = 1 (scale invariance)
            s = np.sum(x)
            if abs(s) < 1e-12:
                continue
            x = x / s
            
            val = _eval_ratio(x)
            
            if val < best_val:
                best_val = val
                best_result = x.copy()
        
        # Also try sign-thresholded version
        if time.time() > deadline:
            break
        sign_x = np.sign(best_result)
        sign_x[sign_x == 0] = 1.0
        s_sign = np.sum(sign_x)
        if abs(s_sign) > 1e-12:
            val_sign = _eval_ratio(sign_x)
            if val_sign < best_val:
                best_val = val_sign
                best_result = sign_x.copy()
    
    # Final validation
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result

# EVOLVE-BLOCK-END
