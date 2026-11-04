import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    return np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]


def _metric(x):
    """max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)


def _phase_optimize(x0, n_iter=500, lr=0.05, deadline=None):
    """
    Optimize phases in frequency domain to minimize max|autoconv|/(sum x)^2.
    We work with the full complex FFT, fixing magnitudes and optimizing phases.
    """
    # Compute FFT of initial signal
    X = np.fft.rfft(x0, _FFT_SIZE)
    # Store magnitudes (fixed) and initialize phases
    mags = np.abs(X)
    # Ensure DC component is positive (sum x > 0)
    phase_dc = 0.0
    phases = np.angle(X)
    # Keep DC phase fixed at 0
    phases[0] = 0.0
    
    best_x = x0.copy()
    best_metric = _metric(x0)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Reconstruct signal from current phases
        X_cur = mags * np.exp(1j * phases)
        # Ensure DC is real and positive
        X_cur[0] = mags[0]
        x_cur = np.fft.irfft(X_cur, _FFT_SIZE)[:N]
        
        # Compute metric
        s = np.sum(x_cur)
        if abs(s) < 1e-12:
            phases += 0.01 * np.random.randn(len(phases))
            phases[0] = 0.0
            continue
        
        c = _autoconv_lin(x_cur)
        metric = float(np.max(np.abs(c))) / (s * s)
        
        if metric < best_metric:
            best_metric = metric
            best_x = x_cur.copy()
        
        # Subgradient step on phases
        # Gradient of max|c| w.r.t. phases: use sign of c at argmax
        k_max = np.argmax(np.abs(c))
        # The autoconvolution c = IFFT(X^2), so dc/dphase involves X
        # Use simple heuristic: perturb phases in direction that reduces peak
        # Compute gradient numerically for a few coordinates
        grad_phases = np.zeros_like(phases)
        # For efficiency, use a few random directions
        n_probe = min(10, len(phases))
        for _ in range(n_probe):
            idx = np.random.randint(0, len(phases))
            if idx == 0:
                continue
            eps = 0.01
            phases[idx] += eps
            X_plus = mags * np.exp(1j * phases)
            X_plus[0] = mags[0]
            x_plus = np.fft.irfft(X_plus, _FFT_SIZE)[:N]
            s_plus = np.sum(x_plus)
            if abs(s_plus) > 1e-12:
                c_plus = _autoconv_lin(x_plus)
                m_plus = float(np.max(np.abs(c_plus))) / (s_plus * s_plus)
            else:
                m_plus = metric + 1.0
            phases[idx] -= 2 * eps
            X_minus = mags * np.exp(1j * phases)
            X_minus[0] = mags[0]
            x_minus = np.fft.irfft(X_minus, _FFT_SIZE)[:N]
            s_minus = np.sum(x_minus)
            if abs(s_minus) > 1e-12:
                c_minus = _autoconv_lin(x_minus)
                m_minus = float(np.max(np.abs(c_minus))) / (s_minus * s_minus)
            else:
                m_minus = metric + 1.0
            phases[idx] += eps
            grad_phases[idx] = (m_plus - m_minus) / (2 * eps)
        
        # Update phases
        phases -= lr * grad_phases
        phases[0] = 0.0
    
    return best_x, best_metric


def construct():
    """
    Minimize max|autoconv| / (sum x)^2 using phase optimization in frequency domain.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Start with constant sequence
    x0 = np.ones(N)
    best_x = x0.copy()
    best_metric = _metric(x0)
    
    # Try phase optimization from constant start
    x_opt, m_opt = _phase_optimize(x0, n_iter=300, lr=0.03, deadline=deadline)
    if m_opt < best_metric:
        best_metric = m_opt
        best_x = x_opt
    
    # Try from random starts
    np.random.seed(42)
    for trial in range(3):
        if time.time() > deadline - 10:
            break
        x_rand = np.random.randn(N)
        x_rand -= np.mean(x_rand)  # Center to avoid DC issues
        x_rand += 1.0  # Ensure positive sum
        x_opt, m_opt = _phase_optimize(x_rand, n_iter=200, lr=0.03, deadline=deadline)
        if m_opt < best_metric:
            best_metric = m_opt
            best_x = x_opt
    
    # Final polish: try local search on best_x
    if time.time() < deadline - 15:
        x_polish, m_polish = _phase_optimize(best_x, n_iter=200, lr=0.01, deadline=deadline)
        if m_polish < best_metric:
            best_metric = m_polish
            best_x = x_polish
    
    return best_x
