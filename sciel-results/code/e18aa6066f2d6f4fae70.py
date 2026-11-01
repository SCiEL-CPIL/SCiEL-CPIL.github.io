"""Spectral shaping: minimize autoconvolution peak via log-sum-exp, threshold to ±1."""
import numpy as np
import time

N = 400
DEADLINE = 145.0  # wall-clock seconds, safety margin under 150s kill
_t0 = time.time()

def _elapsed():
    return time.time() - _t0

def _autoconv_peak(s):
    """Compute the peak magnitude of the (circular) autoconvolution of s."""
    # s has length N; autoconvolution via FFT
    S = np.fft.fft(s)
    ac = np.fft.ifft(S * np.conj(S)).real
    return np.max(np.abs(ac))

def _logsumexp(x, beta=1.0):
    """Smooth approximation of max(x) via (1/beta)*log(sum(exp(beta*x)))."""
    x = np.asarray(x)
    m = np.max(x)
    if beta * (m - m) > 0:  # numerical guard
        return m
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _score_step(s):
    """Score = upper bound on C3 for a ±1 step function s. Lower is better."""
    # C3 upper bound relates to the peak of |s * s| (autoconvolution)
    # For a step function, the autoconvolution peak is the key quantity.
    return _autoconv_peak(s)

def _threshold_signal(x, threshold=None):
    """Map continuous signal to ±1 step function."""
    if threshold is None:
        threshold = np.median(x)
    return np.where(x >= threshold, 1.0, -1.0)

def _optimize_spectral(num_steps, beta_start, beta_end, lr):
    """
    Optimize a continuous signal x (length N) to minimize the smooth peak of its
    autoconvolution, using gradient descent on the log-sum-exp objective.
    Returns the best ±1 step function found and its score.
    """
    # Initialize: random continuous signal
    np.random.seed(42)
    x = np.random.randn(N)
    
    # Frequency domain: work with complex spectrum for efficiency
    # Objective: LSE of |ifft(S * conj(S))| where S = fft(x)
    # Gradient via chain rule
    
    best_s = None
    best_score = float('inf')
    
    step = 0
    while _elapsed() < DEADLINE:
        beta = beta_start + (beta_end - beta_start) * (step / max(num_steps, 1))
        
        # Compute FFT
        S = np.fft.fft(x)
        ac = np.fft.ifft(S * np.conj(S)).real
        ac_abs = np.abs(ac)
        
        # Log-sum-exp objective (smooth max)
        m = np.max(ac_abs)
        lse = m + np.log(np.sum(np.exp(beta * (ac_abs - m)))) / beta
        
        # Gradient of LSE w.r.t. x
        # d/dx LSE = dLSE/dac * dac/dx
        # dac = ifft(S * conj(S))
        # d(ac_i)/dx_j: ac = ifft(|S|^2), so dac/dx = 2*Re(ifft(S * conj(dS/dx)))
        # S = fft(x), dS_k/dx_j = exp(-2*pi*i*k*j/N)
        # This is expensive; use the fact that ac = x * x (circular conv)
        # dac_i/dx_j = delta_{i-j} + delta_{i+j} (for circular, with care)
        # Actually for circular autoconv: ac = ifft(|fft(x)|^2)
        # gradient: grad_x = 2 * ifft(fft(x) * conj(fft(2*ac_lse_grad)))... 
        # Let's use: ac = x ⊛ x (circular), so grad_x = 2 * (ac ⊛ x) is wrong.
        # Correct: ac_k = sum_j x_j * x_{k-j mod N}
        # d ac_k / d x_j = x_{k-j} + x_{k+j} (circular)
        # So grad_x_j = sum_k (dLSE/dac_k) * (x_{k-j} + x_{k+j})
        # This is a convolution: grad_x = 2 * (w ⊛ x) where w_k = dLSE/dac_k
        
        # dLSE/dac_k = beta * exp(beta*(ac_abs_k - m)) / sum(exp(beta*(ac_abs - m)))
        # But ac_abs = |ac|, so dLSE/dac_k = beta * exp(beta*(|ac_k|-m)) * sign(ac_k) / denom
        denom = np.sum(np.exp(beta * (ac_abs - m)))
        w = beta * np.exp(beta * (ac_abs - m)) * np.sign(ac) / denom
        
        # grad_x = 2 * circular_conv(w, x)
        W = np.fft.fft(w)
        grad_x = 2.0 * np.real(np.fft.ifft(W * S))
        
        # Gradient descent step
        x -= lr * grad_x
        
        # Periodically threshold and evaluate
        if step % 5 == 0:
            s = _threshold_signal(x)
            sc = _score_step(s)
            if sc < best_score:
                best_score = sc
                best_s = s.copy()
        
        # Occasionally try different threshold
        if step % 20 == 0 and best_s is not None:
            for t in [np.percentile(x, 30), np.percentile(x, 50), np.percentile(x, 70)]:
                s2 = _threshold_signal(x, t)
                sc2 = _score_step(s2)
                if sc2 < best_score:
                    best_score = sc2
                    best_s = s2.copy()
        
        step += 1
    
    return best_s, best_score

def _local_search(s, num_rounds=1000):
    """Flip individual bits to reduce the autoconvolution peak."""
    best_s = s.copy()
    best_score = _score_step(best_s)
    N = len(best_s)
    
    # Precompute FFT for fast evaluation
    for rnd in range(num_rounds):
        if _elapsed() > DEADLINE:
            break
        
        # Randomly pick a position to flip
        idx = np.random.randint(N)
        trial = best_s.copy()
        trial[idx] = -trial[idx]
        sc = _score_step(trial)
        if sc < best_score:
            best_s = trial
            best_score = sc
    
    return best_s, best_score

def construct():
    """Construct a signed step function with N intervals minimizing C3 upper bound."""
    np.random.seed(123)
    
    best_s = None
    best_score = float('inf')
    
    # Phase 1: Spectral optimization with gradient descent
    if _elapsed() < DEADLINE - 20:
        s1, sc1 = _optimize_spectral(
            num_steps=2000,
            beta_start=5.0,
            beta_end=50.0,
            lr=0.01
        )
        if s1 is not None and sc1 < best_score:
            best_s = s1
            best_score = sc1
    
    # Phase 2: Local search refinement
    if best_s is not None and _elapsed() < DEADLINE - 5:
        s2, sc2 = _local_search(best_s, num_rounds=5000)
        if sc2 < best_score:
            best_s = s2
            best_score = sc2
    
    # Fallback: all ones if nothing found
    if best_s is None:
        best_s = np.ones(N)
    
    return best_s
