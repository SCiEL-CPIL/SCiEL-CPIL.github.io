"""Zero-sum game / minimax approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _minimax_iterate(x, q, lr_x, lr_q, iters=50):
    """
    Iterative gradient updates for the zero-sum game.
    x: sequence (length N)
    q: probability distribution over lags (length L = 2N-1)
    Returns improved (x, q).
    
    The payoff is E_q[|ac[k]|] / (sum x)^2 where ac = autoconv(x).
    Designer minimizes over x, adversary maximizes over q.
    """
    L = 2 * N - 1
    x = x.copy()
    q = q.copy()
    
    for it in range(iters):
        if _check_deadline():
            break
        
        # Compute autoconvolution
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            s = 1e-10
        
        s_sq = s * s
        
        # Adversary update: maximize E_q[|ac|] / s_sq
        # Gradient of E_q[|ac|] w.r.t. q is just |ac|
        # Use softmax-like update to keep q as a distribution
        # q_new[k] proportional to exp(|ac[k]| / tau)
        tau = 5.0  # temperature
        log_q = abs_ac / tau
        log_q -= np.max(log_q)
        q_new = np.exp(log_q)
        q_new /= np.sum(q_new)
        
        # Blend
        q = 0.7 * q + 0.3 * q_new
        
        # Designer update: minimize E_q[|ac|] / s_sq w.r.t. x
        # d/dx [sum_k q[k] |ac[k]|] = sum_k q[k] sign(ac[k]) * d(ac[k])/dx
        # ac[k] = sum_i x[i] x[i+k]
        # d(ac[k])/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
        # This is a convolution-like operation.
        
        # More efficiently: grad_x = sum_k q[k] sign(ac[k]) * (x shifted)
        # Let's compute via FFT or direct summation for each j
        
        # grad_x[j] = sum_k q[k] * sign(ac[k]) * (x[j+k] if j+k<N else 0 + x[j-k] if j-k>=0 else 0)
        # This can be written as: grad_x = sum_k q[k]*sign(ac[k]) * (correlation of x with delta_k)
        
        # Using the fact that ac[k] = sum_i x[i]*x[i+k], we can use FFT:
        # Let w[k] = q[k] * sign(ac[k]) for k=0..L-1
        # Then grad_x[j] = sum_k w[k] * (x[j+k] + x[j-k]) (with boundary handling)
        # This is equivalent to a cross-correlation.
        
        # Efficient computation:
        # grad_x = IFFT(FFT(w_extended) * FFT(x)) where w is extended properly
        # Actually, let's just do it directly for clarity, N=400 is small enough
        
        w = q * np.sign(ac)  # length L
        
        # grad_x[j] = sum_{k=0}^{L-1} w[k] * (x[j+k] if 0<=j+k<N else 0 + x[j-k] if 0<=j-k<N else 0)
        # = sum_{k=0}^{N-1-j} w[k] * x[j+k] + sum_{k=0}^{j} w[k] * x[j-k]
        #   + sum_{k=j+1}^{L-1} w[k] * x[j-(k-N)]  [for the second term when k > j]
        
        # Let's use a different approach: 
        # ac[k] = sum_i x[i] x[i+k]
        # d(ac[k])/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
        # 
        # grad_x[j] = sum_k w[k] * (x[j+k]*I(j+k<N) + x[j-k]*I(j-k>=0))
        #
        # Let's compute this using convolution:
        # Part 1: sum_k w[k] * x[j+k] for k=0..N-1-j
        # Part 2: sum_k w[k] * x[j-k] for k=0..j
        
        # Use FFT-based approach:
        # Extend x to length 2N-1 with zeros
        x_ext = np.zeros(L)
        x_ext[:N] = x
        
        # Part 1: cross-correlation of w with x_ext
        # (w * x_ext)[j] = sum_k w[k] * x_ext[j+k]
        # This is a valid cross-correlation
        
        # Actually, let me just compute directly - N=400, L=799, so 400*799 ~ 320k ops, fast enough
        grad_x = np.zeros(N)
        for j in range(N):
            total = 0.0
            # Sum over k where both x[j+k] and x[j-k] are valid
            for k in range(L):
                wk = w[k]
                if wk == 0.0:
                    continue
                if j + k < N:
                    total += wk * x[j + k]
                if j - k >= 0:
                    total += wk * x[j - k]
            grad_x[j] = total
        
        # Also account for the 1/s_sq term: d(1/s_sq)/dx[j] = -2*x[j]/s^3
        # Total gradient: grad_x / s_sq - 2 * (sum_k q[k]*|ac[k]|) * x[j] / s^3
        
        expected_abs = np.sum(q * abs_ac)
        grad_x = grad_x / s_sq - 2.0 * expected_abs * x / (s_sq * s)
        
        # Update x
        x = x - lr_x * grad_x
        
    return x, q

def _minimax_optimize(x_init, max_time=100):
    """
    Run the minimax game from initial x_init.
    """
    L = 2 * N - 1
    
    # Initial q: uniform over lags
    q = np.ones(L) / L
    
    # Adaptive learning rates
    lr_x = 0.01
    lr_q = 0.1
    
    x = x_init.copy()
    best_x = x.copy()
    best_val = _objective(x)
    
    start = time.time()
    
    while time.time() - start < max_time:
        if _check_deadline():
            break
        
        # Run several iterations
        x, q = _minimax_iterate(x, q, lr_x, lr_q, iters=20)
        
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Adapt learning rate: reduce if oscillating
        # Simple heuristic: keep same for now
        
    return best_x, best_val

def _coordinate_descent_polish(x, max_iters=200):
    """Fast coordinate descent to polish the solution."""
    best_x = x.copy()
    best_val = _objective(best_x)
    
    deltas = [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            for d in deltas:
                trial = best_x.copy()
                trial[j] += d
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Multiple restarts with different initializations
    inits = [
        np.ones(N),
        np.random.RandomState(42).randn(N),
        np.random.RandomState(123).randn(N),
        np.sin(2 * np.pi * np.arange(N) / 50.0),
        np.sin(2 * np.pi * np.arange(N) / 20.0),
        np.random.RandomState(7).rand(N) * 2 - 1,
    ]
    
    for x0 in inits:
        if _check_deadline():
            break
        
        try:
            x_opt, val_opt = _minimax_optimize(x0, max_time=15)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish with coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=100)
    
    return best_x
# EVOLVE-BLOCK-END
