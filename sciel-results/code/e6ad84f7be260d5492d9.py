import numpy as np
import time

N = 400
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
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
        return np.inf
    ac = _autoconv_fft(x)
    # Exclude lag 0 (index 0) for sidelobe optimization
    # Lags 1..N-1 are at indices 1..N-1
    # Lags -(N-1)..-1 are at indices N..2N-2
    peak_sidelobe = np.max(np.abs(ac[1:]))
    return peak_sidelobe / (s * s)

def _smooth_obj_grad(x, tau=0.1):
    """
    Compute a smooth approximation of max(|ac[1:]|)/(sum x)^2 and its gradient.
    Uses soft-max over |ac[1:]| values.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    # We only care about sidelobes (indices 1 to 2N-2)
    ac_side = ac[1:]
    abs_ac = np.abs(ac_side)
    L_side = len(ac_side) # 2N-2
    
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Soft-max approximation of max
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return m / (s * s), np.zeros_like(x)
    weights /= w_sum
    
    soft_max = m + tau * np.log(w_sum)
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx = sum_i w_i * d|ac_side_i|/dx
    # ac_side_i corresponds to lag (i+1)
    # Let lag l = i+1.
    # ac_l = sum_j x_j x_{l-j}
    # d(ac_l)/dx_k = x_{l-k} + x_{k-l} (if indices valid)
    
    # v[i] = w_i * sign(ac_side_i)
    # Note: ac_side[i] is ac[i+1]
    v = np.sign(ac_side) * weights
    
    # grad_k = sum_{l=1}^{2N-2} v[l-1] * (x_{l-k} + x_{k-l})
    # Let's map l to index i in v: i = l-1. l ranges 1..2N-2.
    # So i ranges 0..2N-3.
    # grad_k = sum_{i=0}^{2N-3} v[i] * (x_{i+1-k} + x_{k-(i+1)})
    
    # This is a convolution/correlation operation.
    # Let's compute it efficiently.
    # Term 1: sum_{i} v[i] * x_{i+1-k}
    # Let j = i+1-k => i = j+k-1.
    # sum_{j} v[j+k-1] * x_j
    # This is cross-correlation of v and x, shifted.
    
    # Term 2: sum_{i} v[i] * x_{k-i-1}
    # Let j = k-i-1 => i = k-j-1.
    # sum_{j} v[k-j-1] * x_j
    # This is convolution of v and x.
    
    # We can compute both using FFT.
    # Let's construct an array w of length L_fft such that we can do a single convolution.
    # Actually, let's just compute the gradient via FFT cross-correlation as hinted.
    
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
        
    # Pad x
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    # We need to compute G[k] = sum_{l} v[l-1] * (x_{l-k} + x_{k-l})
    # This is equivalent to the gradient of sum_{l} v[l-1] * |ac_l|? No, v already includes sign.
    # It is the gradient of sum_{l} v[l-1] * ac_l (since v = w * sign(ac), and d|ac|/dx = sign(ac)*dac/dx)
    # Wait, the soft-max objective is sum w_i |ac_i|.
    # Gradient is sum w_i sign(ac_i) dac_i/dx.
    # So yes, v is the coefficient for ac_l in the linear combination.
    
    # Let's define a sequence q of length L_fft where q[l] = v[l-1] for l=1..2N-2.
    # q[0] = 0.
    # q[l] = v[l-1]
    # Then we want the gradient of sum_l q[l] ac_l.
    # ac_l = sum_j x_j x_{l-j}
    # sum_l q[l] sum_j x_j x_{l-j} = sum_j x_j (sum_l q[l] x_{l-j})
    # Let h[j] = sum_l q[l] x_{l-j}. This is cross-correlation of q and x.
    # Then the sum is sum_j x_j h[j] = <x, h>.
    # The gradient of <x, h> w.r.t x is h + h_reversed? No.
    # ac is quadratic. d(sum_l q[l] ac_l)/dx_k = sum_l q[l] (x_{l-k} + x_{k-l})
    # This is exactly (q * x)[k] + (q * x)[N-1-k] ? No.
    # Let's stick to the explicit sum for N=400. It's fast enough.
    # 400 * 800 = 320,000 operations.
    
    grad_sm = np.zeros(N)
    # v is length 2N-2
    # k is 0..N-1
    # l is 1..2N-2
    # x_idx1 = l - k
    # x_idx2 = k - l
    
    # Vectorized approach:
    # For each k, we need:
    # T1[k] = sum_{l=1}^{2N-2} v[l-1] * x[l-k] (where x is 0 outside 0..N-1)
    # T2[k] = sum_{l=1}^{2N-2} v[l-1] * x[k-l]
    
    # We can compute T1 and T2 using FFT.
    # T1[k] = (q conv x)[k] ?
    # q[l] = v[l-1].
    # Convolution (q * x)[k] = sum_l q[l] x[k-l].
    # This matches T2[k] if we shift indices correctly.
    # T2[k] = sum_{l=1}^{2N-2} q[l] x[k-l].
    # So T2 is the standard linear convolution of q and x.
    
    # T1[k] = sum_{l=1}^{2N-2} q[l] x[l-k].
    # This is the cross-correlation of q and x, or convolution with reversed q.
    # Let q_rev[n] = q[-n].
    # (q_rev * x)[k] = sum_n q_rev[n] x[k-n] = sum_n q[-n] x[k-n].
    # Let l = -n => n = -l.
    # sum_l q[l] x[k+l].
    # This is not quite T1.
    # T1[k] = sum_l q[l] x[l-k].
    # Let m = l-k => l = m+k.
    # sum_m q[m+k] x[m].
    # This is the cross-correlation of q and x at lag k.
    # corr[k] = sum_m q[m+k] x[m].
    
    # We can compute both using FFT.
    # q_pad length L_fft. q_pad[l] = v[l-1] for l=1..2N-2.
    q_pad = np.zeros(L_fft)
    q_pad[1:2*N-1] = v
    
    Q = np.fft.rfft(q_pad, n=L_fft)
    X = np.fft.rfft(x_pad, n=L_fft)
    
    # T2 = convolution(q, x)
    # C = ifft(Q * X)
    # C[k] = sum_l q[l] x[k-l] (circular, but we padded enough for linear)
    conv_qx = np.fft.irfft(Q * X, n=L_fft)
    T2 = conv_qx[:N]
    
    # T1 = correlation(q, x)
    # corr[k] = sum_m q[m+k] x[m]
    # This is ifft(Q * conj(X)) ?
    # ifft(Q * conj(X))[k] = sum_n q[n] x[n-k] (circular)
    # Let's check:
    # D = ifft(Q * conj(X))
    # D[k] = sum_n q[n] x[n-k]
    # We want sum_m q[m+k] x[m].
    # Let n = m+k => m = n-k.
    # sum_n q[n] x[n-k].
    # Yes! D[k] is exactly T1[k].
    
    corr_qx = np.fft.irfft(Q * np.conj(X), n=L_fft)
    T1 = corr_qx[:N]
    
    grad_sm = T1 + T2
    
    # Gradient of objective:
    # f = soft_max / s^2
    # df/dx_k = (grad_sm_k * s^2 - soft_max * 2 * s) / s^4
    # = grad_sm_k / s^2 - 2 * soft_max / s^3
    
    grad_obj = grad_sm / (s * s) - 2 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    def time_ok():
        return time.time() < deadline
    
    # Initial guess: random +/-1
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=N).astype(np.float64)
    
    best_h = h.copy()
    best_score = _objective(h)
    
    # Phase 1: Gradient Descent / L-BFGS-like
    # We will use a simple gradient descent with adaptive step size and momentum
    # Or better, use scipy.optimize.minimize if available, but let's stick to numpy for safety
    # and control.
    
    x = h.copy()
    lr = 0.1
    momentum = 0.9
    v_mom = np.zeros_like(x)
    
    iterations = 0
    while time_ok() and iterations < 2000:
        val, grad = _smooth_obj_grad(x, tau=0.1)
        
        # Normalize gradient to prevent explosion
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-6:
            grad = grad / g_norm
            
        # Momentum update
        v_mom = momentum * v_mom - lr * grad
        x_new = x + v_mom
        
        # Project back to +/-1 (or smooth projection)
        # For now, let's keep x continuous but penalize deviation from +/-1
        # Or simply round? Rounding breaks gradient flow.
        # Let's use a soft projection: tanh scaling?
        # Actually, the task asks for "signed step heights", implying integers or reals?
        # "sequence of signed step heights". Seed is int8.
        # If we can use reals, the problem is easier.
        # If we must use +/-1, we should round.
        # Let's assume reals are allowed for the optimization path, 
        # but we will return the rounded version or the continuous version?
        # "construct() -> sequence of signed step heights"
        # Usually implies the discrete set. But let's check the score.
        # If we return continuous values, the evaluator might reject or score differently.
        # The seed returns int8.
        # Let's optimize on continuous domain, but evaluate on rounded domain for tracking best.
        
        # Evaluate rounded version
        x_round = np.sign(x_new)
        # Handle zeros
        x_round[x_round == 0] = 1
        
        score_round = _objective(x_round)
        
        if score_round < best_score:
            best_score = score_round
            best_h = x_round.copy()
            
        # Also check continuous score for gradient direction validity
        if not np.isfinite(val):
            break
            
        # Update x
        x = x_new
        
        # Adaptive learning rate
        if np.random.rand() < 0.1:
            lr *= 0.95
        else:
            lr = min(lr * 1.05, 1.0)
            
        iterations += 1
        
    # Phase 2: Local Search (Bit Flips)
    # Start from best_h (rounded)
    x = best_h.astype(np.float64)
    
    # Try random flips
    max_flips = 50000
    flip_count = 0
    while time_ok() and flip_count < max_flips:
        idx = np.random.randint(N)
        x[idx] *= -1
        score = _objective(x)
        if score < best_score:
            best_score = score
            best_h = x.copy()
        else:
            x[idx] *= -1
        flip_count += 1
        
    # Phase 3: Structured Perturbations
    # Try low-frequency sinusoids
    base = best_h.copy().astype(np.float64)
    for freq in [1, 2, 3, 5, 8]:
        if not time_ok():
            break
        t = np.arange(N)
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            perturbation = np.sin(2 * np.pi * freq * t / N + phase)
            # Mix base and perturbation
            for alpha in [0.1, 0.2, 0.5, 0.8]:
                if not time_ok():
                    break
                x_cand = base + alpha * perturbation
                x_cand = np.sign(x_cand)
                x_cand[x_cand == 0] = 1
                score = _objective(x_cand)
                if score < best_score:
                    best_score = score
                    best_h = x_cand.copy()
    
    # Return as int8 to match seed interface
    return best_h.astype(np.int8)

if __name__ == "__main__":
    # Quick test
    result = construct()
    print(f"Score: {_objective(result.astype(np.float64))}")
    print(f"Length: {len(result)}")
