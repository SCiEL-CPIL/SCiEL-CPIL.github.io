"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import time

def _autoconv(x):
    """Full autoconvolution of x."""
    return fftconvolve(x, x[::-1], mode='full')

def _score(x):
    """Compute max|autoconv| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    return float(np.max(np.abs(ac)) / (s * s))

def _soft_max_abs(x, temperature=10.0):
    """Smooth approximation of max(|x|) using soft-max."""
    ax = np.abs(x)
    # Use log-sum-exp trick for numerical stability
    m = np.max(ax)
    if m < 1e-12:
        return 0.0
    scaled = ax / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    return m + temperature * np.log(np.sum(np.exp(scaled - np.max(scaled)))) / len(ax)

def _soft_objective(x, temperature=10.0):
    """Smooth surrogate for max|autoconv| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-6:
        return 1e10
    ac = _autoconv(x)
    sm = _soft_max_abs(ac, temperature)
    return sm / (s * s)

def _soft_grad(x, temperature=10.0):
    """Gradient of soft objective w.r.t. x.
    
    d/dx [softmax(|ac|) / s^2] where ac = autoconv(x)
    ac[k] = sum_i x[i] * x[i+k-N+1] (for full mode with x[::-1])
    
    We use the fact that autoconv(x) = fftconvolve(x, x[::-1], 'full')
    and compute gradients via FFT.
    """
    s = x.sum()
    if abs(s) < 1e-6:
        return np.zeros_like(x)
    
    ac = _autoconv(x)
    ax = np.abs(ac)
    m = np.max(ax)
    if m < 1e-12:
        return np.zeros_like(x)
    
    scaled = ax / temperature
    scaled = np.clip(scaled, -50, 50)
    smax = np.max(scaled)
    exp_scaled = np.exp(scaled - smax)
    sum_exp = np.sum(exp_scaled)
    
    # d soft_max / d ac[k] = exp_scaled[k] / sum_exp * sign(ac[k])
    dac = exp_scaled / sum_exp * np.sign(ac)
    
    # d ac / d x: ac = fftconvolve(x, x[::-1], 'full')
    # ac[k] = sum_i x[i] * x[i - k + N - 1] for appropriate range
    # Gradient: grad_x[j] = sum_k dac[k] * d(ac[k])/d(x[j])
    # d(ac[k])/d(x[j]) = x[j - k + N - 1] + x[j + k - N + 1] (approximately, for full conv)
    
    # More precisely, for full convolution of x with x[::-1]:
    # ac[k] = sum_{i} x[i] * x_rev[k-i] where x_rev = x[::-1]
    # x_rev[m] = x[N-1-m]
    # So ac[k] = sum_i x[i] * x[N-1-(k-i)] = sum_i x[i] * x[N-1-k+i]
    # d(ac[k])/d(x[j]) = x[N-1-k+j] (if in range) + (if j contributes to other term)
    # Actually for autoconv: ac[k] = sum_i x[i] * x[i+k-N+1]
    # d(ac[k])/d(x[j]) = x[j+k-N+1] + x[j-k+N-1] (symmetric)
    
    # Use FFT-based gradient: 
    # ac = ifft(fft(x) * conj(fft(x))) would be circular, but we have linear.
    # Let's use the direct approach with the structure.
    
    # For linear autoconvolution: ac[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} x[i] * x[i+k-N+1]
    # This is getting complex. Let's use a simpler approach:
    # Compute gradient numerically or use the FFT structure.
    
    # Actually, let's use the fact that:
    # ac = fftconvolve(x, x[::-1], 'full')
    # Let y = x[::-1]. Then ac = fftconvolve(x, y, 'full')
    # d(ac)/d(x[j]) = y convolved with delta at position j, plus x convolved with delta at position j in y
    # = y shifted by j (for the x part) + x shifted by (N-1-j) (for the y part)
    
    # Simpler: use central difference for gradient estimation
    # But that's expensive. Let's use an analytical approach.
    
    # For full linear convolution c = conv(x, y):
    # c[k] = sum_i x[i] * y[k-i]
    # d(c[k])/d(x[j]) = y[k-j]
    # d(c[k])/d(y[j]) = x[k-j]
    
    # Here y = x[::-1], so y[m] = x[N-1-m]
    # d(ac[k])/d(x[j]) = y[k-j] + x[k-j'] where j' is the index in y
    # Since y = x[::-1], changing x[j] changes y[N-1-j]
    # So d(ac[k])/d(x[j]) = y[k-j] + x[k-(N-1-j)] = x[N-1-(k-j)] + x[k-N+1+j]
    
    # Let's compute this properly using FFT
    # Actually, let's just use a simple finite difference for the gradient
    # since N=400 and we can afford it
    
    eps = 1e-7
    grad = np.zeros_like(x)
    
    # For efficiency, we'll compute the gradient using the structure
    # ac[k] = sum_i x[i] * x[i + k - N + 1]
    # d(ac[k])/d(x[j]) = x[j + k - N + 1] + x[j - k + N - 1]
    
    # We can compute this as:
    # grad[j] = sum_k dac[k] * (x[j+k-N+1] + x[j-k+N+1])
    
    # This is a convolution-like operation. Let's use FFT.
    # Let d = dac (length 2N-1)
    # grad[j] = sum_k dac[k] * x[j+k-N+1] + sum_k dac[k] * x[j-k+N-1]
    
    # First term: sum_k dac[k] * x[j+k-N+1]
    # Let m = j+k-N+1, so k = m-j+N-1
    # sum_m dac[m-j+N-1] * x[m]
    # This is a cross-correlation of dac with x
    
    # Use FFT for both terms
    dac_padded = np.zeros(2*N, dtype=np.float64)
    dac_padded[:len(dac)] = dac
    
    # Term 1: cross-correlation of dac with x
    # corr1[j] = sum_k dac[k] * x[j+k-N+1]
    # This is like (dac * x)[j+N-1] if we pad appropriately
    
    # Let's just do it directly since N=400, 2N-1=799
    # grad[j] = sum_k dac[k] * (x[j+k-N+1] if 0<=j+k-N+1<N else 0)
    #         + sum_k dac[k] * (x[j-k+N-1] if 0<=j-k+N-1<N else 0)
    
    for j in range(N):
        total = 0.0
        for k in range(len(dac)):
            # First term: x[j + k - N + 1]
            idx1 = j + k - N + 1
            if 0 <= idx1 < N:
                total += dac[k] * x[idx1]
            # Second term: x[j - k + N - 1]
            idx2 = j - k + N - 1
            if 0 <= idx2 < N:
                total += dac[k] * x[idx2]
        grad[j] = total
    
    # Chain rule: d(softmax/s^2)/dx = (s^2 * d(softmax)/dx - 2s * softmax * ds/dx) / s^4
    # = d(softmax)/dx / s^2 - 2 * softmax * ds/dx / s^3
    # ds/dx = 1 for all j
    
    sm = _soft_max_abs(ac, temperature)
    grad = grad / (s * s) - 2.0 * sm * np.ones_like(x) / (s * s * s)
    
    return grad

def _gradient_descent(x0, max_iters=500, lr=0.01, temperature=10.0, deadline=None):
    """Gradient descent on the soft objective."""
    x = x0.copy().astype(np.float64)
    cur_obj = _soft_objective(x, temperature)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _soft_grad(x, temperature)
        x_new = x - lr * grad
        
        # Project to keep values reasonable
        x_new = np.clip(x_new, -10, 10)
        
        new_obj = _soft_objective(x_new, temperature)
        if new_obj < cur_obj:
            x = x_new
            cur_obj = new_obj
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.9, 1e-4)
    
    return x, cur_obj

def _local_search(x_init, max_iters=2000, deadline=None, rng=None):
    """Local sign-flip search."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    x = x_init.copy().astype(np.float64)
    cur_score = _score(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random element
        idx = rng.integers(0, N)
        x[idx] *= -1
        new_score = _score(x)
        if new_score < cur_score:
            cur_score = new_score
        else:
            x[idx] *= -1  # revert
    
    return x, cur_score

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(123)
    
    best_x = np.ones(N, dtype=np.int64)
    best_score = _score(best_x)
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N, dtype=np.float64))
    
    # 2. Random ±1
    for seed in [1, 2, 3]:
        r = np.random.default_rng(seed)
        inits.append(r.choice([-1.0, 1.0], size=N))
    
    # 3. Smooth random (low-pass filtered random)
    for seed in [10, 20, 30]:
        r = np.random.default_rng(seed)
        x = r.standard_normal(N)
        # Simple low-pass via moving average
        kernel = np.ones(20) / 20
        from scipy.ndimage import uniform_filter1d
        x = uniform_filter1d(x, size=20)
        # Sign
        x = np.sign(x)
        inits.append(x)
    
    # 4. Structured: alternating blocks
    for block_size in [10, 20, 50]:
        x = np.zeros(N)
        for i in range(0, N, block_size * 2):
            x[i:i+block_size] = 1
            x[i+block_size:i+2*block_size] = -1
        inits.append(x)
    
    # 5. Cosine-based
    for freq in [1, 2, 3, 5]:
        t = np.linspace(0, freq * 2 * np.pi, N)
        x = np.sign(np.cos(t))
        inits.append(x)
    
    # Run gradient descent + local search on each initialization
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        # Gradient descent phase
        x_cont, _ = _gradient_descent(x0, max_iters=200, lr=0.05, 
                                       temperature=5.0, deadline=deadline - 20.0)
        
        # Round to ±1
        x_round = np.sign(x_cont)
        # Handle zeros
        x_round[x_round == 0] = 1
        
        # Local search
        x_ls, score_ls = _local_search(x_round, max_iters=1500, 
                                        deadline=deadline - 10.0, rng=rng)
        
        if score_ls < best_score:
            best_score = score_ls
            best_x = np.round(x_ls).astype(np.int64)
    
    # Final local search on best
    if time.time() < deadline - 5.0:
        x_final, score_final = _local_search(best_x.astype(np.float64), 
                                              max_iters=2000, 
                                              deadline=deadline - 2.0, rng=rng)
        if score_final < best_score:
            best_score = score_final
            best_x = np.round(x_final).astype(np.int64)
    
    return best_x
# EVOLVE-BLOCK-END
