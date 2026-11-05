"""Combined program: FFT-based autoconvolution with analytic gradient, structured
initializations (block-alternating, sparse-spike), and L-BFGS-B refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the exact evaluation metric."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _autoconvolve_fft(h):
    """Compute autoconvolution using FFT for speed."""
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:n_conv]
    return conv

def _soft_max_obj(h, beta):
    """
    Soft-max surrogate objective.
    Minimizes a smoothed version of |2*N*max(conv)/S^2| using log-sum-exp.
    Returns val^2 for smoothness.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = _autoconvolve_fft(h)
    m = np.max(conv)
    if m > 1e15:
        return 1e10
    logsum = m + np.log(np.sum(np.exp(beta * (conv - m))))
    soft_max_val = logsum / beta
    val = 2.0 * N * soft_max_val / (s * s)
    return val * val

def _soft_max_grad(h, beta):
    """
    Analytic gradient of the soft-max squared objective.
    Uses FFT-based autoconvolution and the chain rule.
    """
    if not np.all(np.isfinite(h)):
        return np.ones_like(h)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.ones_like(h)
    
    conv = _autoconvolve_fft(h)
    m = np.max(conv)
    if m > 1e15:
        return np.ones_like(h)
    
    exp_conv = np.exp(beta * (conv - m))
    sum_exp = np.sum(exp_conv)
    if sum_exp < 1e-300:
        return np.ones_like(h)
    
    weights = exp_conv / sum_exp  # softmax weights over conv entries
    
    # d(soft_max)/d(conv_k) = weights[k]
    # conv[k] = sum_i h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] if 0<=k-j<N, else 0
    # More precisely: d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N and 0<=j<N
    # Actually: d(conv[k])/d(h[j]) = h[k-j] (from first h) + h[k-j] (from second h)
    # Both terms are h[k-j] when 0<=k-j<N
    # So d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N, else 0
    
    # d(soft_max)/d(h[j]) = (1/beta) * sum_k weights[k] * d(conv[k])/d(h[j])
    # = (2/beta) * sum_{k: 0<=k-j<N} weights[k] * h[k-j]
    # Let l = k-j, so k = j+l, l in [0, N-1]
    # = (2/beta) * sum_{l=0}^{N-1} weights[j+l] * h[l] (where j+l < 2N-1)
    
    # This is a cross-correlation: grad_sm[j] = (2/beta) * sum_l weights[j+l] * h[l]
    # For efficiency, use FFT-based correlation
    # grad_sm[j] = (2/beta) * (weights convolved with h)[j]
    # where the convolution is cross-correlation
    
    # Actually, let's compute it directly using FFT
    # grad_sm[j] = (2/beta) * sum_{l=0}^{min(N-1, 2N-2-j)} weights[j+l] * h[l]
    # This is like: for each j, dot product of h[0:min(N,2N-1-j)] with weights[j:j+min(N,2N-1-j)]
    
    # Use a simple approach: for N=400, 2N-1=799, this is manageable
    # But let's use FFT for speed
    # grad_sm = (2/beta) * ifft(fft(weights) * conj(fft(h)))
    # Wait, that's not quite right. Let me think again.
    
    # grad_sm[j] = (2/beta) * sum_l weights[j+l] * h[l]
    # This is a cross-correlation of weights and h
    # cross_corr[j] = sum_l weights[j+l] * h[l]
    # = sum_l weights[j+l] * h[l]
    # Let's use FFT: cross_corr = ifft(fft(weights) * conj(fft(h)))
    # But we need to be careful with the indexing
    
    # Actually, let's just compute it directly for clarity and correctness
    # For N=400, this loop is O(N^2) = 160000 operations, which is fast
    grad_sm = np.zeros(N)
    for j in range(N):
        max_l = min(N, 2 * N - 1 - j)
        if max_l > 0:
            grad_sm[j] = (2.0 / beta) * np.dot(h[:max_l], weights[j:j + max_l])
    
    val = 2.0 * N * (m + np.log(sum_exp)) / beta / (s * s)
    
    # d(val^2)/d(h[j]) = 2*val * d(val)/d(h[j])
    # d(val)/d(h[j]) = 2*N * d(soft_max)/d(h[j])/S^2 - 2*val * S / S^2 * 1
    #                = 2*N * grad_sm[j]/S^2 - 2*val/S
    
    d_val_d_h = 2.0 * N * grad_sm / (s * s) - 2.0 * val / s
    grad = 2.0 * val * d_val_d_h
    
    # Ensure gradient is finite
    if not np.all(np.isfinite(grad)):
        return np.ones_like(h)
    
    return grad

def _block_alternating_init(block_size, seed=42):
    """Create a block-alternating initialization."""
    rng = np.random.RandomState(seed)
    h = np.zeros(N)
    for i in range(N // block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        h[i * block_size:(i + 1) * block_size] = sign
    # Add small noise
    h += rng.normal(0, 0.01, size=N)
    return h

def _sparse_spike_init(n_spikes, seed=42):
    """Create a sparse spike initialization with positive and negative lobes."""
    rng = np.random.RandomState(seed)
    h = np.zeros(N)
    positions = rng.choice(N, size=n_spikes, replace=False)
    for i, pos in enumerate(positions):
        sign = 1.0 if i % 2 == 0 else -1.0
        h[pos] = sign * rng.uniform(0.5, 2.0)
    # Add small background
    h += rng.uniform(-0.05, 0.05, size=N)
    return h

def _structured_init(seed=42):
    """Create a structured initialization with multiple scales."""
    rng = np.random.RandomState(seed)
    h = np.zeros(N)
    
    # Large-scale blocks
    block_size = 20
    for i in range(N // block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        h[i * block_size:(i + 1) * block_size] = sign
    
    # Add medium-scale spikes
    n_spikes = 10
    positions = rng.choice(N, size=n_spikes, replace=False)
    for i, pos in enumerate(positions):
        sign = 1.0 if i % 2 == 0 else -1.0
        h[pos] += sign * rng.uniform(0.3, 1.0)
    
    # Add small noise
    h += rng.normal(0, 0.02, size=N)
    
    return h

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses FFT-based autoconvolution with analytic gradient for L-BFGS-B optimization.
    Multiple structured initializations to explore different regions of the landscape.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0

    # Generate diverse initializations
    inits = []
    
    # Block-alternating with different block sizes
    for bs in [10, 20, 40, 80]:
        inits.append(_block_alternating_init(bs, seed=42 + bs))
    
    # Sparse spikes
    for ns in [5, 10, 20, 40]:
        inits.append(_sparse_spike_init(ns, seed=100 + ns))
    
    # Structured multi-scale
    for seed in [42, 123, 777]:
        inits.append(_structured_init(seed))
    
    # Seed
    inits.append(np.ones(N))
    
    # Try multiple beta values (soft-max sharpness)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]

    for beta in betas:
        if time.time() > deadline:
            break
        
        for init in inits:
            if time.time() > deadline:
                break
            
            # Normalize initial vector
            s_init = np.sum(init)
            if abs(s_init) < 1e-10:
                continue
            h_init = init * (N / s_init)
            
            try:
                result = minimize(
                    _soft_max_obj,
                    h_init.copy(),
                    jac=_soft_max_grad,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50,
                    }
                )
                
                if result.fun < 1e10:
                    h_opt = result.x
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-10 and np.all(np.isfinite(h_opt)):
                        h_opt = h_opt * (N / s_opt)
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 2: Coordinate descent polish on individual elements
    if time.time() < deadline - 10:
        h_work = best_h.copy()
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        for step in step_sizes:
            if time.time() > deadline - 10:
                break
            improved_any = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                old_val = h_work[i]
                for delta in [step, -step]:
                    h_work[i] = old_val + delta
                    s = np.sum(h_work)
                    if abs(s) < 1e-10:
                        h_work[i] = old_val
                        continue
                    v = _eval_value(h_work)
                    if v < best_val:
                        best_val = v
                        best_h = h_work.copy()
                        improved_any = True
                    else:
                        h_work[i] = old_val
            if not improved_any:
                break

    # Phase 3: Final L-BFGS-B polish with high beta
    if time.time() < deadline - 5:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5:
                break
            
            try:
                result = minimize(
                    _soft_max_obj,
                    best_h.copy(),
                    jac=_soft_max_grad,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50,
                    }
                )
                
                if result.fun < 1e10:
                    h_opt = result.x
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-10 and np.all(np.isfinite(h_opt)):
                        h_opt = h_opt * (N / s_opt)
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
            except Exception:
                continue

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
