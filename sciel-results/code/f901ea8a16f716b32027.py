import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def autoconv_max_abs(x):
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return float(np.max(np.abs(ac)))

    def score(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_max_abs(x) / (s * s)

    def local_optimize(h, max_passes=3, deadline=None):
        h = h.copy()
        best_val = score(h)
        for _ in range(max_passes):
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = -h[i]
                new_val = score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = -h[i]
            if not improved:
                break
        return h, best_val

    def smooth_gradient_descent(h, lr=0.01, steps=2000, deadline=None):
        """Gradient descent on smooth surrogate: log-sum-exp of |autoconv|."""
        h = h.copy()
        best_val = score(h)
        # Use a soft surrogate for the max
        beta = 50.0  # sharpness parameter
        
        for step in range(steps):
            if deadline is not None and time.time() > deadline:
                break
            
            ac = fftconvolve(h, h, mode='full')
            abs_ac = np.abs(ac)
            # Soft max via log-sum-exp
            max_ac = np.max(abs_ac)
            if max_ac < 1e-12:
                break
            # Normalized weights
            exp_ac = np.exp(beta * (abs_ac - max_ac))
            weights = exp_ac / np.sum(exp_ac)
            
            # Gradient of soft-max of |ac| w.r.t. h
            # ac_k = sum_i h_i h_{i+k-N+1} (for full mode, offset)
            # d|ac_k|/dh = sign(ac_k) * (h shifted)
            # For full convolution, ac has length 2N-1
            # ac[k] = sum_i h[i] h[i - (k - (N-1))]
            
            # Compute gradient via FFT
            H = np.fft.rfft(h, n=2*N)
            # Gradient of ac w.r.t. h is h flipped and shifted
            # For autoconv: grad_k = h[N-1-k] for k in [0, N-1], then negative for k >= N
            # More carefully: ac = conv(h, h), so d(ac_k)/dh_i = h_{k-i} + h_{k-i} = 2*h_{k-i} (if in range)
            # Actually for autoconvolution: ac_k = sum_j h_j h_{j+k-(N-1)}
            # d(ac_k)/dh_i = h_{i+k-(N-1)} + h_{i+k-(N-1)} ... no
            # ac_k = sum_j h_j * h_{j + (k - (N-1))}
            # Let m = k - (N-1), so ac_k = sum_j h_j h_{j+m}
            # d(ac_k)/dh_i: if i = j, term is h_{i+m}; if i = j+m, term is h_{i-m}
            # So d(ac_k)/dh_i = h_{i+m} + h_{i-m} (where indices are valid)
            
            # Simpler: use the fact that grad of sum_k w_k |ac_k| w.r.t. h
            # = sum_k w_k sign(ac_k) * (h shifted by k-(N-1) + h shifted by -(k-(N-1)))
            
            # Let's compute it directly
            grad = np.zeros(N)
            for k in range(len(ac)):
                if abs(weights[k]) < 1e-10:
                    continue
                m = k - (N - 1)
                sgn = np.sign(ac[k])
                w = weights[k] * sgn
                # d(ac_k)/dh_i = h_{i+m} + h_{i-m} for valid indices
                for i in range(N):
                    if 0 <= i + m < N:
                        grad[i] += w * h[i + m]
                    if 0 <= i - m < N:
                        grad[i] += w * h[i - m]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Normalize h to keep sum fixed (sum should be 1 for optimal scaling)
            s = np.sum(h)
            if abs(s) < 1e-12:
                break
            
            # Update
            h_new = h - lr * grad / grad_norm
            # Renormalize to keep sum = sign(s)
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                continue
            h_new = h_new * (np.sign(s) * 1.0) / abs(s_new)
            
            val_new = score(h_new)
            if val_new < best_val:
                h = h_new
                best_val = val_new
            else:
                # Reduce learning rate
                lr *= 0.95
        
        return h, best_val

    best_h = np.ones(N)
    best_val = score(best_h)

    rng = np.random.default_rng(42)
    x = np.arange(N)

    # 1. Raised cosine (Hanning) window
    hann = 0.5 * (1 + np.cos(2 * np.pi * x / (N - 1)))
    for sign_pattern in [
        np.ones(N),
        np.sign(np.cos(np.pi * x / (N - 1))),
        np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1)),
    ]:
        h = hann * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 2. Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * x / (N - 1))
    for sign_pattern in [np.ones(N), np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1))]:
        h = hamming * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 3. Triangular window
    tri = 1 - np.abs(2 * x / (N - 1) - 1)
    for sign_pattern in [np.ones(N), np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1))]:
        h = tri * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 4. Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * x / (N - 1)) + 0.08 * np.cos(4 * np.pi * x / (N - 1))
    for sign_pattern in [np.ones(N), np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1))]:
        h = blackman * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 5. Cosine modulation
    for freq in [1, 2, 3, 5, 8]:
        mod = np.cos(2 * np.pi * freq * x / N)
        h = hann * mod
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Smooth gradient descent on best candidate
    if time.time() < deadline:
        h_opt, val_opt = smooth_gradient_descent(best_h, lr=0.05, steps=1500, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Local sign-flip optimization
    if time.time() < deadline:
        h_opt, val_opt = local_optimize(best_h, max_passes=5, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # More diverse restarts
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r % 5 == 0:
            h = rng.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            h = np.ones(N)
            n_neg = rng.integers(N//4, N//2)
            neg_idx = rng.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 5 == 2:
            h = np.sign(np.cos(2 * np.pi * x * (x + 1) / (2 * N)))
        elif r % 5 == 3:
            h = rng.uniform(-1, 1, size=N)
        else:
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = rng.integers(1, N // 10 + 1)
                sign = rng.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.5
            h = np.sign(h)
        
        # Smooth gradient descent first
        h_opt, val_opt = smooth_gradient_descent(h, lr=0.05, steps=800, deadline=deadline)
        
        # Then local sign-flip
        if time.time() < deadline:
            h_opt, val_opt = local_optimize(h_opt, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # Final polish: more iterations of smooth GD + local
    if time.time() < deadline:
        h_opt, val_opt = smooth_gradient_descent(best_h, lr=0.02, steps=1000, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
        if time.time() < deadline:
            h_opt, val_opt = local_optimize(best_h, max_passes=5, deadline=deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()

    return best_h
