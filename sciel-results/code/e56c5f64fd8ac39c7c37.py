import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective_and_grad(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation, with gradient."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18, np.zeros(N)
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    wsum = np.sum(weights)
    if wsum < 1e-300:
        weights = np.zeros_like(conv)
        weights[np.argmax(conv)] = 1.0
        wsum = 1.0
    weights /= wsum
    soft_max = m + np.log(wsum) / beta
    val = 2.0 * N * soft_max / (s * s)
    obj = abs(val)
    
    # d(soft_max)/d(h[j]) = 2 * sum_{k} weights[k] * h[k-j] for valid k-j in [0,N-1]
    # = 2 * sum_{u=0}^{N-1} weights[u+j] * h[u]
    # Compute via FFT cross-correlation
    # We want g[j] = sum_{u=0}^{N-1} h[u] * weights[u+j] for j in [0, N-1]
    # Pad h to length 2N, weights to length 2N
    n_fft = 2 * N
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    w_pad = np.zeros(n_fft)
    w_pad[:2*N-1] = weights  # weights has length 2N-1
    
    # Cross-correlation: g[j] = sum_u h_pad[u] * w_pad[u+j]
    # Using FFT: ifft(fft(h_pad) * conj(fft(w_pad))) gives sum_u h_pad[u] * w_pad[(u-j) mod n]
    # We need sum_u h_pad[u] * w_pad[u+j]
    # This equals ifft(fft(h_pad) * fft(w_pad)[::-1])... let me think again
    # 
    # Standard: corr(a, b)[j] = sum_k a[k] * conj(b[k-j])
    # In numpy: np.correlate(a, b, 'full')[j] = sum_k a[k] * b[k-j] for real
    # 
    # We want: g[j] = sum_u h[u] * weights[u+j]
    # Let b = weights reversed: b[k] = weights[2N-2-k]
    # Then weights[u+j] = b[2N-2-u-j]
    # g[j] = sum_u h[u] * b[2N-2-u-j]
    # Let v = 2N-2-u-j, then u = 2N-2-v-j
    # g[j] = sum_v h[2N-2-v-j] * b[v]
    # This is a cross-correlation of h_reversed and b
    # 
    # Simpler: just use the fact that for N=400, a direct numpy dot product 
    # in a loop is ~400 * 400 = 160k ops, which is fast enough.
    # But let's try FFT properly.
    
    # The cleanest FFT approach:
    # g[j] = sum_{u=0}^{N-1} h[u] * weights[u+j]
    # This is the cross-correlation of h and weights at lag -j (or equivalently, 
    # the correlation at lag j if we define it appropriately)
    # 
    # Using FFT: 
    # Let H = fft(h, n_fft), W = fft(weights_padded, n_fft)
    # ifft(H * conj(W))[j] = sum_k h_padded[k] * weights_padded[(k-j) mod n_fft]
    # For j in [0, N-1], k in [0, N-1]:
    #   (k-j) mod n_fft = k-j if k>=j, else k-j+n_fft
    #   weights_padded[k-j] for k>=j: weights[k-j] (valid since k-j <= N-1-j < 2N-1)
    #   weights_padded[k-j+n_fft] for k<j: k-j+n_fft in [n_fft-j+1, n_fft-1]
    #   weights_padded is zero there (length 2N-1, padded to 2N)
    # So ifft(H * conj(W))[j] = sum_{k=j}^{N-1} h[k] * weights[k-j]
    # But we want sum_{u=0}^{N-1} h[u] * weights[u+j]
    # 
    # Note: sum_{k=j}^{N-1} h[k] * weights[k-j] with m=k-j gives sum_{m=0}^{N-1-j} h[m+j] * weights[m]
    # We want sum_{u=0}^{N-1} h[u] * weights[u+j]
    # These are different!
    # 
    # Let me just compute it directly - for N=400 it's fast enough.
    d_soft_max = np.zeros(N)
    for j in range(N):
        d_soft_max[j] = 2.0 * np.dot(h, weights[j:j+N])
    
    sign = np.sign(2.0 * N * soft_max / (s * s))
    if sign == 0:
        sign = 1.0
    
    grad = np.zeros(N)
    grad[:] = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    return obj, grad


def _eigen_init(max_time=15.0):
    """Initialize using eigenvectors of random PSD matrices."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() - t0 > max_time * 0.8:
            break
        A = rng.normal(0, 1, (N, N))
        A = A @ A.T
        try:
            v = rng.normal(0, 1, N)
            v /= np.linalg.norm(v)
            for _ in range(20):
                v = A @ v
                norm = np.linalg.norm(v)
                if norm < 1e-12:
                    break
                v /= norm
        except Exception:
            continue
        
        h = v.copy()
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        v_val = _value(h)
        if v_val < best_val:
            best_val = v_val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _diverse_init(max_time=15.0):
    """Generate diverse initial candidates."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    candidates.append(np.ones(N))
    
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    for trial in range(30):
        if time.time() - t0 > max_time * 0.4:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(5):
            if time.time() - t0 > max_time * 0.7:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    for n_freq in range(2, 5):
        if time.time() - t0 > max_time * 0.85:
            break
        for trial in range(10):
            freqs = sorted(rng.uniform(1, 40, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.0, n_freq)
            dc = rng.uniform(-0.1, 0.1)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [40.0, 60.0, 100.0]:
        if time.time() - t0 > max_time * 0.75:
            break
        try:
            def obj_fn(h):
                obj, _ = _smooth_objective_and_grad(h, beta=beta)
                return obj
            
            def grad_fn(h):
                _, g = _smooth_objective_and_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=10, max_time=30.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.1 + 0.03 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            def obj_fn(h):
                obj, _ = _smooth_objective_and_grad(h, beta=50.0)
                return obj
            
            def grad_fn(h):
                _, g = _smooth_objective_and_grad(h, beta=50.0)
                return g
            
            res = minimize(
                obj_fn,
                h_init,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    h_div, val_div = _diverse_init(max_time=10.0)
    
    h_eig, val_eig = _eigen_init(max_time=10.0)
    
    candidates = [(h_div, val_div), (h_eig, val_eig)]
    best_h = None
    best_val = 1e18
    for h_c, v_c in candidates:
        if v_c < best_val:
            best_val = v_c
            best_h = h_c.copy()
    
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 15.0))
    
    if time_left() > 25:
        best_h, best_val = _multi_start(best_h, n_starts=10, max_time=min(25.0, time_left() - 12.0))
    
    if time_left() > 12:
        try:
            def obj_fn(h):
                obj, _ = _smooth_objective_and_grad(h, beta=100.0)
                return obj
            
            def grad_fn(h):
                _, g = _smooth_objective_and_grad(h, beta=100.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 600, 'maxfun': 1200, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    if time_left() > 8:
        try:
            def obj_fn(h):
                obj, _ = _smooth_objective_and_grad(h, beta=150.0)
                return obj
            
            def grad_fn(h):
                _, g = _smooth_objective_and_grad(h, beta=150.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
