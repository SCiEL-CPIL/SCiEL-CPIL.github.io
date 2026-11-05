"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _evaluate(h):
    """Compute the objective value for a given h."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    c = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(c) / (s * s))

def _objective_and_grad(h):
    """
    Objective: minimize max(c) where c = conv(h,h), subject to sum(h)=1.
    We use a smoothed max: log-sum-exp of c entries.
    Actually, let's just minimize the max directly via subgradient or use
    a different formulation.
    
    Better: minimize ||c||_2^2 subject to sum(h)=1 and c_k <= 0 for all k != argmax?
    No, let's stick to the directive: minimize squared L2 norm of autoconvolution.
    Objective: sum(c_i^2) subject to sum(h)=1.
    This is a QP.
    """
    c = np.convolve(h, h, mode='full')
    obj = np.sum(c * c)
    
    # Gradient of obj w.r.t. h
    # c_k = sum_i h_i h_{k-i}
    # d(c_k)/d(h_j) = h_{k-j} + h_{k-j} = 2 h_{k-j} (if indices valid)
    # Actually, c_k = sum_{i=0}^{N-1} h_i h_{k-i}
    # d(c_k)/d(h_j) = h_{k-j} + (if j != k-j then h_{k-j} else 2 h_j)?
    # No, simple derivative:
    # c_k = h_0 h_k + h_1 h_{k-1} + ...
    # d(c_k)/d(h_j) = h_{k-j} + h_{k-j} = 2 h_{k-j} ?
    # Let's check: c_k = sum_i h_i h_{k-i}.
    # Term involving h_j: i=j -> h_j h_{k-j}. Also i=k-j -> h_{k-j} h_j.
    # So if j != k-j, there are two terms: 2 h_j h_{k-j}. Derivative is 2 h_{k-j}.
    # If j == k-j (i.e., 2j=k), there is one term h_j^2. Derivative is 2 h_j.
    # In both cases, derivative is 2 h_{k-j} (with h_{k-j} defined as 0 if index out of bounds).
    
    # grad_obj_j = sum_k 2 c_k * d(c_k)/d(h_j) = sum_k 2 c_k * 2 h_{k-j} = 4 sum_k c_k h_{k-j}
    # This is 4 * (c * h)_j (cross-correlation of c and h? No, convolution)
    # Let's verify: sum_k c_k h_{k-j}.
    # This is the cross-correlation of c and h at lag j?
    # (c * h)_j = sum_k c_k h_{j-k}.
    # sum_k c_k h_{k-j} = sum_k c_k h_{-(j-k)} = (c * h_rev)_j?
    # Actually, it's simpler to just compute it via FFT or direct sum if N is small,
    # but N=400, c has length 799.
    # We can compute grad using FFT.
    # Let's use a simpler approach: numerical gradient? No, too slow.
    # Let's derive carefully.
    # obj = sum_k c_k^2
    # d(obj)/d(h_j) = 2 sum_k c_k (d c_k / d h_j)
    # d c_k / d h_j = h_{k-j} + h_{k-j} = 2 h_{k-j} (valid if 0 <= k-j < N, else 0)
    # Wait, c_k = sum_{i: 0<=i<N, 0<=k-i<N} h_i h_{k-i}
    # If j is in range, terms are:
    # 1. i=j: h_j h_{k-j} (if 0<=k-j<N)
    # 2. i=k-j: h_{k-j} h_j (if 0<=k-j<N and k-j != j)
    # So d c_k / d h_j = 2 h_{k-j} if 0<=k-j<N, else 0.
    # Exception: if k-j == j, i.e., k=2j, then the term is h_j^2, derivative 2 h_j.
    # The formula 2 h_{k-j} gives 2 h_j, which matches.
    # So d c_k / d h_j = 2 h_{k-j} (with h_m = 0 if m out of [0, N-1]).
    
    # Thus d(obj)/d(h_j) = 2 sum_k c_k * 2 h_{k-j} = 4 sum_k c_k h_{k-j}.
    # Let g_j = sum_k c_k h_{k-j}.
    # This is the cross-correlation of c and h?
    # (c \star h)_j = sum_k c_k h_{k-j}. Yes.
    # We can compute this via FFT.
    
    # c has length 2N-1. h has length N.
    # We want g of length N.
    # g_j = sum_{k=j}^{j+N-1} c_k h_{k-j}  (since h_{k-j} is 0 outside 0..N-1)
    # This is the convolution of c and reversed h?
    # Let's just compute it directly for clarity, or use np.correlate.
    # np.correlate(c, h, mode='same')? No, lengths differ.
    # Let's pad h to length 2N-1 and use FFT.
    
    # Actually, since N=400, direct computation might be fast enough for gradient?
    # 400 * 800 = 320,000 operations. That's fine.
    # But we are in an optimization loop, so FFT is better.
    
    # Let's use FFT.
    # g = cross_corr(c, h)
    # We can compute this by:
    # G = FFT(c padded), H = FFT(h padded)
    # But cross correlation is FFT(c) * conj(FFT(h)) ?
    
    # Let's stick to the QP solution using scipy.optimize.minimize with SLSQP.
    # We need gradient.
    
    # Let's compute gradient via FFT.
    # c_len = 2*N - 1
    # h_len = N
    # We want g_j = sum_k c_k h_{k-j} for j=0..N-1.
    # This is the "same" mode cross-correlation of c and h?
    # np.correlate(c, h, mode='full') gives length c_len + h_len - 1.
    # We want the part where the overlap is full h.
    # Specifically, g_j = sum_{i=0}^{N-1} h_i c_{i+j}.
    # This is the cross-correlation of c and h at lag j.
    # corr_j = sum_i h_i c_{i+j}.
    # This is exactly what we want.
    
    # We can compute this using FFT:
    # C = FFT(c, n=2*N)
    # H = FFT(h, n=2*N)
    # Corr = IFFT(C * conj(H)) ?
    # Let's check.
    # (C * conj(H))_k = sum_n C_n H_{n-k}^*
    # IFFT gives sum_k ...
    # Standard formula: cross_corr(x, y)_k = sum_n x_n y_{n+k}
    # FFT(x) * conj(FFT(y)) -> IFFT -> cross_corr(x, y)
    # So cross_corr(c, h)_k = sum_n c_n h_{n+k}.
    # This matches our g_k (with indices shifted).
    # g_j = sum_i h_i c_{i+j} = sum_n c_n h_{n+j} = cross_corr(c, h)_j.
    
    # So we can compute g via FFT.
    
    c_len = 2 * N - 1
    fft_len = 2 * N  # power of 2 >= c_len + N - 1? No, just >= c_len for padding.
    # Actually, to avoid circular convolution, we need FFT length >= c_len + N - 1?
    # No, cross-correlation via FFT:
    # Pad c and h to length L >= c_len + N - 1?
    # L >= 2N-1 + N-1 = 3N-2?
    # Let's just use L = 4*N to be safe.
    
    L = 4 * N
    C = np.fft.fft(c, L)
    H = np.fft.fft(h, L)
    G = np.fft.ifft(C * np.conj(H))
    g = G[:N]  # We want the first N lags?
    # cross_corr(c, h)_k for k=0..N-1.
    # The full cross-correlation has length c_len + N - 1.
    # The positive lags are at the end?
    # Let's check numpy.correlate.
    # np.correlate(c, h, 'full')
    # Let's just use np.correlate for simplicity and speed check.
    # N=400, c_len=799.
    # np.correlate is fast.
    
    g = np.correlate(c, h, mode='full')
    # We need g_j = sum_i h_i c_{i+j}.
    # np.correlate(c, h, 'full')[k] = sum_i c_i h_{i-k} ? Or h_i c_{i-k}?
    # Definition: (a * b)[k] = sum_n a[n] b[n-k]
    # np.correlate(a, v) = sum_n a[n+k] v[n]
    # So np.correlate(c, h, 'full')[k] = sum_n c[n+k] h[n]
    # This matches g_k = sum_n h[n] c[n+k].
    # So g = np.correlate(c, h, 'full')
    # We need g[0:N] ?
    # k ranges from -(N-1) to c_len-1.
    # We want k=0 to N-1.
    # So g[0:N] is correct.
    
    grad = 4.0 * g[:N]
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: Gaussian-like
    x = np.linspace(-1, 1, N)
    h0 = np.exp(-10 * x**2)
    h0 /= np.sum(h0)
    
    # Constraints: sum(h) = 1
    cons = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 
             'jac': lambda h: np.ones(N)}]
    
    # Objective: minimize sum(c^2)
    def fun(h):
        return _objective_and_grad(h)[0]
    
    def jac(h):
        return _objective_and_grad(h)[1]
    
    best_h = h0
    best_val = _evaluate(h0)
    
    # Try a few different initial points and methods
    initial_points = [
        h0,
        np.ones(N) / N,
        np.sin(np.pi * np.linspace(0, 1, N)) / np.sum(np.sin(np.pi * np.linspace(0, 1, N))),
        np.random.normal(0, 1, N),
        np.random.normal(0, 1, N)
    ]
    
    for h_init in initial_points:
        if time.time() > deadline:
            break
        
        h_init = h_init.copy()
        h_init /= np.sum(h_init)
        
        try:
            res = minimize(fun, h_init, method='SLSQP', jac=jac, 
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success or res.fun < best_val:
                h_res = res.x
                h_res /= np.sum(h_res)
                val = _evaluate(h_res)
                if val < best_val:
                    best_val = val
                    best_h = h_res
        except Exception as e:
            pass
        
        if time.time() > deadline:
            break
    
    # Post-processing: try to reduce max(c) further
    # The QP minimizes L2 norm, but we care about max(c).
    # We can do a local search to reduce the peak.
    
    # Simple local search: adjust h to reduce the largest c_k
    h = best_h.copy()
    for _ in range(500):
        if time.time() > deadline:
            break
        
        c = np.convolve(h, h, mode='full')
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        
        # Compute gradient of c[max_idx] w.r.t. h
        # c_k = sum_i h_i h_{k-i}
        # d(c_k)/d(h_j) = 2 h_{k-j}
        grad_peak = 2.0 * np.zeros(N)
        for j in range(N):
            k_j = max_idx - j
            if 0 <= k_j < N:
                grad_peak[j] = 2.0 * h[k_j]
        
        # We want to decrease c[max_idx] while keeping sum(h)=1.
        # Project gradient onto constraint: grad_proj = grad_peak - mean(grad_peak)
        grad_proj = grad_peak - np.mean(grad_peak)
        
        # Take a small step in negative gradient direction
        step_size = 0.01
        h_new = h - step_size * grad_proj
        h_new /= np.sum(h_new)
        
        val_new = _evaluate(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new
            h = h_new
        else:
            step_size *= 0.5
    
    return best_h

# EVOLVE-BLOCK-END
