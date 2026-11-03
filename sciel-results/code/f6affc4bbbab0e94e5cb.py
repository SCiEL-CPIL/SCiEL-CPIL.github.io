import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE_SECONDS = 115.0  # Safety margin against 120s limit

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    """Off-diagonal metric: max |c_k| for k!=0 / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _generate_m_sequence(n, poly, init):
    """Generate an m-sequence of length n using a linear feedback shift register."""
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])

def _refine_subgrad(x0, deadline, rng, iters=5000, beta=50.0, mode='offdiag'):
    """Projected subgradient minimization of autoconvolution peaks.
    
    If mode='offdiag', minimizes max |c_k| for k != 0.
    If mode='std', minimizes max |c_k| for all k (but c0 is dominated by sum(x^2)).
    
    We fix sum(x)=1 to handle scale invariance.
    """
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        x = x + 1e-9
        s = np.sum(x)
    x = x / s
    
    best_val = _metric_std(x) if mode == 'std' else _metric_offdiag(x)
    bestx = x.copy()
    lr = 5e-4
    no_improve = 0
    patience = 500
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        c = _autoconv_lin(x)
        
        if mode == 'offdiag':
            if len(c) <= 1:
                break
            abs_c = np.abs(c[1:])
            if np.max(abs_c) < 1e-15:
                break
            # Soft max weights for smoother gradient
            m = np.max(abs_c)
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            
            g = np.zeros(n)
            # Only iterate over lags with significant weight to save time
            # Since N=400, 399 lags is small.
            for k in range(1, n):
                wk = weights[k - 1]
                if wk < 1e-6:
                    continue
                sign_ck = np.sign(c[k])
                # Gradient of c_k = sum_i x_i x_{i+k}
                # dc_k/dx_i = x_{i+k} (for i < n-k) + x_{i-k} (for i >= k)
                # Note: c_k = sum_{i=0}^{n-k-1} x_i x_{i+k}
                # d/dx_j c_k = x_{j+k} (if j+k < n) + x_{j-k} (if j-k >= 0)
                if k < n:
                    g[:n - k] += wk * sign_ck * x[k:]
                    g[k:] += wk * sign_ck * x[:n - k]
        else:
            # Standard metric: include k=0?
            # c0 = sum(x^2). Gradient is 2x.
            # Usually c0 is the max, so we'd just shrink x. But we fixed sum=1.
            # So minimizing max|c| with sum=1 usually means minimizing off-diagonal peaks
            # because c0 >= 1 (by Cauchy-Schwarz, c0 >= 1/N * 1^2? No, c0 >= S^2/N = 1/N).
            # If we normalize sum=1, c0 >= 1/N. For N=400, 1/400 = 0.0025.
            # Off-diagonal peaks can be much smaller.
            # So 'std' mode is tricky because c0 is a hard floor unless we change magnitudes.
            # But the metric is max|c|/S^2. If we don't fix sum, we can scale.
            # Let's stick to offdiag for the heavy lifting, as it's the bottleneck.
            abs_c = np.abs(c)
            m = np.max(abs_c)
            if m < 1e-15:
                break
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            
            g = np.zeros(n)
            for k in range(n):
                wk = weights[k]
                if wk < 1e-6:
                    continue
                sign_ck = np.sign(c[k])
                if k == 0:
                    g += wk * sign_ck * 2 * x
                else:
                    if k < n:
                        g[:n - k] += wk * sign_ck * x[k:]
                        g[k:] += wk * sign_ck * x[:n - k]

        x = x - lr * g
        # Project to sum=1
        x = x - (np.sum(x) - 1.0) / n
        # Keep magnitudes bounded
        x = np.clip(x, -2.0, 2.0)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric_std(x) if mode == 'std' else _metric_offdiag(x)
        if v < best_val:
            best_val = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.02, 2e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.85
                no_improve = 0
                if lr < 1e-6:
                    break
                    
    return bestx, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val_std = _metric_std(best_x)
    
    # Candidate 1: Constant sequence
    cand_const = np.ones(N)
    v_const = _metric_std(cand_const)
    if v_const < best_val_std:
        best_val_std = v_const
        best_x = cand_const
        
    # Generate diverse initializations
    candidates = []
    
    # 1. Random signs
    for _ in range(5):
        x_rand = rng.choice([-1.0, 1.0], size=N).astype(float)
        s = np.sum(x_rand)
        if abs(s) > 1e-6:
            x_rand = x_rand / s
        candidates.append(x_rand)
        
    # 2. M-sequences
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3],
             [8, 4], [9, 5], [10, 7], [11, 6], [12, 8], [13, 4], [14, 10], [15, 12]]
    for poly in polys:
        if time.time() > deadline:
            break
        for init in range(min(4, (1 << len(poly)) - 1)):
            seq = _generate_m_sequence(N, poly, init + 1)
            s = np.sum(seq)
            if abs(s) > 1e-6:
                seq = seq / s
            candidates.append(seq)
            
    # 3. Low frequency sinusoids (signed)
    for freq in [1, 2, 3, 5, 8]:
        t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        x_sin = np.sign(np.sin(t))
        s = np.sum(x_sin)
        if abs(s) > 1e-6:
            x_sin = x_sin / s
        candidates.append(x_sin)

    # Refine each candidate
    for x_init in candidates:
        if time.time() > deadline:
            break
            
        # Try offdiag refinement
        x_opt, v_opt = _refine_subgrad(x_init, deadline, rng, iters=2000, mode='offdiag')
        
        # Check standard metric for the refined candidate
        v_std = _metric_std(x_opt)
        if v_std < best_val_std:
            best_val_std = v_std
            best_x = x_opt
            
        # Also try a quick L-BFGS polish on the best few if time permits
        # But L-BFGS can be slow. Let's stick to subgrad for now.
        
    # Final polish on the best candidate
    if time.time() < deadline - 10:
        x_final, v_final = _refine_subgrad(best_x, deadline, rng, iters=5000, mode='offdiag', beta=100.0)
        v_std_final = _metric_std(x_final)
        if v_std_final < best_val_std:
            best_val_std = v_std_final
            best_x = x_final
            
    return best_x
