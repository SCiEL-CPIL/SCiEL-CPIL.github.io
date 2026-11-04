import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE_SECONDS = 140.0

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric(x):
    """Max |c_k| / S^2 for k != 0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) <= 1:
        return 0.0
    return float(np.max(np.abs(c[1:]))) / (s * s)

def _objective(x, beta=30.0):
    """Soft-max surrogate for off-diagonal max |c_k| / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) <= 1:
        return 0.0
    ac = np.abs(c[1:])
    if len(ac) == 0:
        return 0.0
    m = np.max(ac)
    if m < 1e-15:
        return 0.0
    shifted = beta * (ac - m)
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    return soft_max / (s * s)

def _grad_objective(x, beta=30.0):
    """Gradient of soft-max off-diagonal objective."""
    n = len(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.zeros(n), 1e10
    c = _autoconv(x)
    if len(c) <= 1:
        return np.zeros(n), 0.0
    ac = np.abs(c[1:])
    if len(ac) == 0:
        return np.zeros(n), 0.0
    m = np.max(ac)
    if m < 1e-15:
        return np.zeros(n), 0.0
    shifted = beta * (ac - m)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max_val = m + np.log(sum_exp) / beta
    obj = soft_max_val / (s * s)
    
    g = np.zeros(n)
    for idx, k in enumerate(range(1, len(c))):
        wk = weights[idx]
        if wk < 1e-8:
            continue
        sign_ck = np.sign(c[k])
        g[:n - k] += wk * sign_ck * x[k:]
        g[k:] += wk * sign_ck * x[:n - k]
    
    g_obj = g / (s * s) - 2 * soft_max_val / (s * s * s)
    return g_obj, obj

def _project_sum_one(x):
    s = np.sum(x)
    if abs(s) < 1e-15:
        return x
    return x - (s - 1.0) / len(x)

def _refine_smooth(x0, deadline, beta_start=20.0, beta_end=80.0, iters=2000):
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        x = x + 1e-9
        s = np.sum(x)
    x = x / s
    best_val = _metric(x)
    bestx = x.copy()
    lr = 1e-3
    no_improve = 0
    patience = 150
    
    for it in range(iters):
        if time.time() > deadline:
            break
        progress = it / max(1, iters)
        beta = beta_start + (beta_end - beta_start) * progress
        g, obj = _grad_objective(x, beta=beta)
        x_new = x - lr * g
        x_new = _project_sum_one(x_new)
        x_new = np.clip(x_new, -3.0, 3.0)
        x_new = _project_sum_one(x_new)
        v = _metric(x_new)
        if v < best_val:
            best_val = v
            bestx = x_new.copy()
            x = x_new
            no_improve = 0
            lr = min(lr * 1.03, 3e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.75
                no_improve = 0
                if lr < 1e-7:
                    break
    return bestx, best_val

def _generate_m_sequence(n, poly, init):
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    rng = np.random.default_rng(42)
    
    # Start with a reasonable baseline
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Generate diverse initializations
    candidates = []
    
    # Random signs
    for _ in range(8):
        x_rand = rng.choice([-1.0, 1.0], size=N).astype(float)
        s = np.sum(x_rand)
        if abs(s) > 1e-6:
            x_rand = x_rand / s
        candidates.append(x_rand)
    
    # Low-frequency sinusoids
    for freq in [1, 2, 3, 4, 5, 7, 8, 11, 13]:
        t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        x_sin = np.sign(np.sin(t))
        s = np.sum(x_sin)
        if abs(s) > 1e-6:
            x_sin = x_sin / s
        candidates.append(x_sin)
    
    # M-sequences
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3],
             [8, 4], [9, 5], [10, 7], [11, 6], [12, 8], [13, 4], [14, 10], [15, 12]]
    for poly in polys:
        if time.time() > deadline - 40:
            break
        for init in range(min(2, (1 << len(poly)) - 1)):
            seq = _generate_m_sequence(N, poly, init + 1)
            s = np.sum(seq)
            if abs(s) > 1e-6:
                seq = seq / s
            candidates.append(seq)
    
    # Refine each candidate
    for x_init in candidates:
        if time.time() > deadline - 25:
            break
        x_opt, v_opt = _refine_smooth(x_init, deadline, iters=1500)
        v_std = _metric(x_opt)
        if v_std < best_val:
            best_val = v_std
            best_x = x_opt
    
    # Final polish
    if time.time() < deadline - 10:
        x_final, v_final = _refine_smooth(best_x, deadline, iters=2000, beta_start=40.0, beta_end=150.0)
        v_final_std = _metric(x_final)
        if v_final_std < best_val:
            best_val = v_final_std
            best_x = x_final
    
    return best_x
