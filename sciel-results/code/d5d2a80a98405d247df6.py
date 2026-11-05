import numpy as np
import time

def _eval_at(c, t):
    """Evaluate |P(exp(2j*pi*t))| using Horner's method."""
    z = np.exp(2j * np.pi * t)
    acc = 0j
    for coeff in reversed(c):
        acc = acc * z + coeff
    return np.abs(acc)

def _peak_fast(c):
    """Fast peak estimate via FFT (no refinement)."""
    n = len(c)
    N = 4096
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    return np.abs(F).max()

def _peak(c):
    """Accurate peak: dense FFT sampling + golden-section refinement of top peaks."""
    n = len(c)
    N = 8192
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)

    top_k = 8
    idx = np.argpartition(mags, -top_k)[-top_k:]
    best_peak = float(mags[idx].max())

    gr = (np.sqrt(5) - 1) / 2
    for k in idx:
        lo = max(0.0, (k - 5) / N)
        hi = min(1.0, (k + 6) / N)
        a, b = lo, hi
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = _eval_at(c, c1)
        f2 = _eval_at(c, c2)
        for _ in range(35):
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = _eval_at(c, c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = _eval_at(c, c1)
        v = max(f1, f2)
        if v > best_peak:
            best_peak = v
    return best_peak

def _softmax_peak(c, N=2048, beta=20.0):
    """Compute soft-max of |P(z)| over N points on unit circle.
    Returns (softmax_value, gradient_wrt_coefficients).
    
    soft_max = (1/beta) * log(sum_j exp(beta * |P(z_j)|))
    gradient = (1/beta) * sum_j exp(beta*|P(z_j)|) * d|P(z_j)|/d(c_k) / sum_j exp(beta*|P(z_j)|)
    """
    z = np.exp(2j * np.pi * np.arange(N) / N)
    # P(z_j) = sum_k c_k z_j^k
    # |P(z_j)| = sqrt(Re(P)^2 + Im(P)^2)
    # d|P|/dc_k = Re(z_j^k * conj(P)) / |P|
    
    # Compute P(z_j) for all j
    # P = c @ Z where Z[j,k] = z_j^k
    # For efficiency, use FFT-like approach
    cf = np.zeros(N, dtype=complex)
    cf[:len(c)] = c
    F = np.fft.ifft(cf) * N  # F[j] = P(z_j)
    
    mags = np.abs(F)
    # Avoid division by zero
    mags_safe = np.where(mags > 1e-10, mags, 1.0)
    
    # Soft-max
    max_mag = np.max(mags)
    exp_args = beta * (mags - max_mag)
    exp_vals = np.exp(exp_args)
    sum_exp = np.sum(exp_vals)
    softmax_val = max_mag + np.log(sum_exp) / beta
    
    # Gradient: d(softmax)/dc_k = (1/beta) * sum_j exp_args[j] * Re(z_j^k * conj(F[j])) / (mags_safe[j] * sum_exp)
    # For each k, we need sum_j exp_vals[j] * Re(z_j^k * conj(F[j])) / mags_safe[j]
    # z_j^k = exp(2j*pi*j*k/N)
    # This is essentially an FFT of the weighted sequence
    
    weights = exp_vals / (mags_safe * sum_exp)
    # grad_k = (1/beta) * Re( sum_j weights[j] * conj(F[j]) * z_j^k )
    # Let g[j] = weights[j] * conj(F[j])
    # Then grad_k = (1/beta) * Re( FFT(g)[k] ) ... but careful with FFT conventions
    # sum_j g[j] * z_j^k where z_j = exp(2j*pi*j/N)
    # This is N * ifft(g)[k] in numpy convention? Let's check:
    # numpy ifft: ifft(x)[k] = (1/N) sum_j x[j] exp(2j*pi*j*k/N)
    # So sum_j x[j] exp(2j*pi*j*k/N) = N * ifft(x)[k]
    
    g = weights * np.conj(F)
    grad_fft = np.fft.ifft(g) * N
    grad = np.real(grad_fft) / beta
    
    # Only first len(c) coefficients matter
    grad = grad[:len(c)]
    
    return softmax_val, grad

def _smooth_search(c0, deadline, rng, max_iter=2000, beta_start=10.0, beta_end=100.0):
    """Smooth gradient-based search with sign projection.
    
    Start from c0 (±1 vector), iterate:
    1. Compute soft-max objective and gradient
    2. Update: x = x - lr * grad
    3. Project: c = sign(x)
    4. Anneal beta (increase to make soft-max closer to hard max)
    """
    c = c0.copy().astype(float)
    x = c.copy()
    n = len(c)
    
    best_c = c.copy()
    best_soft = _softmax_peak(c, beta=beta_start)[0]
    
    lr = 0.5
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        beta = beta_start + (beta_end - beta_start) * (it / max_iter)
        val, grad = _softmax_peak(c, beta=beta)
        
        if val < best_soft - 1e-6:
            best_soft = val
            best_c = c.copy()
        
        # Gradient step
        x = x - lr * grad
        
        # Project to ±1
        c_new = np.sign(x)
        # Handle zeros
        c_new[c_new == 0] = 1
        
        # Accept if improved or with small probability (simulated annealing style)
        val_new, _ = _softmax_peak(c_new, beta=beta)
        if val_new < val or rng.random() < 0.1 * np.exp(-(val_new - val) * 5):
            c = c_new
            x = c
        else:
            # Keep old c, adjust x to pull toward sign
            x = 0.9 * x + 0.1 * c
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    return best_c, _peak_fast(best_c)

def _local_search(c, deadline, rng):
    """Local search with 1-bit and 2-bit flips."""
    c = c.copy()
    current_peak = _peak_fast(c)
    n = len(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_flip_peak = current_peak
        best_c_trial = None
        # 1-bit flips
        for i in range(n):
            if time.time() > deadline:
                break
            c[i] = -c[i]
            p_trial = _peak_fast(c)
            if p_trial < best_flip_peak:
                best_flip_peak = p_trial
                best_c_trial = c.copy()
            else:
                c[i] = -c[i]
        # 2-bit flips
        if time.time() < deadline:
            for _ in range(4):
                if time.time() > deadline:
                    break
                i, j = rng.choice(n, size=2, replace=False)
                c[i] = -c[i]; c[j] = -c[j]
                p_trial = _peak_fast(c)
                if p_trial < best_flip_peak:
                    best_flip_peak = p_trial
                    best_c_trial = c.copy()
                else:
                    c[i] = -c[i]; c[j] = -c[j]
        if best_c_trial is not None:
            c = best_c_trial
            current_peak = best_flip_peak
            improved = True
        else:
            break
    return c, _peak(c)

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    n = 24
    rng = np.random.default_rng(42)
    candidates = []

    # Structured / known-good patterns
    candidates.append(np.array([1]*n, dtype=float))
    candidates.append(np.array([1,-1]*12, dtype=float))
    candidates.append(np.array([1,1,-1,-1]*6, dtype=float))
    candidates.append(np.array([1,-1,-1]*8, dtype=float))
    candidates.append(np.array([1,1,1,-1,-1,-1]*4, dtype=float))
    candidates.append(np.array([1,-1,1,1,-1,1,1,-1]*3, dtype=float))
    candidates.append(np.array([1,1,1,1,-1,-1,-1,-1]*3, dtype=float))
    candidates.append(np.array([1,-1,-1,-1,1,1,1,1]*3, dtype=float))
    # GRS construction
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-x for x in q]
    candidates.append(np.array(p[:n], dtype=float))

    # Diverse random restarts
    for seed in range(40):
        r = np.random.default_rng(seed)
        candidates.append(r.choice([-1.0, 1.0], size=n))

    # Phase 1: Fast screening
    screened = []
    for c in candidates:
        if time.time() > deadline - 40:
            break
        pk = _peak_fast(c)
        screened.append((pk, c))
    screened.sort(key=lambda x: x[0])
    top = [c for _, c in screened[:12]]

    best_c = top[0]
    best_score = _peak(best_c)

    # Phase 2: Smooth search on top candidates
    for c in top:
        if time.time() > deadline - 15:
            break
        c_opt, pk_opt = _smooth_search(c, deadline - 10, rng, max_iter=3000)
        if pk_opt < best_score:
            best_score = pk_opt
            best_c = c_opt
        if best_score < 6.162:
            break

    # Phase 3: Local search polish on best
    if time.time() < deadline - 5:
        c_polished, pk_polished = _local_search(best_c, deadline - 2, rng)
        if pk_polished < best_score:
            best_score = pk_polished
            best_c = c_polished

    if best_c is None:
        best_c = candidates[0]
    return [int(x) for x in best_c]
