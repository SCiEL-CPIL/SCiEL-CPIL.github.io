"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(f):
    """Compute full linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    return ac


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)


def _smooth_max_and_grad(ac, beta):
    """
    Compute smooth approximation of max|ac| and its gradient w.r.t. ac.
    Uses log-sum-exp on |ac| values.
    Returns (smooth_max, grad_wrt_ac) where grad_wrt_ac[i] = d(smooth_max)/d(ac[i])
    """
    x = np.abs(ac)
    m = np.max(x)
    # Log-sum-exp: log(sum(exp(x - m))) + m
    log_sum = np.log(np.sum(np.exp(x - m))) + m
    # Weights: exp(x - m) / sum(exp(x - m))
    weights = np.exp(x - m)
    weights /= np.sum(weights)
    # Gradient of log_sum w.r.t. x_i = weights[i]
    # Gradient of log_sum w.r.t. ac_i = weights[i] * sign(ac[i])
    grad = weights * np.sign(ac)
    # For the ratio objective: d(max|ac|/s^2)/df = (grad_ac * s^2 - max|ac|*2*s*1)/s^4
    # But we'll handle the ratio in the outer function
    return log_sum, grad, m


def _objective_and_grad(f, beta=50.0):
    """
    Compute the smooth objective max|autoconv|/s^2 and its gradient w.r.t. f.
    Uses log-sum-exp approximation for the max.
    """
    n = len(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf, np.zeros(n)
    
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    
    smooth_max, grad_ac, true_max = _smooth_max_and_grad(ac, beta)
    
    # The autoconvolution ac = f * f (linear convolution)
    # ac[k] = sum_i f[i] * f[k-i]
    # d(ac[k])/df[j] = f[k-j] + f[j-k] (for valid k, j)
    # So d(smooth_max)/df[j] = sum_k grad_ac[k] * (f[k-j] + f[j-k])
    
    # This is a convolution of grad_ac with f, plus the reverse
    # Let g = grad_ac (length 2n-1)
    # d(smooth_max)/df[j] = sum_k g[k] * f[k-j] + g[k] * f[j-k]
    # = (g * f)[j] + (g_rev * f)[j] where g_rev[k] = g[2n-2-k]
    
    # Compute via FFT
    g = grad_ac
    # First term: (g * f)[j] for j in [0, n-1]
    # g has length 2n-1, f has length n
    # Linear convolution of g and f has length 3n-2
    # We want indices 0 to n-1
    
    conv_size = 1
    while conv_size < (2 * n - 1) + n - 1:
        conv_size *= 2
    
    G = np.fft.rfft(g, conv_size)
    Ff = np.fft.rfft(f, conv_size)
    conv1 = np.fft.irfft(G * Ff, conv_size)[:n]
    
    # Second term: (g_rev * f)[j]
    g_rev = g[::-1]
    G_rev = np.fft.rfft(g_rev, conv_size)
    conv2 = np.fft.irfft(G_rev * Ff, conv_size)[:n]
    
    grad_smooth = conv1 + conv2
    
    # Now the ratio: obj = smooth_max / s^2
    # d(obj)/df[j] = (grad_smooth[j] * s^2 - smooth_max * 2 * s) / s^4
    # = (grad_smooth[j] * s - 2 * smooth_max) / s^3
    
    obj = smooth_max / (s * s)
    grad = (grad_smooth * s - 2 * smooth_max) / (s ** 3)
    
    return obj, grad


def _gradient_descent(f, deadline, max_iters=5000, lr=0.01, beta=50.0):
    """
    Gradient descent on the smooth objective with adaptive learning rate.
    """
    current = f.copy()
    best = f.copy()
    best_obj = _score(f)
    
    lr_current = lr
    patience = 0
    best_local = best_obj
    
    for it in range(max_iters):
        if time.time() > deadline - 15.0:
            break
        
        obj, grad = _objective_and_grad(current, beta)
        
        if not np.isfinite(obj):
            current = f.copy()
            continue
        
        # Gradient step
        current = current - lr_current * grad
        
        # Renormalize to keep sum = N
        s = np.sum(current)
        if abs(s) < 1e-10:
            current = f.copy()
            continue
        current = current * (N / s)
        
        # Evaluate true score
        true_score = _score(current)
        if true_score < best_obj:
            best_obj = true_score
            best = current.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate
        if patience > 200:
            lr_current *= 0.5
            patience = 0
            if lr_current < 0.001:
                break
    
    return best, best_obj


def _local_search_ones(f, deadline, max_iters=20000):
    """
    Fast coordinate descent on ±1 sequences.
    Try flipping individual elements and accept if score improves.
    """
    current = np.sign(f).astype(float)
    current[current == 0] = 1.0
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    np.random.seed(123)
    indices = np.arange(N)
    
    for it in range(max_iters):
        if time.time() > deadline - 15.0:
            break
        
        # Pick a random index
        idx = np.random.randint(N)
        old_val = current[idx]
        new_val = -old_val
        
        # Quick estimate: compute new autoconvolution efficiently
        # ac_new = ac_old - 2*old_val*(f * delta_idx) + ... 
        # Actually, let's just recompute since N=400 is small
        current[idx] = new_val
        
        s = np.sum(current)
        if abs(s) < 1e-12:
            current[idx] = old_val
            continue
        
        ac = _autoconv(current)
        sc = np.max(np.abs(ac)) / (s * s)
        
        if sc < current_score:
            current_score = sc
            if sc < best_score:
                best_score = sc
                best = current.copy()
        else:
            current[idx] = old_val
    
    return best, best_score


def _random_search(deadline, max_trials=100):
    """Try random ±1 sequences."""
    np.random.seed(42)
    best = None
    best_score = np.inf
    
    for _ in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        rand_seq = np.random.choice([-1.0, 1.0], N)
        s = np.sum(rand_seq)
        if abs(s) < 1e-12:
            continue
        sc = _score(rand_seq)
        if sc < best_score:
            best_score = sc
            best = rand_seq
    
    return best, best_score


def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    # Start with all ones
    best = np.ones(N)
    best_score = _score(best)
    
    # Phase 1: Gradient descent on continuous relaxation
    f = best.copy().astype(float)
    f_opt, obj_opt = _gradient_descent(f, deadline, max_iters=3000, lr=0.02, beta=30.0)
    
    sc_opt = _score(f_opt)
    if sc_opt < best_score:
        best_score = sc_opt
        best = f_opt.copy()
    
    # Phase 2: Round to ±1 and do local search
    rounded = np.sign(best)
    rounded[rounded == 0] = 1.0
    rs = _score(rounded)
    if rs < best_score:
        best_score = rs
        best = rounded.copy()
    
    # Phase 3: Local search on ±1
    f_ls, ls_score = _local_search_ones(best, deadline, max_iters=15000)
    if ls_score < best_score:
        best_score = ls_score
        best = f_ls.copy()
    
    # Phase 4: More gradient descent from the rounded solution
    if time.time() < deadline - 30.0:
        f2 = best.copy().astype(float)
        f_opt2, obj_opt2 = _gradient_descent(f2, deadline, max_iters=2000, lr=0.01, beta=50.0)
        sc_opt2 = _score(f_opt2)
        if sc_opt2 < best_score:
            best_score = sc_opt2
            best = f_opt2.copy()
        
        # Round again and local search
        rounded2 = np.sign(best)
        rounded2[rounded2 == 0] = 1.0
        rs2 = _score(rounded2)
        if rs2 < best_score:
            best_score = rs2
            best = rounded2.copy()
        
        f_ls2, ls_score2 = _local_search_ones(best, deadline, max_iters=10000)
        if ls_score2 < best_score:
            best_score = ls_score2
            best = f_ls2.copy()
    
    # Phase 5: Random search
    if time.time() < deadline - 10.0:
        rand_best, rand_score = _random_search(deadline, max_trials=50)
        if rand_best is not None and rand_score < best_score:
            best_score = rand_score
            best = rand_best
    
    # Final: ensure output is ±1
    result = np.sign(best)
    result[result == 0] = 1.0
    
    return result
# EVOLVE-BLOCK-END
