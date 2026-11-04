"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 115.0  # Safety margin before 120s kill

def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3 for a signed step function f with N intervals.
    Returns max |autoconvolution| excluding the zero-lag self term.
    """
    n = len(f)
    # Use rfft for real signals for efficiency
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)
    conv_lags = conv[:n]
    if len(conv_lags) > 1:
        return np.max(np.abs(conv_lags[1:]))
    return 0.0

def _log_sum_exp_max_autoconv(f, beta):
    """
    Compute the Log-Sum-Exp approximation of the max absolute autoconvolution.
    Returns the value and the gradient with respect to f.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)[:n]
    
    if len(conv) <= 1:
        return 0.0, np.zeros_like(f)
    
    abs_conv = np.abs(conv[1:])
    signs = np.sign(conv[1:])
    
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0, np.zeros_like(f)
    
    # Stabilize LSE
    exp_terms = np.exp(beta * (abs_conv - max_val))
    sum_exp = np.sum(exp_terms)
    lse_val = (max_val + np.log(sum_exp)) / beta
    
    # Weights for the soft-max
    weights = exp_terms / sum_exp
    
    # Gradient calculation:
    # We want grad of sum_k weights_k * |C_k|
    # C_k = sum_i f_i f_{i+k}
    # d C_k / d f_j = f_{j+k} + f_{j-k} (with linear convolution boundary handling)
    # Let G_k = weights_k * sign(C_k)
    # grad_j = sum_{k != 0} G_k * (f_{j+k} + f_{j-k})
    
    # This is a convolution of f with a kernel H where H_k = G_k.
    # Since H is symmetric (H_{-k} = G_k), the operator is a convolution.
    # We can compute this via FFT as well, but for N=400, direct loop is O(N^2) which is 160k ops.
    # 500 steps * 160k = 80M ops. NumPy handles this easily in < 1s.
    
    G = np.zeros(N)
    G[1:] = weights * signs
    
    grad = np.zeros(N)
    # Vectorized loop over lags
    for k in range(1, N):
        wk = G[k]
        if wk == 0:
            continue
        # Contribution to grad[j] from lag k:
        # wk * f[j+k] for j < N-k
        # wk * f[j-k] for j >= k
        grad[:N-k] += wk * f[k:]
        grad[k:] += wk * f[:N-k]
        
    return lse_val, grad

def _construct_thue_morse(n):
    """Generate Thue-Morse sequence of length n mapped to {-1, 1}."""
    # t_n = parity of number of 1s in binary representation of n
    # Map 0 -> 1, 1 -> -1
    return 1.0 - 2.0 * (np.array([bin(i).count('1') for i in range(n)], dtype=int) % 2)

def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Initialization: Thue-Morse is a good starting point for low autocorrelation
    best_f = _construct_thue_morse(N)
    best_score = _compute_c3_upper_bound(best_f)
    
    # 2. Continuous Relaxation Optimization
    # We optimize a continuous vector f_cont, then project to signs.
    # This allows us to use gradient descent on the smooth LSE surrogate.
    
    # Start continuous vector near the Thue-Morse signs
    f_cont = best_f.copy()
    
    beta = 5.0
    lr = 0.05
    
    # Adaptive learning rate and beta
    steps = 0
    max_steps = 2000
    
    # Optimization loop
    while steps < max_steps and time.time() < deadline - 15:
        loss, grad = _log_sum_exp_max_autoconv(f_cont, beta)
        
        # Gradient step
        f_cont -= lr * grad
        
        # Project to signs
        f_disc = np.sign(f_cont)
        f_disc[f_disc == 0] = 1.0
        
        # Evaluate discrete score
        score_disc = _compute_c3_upper_bound(f_disc)
        
        if score_disc < best_score:
            best_score = score_disc
            best_f = f_disc.copy()
            
        # Adaptivity
        steps += 1
        if steps % 50 == 0:
            beta *= 1.2
            lr *= 0.95
            
        # Periodic random perturbation to escape local minima
        if steps % 100 == 0:
            mask = np.random.random(N) < 0.1
            f_cont[mask] += np.random.normal(0, 0.5, mask.sum())

    # 3. Discrete Local Search (Greedy Flip)
    # Polish the result by flipping individual signs if it improves the score.
    # This is O(N^2) per pass, but we limit the number of passes.
    
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try flipping each index
            # Optimization: We can compute the effect of flipping index i
            # on the autocorrelation, but for simplicity and robustness,
            # we recompute. N=400 is small.
            # To speed up, we can check indices in random order or just iterate.
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                old_val = best_f[idx]
                best_f[idx] = -old_val
                new_score = _compute_c3_upper_bound(best_f)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    improved = True
                else:
                    best_f[idx] = old_val

    return best_f

# EVOLVE-BLOCK-END
