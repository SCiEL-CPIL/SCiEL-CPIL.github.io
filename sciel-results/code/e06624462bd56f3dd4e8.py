"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3 for a signed step function f with N intervals.
    Returns max |autoconvolution| excluding the zero-lag self term.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real
    conv_lags = conv[:n]
    if len(conv_lags) > 1:
        return np.max(np.abs(conv_lags[1:]))
    return 0.0

def _lse_max_autoconv(f, beta):
    """
    Compute the Log-Sum-Exp approximation of the max absolute autoconvolution.
    This provides a smooth surrogate for the peak minimization objective.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # Exclude lag 0
    if len(conv) <= 1:
        return 0.0, np.zeros_like(f)
    
    abs_conv = np.abs(conv[1:])
    
    # To avoid overflow in exp, subtract max
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0, np.zeros_like(f)
    
    exp_terms = np.exp(beta * (abs_conv - max_val))
    sum_exp = np.sum(exp_terms)
    lse_val = (max_val + np.log(sum_exp)) / beta
    
    # Gradient computation
    # d(LSE)/d(f_j) = sum_k (d|C_k|/d(f_j)) * (exp(beta|C_k|) / sum(exp(beta|C_k|)))
    # Let w_k = exp_terms[k-1] / sum_exp
    weights = exp_terms / sum_exp
    
    # We need d|C_k|/d(f_j).
    # C_k = sum_i f_i f_{i+k}.
    # d C_k / d f_j = f_{j+k} + f_{j-k} (circular/linear handling needed).
    # Since we are doing linear convolution (zero-padded), the derivative is:
    # grad_f = 2 * (f * f)_lag? No.
    # Let's use the property that grad of ||f*f||_2^2 is 2*(f*f)*f.
    # For max, it's a weighted sum of gradients of each C_k.
    
    # Efficient way:
    # grad_k = d|C_k|/df. 
    # |C_k| sign is sign(C_k).
    # d C_k / d f_j = f_{j+k} + f_{j-k} (with 0 padding outside [0, N-1]).
    
    # We can compute the gradient via FFT as well?
    # Let G_k = sign(C_k) * w_k.
    # grad_f = sum_k G_k * (d C_k / d f).
    # This is equivalent to a convolution of G with f?
    # d C_k / d f is the kernel h_k where h_k[j] = 1 if j=k or j=-k?
    # Actually, C_k = <f, shift_k(f)>.
    # grad_f = sum_k G_k * shift_{-k}(f) + shift_k(f)?
    
    # Let's just do it directly for N=400, it's fast enough.
    # grad = zeros(N)
    # for k in range(1, N):
    #     wk = weights[k-1] * sign(conv[k])
    #     # d C_k / d f_j = f_{j+k} + f_{j-k}
    #     # This is 2 * (f shifted by k) + (f shifted by -k)?
    #     # Let's verify: C_k = sum_i f_i f_{i+k}.
    #     # d/df_j (sum_i f_i f_{i+k}) = f_{j+k} (from i=j) + f_{j-k} (from i+k=j => i=j-k).
    #     # So term is f_{j+k} + f_{j-k}.
    #     # This is equivalent to 2 * (f * delta_k) if we consider symmetry?
    #     # f_{j+k} is shift left by k. f_{j-k} is shift right by k.
    #     # So grad += wk * (np.roll(f, -k) + np.roll(f, k)) ? 
    #     # Careful with boundaries. Linear conv means f is 0 outside.
    #     # np.roll wraps around, so we must mask.
    
    # Faster: Use FFT for the gradient too?
    # grad_f = IFFT( FFT(f) * (FFT(sign(C) * w) * 2 ?) )
    # The gradient of the quadratic form C_k = f^T A_k f is (A_k + A_k^T) f.
    # A_k is the shift matrix. A_k + A_k^T is the symmetric shift.
    # So grad_f = sum_k G_k * (shift_k(f) + shift_{-k}(f)).
    # This is a convolution of f with the kernel H where H[k] = G_k and H[-k] = G_k.
    # H is real and even.
    
    G = np.zeros(N)
    G[1:] = weights * np.sign(conv[1:])
    
    # Construct the kernel H of length 2N-1 or use FFT
    # H_k = G_k for k>0, H_{-k} = G_k.
    # This is equivalent to convolving f with [0, G_1, G_2, ..., G_{N-1}, 0, ..., 0, G_{N-1}, ..., G_1]?
    # Let's use FFT.
    # H_freq = FFT([0, G_1, ..., G_{N-1}, 0, ..., 0, G_{N-1}, ..., G_1, 0...])
    # Actually, the operator L_k(f) = shift_k(f) + shift_{-k}(f) has eigenvalues 2*cos(2*pi*k*n/N) in circular domain.
    # In linear domain, it's trickier.
    
    # Given N=400, direct loop is 399 * 400 = 160k ops. 1000 steps = 160M ops. 
    # NumPy vectorized loop over k is fast.
    
    grad = np.zeros(N)
    for k in range(1, N):
        wk = G[k]
        if wk == 0: continue
        # f_{j+k}
        # indices j such that j+k < N => j < N-k
        # f_{j-k}
        # indices j such that j-k >= 0 => j >= k
        
        # shift_left = f[k:] (length N-k)
        # shift_right = f[:N-k] (length N-k)
        
        # grad[0:N-k] += wk * f[k:]
        # grad[k:N] += wk * f[:N-k]
        
        grad[:N-k] += wk * f[k:]
        grad[k:] += wk * f[:N-k]
        
    return lse_val, grad

def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_f = np.ones(N)
    best_score = _compute_c3_upper_bound(best_f)
    
    # Continuous relaxation optimization
    # Start with random continuous values
    np.random.seed(42)
    f_cont = np.random.uniform(-1, 1, N)
    
    # Parameters
    beta = 10.0  # Sharpness of LSE approximation
    lr = 0.01    # Learning rate
    max_steps = 500
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
            
        # Compute loss and gradient
        loss, grad = _lse_max_autoconv(f_cont, beta)
        
        # Gradient step
        f_cont -= lr * grad
        
        # Project to discrete domain: sign()
        f_disc = np.sign(f_cont)
        # Handle zeros (rare)
        f_disc[f_disc == 0] = 1.0
        
        # Evaluate discrete score
        score_disc = _compute_c3_upper_bound(f_disc)
        
        if score_disc < best_score:
            best_score = score_disc
            best_f = f_disc.copy()
            
        # Occasionally, re-initialize continuous part to escape local minima?
        # Or just keep going.
        
        # Adaptive beta: increase beta to sharpen the max approximation
        if step % 50 == 0:
            beta *= 1.5
            lr *= 0.8

    # Final polish: Discrete local search on the best found
    if time.time() < deadline - 5:
        active = np.nonzero(best_f)[0]
        # Simple greedy flip
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in active:
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
