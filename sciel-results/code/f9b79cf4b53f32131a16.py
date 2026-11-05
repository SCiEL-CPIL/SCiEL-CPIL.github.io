"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _hard_value(h):
    """Compute the actual evaluation value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(x, beta=50.0):
    """
    Objective to minimize: 2*N*soft_max(conv) / (sum(h))^2
    Using log-sum-exp for numerical stability.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 0:
        return 2.0 * N * m / (s * s)
    
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    
    return 2.0 * N * log_sum_exp / (s * s)

def _soft_max_grad(x, beta=50.0):
    """
    Gradient of the soft-max objective computed analytically.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    
    if m <= 0:
        # Gradient of 2*N*max(conv)/s^2 when max is negative
        # Just use the argmax
        idx = np.argmax(conv)
        grad_conv = np.zeros_like(h)
        # conv[idx] = sum_k h[k]*h[idx-k]
        # d(conv[idx])/d(h[i]) = h[idx-i] + h[i-idx] if valid
        for i in range(N):
            j1 = idx - i
            j2 = i - idx
            val = 0.0
            if 0 <= j1 < N:
                val += h[j1]
            if 0 <= j2 < N:
                val += h[j2]
            grad_conv[i] = val
        return 2.0 * N * (grad_conv * s - 2.0 * m) / (s * s * s)
    
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    
    # Gradient of soft_max w.r.t. h_i
    # d(soft_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i)
    # d(conv_j)/d(h_i) = h[j-i] + h[i-j] (with zero padding)
    # This is equivalent to a weighted correlation
    
    # Efficient computation using convolution
    # For each i: sum_j weights_j * (h[j-i] if 0<=j-i<N else 0) + sum_j weights_j * (h[i-j] if 0<=i-j<N else 0)
    
    # The first term: sum_j weights_j * h[j-i] for valid j-i
    # This is like correlating weights with h
    # weights has length 2N-1, indexed 0 to 2N-2
    # h has length N, indexed 0 to N-1
    # For fixed i, j-i ranges over valid values when j is in [i, i+N-1]
    # So sum_{j=max(0,i)}^{min(2N-2, i+N-1)} weights[j] * h[j-i]
    
    # Let's compute this as: for each i, (weights * h)_i in a specific sense
    # Actually, let's think of it as:
    # term1[i] = sum_j weights[j] * h[j-i] where 0<=j-i<N
    # = sum_{k=0}^{N-1} weights[i+k] * h[k] where 0<=i+k<2N-1
    # = sum_{k=0}^{N-1} weights[i+k] * h[k] (with weights out of range = 0)
    
    # term2[i] = sum_j weights[j] * h[i-j] where 0<=i-j<N
    # = sum_{k=0}^{N-1} weights[i-k] * h[k] where 0<=i-k<2N-1
    # = sum_{k=0}^{N-1} weights[i-k] * h[k] (with weights out of range = 0)
    
    # We can compute these using convolution or direct summation
    # For N=400, direct summation is O(N^2) = 160000 which is fine
    
    term1 = np.zeros(N)
    term2 = np.zeros(N)
    for i in range(N):
        # term1[i] = sum_{k=0}^{N-1} weights[i+k] * h[k] if i+k < 2N-1
        for k in range(N):
            j = i + k
            if j < 2*N - 1:
                term1[i] += weights[j] * h[k]
        # term2[i] = sum_{k=0}^{N-1} weights[i-k] * h[k] if i-k >= 0
        for k in range(N):
            j = i - k
            if j >= 0:
                term2[i] += weights[j] * h[k]
    
    grad_soft_max = term1 + term2
    
    # Full gradient of 2*N*soft_max/s^2
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_soft_max_i * s^2 - soft_max * 2*s) / s^4
    #          = 2*N * (grad_soft_max_i * s - 2*soft_max) / s^3
    
    return 2.0 * N * (grad_soft_max * s - 2.0 * soft_max) / (s * s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_h = np.ones(N)
    best_val = _hard_value(best_h)
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(8):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [1, 2, 5, 10, 20, 50]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t) + 0.3 * np.sin(3*t)
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Asymmetric patterns - key for breaking symmetry
    for _ in range(5):
        h0 = rng.standard_normal(N)
        # Make it asymmetric by applying a ramp
        ramp = np.linspace(1.0, 0.0, N)
        h0 = h0 * ramp + 0.5 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Block patterns with random signs
    for block_size in [5, 10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            sign = rng.choice([-1.0, 1.0])
            h0[b*block_size:(b+1)*block_size] = sign * abs(rng.standard_normal())
        initializations.append(h0)
    
    # 6. Sparse patterns
    for sparsity in [0.1, 0.2, 0.5]:
        h0 = np.zeros(N)
        mask = rng.random(N) < sparsity
        h0[mask] = rng.standard_normal(np.sum(mask))
        initializations.append(h0)
    
    # Optimize each initialization
    betas = [30.0, 50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            
            h_curr = h0.copy()
            
            # Run L-BFGS-B with analytical gradient
            try:
                res = minimize(_soft_max_obj, h_curr,
                              jac=_soft_max_grad,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _hard_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            
            # If we found something good, do a second pass with higher beta
            if val_new < best_val + 0.5 and time.time() < deadline - 10.0:
                for beta2 in [200.0, 500.0]:
                    if time.time() > deadline - 8.0:
                        break
                    try:
                        res2 = minimize(_soft_max_obj, h_new,
                                       jac=_soft_max_grad,
                                       method='L-BFGS-B',
                                       args=(beta2,),
                                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                        val2 = _hard_value(res2.x)
                        if val2 < best_val:
                            best_val = val2
                            best_h = res2.x.copy()
                    except Exception:
                        pass
    
    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 3.0:
                break
            try:
                res = minimize(_soft_max_obj, best_h,
                              jac=_soft_max_grad,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                val = _hard_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard_value(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
