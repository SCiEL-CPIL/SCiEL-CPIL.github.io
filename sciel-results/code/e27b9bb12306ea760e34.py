"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _objective_hard(h):
    """Compute the hard objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max approximation of the objective.
    Minimizes 2*N*soft_max(conv) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _soft_max_grad(h, beta):
    """
    Analytical gradient of the soft-max objective with respect to h.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # soft_max = m + log(sum_exp)/beta
    # d(soft_max)/d(conv[j]) = exp_shifted[j] / sum_exp
    w = exp_shifted / sum_exp
    
    # conv[j] = sum_{i} h[i] * h[i + j - N + 1]
    # Let k = j - N + 1, so k ranges from -(N-1) to N-1
    # conv[j] = sum_i h[i] * h[i+k]
    # d(conv[j])/dh[i] = h[i+k] (if 0 <= i+k < N) + h[i-k] (if 0 <= i-k < N)
    #                   - but we need to be careful about double counting when k=0
    
    # Actually, let's think more carefully.
    # conv[j] = sum_{i=0}^{N-1} h[i] * h[i + (j - N + 1)]
    # where h[m] = 0 for m outside [0, N-1]
    
    # d(conv[j])/dh[i] = h[i + (j-N+1)] if 0 <= i+(j-N+1) < N
    #                   + h[i - (j-N+1)] if 0 <= i-(j-N+1) < N
    #                   - h[i] * h[i] * (1 if j==N-1 and i==i) ... no
    #                   Actually, when we differentiate sum_k h[k]*h[k+l] w.r.t h[i],
    #                   we get h[i+l] (from the first factor when k=i) + h[i-l] (from the second factor when k+l=i, i.e., k=i-l)
    #                   But if l=0, then both terms are h[i], so we get 2*h[i]
    
    # Let's compute the gradient more directly.
    # grad_soft_max[i] = sum_j w_j * d(conv[j])/dh[i]
    # d(conv[j])/dh[i] = h[i + j - N + 1] (if valid) + h[i - j + N - 1] (if valid)
    # But note: when j = N-1, j-N+1 = 0, so d(conv[N-1])/dh[i] = h[i] + h[i] = 2*h[i]
    # For j != N-1, the two terms are distinct (one is h[i+k], other is h[i-k] with k != 0)
    
    # Let's use a convolution-based approach.
    # Define w as a vector of length 2N-1.
    # The gradient of soft_max(conv) w.r.t. h is:
    # grad[i] = sum_j w_j * (h[i + j - N + 1] + h[i - j + N - 1]) with boundary handling
    
    # Let's define k = j - N + 1, so k ranges from -(N-1) to N-1
    # Then w_j corresponds to w_k where k = j - N + 1
    # grad[i] = sum_k w_{k+N-1} * (h[i+k] if 0<=i+k<N else 0 + h[i-k] if 0<=i-k<N else 0)
    
    # This is essentially a convolution-like operation.
    # Let's create a vector w_k of length 2N-1, indexed by k from -(N-1) to N-1
    # w_k = w_{k+N-1} where w_j is the weight for conv[j]
    
    # Actually, let's just compute it directly using the fact that:
    # grad_soft_max = (w * h) + (flip(w) * h) in some sense
    
    # Let's use a simpler approach: compute the gradient by constructing the Jacobian
    # of conv w.r.t. h, which is a (2N-1) x N matrix, but that's too large.
    
    # Instead, let's use the following observation:
    # d(conv[j])/dh[i] = h[i + j - N + 1] (if 0 <= i+j-N+1 < N) + h[i - j + N + 1 - 2*N + 2] ... 
    # Let me re-derive this.
    
    # conv[j] = sum_{k=0}^{N-1} h[k] * h[k + j - N + 1]
    # Let l = j - N + 1
    # conv[j] = sum_k h[k] * h[k+l]
    # d(conv[j])/dh[i] = h[i+l] (if 0 <= i+l < N) + h[i-l] (if 0 <= i-l < N)
    #                   (the first term comes from d/dh[i] of h[i]*h[i+l], the second from d/dh[i] of h[i-l]*h[i])
    #                   But wait, when l=0, h[i]*h[i] differentiated w.r.t h[i] gives 2*h[i]
    #                   So actually: d(conv[j])/dh[i] = h[i+l] (if valid) + h[i-l] (if valid)
    #                   And when l=0, this gives h[i] + h[i] = 2*h[i], which is correct.
    
    # So: grad_soft_max[i] = sum_j w_j * (h[i + j - N + 1] (if 0<=i+j-N+1<N) + h[i - j + N - 1] (if 0<=i-j+N-1<N))
    
    # Let's compute this efficiently.
    # Define w_shifted such that w_shifted[k] = w_{k+N-1} for k in [-(N-1), N-1]
    # Then grad_soft_max[i] = sum_k w_shifted[k] * (h[i+k] if 0<=i+k<N else 0) + sum_k w_shifted[k] * (h[i-k] if 0<=i-k<N else 0)
    
    # The first sum is a cross-correlation of w_shifted and h.
    # The second sum is also a cross-correlation.
    
    # Let's just compute it using loops for clarity, since N=400 and 2N-1=799, this should be fast enough.
    
    grad_soft = np.zeros(N)
    for j in range(2 * N - 1):
        if w[j] < 1e-15:
            continue
        k = j - N + 1
        # d(conv[j])/dh[i] = h[i+k] (if valid) + h[i-k] (if valid)
        # For the first term: i+k in [0, N-1] => i in [-k, N-1-k]
        # For the second term: i-k in [0, N-1] => i in [k, N-1+k]
        
        # First term contribution
        if k >= 0:
            # i + k in [0, N-1] => i in [0, N-1-k]
            if N - 1 - k >= 0:
                grad_soft[0:N-k] += w[j] * h[k:N]
        else:
            # i + k in [0, N-1] => i in [-k, N-1]
            if -k <= N-1:
                grad_soft[-k:N] += w[j] * h[0:N+k]
        
        # Second term contribution
        if k <= 0:
            # i - k in [0, N-1] => i in [0, N-1-k]
            if N - 1 - k >= 0:
                grad_soft[0:N-k] += w[j] * h[k:N]  # h[i-k] where i-k in [0, N-1], so i in [k, N-1+k]
                # Wait, I need to be more careful.
                # h[i-k] where i-k in [0, N-1] means i in [k, N-1+k]
                # But i must also be in [0, N-1]
                # So i in [max(0,k), min(N-1, N-1+k)]
        else:
            # i - k in [0, N-1] => i in [k, N-1+k]
            # i must be in [0, N-1], so i in [k, N-1]
            if k < N:
                grad_soft[k:N] += w[j] * h[0:N-k]
    
    # Actually, let me redo this more carefully.
    # d(conv[j])/dh[i] = h[i+k] (if 0 <= i+k < N) + h[i-k] (if 0 <= i-k < N)
    # where k = j - N + 1
    
    # For fixed j (and thus fixed k), the contribution to grad_soft[i] is:
    # w[j] * (h[i+k] if 0<=i+k<N else 0) + w[j] * (h[i-k] if 0<=i-k<N else 0)
    
    # Let's compute this by iterating over i.
    grad_soft = np.zeros(N)
    for j in range(2 * N - 1):
        if w[j] < 1e-15:
            continue
        k = j - N + 1
        for i in range(N):
            val = 0.0
            if 0 <= i + k < N:
                val += h[i + k]
            if 0 <= i - k < N:
                val += h[i - k]
            grad_soft[i] += w[j] * val
    
    # Now compute the full gradient of the objective:
    # V = 2*N * soft_max / s^2
    # dV/dh[i] = 2*N * (1/s^2) * grad_soft[i] - 2*N * soft_max * 2/s^3 * 1
    #          = 2*N * grad_soft[i] / s^2 - 4*N * soft_max / s^3
    
    soft_max_val = m + np.log(sum_exp) / beta
    grad_full = 2.0 * N * grad_soft / (s * s) - 4.0 * N * soft_max_val / (s * s * s)
    
    return grad_full

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal with positive mean
    for _ in range(15):
        h0 = rng.standard_normal(N)
        h0 += 0.5  # Ensure positive mean
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Sign sequences with DC offset
    for _ in range(15):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Sine waves with various frequencies and phases
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8  # Positive mean
        initializations.append(h0)
        
    # 4. Polynomial perturbations
    for _ in range(10):
        h0 = np.ones(N)
        t = np.linspace(0, 1, N)
        h0 += 0.2 * (rng.random() * t**2 - 0.1)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 5. Gaussian bumps
    for _ in range(10):
        h0 = np.zeros(N)
        center = rng.integers(0, N)
        width = rng.uniform(10, 50)
        t = np.arange(N)
        h0 += np.exp(-0.5 * ((t - center) / width)**2)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)

    # Optimization using scipy's minimize with L-BFGS-B and analytical gradient
    betas = [30.0, 50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 25.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, h_curr, jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Final polish with high beta and tighter tolerances
    if time.time() < deadline - 15.0:
        for beta in [200.0, 500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 10.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, best_h, jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
