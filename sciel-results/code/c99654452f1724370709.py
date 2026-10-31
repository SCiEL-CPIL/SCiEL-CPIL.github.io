import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, max_iter=1000, ftol=1e-10):
    """
    Optimize weights h_i for i in support to maximize the ratio.
    Uses a smooth surrogate for the max convolution coefficient.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    x0 = np.ones(k) / k
    
    def objective(x):
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0
        shifted = conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted)))
        return log_sum_exp / beta

    def jac(x):
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(k)
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Gradient computation for smooth max of convolution
        # d/dh_j (log_sum_exp / beta) = sum_k weights_k * d conv_k / d h_j
        # d conv_k / d h_j = 2 * h_{k-j} (with appropriate boundary handling)
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                # grad_h[j] = 2 * sum_m h[m] * weights[m+j]
                # For each m in support, add 2*h[m]*weights[m:m+N] to grad_h
                grad_h += 2.0 * hm * weights[m : m+N]
        
        return grad_h[support]

    # Constraints: sum(x) = 1, x >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(k)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       jac=jac, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': ftol})
        x_opt = res.x
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception as e:
        # Fallback to uniform
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        sc = compute_ratio(h_opt)
        return h_opt, sc

def local_search_polish(h, max_iter=50, tol=1e-8):
    """
    Local search polish: try small perturbations to improve the ratio.
    """
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h
    h = h / l1
    
    best_score = compute_ratio(h)
    support = np.where(h > 1e-10)[0]
    
    for _ in range(max_iter):
        improved = False
        for i in range(len(support)):
            for j in range(i+1, len(support)):
                # Try to swap some mass between support[i] and support[j]
                for eps in [0.01, 0.005, 0.001]:
                    h_try = h.copy()
                    h_try[support[i]] += eps
                    h_try[support[j]] -= eps
                    h_try = np.maximum(h_try, 0.0)
                    l1_try = np.sum(h_try)
                    if l1_try < 1e-15:
                        continue
                    h_try = h_try / l1_try
                    sc_try = compute_ratio(h_try)
                    if sc_try > best_score + tol:
                        h = h_try
                        best_score = sc_try
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return h

def spectral_design_optimize(initial_h, max_iter=200, lr=0.01):
    """
    Spectral design approach: optimize in the frequency domain.
    The autoconvolution peak is related to the L1 norm of the Fourier transform.
    We want to minimize the peak, which is equivalent to making the spectrum flat.
    """
    h = np.maximum(initial_h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h
    h = h / l1
    
    # Work in the frequency domain
    # The convolution in time domain is multiplication in frequency domain
    # conv = ifft(fft(h)^2)
    # We want to minimize max(conv)
    
    best_score = compute_ratio(h)
    support = np.where(h > 1e-10)[0]
    
    for _ in range(max_iter):
        # Compute gradient of max(conv) w.r.t. h
        conv = fftconvolve(h, h, mode='full')
        max_idx = np.argmax(conv)
        
        # Gradient of conv[max_idx] w.r.t. h
        # conv[max_idx] = sum_i h[i] h[max_idx - i]
        # d/dh[j] conv[max_idx] = h[max_idx - j] + h[j - max_idx] (if in bounds)
        
        grad = np.zeros(N)
        for j in range(N):
            idx1 = max_idx - j
            idx2 = j - max_idx
            if 0 <= idx1 < N:
                grad[j] += h[idx1]
            if 0 <= idx2 < N:
                grad[j] += h[idx2]
        
        # We want to minimize max(conv), so move in negative gradient direction
        # But we also want to maintain sum(h) = 1 and h >= 0
        
        # Projected gradient descent
        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0.0)
        h_new_sum = np.sum(h_new)
        if h_new_sum > 1e-15:
            h_new = h_new / h_new_sum
        
        sc_new = compute_ratio(h_new)
        if sc_new > best_score:
            h = h_new
            best_score = sc_new
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending
    sidon_sets.sort(key=lambda x: len(x), reverse=True)
    
    # Optimize weights for each Sidon set
    for s in sidon_sets:
        if time.time() > deadline - 15.0:
            break
        
        # Try multiple betas for the smooth surrogate
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 15.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta, max_iter=500, ftol=1e-10)
            
            # Polish with local search
            h_polished = local_search_polish(h_opt, max_iter=30, tol=1e-9)
            sc_polished = compute_ratio(h_polished)
            
            if sc_polished > best_score:
                best_score = sc_polished
                best_h = h_polished.copy()
            
            # Try spectral design polish
            if time.time() > deadline - 15.0:
                break
            
            h_spectral = spectral_design_optimize(h_polished, max_iter=100, lr=0.005)
            sc_spectral = compute_ratio(h_spectral)
            
            if sc_spectral > best_score:
                best_score = sc_spectral
                best_h = h_spectral.copy()
    
    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        best_h = local_search_polish(best_h, max_iter=100, tol=1e-10)
        final_score = compute_ratio(best_h)
        if final_score > best_score:
            best_score = final_score
    
    return list(best_h)
