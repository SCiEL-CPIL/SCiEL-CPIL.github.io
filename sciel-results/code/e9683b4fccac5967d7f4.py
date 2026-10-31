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
    for _ in range(3000):
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
                if len(sidon_sets) > 40:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=50.0):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    def softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    
    def objective_and_grad_z(z):
        x = softmax(z)
        h = np.zeros(N)
        h[support] = x
        
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        # grad_h[j] = 2 * sum_k weights[k] * h[k-j]
        # = 2 * sum_{m in support} h[m] * weights[m+j]
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                start = m
                end = m + N
                if end <= len(weights):
                    grad_h += 2.0 * hm * weights[start:end]
        
        grad_x = grad_h[support]
        
        # Chain rule for softmax: grad_z_j = x_j * (grad_x_j - sum_i grad_x_i * x_i)
        sum_grad_x_x = np.dot(grad_x, x)
        grad_z = x * (grad_x - sum_grad_x_x)
        
        return obj, grad_z
    
    z0 = np.zeros(k)
    try:
        res = minimize(objective_and_grad_z, z0, jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        z_opt = res.x
    except Exception:
        z_opt = z0
    
    x_opt = softmax(z_opt)
    h_opt = np.zeros(N)
    h_opt[support] = x_opt
    sc = compute_ratio(h_opt)
    return h_opt, sc

def local_search_polish(h, support, max_iter=100, tol=1e-10):
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    
    best_score = compute_ratio(h)
    
    for it in range(max_iter):
        improved = False
        for i in range(k):
            for j in range(i + 1, k):
                hi = h[support[i]]
                hj = h[support[j]]
                if hi < 1e-12 or hj < 1e-12:
                    continue
                for frac in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    delta = frac * min(hi, hj)
                    new_hi = hi + delta
                    new_hj = hj - delta
                    if new_hi >= 0 and new_hj >= 0:
                        h_new = h.copy()
                        h_new[support[i]] = new_hi
                        h_new[support[j]] = new_hj
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending - larger sets tend to give better ratios
    sidon_sets.sort(key=len, reverse=True)
    
    for s in sidon_sets:
        if time.time() > deadline - 5.0:
            break
        
        # Try two beta values
        for beta in [30.0, 80.0]:
            if time.time() > deadline - 3.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        
        # Local search polish on the best candidate for this support
        if time.time() > deadline - 2.0:
            break
        h_polished = local_search_polish(best_h if np.allclose(best_h, h_opt) else h_opt, s, max_iter=50)
        sc_polished = compute_ratio(h_polished)
        if sc_polished > best_score:
            best_score = sc_polished
            best_h = h_polished.copy()
    
    return list(best_h)
