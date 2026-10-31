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
    conv_offpeak = np.delete(conv, N-1)
    conv_inf = np.max(conv_offpeak)
    if conv_inf < 1e-15:
        return h2_sq / 1e-15
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

def optimize_weights_for_support(support, beta=100.0):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    x0 = np.zeros(k)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x, beta_val):
        h = softmax(x)
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        conv_offpeak = np.delete(conv, N-1)
        
        max_c = np.max(conv_offpeak)
        if max_c < 1e-15:
            max_c = 1e-15
        
        shifted = conv_offpeak - max_c
        exp_shifted = np.exp(beta_val * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        h2_sq = np.sum(h_full**2)
        if h2_sq < 1e-15:
            h2_sq = 1e-15
            
        obj_val = (1.0 / beta_val) * np.log(sum_exp) + max_c - np.log(h2_sq)
        
        w_full = np.zeros(len(conv))
        w_full[:N-1] = weights[:N-1]
        w_full[N:] = weights[N-1:]
        
        grad_h = np.zeros(N)
        
        m_indices = np.nonzero(w_full)[0]
        lags = m_indices - (N - 1)
        w_vals = w_full[m_indices]
        
        for j in range(N):
            idx_plus = j + lags
            valid_plus = (idx_plus >= 0) & (idx_plus < N)
            term_plus = np.sum(w_vals[valid_plus] * h_full[idx_plus[valid_plus]])
            
            idx_minus = j - lags
            valid_minus = (idx_minus >= 0) & (idx_minus < N)
            term_minus = np.sum(w_vals[valid_minus] * h_full[idx_minus[valid_minus]])
            
            grad_h[j] = term_plus + term_minus - 2.0 * h_full[j] / h2_sq
        
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    def admm_optimize(support, max_iter=200, rho=1.0):
        """ADMM optimization using complex parameterization h_i = |z_i|^2"""
        k = len(support)
        # Initialize z
        np.random.seed(42)
        z = np.ones(k, dtype=np.complex128) * (1.0/np.sqrt(k)) + 0.1j * np.random.randn(k)
        lam = np.zeros(k, dtype=np.complex128)
        
        # Precompute convolution structure
        # For each lag l, the convolution at that lag is sum_i h_i * h_{i+l}
        # In terms of z: sum_i |z_i|^2 * |z_{i+l}|^2
        # This can be written as sum_i (z_i * conj(z_{i+l})) * (conj(z_i) * z_{i+l})
        # = sum_i |z_i * conj(z_{i+l})|^2
        
        # Actually, let's think more carefully.
        # conv[l] = sum_i h_i * h_{i+l} where h_i = |z_i|^2
        # = sum_i |z_i|^2 * |z_{i+l}|^2
        
        # The gradient with respect to z involves these terms.
        # Let's use a simpler approach: treat h directly but use ADMM for the constraint.
        
        # Actually, let's use a different approach: direct optimization with 
        # alternating minimization and projection.
        
        h = np.zeros(N)
        h[support] = 1.0/k
        
        for it in range(max_iter):
            # Update: minimize the soft-max surrogate + penalty
            # For simplicity, use gradient descent on h with projection
            
            conv = fftconvolve(h, h, mode='full')
            conv_offpeak = np.delete(conv, N-1)
            max_c = np.max(conv_offpeak)
            if max_c < 1e-15:
                max_c = 1e-15
            
            h2_sq = np.sum(h**2)
            if h2_sq < 1e-15:
                h2_sq = 1e-15
            
            # Gradient of log-sum-exp surrogate
            shifted = conv_offpeak - max_c
            beta_admm = 50.0
            exp_shifted = np.exp(beta_admm * shifted)
            sum_exp = np.sum(exp_shifted)
            weights = exp_shifted / sum_exp
            
            w_full = np.zeros(len(conv))
            w_full[:N-1] = weights[:N-1]
            w_full[N:] = weights[N-1:]
            
            grad_h = np.zeros(N)
            m_indices = np.nonzero(w_full)[0]
            lags = m_indices - (N - 1)
            w_vals = w_full[m_indices]
            
            for j in range(N):
                idx_plus = j + lags
                valid_plus = (idx_plus >= 0) & (idx_plus < N)
                term_plus = np.sum(w_vals[valid_plus] * h[idx_plus[valid_plus]])
                
                idx_minus = j - lags
                valid_minus = (idx_minus >= 0) & (idx_minus < N)
                term_minus = np.sum(w_vals[valid_minus] * h[idx_minus[valid_minus]])
                
                grad_h[j] = term_plus + term_minus - 2.0 * h[j] / h2_sq
            
            # Gradient step
            lr = 0.01
            h_new = h - lr * grad_h
            
            # Project onto simplex: h >= 0, sum(h[support]) = 1
            h_new = np.maximum(h_new, 0.0)
            h_new_support = h_new[support]
            s = np.sum(h_new_support)
            if s > 1e-15:
                h_new[support] = h_new_support / s
            else:
                h_new[support] = 1.0/k
            
            h = h_new
        
        h_full = h
        sc = compute_ratio(h_full)
        return h_full, sc
    
    try:
        best_local_h = np.zeros(N)
        best_local_score = -np.inf
        best_local_x = x0
        
        np.random.seed(42)
        n_restarts = min(5, max(1, 60 - 5 * len(support)))
        
        for restart in range(n_restarts):
            if restart == 0:
                x_curr = x0.copy()
            else:
                x_curr = np.random.randn(k) * 2.0
            
            try:
                res = minimize(lambda x: objective_and_grad(x, beta), x_curr, method='L-BFGS-B', 
                               jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                x_opt = res.x
                h_opt_local = softmax(x_opt)
                
                h_full = np.zeros(N)
                h_full[support] = h_opt_local
                sc = compute_ratio(h_full)
                
                if sc > best_local_score:
                    best_local_score = sc
                    best_local_h = h_full.copy()
                    best_local_x = x_opt.copy()
            except Exception:
                continue
        
        # Try ADMM as an alternative
        try:
            h_admm, sc_admm = admm_optimize(support, max_iter=100)
            if sc_admm > best_local_score:
                best_local_score = sc_admm
                best_local_h = h_admm.copy()
        except Exception:
            pass
        
        # Final polish with higher beta
        if best_local_score > -np.inf:
            try:
                for beta_polish in [200.0, 400.0]:
                    def obj_polish(x):
                        return objective_and_grad(x, beta_polish)
                    
                    res = minimize(obj_polish, best_local_x, method='L-BFGS-B', 
                                   jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                    x_opt = res.x
                    h_opt_local = softmax(x_opt)
                    h_full = np.zeros(N)
                    h_full[support] = h_opt_local
                    sc = compute_ratio(h_full)
                    
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_h = h_full.copy()
            except Exception:
                pass
        
        return best_local_h, best_local_score
    except Exception as e:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        sc = compute_ratio(h_full)
        return h_full, sc

def local_search_swap(h, support, max_iters=100, deadline=None):
    k = len(support)
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = list(support)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for new_idx in range(N):
            if deadline is not None and time.time() > deadline:
                break
            if new_idx in best_support:
                continue
            new_support = best_support + [new_idx]
            new_support.sort()
            if len(new_support) > 15:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        for remove_idx in list(best_support):
            if deadline is not None and time.time() > deadline:
                break
            new_support = [x for x in best_support if x != remove_idx]
            if len(new_support) < 3:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        for old_idx in best_support:
            if deadline is not None and time.time() > deadline:
                break
            for new_idx in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                if new_idx in best_support:
                    continue
                new_support = [x for x in best_support if x != old_idx]
                new_support.append(new_idx)
                new_support.sort()
                
                h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0)
                if sc_new > best_score + 1e-6:
                    best_h = h_new
                    best_score = sc_new
                    best_support = new_support
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_h, best_score, best_support

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    
    sidon_sets.sort(key=lambda x: -len(x))
    
    for s in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        for beta in [30.0, 50.0, 80.0, 120.0]:
            if time.time() > deadline - 30.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    if time.time() < deadline - 25.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=80, deadline=deadline - 15.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    if time.time() < deadline - 5.0:
        for beta in [100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 5.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    return list(best_h)
