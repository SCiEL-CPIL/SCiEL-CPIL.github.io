import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _soft_max(v, tau=0.1):
    """Soft maximum of vector v."""
    v_max = np.max(v)
    # Prevent overflow
    if v_max > 700:
        return v_max
    return v_max + tau * np.log(np.sum(np.exp((v - v_max) / tau)))

def _autoconv_component(h, k_idx):
    """Compute (h*h)_k for a specific index k (0 to 2N-2)."""
    lo = max(0, k_idx - N + 1)
    hi = min(N - 1, k_idx)
    if lo > hi:
        return 0.0
    i_vals = np.arange(lo, hi + 1)
    j_vals = k_idx - i_vals
    return np.sum(h[i_vals] * h[j_vals])

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using L-BFGS-B on log-scale variables with Soft-Max surrogate
    # Adaptive temperature: start loose for exploration, tighten for exploitation
    tau_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    h_start = best_h.copy()
    h_start[h_start < 1e-8] = 1e-8
    x0 = np.log(h_start)
    
    for tau in tau_schedule:
        if time.time() > deadline - 30:
            break
        def neg_obj_soft(x, tau_val=tau):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            sm = _soft_max(h_conv, tau=tau_val)
            if sm == 0:
                return 0.0
            return -(l2_sq / sm)
        
        try:
            res = minimize(neg_obj_soft, x0, method='L-BFGS-B', 
                          options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                x0 = res.x.copy()
        except Exception:
            break

    # Try multiple restarts with adaptive soft-max
    inits = [best_h.copy()]
    for _ in range(15):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for _ in range(15):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    for h_init in inits:
        if time.time() > deadline - 10: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        h_init[h_init < 1e-8] = 1e-8
        x_init = np.log(h_init)
        
        # Use a few temperature levels for each restart
        for tau in [0.05, 0.02, 0.005]:
            if time.time() > deadline - 5:
                break
            def neg_obj_soft_restart(x, tau_val=tau):
                h = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h**2)
                h_conv = np.convolve(h, h, mode='full')
                sm = _soft_max(h_conv, tau=tau_val)
                if sm == 0:
                    return 0.0
                return -(l2_sq / sm)
            
            try:
                res = minimize(neg_obj_soft_restart, x_init, method='L-BFGS-B', 
                              options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
                    x_init = res.x.copy()
            except Exception:
                break

    # NEW: Iterative peak-index refinement with gradient-boosted convergence
    if time.time() < deadline - 20:
        h_ref = best_h / np.sum(best_h)
        
        for iteration in range(20):
            if time.time() > deadline - 5:
                break
            
            h_conv_ref = np.convolve(h_ref, h_ref, mode='full')
            peak_idx = np.argmax(h_conv_ref)
            
            def neg_obj_fixed_peak(x, k_idx):
                h = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h**2)
                val = _autoconv_component(h, k_idx)
                if val <= 0:
                    return 0.0
                return -(l2_sq / val)
            
            def grad_obj_fixed_peak(x, k_idx):
                h = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h**2)
                val = _autoconv_component(h, k_idx)
                if val <= 0:
                    return np.zeros_like(x)
                
                # Gradient of -(l2_sq / val) w.r.t. x
                # h = exp(x) / sum(exp(x)), so dh/dx_i = h_i * (delta_ij - h_j)
                # l2_sq = sum(h^2), d(l2_sq)/dx_i = 2 * sum(h_j * dh_j/dx_i) = 2 * h_i * (2*h_i - 1) * ... 
                # More carefully: d(sum h_j^2)/dx_i = sum_j 2*h_j * dh_j/dx_i
                # dh_j/dx_i = h_j*(delta_ij - h_j)
                # So d(l2_sq)/dx_i = sum_j 2*h_j * h_j*(delta_ij - h_j) = 2*h_i^2*(1-h_i) - 2*h_i*sum(h_j^2)
                # = 2*h_i^2 - 2*h_i^3 - 2*h_i*l2_sq
                
                # val = sum_{i,j: i+j=k} h_i*h_j
                # d(val)/dx_i = sum_j (delta_{i,j'} * h_{j'} + h_i * delta_{j,i'}) where i'+j'=k
                # = h_{k-i} (if 0<=k-i<N) + h_i * (number of j such that j=i and i+j=k, i.e., 2i=k)
                # More precisely: d(val)/dx_i = sum_{j: i+j=k} (delta_{i,j'} * h_{j'} + h_i * delta_{j,i'})
                # For each pair (i,j) with i+j=k: contributes h_j to dh_i/dx_i and h_i to dh_j/dx_i
                # So d(val)/dx_i = sum_{j: i+j=k} h_j + sum_{j: i+j=k} h_i * [j==i]
                # = h_{k-i} (if valid) + h_i * [2i==k]
                
                d_l2 = 2*h**2 - 2*h**3 - 2*h*l2_sq
                d_val = np.zeros(N)
                for i in range(N):
                    j = k_idx - i
                    if 0 <= j < N:
                        d_val[i] += h[j]
                        if 2*i == k_idx:
                            d_val[i] += h[i]
                
                # d/dx_i [-(l2_sq/val)] = -(d_l2_i * val - l2_sq * d_val_i) / val^2
                grad = -(d_l2 * val - l2_sq * d_val) / (val**2)
                return grad
            
            h_ref[h_ref < 1e-8] = 1e-8
            x_ref = np.log(h_ref)
            
            try:
                res = minimize(neg_obj_fixed_peak, x_ref, args=(peak_idx,),
                              method='L-BFGS-B', jac=lambda x, k_idx: grad_obj_fixed_peak(x, k_idx),
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
                    h_ref = h_candidate
            except Exception:
                # Fallback without gradient
                try:
                    res = minimize(neg_obj_fixed_peak, x_ref, args=(peak_idx,),
                                  method='L-BFGS-B', options={'maxiter': 600, 'ftol': 1e-14})
                    if res.success:
                        h_candidate = np.exp(res.x)
                        h_candidate /= np.sum(h_candidate)
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate
                        h_ref = h_candidate
                except Exception:
                    break

    # Final polish with tighter tolerance on the exact (non-soft) objective
    if time.time() < deadline - 8:
        h_start = best_h / np.sum(best_h)
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_exact(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / max_conv)

        # First pass: moderate tolerance
        try:
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', 
                          options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-10})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                x0 = res.x.copy()
        except Exception:
            pass

        # Second pass: very tight tolerance for final polish
        if time.time() < deadline - 3:
            try:
                res = minimize(neg_obj_exact, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
            except Exception:
                pass

    return list(best_h)

# EVOLVE-BLOCK-END
