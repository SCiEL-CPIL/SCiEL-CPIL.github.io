import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

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

def _soft_max_conv(h, beta=50.0):
    """Smooth approximation of max(conv(h,h)) using log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 0:
        return 0.0, np.zeros_like(h)
    # log-sum-exp for stability
    shifted = beta * (conv - m)
    exp_shifted = np.exp(shifted)
    log_sum = m + np.log(np.sum(exp_shifted)) / beta
    # Gradient of soft-max w.r.t. h
    # d(softmax)/d(conv_k) = exp(beta*(conv_k - m)) / sum(exp(beta*(conv_j - m)))
    weights = exp_shifted / np.sum(exp_shifted)
    # conv_k = sum_i h_i * h_{k-i+N-1} (for appropriate indices)
    # d(conv_k)/d(h_j) = h_{k-j+N-1} + h_{k-j} (if in range)
    # Build gradient
    grad = np.zeros(N)
    for k in range(len(conv)):
        w = weights[k]
        # conv[k] = sum_{i+j=k} h[i]*h[j] where i,j in [0,N-1]
        # d(conv[k])/d(h[j]) = sum_i [delta(i+j=k) h[i]] = h[k-j] if k-j in [0,N-1]
        # Plus symmetric term from the other side
        # Actually conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i]
        # d/dh[j] = h[k-j] (if 0<=k-j<N) + (if j appears in the other factor, but it's symmetric)
        # For self-convolution: d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, else 0
        # But wait, conv[k] = sum_i h[i]*h[k-i], so d/dh[j] = h[k-j] (from first factor) 
        # but also when i=j, there's a factor of 2? No, because it's sum of products.
        # Actually: conv[k] = sum_{i} h[i]*h[k-i]. 
        # d/dh[j]: term where i=j gives h[k-j], term where k-i=j gives i=k-j, gives h[k-j].
        # So if j != k-j (i.e., 2j != k), we get 2*h[k-j]. If 2j==k, we get h[j] (one term, since h[j]*h[j]).
        # Wait no. Let me be more careful.
        # conv[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where h[k-i] is 0 if k-i not in [0,N-1]
        # d(conv[k])/d(h[j]) = sum_i [delta(i,j)*h[k-i] + h[i]*delta(k-i,j)]
        # = h[k-j] (from first term, i=j) + h[k-j] (from second term, k-i=j => i=k-j)
        # = 2*h[k-j] if 0<=k-j<N and k-j != j
        # = h[j] if k-j == j (i.e., k=2j), because then it's h[j]*h[j] and derivative is 2*h[j]... 
        # Wait no. If k=2j, then i=j gives h[j]*h[j], and the derivative of h[j]^2 w.r.t h[j] is 2*h[j].
        # But also from the second sum, k-i=j means i=k-j=j, so same term.
        # So actually: d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] when k-j is valid and k-j != j
        # When k-j == j (k=2j): the term h[j]*h[j] appears once in the sum (when i=j), 
        # and its derivative is 2*h[j]. But in the formula "h[k-j] + h[k-j]", we'd get 2*h[j] too.
        # So it's always 2*h[k-j] when 0<=k-j<N? Let me verify with a simple case.
        # h = [a, b], conv = [a^2, 2ab, b^2]
        # d(conv[0])/da = 2a. Formula: k=0, j=0: 2*h[0-0]=2*h[0]=2a. Correct.
        # d(conv[1])/da = 2b. Formula: k=1, j=0: 2*h[1-0]=2*h[1]=2b. Correct.
        # d(conv[1])/db = 2a. Formula: k=1, j=1: 2*h[1-1]=2*h[0]=2a. Correct.
        # So yes, d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N, else 0.
        
        idx = k - j
        if 0 <= idx < N:
            grad[j] += 2.0 * w * h[idx]
    
    grad *= beta  # Wait, no. Let me re-derive.
    # soft_max = m + (1/beta)*log(sum(exp(beta*(conv_k - m))))
    # d(soft_max)/d(conv_k) = exp(beta*(conv_k - m)) / sum(exp(beta*(conv_j - m))) = weights[k]
    # d(soft_max)/d(h_j) = sum_k weights[k] * d(conv_k)/d(h_j) = sum_k weights[k] * 2*h[k-j] (if valid)
    # So the gradient I computed above (without the beta factor) is correct.
    # Let me recompute without the extra beta.
    
    grad = np.zeros(N)
    for k in range(len(conv)):
        w = weights[k]
        for j in range(N):
            idx = k - j
            if 0 <= idx < N:
                grad[j] += 2.0 * w * h[idx]
    
    return log_sum, grad

def _objective_and_grad(h, beta=50.0):
    """Objective: -L2^2 / (L1 * soft_max_conv). Returns (value, gradient)."""
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0, np.zeros_like(h)
    l2_sq = np.sum(h**2)
    smax, grad_smax = _soft_max_conv(h, beta)
    if smax <= 0:
        return 0.0, np.zeros_like(h)
    
    # f = l2_sq / (l1 * smax)
    # We want to minimize -f
    # df/dh_j = (2*h_j * l1 * smax - l2_sq * (d(l1*smax)/dh_j)) / (l1*smax)^2
    # d(l1)/dh_j = 1
    # d(smax)/dh_j = grad_smax[j]
    # d(l1*smax)/dh_j = smax + l1 * grad_smax[j]
    
    denom = (l1 * smax)**2
    num_grad = 2.0 * h * l1 * smax - l2_sq * (smax + l1 * grad_smax)
    grad_f = num_grad / denom
    
    return -l2_sq / (l1 * smax), -grad_f

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric initializations
    # Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussian
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(1, N/2, 8):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(1, N, 8):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # Sparse peaks
    for _ in range(50):
        if time.time() > deadline - 60: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization with analytic gradients using SLSQP
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)},
        {'type': 'ineq', 'fun': lambda h: h, 'jac': lambda h: np.eye(N)}
    ]
    bounds = [(0.0, 1.0)] * N
    
    h_start = best_h / np.sum(best_h)
    
    # Try different beta values for smooth max
    for beta in [30.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 30: break
        
        def neg_obj_beta(h):
            val, _ = _objective_and_grad(h, beta)
            return val
        
        def neg_obj_grad_beta(h):
            _, grad = _objective_and_grad(h, beta)
            return grad
        
        try:
            res = minimize(neg_obj_beta, h_start, jac=neg_obj_grad_beta, method='SLSQP', 
                          bounds=bounds, constraints=constraints, 
                          options={'maxiter': 300, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(0, h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    h_start = h_candidate / np.sum(h_candidate)
        except Exception:
            pass
    
    # Generate diverse initializations for multi-start
    inits = [best_h / np.sum(best_h), np.ones(N) / N]
    
    for c in np.linspace(5, N-5, 4):
        for w in np.linspace(1, N/2, 4):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    for c in np.linspace(5, N-5, 4):
        for w in np.linspace(1, N, 4):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            h_r /= np.sum(h_r)
            inits.append(h_r)

    # Optimize from each init with smooth objective and analytic gradient
    for h_init in inits:
        if time.time() > deadline - 15: break
        if np.sum(h_init) == 0: continue
        h_init = np.maximum(0, h_init) / np.sum(h_init)
        
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 10: break
            def neg_obj_b(h):
                val, _ = _objective_and_grad(h, beta)
                return val
            def neg_obj_grad_b(h):
                _, grad = _objective_and_grad(h, beta)
                return grad
            try:
                res = minimize(neg_obj_b, h_init, jac=neg_obj_grad_b, method='SLSQP', 
                              bounds=bounds, constraints=constraints, 
                              options={'maxiter': 150, 'ftol': 1e-12})
                if res.success:
                    h_candidate = np.maximum(0, res.x)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Final polish with exact (non-smooth) objective
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_exact(h):
            l1 = np.sum(h)
            if l1 == 0: return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0
            return -(l2_sq / max_conv)
        
        try:
            res = minimize(neg_obj_exact, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-15})
            if res.success:
                h_candidate = np.maximum(0, res.x)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
        
        # Perturbation-based polish
        if time.time() < deadline - 2:
            for scale in [0.005, 0.01, 0.02]:
                if time.time() > deadline - 1: break
                h_pert = best_h * (1 + scale * np.random.randn(N))
                h_pert = np.maximum(0, h_pert)
                if np.sum(h_pert) > 0:
                    h_pert /= np.sum(h_pert)
                    try:
                        res = minimize(neg_obj_exact, h_pert, method='SLSQP', bounds=bounds,
                                      constraints=constraints, options={'maxiter': 300, 'ftol': 1e-15})
                        if res.success:
                            h_candidate = np.maximum(0, res.x)
                            s_candidate = _score_raw(h_candidate)
                            if s_candidate > best_score:
                                best_score = s_candidate
                                best_h = h_candidate.copy()
                    except Exception:
                        pass

    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    return list(best_h)
