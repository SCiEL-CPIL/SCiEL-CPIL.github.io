import numpy as np
import time
import sys
from scipy.optimize import minimize, minimize_scalar

N = 50
DELTA = 0.5 / N
NUM_CONV = 2 * N - 1

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Initial Heuristics
    # Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # Sparse peaks (Sidon-like intuition)
    for _ in range(200):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        # Add some noise
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # 2. Epigraph Optimization (SLSQP)
    # Maximize L2^2 / (L1 * AutoMax) with L1=1.
    # Minimize M subject to Conv(k) <= M for all k.
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0
        return -(l2_sq / max_conv) # Assuming L1=1

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    h_start = best_h / np.sum(best_h)
    
    res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
    
    if res.success:
        h_candidate = res.x
        s_candidate = _score_raw(h_candidate)
        if s_candidate > best_score:
            best_score = s_candidate
            best_h = h_candidate
    
    # Multi-start for SLSQP
    inits = [best_h / np.sum(best_h)]
    inits.append(np.ones(N) / N)
    
    for _ in range(50):
        if time.time() > deadline: break
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for h_init in inits:
        if time.time() > deadline: break
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # 3. Softmax L-BFGS-B Optimization
    # Objective: Maximize log(L2^2) - log(L1) - log(softmax(Conv, T))
    # Variables: log(h) -> h = exp(log_h)
    # Constraint: sum(exp(log_h)) = 1.
    # We will use penalty method or just optimize over h directly with bounds?
    # L-BFGS-B supports bounds. Let's optimize h directly but use soft-max for the max term.
    # This makes the objective smooth.
    
    def softmax_max(conv, T):
        # Numerically stable softmax max
        m = np.max(conv)
        return m + T * np.log(np.sum(np.exp((conv - m) / T)))
    
    def neg_obj_softmax(h, T):
        h = np.clip(h, 1e-12, None)
        l1 = np.sum(h)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        sm_max = softmax_max(h_conv, T)
        # Maximize l2_sq / (l1 * sm_max)
        # Minimize -log(l2_sq) + log(l1) + log(sm_max)
        # To avoid log(0) issues, we clip.
        # Since L1 is not fixed in this formulation, the scale invariance is handled by the optimizer?
        # Actually, the ratio is scale invariant. So we can fix L1=1 via projection or penalty.
        # Let's fix L1=1 by normalizing h inside the function? No, that breaks gradients.
        # Better: optimize over h with sum(h)=1 constraint? L-BFGS-B doesn't do equality constraints.
        # So we optimize over h and accept that the scale is arbitrary?
        # If we don't fix scale, the optimizer might drift.
        # Let's fix scale by normalizing h at the start of each call? No.
        # Let's use the fact that the objective is homogeneous of degree 0.
        # We can just run L-BFGS-B on h, and it will find a direction of improvement.
        # However, without fixing scale, the "length" of h might change.
        # Let's add a penalty for sum(h) != 1.
        
        penalty = 100.0 * (l1 - 1.0)**2
        
        if l2_sq < 1e-12: return 0
        if sm_max < 1e-12: return 0
        
        return -np.log(l2_sq) + np.log(l1) + np.log(sm_max) + penalty
    
    def grad_neg_obj_softmax(h, T):
        h = np.clip(h, 1e-12, None)
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        
        # Gradient of softmax max
        m = np.max(h_conv)
        exp_term = np.exp((h_conv - m) / T)
        sum_exp = np.sum(exp_term)
        # d/dh (m + T log sum exp((conv-m)/T))
        # This is complex. Let's use numerical gradient or finite difference if needed?
        # No, we need analytic for speed.
        
        # Let S = sum(exp((conv_k - m)/T))
        # Softmax Max = m + T log S
        # d(softmax_max)/dh_i = sum_k (d conv_k / dh_i) * (d softmax_max / d conv_k)
        # d softmax_max / d conv_k = (1/T) * (exp((conv_k-m)/T) / S) - (1/T) * (sum_j exp((conv_j-m)/T)/S) * (1/T) * exp((conv_k-m)/T) ?
        # Actually, max is approximated.
        # Let's use a simpler approach: The gradient of the log-softmax is the softmax probabilities.
        # But we have max + T log sum exp.
        # Let's just use finite difference for the gradient? Too slow for L-BFGS-B.
        
        # Alternative: Use the epigraph formulation with SLSQP which handles the non-smoothness better via constraints.
        # The prompt asks for L-BFGS-B with softmax.
        
        # Let's implement the gradient properly.
        # f(h) = log( sum h_i^2 ) - log( sum h_i ) - log( SoftmaxMax(Conv(h)) )
        # d/dh_i log(sum h^2) = 2 h_i / sum h^2
        # d/dh_i log(sum h) = 1 / sum h
        # d/dh_i log(SoftmaxMax) = - (1 / SoftmaxMax) * d(SoftmaxMax)/dh_i
        
        # d(SoftmaxMax)/dh_i = sum_k (d Conv_k / dh_i) * p_k
        # where p_k = exp((Conv_k - M)/T) / sum_j exp((Conv_j - M)/T)
        # Note: d Conv_k / dh_i is 1 if k-i is in [0, N-1] (and symmetric).
        # Specifically, Conv_k = sum_j h_j h_{k-j}.
        # d Conv_k / dh_i = h_{k-i} + h_{i-k} (if indices valid)
        
        p = exp_term / sum_exp
        # p is the "weights" for the gradient of the soft max.
        
        grad_conv = np.zeros(N)
        for i in range(N):
            # h_i contributes to Conv_k where k = i + j for j in 0..N-1
            # So k ranges from i to i+N-1.
            # d Conv_k / dh_i = h_{k-i}
            k_start = i
            k_end = min(i + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - i
                grad_conv[i] = np.sum(p[indices_k] * h[indices_h])
                # Also symmetric part?
                # Conv is symmetric. Conv_k = Conv_{2N-2-k}.
                # The derivative sum accounts for both if we sum over all k.
                # Wait, Conv_k = sum_{j=0}^{N-1} h_j h_{k-j}.
                # If we fix i, h_i appears in term j=i (h_i h_{k-i}) and term j=k-i (h_{k-i} h_i).
                # So d Conv_k / dh_i = h_{k-i} + h_{i-k} (with 0 if out of bounds).
                # My loop above only does one direction?
                # Let's be precise.
                # grad_conv[i] = sum_k p_k * (h_{k-i} + h_{i-k})
                # The loop above sums p[indices_k] * h[indices_h] where indices_h = k-i.
                # This covers h_{k-i}.
                # We need to add h_{i-k}.
                # h_{i-k} is valid if 0 <= i-k < N => k <= i and k >= i-N+1.
                # k_start2 = max(0, i-N+1)
                # k_end2 = min(i, NUM_CONV)
                # Let's just compute the full gradient vector for conv derivatives?
                # It's O(N^2). N=50, so 2500 ops. Fine.
        
        # Recalculate grad_conv properly
        grad_conv = np.zeros(N)
        for k in range(NUM_CONV):
            if p[k] < 1e-15: continue
            # d Conv_k / dh_i
            # h_i h_{k-i} and h_{k-i} h_i
            # If 0 <= k-i < N: term is h_{k-i}
            # If 0 <= i-k < N: term is h_{i-k}
            # Since Conv is symmetric, these are the same set of pairs?
            # No, for a specific k, the terms are distinct unless i = k-i.
            # Let's just iterate i.
            
        # Better:
        for i in range(N):
            # Contribution to grad_conv[i]
            # Sum over k of p_k * (h_{k-i} + h_{i-k})
            # h_{k-i} is non-zero if 0 <= k-i < N => i <= k < i+N
            # h_{i-k} is non-zero if 0 <= i-k < N => i-N < k <= i
            val = 0.0
            # Part 1: k in [i, i+N-1]
            k1_start = i
            k1_end = min(i + N, NUM_CONV)
            if k1_start < NUM_CONV and k1_end > 0:
                idx_k = np.arange(k1_start, k1_end)
                idx_h = idx_k - i
                val += np.sum(p[idx_k] * h[idx_h])
            # Part 2: k in [i-N+1, i]
            k2_start = max(0, i - N + 1)
            k2_end = min(i, NUM_CONV)
            if k2_start < NUM_CONV and k2_end >= 0:
                idx_k = np.arange(k2_start, k2_end + 1)
                idx_h = i - idx_k
                val += np.sum(p[idx_k] * h[idx_h])
            grad_conv[i] = val
            
        grad_log_l2 = 2.0 * h / l2_sq
        grad_log_l1 = 1.0 / l1
        grad_soft_max = grad_conv / softmax_max(h_conv, T)
        
        grad = -grad_log_l2 + grad_log_l1 + grad_soft_max
        # Penalty gradient
        grad += 200.0 * (l1 - 1.0)
        
        return grad

    # Run L-BFGS-B with annealing T
    h_current = best_h / np.sum(best_h)
    
    for T in [1.0, 0.5, 0.2, 0.1, 0.05]:
        if time.time() > deadline: break
        try:
            res_lbfgs = minimize(
                neg_obj_softmax, 
                h_current, 
                method='L-BFGS-B', 
                jac=lambda h: grad_neg_obj_softmax(h, T),
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_lbfgs.success:
                h_candidate = res_lbfgs.x
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
                        h_current = h_candidate
        except Exception:
            pass

    # 4. Final Polish with SLSQP
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
