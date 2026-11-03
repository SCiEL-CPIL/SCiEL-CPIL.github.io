"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Strategy: smooth log-sum-exp approximation of the autoconvolution peak constraint,
transforming the non-differentiable max into a strictly convex penalty. Maximizes
log(L2^2) - log(L1) - logsumexp(conv) via L-BFGS-B on log-parameterized heights.
"""
import numpy as np
import time
import scipy.optimize as opt

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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    x_idx = np.arange(N)
    num_conv = 2 * N - 1
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    def logsumexp(v, beta):
        """Numerically stable log-sum-exp with temperature beta."""
        v_scaled = v * beta
        m = np.max(v_scaled)
        if m < -1e12:
            return -np.inf
        return m + np.log(np.sum(np.exp(v_scaled - m))) / beta
    
    def smooth_objective_and_grad(z, beta):
        """
        Minimize: -log(sum h^2) + log(sum h) + logsumexp(conv, beta)
        where h_i = exp(z_i) (log-parameterization ensures positivity).
        Returns (value, gradient).
        """
        h = np.exp(z)
        l1 = np.sum(h)
        l2_sq = np.sum(h ** 2)
        
        if l1 < 1e-30 or l2_sq < 1e-30:
            return 1e10, np.zeros(N)
        
        conv = np.convolve(h, h, mode='full')
        lse = logsumexp(conv, beta)
        
        val = -np.log(l2_sq) + np.log(l1) + lse
        
        # Gradient of -log(l2_sq) w.r.t. z: -2*h / l2_sq
        # Gradient of log(l1) w.r.t. z: h / l1
        # Gradient of logsumexp(conv, beta) w.r.t. z:
        #   d/dz_i [logsumexp(conv)] = sum_k (conv[k] * exp(conv[k]*beta) / sum(conv*exp(conv*beta))) * d(conv[k])/dz_i
        #   d(conv[k])/dz_i = h_i * (h[k-i] + h[k-(N-1-i)]) for valid indices
        #   Combined: h_i * sum_k (p_k * (h[k-i] + h[k-(N-1-i)]))
        
        # Compute softmax weights for conv
        conv_scaled = conv * beta
        m_conv = np.max(conv_scaled)
        exp_conv = np.exp(conv_scaled - m_conv)
        p = exp_conv / np.sum(exp_conv)  # softmax weights
        
        # Gradient of logsumexp w.r.t. conv: beta * p
        # Gradient of conv w.r.t. z_i: h_i * (h[k-i] + h[k-(N-1-i)])
        # So grad_lse[i] = h_i * sum_k (beta * p_k * (h[k-i] + h[k-(N-1-i)]))
        
        # Precompute: for each i, sum_k p_k * h[k-i] and sum_k p_k * h[k-(N-1-i)]
        # This is a correlation-like operation
        # sum_k p_k * h[k-i] = (p * h)[i] in the appropriate convolution sense
        
        # conv[k] = sum_j h[j] * h[k-j]
        # d(conv[k])/dz_i = h[i] * h[k-i] + h[i] * h[k-(N-1-i)] (with boundary checks)
        # sum_k beta*p[k] * d(conv[k])/dz_i = beta * h[i] * sum_k p[k]*(h[k-i] + h[k-(N-1-i)])
        
        # Compute s1[i] = sum_k p[k] * h[k-i] for k-i in [0, N-1]
        # This is: s1[i] = sum_j p[i+j] * h[j] where i+j in [0, num_conv-1], j in [0, N-1]
        # = sum_j p[i+j] * h[j]
        
        # Using convolution: s1 = correlate(p, h) appropriately
        # Actually: s1[i] = sum_j h[j] * p[i+j]
        # Let's compute via direct indexing for clarity (N=50, num_conv=99, manageable)
        
        s1 = np.zeros(N)
        s2 = np.zeros(N)
        for i in range(N):
            # s1[i] = sum_k p[k] * h[k-i] where k-i in [0,N-1] => k in [i, i+N-1]
            k_lo = i
            k_hi = min(i + N - 1, num_conv - 1)
            if k_lo <= k_hi:
                j_range = np.arange(k_lo - i, k_hi - i + 1)
                s1[i] = np.dot(p[k_lo:k_hi+1], h[j_range])
            
            # s2[i] = sum_k p[k] * h[k-(N-1-i)] where k-(N-1-i) in [0,N-1] => k in [N-1-i, 2N-2-i]
            k_lo2 = N - 1 - i
            k_hi2 = min(2 * N - 2 - i, num_conv - 1)
            if k_lo2 <= k_hi2:
                j_range2 = np.arange(k_lo2 - (N - 1 - i), k_hi2 - (N - 1 - i) + 1)
                s2[i] = np.dot(p[k_lo2:k_hi2+1], h[j_range2])
        
        grad_lse = beta * h * (s1 + s2)
        
        grad = -2.0 * h / l2_sq + h / l1 + grad_lse
        
        return val, grad
    
    # Generate diverse initializations in log-space
    np.random.seed(42)
    z_inits = []
    
    # 1. Uniform
    z_inits.append(np.zeros(N))
    
    # 2. Sparse: few large values
    for trial in range(30):
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        z = np.full(N, -5.0)
        z[idx] = 0.0
        z_inits.append(z)
        
    # 3. Gaussian bumps
    for c in np.linspace(5, N-5, 8):
        for w in [1.0, 2.0, 3.0, 5.0]:
            z = -0.5 * ((x_idx - c) / w) ** 2
            z_inits.append(z)
            
    # 4. Triangular
    for c in np.linspace(5, N-5, 6):
        for w in [3.0, 5.0, 8.0, 12.0]:
            z = np.maximum(-10.0, -np.abs(x_idx - c) / w)
            z_inits.append(z)
            
    # 5. Random smooth
    for _ in range(20):
        z_r = np.random.randn(N) * 1.5
        kernel = np.ones(5) / 5
        z_r = np.convolve(z_r, kernel, mode='same')
        z_inits.append(z_r)
        
    # 6. Symmetric pairs
    for k in range(2, 8):
        for offset in range(2, 10):
            z = np.full(N, -5.0)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    z[pos1] = 0.0
                if 0 <= pos2 < N:
                    z[pos2] = 0.0
            z_inits.append(z)
            
    # 7. Alternating
    for gap in [2, 3, 4, 5, 7]:
        for start in range(gap):
            z = np.full(N, -5.0)
            for i in range(start, N, gap):
                z[i] = 0.0
            z_inits.append(z)
    
    # --- Phase 1: Annealed log-sum-exp optimization with L-BFGS-B ---
    # Start with large beta (smooth) and decrease to small beta (sharp, approximating true max)
    beta_schedule = [50.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.2, 0.15, 0.1]
    
    # Sort inits by raw score, keep top ones
    init_scores = []
    for z in z_inits:
        h = np.exp(z)
        h = np.maximum(h, 1e-30)
        h /= np.sum(h)
        s = _score_raw(h)
        init_scores.append((s, z))
    
    init_scores.sort(key=lambda x: -x[0])
    top_z_inits = [z for _, z in init_scores[:20]]
    
    # For each beta level, optimize from the best current solution
    current_best_z = top_z_inits[0].copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 10:
            break
        
        # Optimize from current best
        try:
            res = opt.minimize(
                lambda z: smooth_objective_and_grad(z, beta)[0],
                current_best_z,
                jac=lambda z: smooth_objective_and_grad(z, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            current_best_z = res.x
        except Exception:
            pass
        
        # Also try a few other inits at this beta level
        for z_init in top_z_inits[1:5]:
            if time.time() > deadline - 10:
                break
            try:
                res = opt.minimize(
                    lambda z: smooth_objective_and_grad(z, beta)[0],
                    z_init,
                    jac=lambda z: smooth_objective_and_grad(z, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
                )
                # Compare
                h_new = np.exp(res.x)
                h_new = np.maximum(h_new, 1e-30)
                h_new /= np.sum(h_new)
                s_new = _score_raw(h_new)
                
                h_cur = np.exp(current_best_z)
                h_cur = np.maximum(h_cur, 1e-30)
                h_cur /= np.sum(h_cur)
                s_cur = _score_raw(h_cur)
                
                if s_new > s_cur:
                    current_best_z = res.x
            except Exception:
                pass
        
        # Update best
        h_candidate = np.exp(current_best_z)
        h_candidate = np.maximum(h_candidate, 1e-30)
        h_candidate /= np.sum(h_candidate)
        s_candidate = _score_raw(h_candidate)
        if s_candidate > best_score:
            best_score = s_candidate
            best_h = h_candidate.copy()
    
    # --- Phase 2: Fine-tune with very small beta from best solution ---
    for beta in [0.1, 0.05, 0.03, 0.02, 0.01]:
        if time.time() > deadline - 5:
            break
        try:
            res = opt.minimize(
                lambda z: smooth_objective_and_grad(z, beta)[0],
                current_best_z,
                jac=lambda z: smooth_objective_and_grad(z, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            h_candidate = np.exp(res.x)
            h_candidate = np.maximum(h_candidate, 1e-30)
            h_candidate /= np.sum(h_candidate)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
                current_best_z = res.x
        except Exception:
            pass
    
    # --- Phase 3: Local search refinement on the best h ---
    if time.time() < deadline - 3:
        h = best_h.copy()
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            improved = False
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                # Try small perturbations
                for eps in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                    h_test = h.copy()
                    h_test[i] += eps
                    h_test[j] -= eps
                    if np.any(h_test < 0):
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_score:
                        best_score = s_test
                        best_h = h_test.copy()
                        h = h_test.copy()
                        improved = True
                        break
    
    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return list(best_h)
