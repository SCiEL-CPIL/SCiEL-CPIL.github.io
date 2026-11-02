import numpy as np
from scipy.optimize import minimize, linprog
import time
import warnings

N = 50
DEADLINE = 145
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _autoconvolution_fft(h):
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_padded = np.zeros(size)
    h_padded[:n] = h
    fft_h = np.fft.fft(h_padded)
    conv = np.fft.ifft(fft_h * fft_h).real
    return conv[:2 * n - 1]

def _score(h):
    h = np.array(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0, None)
    total = np.sum(h)
    if total < 1e-15:
        return 0.0
    h_norm = h / total
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconvolution_fft(h_norm)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / max_conv

def _score_grad(h):
    """Compute score and gradient w.r.t. h (unnormalized)."""
    h = np.clip(np.array(h, dtype=float), 0, None)
    total = np.sum(h)
    if total < 1e-15:
        return 0.0, np.zeros_like(h)
    h_norm = h / total
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconvolution_fft(h_norm)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0, np.zeros_like(h)
    
    score = l2_sq / max_conv
    
    # Gradient of score w.r.t. h
    # score = (sum h_i^2 / total^2) / max_conv
    # = (sum h_i^2) / (total^2 * max_conv)
    # d(score)/d(h_k) = [2*h_k * total^2 * max_conv - sum(h_i^2) * 2*total * max_conv - sum(h_i^2)*total^2 * d(max_conv)/d(h_k)] / (total^4 * max_conv^2)
    
    # Simplify: let S = sum(h_i^2), T = total, M = max_conv
    # score = S / (T^2 * M)
    # dS/dh_k = 2*h_k
    # dT/dh_k = 1
    # dM/dh_k = sum_i h_i * delta_{i+k, argmax} (for the max entry)
    
    S = l2_sq
    T = total
    M = max_conv
    
    # Find argmax of conv
    max_idx = np.argmax(conv)
    
    # dM/dh_k = sum_i h_norm[i] * [i+k == max_idx]
    dM_dh = np.zeros(N)
    for k in range(N):
        i_start = max(0, max_idx - k)
        i_end = min(N - 1, max_idx - k)
        # Actually d(conv[max_idx])/d(h_k) = sum_i h_norm[i] where i+k = max_idx
        # i = max_idx - k, so if 0 <= max_idx-k < N, contribution is h_norm[max_idx-k]
        i = max_idx - k
        if 0 <= i < N:
            dM_dh[k] = h_norm[i]
    
    # d(score)/d(h_k) = [2*h_k * T^2 * M - S * 2*T * M - S * T^2 * dM_dh[k]] / (T^4 * M^2)
    # = [2*h_k / T - 2*S/(T^2) - S*dM_dh[k]/(T*M)] / M  ... let me redo
    # score = S / (T^2 * M)
    # d/dh_k = (dS/dh_k * T^2 * M - S * d(T^2 * M)/dh_k) / (T^4 * M^2)
    # d(T^2*M)/dh_k = 2T*M + T^2*dM/dh_k
    # d/dh_k = (2*h_k * T^2 * M - S*(2T*M + T^2*dM_dh[k])) / (T^4 * M^2)
    # = (2*h_k / (T*M) - 2*S/(T^3) - S*dM_dh[k]/(T^2*M)) / M  ... 
    # Let me just compute numerically for safety in the gradient-based part
    
    grad = np.zeros(N)
    eps = 1e-7
    for k in range(N):
        h_plus = h.copy()
        h_plus[k] += eps
        s_plus = _score(h_plus)
        h_minus = h.copy()
        h_minus[k] = max(0, h[k] - eps)
        s_minus = _score(h_minus)
        if h[k] > eps:
            grad[k] = (s_plus - s_minus) / (2 * eps)
        else:
            grad[k] = (s_plus - s_minus) / eps
    
    return score, grad

def _reweighting_fixed_support(h_init, support, max_iter=300, tol=1e-12):
    """
    Iterative reweighting: for fixed support, find optimal amplitudes.
    Uses the fixed-point: h_i <- h_i * (2*h_i / max_conv) / normalization
    This comes from the KKT conditions of the optimization.
    """
    h = np.zeros(N)
    h[support] = h_init[support].copy()
    total = np.sum(h)
    if total < 1e-15:
        return h
    
    h_norm = h / total
    
    for it in range(max_iter):
        if _time_remaining() < 5:
            break
        
        conv = _autoconvolution_fft(h_norm)
        max_conv = np.max(conv)
        
        if max_conv < 1e-15:
            break
        
        # The optimal h_i for fixed support satisfies:
        # 2*h_i / max_conv = lambda (constant) for all i in support
        # So h_i is proportional to max_conv/2, meaning all equal? No.
        # Actually the KKT: d/dh_i [sum h_j^2 / max_conv] = 0
        # 2*h_i / max_conv = mu + nu_i (nu_i >= 0 for h_i >= 0)
        # For interior points: 2*h_i / max_conv = mu => h_i = mu*max_conv/2 (all equal)
        # But max_conv depends on h, so it's a fixed point.
        
        # Better: use the fact that at optimum, the "pressure" is balanced.
        # Use multiplicative update: h_i <- h_i * (2*h_i / max_conv) / mean(2*h_j/max_conv for j in support)
        # This is like power iteration on the "convolution operator"
        
        # Actually, let's use a different approach: 
        # Maximize sum(h_i^2) subject to max_k conv_k <= 1, sum(h_i) = 1
        # This is a convex optimization (maximize convex? No, sum h^2 is convex)
        # Hmm, maximizing convex is non-convex.
        
        # Let's use gradient ascent on the log barrier formulation
        # Or simply: coordinate descent
        
        # Coordinate descent approach
        improved = False
        for idx in support:
            old_val = h[idx]
            if old_val < 1e-15:
                # Try adding a small amount
                h_trial = h.copy()
                h_trial[idx] = 1e-4
                s_trial = _score(h_trial)
                s_old = _score(h)
                if s_trial > s_old:
                    h[idx] = 1e-4
                    improved = True
                continue
            
            # Golden section search for optimal h[idx]
            a = 0.0
            b = old_val * 5 + 0.1
            gr = (np.sqrt(5) - 1) / 2
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            h_c = h.copy()
            h_c[idx] = c
            h_d = h.copy()
            h_d[idx] = d
            
            fc = _score(h_c)
            fd = _score(h_d)
            
            for _ in range(40):
                if b - a < 1e-10:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    h_c = h.copy()
                    h_c[idx] = c
                    fc = _score(h_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    h_d = h.copy()
                    h_d[idx] = d
                    fd = _score(h_d)
            
            opt_val = (a + b) / 2
            if abs(opt_val - old_val) > 1e-10:
                h[idx] = opt_val
                improved = True
        
        if not improved:
            break
    
    return h

def _gradient_ascent(h_init, support, max_iter=200, lr=0.01):
    """Gradient ascent with adaptive step size."""
    h = np.zeros(N)
    h[support] = h_init[support].copy()
    
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if _time_remaining() < 5:
            break
        
        score, grad = _score_grad(h)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        # Project gradient onto support
        grad_support = np.zeros(N)
        grad_support[support] = grad[support]
        
        # Adaptive step size
        step = lr
        for trial in range(10):
            h_trial = h + step * grad_support
            h_trial = np.clip(h_trial, 0, None)
            s_trial = _score(h_trial)
            if s_trial > score:
                h = h_trial
                break
            step *= 0.5
        
        if step < 1e-15:
            break
    
    return best_h if _score(best_h) > _score(h) else h

def _support_expansion(h_current, max_new_points=5):
    """Try adding new points to the support."""
    current_support = set(np.where(h_current > 1e-10)[0])
    candidates = [i for i in range(N) if i not in current_support]
    
    if not candidates:
        return h_current
    
    best_h = h_current.copy()
    best_score = _score(h_current)
    
    # Try adding each candidate one at a time
    for idx in candidates:
        if _time_remaining() < 10:
            break
        
        h_trial = h_current.copy()
        h_trial[idx] = 0.01  # Small initial value
        
        # Optimize on the new support
        new_support = sorted(current_support | {idx})
        h_opt = _reweighting_fixed_support(h_trial, new_support, max_iter=100)
        s_opt = _score(h_opt)
        
        if s_opt > best_score + 1e-8:
            best_score = s_opt
            best_h = h_opt.copy()
    
    return best_h

def _sdp_relaxation(support):
    """
    SDP relaxation: maximize sum(h_i^2) s.t. max_k conv_k <= 1, sum(h_i) = 1, h_i >= 0
    Relax the rank-1 constraint on M = h h^T to M >= 0 (PSD)
    
    The autoconvolution constraint: for all k, sum_{i+j=k} h_i h_j <= 1
    In terms of M: for all k, sum_{i+j=k} M_{i,j} <= 1
    
    We solve: maximize trace(M) s.t. M >= 0, M_{i,j} = 0 for i,j not in support,
    sum_i M_{i,i} = sum_i h_i^2, and the convolution constraints.
    
    Actually, let's use a simpler approach: solve the dual or use a penalty method.
    """
    s = len(support)
    if s < 2:
        return np.zeros(N)
    
    # Use scipy's minimize with SLSQP on the convex subproblem
    # Maximize sum(h_i^2) s.t. max_k conv_k <= 1, sum(h_i) = 1, h_i >= 0
    
    x0 = np.ones(s) / s
    
    def objective(x):
        """Negative of sum(x^2) for minimization"""
        h_full = np.zeros(N)
        h_full[support] = x
        return -np.sum(x ** 2)
    
    def constraint_conv(x):
        """max_k conv_k <= 1"""
        h_full = np.zeros(N)
        h_full[support] = x
        conv = _autoconvolution_fft(h_full)
        return 1.0 - np.max(conv)
    
    def constraint_sum(x):
        """sum(x) = 1"""
        return np.sum(x) - 1.0
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_conv},
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    bounds = [(0, None)] * s
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
        if result.success:
            h_full = np.zeros(N)
            h_full[support] = np.clip(result.x, 0, None)
            return h_full
    except:
        pass
    
    return None

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Initial candidates
    np.random.seed(12345)
    
    # Candidate 1: Two-point [2/3, 1/3]
    h1 = np.zeros(N)
    h1[0] = 2.0/3
    h1[1] = 1.0/3
    s1 = _score(h1)
    if s1 > best_score:
        best_score = s1
        best_h = h1.copy()
    
    # Candidate 2: Uniform on 2 points
    h2 = np.zeros(N)
    h2[0] = 0.5
    h2[1] = 0.5
    s2 = _score(h2)
    if s2 > best_score:
        best_score = s2
        best_h = h2.copy()
    
    # Candidate 3: Three-point
    h3 = np.zeros(N)
    h3[0] = 0.5
    h3[1] = 0.25
    h3[2] = 0.25
    s3 = _score(h3)
    if s3 > best_score:
        best_score = s3
        best_h = h3.copy()
    
    # Candidate 4: Geometric
    h4 = np.zeros(N)
    for i in range(5):
        h4[i] = 0.5 ** i
    h4 /= np.sum(h4)
    s4 = _score(h4)
    if s4 > best_score:
        best_score = s4
        best_h = h4.copy()
    
    # Generate diverse support sets
    support_candidates = []
    
    # Contiguous supports of various sizes
    for size in range(2, 12):
        for start in range(0, N - size + 1, 1):
            support_candidates.append(list(range(start, start + size)))
    
    # Sparse supports
    for _ in range(50):
        size = np.random.randint(2, 15)
        support = np.random.choice(N, size, replace=False)
        support.sort()
        support_candidates.append(list(support))
    
    # Structured supports: symmetric around center
    center = N // 2
    for radius in range(1, 8):
        support = list(range(max(0, center - radius), min(N, center + radius + 1)))
        support_candidates.append(support)
    
    # Optimize on each support
    for support in support_candidates:
        if _time_remaining() < 15:
            break
        
        # Multiple initializations
        for init_type in range(4):
            if _time_remaining() < 8:
                break
            
            h_init = np.zeros(N)
            if init_type == 0:
                h_init[support] = 1.0 / len(support)
            elif init_type == 1:
                h_init[support] = np.random.rand(len(support))
                h_init /= np.sum(h_init)
            elif init_type == 2:
                weights = np.exp(-np.arange(len(support)) * 0.5)
                h_init[support] = weights
                h_init /= np.sum(h_init)
            else:
                # Bell-shaped
                mid = len(support) / 2
                weights = np.exp(-0.5 * (np.arange(len(support)) - mid) ** 2 / (len(support) / 2))
                h_init[support] = weights
                h_init /= np.sum(h_init)
            
            # Run reweighting
            h_opt = _reweighting_fixed_support(h_init, support, max_iter=200)
            s_opt = _score(h_opt)
            
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
                # Gradient polish
                if _time_remaining() > 10:
                    h_grad = _gradient_ascent(h_opt, support, max_iter=100, lr=0.05)
                    s_grad = _score(h_grad)
                    if s_grad > best_score:
                        best_score = s_grad
                        best_h = h_grad.copy()
    
    # Support expansion phase
    if _time_remaining() > 20:
        for _ in range(5):
            if _time_remaining() < 10:
                break
            h_expanded = _support_expansion(best_h, max_new_points=3)
            s_expanded = _score(h_expanded)
            if s_expanded > best_score:
                best_score = s_expanded
                best_h = h_expanded.copy()
            else:
                break
    
    # Final high-precision polish
    if _time_remaining() > 10:
        support_best = list(np.where(best_h > 1e-10)[0])
        if len(support_best) > 0:
            h_final = _reweighting_fixed_support(best_h, support_best, max_iter=300)
            s_final = _score(h_final)
            if s_final > best_score:
                best_score = s_final
                best_h = h_final.copy()
            
            # Gradient polish
            h_grad = _gradient_ascent(best_h, support_best, max_iter=100, lr=0.02)
            s_grad = _score(h_grad)
            if s_grad > best_score:
                best_score = s_grad
                best_h = h_grad.copy()
    
    # Ensure valid output
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) < 1e-15:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
    
    return best_h
