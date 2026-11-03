import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N

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

def _score_normalized(h):
    """Score assuming sum(h)=1. Returns l2_sq / max_conv."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h = h / l1
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / autoconv_max)

def _project_simplex(v, s=1.0):
    """Project v onto the simplex {x >= 0, sum(x) = s}."""
    n = len(v)
    if np.sum(v) < 1e-15:
        return np.ones(n) / n * s
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def _gradient_ascent(h_init, max_iter=200, deadline=None):
    h = h_init.copy()
    h = _project_simplex(h, 1.0)
    best_score = _score_normalized(h)
    best_h = h.copy()
    
    # Adaptive step size
    alpha = 0.1
    last_score = best_score
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        Q = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        k_star = np.argmax(conv)
        M = conv[k_star]
        
        if M < 1e-15:
            break
            
        # Gradient of M w.r.t h
        # v_j = h_{k_star - j}
        v = np.zeros(N)
        for j in range(N):
            idx = k_star - j
            if 0 <= idx < N:
                v[j] = h[idx]
        grad_M = 2 * v
        
        # Gradient of Q w.r.t h
        grad_Q = 2 * h
        
        # Gradient of J = Q/M
        # dJ = (dQ * M - Q * dM) / M^2
        grad_J = (grad_Q * M - Q * grad_M) / (M * M)
        
        # Update
        h_new = h + alpha * grad_J
        h_new = _project_simplex(h_new, 1.0)
        
        score_new = _score_normalized(h_new)
        
        if score_new > best_score:
            best_score = score_new
            best_h = h_new.copy()
            
        # Simple adaptive step size
        if score_new > last_score:
            alpha *= 1.1
        else:
            alpha *= 0.5
            
        h = h_new
        last_score = score_new
        
        # Convergence check
        if abs(score_new - best_score) < 1e-10 and i > 20:
             # Try smaller steps for a bit
             if i > 50:
                 break
                 
    return best_h, best_score

def _nelder_mead_symmetric(z_init, max_iter=100, deadline=None):
    """
    Optimize symmetric h using Nelder-Mead on half-vector z.
    h is constructed from z: h_i = z_i for i < N/2, h_i = z_{N-1-i} for i >= N/2.
    Constraint: sum(h) = 1 => 2*sum(z) = 1 => sum(z) = 0.5.
    We reparametrize z = 0.5 * w, where w is on the standard simplex of dimension N/2.
    """
    half_N = N // 2
    
    def obj(w):
        if np.any(w < -1e-10):
            return 1e6
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s < 1e-10:
            return 1e6
        w = w / s
        z = 0.5 * w
        # Construct h
        h = np.zeros(N)
        h[:half_N] = z
        h[half_N:] = z[::-1]
        # h should sum to 1
        s_h = np.sum(h)
        if s_h < 1e-10:
            return 1e6
        h = h / s_h
        return -_score_normalized(h)
        
    res = opt.minimize(obj, z_init, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
    
    w = res.x
    w = np.maximum(w, 0)
    s = np.sum(w)
    if s < 1e-10:
        return None, 0.0
    w = w / s
    z = 0.5 * w
    h = np.zeros(N)
    h[:half_N] = z
    h[half_N:] = z[::-1]
    h = h / np.sum(h)
    
    return h, _score_normalized(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    half_N = N // 2
    
    # Generate diverse initializations for symmetric search
    inits_w = []
    
    # 1. Uniform
    inits_w.append(np.ones(half_N) / half_N)
    
    # 2. Sparse (centered)
    for k in [1, 2, 3, 5, 10]:
        w = np.zeros(half_N)
        center = half_N // 2
        for i in range(k):
            idx = center - i
            if 0 <= idx < half_N:
                w[idx] = 1.0
            idx2 = center + i
            if 0 <= idx2 < half_N:
                w[idx2] = 1.0
        s = np.sum(w)
        if s > 0:
            w /= s
            inits_w.append(w)
            
    # 3. Random
    np.random.seed(42)
    for _ in range(20):
        w = np.random.rand(half_N)
        w /= np.sum(w)
        inits_w.append(w)
        
    # 4. Gaussian-like
    for sigma in [1, 2, 5, 10]:
        x = np.arange(half_N)
        w = np.exp(-0.5 * ((x - half_N/2) / sigma)**2)
        w /= np.sum(w)
        inits_w.append(w)

    # Phase 1: Nelder-Mead on symmetric subspace
    for w_init in inits_w:
        if time.time() > deadline - 30:
            break
        h_candidate, score_candidate = _nelder_mead_symmetric(w_init, max_iter=200, deadline=deadline - 30)
        if h_candidate is not None:
            raw_score = _score_raw(h_candidate)
            if raw_score > best_score:
                best_score = raw_score
                best_h = h_candidate.copy()
                
    # Phase 2: Gradient Ascent on full space, initialized from best symmetric solution
    # We run multiple gradient ascent runs with different step sizes/initializations derived from best_h
    
    # 1. Direct refinement of best_h
    if time.time() < deadline - 10:
        h_refined, score_refined = _gradient_ascent(best_h, max_iter=100, deadline=deadline - 10)
        raw_score = _score_raw(h_refined)
        if raw_score > best_score:
            best_score = raw_score
            best_h = h_refined.copy()

    # 2. Perturb best_h and refine
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        h_perturbed = best_h + np.random.normal(0, 0.01, N)
        h_perturbed = np.maximum(h_perturbed, 0)
        h_perturbed = _project_simplex(h_perturbed, 1.0)
        
        h_refined, score_refined = _gradient_ascent(h_perturbed, max_iter=50, deadline=deadline - 5)
        raw_score = _score_raw(h_refined)
        if raw_score > best_score:
            best_score = raw_score
            best_h = h_refined.copy()

    # Phase 3: Final polish
    if time.time() < deadline - 2:
        h_refined, score_refined = _gradient_ascent(best_h, max_iter=50, deadline=deadline - 2)
        raw_score = _score_raw(h_refined)
        if raw_score > best_score:
            best_score = raw_score
            best_h = h_refined.copy()

    return list(best_h)

# EVOLVE-BLOCK-END
