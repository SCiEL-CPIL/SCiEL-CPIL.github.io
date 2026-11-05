import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_conv(h, beta):
    conv = np.convolve(h, h)
    # Soft max via log-sum-exp, scaled by beta
    m = np.max(conv)
    return m + beta * np.log(np.sum(np.exp((conv - m) / beta)))

def _smooth_objective(h, beta=20.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    sm = _soft_max_conv(h, beta)
    return abs(2.0 * N * sm / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    i = np.arange(N)
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # 2.0
    
    # DCT basis: use type-II DCT for efficient spectral representation
    # h[i] = a + sum_k c_k * cos(pi*(i+0.5)*k/N)
    # This gives orthogonal basis on [0, N)
    
    def dct_basis(k):
        return np.cos(np.pi * (i + 0.5) * k / N)
    
    # Precompute DCT basis for first M modes
    M = 12  # number of spectral modes to optimize
    
    def params_to_h(params):
        """Convert [a, c_1, ..., c_M] to h vector"""
        h = np.full(N, params[0])
        for k in range(1, M + 1):
            h += params[k] * dct_basis(k)
        return h
    
    def h_to_params(h):
        """Project h onto DCT basis"""
        a = np.mean(h)
        params = [a]
        for k in range(1, M + 1):
            c_k = 2.0 / N * np.sum(h * dct_basis(k))
            params.append(c_k)
        return np.array(params)
    
    # Try several different initial spectral configurations
    configs = []
    
    # Config 1: Pure DC + first harmonic (like seed)
    configs.append(np.concatenate([[1.0], [1.0] + np.zeros(M - 1)]))
    
    # Config 2: DC + alternating signs on low modes
    p2 = np.concatenate([[1.0], np.array([(-1)**k * 0.5 for k in range(1, M+1)])])
    configs.append(p2)
    
    # Config 3: Emphasize mode 1 and 2
    p3 = np.zeros(M + 1)
    p3[0] = 1.0
    p3[1] = 0.8
    p3[2] = -0.4
    configs.append(p3)
    
    # Config 4: More aggressive low-frequency
    p4 = np.zeros(M + 1)
    p4[0] = 1.0
    for k in range(1, min(5, M+1)):
        p4[k] = 0.3 * (-1)**k / k
    configs.append(p4)
    
    # Config 5: Random low-frequency
    rng = np.random.RandomState(42)
    p5 = np.zeros(M + 1)
    p5[0] = 1.0
    for k in range(1, M + 1):
        p5[k] = rng.randn() * 0.3 / k
    configs.append(p5)
    
    # Config 6: Another random
    rng2 = np.random.RandomState(123)
    p6 = np.zeros(M + 1)
    p6[0] = 1.0
    for k in range(1, M + 1):
        p6[k] = rng2.randn() * 0.5 / (k + 1)
    configs.append(p6)
    
    # Config 7: Sine-like (mode 1 dominant)
    p7 = np.zeros(M + 1)
    p7[0] = 1.0
    p7[1] = 1.2
    configs.append(p7)
    
    # Config 8: Cosine-like with phase shift (mode 2 dominant)
    p8 = np.zeros(M + 1)
    p8[0] = 1.0
    p8[2] = 0.9
    configs.append(p8)
    
    for idx, p_init in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        h_init = params_to_h(p_init)
        
        # Check initial value
        v_init = _compute_value(h_init)
        if v_init < best_val:
            best_h = h_init.copy()
            best_val = v_init
        
        # Optimize in parameter space with soft-max surrogate
        try:
            def obj_in_params(p):
                h = params_to_h(p)
                return _smooth_objective(h, beta=15.0)
            
            result = minimize(obj_in_params, p_init, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                                      'maxfun': 5000})
            
            if time.time() > deadline - 15:
                break
            
            h_candidate = params_to_h(result.x)
            v_candidate = _compute_value(h_candidate)
            
            if v_candidate < best_val:
                best_h = h_candidate.copy()
                best_val = v_candidate
                
                # Polish with tighter optimization on the full h vector
                if time.time() < deadline - 20:
                    try:
                        def obj_full(h_vec):
                            return _smooth_objective(h_vec, beta=8.0)
                        
                        res2 = minimize(obj_full, best_h, method='L-BFGS-B',
                                       options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12,
                                                'maxfun': 3000})
                        h_polish = res2.x
                        v_polish = _compute_value(h_polish)
                        if v_polish < best_val:
                            best_h = h_polish.copy()
                            best_val = v_polish
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Final high-precision polish on the best solution found
    if time.time() < deadline - 10 and best_val < 2.0:
        try:
            # Use very tight soft-max to approximate exact max
            def obj_final(h_vec):
                return _smooth_objective(h_vec, beta=3.0)
            
            res_final = minimize(obj_final, best_h, method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14,
                                         'maxfun': 5000})
            h_final = res_final.x
            v_final = _compute_value(h_final)
            if v_final < best_val:
                best_h = h_final.copy()
                best_val = v_final
        except Exception:
            pass
    
    # One more attempt: direct optimization from best_h with exact-like surrogate
    if time.time() < deadline - 5:
        try:
            def obj_direct(h_vec):
                s = np.sum(h_vec)
                if abs(s) < 1e-10:
                    return 1e10
                conv = np.convolve(h_vec, h_vec)
                # Use a very sharp soft-max
                beta = 2.0
                m = np.max(conv)
                sm = m + beta * np.log(np.sum(np.exp((conv - m) / beta)))
                return abs(2.0 * N * sm / (s * s))
            
            res_d = minimize(obj_direct, best_h, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14,
                                     'maxfun': 4000})
            h_d = res_d.x
            v_d = _compute_value(h_d)
            if v_d < best_val:
                best_h = h_d.copy()
                best_val = v_d
        except Exception:
            pass
    
    return best_h
