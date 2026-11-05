"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def _smooth_max_surrogate(h, beta=100.0):
    """Soft-max surrogate for the autoconvolution max."""
    conv = np.convolve(h, h)
    # Use log-sum-exp for smooth max
    m = np.max(conv)
    if m > 0:
        logsumexp = m + np.log(np.sum(np.exp(conv - m)))
    else:
        logsumexp = m + np.log(np.sum(np.exp(conv - m)))
    # Derivative-friendly approximation
    weights = np.exp((conv - m) * beta)
    weights /= np.sum(weights)
    return np.dot(weights, conv)

def _objective(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    return abs(2 * N * max_val / (s * s))

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # Start from seed
    h0 = np.ones(N)
    
    # Try multiple initializations
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    inits = [h0.copy()]
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        # Random perturbation
        h_try = h0 + 0.1 * rng.randn(N)
        inits.append(h_try)
    
    # Also try structured initializations
    # Alternating
    h_alt = np.ones(N)
    h_alt[::2] = -0.5
    inits.append(h_alt)
    
    # Block structure
    h_block = np.ones(N)
    h_block[:N//4] = -1.0
    inits.append(h_block)
    
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        # Use BFGS with smooth surrogate
        def obj_smooth(x):
            s = np.sum(x)
            if abs(s) < 1e-12:
                return 1e10
            conv = np.convolve(x, x)
            # Smooth max via softmax
            beta = 50.0
            m = np.max(conv)
            exps = np.exp((conv - m) * beta)
            smax = m + np.log(np.sum(exps)) / beta
            val = 2 * N * smax / (s * s)
            if val < 0:
                val = -val
            return val
        
        def grad_smooth(x):
            s = np.sum(x)
            if abs(s) < 1e-12:
                return np.zeros(N)
            conv = np.convolve(x, x)
            beta = 50.0
            m = np.max(conv)
            exps = np.exp((conv - m) * beta)
            weights = exps / np.sum(exps)
            smax = np.dot(weights, conv)
            
            # d(smax)/dx: need gradient of softmax-weighted conv
            # conv[k] = sum_i x[i]*x[i+k-(N-1)] (with appropriate bounds)
            # This is complex; use numerical gradient as fallback
            eps = 1e-5
            grad = np.zeros(N)
            for i in range(N):
                xp = x.copy()
                xp[i] += eps
                grad[i] = (obj_smooth(xp) - obj_smooth(x)) / eps
            return grad
        
        try:
            res = minimize(obj_smooth, h_init, method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 5000,
                                   'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < best_val:
                h_opt = res.x
                val_opt = _compute_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        # Also try coordinate descent on the true objective
        if time.time() > deadline - 10:
            break
        h_cd = h_init.copy()
        for it in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h_cd[i]
                # Try small perturbations
                for delta in [-0.01, 0.01, -0.001, 0.001, -0.1, 0.1]:
                    h_cd[i] = old + delta
                    v = _compute_value(h_cd)
                    if v < _compute_value(h_cd + np.array([0]*i + [old - h_cd[i]] + [0]*(N-i-1))):
                        pass
                # Simpler: just evaluate and keep if better
                h_cd[i] = old
                best_delta = 0
                best_v = _compute_value(h_cd)
                for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    h_cd[i] = old + delta
                    v = _compute_value(h_cd)
                    if v < best_v:
                        best_v = v
                        best_delta = delta
                h_cd[i] = old + best_delta
                if best_delta != 0:
                    improved = True
            if not improved:
                break
        val_cd = _compute_value(h_cd)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Final check
    final_val = _compute_value(best_h)
    if final_val < 2.0:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
