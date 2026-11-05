import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fftpack import dct, idct

N = 400

def _autoconv(h):
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c

def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_conv(h, beta):
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z
    
    # Gradient calculation
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]
    
    B = np.correlate(h, w, mode='full')
    term1 = B[n-1:2*n-1]
    
    grad = term1 + term2
    return sm, grad

def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g

def _dct_to_h(p):
    m = len(p)
    coeffs = np.zeros(N)
    coeffs[:m] = p
    h = idct(coeffs, type=2, norm='ortho')
    return h

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    
    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _hard(best_h)
    
    # Use a subset of DCT coefficients for optimization to reduce dimensionality
    # and encourage smooth/structured solutions
    m_dct = 50
    
    # Phase 1: Differential Evolution on DCT coefficients
    # DE is good for global search but slow. We run it for a short time.
    bounds = [(-1, 1)] * m_dct
    
    def de_obj(p):
        h = _dct_to_h(p)
        # Normalize to keep sum stable? No, objective is scale invariant.
        # But DE might produce huge/small values.
        # Let's normalize h to sum=1 inside the objective for DE stability?
        # No, the objective is scale invariant, so it's fine.
        # However, if sum is 0, it's invalid.
        s = h.sum()
        if abs(s) < 1e-6:
            return 1e12
        return _hard(h)

    if time_left() > 30:
        try:
            res = differential_evolution(de_obj, bounds, maxiter=50, popsize=15, 
                                         seed=42, tol=1e-8, polish=False, 
                                         time_limit=time_left() - 30)
            if res.success or res.fun < best_val:
                h_de = _dct_to_h(res.x)
                v = _hard(h_de)
                if v < best_val:
                    best_val = v
                    best_h = h_de.copy()
        except Exception:
            pass

    # Phase 2: Local Optimization with L-BFGS-B on full DCT vector
    # Start from best_h found so far, map to DCT
    h_curr = best_h.copy()
    # Normalize h_curr to have sum 1 to help convergence?
    # The objective is scale invariant, but the gradient might be sensitive.
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    
    # Project to DCT
    h_dct = dct(h_curr, type=2, norm='ortho')
    p0 = h_dct[:m_dct].copy()
    
    def param_obj(p):
        h = _dct_to_h(p)
        s = h.sum()
        if abs(s) < 1e-9:
            return 1e12
        return _hard(h)
    
    # Run L-BFGS-B with increasing beta for soft-max surrogate
    beta_seq = [10.0, 20.0, 50.0, 100.0]
    
    for beta in beta_seq:
        if time_left() < 15:
            break
        
        def obj(x):
            h = _dct_to_h(x)
            s = h.sum()
            if abs(s) < 1e-9:
                return 1e12
            sm, g_sm = _soft_max_conv(h, beta)
            val = 2.0 * N * sm / (s * s)
            return val
        
        def gr(x):
            h = _dct_to_h(x)
            s = h.sum()
            if abs(s) < 1e-9:
                return np.zeros_like(x)
            sm, g_sm = _soft_max_conv(h, beta)
            # Chain rule: grad_p = grad_h * (dh/dp)
            # dh/dp is the inverse DCT matrix columns
            # We can compute grad_h and then project back to DCT space
            # grad_h = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
            g_h = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
            # Project g_h to DCT space
            g_p = dct(g_h, type=2, norm='ortho')
            return g_p[:m_dct]
        
        try:
            res = minimize(obj, p0, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            p_new = res.x
            h_new = _dct_to_h(p_new)
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                p0 = p_new.copy()
        except Exception:
            pass

    # Phase 3: Full vector polish
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
        
    for beta in [50.0, 100.0, 200.0]:
        if time_left() < 10:
            break
        def obj_full(x):
            val, _ = _obj_and_grad(x, beta)
            return val
        def gr_full(x):
            _, g = _obj_and_grad(x, beta)
            return g
            
        try:
            res = minimize(obj_full, h_curr, jac=gr_full, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # Phase 4: Coordinate refinement
    h_curr = best_h.copy()
    best_local = _hard(h_curr)
    for _ in range(50):
        if time_left() < 5:
            break
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        lag = kmax - (N - 1)
        sg = np.zeros(N)
        for i in range(N):
            a = lag - i
            b = i - lag
            if 0 <= a < N:
                sg[i] += h_curr[a]
            if 0 <= b < N:
                sg[i] += h_curr[b]
        
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        sg_centered = sg - sg.mean()
        
        for alpha in [1e-2, 5e-3, 1e-3, 5e-4]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break

    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)

    return list(best_h)
