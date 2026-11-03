import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    h = 1.0 / N
    
    # Initialize with a smooth symmetric profile
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    a = np.cos(np.pi * x / 0.5)
    a = np.maximum(a, 0)
    
    def compute_autoconv_max(a, upsample=1):
        N_up = N * upsample
        a_up = np.repeat(a, upsample)
        size = 2 * N_up
        f = np.fft.rfft(a_up, n=size)
        ac = np.fft.irfft(f * f, n=size)[:N_up]
        ac_scaled = ac * h * upsample
        return np.max(ac_scaled)
    
    best_a = a.copy()
    best_val = compute_autoconv_max(a)
    
    try:
        from scipy.optimize import minimize
        def objective(a_flat):
            a = np.maximum(a_flat, 0)
            return compute_autoconv_max(a, upsample=2)
        res = minimize(objective, best_a, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.x is not None and res.fun < best_val:
            best_a = np.maximum(res.x, 0)
            best_val = res.fun
    except Exception:
        pass
    
    if time.time() < deadline - 10:
        lr = 0.05
        for it in range(5000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N // 2):
                j = N - 1 - i
                a_i, a_j = best_a[i], best_a[j]
                delta = lr * max(0.02, a_i)
                
                best_a[i], best_a[j] = a_i + delta, a_j + delta
                v_inc = compute_autoconv_max(best_a, upsample=1)
                
                best_a[i], best_a[j] = max(0.0, a_i - delta), max(0.0, a_j - delta)
                v_dec = compute_autoconv_max(best_a, upsample=1)
                
                if v_dec < best_val:
                    best_val = v_dec; improved = True; lr = max(0.005, lr * 0.95)
                elif v_inc < best_val:
                    best_a[i], best_a[j] = a_i + delta, a_j + delta
                    best_val = v_inc; improved = True; lr = max(0.005, lr * 0.95)
                else:
                    best_a[i], best_a[j] = a_i, a_j; lr = max(0.005, lr * 0.9)
            if not improved and lr <= 0.005:
                break
    
    return list(best_a)
