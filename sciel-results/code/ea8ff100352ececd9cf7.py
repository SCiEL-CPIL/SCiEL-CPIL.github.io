import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def value(h):
        S = np.sum(h)
        if S == 0:
            return np.inf
        H = np.fft.fft(h)
        C = np.fft.ifft(H ** 2).real
        max_conv = np.max(C)
        return abs(2 * N * max_conv / (S ** 2))

    best_val = 2.0
    best_h = np.ones(N)
    rng = np.random.default_rng(42)
    
    t = np.arange(N)
    
    # Family 1: Cosine
    def obj_cos(params):
        C, A, k, phi = params
        h = C + A * np.cos(2 * np.pi * k * t / N + phi)
        return value(h)

    # Family 2: Ramp
    def obj_ramp(params):
        C, A = params
        h = C + A * (t / N - 0.5)
        return value(h)

    # Family 3: Alternating Ramp
    def obj_alt_ramp(params):
        C, A = params
        h = C + A * (t / N - 0.5) * ((-1) ** t)
        return value(h)

    families = [
        (obj_cos, [1.0, 0.5, 10.0, 0.0]),
        (obj_ramp, [1.0, 0.5]),
        (obj_alt_ramp, [1.0, 0.5])
    ]
    
    for obj, x0 in families:
        # Try multiple random starts for each family
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Perturb x0
            if len(x0) == 4:
                x_try = [x0[0] + rng.normal(0, 0.5), x0[1] + rng.normal(0, 0.5), x0[2] + rng.normal(0, 5.0), x0[3] + rng.normal(0, 1.0)]
            elif len(x0) == 2:
                x_try = [x0[0] + rng.normal(0, 0.5), x0[1] + rng.normal(0, 0.5)]
            
            res = minimize(obj, x_try, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
            
            if res.fun < best_val:
                best_val = res.fun
                # Reconstruct h
                if len(res.x) == 4:
                    C, A, k, phi = res.x
                    best_h = C + A * np.cos(2 * np.pi * k * t / N + phi)
                elif len(res.x) == 2:
                    C, A = res.x
                    if obj == obj_ramp:
                        best_h = C + A * (t / N - 0.5)
                    else:
                        best_h = C + A * (t / N - 0.5) * ((-1) ** t)

    return best_h
