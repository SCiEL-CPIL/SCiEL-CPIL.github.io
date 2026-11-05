import numpy as np
from scipy.optimize import minimize
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 140

def fast_autoconv(h):
    """
    Compute linear auto-convolution of h using FFT.
    """
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    S = H * np.conj(H)
    conv = np.fft.irfft(S, n=n_fft)
    return conv[:2*N-1]

def soft_max(x, beta=10.0):
    """
    Soft maximum approximation.
    """
    # Scale to avoid overflow
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def objective(v):
    """
    v: deviation vector of length N.
    h = 1/N + v - mean(v)
    """
    h = np.full(N, 1.0 / N)
    h += v
    h -= np.mean(h)
    
    # Ensure h is finite
    if not np.all(np.isfinite(h)):
        return 1e10
        
    conv = fast_autoconv(h)
    max_conv = soft_max(conv, beta=50.0)
    
    # Value is 2 * N * max_conv / sum(h)^2
    # sum(h) is 1 by construction
    return 2.0 * N * max_conv

def construct():
    global START_TIME, TIME_LIMIT
    
    # Fallback: Uniform
    best_h = np.ones(N) / N
    best_val = 2.0
    
    # Initial guesses for v
    # 1. Zero (Uniform)
    # 2. Random low freq
    # 3. Random high freq
    
    rng = np.random.default_rng(42)
    
    inits = [np.zeros(N)]
    
    # Random smooth
    for i in range(3):
        freqs = np.fft.rfftfreq(N)
        # Low freq emphasis
        weights = 1.0 / (1.0 + freqs * 10)
        spec = rng.normal(0, 1, N//2 + 1) * weights
        spec[0] = 0 # Zero mean
        v_rand = np.fft.irfft(spec, n=N)
        inits.append(v_rand)
        
    # Random high freq (noise)
    v_noise = rng.normal(0, 1, N)
    v_noise -= np.mean(v_noise)
    inits.append(v_noise)

    # Evaluate inits
    scored_inits = []
    for v_init in inits:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        val = objective(v_init)
        scored_inits.append((val, v_init))
        if val < best_val:
            best_val = val
            h_tmp = np.full(N, 1.0 / N)
            h_tmp += v_init
            h_tmp -= np.mean(h_tmp)
            best_h = h_tmp

    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    for val, v_init in scored_inits[:3]:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        try:
            # Use L-BFGS-B
            result = minimize(
                objective,
                v_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                v_opt = result.x
                h_opt = np.full(N, 1.0 / N)
                h_opt += v_opt
                h_opt -= np.mean(h_opt)
                best_h = h_opt
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N) / N
        
    return best_h

if __name__ == "__main__":
    h = construct()
    s = np.sum(h)
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / s**2)
    print(f"Score: {val}")
