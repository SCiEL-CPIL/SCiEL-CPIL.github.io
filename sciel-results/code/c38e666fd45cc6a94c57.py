import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_value(h):
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta=50.0):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp((x - x_max) * beta)) / beta) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    def objective_and_grad(h):
        """Compute soft objective and gradient using FFT."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(h)
        
        # FFT-based convolution
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
        
        beta = 50.0
        conv_max = np.max(conv)
        exp_conv = np.exp(beta * (conv - conv_max))
        soft_max_val = conv_max + np.log(np.sum(exp_conv) / beta) / beta
        
        val = abs(2.0 * N * soft_max_val / (s * s))
        
        # Gradient w.r.t. h
        # d(soft_max)/d(conv[j]) = beta * exp_conv[j] / sum(exp_conv)
        d_conv = beta * exp_conv / np.sum(exp_conv)
        
        # d(conv)/d(h[i]) = h convolved with delta, i.e., conv_shifted
        # conv[j] = sum_k h[k] * h[j-k]
        # d(conv[j])/d(h[i]) = h[j-i] + h[i-j] (for valid indices)
        # This is 2 * correlate(h, h) but with convolution indexing...
        # Actually d(conv[j])/d(h[i]) = sum over valid: h[j-i] + h[i-j]
        # = 2*h[j-i] if we consider full correlation
        
        # Simpler: use the fact that d(conv)/d(h) = 2 * np.correlate(h, h, 'full')
        # but indexed properly. Let's use: the gradient of conv w.r.t. h is
        # the convolution of h with h, but we need d(conv[j])/d(h[i]) for all j,i.
        # 
        # Alternative: use the identity that grad of sum_j d_conv[j]*conv[j] w.r.t. h[i]
        # = sum_j d_conv[j] * (h[j-i] + h[i-j]) for valid j,i
        # = 2 * sum_j d_conv[j] * h[j-i] (by symmetry, approximately)
        
        # Let's compute it via FFT: the gradient is 2 * ifft(fft(d_conv) * conj(fft(h)))
        D = np.fft.rfft(d_conv, n=n_fft)
        H_conj = np.conj(H)
        grad_conv = 2.0 * np.fft.irfft(D * H_conj, n=n_fft)[:N]
        
        # Now val = 2*N*soft_max/s^2 (assuming soft_max > 0)
        # d(val)/d(h[i]) = 2*N * (d(soft_max)/d(h[i]) * s^2 - soft_max * 2*s*dh[i]/ds) / s^4
        # = 2*N * (grad_conv[i]/s^2 - 2*soft_max*grad_conv[i]/s^3 * s + ...) 
        # Let me be more careful:
        # val = 2*N*sm/s^2 where sm = soft_max
        # d(val)/d(h[i]) = 2*N * (dsm/dh[i] * s^2 - sm * 2*s * 1) / s^4
        # = 2*N * (dsm/dh[i] * s - 2*sm) / s^3
        # where dsm/dh[i] = grad_conv[i]
        
        dsm = grad_conv
        grad_val = 2.0 * N * (dsm * s - 2.0 * soft_max) / (s ** 3)
        
        # Handle the abs: if 2*N*sm/s^2 < 0, gradient flips. But sm >= 0 typically.
        if 2.0 * N * soft_max_val / (s * s) < 0:
            grad_val = -grad_val
            val = -val
        
        return val, grad_val
    
    # Try multiple initializations
    rng = np.random.default_rng(123)
    inits = [
        np.ones(N),
        np.concatenate([np.ones(N//2), -0.5*np.ones(N//2)]),
        rng.standard_normal(N),
        np.sin(np.linspace(0, 6*np.pi, N)),
        np.exp(-np.linspace(0, 3, N)) - 0.3,
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        h0 = init.copy()
        s = np.sum(h0)
        if abs(s) < 1e-10:
            h0[N//2] += 0.1
            s = np.sum(h0)
        
        try:
            res = minimize(
                lambda h: objective_and_grad(h)[0],
                h0,
                jac=lambda h: objective_and_grad(h)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 2000}
            )
            val = _compute_value(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Polish with coordinate descent
    if time.time() < deadline - 10.0:
        h = best_h.copy()
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(0, N, 4):  # every 4th index for speed
                old = h[i]
                best_hi = old
                best_v = _compute_value(h)
                for step in [0.01, -0.01, 0.1, -0.1]:
                    h[i] = old + step
                    v = _compute_value(h)
                    if v < best_v:
                        best_v = v
                        best_hi = h[i]
                h[i] = best_hi
                if h[i] != old:
                    improved = True
            if not improved:
                break
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h
    
    return best_h
