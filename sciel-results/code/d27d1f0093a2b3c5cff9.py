import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) > 1e-12:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    i = np.arange(N)
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)  # 0.5

    # Inspiration 1: Simple Fourier mode often provides a very good starting point
    # h = 0.5 + 0.5 * cos(2*pi*i/N)
    h_fourier = 0.5 + 0.5 * np.cos(2 * np.pi * i / N)
    h_fourier = project(h_fourier)
    val_fourier = compute_value(h_fourier)
    if val_fourier < best_val:
        best_val = val_fourier
        best_h = h_fourier.copy()

    # Gradient-based refinement using soft-max smoothing of the max correlation
    # This avoids the non-differentiability of max()
    def objective_and_grad(x, beta):
        h = x + 0.5
        c = np.correlate(h, 1.0 - h, mode="full")
        c_max = np.max(c)
        c_shifted = c - c_max
        exp_c = np.exp(beta * c_shifted)
        sum_exp = np.sum(exp_c)
        obj = c_max + np.log(sum_exp) / beta
        weights = exp_c / sum_exp

        # Gradient computation
        # d/dh_k (sum_m w_m * c_m) where c_m = sum_n h[n] * (1-h[n-m])
        # c_m = sum_n h[n] - sum_n h[n]h[n-m]
        # Let A = sum_n h[n] = constant (sum is fixed at N/2)
        # So c_m = A - sum_n h[n]h[n-m]
        # We want grad_k of sum_m w_m c_m = - sum_m w_m grad_k (sum_n h[n]h[n-m])
        # sum_n h[n]h[n-m] involves h[k] when n=k (term h[k]h[k-m]) or n=k+m (term h[k+m]h[k])
        # So grad_k (sum_n h[n]h[n-m]) = h[k-m] + h[k+m] (with boundary handling)
        
        grad_h = np.zeros(N)
        for m in range(2 * N - 1):
            w = weights[m]
            if w < 1e-10:
                continue

            # Term 1: h[k-m]
            # Valid when 0 <= k-m < N => m <= k < N+m
            if m < N:
                # k ranges from m to N-1
                # h[k-m] ranges from h[0] to h[N-1-m]
                grad_h[m:] -= w * h[:N - m]
            # If m >= N, k-m >= 0 implies k >= m >= N, impossible since k < N
            
            # Term 2: h[k+m]
            # Valid when 0 <= k+m < N => -m <= k < N-m
            if m < N:
                # k ranges from 0 to N-1-m
                # h[k+m] ranges from h[m] to h[N-1]
                grad_h[:N - m] -= w * h[m:]
            # If m >= N, k+m < N implies k < N-m <= 0, impossible since k >= 0

        # Note: The objective is obj = c_max + log(sum exp) / beta
        # The gradient of log(sum exp) w.r.t h is sum w_m * grad c_m
        # So grad obj = grad (sum w_m c_m) = - (grad computed above)
        # Wait, my derivation above gave grad of sum_n h[n]h[n-m].
        # c_m = A - sum_n h[n]h[n-m]
        # grad c_m = - (h[k-m] + h[k+m])
        # So grad obj = sum w_m * (- (h[k-m] + h[k+m]))
        # My code above does grad_h -= w * h[...], which matches this.
        
        return obj, grad_h

    bounds = [(-0.5, 0.5)] * N

    # Multi-start optimization
    starts = [best_h - 0.5]
    
    # Add some diverse starts
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.randn(N)
        x_rand -= np.mean(x_rand)
        x_rand /= max(np.max(np.abs(x_rand)), 1e-10)
        x_rand *= 0.4
        starts.append(x_rand)

    betas = [20.0, 50.0, 100.0]

    for beta in betas:
        if remaining() < 10:
            break
            
        for x0 in starts:
            if remaining() < 5:
                break
            
            # Ensure start is valid
            x0 = np.clip(x0, -0.5, 0.5)
            h0 = x0 + 0.5
            h0 = project(h0)
            x0 = h0 - 0.5

            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    args=(beta,),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                x_opt = res.x
                h_opt = x_opt + 0.5
                h_opt = project(h_opt)
                val = compute_value(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    # Update starts for next beta to include the new best
                    starts[0] = best_h - 0.5

            except Exception:
                pass

    return best_h
