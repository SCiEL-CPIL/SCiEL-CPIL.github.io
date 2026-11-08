import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 0.5 / N  # interval width
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

def _autoconv_max(heights):
    """Compute the max of normalized autoconvolution."""
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    return np.max(conv)

def _autoconv_max_smooth(heights, beta=100.0):
    """Compute smooth max of normalized autoconvolution."""
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    mx = np.max(conv)
    if mx < 1e-12:
        return 0.0
    shifted = conv - mx
    return np.log(np.sum(np.exp(beta * shifted))) / beta + mx

def _objective(x):
    return _autoconv_max_smooth(x)

def _objective_grad(x, beta=100.0):
    """Compute objective value and gradient."""
    h = np.maximum(x, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10, np.zeros(N)
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    F = np.fft.fft(h_pad)
    conv = np.fft.ifft(F ** 2).real[:2 * N - 1] * dx

    mx = np.max(conv)
    if mx < 1e-12:
        return 0.0, np.zeros(N)
    shifted = conv - mx
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp
    val = np.log(sum_exp) / beta + mx

    # Gradient w.r.t. h_norm
    w_pad = np.zeros(n_fft)
    w_pad[:2 * N - 1] = w

    # term2[m] = sum_k w[k]*h_norm[m-k] = (w * h_norm)[m]
    term2 = np.fft.ifft(np.fft.fft(w_pad) * F).real[:N]

    # term1[m] = sum_k w[k]*h_norm[k-m] = sum_j h_norm[j]*w[j+m]
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:2 * N - 1] = w_rev
    hw_rev_conv = np.fft.ifft(np.fft.fft(w_rev_pad) * F).real
    term1 = np.zeros(N)
    for m in range(N):
        p = 2 * N - 2 - m
        if 0 <= p < len(hw_rev_conv):
            term1[m] = hw_rev_conv[p]

    grad_hnorm = dx * (term1 + term2)

    # Chain rule: h_norm = h / S
    S = integral
    chain = (1.0 - h_norm * dx) / S
    grad_h = grad_hnorm * chain
    grad_h[h < 0] = 0.0

    return val, grad_h

def construct():
    global START_TIME
    best_x = np.ones(N)
    best_val = _autoconv_max(best_x)

    # Phase 1: L-BFGS-B with gradient, annealing beta
    betas = [20, 50, 100, 200]
    current_x = best_x.copy()
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        def obj_b(x):
            return _autoconv_max_smooth(x, beta=beta)
        def grad_b(x):
            return _objective_grad(x, beta=beta)[1]
        try:
            res = minimize(obj_b, current_x, method='L-BFGS-B', jac=grad_b,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100})
            current_x = np.maximum(res.x, 0.0)
            v = _autoconv_max(current_x)
            if v < best_val:
                best_x = current_x.copy()
                best_val = v
        except Exception:
            pass

    # Phase 2: Try different initializations with gradient
    inits = [
        np.ones(N),
        np.exp(-4.0 * (np.linspace(-1, 1, N)) ** 2),
        np.cos(np.pi * np.linspace(-1, 1, N)) ** 2,
    ]
    for x0 in inits:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        def grad_def(x):
            return _objective_grad(x)[1]
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=grad_def,
                           options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            x_new = np.maximum(res.x, 0.0)
            v = _autoconv_max(x_new)
            if v < best_val:
                best_x = x_new
                best_val = v
        except Exception:
            pass

    # Phase 3: High-precision polish
    if time.time() - START_TIME < TIME_LIMIT - 5:
        def grad_def2(x):
            return _objective_grad(x)[1]
        try:
            res = minimize(_objective, best_x.copy(), method='L-BFGS-B', jac=grad_def2,
                           options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 200})
            x_new = np.maximum(res.x, 0.0)
            v = _autoconv_max(x_new)
            if v < best_val:
                best_x = x_new
                best_val = v
        except Exception:
            pass

    return best_x.tolist()
