"""
New strategy: Direct optimization via scipy.optimize.minimize with smooth
epigraph formulation of the autoconvolution peak. The key insight is that
the autoconvolution at lag k is x * x[k:] shifted, and we can express the
peak constraint smoothly using log-sum-exp. We optimize the raw sequence
values directly, using L-BFGS-B which is efficient for smooth objectives.

This differs from the previous phase-only approach by working in the
time domain directly with a proper gradient-based optimizer, and by using
a different smooth approximation (log-sum-exp with adaptive temperature).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv(x):
    """Compute linear autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _metric(x):
    """Compute the C3' metric: max|autoconv| / (sum(x))^2."""
    conv = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(conv))) / (s * s)


def _soft_peak(conv, beta):
    """Smooth approximation of max|conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val < 1e-15:
        return 1e-15
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _objective(x, beta, s_target=1.0):
    """
    Objective: soft_peak(autoconv(x)) / (sum(x))^2
    We fix the sum to be s_target to avoid the scale invariance issue.
    Actually, the metric is scale-invariant: if we scale x by c, both
    max|conv| and (sum)^2 scale by c^2. So we can fix sum(x) = 1.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv(x)
    peak = _soft_peak(conv, beta)
    return peak / (s * s)


def _objective_with_sum_constraint(x, beta):
    """
    For optimization, we want to minimize max|conv|/(sum)^2.
    Since this is scale-invariant, we can fix sum(x)=1 and minimize max|conv|.
    But fixing sum(x)=1 with L-BFGS-B is tricky. Instead, we optimize
    the ratio directly and rely on the optimizer to find a good scale.
    
    Actually, a better approach: minimize soft_peak(autoconv(x)) subject
    to sum(x) = 1. We can use the fact that the problem is scale-invariant,
    so we can normalize after each step.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    x_norm = x / s
    conv = _autoconv(x_norm)
    peak = _soft_peak(conv, beta)
    return peak


def _gradient(x, beta):
    """
    Compute gradient of soft_peak(autoconv(x)/s^2) with respect to x.
    Since the metric is scale-invariant, we work with the normalized version.
    
    Let y = x / sum(x). Then metric = max|conv(y)|.
    The gradient of max|conv(y)| w.r.t. y, multiplied by d(y)/d(x).
    
    d(y)/d(x) is complicated. Instead, let's just use finite differences
    or compute the gradient of the soft objective directly.
    
    For the soft objective: f(x) = soft_peak(_autoconv(x)) / (sum(x))^2
    This is complicated. Let's use a simpler approach.
    """
    # Use numerical gradient as fallback, or implement analytically
    eps = 1e-6
    grad = np.zeros(N)
    f0 = _objective_with_sum_constraint(x, beta)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        fp = _objective_with_sum_constraint(x_plus, beta)
        fm = _objective_with_sum_constraint(x_minus, beta)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def _gradient_analytic(x, beta):
    """
    Analytical gradient of soft_peak(_autoconv(x_norm)) where x_norm = x/sum(x).
    
    Let s = sum(x), y = x/s.
    conv(y)[k] = sum_j y[j]*y[j+k]
    
    soft_peak = (1/beta) * log(sum_k exp(beta * |conv_k|)) + max|conv|
    
    d(soft_peak)/d(conv_k) = (1/sum_j exp(beta*|conv_j|)) * exp(beta*|conv_k|) * sign(conv_k)
    
    d(conv_k)/dy[j] = y[j+k] + y[j-k] (with appropriate boundary handling)
    
    dy[j]/dx[i] = delta(j,i)/s - y[j]/s^2
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.ones(N)
    y = x / s
    conv = _autoconv(y)
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val < 1e-15:
        max_val = 1e-15
    shifted = abs_conv - max_val
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft_peak w.r.t. conv
    dsoft_dconv = weights * np.sign(conv)
    
    # Now compute d(soft_peak)/dx[i] = sum_k dsoft_dconv[k] * d(conv_k)/dx[i]
    # d(conv_k)/dy[j] = y[j+k] + y[j-k] (for valid indices)
    # d(y[j])/dx[i] = delta(j,i)/s - y[j]/s^2
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # d(conv_k)/dy[j] is like a "cross-correlation" kernel.
    
    # Let's build the gradient step by step.
    # grad_y[j] = sum_k dsoft_dconv[k] * d(conv_k)/dy[j]
    #            = sum_k dsoft_dconv[k] * (y[j+k] + y[j-k])
    
    # This is: grad_y = convolve(dsoft_dconv, y) + convolve(dsoft_dconv[::-1], y)
    # where convolve here means cross-correlation.
    
    # Using FFT:
    dff = np.fft.rfft(dsoft_dconv, _FFT_SIZE)
    yff = np.fft.rfft(y, _FFT_SIZE)
    
    # conv_k = sum_j y[j]*y[j+k], so d(conv_k)/dy[j] = y[j+k]
    # sum_k dsoft_dconv[k] * y[j+k] = (dsoft_dconv * y)[j] in correlation sense
    # In FFT: if A[k] = dsoft_dconv[k], B[k] = y[k], then
    # C[j] = sum_k A[k] * B[j+k] = sum_k A[k] * B[j+k]
    # This is cross-correlation. In FFT: C = ifft(conj(fft(A)) * fft(B))
    
    # Actually, let me think again.
    # conv_y[k] = sum_j y[j] * y[j+k]
    # d(conv_y[k])/dy[j] = y[j+k] + y[j-k] (by symmetry, since conv is symmetric)
    
    # So grad_y[j] = sum_k dsoft_dconv[k] * (y[j+k] + y[j-k])
    #              = sum_k dsoft_dconv[k] * y[j+k] + sum_k dsoft_dconv[k] * y[j-k]
    
    # First term: sum_k A[k] * y[j+k] where A = dsoft_dconv
    # This is (A * y) at position j, where * denotes cross-correlation.
    # In terms of convolution: if we define A_rev[k] = A[-k], then
    # sum_k A[k] * y[j+k] = sum_k A_rev[-k] * y[j+k]
    # Let m = -k: = sum_m A_rev[m] * y[j-m] = (A_rev * y)[j]
    
    # In FFT: A_rev * y = ifft(fft(A_rev) * fft(y))
    # fft(A_rev)[f] = conj(fft(A)[f]) for real A
    
    # Similarly for second term: sum_k A[k] * y[j-k] = (A * y)[j] = ifft(fft(A) * fft(y))
    
    # So grad_y = ifft(conj(fft(A)) * fft(y)) + ifft(fft(A) * fft(y))
    #           = 2 * Re(ifft(fft(A) * fft(y)))
    
    cross = np.fft.irfft(dff * yff, _FFT_SIZE)
    grad_y = 2.0 * cross[:N]
    
    # Now d(y[j])/dx[i] = delta(j,i)/s - y[j]/s^2
    # grad_x[i] = sum_j grad_y[j] * d(y[j])/dx[i]
    #           = grad_y[i]/s - y[i]/s^2 * sum_j grad_y[j]
    
    sum_grad_y = np.sum(grad_y)
    grad_x = grad_y / s - y / (s * s) * sum_grad_y
    
    return grad_x


def _initialize_random(rng):
    """Generate a random initial sequence."""
    x = rng.normal(0, 1, size=N)
    # Ensure non-zero sum
    s = np.sum(x)
    if abs(s) < 1e-10:
        x[0] += 1.0
    return x


def _initialize_structured(rng, start_type):
    """Generate structured initial sequences."""
    if start_type == 0:
        # Random
        return _initialize_random(rng)
    elif start_type == 1:
        # Low-pass filtered random
        x = rng.normal(0, 1, size=N)
        fx = np.fft.rfft(x, _FFT_SIZE)
        cutoff = _FFT_SIZE // 8
        fx[cutoff:] = 0
        x = np.fft.irfft(fx, _FFT_SIZE)[:N]
        return x
    elif start_type == 2:
        # Chirp-like
        n = np.arange(N)
        x = np.cos(2 * np.pi * (0.5 * n * n / N + rng.random()))
        x += rng.normal(0, 0.1, size=N)
        return x
    elif start_type == 3:
        # Block pattern
        x = np.ones(N)
        block_size = rng.integers(10, 50)
        for i in range(0, N, block_size):
            if rng.random() < 0.5:
                x[i:i+block_size] *= -1
        return x
    else:
        # Mixed
        x = rng.normal(0, 1, size=N)
        x[:N//2] += np.cos(2 * np.pi * np.arange(N//2) / N)
        return x


def _optimize_from_start(x_init, deadline, beta_initial, rng):
    """Run L-BFGS-B optimization from a given start."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Adaptive beta: start low, increase
    beta = beta_initial
    current_x = x_init.copy()
    
    max_time = 20
    start_time = time.time()
    
    # Multiple passes with increasing beta
    n_passes = 3
    pass_time = min(max_time, (deadline - time.time()) / (n_passes + 1))
    
    for p in range(n_passes):
        if time.time() > deadline - 5:
            break
        
        beta = beta_initial * (1 + p * 2)
        timeout_this = time.time() + pass_time
        
        def obj(x):
            if time.time() > timeout_this:
                return 1e10
            return _objective_with_sum_constraint(x, beta)
        
        def grad(x):
            if time.time() > timeout_this:
                return np.zeros(N)
            return _gradient_analytic(x, beta)
        
        try:
            result = minimize(
                obj,
                current_x,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                         'maxfun': 500, 'disp': False}
            )
            if result.fun < 1e9:
                x_new = result.x
                # Normalize
                s = np.sum(x_new)
                if abs(s) > 1e-10:
                    x_new = x_new / s
                v = _metric(x_new)
                if v < best_val:
                    best_val = v
                    best_x = x_new.copy()
                current_x = x_new
        except Exception:
            pass
    
    return best_x, best_val


def _local_refine(x, deadline, rng):
    """Fast local refinement using random perturbations."""
    best_x = x.copy()
    best_val = _metric(best_x)
    step_size = 0.05
    no_improve = 0
    
    while time.time() < deadline - 2:
        if no_improve > 200:
            step_size *= 0.9
            no_improve = 0
            if step_size < 0.001:
                break
        
        k = rng.integers(1, 5)
        indices = rng.choice(N, size=k, replace=False)
        delta = rng.normal(0, step_size, size=N)
        delta[~np.isin(np.arange(N), indices)] = 0
        
        x_test = best_x + delta
        v = _metric(x_test)
        if v < best_val:
            best_val = v
            best_x = x_test
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(123)
    
    best_x = None
    best_val = float('inf')
    
    # Multi-start
    n_starts = 4
    time_budget = deadline - time.time() - 20  # Reserve 20s for local refine
    time_per_start = time_budget / n_starts
    
    for start in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        x_init = _initialize_structured(rng, start)
        # Normalize
        s = np.sum(x_init)
        if abs(s) > 1e-10:
            x_init = x_init / s
        
        x_opt, val_opt = _optimize_from_start(
            x_init,
            min(deadline - 25, time.time() + time_per_start),
            beta_initial=50.0,
            rng=rng
        )
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Local refinement
    if best_x is not None and time.time() < deadline - 5:
        best_x, best_val = _local_refine(best_x, deadline - 3, rng)
    
    if best_x is None:
        # Fallback
        best_x = np.ones(N)
    
    return best_x
