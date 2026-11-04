"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
_DEADLINE = 150.0  # seconds total budget
_START_TIME = None


def _objective(x):
    """Compute the value to minimize: max(conv(h,h)) / sum(h)^2, scaled."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # Use soft-max for differentiability, but we can also just use np.max
    # For gradient-based optimization, soft-max is smoother
    # Temperature for soft-max
    temp = 0.01
    m = np.max(conv)
    # Soft-max approximation
    smax = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
    val = smax / (s * s)
    return val


def _objective_hard(x):
    """Hard objective for evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    return abs(2 * N * m / (s * s))


def _soft_max_obj(x, temp=0.01):
    """Differentiable soft-max objective."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    smax = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
    return smax / (s * s)


def _soft_max_grad(x, temp=0.01):
    """Gradient of soft-max objective."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    exp_terms = np.exp((conv - m) / temp)
    softmax_weights = exp_terms / np.sum(exp_terms)
    
    # d(smax)/d(conv_i) = softmax_weights[i]
    # d(conv)/d(h) involves the autoconvolution gradient
    
    # conv[k] = sum_j h[j] * h[k-j] for valid k
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] (with boundary handling)
    
    # For full convolution, conv has length 2N-1
    # conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j] * h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] if 0<=k-i<N else 0, plus h[i-k] if 0<=i-k<N else 0
    
    # More carefully: conv[k] = sum_j h[j]*h[k-j] where j ranges so both indices valid
    # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[i-k] (if 0<=i-k<N)
    # But when i = k-i, i.e., 2i=k, we get 2*h[i], so it's h[k-i] + h[i-k] which is 2*h[i] when k=2i
    
    # Let's compute the gradient numerically or use the structure
    # Actually, let's just compute it properly
    
    # d(smax)/d(h[i]) = sum_k softmax_weights[k] * d(conv[k])/d(h[i])
    
    # d(conv[k])/d(h[i]) = h[k-i] if 0 <= k-i < N, else 0
    #                    + h[i-k] if 0 <= i-k < N, else 0
    
    # This is equivalent to: for each i, d(conv)/d(h[i]) is a vector where
    # entry k is h[k-i] (shifted) + h[i-k] (shifted)
    
    # Let's compute it as: grad_conv_i = shift(h, i) + shift(h, -i) with proper boundaries
    # Actually, conv = full autoconvolution, so:
    # conv[k] = sum_{j} h[j] h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] when both indices valid
    
    # The vector d(conv)/d(h[i]) has entries at positions k=i+j for j in [0,N-1] giving h[j],
    # and positions k=i-j for j in [0,N-1] giving h[j].
    # So it's h shifted by i (forward) plus h shifted by -i (backward), but only where valid.
    
    # Let me just compute it numerically for simplicity given N=400
    # Actually, let's do it analytically:
    
    # For fixed i, d(conv[k])/d(h[i]) = h[k-i] if 0<=k-i<N else 0, plus h[i-k] if 0<=i-k<N else 0
    # Note: when k-i == i-k, i.e., k=2i, we have h[i] counted twice, which is correct.
    
    # So the gradient contribution from position i is:
    # g[i] = sum_k softmax_weights[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
    
    # This can be written as:
    # g[i] = sum_k softmax_weights[k] * h[k-i] * 1_{0<=k-i<N} + sum_k softmax_weights[k] * h[i-k] * 1_{0<=i-k<N}
    
    # First term: for fixed i, sum over k such that 0<=k-i<N, i.e., i<=k<i+N
    # Let j = k-i, then j in [0,N-1], k = i+j
    # sum_{j=0}^{N-1} softmax_weights[i+j] * h[j] (where i+j < 2N-1)
    
    # Second term: for fixed i, sum over k such that 0<=i-k<N, i.e., i-N+1<=k<=i
    # Let j = i-k, then j in [0,N-1], k = i-j
    # sum_{j=0}^{N-1} softmax_weights[i-j] * h[j] (where i-j >= 0)
    
    # So g[i] = sum_{j=0}^{N-1} softmax_weights[i+j] * h[j] + sum_{j=0}^{N-1} softmax_weights[i-j] * h[j]
    # with appropriate bounds checking on the softmax_weights indices.
    
    # This is a convolution-like operation. Let's compute it efficiently.
    
    # Create a padded softmax_weights vector
    sw = softmax_weights  # length 2N-1
    padded_sw = np.zeros(2 * N - 1 + N)  # extra padding
    
    # Actually, let's just do it with loops for correctness, N=400 is small enough
    # But that's O(N^2) which is 160000, fine for gradient computation.
    
    # Better: use the fact that this is a convolution
    # g[i] = (sw * h)[i] where * is a specific convolution
    
    # Let me just compute it directly:
    grad = np.zeros(N)
    for i in range(N):
        # First sum: k from i to min(i+N-1, 2N-2)
        k_start = i
        k_end = min(i + N, 2 * N - 1)  # k < 2N-1
        if k_start < 2 * N - 1:
            j_range = np.arange(0, k_end - k_start)
            k_vals = k_start + j_range
            valid = k_vals < (2 * N - 1)
            if np.any(valid):
                grad[i] += np.sum(sw[k_vals[valid]] * h[j_range[valid]])
        
        # Second sum: k from max(0, i-N+1) to i
        k_start2 = max(0, i - N + 1)
        k_end2 = i + 1
        j_range2 = i - np.arange(k_start2, k_end2)
        valid2 = (j_range2 >= 0) & (j_range2 < N)
        if np.any(valid2):
            grad[i] += np.sum(sw[k_start2 + np.arange(k_end2 - k_start2)[valid2]] * h[j_range2[valid2]])
    
    # Now d(smax)/d(h[i]) = grad[i] / s^2 - 2*smax * d(s)/d(h[i]) / s^2
    # d(s)/d(h[i]) = 1
    # d(smax/s^2)/d(h[i]) = (grad[i] * s^2 - smax * 2*s) / s^4 = grad[i]/s^2 - 2*smax/(s^3)
    
    smax = m + temp * np.log(np.sum(exp_terms))
    d_obj = grad / (s * s) - 2 * smax / (s * s * s)
    
    return d_obj


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0  # seed value
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0  # 5 second safety margin
    
    # Strategy: try multiple initializations and optimize
    # Known good structure: non-symmetric sequences
    # Try various initializations
    
    initializations = []
    
    # 1. Random normal
    rng = np.random.RandomState(42)
    for seed in range(5):
        r = np.random.RandomState(seed)
        initializations.append(r.randn(N) * 0.1)
    
    # 2. Structured: positive on one side, negative on other
    x = np.linspace(-1, 1, N)
    initializations.append(x)
    initializations.append(-x)
    
    # 3. Step function
    h = np.zeros(N)
    h[:N//2] = 1.0
    h[N//2:] = -0.5
    initializations.append(h)
    
    # 4. Cosine wave
    t = np.linspace(0, 2*np.pi, N)
    initializations.append(np.cos(t))
    initializations.append(np.sin(t))
    
    # 5. Triangular
    h = np.abs(np.linspace(-1, 1, N))
    initializations.append(h - np.mean(h))
    
    for init_h in initializations:
        if time.time() > deadline:
            break
        
        h = init_h.copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        
        # Optimize using L-BFGS-B
        try:
            res = minimize(
                _soft_max_obj,
                h,
                jac=_soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            if res.success or res.fun < _soft_max_obj(h):
                h_opt = res.x
                val = _objective_hard(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try a few more iterations from the best found
    if time.time() < deadline:
        try:
            res = minimize(
                _soft_max_obj,
                best_h,
                jac=_soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if res.fun < _soft_max_obj(best_h):
                h_opt = res.x
                val = _objective_hard(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h


# Keep the original function name
def construct():
    """Construct 400 step heights to minimize the autocorrelation inequality value."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Random
    for i in range(8):
        r = np.random.RandomState(100 + i)
        inits.append(r.randn(N) * 0.5)
    
    # Linear ramps
    inits.append(np.linspace(-1, 1, N))
    inits.append(np.linspace(1, -1, N))
    
    # Step functions
    for frac in [0.3, 0.5, 0.7]:
        h = np.zeros(N)
        h[:int(frac*N)] = 1.0
        h[int(frac*N):] = -0.5
        inits.append(h - np.mean(h))
    
    # Waves
    t = np.linspace(0, 4*np.pi, N)
    inits.append(np.cos(t))
    inits.append(np.sin(t))
    inits.append(np.cos(2*t))
    
    # Parabolic
    x = np.linspace(-1, 1, N)
    inits.append(x**2 - 1/3)
    inits.append(x**3)
    
    for init_h in inits:
        if time.time() > deadline:
            break
        
        h = init_h.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        
        try:
            res = minimize(
                _soft_max_obj,
                h,
                jac=_soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            current_val = _objective_hard(res.x)
            if current_val < best_val:
                best_val = current_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _soft_max_obj,
                best_h,
                jac=_soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current_val = _objective_hard(res.x)
            if current_val < best_val:
                best_val = current_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h


N = 400
_START_TIME = None
_DEADLINE = 150.0

# Re-define to avoid duplication issues - actually let me restructure

import numpy as np
from scipy.optimize import minimize
import time

N = 400
_START_TIME = None
_DEADLINE = 150.0


def _soft_max_obj(x, temp=0.01):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    smax = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
    return smax / (s * s)


def _soft_max_grad(x, temp=0.01):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    exp_terms = np.exp((conv - m) / temp)
    softmax_weights = exp_terms / np.sum(exp_terms)
    sw = softmax_weights
    
    grad = np.zeros(N)
    for i in range(N):
        # First sum: k from i to min(i+N-1, 2N-2)
        k_start = i
        k_end = min(i + N, 2 * N - 1)
        if k_start < 2 * N - 1:
            j_range = np.arange(0, k_end - k_start)
            k_vals = k_start + j_range
            valid = k_vals < (2 * N - 1)
            if np.any(valid):
                grad[i] += np.sum(sw[k_vals[valid]] * h[j_range[valid]])
        
        # Second sum: k from max(0, i-N+1) to i
        k_start2 = max(0, i - N + 1)
        k_end2 = i + 1
        j_range2 = i - np.arange(k_start2, k_end2)
        valid2 = (j_range2 >= 0) & (j_range2 < N)
        if np.any(valid2):
            grad[i] += np.sum(sw[k_start2 + np.arange(k_end2 - k_start2)[valid2]] * h[j_range2[valid2]])
    
    smax = m + temp * np.log(np.sum(exp_terms))
    d_obj = grad / (s * s) - 2 * smax / (s * s * s)
    
    return d_obj


def _objective_hard(x):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    return abs(2 * N * m / (s * s))


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0
    
    inits = []
    
    for i in range(10):
        r = np.random.RandomState(100 + i)
        inits.append(r.randn(N) * 0.5)
    
    inits.append(np.linspace(-1, 1, N))
    inits.append(np.linspace(1, -1, N))
    
    for frac in [0.3, 0.5, 0.7]:
        h = np.zeros(N)
        h[:int(frac*N)] = 1.0
        h[int(frac*N):] = -0.5
        inits.append(h - np.mean(h))
    
    t = np.linspace(0, 4*np.pi, N)
    inits.append(np.cos(t))
    inits.append(np.sin(t))
    
    x = np.linspace(-1, 1, N)
    inits.append(x**2 - 1/3)
    inits.append(x**3)
    
    for init_h in inits:
        if time.time() > deadline:
            break
        
        h = init_h.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        
        try:
            res = minimize(
                _soft_max_obj,
                h,
                jac=_soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            current_val = _objective_hard(res.x)
            if current_val < best_val:
                best_val = current_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _soft_max_obj,
                best_h,
                jac=_soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current_val = _objective_hard(res.x)
            if current_val < best_val:
                best_val = current_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h
