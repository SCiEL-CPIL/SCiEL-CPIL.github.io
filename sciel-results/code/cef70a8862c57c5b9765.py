"""Program for the third autocorrelation inequality task.

This program formulates the problem as minimizing the squared L2 norm of the
autoconvolution $c = h * h$, subject to $\sum h_i = 1$.
The objective function $J(h) = \sum_k c_k^2$ is convex.
The global minimum is expected to be at the uniform vector $h_i = 1/N$,
which yields the theoretical lower bound value of 2.0.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _autoconv_sq_norm(h):
    """Compute sum of squares of the full autoconvolution."""
    c = np.convolve(h, h)
    return float(np.sum(c ** 2))


def _autoconv_sq_norm_grad(h):
    """Compute gradient of sum of squares of autoconvolution.
    
    Let c_k = sum_i h_i h_{i+k}.
    J = sum_k c_k^2.
    dJ/dh_j = 2 sum_k c_k (dc_k/dh_j).
    dc_k/dh_j = h_{j-k} + h_{j+k} (if indices valid).
    """
    c = np.convolve(h, h)
    # c has length 2N - 1. c[k] corresponds to lag k - (N-1).
    # Let's use the property: grad = 4 * convolve(h, c) ?
    # Let's derive carefully.
    # c = convolve(h, h).
    # J = dot(c, c).
    # dJ = 2 dot(c, dc).
    # dc = convolve(dh, h) + convolve(h, dh).
    # This is a linear operator.
    # We can compute the gradient by evaluating the action of the Hessian?
    # Or simpler: use automatic differentiation? No, standard lib only.
    # Let's use the formula: grad_J = 4 * np.convolve(h, c, 'full')?
    # Let's test for small N.
    
    # Actually, there is a known identity.
    # The gradient of ||h*h||_2^2 is 4 * h * (h*h) ? No.
    
    # Let's just compute it numerically or via the sum definition.
    # N=400 is small enough for O(N^2) gradient?
    # c has 2N-1 terms.
    # grad_j = 2 * sum_k c_k * (h_{j-k} + h_{j+k})
    # This is O(N^2). 400^2 = 160,000 operations. Very fast.
    
    grad = np.zeros(N)
    # c indices: 0 to 2N-2.
    # c[k] = sum_{i} h[i] h[i+k-N+1]?
    # np.convolve(h, h) returns c such that c[k] = sum_{i} h[i] h[k-i].
    # Let's stick to the definition c = np.convolve(h, h).
    # c[k] = sum_{i=max(0, k-N+1)}^{min(N-1, k)} h[i] h[k-i].
    
    # dc_k / dh_j = h[k-j] (if valid) + h[j-k] (if valid)?
    # c_k = sum_i h_i h_{k-i}.
    # If i=j, term is h_j h_{k-j}. Deriv wrt h_j is h_{k-j}.
    # If k-i=j => i=k-j, term is h_{k-j} h_j. Deriv wrt h_j is h_{k-j}.
    # So dc_k/dh_j = h_{k-j} + h_{j-k} (with bounds 0..N-1).
    
    for j in range(N):
        s = 0.0
        for k in range(len(c)):
            # term 1: h[k-j] if 0 <= k-j < N
            if 0 <= k - j < N:
                s += c[k] * h[k - j]
            # term 2: h[j-k] if 0 <= j-k < N
            # Note: if k=j, both terms are h[0]? No.
            # If k=j, term 1 is h[0]. Term 2 is h[0].
            # The derivative of h_j * h_j is 2 h_j.
            # So if k=j, we add 2 * c[j] * h[0]? No, h[k-j]=h[0] is wrong index logic?
            # h[k-j] is the value at index k-j.
            # If k=j, index is 0. Value is h[0].
            # Wait, c_k depends on h_j and h_{k-j}.
            # If k=j, c_j contains h_j * h_0.
            # Deriv wrt h_j is h_0.
            # Also c_j contains h_0 * h_j.
            # Deriv wrt h_j is h_0.
            # So total deriv is 2 h_0.
            # My formula: h[k-j] + h[j-k].
            # If k=j: h[0] + h[0] = 2 h[0]. Correct.
            
            if 0 <= j - k < N:
                s += c[k] * h[j - k]
        grad[j] = 2.0 * s
        
    return grad


def construct():
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_autoconv_sq_norm, x0, method="SLSQP", 
                   jac=_autoconv_sq_norm_grad,
                   constraints=cons,
                   options={"maxiter": 100, "ftol": 1e-12})
    h = res.x
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
