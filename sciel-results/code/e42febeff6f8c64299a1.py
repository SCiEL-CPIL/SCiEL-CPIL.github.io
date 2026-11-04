import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 0.5 / N  # interval width

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function h on [-1/4, 1/4]."""
    # Continuous autoconvolution: (h*h)(t) = integral h(s) h(t-s) ds
    # For step functions, this is piecewise linear. Sample at grid points.
    # The autoconvolution of N step functions has 2N-1 "cells" in the natural grid.
    # Let's compute it on a fine grid to find the sup.
    # Actually, for step functions the autoconvolution is a piecewise linear function
    # with breakpoints at multiples of DELTA. The max occurs at a breakpoint.
    
    # Discrete autoconvolution (full): c[k] = sum_i h[i] * h[k-i] * DELTA^2
    # But we need the continuous sup norm.
    
    # For step functions h(s) = h_i for s in [x_i, x_{i+1}),
    # (h*h)(t) = sum_{i,j} h_i h_j * length of {s: s in [x_i,x_{i+1}) and t-s in [x_j,x_{j+1})}
    # = sum_{i,j} h_i h_j * max(0, min(x_{i+1}, t + x_{j+1}) - max(x_i, t + x_j))
    
    # This is piecewise linear in t, with breakpoints at x_i - x_j for all i,j.
    # The breakpoints are at multiples of DELTA from -(N-1)*DELTA to (N-1)*DELTA.
    # So evaluate at t = k*DELTA for k = -(N-1), ..., N-1.
    
    # At t = k*DELTA, (h*h)(t) = DELTA^2 * sum_{i,j: overlap} h_i h_j
    # Actually let's just compute the discrete convolution and scale appropriately.
    
    # The autoconvolution at t = m*DELTA (m integer) equals:
    # sum_i h_i * h_{i+m} * DELTA  (when indices are valid), but this isn't quite right.
    
    # Let me think more carefully. h(s) = sum_i h_i * 1_{[x_i, x_{i+1})}(s)
    # (h*h)(t) = integral h(s) h(t-s) ds
    # = sum_i sum_j h_i h_j * integral 1_{[x_i,x_{i+1})}(s) 1_{[x_j,x_{j+1})}(t-s) ds
    # = sum_i sum_j h_i h_j * length([x_i,x_{i+1}) ∩ [t-x_{j+1}, t-x_j])
    
    # For t = m*DELTA, the intersection length is DELTA if i+j+m is in a valid range...
    
    # Actually, let's just use the fact that for step functions on a uniform grid,
    # the autoconvolution evaluated at the grid points t_k = k*DELTA is:
    # (h*h)(t_k) = DELTA * sum_i h_i * h_{i-k}  (with h_j = 0 outside [0,N-1])
    # Wait, that's not right either.
    
    # Let me just compute it numerically on a fine grid.
    x = np.linspace(-0.25, 0.25, N, endpoint=False)  # left edges
    # Actually the intervals are [x_i, x_i + DELTA) for i=0..N-1
    # x_i = -0.25 + i*DELTA
    
    # For efficiency, compute the autoconvolution at the breakpoints.
    # The breakpoints are at t = (i - j) * DELTA for i,j in 0..N-1
    # So t ranges from -(N-1)*DELTA to (N-1)*DELTA in steps of DELTA.
    
    # At t = m*DELTA, (h*h)(t) = DELTA^2 * sum_{i: 0<=i<N, 0<=i-m<N} h_i * h_{i-m}
    # Wait, I need to be more careful.
    
    # Let's just use np.convolve for the discrete part and scale.
    # The continuous autoconvolution at t = m*DELTA:
    # (h*h)(m*DELTA) = sum_i h_i * (integral of h(m*DELTA - s) over s in [x_i, x_i+DELTA))
    # = sum_i h_i * DELTA * h_{i-m}  (if the interval [x_i, x_i+DELTA) maps to [x_{i-m}, x_{i-m}+DELTA) under s -> m*DELTA - s)
    # This is only exact if m*DELTA - [x_i, x_i+DELTA) = [x_i - m*DELTA - DELTA, x_i - m*DELTA) = [x_{i-m-1}, x_{i-m})
    # So (h*h)(m*DELTA) = DELTA * sum_i h_i * h_{i-m-1} ... this is getting confusing.
    
    # Let me just use a fine numerical integration.
    t_fine = np.linspace(-(N-1)*DELTA, (N-1)*DELTA, 2001)
    # h(s) for s in [-0.25, 0.25]
    s_fine = np.linspace(-0.25, 0.25, 2001)
    h_s = np.interp(s_fine, np.concatenate([[-0.25], x, [0.25]]), np.concatenate([[0], h, [0]]), left=0, right=0)
    
    # Actually this is getting complicated. Let me use a simpler approach:
    # Compute the autoconvolution using the discrete convolution formula.
    # For step functions, (h*h)(t) at t = k*DELTA equals:
    # DELTA * sum_i h_i * h_{i-k}  for k = -(N-1), ..., N-1
    # where h_j = 0 if j < 0 or j >= N.
    
    # Let me verify: for uniform h_i = 1, (h*h)(0) should be 0.5 (the area of the rect).
    # DELTA * sum_i h_i * h_i = DELTA * N = 0.01 * 50 = 0.5. Yes!
    
    # So (h*h)(k*DELTA) = DELTA * sum_i h_i * h_{i-k}
    # The sup norm is the max over k of this.
    
    conv = np.convolve(h, h, mode='full')  # conv[k] = sum_i h[i] * h[k-i]
    # conv has length 2N-1, conv[k] corresponds to t = (k - (N-1))*DELTA
    sup_val = np.max(conv) * DELTA
    return sup_val

def _objective(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    l2_sq = np.sum(h**2) * DELTA
    l1 = np.sum(h) * DELTA
    if l1 < 1e-15:
        return -1e15
    sup_ac = _autoconv_sup(h)
    if sup_ac < 1e-15:
        return -1e15
    return l2_sq / (l1 * sup_ac)

def _neg_objective(h):
    return -_objective(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = _objective(best_h)
    
    # Generate diverse initial conditions
    rng = np.random.default_rng(42)
    
    initializations = []
    # Uniform
    initializations.append(np.ones(N))
    # Gaussian-like profiles
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        x = np.linspace(-0.25, 0.25, N)
        profile = np.exp(-x**2 / (2*sigma**2))
        profile /= np.max(profile)
        initializations.append(profile)
    # Cosine profiles
    for m in [1, 2, 3, 4]:
        x = np.linspace(-0.25, 0.25, N)
        profile = 1.0 + 0.5 * np.cos(2*np.pi*m*x)
        profile = np.maximum(profile, 0.1)
        profile /= np.max(profile)
        initializations.append(profile)
    # Random smooth profiles
    for _ in range(20):
        # Generate via random Fourier coefficients
        coeffs = rng.standard_normal(10) + 1j * rng.standard_normal(10)
        x = np.linspace(-0.25, 0.25, N)
        profile = np.ones(N)
        for k, c in enumerate(coeffs):
            profile += np.abs(c) * np.cos(2*np.pi*(k+1)*x + np.angle(c))
        profile = np.maximum(profile, 0.01)
        profile /= np.max(profile)
        initializations.append(profile)
    # Two-peak profiles
    for sep in [0.1, 0.15, 0.2]:
        x = np.linspace(-0.25, 0.25, N)
        profile = np.exp(-(x-sep/2)**2/0.005) + np.exp(-(x+sep/2)**2/0.005)
        profile /= np.max(profile)
        initializations.append(profile)
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        try:
            res = minimize(_neg_objective, h0, method='L-BFGS-B',
                          bounds=[(0, None)]*N,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = res.x
            val = _objective(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Additional local search around best
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline:
                break
            h_pert = best_h + rng.standard_normal(N) * 0.1 * np.max(best_h)
            h_pert = np.maximum(h_pert, 0)
            try:
                res = minimize(_neg_objective, h_pert, method='L-BFGS-B',
                              bounds=[(0, None)]*N,
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                h_opt = res.x
                val = _objective(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    return [float(x) for x in best_h]
