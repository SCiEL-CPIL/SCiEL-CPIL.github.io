"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import time

N = 600
# Interval width
dx = (1.0/4 - (-1.0/4)) / N  # 0.5 / 600 = 1/1200

# Precompute the autoconvolution kernel structure.
# The autoconvolution of a step function with heights h_i on intervals [x_i, x_i+dx]
# is a piecewise linear function. We want to minimize the maximum of this autoconvolution
# normalized appropriately.
#
# Let f(x) = sum_i h_i * I(x in [x_i, x_i+dx])
# The autoconvolution (f * f)(t) = integral f(t-s) f(s) ds
# For a step function, this is a piecewise linear function (trapezoidal shape for each pair).
#
# Actually, for discrete steps, the autoconvolution at the grid points can be computed via
# discrete convolution. But the maximum might occur between grid points.
#
# Let's think about the continuous autoconvolution.
# f(x) is constant on each interval.
# (f*f)(t) = integral_{-1/4}^{1/4} f(t-s) f(s) ds
#
# Since f is a step function, f(t-s) changes when t-s crosses an interval boundary, i.e., s = t - x_j.
# So (f*f)(t) is a piecewise linear function of t. The breakpoints are at t = x_i + x_j for all i,j.
# There are O(N^2) breakpoints, which is 360,000 for N=600. This is manageable.
#
# However, computing the value at all breakpoints and finding the max is O(N^2) per evaluation.
# With N=600, that's 360k operations. In Python, this might be slow if done naively.
#
# Alternative: Use FFT to compute the discrete autoconvolution at the grid points.
# But the maximum of the continuous autoconvolution might be higher than at grid points.
#
# Let's estimate: the autoconvolution of a non-negative function is non-negative.
# The maximum likely occurs near the center t=0.
#
# For a more efficient approach, note that the autoconvolution of a step function can be
# computed as follows:
# Let the intervals be [a_i, a_{i+1}] with height h_i.
# (f*f)(t) = sum_i h_i * (integral over s where f(t-s) is non-zero)
#
# Actually, let's use the fact that f*f is the convolution of two identical step functions.
# We can compute it using the "trapezoid" rule or by direct summation.
#
# For efficiency, let's use numpy's correlate or convolve with mode='full' to get the discrete
# autoconvolution at the grid points. Then we can assume the max is close to the max of the
# discrete autoconvolution, or we can refine.
#
# But wait, the task says "normalized maximum of its autoconvolution". What is the normalization?
# Typically, for such problems, it's max(f*f) / (integral f)^2 or similar.
# Let's assume the score is max(f*f) / (sum(h_i) * dx)^2, which is the max value divided by the square of the L1 norm.
# Or perhaps it's just max(f*f) with the constraint that integral f = 1?
# The problem says "nonnegative step function" without specifying normalization.
# But "normalized maximum" suggests we divide by something.
#
# Let's look at the seed: [1.0]*N.
# integral f = N * 1.0 * dx = 600 * (1/1200) = 0.5.
# autoconvolution at t=0: integral f(s)^2 ds = N * 1.0^2 * dx = 0.5.
# If normalized by (integral f)^2 = 0.25, then score = 0.5/0.25 = 2.
# If normalized by integral f = 0.5, then score = 0.5/0.5 = 1.
#
# The task says "upper bound on C1". C1 is likely a constant in an inequality.
# Let's assume the score is max(f*f) / (integral f)^2.
#
# Let's implement a function to compute the score.
# For speed, we'll use FFT-based convolution for the discrete approximation, and then
# maybe refine by checking the midpoint between grid points if needed.
# But for now, let's use the discrete convolution at the grid points as a proxy.
#
# Actually, the continuous autoconvolution of a step function is piecewise linear.
# The maximum of a piecewise linear function on an interval occurs at an endpoint.
# The endpoints are the breakpoints t = x_i + x_j.
# So the maximum is at one of the O(N^2) breakpoints.
#
# To compute the value at a breakpoint t, we need to sum over all pairs (i,j) such that
# the interval of integration contributes.
#
# This is getting complex. Let's try a simpler approach:
# 1. Use discrete convolution via FFT to get an approximation.
# 2. Use coordinate descent or gradient-based optimization to minimize the max.
#
# But the max is non-smooth. We can use a smooth approximation like log-sum-exp.
#
# Let's define the objective as the log-sum-exp of the autoconvolution values at the grid points,
# scaled by a temperature parameter. This gives a smooth approximation to the max.
#
# Steps:
# 1. Compute the discrete autoconvolution using FFT.
# 2. Compute the log-sum-exp of the positive parts.
# 3. Minimize this using scipy.optimize.
#
# We need to be careful with the normalization.
# Let's define the score as max(f*f) / (sum(h) * dx)^2.
# In log-sum-exp form, this is exp(logsumexp(f*f) - 2*log(sum(h)*dx)).
#
# Let's implement this.

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=float)
    # Discrete autoconvolution via FFT
    n = len(h)
    # We need the full convolution, so we pad to 2*n - 1
    f = np.fft.rfft(h, n=2*n)
    conv = np.fft.irfft(f * f, n=2*n-1)
    # The autoconvolution is symmetric, so we can take the center part
    # The center is at index n-1
    # The max of the continuous autoconvolution is approximated by the max of the discrete one
    # But we should be careful: the discrete convolution sums h_i * h_j * dx^2?
    # Actually, np.convolve(h, h) gives the sum of products, not multiplied by dx.
    # The continuous autoconvolution is approximated by dx * sum_i h_i h_{i+k} for the k-th lag.
    # So the discrete convolution result needs to be multiplied by dx.
    conv *= dx
    # The max value
    max_val = np.max(conv)
    # Normalization: (integral f)^2 = (sum(h) * dx)^2
    integral_f = np.sum(h) * dx
    norm = integral_f ** 2
    if norm < 1e-15:
        return 1e15
    return max_val / norm

def logsumexp(x, beta=100.0):
    """Approximate max(x) using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)  # Only consider positive parts
    if len(x) == 0:
        return 0
    m = np.max(x)
    if m == 0:
        return 0
    return m + np.log(np.sum(np.exp((x - m) * beta)) / len(x)) / beta

def objective(h, beta=50.0):
    """Smooth approximation to the score."""
    h = np.asarray(h, dtype=float)
    # Ensure non-negative
    h = np.clip(h, 0, None)
    n = len(h)
    f = np.fft.rfft(h, n=2*n)
    conv = np.fft.irfft(f * f, n=2*n-1)
    conv *= dx
    # Compute log-sum-exp of conv
    lse = logsumexp(conv, beta)
    # Normalization
    integral_f = np.sum(h) * dx
    norm = integral_f ** 2
    if norm < 1e-15:
        return 1e15
    return lse / norm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Try a few different initializations
    inits = [
        np.ones(N),
        np.exp(-np.linspace(-1, 1, N)**2 * 10),
        np.sin(np.linspace(0, np.pi, N))**2,
    ]
    
    best_h = h0.copy()
    best_score = compute_score(h0)
    
    for h_init in inits:
        if time.time() > deadline:
            break
        h_init = np.clip(h_init, 0, None)
        # Normalize so that integral = 1
        s = np.sum(h_init) * dx
        if s > 0:
            h_init = h_init / s
        
        # Use L-BFGS-B for optimization
        try:
            res = minimize(
                lambda x: objective(x, beta=50.0),
                h_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() > deadline:
                break
            h_res = np.clip(res.x, 0, None)
            score_res = compute_score(h_res)
            if score_res < best_score:
                best_score = score_res
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Final polish: coordinate descent with exact 1D minimization
    if time.time() < deadline - 10:
        h = best_h.copy()
        # A few passes of coordinate descent
        for _ in range(3):
            if time.time() > deadline:
                break
            for i in range(N):
                if time.time() > deadline:
                    break
                # Minimize over h[i]
                def obj_1d(x):
                    h[i] = x
                    return compute_score(h)
                
                try:
                    res = minimize_scalar(
                        obj_1d,
                        bounds=(0, max(1.0, 2*best_h[i])),
                        method='bounded',
                        options={'xatol': 1e-8}
                    )
                    if res.success:
                        h[i] = max(0, res.x)
                except Exception:
                    pass
        score_final = compute_score(h)
        if score_final < best_score:
            best_score = score_final
            best_h = h.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
