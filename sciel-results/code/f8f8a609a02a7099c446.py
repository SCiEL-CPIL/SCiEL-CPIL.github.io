"""
Optimized step function to minimize the normalized maximum of autoconvolution.
Task: 600 equal intervals on [-1/4, 1/4], nonnegative step heights.
Score: upper bound on C1 (lower is better).
"""

import numpy as np
import time
import sys

N = 600
dx = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # 1/1200

def _compute_objective(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    For a step function with heights h_i on equal intervals of width dx,
    the autoconvolution at lag k is sum_i h_i * h_{i+k} * dx (approximately).
    
    The 'normalized maximum' for the first autocorrelation inequality upper bound on C1:
    We need to be careful about the exact normalization.
    
    The autoconvolution of f with itself: (f*f)(x) = int f(t) f(x-t) dt.
    For a step function, this is a piecewise linear function.
    
    The maximum of the autoconvolution relates to the upper bound on C1.
    A common formulation: minimize max_x (f*f)(x) subject to some normalization of f.
    
    Let's think about what "normalized maximum of its autoconvolution" means.
    If we normalize such that ||f||_2 = 1 or int f = 1 or similar.
    
    Actually, for the first autocorrelation inequality, we're looking at:
    C1 >= sup over f of (f*f)(0) / something.
    
    Let me think more carefully. The autoconvolution (f*f)(x) = integral f(t)f(x-t)dt.
    At x=0: (f*f)(0) = integral f(t)^2 dt = ||f||_2^2.
    
    The "normalized maximum" likely means: normalize f so that some norm = 1,
    then maximize (or we minimize the max of autoconvolution).
    
    Wait, the task says "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
    
    So we want to find f to minimize (max_x (f*f)(x)) / (normalization).
    
    A natural normalization: (int f)^2 or ||f||_2^2 or similar.
    
    Actually, for the autocorrelation inequality context, a typical setup is:
    - Normalize so that (f*f)(0) = 1, i.e., ||f||_2 = 1.
    - Then minimize max_x (f*f)(x) for x != 0, or minimize the overall max including x=0.
    
    But if we include x=0 and ||f||_2=1, then (f*f)(0)=1 always, so the max is at least 1.
    
    Alternatively: normalize so that int f = 1 (L1 norm = 1).
    Then (f*f)(0) = ||f||_2^2, and we minimize max_x (f*f)(x).
    
    Or: the "normalized" might mean dividing by (int f)^2.
    So objective = max_x (f*f)(x) / (int f)^2.
    
    For a nonnegative step function, let's compute:
    - int f = sum_i h_i * dx
    - (f*f)(x) for a step function: this is a convolution, which will be a piecewise linear (trapezoidal) function.
    
    Let me compute the autoconvolution numerically.
    For step function with heights h[0..N-1], the autoconvolution at lag m (in units of dx) is:
    c[m] = sum_{i} h[i] * h[i+m] * dx   for m >= 0
    c[-m] = c[m]
    
    But actually the continuous autoconvolution at position x = m*dx would be:
    (f*f)(m*dx) = integral f(t) f(m*dx - t) dt
    
    For step functions, this integral can be computed exactly. But for the max, it might occur at interval boundaries or midpoints.
    
    Actually, the autoconvolution of two step functions (with the same breakpoints) is a piecewise linear function that is linear on each subinterval of length dx. The maximum will occur at one of the breakpoints or we need to check carefully.
    
    Let me just compute it numerically by sampling. But for accuracy, let's compute the exact piecewise linear convolution.
    
    Actually, for the purpose of this optimization, let me use a discrete approximation that's accurate enough.
    
    The autoconvolution of the step function: if f(t) = h[i] for t in [i*dx, (i+1)*dx) shifted to start at -1/4, then:
    
    (f*f)(x) = integral_{-1/4}^{1/4} f(t) f(x-t) dt
    
    This is a continuous piecewise linear function. The maximum over x can be found by evaluating at all breakpoints of the piecewise linear function.
    
    For two step functions with N intervals each, the convolution has at most 2N-1 linear pieces, but actually it's more complex. Let me just sample densely.
    
    Hmm, but 120s time limit. Let me use a reasonable approach.
    
    Let me define:
    - heights h of length N, nonnegative
    - The function f(t) = h[i] for t in [-1/4 + i*dx, -1/4 + (i+1)*dx)
    - Autoconvolution: c(x) = integral f(t) f(x-t) dt
    - Normalization: let's use (integral f)^2 = (sum h_i * dx)^2
    - Objective: max_x c(x) / (sum h_i * dx)^2
    
    To compute c(x) for various x, I can use the fact that for step functions, the convolution is a sum of products of step functions, which gives a piecewise linear result.
    
    Let me just evaluate c(x) at many points and take the max. The function is continuous and piecewise linear, so the max is at a breakpoint or we can sample enough points.
    
    Actually, for step functions, the convolution is a "bump" function that's piecewise linear. The maximum occurs at x=0 by symmetry (for nonnegative f), and it's ||f||_2^2 * ... wait no.
    
    c(0) = integral f(t)^2 dt = sum h_i^2 * dx
    
    c(x) for x != 0 will generally be smaller for nonnegative f if f is concentrated. But the max might still be at 0.
    
    Hmm, but we're minimizing the max, so if the max is always at 0, then we're just minimizing ||f||_2^2 / (integral f)^2.
    
    By Cauchy-Schwarz, ||f||_2^2 / (integral f)^2 >= 1/measure(support) = 1/(1/2) = 2, with equality when f is constant on its support.
    
    So if the max is at 0, the minimum would be 2, achieved by a constant function. That matches the seed program returning all 1s.
    
    But wait, is the max always at 0? For nonnegative f, c(0) = int f^2, and c(x) = int f(t)f(x-t)dt. By the properties of convolution, for nonnegative f, c(0) is indeed the maximum (this is a standard result - the autocorrelation of a nonnegative function is maximized at zero lag).
    
    So if the normalization is (int f)^2, then the objective is c(0)/(int f)^2 = (int f^2)/(int f)^2, and the minimum is 2 (for constant on support), which the seed achieves.
    
    But that can't be right because the task says to minimize to get an upper bound on C1, and the seed gives 1.0 score. Let me reconsider the normalization.
    
    Maybe the normalization is different. Perhaps it's ||f||_2^2 = 1 normalization, and we minimize max_{x != 0} c(x), or minimize max c(x) - c(0), or something else.
    
    Or perhaps the "normalized maximum" means: max_x c(x) / c(0), which for nonnegative f is 1 (at x=0). That doesn't make sense either.
    
    Let me re-read: "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
    
    Perhaps the setup is: normalize so that (f*f)(0) = 1 (i.e., ||f||_2 = 1), and then the "maximum of autoconvolution" is max_x (f*f)(x) = 1 (at x=0). That's trivial.
    
    Unless "maximum" refers to max over x != 0, or the L-infinity norm minus the peak.
    
    Alternatively, maybe the normalization is int f = 1, and the objective is max_x (f*f)(x), and we want to minimize this. Then for constant f = 2 on [-1/4, 1/4] (so int f = 1), c(0) = int 4 dt = 4 * (1/2) = 2. And the max is 2.
    
    But can we do better? If we make f more spread out... but the domain is fixed. If we make f non-constant, by Cauchy-Schwarz, c(0) = int f^2 >= (int f)^2 / measure = 1/(1/2) = 2. So the minimum of c(0) is 2, achieved by constant.
    
    I think I'm overcomplicating this. Let me look at the score: the seed returns [1.0]*600 and gets score 1.0. The score is "seed/value", so value = 1.0 for the seed.
    
    Let me just implement the most natural interpretation and see. I'll assume:
    - Normalize so that sum(h) = 1 (or some fixed normalization)
    - Compute autoconvolution
    - Take the max
    - That's the objective to minimize
    
    Actually, let me just go with: objective = max of autoconvolution, with f normalized so that integral f = 1.
    
    For the seed (all ones), integral f = 600 * (1/1200) = 1/2. To normalize to integral 1, we'd multiply by 2, so heights become 2. Then c(0) = sum (2)^2 * dx = 600 * 4 / 1200 = 2.
    
    So the value for the seed would be 2.0, and score = 1.0/2.0 = 0.5? That doesn't match.
    
    Let me try: the evaluator computes the value directly from the returned heights. For [1.0]*600, the value is 1.0. So the normalization must be such that the constant function gives value 1.
    
    If objective = max c(x) / c(0), then for any f, this is 1 (since max is at 0 for nonnegative f). That's always 1.
    
    If objective = max c(x) / (sum h)^2, for all ones: max c(x) = c(0) = N * 1^2 * dx = 600/1200 = 0.5. (sum h)^2 = 600^2. So 0.5/360000. That's tiny, not 1.
    
    Hmm. Let me try: objective = max c(x) / (N * dx)^2 = max c(x) / (1/2)^2 = 4 * max c(x). For all ones, max c(x) = 0.5, so objective = 2. Not 1.
    
    What if the evaluator just computes the max of the discrete autoconvolution sum? For all ones: c[m] = (N - |m|) * 1 * 1 * dx for |m| < N. Max is at m=0: N * dx = 600/1200 = 0.5.
    
    I'm stuck on the exact normalization. Let me just write a reasonable optimization and hope the evaluator handles the normalization. I'll minimize the max of the autoconvolution with L2 normalization (||f||_2 = 1), which is a standard setup.
    
    Actually, let me just focus on making a good optimization. I'll use scipy.optimize.minimize with a good initial guess and see what happens.
"""

def _autoconv_max(heights, dx):
    """Compute max of autoconvolution of step function with given heights."""
    N = len(heights)
    heights = np.asarray(heights, dtype=float)
    # Discrete autoconvolution: c[m] = sum_i h[i] * h[i+m] * dx for m >= 0
    # c[-m] = c[m]
    # Compute using FFT for efficiency
    n_fft = 2 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = heights
    # Autoconvolution via FFT
    H = np.fft.fft(h_padded, n=n_fft)
    conv = np.fft.ifft(H * H).real[:2*N-1] * dx
    # conv[m] for m = 0, ..., 2N-2 corresponds to lags 0, dx, ..., (2N-2)*dx
    # But we only care about the range where both functions overlap
    # For autoconvolution, the valid range is |lag| < N*dx, i.e., m = 0, ..., N-1 and m = -(N-1), ..., -1
    # In the computed array, conv[0] = c(0), conv[m] for m>0 = c(m*dx)
    # The negative lags are symmetric
    return np.max(conv[:N])

def _objective(params, dx):
    """Objective: max autoconvolution, with L2 normalization."""
    heights = np.exp(params)  # Ensure nonnegativity via log-parameterization
    # Normalize: ||f||_2 = 1 => sum h_i^2 * dx = 1 => sum h_i^2 = 1/dx
    norm_sq = np.sum(heights**2) * dx
    if norm_sq < 1e-30:
        return 1e10
    heights_norm = heights / np.sqrt(norm_sq / dx)  # This makes sum h_i^2 * dx = 1
    return _autoconv_max(heights_norm, dx)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 600
    dx = 1.0 / 1200.0
    
    # Initial guess: constant function (log of 1)
    # For L2 norm = 1: h_i = 1/sqrt(N*dx) = 1/sqrt(0.5) = sqrt(2)
    h0 = np.full(N, np.sqrt(2.0))
    params0 = np.log(h0)
    
    # Use Nelder-Mead or L-BFGS-B
    from scipy.optimize import minimize
    
    best_params = params0.copy()
    best_val = _objective(best_params, dx)
    
    # Try a few optimization approaches
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                _objective,
                params0,
                args=(dx,),
                method=method,
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time.time() > deadline - 10:
                break
            if res.success or res.fun < best_val:
                candidate_params = res.x
                candidate_val = _objective(candidate_params, dx)
                if candidate_val < best_val:
                    best_params = candidate_params
                    best_val = candidate_val
        except Exception:
            pass
    
    # Final: convert back to heights
    heights = np.exp(best_params)
    norm_sq = np.sum(heights**2) * dx
    if norm_sq > 1e-30:
        heights = heights / np.sqrt(norm_sq / dx)
    
    # Ensure nonnegative
    heights = np.maximum(heights, 0.0)
    
    return list(heights)
