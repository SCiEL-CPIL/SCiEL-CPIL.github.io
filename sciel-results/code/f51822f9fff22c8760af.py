import numpy as np
from scipy.optimize import linprog
import time

N = 600

def _autoconvolve(f):
    """Compute the autoconvolution of a step function represented by N heights on [-1/4, 1/4].
    The autoconvolution (f*f)(t) for t in [-1/2, 1/2] is piecewise linear with 2N-1 peaks."""
    # f is an array of N nonnegative values, each representing the height on an interval of width 1/(2N)
    # Wait, the interval is [-1/4, 1/4], length 1/2. So each interval has width (1/2)/N = 1/(2N).
    # The autoconvolution of two such step functions is a piecewise linear function.
    # The peaks of the autoconvolution occur at the midpoints of the convolution intervals.
    # For a step function with N intervals, the autoconvolution has 2N-1 linear pieces.
    # The peaks (local maxima) are at the centers of these pieces.
    # Actually, for a step function, the autoconvolution is a "tent" shape if f is constant.
    # The maximum of the autoconvolution is at t=0 if f is symmetric, but generally the max
    # occurs at one of the 2N-1 peak points.
    
    # The autoconvolution at the peak points can be computed via FFT or direct summation.
    # For a step function f with values f[0], f[1], ..., f[N-1] on intervals of width w = 1/(2N),
    # the autoconvolution (f*f)(t) at t = k*w/2 for k=0,...,2N-2 (the peak points) is:
    # (f*f)(k*w/2) = w * sum_{i} f[i] * f[j] where the overlap is maximized.
    # Actually, the peak values are proportional to the linear convolution of f with itself,
    # evaluated at the appropriate points.
    
    # Let's use the fact that the autoconvolution of step functions is a piecewise linear function
    # whose peaks are given by the linear convolution of the step heights.
    
    # The linear convolution of f with f has length 2N-1.
    # The peak values of the autoconvolution are (w^2) * (linear convolution of f with f).
    # Wait, let's think about this more carefully.
    
    # If f(s) = f[i] for s in [i/(2N), (i+1)/(2N)) - 1/4, i.e., s in [-1/4 + i/(2N), -1/4 + (i+1)/(2N)),
    # then (f*f)(t) = integral f(s) f(t-s) ds.
    
    # The peaks of (f*f) occur at t = (k+1)/(2N) - 1/4 + 1/(4N) = (k+0.5)/(2N) - 1/4 + 1/(4N)? 
    # This is getting complicated. Let's use a simpler approach.
    
    # The autoconvolution of a step function with N equal intervals is a piecewise linear function
    # with 2N-1 linear segments. The peaks are at the junctions of these segments.
    # The peak values can be computed as:
    # peak[k] = (1/(2N)) * sum_{i=0}^{min(k, N-1)} f[i] * f[k-i] for k=0,...,N-1
    # and by symmetry for the other half.
    # Wait, that's not quite right either.
    
    # Let's use the standard result: for a step function f with values f[0],...,f[N-1] on N equal
    # intervals, the autoconvolution at the peak points is given by the linear convolution of f with f,
    # scaled by the interval width.
    
    # The linear convolution c = f * f has length 2N-1, where c[k] = sum_{i} f[i]*f[k-i].
    # The peak values of the autoconvolution are c[k] * w^2 where w = 1/(2N) is the interval width?
    # No, that's not right. Let's think about the units.
    
    # f(s) has units of "height". The autoconvolution (f*f)(t) = integral f(s)f(t-s)ds has units of
    # "height^2 * length". If f is a probability density (integral = 1), then (f*f) is also a pdf.
    
    # For a step function with N intervals of width w = 1/(2N), f(s) = f[i]/(w) * w? No.
    # If f[i] is the height on the i-th interval, then f(s) = f[i] for s in the i-th interval.
    # The integral of f is sum f[i] * w.
    
    # The autoconvolution at the peak point t_k (the center of the k-th linear segment of the
    # autoconvolution) is:
    # (f*f)(t_k) = sum_{i,j: intervals i and j overlap at t_k} f[i]*f[j] * (overlap length)
    
    # For a step function, the autoconvolution is a "trapezoidal" shape. The peaks are:
    # peak[k] = (1/(2N)) * sum_{i=0}^{min(k, N-1)} f[i] * f[k-i]  ??? 
    # This doesn't look right. Let's use a concrete example.
    
    # If f = [1, 1, ..., 1] (constant), then (f*f)(t) is a triangle with peak at t=0 (or t=1/4?)
    # of height 1 * (1/2) = 1/2? No.
    # If f(s) = 1 for s in [-1/4, 1/4], then (f*f)(t) = integral f(s)f(t-s)ds.
    # For t=0: integral_{-1/4}^{1/4} 1*1 ds = 1/2.
    # For t=1/4: integral where s and t-s are both in [-1/4,1/4]: s in [0,1/4], t-s in [-1/4,1/4].
    # The overlap length is 1/4. So (f*f)(1/4) = 1/4.
    # The peak is at t=0 with value 1/2.
    
    # So for a constant step function with height h, the peak autoconvolution is h^2 * (1/2).
    
    # Now, for a general step function, the peak values of the autoconvolution are:
    # peak[k] = (1/(2N)) * sum_{i=0}^{k} f[i] * f[k-i] for k = 0, ..., N-1
    # Wait, for k=0: peak[0] = (1/(2N)) * f[0]*f[0]. But for the constant case, the peak at t=0
    # (which corresponds to k=0 if we center at 0) should be h^2/2.
    # (1/(2N)) * h^2 is not h^2/2 unless N=1.
    
    # I think the correct formula is:
    # The autoconvolution of a step function with N intervals of width w has peaks at the
    # 2N-1 points t_k = -1/4 + (k+0.5)*w for k=0,...,2N-2.
    # The peak value at t_k is:
    # peak[k] = w * sum_{i} f[i] * f[j] where j = k - i and the intervals overlap.
    
    # Actually, let's use the FFT-based approach or just compute the linear convolution.
    # The key insight is that for a step function, the autoconvolution's peak values are
    # proportional to the linear convolution of the step heights.
    
    # Let's use the fact that the maximum of the autoconvolution is:
    # max_t (f*f)(t) = (w/2) * max_k (linear_convolution_of_f_with_f)[k]
    # where w = 1/(2N).
    
    # Wait, let's verify with the constant case: f = [h, h, ..., h], N intervals, width w = 1/(2N).
    # Linear convolution of f with f: c[k] = sum_{i} f[i]*f[k-i] = h^2 * (min(k+1, N, 2N-1-k)) for k=0,...,2N-2.
    # The max of c[k] is h^2 * N (at k=N-1).
    # So max autoconvolution = (w/2) * h^2 * N = (1/(4N)) * h^2 * N = h^2/4.
    # But we computed the peak at t=0 as h^2/2. So this formula gives h^2/4, which is wrong.
    
    # Let me reconsider. The autoconvolution (f*f)(t) for a step function is a piecewise linear
    # function. The "peak" values I'm computing are actually the values at the midpoints of the
    # linear segments, not the local maxima.
    
    # For a step function, the autoconvolution is a "tent" function (if f is symmetric).
    # The local maxima of the autoconvolution occur at the points where the slope changes.
    # These are the points t = -1/4 + k*w for k=0,...,2N-1.
    # At these points, the autoconvolution has a "kink".
    
    # The value of the autoconvolution at the kink point t_k = -1/4 + k*w is:
    # (f*f)(t_k) = sum_{i,j: interval_i and interval_j are both to the left or right of t_k} ...
    
    # This is getting too complicated. Let's just use a direct numerical computation.
    # For a step function with N=600, we can compute the autoconvolution at all 2N-1 peak points
    # using the linear convolution.
    
    # The correct formula: for a step function f with values f[0],...,f[N-1] on N equal intervals
    # of width w, the autoconvolution (f*f)(t) is a piecewise linear function. The local maxima
    # (peaks) occur at t = -1/4 + (k+1)*w for k=0,...,2N-2.
    # The peak value at t_k is:
    # peak[k] = (w/2) * (f * f)[k]  ??? 
    # No, let's just compute it directly.
    
    # Actually, the simplest approach: the autoconvolution of a step function can be computed
    # as follows. The step function f can be written as f(s) = sum_i f[i] * I_{[a_i, b_i)}(s)
    # where I is the indicator function. Then (f*f)(t) = sum_i sum_j f[i]f[j] * (length of 
    # intersection of [a_i, b_i) and [t-b_j, t-a_j)).
    
    # The peaks of this function occur at t = a_i - b_j for various i,j.
    # The maximum of (f*f) is the maximum over all such t of the sum.
    
    # For N=600, there are O(N^2) such points, which is 360,000. That's manageable.
    # But we can do better using the linear convolution.
    
    # Key fact: for a step function with N equal intervals, the autoconvolution's peak values
    # are given by:
    # peak[k] = (w/2) * sum_{i=0}^{min(k, N-1)} f[i] * f[k-i]  for k = 0, ..., 2N-2
    # Wait, I think the correct formula is:
    # The autoconvolution of a step function with N equal intervals of width w is a piecewise
    # linear function with 2N-1 linear segments. The peaks (local maxima) are at the junctions
    # of these segments, and the peak values are:
    # peak[k] = w * (f * f)[k]  where (f*f) is the linear convolution? 
    # No, that gives the wrong units.
    
    # Let's just use the fact that we can compute the autoconvolution at all relevant points
    # using the linear convolution. The key insight from the literature is that for a step
    # function, the maximum of the autoconvolution is:
    # max (f*f)(t) = (w^2) * max_k (linear_conv(f, f))[k]  ??? 
    # Let's test: f = [1,1,...,1], N=600, w=1/1200.
    # linear_conv(f,f) has max value = N = 600.
    # w^2 * 600 = (1/1200)^2 * 600 = 600/1440000 = 1/2400.
    # But the true peak is 1/2. So this is wrong.
    
    # I think the issue is that I'm confusing the step height with the probability mass.
    # If f(s) = h on each interval, then the probability mass on each interval is h*w.
    # The autoconvolution of the probability density f is (f*f)(t).
    # If f is a pdf with f(s) = h, then h * N * w = 1, so h = 1/(N*w) = 1/(1/2) = 2.
    # So h=2 for a uniform pdf on [-1/4, 1/4].
    # (f*f)(0) = integral f(s)f(-s)ds = integral_{-1/4}^{1/4} 2*2 ds = 4 * 1/2 = 2.
    # So the peak is 2 for a uniform pdf.
    
    # If we use step heights h_i, the autoconvolution peak is:
    # (f*f)(t) = sum_{i,j} h_i * h_j * (overlap_length(i,j,t))
    # The maximum over t of this is what we want to minimize.
    
    # For the purpose of this problem, I'll use the fact that the autoconvolution of a step
    # function can be computed via the linear convolution of the step heights, scaled appropriately.
    # The exact scaling doesn't matter for optimization since we're minimizing the max, and
    # the scaling is a constant factor.
    
    # So: max_autoconv = max_k (linear_conv(f, f))[k] * (some constant)
    # We minimize max_k (linear_conv(f, f))[k] subject to sum f[i] = 1 (normalization) and f[i] >= 0.
    
    # The linear convolution of f with f can be computed efficiently using FFT.
    
    # So the optimization problem is:
    # min max_k c[k]  where c = linear_conv(f, f)
    # subject to: sum f[i] = 1, f[i] >= 0
    
    # This is a convex optimization problem (minimizing the max of convex functions).
    # We can solve it using binary search on the max value + LP feasibility check,
    # or directly using a convex solver.
    
    # For N=600, the linear convolution has length 1199. We can use scipy.optimize.linprog
    # with the constraint that all c[k] <= t, and minimize t.
    
    # The constraints are:
    # sum f[i] = 1
    # f[i] >= 0
    # c[k] = sum_i f[i]*f[k-i] <= t  for all k
    # The c[k] constraints are quadratic in f, so this is a QP, not an LP.
    
    # We can use scipy.optimize.minimize with a suitable method, or we can use the
    # fact that this is a special structure.
    
    # Alternative: use the Fourier domain. The autoconvolution in Fourier space is
    # the square of the Fourier transform. The peak of the autoconvolution is related
    # to the L-infinity norm of the inverse Fourier transform of |F(f)|^2.
    
    # For now, let's use a simple approach: start with a uniform distribution and use
    # gradient descent or a convex optimizer to minimize the max autoconvolution.
    
    # Actually, let's use a different approach. The problem is equivalent to finding a
    # probability distribution on {0,...,N-1} that minimizes the maximum of its
    # self-convolution. This is a well-studied problem in probability and signal processing.
    
    # The optimal distribution is known to be related to the "flat-top" window or the
    # "prolate spheroidal" sequence. For discrete distributions, the optimal solution
    # is to spread the mass as evenly as possible, but with some structure.
    
    # Let's use scipy.optimize to solve this. We'll use the fact that the objective is
    # convex and use a gradient-based method.
    
    # For the initial proposal, let's use a simple heuristic: use a cosine window or
    # a Gaussian-like window, which should give a better peak than uniform.
    
    # Actually, let's just return a slightly modified uniform distribution for now,
    # and refine in later iterations.
    
    # For a first pass, let's use a cosine-tapered window:
    # f[i] = cos(pi * (i - (N-1)/2) / (N-1))  for i in the middle, 0 at the edges
    # This should reduce the peak of the autoconvolution compared to uniform.
    
    # Wait, but the task says "nonnegative step function", so we need f[i] >= 0.
    # A cosine window that goes negative won't work. Let's use a raised cosine:
    # f[i] = 0.5 * (1 + cos(pi * (i - (N-1)/2) / ((N-1)/2)))  for |i - (N-1)/2| <= (N-1)/2
    # This is nonnegative and goes to 0 at the edges.
    
    # Actually, the uniform distribution is already a valid starting point. Let's
    # compute the autoconvolution peak for uniform and then try to improve.
    
    # For the purpose of this code, let's return the uniform distribution for now.
    # In future iterations, we can use optimization to improve.
    
    return np.ones(N) / N

def _compute_score(f):
    """Compute the upper bound on C1, which is the normalized maximum of the autoconvolution."""
    # Normalize f so that sum f[i] = 1 (it should already be, but just in case)
    f = np.array(f, dtype=np.float64)
    f = f / np.sum(f)
    
    # Compute the linear convolution of f with f
    c = np.convolve(f, f, mode='full')
    
    # The peak of the autoconvolution is proportional to the max of c
    # The normalization factor depends on the interval width.
    # For the purpose of comparing, we can just use max(c).
    # The actual C1 upper bound is max(c) * (1/(2N)) or similar, but the scaling
    # doesn't matter for optimization since it's a constant.
    
    return np.max(c)

def construct():
    """Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with uniform distribution
    f = np.ones(N) / N
    best_f = f.copy()
    best_score = _compute_score(f)
    
    # Try a few simple heuristics
    # 1. Cosine-tapered window
    i = np.arange(N)
    cos_f = 0.5 * (1 + np.cos(np.pi * (i - (N-1)/2) / ((N-1)/2)))
    cos_f = np.maximum(cos_f, 0)
    if np.sum(cos_f) > 0:
        cos_f = cos_f / np.sum(cos_f)
        score = _compute_score(cos_f)
        if score < best_score:
            best_score = score
            best_f = cos_f
    
    # 2. Gaussian-like window
    center = (N-1)/2
    sigma = N / 4
    gauss_f = np.exp(-0.5 * ((i - center) / sigma) ** 2)
    gauss_f = gauss_f / np.sum(gauss_f)
    score = _compute_score(gauss_f)
    if score < best_score:
        best_score = score
        best_f = gauss_f
    
    # 3. Try to optimize using gradient descent
    if time.time() < deadline - 10:
        f = best_f.copy()
        lr = 0.01
        for it in range(1000):
            if time.time() > deadline - 5:
                break
            # Compute gradient of max(c) w.r.t. f
            c = np.convolve(f, f, mode='full')
            max_idx = np.argmax(c)
            # Gradient of c[max_idx] w.r.t. f
            # c[k] = sum_i f[i]*f[k-i]
            # dc[k]/df[j] = f[k-j] + f[j-k] (if valid indices)
            grad = np.zeros(N)
            k = max_idx
            for j in range(N):
                if 0 <= k - j < N:
                    grad[j] += f[k - j]
                if 0 <= j - k < N:
                    grad[j] += f[j - k]
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            # Update
            f = f - lr * grad
            f = np.maximum(f, 0)
            f = f / np.sum(f)
            score = _compute_score(f)
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    return list(best_f)
