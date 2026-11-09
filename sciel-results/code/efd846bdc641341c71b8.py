"""Optimized program for the second autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _ratio(h):
    """Compute the ratio: (L2 norm squared) / (L1 norm * sup norm of autoconvolution).
    
    For nonnegative heights h of length N:
    - L2 norm squared = sum(h_i^2)
    - L1 norm = sum(h_i)
    - Autoconvolution: c_k = sum_i h_i * h_{i+k} (for all lags)
    - Sup norm of autoconvolution = max_k c_k
    
    Note: For nonnegative h, the autoconvolution at lag 0 is sum(h_i^2) = L2^2,
    and since h_i >= 0, c_0 >= c_k for all k. So sup norm = c_0 = L2^2.
    
    Wait, that would make the ratio = L2^2 / (L1 * L2^2) = 1/L1, which is maximized
    by making L1 as small as possible. But we need nonnegative heights, and if they
    can be arbitrarily small, the ratio goes to infinity.
    
    Let me re-read: "squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    
    Actually, let me reconsider. The autoconvolution of a discrete signal h is:
    (h * h)[k] = sum_i h[i] * h[i+k]
    
    For nonnegative h, the maximum of the autoconvolution is indeed at lag 0:
    (h*h)[0] = sum_i h[i]^2 = ||h||_2^2
    
    So the ratio would be ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.
    
    This doesn't make sense for an optimization problem unless there's a constraint.
    
    Let me reconsider the problem. Perhaps the "autoconvolution" here means the 
    circular/periodic autoconvolution, or perhaps the heights define a piecewise
    constant function and we're looking at the convolution of that function with itself.
    
    Actually, re-reading: "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4]"
    
    So we have a piecewise constant function f(x) defined on [-1/4, 1/4] with 50 equal
    subintervals. The autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    For a piecewise constant function with heights h_0, ..., h_{N-1} on intervals
    [x_i, x_{i+1}] where x_i = -1/4 + i*(1/2)/N, the convolution is more complex.
    
    The sup norm of the autoconvolution is max_t |f*f(t)|.
    
    L1 norm of f = integral |f(x)| dx = sum h_i * dx where dx = (1/2)/N = 1/100
    L2 norm squared = integral f(x)^2 dx = sum h_i^2 * dx
    
    So the ratio is: (sum h_i^2 * dx) / ((sum h_i * dx) * sup|f*f|)
    = (sum h_i^2) / ((sum h_i) * sup|f*f| / dx)
    
    Let me compute f*f properly. If f is piecewise constant with height h_j on interval
    I_j = [a_j, b_j] where a_j = -1/4 + j*dx, b_j = a_j + dx, dx = 1/100.
    
    (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    
    This is a convolution of a box-like function. The result will be piecewise linear
    (trapezoidal) in general.
    
    For the seed program h = [1,1,...,1], f is the indicator of [-1/4, 1/4], so
    f*f is a triangle function with peak 1/2 at t=0 and support [-1/2, 1/2].
    sup|f*f| = 1/2.
    L1 = 1/2, L2^2 = 1/2.
    Ratio = (1/2) / ((1/2)*(1/2)) = (1/2)/(1/4) = 2.
    
    So the seed score should correspond to ratio 2? But the task says seed score is 1.0
    and it's value/seed. So the raw value for the seed is some C2, and score = value/seed_value.
    
    Let me just compute the ratio numerically and optimize it.
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 0.0
    
    dx = 1.0 / 100.0  # interval width: (1/2) / 50
    
    # L1 norm and L2 norm squared of the piecewise constant function
    l1 = np.sum(h) * dx
    l2_sq = np.sum(h**2) * dx
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    
    # Compute autoconvolution f*f
    # f is defined on [-1/4, 1/4], so f*f is defined on [-1/2, 1/2]
    # We need to find sup |f*f|
    
    # The convolution of two piecewise constant functions is piecewise linear
    # with breakpoints at all sums of interval endpoints.
    
    # Interval endpoints for f:
    # a_j = -1/4 + j*dx for j=0..N-1, plus a_N = 1/4
    a = np.linspace(-0.25, 0.25, N + 1)  # endpoints
    
    # f*f(t) = integral f(s) f(t-s) ds
    # Breakpoints of f*f occur at a_i + a_j for all i,j
    # But more efficiently, the breakpoints are at a_i - a_j (shifts)
    
    # Actually, let's use the fact that f*f is the convolution of a step function with itself.
    # The breakpoints of the convolution are at all values a_i + a_j where a_i and a_j
    # are endpoints of the intervals. Since both functions have the same support [-1/4,1/4],
    # the convolution support is [-1/2, 1/2].
    
    # Let me just numerically evaluate f*f on a fine grid and find the max.
    # But for optimization, I need a differentiable-ish function. Let me use a moderate grid.
    
    # Actually, for the optimization, let me compute f*f analytically.
    # f*f(t) = sum_j h_j * (length of intersection of [a_j, a_{j+1}] and [t - 1/4, t + 1/4])
    # Wait, that's not right either.
    
    # (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    # = sum_j h_j * integral_{s in I_j} f(t-s) ds
    # = sum_j h_j * sum_k h_k * length(I_j ∩ (t - I_k))
    
    # where t - I_k = [t - a_{k+1}, t - a_k]
    
    # This is getting complex. Let me just use FFT for speed during optimization.
    
    # Pad h to handle the convolution properly.
    # The function f is nonzero on [-1/4, 1/4], length 1/2.
    # f*f is nonzero on [-1/2, 1/2], length 1.
    # 
    # To compute f*f via FFT, I can represent f on a grid and use circular convolution
    # with enough padding.
    
    # Let's use a grid with M points over [-1/2, 1/2] for the convolution result.
    # The original function has N=50 intervals of width dx=1/100 over [-1/4, 1/4].
    # Let M = 200 points over [-1/2, 1/2], so grid spacing = 1/200.
    
    M = 200
    t_grid = np.linspace(-0.5, 0.5, M, endpoint=False)
    
    # Evaluate f on the grid for the convolution
    # f(x) = h_j if x in [a_j, a_{j+1})
    # For the convolution, I need f(s) and f(t-s)
    
    # Alternative: use the discrete convolution formula directly.
    # (f*f)(t) = sum_{j,k} h_j h_k * length(I_j ∩ (t - I_k))
    
    # I_j = [a_j, a_{j+1}], t - I_k = [t - a_{k+1}, t - a_k]
    # Intersection length = max(0, min(a_{j+1}, t - a_k) - max(a_j, t - a_{k+1}))
    
    # This is O(N^2 * M) which is 50*50*200 = 500,000 operations. Acceptable.
    
    # But for optimization with many evaluations, this might be slow.
    # Let me precompute the structure.
    
    # Actually, let me use a simpler approach: since the breakpoints of f*f are at
    # a_i + a_j (or a_i - a_j), I can evaluate f*f at all breakpoints and take the max.
    # The number of breakpoints is O(N^2) = 2500.
    
    # f*f(t) is piecewise linear, so its maximum occurs at a breakpoint or at the
    # endpoints of its support.
    
    # Breakpoints: all values a_i - a_j for i,j in 0..N (these are where the
    # intersection length formula changes).
    
    # Actually, the breakpoints of (f*f)(t) as a function of t are at all values
    # a_i + a_j where a_i, a_j are endpoints. But since we're computing f*f(t) = 
    # integral f(s)f(t-s)ds, the breakpoints are at t = a_i - a_j for all i,j.
    
    # Let me collect all unique breakpoints in [-1/2, 1/2].
    all_bp = np.unique(a[:, None] - a[None, :])
    all_bp = all_bp[(all_bp >= -0.5) & (all_bp <= 0.5)]
    
    # Evaluate f*f at each breakpoint
    def _conv_at(t_val):
        total = 0.0
        for j in range(N):
            for k in range(N):
                # Intersection of [a_j, a_{j+1}] and [t_val - a_{k+1}, t_val - a_k]
                lo = max(a[j], t_val - a[k+1])
                hi = min(a[j+1], t_val - a[k])
                if hi > lo:
                    total += h[j] * h[k] * (hi - lo)
        return total
    
    # This is O(N^2 * len(all_bp)) = 2500 * 5000 = 12.5M operations per evaluation.
    # That's too slow for optimization.
    
    # Let me use a coarser grid approach for the optimization.
    # Evaluate f*f on a grid of ~1000 points and take the max.
    
    # Better: use FFT-based convolution.
    # Represent f on a fine grid, zero-pad, FFT, multiply, IFFT.
    
    # Grid for f: 100 points over [-1/4, 1/4] (since dx = 1/100, N=50 intervals)
    # Actually, f is already defined by 50 intervals. Let me sample it on a finer grid.
    
    # Let me use M_fft = 400 points over [-1/2, 1/2] for the convolution.
    # f is supported on [-1/4, 1/4], so I place it in the middle.
    
    M_fft = 400
    dt = 1.0 / M_fft  # = 1/400
    
    # Create grid for the convolution
    t_full = np.linspace(-0.5, 0.5, M_fft, endpoint=False)
    
    # Evaluate f on this grid
    # f(x) = h_j for x in [a_j, a_{j+1})
    f_vals = np.zeros(M_fft)
    for j in range(N):
        mask = (t_full >= a[j]) & (t_full < a[j+1])
        f_vals[mask] = h[j]
    # Handle the last interval
    mask_last = (t_full >= a[N-1]) & (t_full <= a[N])
    f_vals[mask_last] = h[N-1]
    
    # Circular convolution via FFT
    F = np.fft.rfft(f_vals, n=2*M_fft)
    conv = np.fft.irfft(F * F, n=2*M_fft)
    conv *= dt  # Scale by dt for the integral
    
    # The convolution result is on a grid of 2*M_fft points over [-1, 1]
    # But we only care about [-1/2, 1/2], which is the first M_fft points
    # Actually, with zero-padding, the linear convolution is in the first M_fft points.
    
    # The sup norm of f*f
    sup_conv = np.max(np.abs(conv[:M_fft]))
    
    if sup_conv < 1e-15:
        return 0.0
    
    ratio = l2_sq / (l1 * sup_conv)
    return ratio


def _neg_ratio(h):
    return -_ratio(h)


# EVOLVE-BLOCK-START
def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    # Strategy: Use sparse support with geometric decay to make L2^2 large relative
    # to the autoconvolution peak.
    
    # Key insight: If we put all mass in a few intervals, the L1 norm is small
    # relative to L2^2, and the autoconvolution peak is controlled.
    
    # For a single nonzero height h at position j:
    # L1 = h*dx, L2^2 = h^2*dx
    # f*f is a triangle of height h^2*dx at t=0
    # Ratio = h^2*dx / (h*dx * h^2*dx) = 1/(h*dx)
    # This goes to infinity as h->0, which can't be right.
    
    # Wait, I think I'm making an error. Let me recompute for the seed case.
    # Seed: h = [1,1,...,1] (50 ones)
    # f(x) = 1 for x in [-1/4, 1/4]
    # L1 = integral |f| = 1/2
    # L2^2 = integral f^2 = 1/2
    # f*f(t) = triangle: f*f(0) = 1/2, support [-1/2, 1/2]
    # sup|f*f| = 1/2
    # Ratio = (1/2) / ((1/2) * (1/2)) = 2
    
    # For a single spike h=[0,...,0,h,0,...,0]:
    # L1 = h*dx = h/100
    # L2^2 = h^2*dx = h^2/100
    # f*f(t): f is h on interval [a_j, a_{j+1}], width dx=1/100
    # f*f is a triangle of height h^2*dx = h^2/100 at t=0
    # sup|f*f| = h^2/100
    # Ratio = (h^2/100) / ((h/100) * (h^2/100)) = (h^2/100) / (h^3/10000) = 100/h
    # As h->infinity, ratio->0. As h->0, ratio->infinity.
    
    # This suggests the ratio is unbounded, which means I'm misunderstanding the problem.
    
    # Let me re-read: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    
    # Maybe it's: (L2 norm)^2 / (L1 norm) / (sup norm of autoconvolution)
    # = L2^2 / (L1 * sup|f*f|)
    
    # Or maybe: L2^2 / (L1 * sup|f*f|) where these are norms of the function,
    # not scaled by dx.
    
    # Or perhaps the "heights" are the values, and the norms are computed as:
    # L1 = sum h_i (not sum h_i * dx)
    # L2^2 = sum h_i^2
    # Autoconvolution computed discretely
    
    # Let me try: ratio = sum(h_i^2) / (sum(h_i) * max_k sum_i h_i h_{i+k})
    
    # For seed h=[1,...,1]:
    # sum h_i^2 = 50
    # sum h_i = 50
    # autoconv at lag 0: sum h_i^2 = 50
    # autoconv at lag k: 50-|k| for |k|<50
    # sup = 50
    # ratio = 50/(50*50) = 1/50 = 0.02
    
    # That gives a very small number. The seed score is 1.0 = value/seed_value.
    # So seed_value = 0.02, and we want to maximize value.
    
    # For a single spike h=[0,...,1,0,...]:
    # sum h_i^2 = 1
    # sum h_i = 1  
    # autoconv at lag 0: 1
    # sup = 1
    # ratio = 1/(1*1) = 1
    
    # That's 50x better! Let me verify with two spikes.
    # h = [1, 0, ..., 1, 0, ...] at positions 0 and 49:
    # sum h_i^2 = 2
    # sum h_i = 2
    # autoconv at lag 0: 2
    # autoconv at lag 49: 1
    # sup = 2
    # ratio = 2/(2*2) = 0.5
    
    # So spreading mass reduces the ratio. The best is to concentrate mass.
    
    # What about h = [c, 0, 0, ..., 0]?
    # sum h_i^2 = c^2
    # sum h_i = c
    # autoconv at lag 0: c^2
    # sup = c^2
    # ratio = c^2/(c*c^2) = 1/c
    
    # As c -> infinity, ratio -> 0. As c -> 0, ratio -> infinity.
    # But heights must be nonnegative, not necessarily bounded.
    
    # Hmm, but if the ratio is unbounded, the problem is ill-posed. There must be
    # a constraint I'm missing. Perhaps the heights must sum to 1, or the L1 norm
    # must be 1, or something else.
    
    # Let me re-read the task: "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4] to maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
    
    # Maybe the autoconvolution is of the indicator function (not weighted by heights),
    # and the heights just define where the function is nonzero?
    
    # Or maybe it's: the function is f(x) = h_i on interval i, and we compute:
    # ||f||_2^2 / (||f||_1 * ||f*f||_inf)
    
    # Where ||f||_2^2 = integral f^2 dx, ||f||_1 = integral |f| dx,
    # ||f*f||_inf = sup |f*f|
    
    # For f = h * 1_{[a,b]} (constant h on interval [a,b] of width w):
    # ||f||_2^2 = h^2 * w
    # ||f||_1 = h * w
    # f*f is a triangle of height h^2*w at t=0
    # ||f*f||_inf = h^2*w
    # Ratio = h^2*w / (h*w * h^2*w) = 1/(h*w)
    
    # Still unbounded as h->0.
    
    # I think there might be a normalization constraint. Let me assume the heights
    # must satisfy sum h_i = 1 (L1 norm of the discrete sequence is 1) or
    # integral f = 1.
    
    # If sum h_i = 1:
    # For single spike h=[1,0,...,0]: ratio = 1/(1*1) = 1
    # For uniform h=[1/50,...,1/50]: 
    #   sum h_i^2 = 50*(1/50)^2 = 1/50
    #   sum h_i = 1
    #   autoconv at lag 0 = 1/50
    #   sup = 1/50
    #   ratio = (1/50)/(1*1/50) = 1
    # Same!
    
    # For two spikes h=[1/2, 0,..., 1/2, 0,...] at positions 0 and 49:
    #   sum h_i^2 = 2*(1/4) = 1/2
    #   sum h_i = 1
    #   autoconv at lag 0 = 1/2
    #   autoconv at lag 49 = 1/4
    #   sup = 1/2
    #   ratio = (1/2)/(1*1/2) = 1
    
    # Interesting, all give ratio 1 with sum h_i = 1?
    
    # Let me try h=[0.8, 0.2, 0,...,0]:
    #   sum h_i^2 = 0.64 + 0.04 = 0.68
    #   sum h_i = 1
    #   autoconv at lag 0 = 0.68
    #   autoconv at lag 1 = 0.8*0.2 = 0.16
    #   sup = 0.68
    #   ratio = 0.68/(1*0.68) = 1
    
    # It seems like with discrete autoconvolution and sum h_i = 1, the ratio is always 1!
    # Because autoconv at lag 0 = sum h_i^2 = L2^2, and sup >= L2^2, so
    # ratio = L2^2/(L1*sup) <= L2^2/(L1*L2^2) = 1/L1 = 1.
    # And equality when sup = L2^2, i.e., when the autoconvolution peak is at lag 0.
    
    # So the maximum is 1, achieved by any configuration where the autoconvolution
    # doesn't exceed its lag-0 value. For nonnegative h, this is always true!
    
    # So with the discrete interpretation and L1=1 constraint, the answer is always 1.
    
    # This means I must be misunderstanding the problem. Let me try the continuous
    # interpretation without any constraint, and see what gives a finite maximum.
    
    # Perhaps the ratio is: ||f||_2^2 / (||f||_1 * ||f*f||_inf) where f is the
    # piecewise constant function, and there's no constraint. But as we saw, this
    # is unbounded.
    
    # Unless... the "autoconvolution" is normalized differently, or it's the
    # autocorrelation (not convolution).
    
    # Autocorrelation: R(t) = integral f(s) f(s+t) ds
    # For nonnegative f, R(0) = ||f||_2^2, and |R(t)| <= R(0) by Cauchy-Schwarz.
    # So sup|R| = R(0) = ||f||_2^2.
    # Ratio = ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1/||f||_1.
    # Still unbounded.
    
    # I'm stuck. Let me just use the numerical approach and optimize whatever
    # metric gives a reasonable result. Let me try the discrete interpretation
    # without the L1=1 constraint, and see if the optimizer finds something.
    
    # Actually, let me just try a few candidate configurations and see which gives
    # the best score according to the evaluator's metric.
    
    # Given the inspiration says "Sparse support optimization finds configurations
    # (like geometric decays) where L2^2 is large relative to the autoconvolution peak,
    # achieving ratios > 1.25 vs seed 1", let me try geometric decays.
    
    # Let me try: h_i = r^i for i=0..49, with various r.
    
    best_h = None
    best_ratio = 0.0
    
    # Try various patterns
    candidates = []
    
    # Geometric decay
    for r in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        h = [r**i for i in range(N)]
        candidates.append(h)
    
    # Geometric decay from center
    for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
        h = [r**abs(i - N//2) for i in range(N)]
        candidates.append(h)
    
    # Single spike
    for pos in [0, N//2, N-1]:
        h = [0.0]*N
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes
    for pos1 in [0, N//4, N//2]:
        for pos2 in [N//4, N//2, 3*N//4, N-1]:
            if pos1 != pos2:
                h = [0.0]*N
                h[pos1] = 1.0
                h[pos2] = 1.0
                candidates.append(h)
    
    # Uniform (seed)
    candidates.append([1.0]*N)
    
    # Evaluate all candidates
    for h in candidates:
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h[:]
    
    # Try local optimization from the best candidate
    if best_h is not None and _elapsed() < _DEADLINE - 60:
        try:
            from scipy.optimize import minimize
            x0 = np.array(best_h, dtype=np.float64)
            
            def _neg_ratio_opt(x):
                h = np.maximum(x, 0)  # Enforce nonnegativity
                return -_ratio(h)
            
            result = minimize(_neg_ratio_opt, x0, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < -best_ratio and _elapsed() < _DEADLINE - 30:
                best_h = np.maximum(result.x, 0).tolist()
                best_ratio = -result.fun
        except Exception:
            pass
    
    # Try differential evolution for broader search
    if _elapsed() < _DEADLINE - 40:
        try:
            bounds = [(0, 10)] * N
            de_result = differential_evolution(_neg_ratio_opt, bounds, 
                                               maxiter=50, popsize=10,
                                               seed=42, tol=1e-6,
                                               polish=True)
            if de_result.fun < -best_ratio:
                best_h = np.maximum(de_result.x, 0).tolist()
                best_ratio = -de_result.fun
        except Exception:
            pass
    
    if best_h is None:
        best_h = [1.0] * N
    
    # Ensure nonnegativity
    return [max(0.0, float(x)) for x in best_h]
# EVOLVE-BLOCK-END
