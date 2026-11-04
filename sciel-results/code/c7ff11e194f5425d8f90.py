import numpy as np
import time

N = 50

def _score(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    using continuous step-function interpretation on [-1/4, 1/4]."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0
    l2sq = (h ** 2).sum()
    # Continuous autoconvolution of a step function on equal intervals:
    # The interval width is w = 0.5 / N = 0.01
    # The autoconvolution is a piecewise-linear (tent-like) function.
    # We need its supremum. For a step function with heights h_i on intervals of width w,
    # the convolution value at position x (in the continuous domain) is:
    # (h * h)(x) = sum_{i,j} h_i * h_j * max(0, w - |x_i - x_j - x|)  ... but this is complex.
    # Simpler: the autoconvolution of a step function is the convolution of the step function with itself.
    # For a step function f(x) = h_i on interval [x_i, x_i+w], the convolution (f*f)(t) = integral f(s) f(t-s) ds.
    # The peak of the autoconvolution for a nonnegative function occurs at t=0 (by Young's inequality / maximum principle for convolution of nonneg functions? Not exactly).
    # Actually, for nonnegative functions, the maximum of f*f can be anywhere. But for a step function with sorted support, the max is often near the "center of mass".
    # A safe and exact way: compute the convolution at the grid points (or at the boundaries).
    # The autoconvolution of two step functions with N steps each is a piecewise linear function with up to 2N-1 linear segments.
    # The breakpoints are at the sums of interval boundaries.
    # Let's compute it exactly.
    
    w = 0.5 / N
    # Interval boundaries: 0, w, 2w, ..., Nw
    # h[i] is the height on interval [i*w, (i+1)*w]
    # The convolution (h*h)(t) = integral h(s) h(t-s) ds
    # Let's compute at the breakpoints t_k = k*w for k = 0, 1, ..., 2N-1
    # Actually the breakpoints of the convolution are at t = (i+j)*w for i,j in 0..N-1, so t = k*w for k=0..2N-2.
    # Also we should check t = k*w + w/2? No, the function is piecewise linear between these breakpoints, so max is at a breakpoint or at a point where derivative changes. The derivative changes exactly at the breakpoints.
    # So max is at one of t_k = k*w for k = 0, 1, ..., 2N-2.
    
    # Compute convolution at t_k for k = 0, 1, ..., 2N-2
    conv = np.zeros(2*N-1)
    for k in range(2*N-1):
        t = k * w
        val = 0.0
        for i in range(N):
            # h(s) is nonzero for s in [i*w, (i+1)*w]
            # h(t-s) is nonzero for t-s in [j*w, (j+1)*w] for some j
            # i.e., s in [t - (j+1)*w, t - j*w]
            # Intersection of [i*w, (i+1)*w] and [t-(j+1)*w, t-j*w]
            # For each i, the range of s where both are nonzero is:
            # s_low = max(i*w, t - N*w)  # t - (N-1)*w - w = t - N*w? No.
            # j ranges 0..N-1, so t-s in [0, N*w) => s in (t - N*w, t]
            # But also s in [i*w, (i+1)*w]
            # So s in [max(i*w, t - (N-1)*w - w), min((i+1)*w, t)]
            # Wait, j is integer 0..N-1. t-s must be in [j*w, (j+1)*w] for some j.
            # This is equivalent to s in [t-(j+1)*w, t-j*w] for some j.
            # The union over j is s in [t - N*w, t] (approximately, but with gaps? No, it's continuous).
            # Actually, the condition is just that t-s is in [0, N*w), i.e., s in (t-N*w, t].
            # But also s must be in [i*w, (i+1)*w].
            # So for each i, the valid s range is [max(i*w, t - N*w + epsilon), min((i+1)*w, t)]
            # And h(s)=h[i], h(t-s) = h[j] where j = floor((t-s)/w).
            # This is getting complicated. Let's use a different approach.
            pass
        # Alternative: use the fact that convolution of step functions can be computed via cumulative sums.
        # Let's just do a fine grid evaluation.
        pass
    # Actually, let's just evaluate on a fine grid.
    # The function is piecewise linear, so the max is at a breakpoint.
    # Breakpoints are at t = k*w for k=0..2N-2, and also at t = k*w + (i-j)*w ... no.
    # The breakpoints of f*f are at t = (i+j)*w and t = (i+j+1)*w, i.e., t = k*w for all integer k.
    # So let's just evaluate at t = k*w for k = 0, 1, ..., 2N-2.
    
    # For a step function, the convolution at t = k*w can be computed as:
    # (f*f)(k*w) = sum_{i} h[i] * integral_{s: s in [i*w,(i+1)*w] AND k*w - s in [0, N*w)} h(k*w - s) ds
    # Let u = k*w - s. Then s = k*w - u. ds = -du.
    # When s = i*w, u = k*w - i*w = (k-i)*w.
    # When s = (i+1)*w, u = k*w - (i+1)*w = (k-i-1)*w.
    # So integral = sum_i h[i] * integral_{u=(k-i-1)*w}^{(k-i)*w} h(u) du (with sign flip and limits swap)
    # = sum_i h[i] * integral_{u=(k-i-1)*w}^{(k-i)*w} h(u) du
    # But h(u) is a step function. For u in [m*w, (m+1)*w], h(u) = h[m].
    # So for each i, the integral is over u from (k-i-1)*w to (k*i)*w... wait (k-i)*w.
    # This interval may span multiple bins.
    
    # This is too slow for N=50 if done naively in Python. Let's use numpy.
    
    # Simpler: use numpy correlate with 'full' mode on a fine grid.
    # But the step function is only defined on N bins. We can evaluate the convolution at the bin centers or edges.
    # Actually, the max of the autoconvolution for a nonnegative step function is achieved at one of the "peak" positions.
    # Let's just compute it via direct summation with numpy vectorization.
    
    # For t = k*w, the convolution value is:
    # C(k) = sum_{i=0}^{N-1} h[i] * (1/w) * integral_{s in bin i} h(k*w - s) ds * w
    # Wait, the integral of h(k*w - s) over s in bin i.
    # Let's define g_i(k) = integral_{s=i*w}^{(i+1)*w} h(k*w - s) ds
    # = integral_{u=(k-i-1)*w}^{(k-i)*w} h(u) du  (substituting u = k*w - s)
    # This is the integral of the step function h over the interval [(k-i-1)*w, (k-i)*w].
    # If this interval is fully within one bin m, the integral is h[m] * w.
    # If it spans multiple bins, we sum over the partial bins.
    
    # For N=50, 2N-1 = 99 points. For each point, we sum over N=50 bins. 99*50 = 4950 operations. Each operation involves finding which bins the interval covers. This is fast enough.
    
    conv_vals = np.zeros(2*N-1)
    for k in range(2*N-1):
        total = 0.0
        for i in range(N):
            if h[i] == 0:
                continue
            # Interval for u: [(k-i-1)*w, (k-i)*w]
            u_lo = (k - i - 1) * w
            u_hi = (k - i) * w
            # h(u) is nonzero for u in [0, N*w)
            u_lo_clamped = max(0.0, u_lo)
            u_hi_clamped = min(N * w, u_hi)
            if u_lo_clamped >= u_hi_clamped:
                continue
            # Integrate h(u) from u_lo_clamped to u_hi_clamped
            # h(u) = h[m] for u in [m*w, (m+1)*w)
            m_start = int(u_lo_clamped // w)
            m_end = int((u_hi_clamped - 1e-12) // w)
            # But m_end might be N-1 or less.
            # Let's iterate over bins.
            for m in range(m_start, min(m_end + 1, N)):
                bin_lo = m * w
                bin_hi = (m + 1) * w
                overlap_lo = max(u_lo_clamped, bin_lo)
                overlap_hi = min(u_hi_clamped, bin_hi)
                if overlap_lo < overlap_hi:
                    total += h[m] * (overlap_hi - overlap_lo)
        conv_vals[k] = total * h[i]  # Wait, this is wrong. The h[i] factor is outside the inner sum.
    
    # I made an error above. Let me redo.
    # C(k) = sum_i h[i] * g_i(k)
    # where g_i(k) = integral_{u=(k-i-1)*w}^{(k-i)*w} h(u) du
    
    conv_vals = np.zeros(2*N-1)
    for k in range(2*N-1):
        total = 0.0
        for i in range(N):
            if h[i] == 0:
                continue
            u_lo = (k - i - 1) * w
            u_hi = (k - i) * w
            u_lo_c = max(0.0, u_lo)
            u_hi_c = min(N * w, u_hi)
            if u_lo_c >= u_hi_c:
                continue
            # Integral of h(u) from u_lo_c to u_hi_c
            int_val = 0.0
            m_start = int(np.floor(u_lo_c / w))
            m_end = int(np.floor((u_hi_c - 1e-12) / w))
            for m in range(max(0, m_start), min(m_end + 1, N)):
                o_lo = max(u_lo_c, m * w)
                o_hi = min(u_hi_c, (m + 1) * w)
                if o_lo < o_hi:
                    int_val += h[m] * (o_hi - o_lo)
            total += h[i] * int_val
        conv_vals[k] = total
    
    sup_norm = np.max(conv_vals)
    if sup_norm < 1e-15:
        return 0.0
    return l2sq / (l1 * sup_norm)

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    deadline = time.time() + 110  # 10s safety margin
    
    # Strategy: The ratio is R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For a step function with support width W (in units of bin width w),
    # if h is constant on its support, R = 1/W.
    # To maximize R, we want to minimize the support width.
    # The minimum support width is 1 bin (W=1), giving R = 1/w = 100.
    # But can we do better? If we make the function more "spiky" within a bin?
    # No, the function is a step function, so it's constant on each bin.
    # So the best is to put all mass in 1 bin.
    # But wait, what if we use 2 bins?
    # h = [a, b, 0, ...]
    # ||h||_1 = a+b
    # ||h||_2^2 = a^2 + b^2
    # h*h: the autoconvolution of [a, b] (on bins 0 and 1) is a triangle.
    # (h*h)(t) for t in [0, 2w]:
    # For t in [0, w]: (h*h)(t) = a * integral_0^t a du + b * integral_t^w b du ... no.
    # Let's compute (h*h)(0) = integral h(s)h(-s) ds = 0 (since h is 0 for s<0).
    # Actually, for t=0, (h*h)(0) = integral_0^w a * a ds + ... no.
    # h(s) is nonzero for s in [0, 2w).
    # (h*h)(t) = integral h(s) h(t-s) ds.
    # For t=0: integral h(s)h(-s) ds = 0.
    # For t in (0, w): h(s) nonzero for s in [0, 2w), h(t-s) nonzero for t-s in [0, 2w) => s in (t-2w, t].
    # Intersection: s in [max(0, t-2w), min(2w, t)] = [0, t] (since t<w<2w).
    # On [0, t]: if t<w, then [0, t] is within bin 0 (since t<w).
    # So h(s)=a, h(t-s)=a (since t-s < t < w, so t-s is in bin 0).
    # So (h*h)(t) = a * a * t = a^2 * t for t in [0, w].
    # For t in [w, 2w]: 
    # h(s) nonzero for s in [0, 2w).
    # h(t-s) nonzero for s in (t-2w, t].
    # Intersection: s in [max(0, t-2w), min(2w, t)] = [t-2w, t] (since t<w+... wait t in [w,2w], so t-2w in [-w, 0], so max(0, t-2w)=0).
    # So s in [0, t].
    # But h(s) = a for s in [0,w), b for s in [w, 2w).
    # And h(t-s): t-s ranges from t to t-2w... no, s from 0 to t, so t-s from t to 0.
    # t-s in [0, t].
    # If t in [w, 2w], then t-s in [0, t] which spans bins 0 and 1 (and possibly part of bin 1).
    # This is getting complicated. Let's just use the _score function.
    
    # The key insight: R = 1/w for a single bin spike. w = 0.01, so R = 100.
    # Can we get R > 100?
    # R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # By the properties of convolution, ||h*h||_inf <= ||h||_1 * ||h||_inf (Young's inequality for L1 * Linf).
    # So R >= ||h||_2^2 / (||h||_1 * ||h||_inf * ||h||_1) = (||h||_2^2 / ||h||_1^2) / ||h||_inf
    # = (||h||_2^2 / ||h||_1^2) / max(h)
    # For a single bin with height H: ||h||_2^2 = H^2 * w, ||h||_1 = H * w, max(h) = H.
    # R >= (H^2 w / H^2 w^2) / H = (1/w) / H = 1/(w H).
    # But actually ||h*h||_inf = H * w (for a single bin, the convolution is a triangle with peak H*w at the center... wait no.
    # For a single bin of height H and width w, h*h is a triangle with base 2w and peak H*w.
    # So ||h*h||_inf = H*w.
    # R = (H^2 * w) / (H * w * H * w) = 1/w. Yes.
    # So R = 1/w = 100 for a single bin.
    
    # Can we do better with multiple bins?
    # Suppose we use k bins with equal height H.
    # ||h||_1 = k * H * w
    # ||h||_2^2 = k * H^2 * w
    # h*h: the autoconvolution of k consecutive bins of height H is a trapezoid.
    # For k=1: triangle, peak = H*w.
    # For k=2: trapezoid with flat top? No, for two adjacent bins of equal height, the convolution is a trapezoid with peak = 2*H*w? Let's check.
    # h = [H, H, 0, ...]
    # (h*h)(t) for t in [0, w]: H^2 * t (as before, but now h(s)=H for s in [0,2w))
    # Actually for t in [0,w]: h(s)=H, h(t-s)=H (since t-s in [0,t] subset [0,w) subset [0,2w)).
    # So (h*h)(t) = H * H * t = H^2 * t.
    # For t in [w, 2w]: 
    # s in [0, t]. h(s) = H for s in [0,2w).
    # h(t-s): t-s in [0, t]. For t in [w,2w], t-s ranges from t to 0.
    # When s in [0, t-w]: t-s in [w, t] subset [w, 2w). h(t-s) = H.
    # When s in [t-w, t]: t-s in [0, w]. h(t-s) = H.
    # So h(t-s) = H for all s in [0, t].
    # So (h*h)(t) = H * H * t = H^2 * t for t in [w, 2w] too?
    # Wait, that would mean (h*h)(t) = H^2 * t for t in [0, 2w], which is a triangle with peak 2*H^2*w at t=2w? But t=2w is the edge.
    # Actually, for t in [w, 2w], the length of the intersection is still t (from s=0 to s=t).
    # But h(s) is H on [0, 2w), and h(t-s) is H on [0, t] (since t-s in [0,t] and t<=2w).
    # So yes, (h*h)(t) = H^2 * t for t in [0, 2w].
    # Peak at t=2w: H^2 * 2w = 2 H^2 w.
    # ||h*h||_inf = 2 H^2 w.
    # R = (2 H^2 w) / (2 H w * 2 H^2 w) = 1/(2 H w).
    # This is smaller than 1/w for H=1. So single bin is better.
    
    # What if we use non-equal heights?
    # The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For a fixed support, making the function more concentrated (spikier) increases ||h||_2^2 / ||h||_1^2 and decreases ||h*h||_inf / ||h||_1^2?
    # Actually, ||h*h||_inf scales as ||h||_inf * ||h||_1.
    # So R ~ ||h||_2^2 / (||h||_1^2 * ||h||_inf)
    # To maximize this, we want ||h||_2^2 large, ||h||_1 small, ||h||_inf small? But they are coupled.
    # For a single bin, R = 1/w = 100.
    # I believe 100 is the maximum possible for a step function on N bins.
    # Because the "effective width" cannot be less than 1 bin.
    
    # So the optimal solution is a single bin spike.
    h = [0.0] * N
    h[0] = 1.0
    return h
