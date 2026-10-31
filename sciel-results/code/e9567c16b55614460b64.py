"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, divided into 50 intervals

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) for step function heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * DELTA
    l2_sq = np.sum(h ** 2) * DELTA
    
    if l1 < 1e-15:
        return -np.inf
    
    # Compute autoconvolution h*h
    # h is a step function: h(x) = h[i] for x in [x_i, x_{i+1})
    # (h*h)(t) = integral h(s) h(t-s) ds
    # For step functions on equal intervals, this is piecewise linear
    # We can compute it via FFT or direct convolution
    
    # The autoconvolution of a step function can be computed as:
    # (h*h)(t) = sum over overlaps
    # Using the fact that for discrete steps, the continuous convolution at 
    # integer multiples of DELTA gives us the key values
    
    # Actually, let's compute it properly.
    # h(x) = sum_i h[i] * 1_{[x_i, x_{i+1})}(x) where x_i = -1/4 + i*DELTA
    # (h*h)(t) = integral h(s) h(t-s) ds
    # = sum_i sum_j h[i] h[j] * length of intersection of [x_i, x_{i+1}) and [t - x_j, t - x_{j+1})
    
    # The intersection length of two intervals of width DELTA whose starts differ by d
    # is max(0, DELTA - |d|)
    
    # So (h*h)(t) = sum_i sum_j h[i] h[j] * max(0, DELTA - |t - (x_i - x_j)|)
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, for the convolution of two step functions:
    # (h*h)(t) = integral_{-inf}^{inf} h(s) h(t-s) ds
    # h(s) = h[i] for s in [x_i, x_{i+1})
    # h(t-s) = h[j] for t-s in [x_j, x_{j+1}) i.e., s in [t-x_{j+1}, t-x_j)
    
    # So the integral is sum_i sum_j h[i] h[j] * length([x_i, x_{i+1}) ∩ [t-x_{j+1}, t-x_j))
    
    # The x_i are equally spaced with spacing DELTA.
    # x_i = -1/4 + i * DELTA for i = 0, ..., N-1
    
    # Let's compute the autoconvolution on a fine grid and take the max.
    # The autoconvolution is piecewise linear with breakpoints at t = x_i - x_j for all i,j.
    # These are all multiples of DELTA.
    
    # t ranges from -1/2 to 1/2 (since h is supported on [-1/4, 1/4])
    # Breakpoints are at t = k * DELTA for k = -(N-1), ..., N-1
    
    # At t = k*DELTA, the value can be computed.
    # Between breakpoints, it's linear.
    
    # The maximum of a piecewise linear function occurs at a breakpoint or endpoint.
    # So we just need to evaluate at all breakpoints.
    
    # t_k = k * DELTA for k = -(N-1), ..., N-1
    # (h*h)(t_k) = sum_i sum_j h[i] h[j] * max(0, DELTA - |t_k - (x_i - x_j)|) ... 
    
    # Wait, I think I'm overcomplicating this. Let me use a different approach.
    
    # For a step function with values h[0..N-1] on intervals of width DELTA,
    # the autoconvolution at t=0 is:
    # (h*h)(0) = sum_i h[i]^2 * DELTA
    # because at t=0, the overlap of interval i with itself is DELTA.
    
    # But the supremum might be elsewhere. For nonnegative functions, 
    # (h*h)(t) is maximized at t=0 by the rearrangement inequality?
    # Actually no, that's not true in general.
    
    # Let me just compute it numerically on a fine grid.
    
    # Support of h: [-1/4, 1/4]
    # Support of h*h: [-1/2, 1/2]
    # Use a fine grid to find the max
    
    t_vals = np.linspace(-0.5, 0.5, 4*N + 1)
    
    # (h*h)(t) = integral h(s) h(t-s) ds
    # We can compute this by noting that for each pair (i,j),
    # the contribution is h[i]*h[j]*overlap_length(i, j, t)
    # where overlap_length is the length of intersection of [x_i, x_{i+1}) and [t-x_j, t-x_{j+1})
    
    # x_i = -0.25 + i * DELTA
    # Interval i: [x_i, x_i + DELTA)
    # Interval for h(t-s): s in [t - x_j - DELTA, t - x_j)
    
    # Overlap of [a, a+DELTA) and [b, b+DELTA) = max(0, DELTA - |a-b|)
    # Here a = x_i, b = t - x_j - DELTA... 
    
    # Hmm, let me just use FFT-based convolution for accuracy.
    
    # Pad h to handle the full convolution
    # The autoconvolution of a function on 50 intervals will have support on 99 intervals
    # (since 50 + 50 - 1 = 99)
    
    # Actually, for continuous step functions, the convolution is a piecewise linear 
    # function. Let me compute it directly.
    
    # x_i = -0.25 + i * DELTA for i in range(N)
    # (h*h)(t) = sum_i sum_j h[i] h[j] * max(0, DELTA - |t - (x_i - x_j)|)
    
    # Wait, that formula: the overlap of [x_i, x_i+DELTA) and [t-x_j-DELTA, t-x_j)
    # = max(0, min(x_i+DELTA, t-x_j) - max(x_i, t-x_j-DELTA))
    # = max(0, DELTA - |t - x_j - x_i|) ... let me verify
    # 
    # If t = x_i + x_j, then overlap = max(0, min(x_i+DELTA, x_i+x_j-x_j) - max(x_i, x_i+x_j-x_j-DELTA))
    # = max(0, min(x_i+DELTA, x_i) - max(x_i, x_i-DELTA))
    # = max(0, x_i - x_i) = 0. That's wrong.
    
    # Let me redo. h(s) = h[i] for s in [x_i, x_{i+1})
    # h(t-s) = h[j] for t-s in [x_j, x_{j+1}) iff s in [t-x_{j+1}, t-x_j)
    # 
    # Overlap of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j):
    # left = max(x_i, t - x_{j+1})
    # right = min(x_{i+1}, t - x_j)
    # overlap = max(0, right - left)
    
    # x_i = -0.25 + i*d, x_{i+1} = -0.25 + (i+1)*d where d=DELTA
    # t - x_{j+1} = t + 0.25 - (j+1)*d
    # t - x_j = t + 0.25 - j*d
    
    # left = max(-0.25 + i*d, t + 0.25 - (j+1)*d)
    # right = min(-0.25 + (i+1)*d, t + 0.25 - j*d)
    
    # This is getting complex. Let me just sample on a fine grid.
    
    # For N=50, the autoconvolution has breakpoints at t = k*DELTA for k=-49,...,49
    # So 99 breakpoints. The function is linear between them.
    # Max occurs at a breakpoint.
    
    # Let me compute (h*h) at t = k*DELTA for k = -49,...,49
    
    d = DELTA
    # x_i = -0.25 + i*d
    # At t = k*d:
    # (h*h)(k*d) = sum_i sum_j h[i] h[j] * max(0, min(-0.25+(i+1)*d, k*d+0.25-j*d) - max(-0.25+i*d, k*d+0.25-(j+1)*d))
    
    # Let me simplify. Let u = i, v = j, and t = k*d.
    # left = max(-0.25 + u*d, k*d + 0.25 - (v+1)*d)
    # right = min(-0.25 + (u+1)*d, k*d + 0.25 - v*d)
    
    # left = max(-0.25 + u*d, (k-v-1)*d + 0.25)
    # right = min(-0.25 + (u+1)*d, (k-v)*d + 0.25)
    
    # Let m = k - v. Then:
    # left = max(-0.25 + u*d, (m-1)*d + 0.25)
    # right = min(-0.25 + (u+1)*d, m*d + 0.25)
    
    # overlap = max(0, right - left)
    
    # This is still complex. Let me just do it numerically with a fine grid.
    
    # Actually, the simplest correct approach: use numpy to compute the convolution.
    # The step function can be represented, and its autoconvolution computed.
    
    # For a step function with values h on N intervals of width d,
    # the autoconvolution (h*h)(t) for t in [-N*d, N*d] can be computed as:
    # (h*h)(t) = d * sum_i h[i] * h_interp(t/d - i)
    # where h_interp is the linear interpolation... no, that's not right either.
    
    # Let me just use a dense evaluation.
    
    n_grid = 4 * N + 1
    t_grid = np.linspace(-0.5, 0.5, n_grid)
    
    # For each t, compute (h*h)(t) = integral h(s) h(t-s) ds
    # h(s) is nonzero only for s in [-0.25, 0.25]
    # h(t-s) is nonzero only for t-s in [-0.25, 0.25] i.e., s in [t-0.25, t+0.25]
    # So s in [max(-0.25, t-0.25), min(0.25, t+0.25)]
    
    # On this range, h(s) = h[floor((s+0.25)/d)] and h(t-s) = h[floor((t-s+0.25)/d)]
    
    # This is a Riemann sum. Let me compute it as a sum over pairs.
    
    # Actually, for step functions, the convolution is piecewise linear and can be 
    # computed exactly at the breakpoints.
    
    # The breakpoints of h*h are at t = x_i - x_j for all i,j, which are k*d for k=-(N-1),...,N-1
    # plus the endpoints of the support.
    
    # Let me evaluate at all breakpoints t_k = k*d for k = -N+1, ..., N-1
    
    max_val = 0.0
    for k in range(-N + 1, N):
        t = k * d
        # (h*h)(t) = sum_i sum_j h[i]*h[j] * overlap(i, j, t)
        # where overlap is computed as above
        val = 0.0
        for i in range(N):
            for j in range(N):
                # left = max(x_i, t - x_j - d) where x_j + d = x_{j+1}
                # Actually: h(t-s) nonzero when t-s in [x_j, x_j+d], i.e., s in [t-x_j-d, t-x_j]
                # h(s) nonzero when s in [x_i, x_i+d]
                # Overlap of [x_i, x_i+d] and [t-x_j-d, t-x_j]
                a = -0.25 + i * d
                b = -0.25 + j * d
                left = max(a, t - b - d)
                right = min(a + d, t - b)
                if right > left:
                    val += h[i] * h[j] * (right - left)
        if val > max_val:
            max_val = val
    
    if max_val < 1e-15:
        return -np.inf
    
    return l2_sq / (l1 * max_val)

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Try a few structured patterns
    patterns = []
    
    # Uniform
    patterns.append(np.ones(N))
    
    # Sparse: only a few nonzero
    for k in [5, 10, 15, 20, 25]:
        h = np.zeros(N)
        # Spread out
        idx = np.linspace(0, N-1, k).astype(int)
        h[idx] = 1.0
        patterns.append(h)
    
    # Sidon-like: positions with distinct pairwise differences
    # For small sets, use well-separated points
    for k in [8, 12, 16]:
        h = np.zeros(N)
        # Use a geometric spacing
        positions = []
        pos = 0
        gap = max(1, N // k)
        for i in range(k):
            if pos < N:
                positions.append(pos)
            pos += gap
        if len(positions) > 0:
            h[positions] = 1.0
            patterns.append(h)
    
    # Evaluate all patterns
    for h in patterns:
        if time.time() > deadline:
            break
        score = _compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Now try local optimization from the best pattern
    if time.time() < deadline - 30:
        def neg_ratio(h_flat):
            h = np.maximum(h_flat, 0)
            r = _compute_ratio(h)
            if r == -np.inf or np.isnan(r):
                return 1e10
            return -r
        
        # Normalize to sum=1 to reduce dimensionality
        h0 = best_h.copy()
        s = np.sum(h0)
        if s > 0:
            h0 = h0 / s
        
        result = minimize(neg_ratio, h0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if time.time() < deadline:
            h_opt = np.maximum(result.x, 0)
            score_opt = _compute_ratio(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    
    # Return as list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
