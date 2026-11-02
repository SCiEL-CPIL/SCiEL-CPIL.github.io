import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 1.0 / N

def autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    # The autoconvolution (f*f)(t) = integral f(s)f(t-s)ds
    # For a step function with heights h[i] on [i*DELTA, (i+1)*DELTA],
    # we can compute this efficiently.
    
    # (f*f)(t) for t in [k*DELTA, (k+1)*DELTA) can be computed
    # But we need the sup over all t. The sup occurs at specific points.
    # Actually, for step functions, the autoconvolution is piecewise quadratic,
    # and the max occurs at the endpoints of the intervals or at the center.
    
    # Let's use a more direct approach:
    # (f*f)(t) = sum_{i,j} h[i]*h[j] * length of intersection of [i*DELTA, (i+1)*DELTA] and [t-(j+1)*DELTA, t-j*DELTA]
    
    # The support of f is [0, N*DELTA] = [0, 1]
    # So (f*f)(t) is nonzero for t in [0, 2]
    
    # For efficiency, let's compute at the critical points:
    # t = k*DELTA for k = 0, 1, ..., 2*(N-1)
    # and t = (k+0.5)*DELTA for k = 0, 1, ..., 2*(N-1)-1
    
    # Actually, the autoconvolution of a step function is a piecewise quadratic function.
    # The maximum occurs at one of the breakpoints or at the vertex of a parabolic segment.
    
    # Let's use a simpler approach: compute (f*f)(t) at many points and take the max.
    # But we need to be careful about the exact sup.
    
    # For step functions, the autoconvolution is:
    # (f*f)(t) = sum_{i,j} h[i]*h[j] * max(0, min(i+1, (t-j*DELTA)/DELTA + 1) - max(i, (t-j*DELTA)/DELTA)) * DELTA
    
    # Let's define g(t) = (f*f)(t)/DELTA^2 as a function of t/DELTA
    # Then g(u) where u = t/DELTA is:
    # g(u) = sum_{i,j} h[i]*h[j] * max(0, min(i+1, u-j+1) - max(i, u-j))
    
    # Let's compute this at u = 0, 0.5, 1, 1.5, ..., 2*(N-1), 2*(N-1)+0.5
    # Actually, the breakpoints are at u = integer values.
    
    # For a more efficient computation, let's use the fact that:
    # (f*f)(t) = integral_0^1 f(s)f(t-s)ds
    # = sum_i h[i] * integral_{i*DELTA}^{(i+1)*DELTA} f(t-s)ds
    # = sum_i h[i] * sum_j h[j] * length([i*DELTA, (i+1)*DELTA] ∩ [t-(j+1)*DELTA, t-j*DELTA])
    
    # Let's compute at the critical points. The function (f*f)(t) is piecewise quadratic
    # with breakpoints at t = k*DELTA for integer k.
    
    # For t = k*DELTA:
    # (f*f)(k*DELTA) = DELTA^2 * sum_{i,j} h[i]*h[j] * overlap(i, j, k)
    # where overlap(i, j, k) = max(0, min(i+1, k-j+1) - max(i, k-j))
    
    # Let's compute this for k = -(N-1), ..., N-1
    # Actually t ranges from 0 to 2, so k ranges from 0 to 2*(N-1) approximately.
    
    # Let's use a vectorized approach:
    # For each k (shift in units of DELTA), compute the sum
    
    # The autoconvolution at t = k*DELTA is:
    # sum_{i,j: |i-j| related to k} h[i]*h[j] * overlap_length
    
    # Let's just compute it directly but efficiently using numpy
    
    # Method: compute the full autoconvolution by convolution
    # f is a step function. We can represent it as a vector of values at midpoints
    # and use convolution, but we need to be careful.
    
    # Actually, let's use the following approach:
    # (f*f)(t) for t in [k*DELTA, (k+1)*DELTA) is a quadratic function.
    # The maximum on each interval occurs at the endpoints or at the vertex.
    
    # For simplicity and correctness, let's compute at the endpoints:
    # t = 0, DELTA, 2*DELTA, ..., 2*(N-1)*DELTA, 2*(N-1)*DELTA + DELTA
    
    # And also at the midpoints to catch any interior maxima.
    
    # Let's compute (f*f)(t) at t = m*DELTA/2 for m = 0, 1, ..., 4*(N-1)
    # This should be sufficient for step functions.
    
    # Actually, for a step function, the autoconvolution is a sum of "tent" functions
    # (triangular functions), so it's piecewise linear, not quadratic!
    # Wait, no. The autoconvolution of two step functions is piecewise quadratic.
    
    # Let me reconsider. f is a step function. f*f(t) = integral f(s)f(t-s)ds.
    # For fixed t, as we integrate, the product f(s)f(t-s) is piecewise constant,
    # so the integral is piecewise linear in t? No, it's piecewise quadratic.
    
    # Actually, let's just compute it numerically at enough points.
    
    # Number of points: let's use 4*N points to be safe.
    t_vals = np.linspace(0, 2.0, 4*N + 1)
    
    ff_vals = np.zeros(len(t_vals))
    
    for idx, t in enumerate(t_vals):
        total = 0.0
        for i in range(n):
            s_lo = i * DELTA
            s_hi = (i + 1) * DELTA
            # f(t-s) is nonzero when t-s is in [0, N*DELTA] = [0, 1]
            # so s is in [t-1, t]
            # overlap of [s_lo, s_hi] with [t-1, t]
            lo = max(s_lo, t - 1.0)
            hi = min(s_hi, t)
            if hi > lo:
                # For s in [lo, hi], f(s) = h[i], f(t-s) = h[j] where j = floor((t-s)/DELTA)
                # But (t-s)/DELTA varies over [(t-hi)/DELTA, (t-lo)/DELTA]
                # So we need to sum over j
                u_lo = (t - hi) / DELTA
                u_hi = (t - lo) / DELTA
                j_min = max(0, int(np.ceil(u_lo - 1e-12)))
                j_max = min(n - 1, int(np.floor(u_hi + 1e-12)))
                for j in range(j_min, j_max + 1):
                    # s range where (t-s)/DELTA is in [j, j+1]
                    s_lo_j = t - (j + 1) * DELTA
                    s_hi_j = t - j * DELTA
                    s_lo_jj = max(lo, s_lo_j)
                    s_hi_jj = min(hi, s_hi_j)
                    if s_hi_jj > s_lo_jj:
                        total += h[i] * h[j] * (s_hi_jj - s_lo_jj)
        ff_vals[idx] = total
    
    return np.max(ff_vals)

def ratio(h):
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N)
    best_score = ratio(best_h)
    
    np.random.seed(123)
    
    n_trials = 30
    
    for trial in range(n_trials):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if trial == 0:
            h0 = np.ones(N)
        elif trial == 1:
            # Triangular
            h0 = np.array([1.0 - abs(i - N//2) / (N//2) for i in range(N)])
            h0 = np.maximum(h0, 0.01)
        elif trial == 2:
            # Inverse triangular (peaked in middle)
            h0 = np.array([abs(i - N//2) / (N//2) + 0.1 for i in range(N)])
        elif trial == 3:
            # Cosine bell
            t = np.linspace(-1, 1, N)
            h0 = np.maximum(1 + 0.8*np.cos(np.pi*t), 0.05)
        elif trial == 4:
            # Two peaks
            t = np.linspace(-1, 1, N)
            h0 = np.maximum(1.0 + 1.5*np.exp(-((t-0.5)**2)/0.05) + 1.5*np.exp(-((t+0.5)**2)/0.05), 0.05)
        else:
            # Random
            h0 = np.random.uniform(0.1, 2.0, N)
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 > 0:
            h0 = h0 / s0
        
        def neg_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -ratio(x / s)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, None)] * N
        
        try:
            result = minimize(neg_ratio, h0.copy(), method='SLSQP',
                             bounds=bounds, constraints=constraints,
                             options={'maxiter': 300, 'ftol': 1e-10})
            
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if result.fun < -best_score + 1e-10:
                h_candidate = np.maximum(result.x, 0)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate = h_candidate / s
                    sc = ratio(h_candidate)
                    if sc > best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
        except Exception:
            continue
    
    return best_h.tolist()
