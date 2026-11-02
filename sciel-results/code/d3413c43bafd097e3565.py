import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    num_points = 2 * n - 1
    
    # The autoconvolution of a step function f(x) = h_i for x in [i*DELTA, (i+1)*DELTA)
    # is (f*f)(t) = integral f(x) f(t-x) dx
    # At t = k*DELTA, this is DELTA * sum over pairs where the overlap is non-zero
    # More precisely, for each pair (i, j), the contribution to the integral at lag k
    # is h[i]*h[j] * overlap_length(i, j, k) * DELTA
    # where overlap_length(i, j, k) is the length of intersection of [i, i+1] and [k-j, k-j+1]
    
    # Vectorized computation
    # For lag k, we need sum_{i,j} h[i]*h[j] * max(0, min(i+1, k-j+1) - max(i, k-j))
    
    ff_vals = np.zeros(num_points)
    
    # Precompute for each k, the valid (i, j) pairs and their overlaps
    # For a given k, i ranges over [max(0, k-(n-1)), min(n-1, k)]
    # For a given i, j ranges over [max(0, k-i-1+1), min(n-1, k-i+1-1)] = [max(0, k-i), min(n-1, k-i)]
    # Wait, let me re-derive.
    # f(x) = h[i] for x in [i*DELTA, (i+1)*DELTA)
    # (f*f)(t) = integral_0^t f(x) f(t-x) dx (for t >= 0, and by symmetry for t < 0)
    # Actually f*f(t) = integral_{-inf}^{inf} f(x) f(t-x) dx
    # Since f is supported on [0, N*DELTA), for t in [0, 2*N*DELTA),
    # (f*f)(t) = integral_0^{min(t, N*DELTA)} f(x) f(t-x) dx
    # 
    # Let t = k * DELTA. Then:
    # (f*f)(k*DELTA) = DELTA * sum_{i=0}^{n-1} h[i] * f((k-i)*DELTA)
    # where f((k-i)*DELTA) is the value of f at point (k-i)*DELTA
    # But f is a step function, so f(x) = h[j] for x in [j*DELTA, (j+1)*DELTA)
    # So f((k-i)*DELTA) = h[j] where j = floor((k-i)/1) = k-i, if 0 <= k-i <= n-1
    # 
    # Wait, this is for point evaluation, but the integral is over an interval.
    # Let me be more careful.
    # (f*f)(t) = integral f(x) f(t-x) dx
    # For t = k*DELTA, x ranges over [0, min(t, N*DELTA)]
    # f(x) = h[i] for x in [i*DELTA, (i+1)*DELTA), i = 0, ..., n-1
    # f(t-x) = f(k*DELTA - x) = h[j] for k*DELTA - x in [j*DELTA, (j+1)*DELTA)
    # i.e., x in [(k-j-1)*DELTA, (k-j)*DELTA), j = 0, ..., n-1
    # 
    # So for each i, the range of x where f(x) = h[i] is [i*DELTA, (i+1)*DELTA)
    # And f(t-x) = h[j] when x in [(k-j-1)*DELTA, (k-j)*DELTA)
    # The overlap of [i, i+1] and [k-j-1, k-j] (in units of DELTA) is:
    # max(0, min(i+1, k-j) - max(i, k-j-1))
    # 
    # So (f*f)(k*DELTA) = DELTA * sum_{i=0}^{n-1} sum_{j=0}^{n-1} h[i]*h[j] * max(0, min(i+1, k-j) - max(i, k-j-1))
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i)
            j_max = min(n - 1, k - i)
            if j_min > j_max:
                continue
            # For this i, j can only be k-i (a single value) if it's in range
            j = k - i
            if 0 <= j < n:
                # Overlap of [i, i+1] and [j, j+1] shifted... 
                # Actually, the overlap of [i, i+1] and [k-j-1, k-j] = [i-1+1, i] wait
                # k-j = i, so [k-j-1, k-j] = [i-1, i]
                # Overlap of [i, i+1] and [i-1, i] is just the point i, length 0
                # Hmm, that can't be right.
                
                # Let me re-derive more carefully.
                # f(t-x) where t = k*DELTA, x in [i*DELTA, (i+1)*DELTA)
                # t-x in [k*DELTA - (i+1)*DELTA, k*DELTA - i*DELTA) = [(k-i-1)*DELTA, (k-i)*DELTA)
                # So f(t-x) = h[j] where (k-i-1) <= (t-x)/DELTA < (k-i)
                # i.e., j = k-i-1 if k-i-1 >= 0, or we need to check which step it falls in
                # (t-x)/DELTA ranges over [k-i-1, k-i)
                # So f(t-x) = h[k-i-1] if k-i-1 >= 0 and k-i-1 < n
                # But also, (t-x)/DELTA could be in [k-i-1, k-i), which is a single interval
                # So the value is h[k-i-1] if 0 <= k-i-1 < n
                # 
                # Wait, but this is a pointwise argument. The integral over x in [i*DELTA, (i+1)*DELTA)
                # of f(x) f(t-x) dx = h[i] * integral_{i*DELTA}^{(i+1)*DELTA} f(t-x) dx
                # Let u = t-x, du = -dx, so integral = h[i] * integral_{t-(i+1)*DELTA}^{t-i*DELTA} f(u) du
                # = h[i] * integral_{(k-i-1)*DELTA}^{(k-i)*DELTA} f(u) du
                # 
                # Now f(u) = h[j] for u in [j*DELTA, (j+1)*DELTA)
                # The integral of f(u) over [(k-i-1)*DELTA, (k-i)*DELTA) is:
                # If k-i-1 >= 0 and k-i-1 < n: this is h[k-i-1] * DELTA (since the interval has length DELTA)
                # But wait, the interval [(k-i-1)*DELTA, (k-i)*DELTA) might span multiple steps?
                # No, it has length exactly DELTA, so it's within a single step or spans the boundary.
                # Actually, [(k-i-1)*DELTA, (k-i)*DELTA) is exactly the interval for step j = k-i-1.
                # So the integral is h[k-i-1] * DELTA if 0 <= k-i-1 < n, else 0.
                # 
                # Hmm, but that would make the autoconvolution very simple:
                # (f*f)(k*DELTA) = DELTA * sum_{i} h[i] * h[k-i-1] * DELTA (if valid)
                # = DELTA^2 * sum_{i} h[i] * h[k-i-1]
                # which is just DELTA^2 times the discrete cross-correlation.
                # 
                # But that doesn't account for the overlap when the interval spans two steps.
                # Wait, the interval [(k-i-1)*DELTA, (k-i)*DELTA) has length DELTA and is exactly one step's worth.
                # So it doesn't span two steps. The value of f(u) is constant on this interval.
                # 
                # So actually, (f*f)(k*DELTA) = DELTA^2 * sum_{i: 0<=i<n, 0<=k-i-1<n} h[i] * h[k-i-1]
                # 
                # Let me verify with a simple example. N=2, h=[1,1], DELTA=1/2
                # f(x) = 1 for x in [0,1)
                # (f*f)(t) = integral_0^t f(x) f(t-x) dx for t in [0,2]
                # For t in [0,1]: (f*f)(t) = t
                # For t in [1,2]: (f*f)(t) = 2-t
                # 
                # At t=0: (f*f)(0) = 0. Formula: DELTA^2 * sum h[i] h[0-i-1] = 0 (no valid i). Correct.
                # At t=1/2 (k=1): (f*f)(1/2) = 1/2. Formula: DELTA^2 * sum_{i} h[i] h[1-i-1] = DELTA^2 * h[0]*h[0] = (1/4)*1 = 1/4. 
                # But the true value is 1/2. So my formula is wrong.
                # 
                # The issue: (f*f)(t) for t=1/2 is integral_0^{1/2} 1*1 dx = 1/2.
                # My formula gave DELTA^2 * h[0]*h[0] = 1/4. Off by a factor of 2.
                # 
                # I think the issue is that I'm evaluating at discrete points, but the integral is continuous.
                # The correct formula for (f*f)(t) at t = k*DELTA is not just a single term.
                # 
                # Let me just compute it directly for each k.
                # (f*f)(k*DELTA) = integral_0^{min(k*DELTA, N*DELTA)} f(x) f(k*DELTA - x) dx
                # 
                # This integral can be computed by breaking it into intervals where f(x) and f(k*DELTA-x) are constant.
                # f(x) = h[i] for x in [i*DELTA, (i+1)*DELTA)
                # f(k*DELTA - x) = h[j] for k*DELTA - x in [j*DELTA, (j+1)*DELTA), i.e., x in [(k-j-1)*DELTA, (k-j)*DELTA)
                # 
                # So the breakpoints are at x = i*DELTA and x = (k-j)*DELTA for various i, j.
                # 
                # For a given k, the relevant x range is [0, min(k, n)*DELTA)
                # The breakpoints within this range are at x = m*DELTA for m = 0, 1, ..., min(k, n)
                # 
                # On each subinterval [m*DELTA, (m+1)*DELTA) within [0, min(k,n)*DELTA),
                # f(x) = h[m] (if m < n)
                # f(k*DELTA - x): k*DELTA - x ranges over [k*DELTA - (m+1)*DELTA, k*DELTA - m*DELTA) = [(k-m-1)*DELTA, (k-m)*DELTA)
                # So f(k*DELTA - x) = h[k-m-1] if 0 <= k-m-1 < n
                # 
                # So the integral over [m*DELTA, (m+1)*DELTA) is h[m] * h[k-m-1] * DELTA (if valid)
                # 
                # Therefore: (f*f)(k*DELTA) = DELTA * sum_{m=0}^{min(k-1, n-1)} h[m] * h[k-m-1] (with appropriate bounds)
                # 
                # Let me re-check with N=2, h=[1,1], k=1 (t=1/2):
                # (f*f)(1/2) = DELTA * sum_{m=0}^{0} h[m] * h[1-m-1] = DELTA * h[0]*h[0] = (1/2)*1 = 1/2. Correct!
                # 
                # For k=2 (t=1): (f*f)(1) = DELTA * sum_{m=0}^{1} h[m]*h[2-m-1] = DELTA*(h[0]*h[1] + h[1]*h[0]) = (1/2)*2 = 1. Correct!
                # 
                # For k=3 (t=3/2): (f*f)(3/2) = DELTA * sum_{m=0}^{2} h[m]*h[3-m-1]
                # m=0: h[0]*h[2] - invalid
                # m=1: h[1]*h[1] = 1
                # m=2: h[2]*h[0] - invalid
                # So = DELTA * 1 = 1/2. Correct!
                # 
                # Great, so the formula is:
                # (f*f)(k*DELTA) = DELTA * sum_{m} h[m] * h[k-1-m] where 0<=m<n and 0<=k-1-m<n
                
                pass
        
        # Using the derived formula
        for m in range(n):
            j = k - 1 - m
            if 0 <= j < n:
                total += h[m] * h[j]
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=100.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        for m in range(n):
            j = k - 1 - m
            if 0 <= j < n:
                total += h[m] * h[j]
        ff_vals[k] = total * DELTA
    
    max_val = np.max(ff_vals)
    if max_val < 1e-15:
        return 0.0
    # Smooth max via log-sum-exp
    shifted = (ff_vals - max_val) * beta
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _smooth_ratio(h, beta=100.0):
    """Compute smooth ratio."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, beta)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_mian_chowla(n, max_size):
    """Generate a Mian-Chowla (Sidon) set."""
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Mian-Chowla sets of various sizes
    for size in [8, 10, 12, 14, 16, 18]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
        
    # 2. Random sparse vectors
    for _ in range(25):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(5, 18)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.5, 1.5, num_ones)
        initializations.append(h_sp)
        
    # 3. Random dense vectors
    for _ in range(10):
        h_rd = np.random.uniform(0, 1, N)
        initializations.append(h_rd)
        
    # 4. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    # 5. Every 3
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    # 6. Gaussian
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2/0.3))
    
    # 7. Concentrated at center
    h_cen = np.zeros(N)
    h_cen[N//2-3:N//2+4] = 1.0
    initializations.append(h_cen)
    
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Optimize using L-BFGS-B on the smooth surrogate
        def neg_smooth_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -_smooth_ratio(x / s, beta=50.0)
        
        try:
            result = minimize(neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                             bounds=[(0, None)] * N,
                             options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            continue
    
    # Final polish with Nelder-Mead on the exact ratio
    if time.time() - start_time < DEADLINE - 15:
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h.tolist()
