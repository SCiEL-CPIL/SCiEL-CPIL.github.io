"""
Second autocorrelation inequality - spectral design approach.

We maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf) for nonnegative step heights on
N=50 equal intervals covering [-1/4, 1/4].

Key insight: the ratio is scale-invariant, so we can fix ||h||_2 = 1 and minimize
||h||_1 * ||h*h||_inf. The autoconvolution peak is related to the maximum of the
power spectrum. We use gradient-based optimization with a smooth approximation
of the max (log-sum-exp) to make the objective differentiable.
"""
import numpy as np
import time

N = 50
DELTA = 1.0 / (2 * N)  # 0.01

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) for step function heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * DELTA
    l2_sq = np.sum(h ** 2) * DELTA
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return -np.inf
    
    # Compute autoconvolution exactly at all breakpoints
    # The autoconvolution of a step function on N intervals of width d is piecewise
    # linear with breakpoints at t = k*d for k = -(N-1), ..., N-1
    # The maximum occurs at one of these breakpoints.
    
    d = DELTA
    max_val = 0.0
    
    # For each lag k (from -(N-1) to N-1), compute (h*h)(k*d)
    # (h*h)(k*d) = sum_i sum_j h[i]*h[j] * overlap(i, j, k*d)
    # where overlap is the length of intersection of interval i and the shifted interval
    
    # More efficiently: for lag k, the overlap between interval i and interval j
    # when shifted by k*d is max(0, d - |i - j - k| * d) ... no, that's not right.
    
    # Let me think again. h(s) = h[i] for s in [x_i, x_{i+1}) where x_i = -1/4 + i*d
    # (h*h)(t) = integral h(s) h(t-s) ds
    # For t = k*d:
    # h(t-s) = h[j] when t-s in [x_j, x_{j+1}), i.e., s in [t-x_{j+1}, t-x_j)
    # = [k*d - x_{j+1}, k*d - x_j)
    # = [k*d - (-1/4+(j+1)*d), k*d - (-1/4+j*d))
    # = [k*d + 1/4 - (j+1)*d, k*d + 1/4 - j*d)
    # = [(k-j-1)*d + 1/4, (k-j)*d + 1/4)
    
    # h(s) = h[i] for s in [x_i, x_{i+1}) = [-1/4+i*d, -1/4+(i+1)*d)
    
    # Overlap of [-1/4+i*d, -1/4+(i+1)*d) and [(k-j-1)*d+1/4, (k-j)*d+1/4)
    
    # Let m = i - (k-j) = i + j - k
    # Interval 1: [-1/4+i*d, -1/4+(i+1)*d)
    # Interval 2: [(k-j-1)*d+1/4, (k-j)*d+1/4) = [-(i-m-1)*d+1/4, -(i-m)*d+1/4) ... this is getting messy.
    
    # Let me just use the direct formula. For two intervals of width d:
    # [a, a+d) and [b, b+d) have overlap max(0, d - |a-b|)
    
    # a = -1/4 + i*d
    # b = (k-j-1)*d + 1/4 = -1/4 + (k-j)*d + d - d/2 ... 
    # Actually b = (k-j-1)*d + 1/4
    # a - b = -1/4 + i*d - (k-j-1)*d - 1/4 = i*d - (k-j-1)*d - 1/2
    # = (i - k + j + 1)*d - 1/2
    # = (i + j - k + 1)*d - 1/2
    
    # Since d = 1/(2N) = 0.01, and 1/2 = 25*d
    # a - b = (i + j - k + 1)*d - 25*d = (i + j - k + 1 - 25)*d
    
    # So |a-b| = |i + j - k - 24| * d
    
    # Overlap = max(0, d - |i + j - k - 24| * d) = d * max(0, 1 - |i + j - k - 24|)
    
    # So (h*h)(k*d) = d * sum_i sum_j h[i]*h[j] * max(0, 1 - |i + j - k - 24|)
    
    # This means the overlap is nonzero only when |i + j - k - 24| < 1, i.e., i + j - k - 24 = 0
    # i.e., i + j = k + 24
    
    # Wait, that can't be right. Let me recheck.
    
    # a = -1/4 + i*d
    # b = (k-j-1)*d + 1/4
    
    # a - b = -1/4 + i*d - (k-j-1)*d - 1/4 = i*d - (k-j-1)*d - 1/2
    # = (i - k + j + 1)*d - 1/2
    
    # With d = 0.01, 1/2 = 50*d
    # a - b = (i - k + j + 1)*d - 50*d = (i + j - k - 49)*d
    
    # Hmm, let me recompute. 1/2 = 0.5, d = 0.01, so 1/2 = 50*d.
    
    # a - b = (i - k + j + 1)*d - 50*d = (i + j - k + 1 - 50)*d = (i + j - k - 49)*d
    
    # So |a-b| = |i + j - k - 49| * d
    # Overlap = max(0, d - |i + j - k - 49| * d) = d * max(0, 1 - |i + j - k - 49|)
    
    # This is nonzero only when i + j - k - 49 = 0, i.e., i + j = k + 49.
    
    # But that would mean the autoconvolution is zero for most lags, which seems wrong.
    
    # Let me verify with a simple case. N=2, d=0.25, x_0=-0.25, x_1=0
    # h = [1, 0]
    # (h*h)(0) = integral h(s)h(s) ds = integral_0^0.25 1*1 ds = 0.25 = d
    # Using formula: k=0, i=0, j=0: i+j-k-49 = -49, overlap = 0. Wrong!
    
    # I think I made an error. Let me redo.
    # For N=2: x_0 = -0.25, x_1 = 0, d = 0.25
    # h(s) = 1 for s in [-0.25, 0), 0 for s in [0, 0.25)
    # (h*h)(t) = integral h(s) h(t-s) ds
    # For t=0: integral h(s)h(s) ds = integral_{-0.25}^0 1*1 ds = 0.25
    # For t=0.25: integral h(s)h(0.25-s) ds
    # h(s) nonzero for s in [-0.25, 0)
    # h(0.25-s) nonzero for 0.25-s in [-0.25, 0) => s in (0.25, 0.5]
    # No overlap, so (h*h)(0.25) = 0
    
    # For t=-0.25: similar, no overlap.
    
    # So for N=2, h=[1,0], the autoconvolution is a triangle: peak at t=0 with value 0.25.
    
    # Let me recompute the overlap formula.
    # h(s) = h[i] for s in [x_i, x_{i+1})
    # h(t-s) = h[j] for t-s in [x_j, x_{j+1}) iff s in [t-x_{j+1}, t-x_j)
    
    # Overlap of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j)
    # = max(0, min(x_{i+1}, t-x_j) - max(x_i, t-x_{j+1}))
    
    # x_i = -0.25 + i*d, x_{i+1} = -0.25 + (i+1)*d
    # t-x_j = t + 0.25 - j*d
    # t-x_{j+1} = t + 0.25 - (j+1)*d
    
    # For t = k*d:
    # t-x_j = k*d + 0.25 - j*d = (k-j)*d + 0.25
    # t-x_{j+1} = (k-j-1)*d + 0.25
    
    # left = max(-0.25+i*d, (k-j-1)*d+0.25)
    # right = min(-0.25+(i+1)*d, (k-j)*d+0.25)
    
    # Let me define u = i, v = k-j
    # left = max(-0.25+u*d, (v-1)*d+0.25)
    # right = min(-0.25+(u+1)*d, v*d+0.25)
    
    # left = max(-0.25+u*d, v*d - d + 0.25)
    # right = min(-0.25+u*d+d, v*d+0.25)
    
    # For the overlap to be positive, we need right > left.
    
    # This is a standard interval overlap. The overlap length is:
    # max(0, min(A+d, B) - max(A, B-d)) where A = -0.25+u*d, B = v*d+0.25
    
    # Let me just compute it numerically for correctness.
    
    max_val = 0.0
    for k in range(-N + 1, N):
        t = k * d
        val = 0.0
        for i in range(N):
            for j in range(N):
                a = -0.25 + i * d
                b = -0.25 + j * d
                # h(t-s) nonzero when s in [t-b-d, t-b]
                left = max(a, t - b - d)
                right = min(a + d, t - b)
                if right > left:
                    val += h[i] * h[j] * (right - left)
        if val > max_val:
            max_val = val
    
    if max_val < 1e-15:
        return -np.inf
    
    return l2_sq / (l1 * max_val)


def _compute_ratio_fast(h):
    """Faster computation using the structure of the problem."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * DELTA
    l2_sq = np.sum(h ** 2) * DELTA
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return -np.inf
    
    d = DELTA
    max_val = 0.0
    
    # Vectorized computation for each lag k
    # For lag k, (h*h)(k*d) = sum_i sum_j h[i]*h[j]*overlap(i,j,k)
    
    # Precompute overlaps
    # For each k, the overlap between interval i and the shifted interval from j
    # is a function of i, j, k
    
    # Let me use a different approach: compute the convolution via FFT-like method
    # or just do it directly but more efficiently.
    
    # For N=50, the O(N^3) approach (50^3 = 125000 operations) should be fast enough.
    # But let's try to be smarter.
    
    # Actually, for each k, the sum over i,j can be reorganized.
    # Let s = i + j. Then for fixed s and k, we sum over i where j = s-i.
    # The overlap depends on i, j, k.
    
    # Let me just do the direct computation but with numpy vectorization over i for each (k,j)
    
    for k in range(-N + 1, N):
        t = k * d
        # For each j, compute the contribution from all i
        # overlap(i, j, t) = max(0, min(-0.25+(i+1)*d, t+0.25-j*d) - max(-0.25+i*d, t+0.25-(j+1)*d))
        
        val = 0.0
        for j in range(N):
            if h[j] == 0:
                continue
            # For all i:
            # a_i = -0.25 + i*d
            # left_i = max(a_i, t + 0.25 - (j+1)*d)
            # right_i = min(a_i + d, t + 0.25 - j*d)
            # overlap_i = max(0, right_i - left_i)
            
            t_j = t + 0.25 - j * d
            t_j1 = t + 0.25 - (j + 1) * d
            
            # a_i = -0.25 + i*d for i = 0,...,N-1
            a_vals = -0.25 + np.arange(N) * d
            
            left_vals = np.maximum(a_vals, t_j1)
            right_vals = np.minimum(a_vals + d, t_j)
            overlaps = np.maximum(0, right_vals - left_vals)
            
            val += h[j] * np.sum(h * overlaps)
        
        if val > max_val:
            max_val = val
    
    if max_val < 1e-15:
        return -np.inf
    
    return l2_sq / (l1 * max_val)


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    best_h = np.ones(N)
    best_score = _compute_ratio_fast(best_h)
    
    # Try structured initial patterns
    patterns = []
    patterns.append(np.ones(N))
    
    # Sparse patterns with Sidon-like properties
    for k in [3, 5, 8, 10, 15, 20]:
        h = np.zeros(N)
        if k <= N:
            # Use well-separated positions
            positions = np.linspace(0, N-1, k, dtype=int)
            h[positions] = 1.0
            patterns.append(h.copy())
    
    # Evaluate patterns
    for h in patterns:
        if time.time() > deadline - 20:
            break
        score = _compute_ratio_fast(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Gradient-based optimization
    # We maximize the ratio. Since it's scale-invariant, fix ||h||_2 = 1 and minimize l1 * max_conv.
    # Use log-sum-exp for smooth max approximation.
    
    def smooth_max_conv(h, temperature=0.01):
        """Compute a smooth approximation of the max of the autoconvolution."""
        d = DELTA
        conv_vals = np.zeros(2*N - 1)
        
        for k_idx, k in enumerate(range(-N + 1, N)):
            t = k * d
            val = 0.0
            for j in range(N):
                if h[j] == 0:
                    continue
                t_j = t + 0.25 - j * d
                t_j1 = t + 0.25 - (j + 1) * d
                a_vals = -0.25 + np.arange(N) * d
                left_vals = np.maximum(a_vals, t_j1)
                right_vals = np.minimum(a_vals + d, t_j)
                overlaps = np.maximum(0, right_vals - left_vals)
                val += h[j] * np.sum(h * overlaps)
            conv_vals[k_idx] = val
        
        # Log-sum-exp for smooth max
        max_cv = np.max(conv_vals)
        if max_cv < 1e-15:
            return 1e10
        log_sum_exp = max_cv + np.log(np.sum(np.exp((conv_vals - max_cv) / temperature))) * temperature
        return log_sum_exp
    
    def objective_and_grad(h_raw):
        """Objective: minimize l1 * smooth_max_conv, with h = softmax(h_raw) normalized."""
        # Convert to nonnegative heights
        h = np.exp(h_raw - np.max(h_raw))  # softmax-like for stability
        h = np.maximum(h, 0)
        
        # Normalize to ||h||_2 = 1
        norm = np.sqrt(np.sum(h**2))
        if norm < 1e-15:
            return 1e10, np.zeros(N)
        h = h / norm
        
        l1 = np.sum(h) * DELTA
        max_cv = smooth_max_conv(h)
        
        obj = l1 * max_cv
        return obj, h
    
    # Multi-start gradient descent
    remaining = deadline - time.time()
    if remaining > 10:
        # Try several random initializations
        np.random.seed(42)
        
        best_g = None
        best_obj = 1e10
        
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize
            if trial == 0:
                g = np.zeros(N)
            elif trial == 1:
                g = np.random.randn(N) * 0.5
            elif trial == 2:
                # Sparse init
                g = np.full(N, -5.0)
                positions = np.random.choice(N, size=10, replace=False)
                g[positions] = 0.0
            else:
                g = np.random.randn(N) * 1.0
            
            # Gradient descent with finite differences
            lr = 0.05
            for iteration in range(500):
                if time.time() > deadline - 15:
                    break
                
                obj, h = objective_and_grad(g)
                if obj < best_obj:
                    best_obj = obj
                    best_g = g.copy()
                
                # Compute gradient via finite differences
                grad = np.zeros(N)
                eps = 1e-4
                for i in range(N):
                    g_plus = g.copy()
                    g_plus[i] += eps
                    obj_plus, _ = objective_and_grad(g_plus)
                    grad[i] = (obj_plus - obj) / eps
                
                # Update
                g = g - lr * grad
                lr *= 0.99  # decay learning rate
        
        # Evaluate best found
        if best_g is not None:
            _, h_opt = objective_and_grad(best_g)
            score_opt = _compute_ratio_fast(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    
    # Also try the original best_h
    return [float(x) for x in best_h]
