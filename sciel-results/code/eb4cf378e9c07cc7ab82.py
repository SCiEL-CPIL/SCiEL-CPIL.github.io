import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
# Interval width
w = (1/4 - (-1/4)) / N  # 0.5 / 600

def _autoconv_max(h):
    """Compute the maximum of the autoconvolution of the step function h.
    
    h is a length-N array of nonnegative heights.
    The step function is f(x) = h[i] for x in [-1/4 + i*w, -1/4 + (i+1)*w).
    
    The autoconvolution (f * f)(x) = integral f(y) f(x-y) dy.
    For step functions, this is a piecewise linear function.
    
    We compute it via discrete convolution:
    The continuous autoconvolution at sample points can be computed as:
    (f*f)(x_k) where x_k are the midpoints or breakpoints.
    
    Actually, for step functions on equal intervals, the autoconvolution is a 
    piecewise quadratic (or linear?) function. Let me think more carefully.
    
    f(x) = sum_i h[i] * 1_{[-1/4+iw, -1/4+(i+1)w)}(x)
    
    (f*f)(z) = integral f(y) f(z-y) dy
    
    This is a convolution of two step functions. The result is a piecewise 
    linear function (since convolution of two step functions is piecewise linear).
    
    The breakpoints occur at z = -1/4 + i*w for i = 0, 1, ..., 2N.
    So there are 2N+1 intervals for the autoconvolution.
    
    We can compute the autoconvolution values at the midpoints of these intervals,
    or better, compute the exact maximum.
    
    Actually, let's use the discrete approach. The autoconvolution of step functions
    can be computed as follows:
    
    Let g = f*f. The breakpoints are at t_j = -1/4 + j*w for j=0,...,2N.
    On each interval [t_j, t_{j+1}], g is linear (actually, it's the convolution
    of two piecewise constant functions, so it's piecewise linear with slope changes
    at the breakpoints).
    
    Wait, convolution of two step functions: each is piecewise constant. The 
    convolution is piecewise linear. The maximum occurs at one of the breakpoints
    or at a local maximum of the piecewise linear function, which would be at a breakpoint.
    
    Actually, for a piecewise linear function, the maximum on a closed interval occurs
    at an endpoint. So the maximum of g occurs at one of the breakpoints t_j.
    
    Let me compute g at the breakpoints t_j = -1/4 + j*w for j=0,...,2N.
    
    g(t_j) = integral f(y) f(t_j - y) dy
    
    f(y) is nonzero for y in [-1/4, 1/4], i.e., y in [-1/4 + i*w, -1/4 + (i+1)*w) for i=0..N-1.
    f(t_j - y) is nonzero for t_j - y in [-1/4, 1/4], i.e., y in [t_j - 1/4, t_j + 1/4].
    
    So the integral is over y in [-1/4, 1/4] ∩ [t_j - 1/4, t_j + 1/4].
    
    This is getting complex. Let me use a simpler numerical approach.
    
    Alternative: use FFT-based convolution. The step function f can be represented by
    its values h[0..N-1] on intervals of width w. The autoconvolution can be computed
    via:
    
    g(x) = integral f(y) f(x-y) dy
    
    If we sample f at the midpoints of the intervals, we get a discrete signal. But this
    isn't exact.
    
    Better approach: The autoconvolution of two step functions with the same interval 
    structure is a piecewise linear function. We can compute it exactly.
    
    Let me define the intervals. f is constant h[i] on I_i = [-1/4 + i*w, -1/4 + (i+1)*w).
    
    g(z) = sum_i sum_j h[i] h[j] * length(I_i ∩ (z - I_j))
    
    where z - I_j = {z - y : y in I_j} = [z - (-1/4 + (j+1)*w), z - (-1/4 + j*w)) = [z + 1/4 - (j+1)*w, z + 1/4 - j*w).
    
    The intersection length of I_i = [a_i, b_i) and [c_j(z), d_j(z)) where 
    a_i = -1/4 + i*w, b_i = -1/4 + (i+1)*w,
    c_j(z) = z + 1/4 - (j+1)*w, d_j(z) = z + 1/4 - j*w.
    
    This is O(N^2) per evaluation, which for N=600 is 360,000 operations. That might be 
    slow for many iterations.
    
    Let me use FFT instead. The autoconvolution of step functions can be computed as:
    
    Consider the continuous function f. Its autoconvolution g = f*f.
    
    We can compute g at the breakpoints using the fact that g is piecewise linear.
    
    Actually, let me just use a dense sampling approach with enough points to capture
    the maximum accurately.
    
    The breakpoints of g are at z = -1/4 + k*w for k = 0, 1, ..., 2N. There are 2N+1 
    breakpoints. The maximum of g occurs at one of these breakpoints (since g is piecewise
    linear).
    
    So I need to compute g at z_k = -1/4 + k*w for k=0,...,2N.
    
    g(z_k) = integral_{-1/4}^{1/4} f(y) f(z_k - y) dy
    
    Let me change variables. Let u = y + 1/4, so u in [0, 1/2] = [0, N*w].
    f(y) = h[floor(u/w)] for u in [i*w, (i+1)*w), i.e., h[i] when u in [i*w, (i+1)*w).
    
    z_k - y = z_k - u + 1/4 = -1/4 + k*w - u + 1/4 = k*w - u.
    f(z_k - y) = f(k*w - u).
    
    f(v) for v in [-1/4, 1/4]: let v = m*w - 1/4 + delta... this is getting messy.
    
    Let me just use a direct O(N^2) computation but vectorize it with numpy.
    """
    # h is length N, nonnegative
    # Compute g at breakpoints z_k = -1/4 + k*w for k=0,...,2N
    # g(z_k) = sum_i sum_j h[i] h[j] * overlap(I_i, z_k - I_j)
    
    # I_i = [-1/4 + i*w, -1/4 + (i+1)*w)
    # z_k - I_j = [z_k - (-1/4 + (j+1)*w), z_k - (-1/4 + j*w))
    #            = [-1/4 + k*w + 1/4 - (j+1)*w, -1/4 + k*w + 1/4 - j*w)
    #            = [(k - j - 1)*w, (k - j)*w)
    
    # So z_k - I_j = [(k-j-1)*w, (k-j)*w)
    
    # I_i = [-1/4 + i*w, -1/4 + (i+1)*w) = [(-N/2 + i)*w, (-N/2 + i + 1)*w)
    
    # Overlap of [(-N/2+i)*w, (-N/2+i+1)*w) and [(k-j-1)*w, (k-j)*w)
    
    # Let a1 = -N/2 + i, b1 = -N/2 + i + 1  (in units of w)
    # Let a2 = k - j - 1, b2 = k - j  (in units of w)
    
    # Overlap length = max(0, min(b1, b2) - max(a1, a2)) * w
    
    # g(z_k) = w * sum_i sum_j h[i] h[j] * max(0, min(b1,b2) - max(a1,a2))
    
    # This is O(N^2 * 2N) = O(N^3) which is 600^3 = 216 million. Too slow.
    
    # Let me use FFT. The autoconvolution can be computed via:
    # g = f * f where * is convolution.
    # 
    # Actually, I realize I can compute this more efficiently.
    # 
    # Note that g(z_k) for k=0,...,2N is the discrete autoconvolution of the sequence
    # h padded appropriately.
    #
    # Let me think of it differently. Define a sequence a of length 2N+1 where
    # a[m] = h[m] for m=0,...,N-1 and a[m] = 0 otherwise.
    #
    # Then the autoconvolution at the breakpoints is related to the discrete 
    # convolution of a with itself, but scaled.
    #
    # Actually, let me just use scipy.signal.fftconvolve on a densified version.
    
    # Simpler approach: sample the step function at many points and use numerical convolution.
    # But this might lose precision.
    
    # Let me try a different approach: the autoconvolution of step functions is a 
    # piecewise linear function. I can compute its values at the breakpoints using 
    # a cumulative sum trick.
    
    # g(z_k) = w * sum_i sum_j h[i] h[j] * max(0, min(-N/2+i+1, k-j) - max(-N/2+i, k-j-1))
    
    # Let me define d = k - j. Then the term is max(0, min(-N/2+i+1, d) - max(-N/2+i, d-1)).
    
    # This is the overlap of intervals [i-N/2, i-N/2+1] and [d-1, d].
    
    # For fixed i and d, this overlap is:
    # If d-1 >= i-N/2+1 or d <= i-N/2: 0
    # Otherwise: min(i-N/2+1, d) - max(i-N/2, d-1)
    
    # This is a "tent" function in d for each i.
    
    # I think the cleanest approach for N=600 is to just do the O(N^2) computation 
    # for each of the 2N+1 breakpoints, but vectorize over i and j using numpy.
    
    # Actually, let me just use a dense grid and FFT.
    
    M = 4 * N  # oversample
    # Create a dense representation of f
    # f(x) on [-1/4, 1/4]
    # Sample at M points
    xs = np.linspace(-1/4, 1/4, M, endpoint=False)
    # Determine which interval each x falls into
    idx = np.clip(((xs + 1/4) / w).astype(int), 0, N-1)
    f_dense = h[idx]
    
    # Autoconvolution via FFT
    g_dense = fftconvolve(f_dense, f_dense, mode='full') * (1/M)  # scale by dx
    
    # But this gives values at xs + xs, which range from -1/2 to 1/2.
    # The maximum of g_dense should be close to the true maximum.
    
    # Actually, this isn't quite right because we're treating f_dense as samples, 
    # not as a step function.
    
    # Let me use a different approach. Since N=600 is not too large, let me just 
    # compute the autoconvolution exactly using the O(N^2) formula, vectorized.
    
    # g(z_k) for k=0,...,2N
    # g(z_k) = w * sum_{i=0}^{N-1} sum_{j=0}^{N-1} h[i]*h[j] * overlap(i, j, k)
    # where overlap(i,j,k) = max(0, min(i+1, k-j) - max(i, k-j-1))  [in units of w, with offset]
    
    # Wait, let me redo the coordinate system. Let me shift so that the intervals are 
    # [0, w), [w, 2w), ..., [(N-1)w, Nw).
    
    # f(x) = h[i] for x in [i*w, (i+1)*w), i=0,...,N-1.
    
    # g(z) = integral f(y) f(z-y) dy
    
    # Breakpoints at z = k*w for k=0,...,2N.
    
    # g(k*w) = w * sum_i sum_j h[i] h[j] * max(0, min(i+1, k-j) - max(i, k-j-1))
    
    # Let me verify: I_i = [i*w, (i+1)*w), z - I_j = [k*w - (j+1)*w, k*w - j*w) = [(k-j-1)*w, (k-j)*w).
    # Overlap of [i, i+1) and [k-j-1, k-j) in units of w:
    # max(0, min(i+1, k-j) - max(i, k-j-1))
    
    # This is correct.
    
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    k_idx = np.arange(2*N + 1)
    
    # For each k, compute sum over i,j of h[i]*h[j]*overlap
    # This is O(N^2 * 2N) = O(N^3). For N=600, that's 216M operations. Might be slow.
    
    # Let me try a smarter approach using the fact that the overlap is a tent function.
    
    # For fixed i, the overlap as a function of d = k-j is:
    # T_i(d) = max(0, min(i+1, d) - max(i, d-1))
    
    # This is 1 for d in [i, i+1], and linearly decreases to 0 at d=i-1 and d=i+2.
    # Actually: 
    # If d-1 >= i+1, i.e., d >= i+2: 0
    # If d <= i: 0 (since max(i, d-1) >= i and min(i+1, d) <= d <= i, so min - max <= 0)
    # Wait, let me recompute.
    # min(i+1, d) - max(i, d-1)
    # Case 1: d <= i: min(i+1,d)=d, max(i,d-1)=i (since d-1 <= i-1 < i). So d-i <= 0. Overlap=0.
    # Case 2: i < d <= i+1: min(i+1,d)=d, max(i,d-1)=i (since d-1 <= i). So d-i. Overlap=d-i.
    # Case 3: i+1 < d <= i+1: not possible since i < d <= i+1 covers up to i+1.
    # Case 3: i+1 < d: min(i+1,d)=i+1, max(i,d-1)=d-1 (since d-1 > i). So i+1-(d-1)=i+2-d.
    #   For this to be positive: d < i+2.
    # So:
    # d in [i, i+1]: overlap = d - i
    # d in [i+1, i+2]: overlap = i+2 - d
    # Otherwise: 0
    
    # So T_i(d) is a tent function centered at d=i+1, with support [i, i+2].
    
    # g(k*w) = w * sum_j h[j] * sum_i h[i] * T_i(k-j)
    
    # Let me define for each j, the inner sum is (h * T)(k-j) where T is the sum of tents.
    # Actually, this is a convolution.
    
    # Let me define S[m] = sum_i h[i] * T_i(m) for m = 0, 1, ..., 2N.
    # Then g(k*w) = w * sum_j h[j] * S[k-j]
    
    # S[m] = sum_i h[i] * T_i(m)
    # T_i(m) = m-i for m in [i, i+1], i+2-m for m in [i+1, i+2], 0 otherwise.
    
    # Since m and i are integers (or half-integers?), let me be careful.
    # k, i, j are integers. d = k-j is an integer.
    # T_i(d) for integer d:
    # If d = i: T_i(i) = 0 (since d-i=0)
    # If d = i+1: T_i(i+1) = 1 (from first case: d-i=1, or second case: i+2-d=1)
    # If d = i+2: T_i(i+2) = 0
    # So for integer d, T_i(d) = 1 if d=i+1, else 0.
    
    # Wait, that can't be right. Let me recheck.
    # The overlap of [i, i+1) and [d-1, d) when d is an integer:
    # If d = i+1: [i, i+1) and [i, i+1) -> overlap = 1.
    # If d = i: [i, i+1) and [i-1, i) -> overlap = 0.
    # If d = i+2: [i, i+1) and [i+1, i+2) -> overlap = 0.
    
    # So indeed, for integer d, T_i(d) = 1 if d = i+1, else 0.
    
    # That means g(k*w) = w * sum_j h[j] * S[k-j] where S[m] = sum_i h[i] * delta(m - i - 1) = h[m-1] if 0<=m-1<N else 0.
    
    # So S[m] = h[m-1] for m=1,...,N, and 0 otherwise.
    
    # g(k*w) = w * sum_j h[j] * h[k-j-1] (where h[m]=0 if m<0 or m>=N)
    
    # This is just w * (h * h)[k-1] where * is discrete convolution!
    
    # So the autoconvolution at the breakpoints is w times the discrete autoconvolution of h!
    
    # And the maximum of the piecewise linear function g occurs at one of the breakpoints.
    
    # So max(g) = w * max(discrete_autoconv(h))
    
    # And the discrete autoconvolution can be computed via FFT!
    
    h_arr = np.asarray(h, dtype=float)
    autoconv = fftconvolve(h_arr, h_arr, mode='full')
    return w * np.max(autoconv)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a reasonable initial guess
    # The optimal step function for minimizing the max autoconvolution 
    # is related to the Fejer kernel or similar.
    # Let me start with a cosine-shaped function.
    x = np.linspace(0, 1, N)
    h = 0.5 * (1 + np.cos(2 * np.pi * x))  # cosine bump
    h = np.maximum(h, 0)
    
    # Normalize so that the integral is fixed? The problem says "nonnegative step function"
    # but doesn't specify normalization. Let me re-read the task.
    # "minimize the normalized maximum of its autoconvolution"
    # The normalization is likely by the L1 norm or L2 norm squared.
    # 
    # Actually, the task says "normalized maximum". Let me assume it's 
    # max(f*f) / (integral f)^2 or something similar.
    # 
    # Looking at the seed program, it returns [1.0]*N and gets score 1.0.
    # Let me compute what the score would be for the constant function.
    # For h = [1]*N, the autoconvolution max is w * max(fftconvolve(ones, ones))
    # fftconvolve(ones(N), ones(N)) has max value N at the center.
    # So max autoconv = w * N = (0.5/N) * N = 0.5.
    # 
    # The "normalized" part: if we normalize by (integral f)^2 = (w*N)^2 = (0.5)^2 = 0.25,
    # then the normalized max is 0.5 / 0.25 = 2.
    # 
    # But the score is 1.0 for the seed. So maybe the score is 1/normalized_max or something.
    # Or maybe the normalization is different.
    #
    # Let me just compute the raw autoconvolution max and see.
    # The evaluator says "score is the upper bound on C1" and "lower is better".
    # And "search score is seed/value, so 1.0 equals the seed program".
    # So if seed value = V_seed, then score = V_seed / V_child.
    # For seed, score = 1.0, so V_seed / V_seed = 1.
    # 
    # I need to figure out what the "value" is. Let me just compute 
    # max(autoconv) and use that as the value. The evaluator will handle the rest.
    
    # Actually, I think the value is just the maximum of the autoconvolution,
    # possibly normalized by the L1 norm squared to make it scale-invariant.
    # 
    # Let me assume the value = max(f*f) / (integral f)^2.
    # For the constant function: max(f*f) = 0.5, (integral f)^2 = 0.25, ratio = 2.
    # 
    # But I'm not sure. Let me just return the step heights and let the evaluator
    # compute the score. My job is to minimize whatever metric they use.
    
    # Let me implement a coordinate descent / local search.
    
    def compute_value(h):
        """Compute the normalized max autoconvolution."""
        h_arr = np.asarray(h, dtype=float)
        h_arr = np.maximum(h_arr, 0)
        integral = np.sum(h_arr) * w
        if integral < 1e-12:
            return 1e10
        autoconv = fftconvolve(h_arr, h_arr, mode='full')
        max_autoconv = np.max(autoconv)
        return max_autoconv / (integral * integral)
    
    best_h = h.copy()
    best_val = compute_value(best_h)
    
    # Coordinate descent with line search
    # For each coordinate, try different values and pick the best.
    
    # Geometrically spaced candidates
    def get_candidates(current_val, best_val):
        """Generate candidate values for a coordinate."""
        candidates = [0.0]
        if current_val > 0:
            # Try fractions of current value
            for frac in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
                candidates.append(current_val * frac)
        # Try absolute values based on best_val scale
        scale = max(0.01, best_val * 0.1)
        for s in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            candidates.append(s * scale)
        return candidates
    
    # Iterate over coordinates
    n_iter = 0
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            current_val = best_h[i]
            candidates = get_candidates(current_val, best_val)
            
            best_candidate_val = current_val
            best_candidate_obj = best_val
            
            for cand in candidates:
                if cand < 0:
                    continue
                trial_h = best_h.copy()
                trial_h[i] = cand
                trial_obj = compute_value(trial_h)
                if trial_obj < best_candidate_obj:
                    best_candidate_obj = trial_obj
                    best_candidate_val = cand
            
            if best_candidate_val != current_val:
                best_h[i] = best_candidate_val
                best_val = best_candidate_obj
                improved = True
        
        n_iter += 1
        if not improved and n_iter > 1:
            break
    
    # Final polish: try fine-tuning
    if time.time() < deadline - 5:
        # Do a few more passes with finer candidates
        for _ in range(3):
            if time.time() > deadline:
                break
            for i in range(N):
                if time.time() > deadline:
                    break
                current_val = best_h[i]
                if current_val == 0:
                    continue
                # Fine-tune around current value
                for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                    trial_val = current_val + delta
                    if trial_val < 0:
                        continue
                    trial_h = best_h.copy()
                    trial_h[i] = trial_val
                    trial_obj = compute_value(trial_h)
                    if trial_obj < best_val:
                        best_h[i] = trial_val
                        best_val = trial_obj
    
    return list(best_h)
