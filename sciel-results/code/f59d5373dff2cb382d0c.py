"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Strategy:
1. Start from structured candidates (Golay, Barker, random).
2. Use a continuous relaxation: allow f[i] to be real values in [-1, 1].
3. Minimize the smooth objective: sum_k (ac[k])^2 / (sum f)^2 using gradient descent.
   This is equivalent to minimizing the L2 norm of the autoconvolution relative to the sum.
   The L2 norm bounds the max norm, so reducing L2 helps reduce max.
4. After continuous optimization, project to {-1, 1} by sign().
5. Polish with discrete local search (single flips) on the projected sequence.
6. Repeat from multiple random restarts within the time budget.

The key insight: the continuous relaxation allows gradient-based optimization
that can find configurations with much lower autoconvolution energy than
purely discrete search, and the projection back to binary gives a good
starting point for discrete refinement.
"""
import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2 * n - 1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _smooth_objective(f):
    """
    Smooth objective: ||ac||_2^2 / (sum f)^2.
    This is differentiable and easier to optimize than max|ac|/(sum f)^2.
    """
    s = np.sum(f)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(f)
    ac = _autoconv_fft(f)
    obj = np.sum(ac ** 2) / (s * s)
    # Gradient of obj w.r.t. f
    # d/d f[j] [ sum_k ac[k]^2 / s^2 ]
    # = (1/s^2) * 2 * sum_k ac[k] * d(ac[k])/d(f[j]) - 2*s * sum(ac^2) / s^4 * 1
    # d(ac[k])/d(f[j]) = f[k-j] + f[j-k] (with zero padding), but for linear ac:
    # ac[k] = sum_i f[i] f[k-i]
    # d(ac[k])/d(f[j]) = f[k-j] if 0<=k-j<N else 0  (from i=j term)
    #                   + f[k-j] if 0<=k-j<N else 0  (from k-i=j term, i=k-j)
    # Wait, more carefully:
    # ac[k] = sum_i f[i] f[k-i]
    # Term where i=j: f[j] f[k-j] -> derivative w.r.t f[j] is f[k-j]
    # Term where k-i=j, i.e., i=k-j: f[k-j] f[j] -> derivative w.r.t f[j] is f[k-j]
    # But if j == k-j (i.e., k=2j), these are the same term, so only one f[j] factor.
    # Actually no: ac[k] = sum_i f[i] f[k-i]. The term i=j gives f[j]*f[k-j].
    # The term i=k-j gives f[k-j]*f[j]. These are distinct terms in the sum unless j=k-j.
    # If j != k-j: both terms exist, derivative = f[k-j] + f[k-j] = 2*f[k-j]
    # If j == k-j (k=2j): only one term f[j]*f[j]=f[j]^2, derivative = 2*f[j]
    # So d(ac[k])/d(f[j]) = 2*f[k-j] if 0<=k-j<N, with the understanding that
    # when k-j=j, it's 2*f[j] which is correct.
    
    # For efficiency, compute via correlation:
    # sum_k ac[k] * d(ac[k])/d(f[j]) = sum_k ac[k] * (f[k-j] + f[k-j]) for valid k-j
    # = 2 * sum_k ac[k] * f[k-j] where 0<=k-j<N
    # Let m = k-j, then k = m+j, m in [0, N-1], k in [j, j+N-1]
    # = 2 * sum_{m=0}^{N-1} ac[m+j] * f[m]
    # This is the cross-correlation of ac (padded) with f.
    
    # More simply: the gradient of sum_k ac[k]^2 w.r.t. f is 2 * (ac * f) where * is
    # the appropriate convolution/correlation. Specifically:
    # d/d(f[j]) sum_k ac[k]^2 = 2 * sum_k ac[k] * d(ac[k])/d(f[j])
    # = 2 * sum_k ac[k] * (f[k-j] + f[k-j]) for valid indices
    # = 4 * sum_{m: 0<=m<N, 0<=m+j<2N-1} ac[m+j] * f[m]
    # Hmm, let's just use FFT to compute this efficiently.
    
    # Actually, the cleanest way: 
    # sum_k ac[k]^2 = ||ac||^2. And ac = f * f (linear convolution).
    # The gradient of ||f*f||^2 w.r.t. f is 4 * (f * (f * f)) = 4 * (f * ac)
    # where the first * is linear convolution. Let's verify:
    # ||f*f||^2 = sum_k (sum_i f[i]f[k-i])^2
    # d/d(f[j]) = 2 sum_k (sum_i f[i]f[k-i]) * (f[k-j] + f[j-k])
    #            = 2 sum_k ac[k] * (f[k-j] + f[j-k])
    # The term sum_k ac[k] f[k-j]: let m=k-j, = sum_m ac[m+j] f[m] = (ac_padded * f)[j]
    # where ac_padded is ac zero-padded to length 2N-1, and * is cross-correlation.
    # Similarly sum_k ac[k] f[j-k] = sum_m ac[m] f[j-m] = (ac * f)[j] (convolution)
    # By symmetry of ac (ac[k]=ac[-k] for the relevant range... actually ac is not symmetric
    # for linear convolution, but ac[k] = sum_i f[i]f[k-i], and ac[2N-2-k] = sum_i f[i]f[2N-2-k-i]
    # which is not the same as ac[k] in general for finite sequences).
    
    # Let me just compute it numerically via FFT for correctness.
    # grad_obj = d/d(f) [sum(ac^2) / s^2]
    # = (2 * grad_sum_ac_sq / s^2) - (2 * sum_ac_sq / s^3) * ones
    # where grad_sum_ac_sq[j] = 2 * sum_k ac[k] * (f[k-j] + f[j-k]) for valid indices
    
    # Efficient computation via FFT:
    # sum_k ac[k] f[k-j] for k in [j, j+N-1] = cross-correlation
    # Let's use the fact that this is the j-th element of the convolution of ac_rev with f
    # where ac_rev[k] = ac[2N-2-k].
    
    # For simplicity and correctness, let's compute via direct formula using FFT:
    # g[j] = sum_k ac[k] * d(ac[k])/d(f[j])
    # We know ac = ifft(fft(f)^2), so let's work in frequency domain.
    # Actually, let's just do it the straightforward way with FFT:
    
    size = 1
    while size < 2 * N - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    A = np.fft.rfft(ac, size)  # ac is length 2N-1, padded to size
    # The gradient of sum|A|^2 w.r.t. F is 2*conj(F)*A... no, that's not right either.
    
    # Let me just compute it directly for N=400. It's O(N^2) but N=400 is fine.
    # Actually, let's use a cleaner FFT approach:
    # sum_k ac[k]^2 = (1/2pi) integral |F(omega)|^4 domega (Parseval for convolution)
    # Actually sum_k |ac[k]|^2 = (1/2pi) ∫ |F(ω)|^4 dω for the appropriate normalization.
    # The gradient w.r.t. f is then related to the inverse FFT of 4|F|^2 * conj(F).
    
    # Let's use the standard result: if ac = f ⊛ f (linear convolution), then
    # ∇_f ||ac||_2^2 = 4 (f ⊛ ac) where ⊛ is linear convolution.
    # Let's verify: ||f⊛f||^2 = <f⊛f, f⊛f> = <f, (f⊛f)⊛f> (by associativity and commutativity)
    # Wait, <f⊛f, f⊛f> = <f, f⊛(f⊛f)>? Let's check:
    # sum_k (f⊛f)[k]^2 = sum_k sum_i sum_j f[i]f[k-i]f[j]f[k-j]
    # = sum_i sum_j f[i]f[j] sum_k f[k-i]f[k-j]
    # = sum_i sum_j f[i]f[j] (f⊛f_rev)[i-j] where f_rev[m]=f[-m]... this is getting complex.
    
    # Let me just compute the gradient numerically for correctness since N=400 is small.
    # But O(N^2) per gradient step might be slow for many steps. Let's use FFT.
    
    # The correct formula: ∇_f ||f⊛f||_2^2 = 4 (f ⊛ (f⊛f)) = 4 (f ⊛ ac)
    # This is a standard result from the chain rule and the fact that the adjoint of
    # convolution by f is convolution by f (for real signals).
    
    f_ac = np.convolve(f, ac, mode='full')  # length 2N-1 + N - 1 = 3N-2
    # The j-th element of f ⊛ ac (linear convolution) is f_ac[j]
    # But we need to be careful about the indexing.
    # (f ⊛ ac)[j] = sum_i f[i] ac[j-i]
    # This should give us grad_sum_ac_sq[j] / 2 (since we had a factor of 2).
    
    grad_sum_ac_sq = 4.0 * f_ac[:N]  # Take the first N elements
    
    # Now divide by s^2 and subtract the s-dependent term
    sum_ac_sq = np.sum(ac ** 2)
    grad = (2.0 * grad_sum_ac_sq / (s * s)) - (2.0 * sum_ac_sq / (s ** 3)) * np.ones(N)
    
    return obj, grad


def _continuous_optimize(f_init, max_iters=5000, lr=0.01):
    """
    Optimize the continuous relaxation of the smooth objective.
    f is allowed to be any real vector; we minimize ||ac||_2^2 / (sum f)^2.
    """
    f = f_init.astype(float).copy()
    s = np.sum(f)
    if abs(s) < 1e-12:
        f += np.random.randn(N) * 0.1
        s = np.sum(f)
    
    best_obj = np.inf
    best_f = f.copy()
    
    for it in range(max_iters):
        if _remaining() < 10.0:
            break
        
        obj, grad = _smooth_objective(f)
        if obj < best_obj:
            best_obj = obj
            best_f = f.copy()
        
        # Gradient descent step
        f_new = f - lr * grad
        # Keep values in a reasonable range to avoid numerical issues
        f_new = np.clip(f_new, -2.0, 2.0)
        
        new_obj, _ = _smooth_objective(f_new)
        if new_obj < obj:
            f = f_new
            # Adaptive learning rate
            lr = min(lr * 1.05, 0.5)
        else:
            lr = max(lr * 0.9, 0.001)
    
    return best_f, best_obj


def _project_and_polish(f_cont, max_iters=2000):
    """
    Project continuous solution to {-1, 1} and polish with discrete local search.
    """
    f = np.sign(f_cont)
    f[f == 0] = 1.0  # Handle zeros
    
    s = np.sum(f)
    if abs(s) < 1e-12:
        return f, np.inf
    
    ac = _autoconv_fft(f)
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    # Discrete local search: single-bit flips
    rng = np.random.RandomState(int(hash(tuple(f.astype(int))) % 100000) % 2**31)
    improved = True
    while improved and _remaining() > 5.0:
        improved = False
        for idx in range(N):
            f[idx] = -f[idx]
            s_new = np.sum(f)
            if abs(s_new) < 1e-12:
                f[idx] = -f[idx]
                continue
            ac_new = _autoconv_fft(f)
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            if new_score < cur_score:
                cur_score = new_score
                ac = ac_new
                s = s_new
                improved = True
            else:
                f[idx] = -f[idx]
    
    return f, cur_score


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    rng = np.random.RandomState(42)
    
    best_f = None
    best_score = np.inf
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay-like complementary sequence construction
    def golay_pair(n):
        """Generate Golay complementary pair of length n (must be power of 2)."""
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2
    
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
    else:
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
    
    # Barker-like sequence
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()
    
    # Continuous optimization from multiple starts
    n_restarts = 5
    for r in range(n_restarts):
        if _remaining() < 20.0:
            break
        
        # Use the best discrete solution as a starting point, perturbed
        if r == 0:
            f_init = best_f.astype(float)
        else:
            # Random continuous initialization
            f_init = rng.randn(N) * 0.5
        
        # Run continuous optimization
        f_cont, obj_cont = _continuous_optimize(f_init, max_iters=3000)
        
        # Project to binary and polish
        f_bin, score_bin = _project_and_polish(f_cont)
        
        if score_bin < best_score:
            best_score = score_bin
            best_f = f_bin.copy()
    
    # Final discrete polish from the best solution
    if _remaining() > 10.0:
        f = best_f.copy()
        s = np.sum(f)
        if abs(s) >= 1e-12:
            ac = _autoconv_fft(f)
            cur_score = np.max(np.abs(ac)) / (s * s)
            
            # Greedy local search
            improved = True
            while improved and _remaining() > 5.0:
                improved = False
                for idx in range(N):
                    f[idx] = -f[idx]
                    s_new = np.sum(f)
                    if abs(s_new) < 1e-12:
                        f[idx] = -f[idx]
                        continue
                    ac_new = _autoconv_fft(f)
                    new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
                    if new_score < cur_score:
                        cur_score = new_score
                        ac = ac_new
                        s = s_new
                        improved = True
                    else:
                        f[idx] = -f[idx]
            
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
    
    return best_f
