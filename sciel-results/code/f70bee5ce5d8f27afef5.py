"""Third autocorrelation inequality — structural rewrite with block-flip SA."""
import numpy as np
import time
import random

N = 400
DEADLINE = 115.0  # safety margin below 150s evaluator limit


def _compute_c3(h):
    """
    Compute C3[k] = sum_i h[i] * h[i+k] * h[i+2k] for valid i, for each lag k.
    Returns array of length N//2 + 1 (lags 0..N//2).
    Uses FFT-based approach: C3 is the cross-correlation structure of the triple product.
    
    For a ±1 signal, C3[k] = sum_{i=0}^{N-1-2k} h[i]*h[i+k]*h[i+2k].
    We can compute this via: let g = h * h (elementwise, which is 1 for ±1),
    but that's trivial. Instead, use the direct FFT trick:
    
    C3[k] = sum_i h[i] h[i+k] h[i+2k]
    
    This is the 3rd-order autocorrelation. We can compute it as:
    Let H = fft(h, 2N). Then the circular autocorrelation of order 2 is ifft(H * H * conj(H))... 
    Actually, for the triple sum, we need a different approach.
    
    Direct approach: C3[k] = sum_i h[i] * h[i+k] * h[i+2k]
    This is a linear convolution-like sum. We can compute it efficiently:
    
    Method: For each k, C3[k] = <h[0:N-2k], h[k:N-k] * h[2k:N]>
    This is O(N^2) total which is 160000 ops — fast enough.
    """
    n = len(h)
    half = n // 2
    c3 = np.zeros(half + 1)
    for k in range(half + 1):
        # sum over i=0 to n-1-2k of h[i]*h[i+k]*h[i+2k]
        m = n - 2 * k
        c3[k] = np.sum(h[:m] * h[k:k+m] * h[2*k:2*k+m])
    return c3


def _score(h):
    """Return max |C3[k]| over all valid lags k."""
    c3 = _compute_c3(h)
    return np.max(np.abs(c3))


def _score_fast(h):
    """
    Faster scoring using the fact that for ±1 signals, we can use FFT.
    C3[k] = sum_i h[i] h[i+k] h[i+2k]
    
    Alternative: note that h[i]h[i+k]h[i+2k] = h[i] * (h[i+k]*h[i+2k])
    Let d[i] = h[i]*h[i+1] (the difference signal). Then h[i+k]*h[i+2k] can be related.
    
    Actually, let's just use the direct method but vectorize over k.
    For N=400, the direct loop over k (201 iterations) with numpy sum is fast.
    """
    n = len(h)
    half = n // 2
    c3 = np.zeros(half + 1)
    for k in range(half + 1):
        m = n - 2 * k
        if m <= 0:
            break
        c3[k] = np.dot(h[:m], h[k:k+m] * h[2*k:2*k+m])
    return np.max(np.abs(c3))


def _random_sign_vector():
    return np.random.choice([-1.0, 1.0], size=N)


def _block_flip(h, start, length):
    """Flip a contiguous block of signs."""
    h2 = h.copy()
    h2[start:start+length] *= -1
    return h2


def _construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Phase 1: Evaluate several structured initial patterns
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    # Various block patterns
    for period in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 40, 50, 80, 100, 200]:
        for duty in range(1, period):
            pattern = np.ones(period)
            pattern[duty:] = -1.0
            h = np.tile(pattern, N // period + 1)[:N]
            candidates.append(h)
    
    # Sine-based patterns
    x = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        for phase in np.linspace(0, np.pi, 10, endpoint=False):
            h = np.sign(np.sin(2 * np.pi * freq * x + phase))
            h[h == 0] = 1.0
            candidates.append(h)
    
    # Random patterns
    for _ in range(20):
        candidates.append(_random_sign_vector())
    
    # Score all candidates
    best_h = None
    best_score = np.inf
    for h in candidates:
        if not elapsed_ok():
            break
        s = _score_fast(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score_fast(best_h)
    
    # Phase 2: Simulated annealing with block flips
    # This is the key structural improvement: block flips allow coordinated changes
    h_curr = best_h.copy()
    score_curr = best_score
    h_best = best_h.copy()
    score_best = best_score
    
    # Annealing schedule
    t0 = 1.0
    t_min = 0.001
    # We'll do many iterations, checking time
    n_iter_budget = 50000
    
    # Block sizes to try
    block_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 80]
    
    iter_count = 0
    while iter_count < n_iter_budget and elapsed_ok():
        # Temperature schedule: linear cooling
        progress = iter_count / n_iter_budget
        temp = t0 * (t_min / t0) ** progress
        
        # Choose a block size (biased toward smaller blocks as temperature drops)
        if temp > 0.1:
            blen = block_sizes[int(rng.integers(0, len(block_sizes)))]
        else:
            blen = block_sizes[int(rng.integers(0, max(1, len(block_sizes)//2)))]
        
        if blen >= N:
            blen = N // 2
        
        start = int(rng.integers(0, N - blen + 1))
        
        h_new = _block_flip(h_curr, start, blen)
        score_new = _score_fast(h_new)
        
        delta = score_new - score_curr
        
        # Accept if better, or with probability exp(-delta/temp)
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            h_curr = h_new
            score_curr = score_new
            if score_curr < score_best:
                score_best = score_curr
                h_best = h_curr.copy()
        
        iter_count += 1
    
    # Phase 3: Local refinement with single-bit flips (greedy)
    h_work = h_best.copy()
    score_work = score_best
    improved = True
    while improved and elapsed_ok():
        improved = False
        order = rng.permutation(N)
        for idx in order:
            if not elapsed_ok():
                break
            h_try = h_work.copy()
            h_try[idx] *= -1
            s_try = _score_fast(h_try)
            if s_try < score_work - 1e-12:
                score_work = s_try
                h_work = h_try
                if s_try < score_best:
                    score_best = s_try
                    h_best = h_work.copy()
                improved = True
    
    # Phase 4: A few more SA restarts from random starts if time allows
    for restart in range(3):
        if not elapsed_ok():
            break
        
        h_rand = _random_sign_vector()
        s_rand = _score_fast(h_rand)
        
        if s_rand < score_best:
            score_best = s_rand
            h_best = h_rand.copy()
        
        # Quick SA from this start
        h_sa = h_rand.copy()
        s_sa = s_rand
        for it in range(2000):
            if not elapsed_ok():
                break
            temp = 0.5 * (0.01) ** (it / 2000.0)
            blen = int(rng.integers(1, 21))
            if blen >= N:
                blen = N // 4
            start = int(rng.integers(0, N - blen + 1))
            h_try = _block_flip(h_sa, start, blen)
            s_try = _score_fast(h_try)
            delta = s_try - s_sa
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                h_sa = h_try
                s_sa = s_try
                if s_sa < score_best:
                    score_best = s_sa
                    h_best = h_sa.copy()
        
        # Local refinement
        h_work2 = h_best.copy()
        s_work2 = score_best
        improved2 = True
        while improved2 and elapsed_ok():
            improved2 = False
            order = rng.permutation(N)
            for idx in order:
                if not elapsed_ok():
                    break
                h_try = h_work2.copy()
                h_try[idx] *= -1
                s_try = _score_fast(h_try)
                if s_try < s_work2 - 1e-12:
                    s_work2 = s_try
                    h_work2 = h_try
                    if s_try < score_best:
                        score_best = s_try
                        h_best = h_work2.copy()
                    improved2 = True
    
    # Ensure output is ±1
    result = np.sign(h_best).astype(np.float64)
    result[result == 0] = 1.0
    return result


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Returns a sequence of signed step heights.
    Score is the upper bound on C3 (max |C3|), lower is better.
    """
    return _construct()


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score_fast(h))
