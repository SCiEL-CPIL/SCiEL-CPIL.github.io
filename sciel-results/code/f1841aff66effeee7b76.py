import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n."""
    # Thue-Morse is defined by the parity of the number of 1s in the binary representation.
    # Alternatively, it can be generated recursively: t = 0, 1-t.
    # For efficiency with n=400, we can just compute bit counts.
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count bits set in i
        bits = bin(i).count('1')
        seq[i] = 1.0 if bits % 2 == 0 else -1.0
    return seq

def _linear_acf(f):
    """
    Compute the linear autoconvolution of f.
    The result has length 2*N - 1.
    The peak (zero-lag) is at index N-1.
    """
    # Use FFT for efficient convolution
    # Zero pad to next power of 2 or just 2*N to avoid circular wrap
    size = 2 * N
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real
    # The linear convolution is in conv[:2*N-1]
    # The zero-lag term is at index N-1
    return conv[:2*N-1]

def _psl_value(f):
    """
    Compute the Peak Sidelobe Level (PSL).
    This is the maximum absolute value of the autoconvolution
    excluding the zero-lag (peak) term.
    """
    acf = _linear_acf(f)
    # Zero lag is at index N-1
    # We want max abs of acf[0:N-1] and acf[N:]
    # Note: acf is symmetric for real inputs, but let's check all non-zero lags.
    # Lags 0 to N-2 correspond to positive lags in one direction?
    # Standard linear convolution of [a0, a1] and [a0, a1] is [a0^2, a0a1+a1a0, a1^2].
    # Index 0 is lag -(N-1)? No.
    # Let's verify indices.
    # f = [f0, ..., f_{N-1}]
    # (f * f)[k] = sum f[i] f[k-i]
    # Max k is 2N-2. Min k is 0.
    # Zero lag (sum f[i]^2) is at k = N-1.
    # So sidelobes are indices 0 to N-2 and N to 2N-2.
    if N == 1:
        return 0.0
    return np.max(np.abs(np.concatenate([acf[:N-1], acf[N:]])))

def _smooth_psl(f, beta=10.0):
    """
    A smooth approximation of the max sidelobe to aid gradient-like search.
    Uses log-sum-exp trick.
    """
    acf = _linear_acf(f)
    sidelobes = np.concatenate([acf[:N-1], acf[N:]])
    # To avoid overflow in exp, subtract max
    m = np.max(np.abs(sidelobes))
    if m < 1e-10:
        return 0.0
    # We want to minimize the max. 
    # Using soft-max on squared values or just values.
    # Let's use a soft-max on the absolute values.
    # Actually, for local search, the hard max is often sufficient if we do many iterations.
    # But smooth helps escape local minima if we use it in a simulated annealing context.
    # For now, let's stick to hard max for simplicity and speed, 
    # but maybe add a small penalty for sum of squares to break ties?
    # Score = max_psl + 0.001 * sum(sidelobes^2)
    return m + 0.001 * np.sum(sidelobes * sidelobes)

def _local_search(f, deadline, max_iter=100000):
    """
    Greedy local search: flip bits to improve the score.
    """
    current_score = _smooth_psl(f)
    improved = True
    it = 0
    # Use a random order of indices to avoid bias
    indices = np.arange(N)
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to randomize search path
        np.random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                return f, current_score
            it += 1
            if it > max_iter:
                return f, current_score
            
            # Flip sign
            f[idx] = -f[idx]
            new_score = _smooth_psl(f)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                f[idx] = -f[idx] # Revert
    return f, current_score

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s
    deadline = start_time + 140.0
    
    best_f = None
    best_score = np.inf
    
    # Candidate 1: Thue-Morse
    f_tm = _thue_morse(N)
    score_tm = _psl_value(f_tm)
    if score_tm < best_score:
        best_score = score_tm
        best_f = f_tm.copy()
    
    # Local search on Thue-Morse
    if time.time() < deadline - 10:
        f_tm_opt, score_tm_opt = _local_search(f_tm.copy(), deadline - 5)
        # Re-evaluate with hard metric
        hard_score = _psl_value(f_tm_opt)
        if hard_score < best_score:
            best_score = hard_score
            best_f = f_tm_opt.copy()
            
    # Candidate 2: Alternating signs (1, -1, 1, -1...)
    # This has very specific autocorrelation properties.
    f_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    score_alt = _psl_value(f_alt)
    if score_alt < best_score:
        best_score = score_alt
        best_f = f_alt.copy()
        
    if time.time() < deadline - 10:
        f_alt_opt, _ = _local_search(f_alt.copy(), deadline - 5)
        hard_score = _psl_value(f_alt_opt)
        if hard_score < best_score:
            best_score = hard_score
            best_f = f_alt_opt.copy()
            
    # Candidate 3: Random initialization (if time permits)
    if time.time() < deadline - 20:
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        score_rand = _psl_value(f_rand)
        if score_rand < best_score:
            best_score = score_rand
            best_f = f_rand.copy()
            
        if time.time() < deadline - 10:
            f_rand_opt, _ = _local_search(f_rand.copy(), deadline - 5)
            hard_score = _psl_value(f_rand_opt)
            if hard_score < best_score:
                best_score = hard_score
                best_f = f_rand_opt.copy()
                
    # Final polish on the best candidate found
    if best_f is not None and time.time() < deadline - 2:
        best_f, _ = _local_search(best_f.copy(), deadline)
        # Final check
        final_score = _psl_value(best_f)
        # Note: _local_search minimizes _smooth_psl, which might not be perfectly aligned with _psl_value
        # But usually they are correlated. We return the f that gave the best _psl_value in the loop above?
        # Actually, the last local search might have degraded the hard PSL slightly if smooth != hard.
        # Let's keep the best_f from the loop if its hard score was better, 
        # but the final local search operated on a copy. 
        # Let's re-evaluate the final best_f's hard score.
        # If the final local search improved the smooth score, it likely improved the hard score too.
        # But to be safe, we should probably track the best hard score separately.
        
        # Let's refine: We should track the best hard score explicitly.
        # The code above updates best_f and best_score based on hard_score checks.
        # The final local search modifies best_f in place? No, it takes a copy.
        # Wait, `_local_search(best_f.copy(), ...)` returns a new array.
        # I need to capture that result.
        
    # Let's re-structure the final polish to be safe
    if best_f is not None:
        # Try one more round of local search on the current best
        # But we must ensure we don't exceed time.
        remaining = deadline - time.time()
        if remaining > 5:
            # Use a tighter deadline for this final step
            f_final, _ = _local_search(best_f.copy(), start_time + remaining - 1)
            score_final = _psl_value(f_final)
            if score_final < best_score:
                best_f = f_final
                best_score = score_final

    if best_f is None:
        # Fallback: all ones
        best_f = np.ones(N)
        
    return best_f
