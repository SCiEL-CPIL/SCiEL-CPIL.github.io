"""
Combined program: Thue-Morse initialization + Incremental O(N) local sign-flip optimization.
"""
import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/- 1."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    # 0 -> 1, 1 -> -1
    seq = np.array([1.0])
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n]

def compute_autoconv(a):
    """Compute full autoconvolution of a using FFT for initial setup."""
    return np.correlate(a, a, mode='full')

def compute_objective(a, ac=None):
    """Compute max abs autoconv / (sum a)^2. 
    If ac is not provided, it is computed.
    """
    s = np.sum(a)
    if s == 0:
        return np.inf, None
    if ac is None:
        ac = compute_autoconv(a)
    return np.max(np.abs(ac)) / (s * s), ac

def local_sign_flip_optimize_incremental(a, max_rounds=200, deadline=None):
    """
    Iteratively flip signs to reduce max abs autoconv.
    Uses incremental update of autoconvolution for O(N) per flip check.
    """
    a = a.copy().astype(np.float64)
    ac = compute_autoconv(a)
    current_max_abs = np.max(np.abs(ac))
    s = np.sum(a)
    # Note: s changes sign if we flip, but s^2 is invariant to sign flips? 
    # Wait, s = sum(a). If we flip a[j], s_new = s - 2*a[j].
    # So s^2 changes. We must track s.
    
    best_obj = (current_max_abs) / (s * s)
    best_a = a.copy()
    best_ac = ac.copy()
    best_s = s
    best_max_abs = current_max_abs
    
    for r in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        n = len(a)
        
        # We scan through indices. 
        # For each index i, we simulate flipping a[i].
        # Update ac incrementally.
        # Update s incrementally.
        
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_a, best_obj
            
            # Simulate flip at i
            # a[i] becomes -a[i]
            # Change in s: ds = -2 * a[i]
            # Change in ac: 
            # ac[k] = sum_j a[j] a[j+k]
            # The terms involving a[i] are where j=i (contribution a[i]*a[i+k]) 
            # and where j+k=i (contribution a[i-k]*a[i]).
            # So ac_new[k] = ac_old[k] - 2 * a[i] * (a[i+k] + a[i-k]) 
            # with boundary checks.
            
            # Vectorized update for ac
            # We need to add -2 * a[i] * (a shifted by i) and -2 * a[i] * (a shifted by -i)
            
            # Let's construct the delta_ac
            delta_ac = np.zeros_like(ac)
            
            # Term 1: j=i -> affects lags k such that i+k is valid.
            # a[i] * a[i+k]. In full correlation, index in ac is (i+k) - 0? 
            # np.correlate(a, a, 'full') index k corresponds to lag k.
            # ac[k] = sum_{i} a[i] * a[i+k] (where a is 0 outside [0, N-1])
            # If we flip a[i], the term a[i]*a[i+k] becomes -a[i]*a[i+k].
            # Change is -2 * a[i] * a[i+k].
            # This affects ac at index corresponding to lag k = (i+k) - i? No.
            # The index in the 'full' output for lag L is L + N - 1.
            # Here, the lag is effectively determined by the pair (i, j).
            # Let's use the property: ac = fftconvolve(a, a[::-1], 'full')? 
            # No, np.correlate(a, a, 'full') is equivalent to convolution of a and reversed a?
            # Actually, it's easier to just compute the difference directly.
            
            # Let's revert to a safer O(N) update if vectorization is tricky.
            # But O(N) vectorized update:
            # The autoconvolution ac has length 2N-1.
            # ac[idx] = sum_{p} a[p] * a[p + (idx - (N-1))]
            
            # Let's try a direct update.
            # Flip a[i].
            # New ac = ac - 2 * a[i] * (a * delta_i) ?
            # The contribution of a[i] to ac is:
            # For each j, a[i] * a[j] contributes to ac at lag j-i.
            # So we subtract 2 * a[i] * a[j] from ac[j-i + N-1].
            
            # This can be done by:
            # delta_ac = -2 * a[i] * np.correlate(a, np.zeros(N), 'full')? No.
            
            # Let's use the fact that ac = conv(a, a).
            # If we change a[i] by -2*a[i], the change in ac is -2*a[i] * conv(delta_i, a).
            # conv(delta_i, a) is just a shifted a.
            
            # Let's implement the shift.
            # ac_new = ac
            # We need to subtract 2 * a[i] * a from the appropriate part of ac.
            
            # The term a[i] * a[j] is at index j - i + N - 1 in ac.
            # So we subtract 2 * a[i] * a[j] from ac[j - i + N - 1] for all j.
            
            # This is equivalent to:
            # delta = np.zeros(2*N - 1)
            # For j in 0..N-1:
            #   idx = j - i + N - 1
            #   if 0 <= idx < 2*N-1:
            #      delta[idx] -= 2 * a[i] * a[j]
            
            # This loop is O(N).
            # Let's vectorize it.
            # Indices j: 0..N-1
            # Indices in ac: (0..N-1) - i + N - 1
            # start = N - 1 - i
            # end = start + N
            # The slice of ac affected is ac[start : start + N]
            # The values to subtract are 2 * a[i] * a
            
            start = N - 1 - i
            end = start + N
            
            # Boundary checks
            start_idx = max(0, start)
            end_idx = min(2 * N - 1, end)
            
            # Adjust a slice
            a_start = start_idx - start
            a_end = a_start + (end_idx - start_idx)
            
            # Subtract
            # We are modifying ac in place for the simulation
            ac[start_idx:end_idx] -= 2 * a[i] * a[a_start:a_end]
            
            # Calculate new max
            # We don't need to recompute the whole max if we track it, but for simplicity 
            # and since N=400, max is fast.
            new_max_abs = np.max(np.abs(ac))
            
            # New s
            new_s = s - 2 * a[i]
            
            if new_s == 0:
                new_obj = np.inf
            else:
                new_obj = new_max_abs / (new_s * new_s)
                
            if new_obj < best_obj:
                # Accept the flip
                a[i] *= -1
                s = new_s
                best_obj = new_obj
                best_a = a.copy()
                best_ac = ac.copy()
                best_s = s
                best_max_abs = new_max_abs
                improved = True
            else:
                # Revert ac
                ac[start_idx:end_idx] += 2 * a[i] * a[a_start:a_end]
                # a[i] is not flipped in a, so no revert needed for a
            
        if not improved:
            break
            
    return best_a, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_a = None
    best_obj = np.inf
    
    rng = np.random.default_rng(42)
    
    # 1. Thue-Morse (Strong structural initializer)
    a_tm = _thue_morse(N)
    obj_tm, _ = compute_objective(a_tm)
    if obj_tm < best_obj:
        best_obj = obj_tm
        best_a = a_tm.copy()
    
    # 2. Random signs
    for trial in range(10):
        if time.time() > deadline:
            break
        a = rng.choice([-1.0, 1.0], size=N)
        obj, _ = compute_objective(a)
        if obj < best_obj:
            best_obj = obj
            best_a = a.copy()
            
    # 3. Sparse random
    for density in [0.3, 0.5]:
        for trial in range(5):
            if time.time() > deadline:
                break
            mask = rng.random(N) < density
            a = np.zeros(N)
            a[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            obj, _ = compute_objective(a)
            if obj < best_obj:
                best_obj = obj
                best_a = a.copy()
                
    # Run local optimization on the best candidate found so far
    # And potentially on Thue-Morse if it wasn't the best
    candidates = [best_a.copy()]
    if not np.array_equal(best_a, a_tm):
        candidates.append(a_tm)
        
    for cand in candidates:
        if time.time() > deadline:
            break
        opt_a, opt_obj = local_sign_flip_optimize_incremental(cand, max_rounds=100, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    # Final polish on the absolute best
    if time.time() < deadline - 5:
        opt_a, opt_obj = local_sign_flip_optimize_incremental(best_a, max_rounds=200, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    return best_a
# EVOLVE-BLOCK-END
