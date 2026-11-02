import numpy as np
import time

N = 400
N_FFT = 1024

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=N_FFT)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=N_FFT)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    """Calculate the objective value: max absolute autocorrelation for lags 1..N-1."""
    return float(np.max(np.abs(A[1:])))

def _apply_flip(s, A, i):
    """
    Apply a single bit flip at index i to s and A.
    Returns the new A (modified in place or new array? Let's return new A to keep it clean, 
    but for performance, in-place is better if we are careful. 
    However, we need to compare E_new with E_old.
    We can compute delta and add to A.
    """
    # We need to update A based on the current s BEFORE flipping s[i]
    old_si = s[i]
    
    # Delta calculation
    # A_new[k] = A[k] - 2*old_si*(s[i+k] + s[i-k])
    
    # We create a temporary delta array
    # To avoid allocation overhead, we can reuse a buffer, but for clarity:
    
    # Part 1: s[i+k] for k=0..N-1-i
    # s[i:N] has length N-i
    # Part 2: s[i-k] for k=0..i
    # s[i::-1] has length i+1
    
    # We can do this with slicing on a pre-allocated delta array?
    # Let's just do it directly.
    
    # Optimization: 
    # The change is localized? No, it affects all k.
    
    # Let's construct the new A
    # A is an array of floats.
    
    # Vectorized update:
    # delta = np.zeros(N)
    # This allocation is expensive.
    
    # Alternative:
    # We know the formula.
    # Let's just recompute A from scratch if we do many flips?
    # No, incremental is better.
    
    # Let's try to do it without allocating a new N-sized array every time.
    # We can update A in place, but we need the old A to calculate delta?
    # No, delta depends on s, not A.
    # So we can compute delta, add to A, then flip s.
    
    # To minimize allocation:
    # We can use a pre-allocated delta buffer?
    # But the length of the slices varies.
    
    # Let's stick to a simple implementation first and see if it's fast enough.
    # If not, we optimize.
    
    # Actually, we can compute the new A values directly for the affected indices?
    # No, all indices are affected.
    
    # Let's use a trick:
    # A_new = A - 2 * old_si * (convolution of indicator of i with s)
    
    # Let's just do the slice operations.
    
    # Note: s is int array, A is float array.
    
    # Update A
    # A[:N-i] -= 2.0 * old_si * s[i:N]
    # A[:i+1] -= 2.0 * old_si * s[i::-1]
    
    # Wait, this modifies A in place.
    # We need to check if this is correct.
    # A[k] = sum s[j]s[j+k]
    # If we flip s[i] to -s[i]:
    # Terms involving s[i]:
    # 1. j=i: s[i]s[i+k] -> -s[i]s[i+k]. Change: -2 s[i]s[i+k].
    # 2. j+k=i => j=i-k: s[i-k]s[i] -> s[i-k](-s[i]). Change: -2 s[i-k]s[i].
    # 
    # For a given k:
    # If i+k < N, term 1 contributes -2 s[i]s[i+k].
    # If i-k >= 0, term 2 contributes -2 s[i-k]s[i].
    #
    # So A_new[k] = A[k] - 2 s[i] ( s[i+k] * [i+k<N] + s[i-k] * [i-k>=0] )
    #
    # The slices:
    # For k in 0..N-1-i: s[i+k] is s[i:N][k].
    # So A[0:N-i] -= 2 * old_si * s[i:N]
    #
    # For k in 0..i: s[i-k] is s[i::-1][k].
    # So A[0:i+1] -= 2 * old_si * s[i::-1]
    #
    # This looks correct.
    
    A[:N-i] -= 2.0 * old_si * s[i:N]
    A[:i+1] -= 2.0 * old_si * s[i::-1]
    
    # Flip s
    s[i] = -old_si
    
    return A

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_val = float('inf')
    best_seq = None
    
    # Simulated Annealing Parameters
    # We will do multiple restarts
    num_restarts = 20
    steps_per_restart = 5000
    
    # If time allows, we can do more.
    # Let's estimate time for one restart.
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # SA Loop
        T = 10.0  # Initial temperature
        T_min = 0.01
        alpha = 0.99  # Cooling rate
        
        # We need to be careful with time per step.
        # Let's run a fixed number of steps or until time limit.
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break
            
            # Choose move: 1, 2, or 3 bit flip
            # Probabilities: 0.6 for 1, 0.3 for 2, 0.1 for 3
            u = np.random.random()
            if u < 0.6:
                m = 1
            elif u < 0.9:
                m = 2
            else:
                m = 3
                
            # Choose random indices
            indices = np.random.choice(N, size=m, replace=False)
            
            # Save state to revert if rejected
            # We need to save s and A?
            # A is updated in place. 
            # We can save the old s values and the old A?
            # A is large. 
            # Alternative: Compute delta_A for the whole move, check E_new, then apply or revert.
            
            # Let's compute the new A in a temporary array?
            # That requires copying A (O(N)).
            # Or we can apply the flips, compute E, and if rejected, revert.
            # Reverting is the same cost as applying.
            # So total cost is 2 * cost(move).
            
            # Let's try:
            # 1. Save old s values
            old_s_vals = s[indices].copy()
            
            # 2. Apply flips
            for i in indices:
                _apply_flip(s, A, i)
                
            # 3. Compute new E
            new_val = _max_abs_acf_from_array(A)
            
            # 4. Accept/Reject
            delta_E = new_val - current_val
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_seq = s.copy()
            else:
                # Revert
                for i in indices:
                    _apply_flip(s, A, i)
                    # Note: _apply_flip flips the sign. 
                    # If we applied it once, applying it again reverts it.
                    # But we must ensure s is reverted correctly.
                    # _apply_flip uses s[i] to calculate delta.
                    # If we revert, we are flipping back.
                    # The logic in _apply_flip is symmetric.
                    
            # Cool down
            T = max(T * alpha, T_min)
            
            # Periodically check time
            if step % 100 == 0 and time.time() > deadline - 5.0:
                break

    if best_seq is None:
        # Fallback
        best_seq = np.ones(N)
        
    return best_seq
