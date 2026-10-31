import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initialize with a Chirp-like sequence to have a good starting point
    # Zadoff-Chu: exp(1j * pi * i^2 / N)
    # We take the sign of the real part or imaginary part to get +/- 1
    i = np.arange(N)
    # Use a quadratic chirp
    phase = (np.pi * i**2) / N
    s = np.sign(np.cos(phase))
    # Ensure non-zero
    s[s == 0] = 1
    
    # Calculate initial autocorrelation
    # A[k] = sum(s[j] * s[j+k]) for j=0 to N-1-k
    A = np.zeros(N)
    for k in range(N):
        A[k] = np.sum(s[:N-k] * s[k:])
        
    max_side_lobe = np.max(np.abs(A[1:]))
    
    # Local Search
    # We try to flip bits to reduce the maximum absolute sidelobe
    # We maintain the autocorrelation array A
    
    best_s = s.copy()
    best_max = max_side_lobe
    
    # Pre-compute indices for efficient updates? 
    # Flipping s[i] affects A[k] for k where i < N-k (i+k < N) or i-k >= 0 (k <= i).
    # i.e., k in [0, N-1-i] (as first term) and k in [0, i] (as second term).
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        # Randomize the order of indices to avoid local minima cycles
        indices = np.random.permutation(N)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Flip s[idx]
            old_val = s[idx]
            s[idx] = -old_val
            
            # Update A
            # A[k] changes by:
            # Term 1: s[idx] * s[idx+k] for k=0 to N-1-idx
            # Term 2: s[idx-k] * s[idx] for k=0 to idx
            # Note: A[0] is always N, so we don't care about k=0 for the max sidelobe (k!=0)
            
            # For k > 0:
            # If idx + k < N: A[k] += -2 * old_val * s[idx+k]
            # If idx - k >= 0: A[k] += -2 * old_val * s[idx-k]
            
            # We need to check if the new max is lower.
            # Instead of updating all and checking max, we can just update and track.
            
            # Optimization: Only update if we think it might help?
            # Hard to know. Just update.
            
            # Update A for k=1 to N-1
            # k range for term 1: 1 to N-1-idx
            if N - 1 - idx >= 1:
                k_range_1 = np.arange(1, N - idx)
                A[k_range_1] += -2 * old_val * s[idx + k_range_1]
                
            # k range for term 2: 1 to idx
            if idx >= 1:
                k_range_2 = np.arange(1, idx + 1)
                A[k_range_2] += -2 * old_val * s[idx - k_range_2]
                
            # Calculate new max sidelobe
            # We can optimize this by only checking the changed k's and the previous max?
            # If the previous max k was not changed, it's still the max.
            # If the previous max k was changed, it might be lower.
            # New peaks might appear in the changed k's.
            
            # For simplicity and safety within time limit, compute max of A[1:]
            # N=400, this is very fast.
            current_max = np.max(np.abs(A[1:]))
            
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
                improved = True
                no_improve_count = 0
            else:
                # Revert
                s[idx] = old_val
                # Revert A
                if N - 1 - idx >= 1:
                    k_range_1 = np.arange(1, N - idx)
                    A[k_range_1] -= -2 * old_val * s[idx + k_range_1]
                if idx >= 1:
                    k_range_2 = np.arange(1, idx + 1)
                    A[k_range_2] -= -2 * old_val * s[idx - k_range_2]
                    
                no_improve_count += 1
        
        if no_improve_count > N:
            # Restart with new random init if stuck
            s = np.random.choice([-1, 1], N)
            for k in range(N):
                A[k] = np.sum(s[:N-k] * s[k:])
            best_max = np.max(np.abs(A[1:]))
            best_s = s.copy()
            no_improve_count = 0
            
    return best_s

# EVOLVE-BLOCK-END
