import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _score(U):
    U = sorted(set(U))
    if not U:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    # Compute sumset and difference set sizes using FFT for efficiency
    n = len(U)
    max_val = M
    # For sumset U+U, elements range from 0 to 2*M
    # For diffset U-U, elements range from -M to M (or 0 to 2*M if we shift)
    
    # Use convolution via FFT
    # Represent U as a binary vector of length max_val+1
    size_sum = 2 * M + 1
    size_diff = 2 * M + 1  # We'll shift diffset to 0..2M
    
    # Padding for FFT
    N = 1
    while N < size_sum:
        N *= 2
    
    u_vec = np.zeros(N, dtype=np.float64)
    for x in U:
        u_vec[x] = 1.0
    
    # Sumset: u_vec convolved with u_vec
    f = np.fft.fft(u_vec)
    f_sum = f * f
    c_sum = np.fft.ifft(f_sum).real
    sumset_size = np.sum(c_sum[:size_sum] >= 0.5)
    
    # Difference set: u_vec convolved with reversed u_vec
    u_rev = np.zeros(N, dtype=np.float64)
    for x in U:
        u_rev[M - x] = 1.0  # Shift so that diff x-y maps to (M-x)+(y) = M - (x-y)
    # Actually, standard diffset convolution:
    # Let v[x] = 1 if x in U.
    # (v * v_rev)[k] = sum_j v[j] v_rev[k-j] = sum_j v[j] v[M-(k-j)] = sum_j v[j] v[M-k+j]
    # This is messy.
    # Simpler: Use numpy convolve for small sizes, or just iterate for small M.
    # Since we are optimizing, M might be up to a few thousand. O(M^2) is too slow for large M in Python loops.
    # But FFT is O(M log M).
    # Let's do the diffset FFT correctly.
    # Diffset D = {a - b | a, b in U}.
    # Let U' = {x | x in U}.
    # The indicator function of U is f_U.
    # The indicator of U-U is the cross-correlation of f_U with itself.
    # Cross correlation (f_U *~ f_U)[k] = sum_j f_U[j] f_U[j-k].
    # This is equivalent to f_U convolved with reversed f_U.
    # Let's use the property that |U-U| = |U+(-U)|.
    # Let V = {-x | x in U}. Then U+V = {u - v | u, v in U}.
    # V is in range [-M, 0]. Shift V by M: V' = {M - x | x in U}. V' is in [0, M].
    # Then U+V' is in [M, 2M].
    # So we can convolve f_U (shifted to 0..M) with f_V' (shifted to 0..M).
    
    v_vec = np.zeros(N, dtype=np.float64)
    for x in U:
        v_vec[M - x] = 1.0
        
    f_v = np.fft.fft(v_vec)
    f_diff = f * f_v  # Convolution of U and V'
    c_diff = np.fft.ifft(f_diff).real
    # The result is shifted by M. The diffset is in 0..2M, but we only care about the count.
    diffset_size = np.sum(c_diff[:size_diff] >= 0.5)
    
    if sumset_size == 0 or diffset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    denom = math.log2(2 * M + 1)
    if denom == 0:
        return 0.0
    return 1 + math.log2(ratio) / denom

def _local_search(U, iterations=1000, deadline=None):
    U = sorted(set(U))
    if not U:
        return U
    best_score = _score(U)
    best_U = U[:]
    N = len(U)
    M = max(U)
    
    for i in range(iterations):
        if deadline and time.time() > deadline:
            break
        
        # Choose a random index to modify
        idx = np.random.randint(0, N)
        old_val = U[idx]
        
        # Propose a new value
        # Try adding a new element, removing one, or moving one
        if np.random.random() < 0.1 and N < 100:
            # Add new element
            new_val = np.random.randint(0, M + 10)
            U_new = U + [new_val]
        elif np.random.random() < 0.1 and N > 1:
            # Remove element
            U_new = U[:idx] + U[idx+1:]
        else:
            # Move element
            # Propose a new value close to old or random
            if np.random.random() < 0.5:
                new_val = max(0, old_val + np.random.randint(-5, 6))
            else:
                new_val = np.random.randint(0, M + 10)
            
            # Ensure uniqueness
            if new_val in U and new_val != old_val:
                continue
            
            U_new = U[:]
            U_new[idx] = new_val
            U_new = sorted(U_new)
        
        # Ensure 0 is in the set
        if 0 not in U_new:
            if len(U_new) > 0:
                U_new[0] = 0 # This is a hack, better to just reject or add 0
            else:
                continue
            
        # Actually, the contract says "containing 0".
        # If 0 is not in U_new, it's invalid.
        if 0 not in U_new:
            continue
            
        s = _score(U_new)
        if s > best_score:
            best_score = s
            best_U = U_new[:]
        else:
            # Accept worse with small probability (Simulated Annealing lite)
            if np.random.random() < 0.05:
                best_U = U_new[:]
                # Don't update best_score here, just move current state
                # For simplicity, let's stick to Hill Climbing first
                pass
        
        U = best_U[:]
        N = len(U)
        M = max(U)
        
    return best_U

def _global_search(deadline):
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Initialize with a few good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 2, 5, 7],
        [0, 1, 4, 8],
        [0, 1, 3, 6, 10],
        [0, 1, 2, 4, 8, 15],
        [0, 1, 4, 9, 16],
    ]
    
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_U = c[:]
            
    # Local search from best candidate
    time_left = deadline - time.time()
    if time_left > 5:
        best_U = _local_search(best_U, iterations=500, deadline=deadline - 5)
        s = _score(best_U)
        if s > best_score:
            best_score = s
            
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # Start with a small random or structured set
    U = _global_search(deadline)
    
    # If we have more time, do more local search
    time_left = deadline - time.time()
    if time_left > 10:
        U = _local_search(U, iterations=1000, deadline=deadline - 5)
        
    # Final polish
    time_left = deadline - time.time()
    if time_left > 5:
        U = _local_search(U, iterations=200, deadline=deadline - 2)

    return sorted(set(U))
# EVOLVE-BLOCK-END
