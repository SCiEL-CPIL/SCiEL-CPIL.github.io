"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from scipy.signal import fftconvolve

    start_time = time.time()
    TIME_LIMIT = 110  # Leave 10s margin

    # Increase N_MAX to find more representations
    # Memory constraint: FFT size ~ 2*N_MAX. 
    # 2*20M = 40M. Next power of 2 is 67M. 
    # 67M * 8 bytes = 536MB. Safe.
    N_MAX = 20_000_000

    # Generate powerful numbers up to N_MAX
    pw_set = set()
    
    max_b = int(N_MAX ** (1/3)) + 2
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_square_free = True
        d = 2
        temp_b = b
        while d * d <= temp_b:
            if temp_b % (d * d) == 0:
                is_square_free = False
                break
            d += 1
        if is_square_free:
            b3 = b * b * b
            max_a = int((N_MAX // b3) ** 0.5) + 1
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= N_MAX:
                    pw_set.add(n)
        
        if time.time() - start_time > TIME_LIMIT - 10:
            break

    # Create array for convolution
    size = 1
    while size < 2 * N_MAX:
        size *= 2
        
    A = np.zeros(size, dtype=np.float64)
    for p in pw_set:
        A[p] = 1.0
        
    # Compute convolution to count sums
    C = fftconvolve(A, A, mode='full')
    
    # Find N that maximizes the count
    # We only care about N up to 2 * N_MAX
    max_idx = np.argmax(C[:2 * N_MAX])
    best_N = int(max_idx)
    
    # Retrieve pairs
    pairs = []
    pw_set_local = pw_set
    # Iterate through powerful numbers
    # To be efficient, we can just iterate through the set
    for a in pw_set:
        if a > best_N - a:
            continue
        b_val = best_N - a
        if b_val in pw_set_local:
            pairs.append([a, b_val])
            
    # Sort pairs for consistency (though order might not matter for score, good practice)
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
