import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe (150s wall clock)
    
    # Choose L based on time budget
    # We need to balance generation time and counting time.
    # Generation: O(L^(1/3) * L^(1/6)) roughly.
    # Counting: O(K^2) where K ~ L^(1/3).
    # Let's try L = 500_000_000.
    # K ~ 500M^(1/3) ~ 800.
    # K^2 ~ 640,000. Very fast.
    # Wait, K is number of powerful numbers.
    # Number of powerful numbers up to N is ~ c * N^(1/3).
    # For N=10^18, K ~ 10^6. K^2 is 10^12. Too slow.
    # For N=10^9, K ~ 1000. K^2 is 10^6. Fast.
    # So L should be around 10^9 or 10^10.
    
    L = 1_000_000_000  # 1e9
    
    # Generate powerful numbers up to L
    powerful = set()
    max_y = int(round(L ** (1/3))) + 2
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > L:
            break
        max_x = int(math.isqrt(L // y3)) + 1
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= L:
                powerful.add(val)
    
    P = sorted(powerful)
    K = len(P)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 10:
        # If generation took too long, we might need to reduce L?
        # But 1e9 should be fast.
        pass
    
    # Count sums using NumPy array for memory efficiency
    # Max sum is 2*L
    max_sum = 2 * L
    # Use int32, max count won't exceed 2^31
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Iterate pairs
    P_list = P
    counts_arr = counts
    K_local = K
    
    # To speed up, we can use vectorization for the inner loop?
    # For each i, a = P[i]
    # b = P[i:]
    # sums = a + b
    # np.add.at(counts, sums, 1)
    # This might be faster than a Python loop.
    
    for i in range(K_local):
        a = P_list[i]
        b_slice = P_list[i:]
        sums = a + b_slice
        np.add.at(counts_arr, sums, 1)
        
        # Check time every 100 iterations of i
        if i % 100 == 0 and time.time() > deadline - 10:
            break
    
    # Find max
    best_N = int(np.argmax(counts_arr))
    best_count = int(counts_arr[best_N])
    
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Reconstruct pairs
    P_set = set(P_list)
    pairs = []
    for a in P_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
