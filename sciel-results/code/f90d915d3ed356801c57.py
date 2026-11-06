import numpy as np
from math import isqrt
import time

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s buffer
    
    # Step 1: Generate powerful numbers up to L
    # L = 10^8 gives ~10,000 numbers. 
    # Let's try L = 2 * 10^8 if time permits, but start with 10^8 for safety.
    L = 100000000  # 10^8
    
    # Precompute square-free numbers up to L^(1/3) ~ 464
    # Actually, we iterate b up to L^(1/3).
    # b^3 <= L => b <= L^(1/3)
    max_b = isqrt(isqrt(L)) + 2 # Approx, better to just use integer cube root logic or simple loop
    # Actually b^3 <= 10^8 => b <= 464.
    # Let's just loop b from 1 to 1000 and break if b^3 > L.
    
    # Sieve for square-free check up to 1000
    limit_sf = 1000
    is_square_free = [True] * (limit_sf + 1)
    is_square_free[0] = False
    is_square_free[1] = True
    for p in range(2, isqrt(limit_sf) + 1):
        if is_square_free[p]:
            for j in range(p*p, limit_sf + 1, p*p):
                is_square_free[j] = False
    # Wait, square-free means no square factor.
    # My sieve above marks multiples of p^2.
    # This is correct.
    
    powerful = set()
    # Iterate b
    b = 1
    while b**3 <= L:
        if is_square_free[b]:
            b3 = b**3
            # a^2 * b^3 <= L => a <= sqrt(L / b^3)
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                val = a*a * b3
                if val <= L:
                    powerful.add(val)
        b += 1
        if time.time() > deadline:
            break
            
    P = sorted(powerful)
    # If P is too large for numpy outer, truncate?
    # Or just proceed.
    
    if not P:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    P_arr = np.array(P, dtype=np.int64)
    
    # Check memory/time for outer sum
    # If len(P) is ~10000, 10000x10000 is 100M elements.
    # np.int64 is 8 bytes -> 800MB.
    # np.int32 is 4 bytes -> 400MB.
    # Max sum is 2*L = 2*10^8 < 2^31. So int32 is safe.
    
    # Time check
    if time.time() > deadline - 10:
         # Fallback to smaller set or naive
         pass

    try:
        # Compute sums
        # To save memory, we can do it in chunks or just try it.
        # Let's try int32.
        S = np.add.outer(P_arr.astype(np.int32), P_arr.astype(np.int32))
        
        # Count occurrences
        # np.unique on 100M elements might be slow/memory heavy.
        # Alternative: use a dictionary on the flat array? No, 100M entries.
        # Better: np.bincount if range is small?
        # Range is 0 to 2*L = 2*10^8.
        # Bincount array of size 2*10^8 is 800MB (int64) or 400MB (int32).
        # This is feasible.
        
        max_sum = 2 * L
        counts = np.bincount(S.flatten(), minlength=max_sum + 1)
        
        # Find the sum with max count
        # We want N >= 2.
        # Also we want distinct pairs a<=b.
        # The count from bincount counts all ordered pairs (a,b) and (b,a) and (a,a).
        # If a != b, count is 2.
        # If a == b, count is 1.
        # We want the number of unordered pairs {a,b} with a<=b.
        # Let C_N be the total count of pairs (a,b) such that a+b=N.
        # Let k be the number of pairs with a<b.
        # Let m be 1 if N is even and N/2 is in P, else 0.
        # Then C_N = 2*k + m.
        # The number of distinct pairs a<=b is k + m.
        # Score = k + m.
        
        # We can just find the max C_N, then calculate score.
        # Or we can just find the max score directly?
        # It's easier to find max C_N, then verify.
        
        # However, bincount gives counts for ALL sums.
        # We need to find the N that maximizes (C_N + m_N) / 2 ? No.
        # Score = (C_N + m_N) // 2 is not quite right if C_N is odd.
        # If C_N is odd, m_N=1, Score = (C_N+1)/2.
        # If C_N is even, m_N=0, Score = C_N/2.
        # So Score = (C_N + (C_N % 2)) // 2.
        
        # Let's find the N with max Score.
        # We can iterate over candidate Ns?
        # There are 2*10^8 candidates.
        # Iterating 2*10^8 in Python is slow (10-20s).
        # We can use numpy to compute scores for all N?
        # m_N is 1 if N is even and N/2 in P.
        # We can create an array `is_in_P` of size L+1.
        is_in_P = np.zeros(L + 1, dtype=bool)
        is_in_P[P_arr] = True
        
        # m_N array:
        # m_N = 1 if N is even and is_in_P[N//2]
        # We can compute this for all N.
        # N ranges from 0 to 2*L.
        # This is doable with numpy.
        
        N_max = 2 * L
        # Initialize m array
        m = np.zeros(N_max + 1, dtype=np.int32)
        # Even indices
        even_indices = np.arange(2, N_max + 1, 2)
        half_indices = even_indices // 2
        # Check if half_indices are in P
        # half_indices max is L.
        valid_half = half_indices <= L
        # m[even_indices[valid_half]] = is_in_P[half_indices[valid_half]]
        # Careful with broadcasting.
        m_vals = is_in_P[half_indices[valid_half]]
        m[even_indices[valid_half]] = m_vals.astype(np.int32)
        
        # Calculate score for each N
        # C_N is counts[N]
        # Score = (C_N + m_N) // 2
        # We want max Score.
        
        # To avoid division, maximize 2*Score = C_N + m_N
        total_score_2 = counts + m
        
        # Find max
        # Ignore N < 2
        total_score_2[:2] = 0
        
        best_N = np.argmax(total_score_2)
        best_score_2 = total_score_2[best_N]
        
        if best_score_2 == 0:
             return {'N': 2, 'pairs': [[1, 1]]}
             
        best_N = int(best_N)
        
        # Reconstruct pairs
        pairs = []
        P_set = set(P)
        for a in P:
            b = best_N - a
            if b < a:
                continue
            if b in P_set:
                pairs.append([a, b])
                
        # Verify time
        if time.time() > deadline:
             # Fallback?
             pass

        return {'N': best_N, 'pairs': pairs}
        
    except MemoryError:
        # Fallback to smaller L
        L = 50000000
        powerful = set()
        b = 1
        while b**3 <= L:
            if is_square_free[b]:
                b3 = b**3
                max_a = isqrt(L // b3)
                for a in range(1, max_a + 1):
                    val = a*a * b3
                    if val <= L:
                        powerful.add(val)
            b += 1
        P = sorted(powerful)
        P_arr = np.array(P, dtype=np.int32)
        S = np.add.outer(P_arr, P_arr)
        max_sum = 2 * L
        counts = np.bincount(S.flatten(), minlength=max_sum + 1)
        
        is_in_P = np.zeros(L + 1, dtype=bool)
        is_in_P[P_arr] = True
        m = np.zeros(max_sum + 1, dtype=np.int32)
        even_indices = np.arange(2, max_sum + 1, 2)
        half_indices = even_indices // 2
        valid_half = half_indices <= L
        m_vals = is_in_P[half_indices[valid_half]]
        m[even_indices[valid_half]] = m_vals.astype(np.int32)
        
        total_score_2 = counts + m
        total_score_2[:2] = 0
        best_N = int(np.argmax(total_score_2))
        
        pairs = []
        P_set = set(P)
        for a in P:
            b = best_N - a
            if b < a:
                continue
            if b in P_set:
                pairs.append([a, b])
        
        return {'N': best_N, 'pairs': pairs}
