import math
import time
import numpy as np
from collections import Counter

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to 'limit'.
    A powerful number is of the form x^2 * y^3 where y is square-free.
    """
    powerful = set()
    # y^3 <= limit => y <= limit^(1/3)
    max_y = int(math.ceil(limit ** (1.0/3.0))) + 2
    
    # Sieve for square-free numbers up to max_y
    sq_free = [True] * (max_y + 1)
    sq_free[0] = False
    sq_free[1] = True
    p = 2
    while p * p <= max_y:
        if sq_free[p]:
            sq = p * p
            for j in range(sq, max_y + 1, sq):
                sq_free[j] = False
        p += 1
    
    for y in range(1, max_y + 1):
        if not sq_free[y]:
            continue
        y3 = y * y * y
        if y3 > limit:
            break
        # x^2 * y^3 <= limit => x <= sqrt(limit / y^3)
        max_x = int(math.isqrt(limit // y3))
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerful.add(n)
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    best_n = 0
    best_pairs = []
    
    # We can go much higher than 10^7 with sparse convolution.
    # 10^8 powerful numbers count is ~15,000. 
    # Pairs ~ 15000^2 / 2 = 112.5M operations. In Python this might be slow (~5-10s).
    # Let's try 5*10^7 first, then 10^8 if time permits.
    limits = [5 * 10**7, 10**8]
    
    for L in limits:
        if time.time() > deadline - 30:
            break
            
        try:
            P = generate_powerful_numbers(L)
            if not P:
                continue
                
            P_list = list(P)
            P_set = set(P)
            n_pows = len(P_list)
            
            # Sparse convolution using Counter
            # We want to count pairs (a, b) with a + b = N.
            # To avoid double counting and handle a==b, we can just count all pairs
            # and then adjust, or just count a <= b.
            # Given n_pows ~ 15k, n_pows^2 is ~225M. This is too slow in pure Python loops.
            # We need to be smarter.
            
            # Alternative: Iterate over a in P, b in P.
            # Actually, 15k^2 is 225 million. In Python, a simple loop takes ~50ns per op?
            # 225M * 50ns = 11 seconds. This is risky.
            # Let's stick to L=2*10^7 or 5*10^7 but optimize the inner loop.
            
            # If we use L=5*10^7, n_pows is ~8000. 8000^2 = 64M. ~3 seconds.
            # If we use L=10^8, n_pows is ~15000. 15000^2 = 225M. ~11 seconds.
            
            # Let's try to fit as many as possible.
            # We will use a list of lists or a dictionary to store counts.
            # Dictionary of sum -> count.
            sum_counts = Counter()
            
            # Optimization: Use local variables for speed
            p_list = P_list
            n = len(p_list)
            
            # If n is large, this loop is the bottleneck.
            # We can break early if time is running out.
            # But we need the full distribution to find the max.
            
            # Let's try a hybrid: 
            # If n > 10000, we might timeout. 
            # Let's cap L such that n_pows is manageable.
            # 10^7 -> ~3000 pows. 3000^2 = 9M. Fast.
            # 5*10^7 -> ~8000 pows. 8000^2 = 64M. Medium.
            # 10^8 -> ~15000 pows. 15000^2 = 225M. Slow.
            
            # Let's stick to 5*10^7 for safety, or even 10^8 if we optimize.
            # Actually, the current program uses FFT which is O(N log N) where N is the range.
            # For L=10^7, N=2*10^7. FFT is fast.
            # For L=5*10^7, N=10^8. FFT is ~1s.
            # The sparse method is O(K^2).
            # For K=8000, K^2=64M.
            # Python loops are slow. 64M iterations might take 5-10s.
            # Let's try it.
            
            # To speed up, we can use a list for P and nested loops.
            # We only care about sums up to 2L.
            
            # Let's check time before starting the heavy loop
            t_start_loop = time.time()
            
            # We will iterate a from P, b from P.
            # To ensure a <= b, we can iterate b from index of a onwards?
            # No, P is sorted. 
            # for i, a in enumerate(P):
            #    for b in P[i:]:
            #       sum_counts[a+b] += 1
            # This counts each pair {a,b} with a<=b exactly once.
            
            # This is efficient for counting distinct pairs.
            
            for i in range(n):
                a = p_list[i]
                # If a > L, we might have sums > 2L? No, max sum is 2L.
                # We want to find N.
                # Inner loop
                # Using local variable for sum_counts update
                sc = sum_counts
                # b must be >= a
                for j in range(i, n):
                    b = p_list[j]
                    s = a + b
                    # We don't need to check s <= 2L because a,b <= L
                    sc[s] += 1
                    
                # Periodic time check?
                # Checking time inside the inner loop is expensive.
                # Checking every 1000 outer iterations?
                if i % 1000 == 0:
                    if time.time() > deadline - 10:
                        break
                        
            if time.time() > deadline - 10:
                continue # Skip verification if we ran out of time during counting
                
            # Find top candidates
            # Get most common sums
            # Counter.most_common() is O(K log K) or O(K) for top K?
            # It sorts all. K is number of unique sums.
            # Unique sums are up to 2L, but only those formed by pairs.
            # Number of unique sums is likely much smaller than 2L, maybe ~K^2/2?
            # Actually, sums are dense in the middle.
            # Let's just get the top 20.
            
            top_sums = sum_counts.most_common(20)
            
            current_best_n = 0
            current_best_pairs = []
            
            for N, count in top_sums:
                if N < 2:
                    continue
                
                # Verify and collect pairs
                pairs = []
                # We need to find all pairs a <= b such that a+b=N and a,b in P.
                # We can iterate a in P.
                # If a > N/2, break.
                for a in P:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in P_set:
                        pairs.append([a, b])
                        
                if len(pairs) > len(current_best_pairs):
                    current_best_pairs = pairs
                    current_best_n = N
                    
            if len(current_best_pairs) > len(best_pairs):
                best_pairs = current_best_pairs
                best_n = current_best_n
                
        except Exception as e:
            continue
            
    # Fallback
    if not best_pairs:
        best_n = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
