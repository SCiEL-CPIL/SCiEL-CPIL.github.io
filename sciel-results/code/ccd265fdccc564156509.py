import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # Use NumPy for efficient sieve-based totient computation
    # We can afford a larger N with NumPy. 10^7 is feasible.
    N = 10000000  # 10 million
    
    # Initialize phi array
    phi_arr = np.arange(N + 1, dtype=np.int64)
    
    # Sieve for Euler's totient function
    # Iterate through primes
    for i in range(2, N + 1):
        if phi_arr[i] == i:  # i is prime
            # For all multiples of i, phi[j] = phi[j] * (1 - 1/i)
            # Vectorized update
            multiples = np.arange(i, N + 1, i)
            phi_arr[multiples] -= phi_arr[multiples] // i
            
    # Convert to list for faster indexing in the sliding window loop? 
    # Or keep as numpy array? Numpy indexing is slightly slower than list for single elements, 
    # but the sieve was the bottleneck. Let's keep it as numpy array for memory locality if needed, 
    # but actually a list of ints is faster for random access in Python loops.
    # However, converting 10^7 elements to a list takes time. 
    # Let's stick to numpy array and use it directly.
    
    # Find longest run of distinct totients in [1, N]
    best_k = 1
    best_s = 1
    
    # Sliding window approach
    # We want to find the max k such that phi[s], ..., phi[s+k-1] are distinct.
    # We can iterate s from 1 to N.
    # To make it faster, we can break early if the remaining range is less than best_k.
    
    # Optimization: 
    # A naive O(N * K) scan is too slow if K is large and N is 10^7.
    # But K is likely small (e.g., < 50).
    # Let's use a set-based check for each start s.
    
    # Alternative: 
    # We can scan for collisions. 
    # If phi[i] == phi[j] with i < j, then any window [s, e] containing both i and j is invalid.
    # This is complex.
    
    # Let's stick to the simple scan but optimized.
    # Since N=10^7, and we check up to 200 values, that's 2*10^9 operations worst case, which is too slow in Python.
    # We need to be smarter.
    
    # Observation: Collisions are rare for small n, but become more frequent?
    # Actually, phi(n) is even for n > 2.
    # phi(n) = n-1 for primes.
    # phi(2p) = p-1.
    # Collisions like phi(3)=2, phi(4)=2.
    
    # Let's limit the window size to something reasonable, say 100, initially.
    # And break early if we find a collision.
    
    # To speed up the Python loop, we can skip steps or use a heuristic.
    # But let's try a direct scan first with a tighter deadline.
    # If N=10^7, and average run length is ~10, then 10^8 operations. Might be too slow.
    
    # Let's reduce N to 2*10^6 if time is tight, or optimize the inner loop.
    # Actually, the previous program did N=10^6 in pure Python.
    # NumPy sieve is ~100x faster.
    # So N=10^7 sieve takes ~0.5s.
    # The search loop is the bottleneck.
    
    # Let's try N=2000000 to be safe with the search loop time.
    # Or N=5000000.
    
    # Let's reset N based on time budget.
    # If we have 115s, and sieve takes ~1s for 10^7, we have ~110s for search.
    # 10^7 * 20 (avg check) = 2*10^8 ops. In Python, ~20-40s. This is risky.
    # Let's use N=5000000.
    
    # Re-doing with N=5000000
    N = 5000000
    phi_arr = np.arange(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if phi_arr[i] == i:
            multiples = np.arange(i, N + 1, i)
            phi_arr[multiples] -= phi_arr[multiples] // i
            
    # Search
    # We can convert to a list of ints for faster access in the tight loop?
    # List access is faster than numpy scalar access.
    # phi_list = phi_arr.tolist()
    # This conversion takes ~0.1s. Worth it.
    phi_list = phi_arr.tolist()
    
    best_k = 1
    best_s = 1
    
    # We can stop early if s + best_k > N
    for s in range(1, N - best_k):
        if time.time() - start_time > deadline - 5:
            break
        seen = set()
        k = 0
        # Limit k to 200 as per task, but we can break earlier if we find collision
        # Also, if s + k > N, break
        while s + k <= N and k < 200:
            val = phi_list[s + k]
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            # If we hit the max possible k (200), we can't do better in this range
            if best_k == 200:
                break

    # Now use remaining time to do random sampling at larger scales
    # This is a fallback, but the sieve result is likely better.
    remaining = deadline - (time.time() - start_time)
    if remaining > 2:
        import random
        random.seed(42)
        # Try ranges around multiples of small LCMs or just random large starts
        for _ in range(100):
            if time.time() - start_time > deadline - 1:
                break
            # Choose a random start point
            scale = random.choice([100000000, 1000000000, 10000000000, 100000000000])
            s = random.randint(1, max(1, 10**12 - 200))
            
            # Check a window of up to 200 values
            # We need a fast phi function for large numbers
            def phi(x):
                result, d, y = x, 2, x
                while d * d <= y:
                    if y % d == 0:
                        result -= result // d
                        while y % d == 0:
                            y //= d
                    d += 1
                return result - result // y if y > 1 else result

            seen = set()
            k = 0
            for i in range(200):
                val = s + i
                if val > 10**12:
                    break
                pv = phi(val)
                if pv in seen:
                    break
                seen.add(pv)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
