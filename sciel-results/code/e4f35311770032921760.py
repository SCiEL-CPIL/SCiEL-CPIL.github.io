"""
Strategy: Sparse pairwise summation for powerful number pair counting.
Instead of FFT on a dense array of size O(L), we generate all powerful numbers 
up to L, compute the sum of every pair (i, j) with i <= j, and use np.unique 
with return_counts to find the most frequent sums. This avoids large memory 
allocations for FFT and may be faster for sparse signals if the number of 
powerful numbers P is small enough that P^2 is manageable.
For L=10^8, P is approx L^0.5 ~ 10^4. P^2 = 10^8 sums. This is feasible.
"""

# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np
import bisect

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    # b must be square-free and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    
    # Primes up to sqrt(max_b) for square-free check
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    
    return powerfuls

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit (10s buffer)
    
    # Target L = 10^8. 
    # Number of powerful numbers P ~ sqrt(L) * C. For L=10^8, P ~ 10^4 * C.
    # P^2 ~ 10^8. Computing 10^8 sums and sorting/unique-ing them is feasible in numpy.
    L = 100_000_000
    
    # Check if we have enough time for generation
    # Generation is O(L^(2/3)) roughly.
    powerfuls = _generate_powerfuls(L)
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls_list)
    
    elapsed_gen = time.time() - start_time
    if elapsed_gen > 10:
        # If generation was slow, reduce L
        L = 50_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        
    if time.time() > deadline - 20:
        # Fallback to small brute force if time is tight
        small_limit = 10000
        small_pows = [p for p in powerfuls_list if p <= small_limit]
        count_dict = {}
        pair_dict = {}
        for i in range(len(small_pows)):
            a = small_pows[i]
            for j in range(i, len(small_pows)):
                b = small_pows[j]
                s = a + b
                if s > small_limit * 2: break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
        return {'N': 2, 'pairs': [[1, 1]]}

    # Convert to numpy array for vectorized operations
    pows_arr = np.array(powerfuls_list, dtype=np.int64)
    P = len(pows_arr)
    
    # We want to find N = a + b with a <= b.
    # Iterate over i, and for each i, compute pows_arr[i] + pows_arr[i:]
    # This is O(P^2) memory if we store all sums, which is 10^8 * 8 bytes = 800MB.
    # This might be too much for some environments. 
    # Let's check memory. If P=10000, P^2/2 = 50M sums. 50M * 8 bytes = 400MB. 
    # This is usually acceptable.
    
    # However, if P is larger (e.g., L=10^9, P~30000, P^2/2 ~ 450M sums ~ 3.6GB), it might fail.
    # For L=10^8, P is around 10^4-2*10^4. Let's estimate P.
    # Number of powerful numbers up to x is ~ C * x^0.5.
    # For x=10^8, x^0.5 = 10000. C is approx 1.5-2? So P ~ 15000-20000.
    # P^2/2 ~ 2*10^8 sums. 2*10^8 * 8 bytes = 1.6GB. 
    # This is risky. 
    # Alternative: Do it in chunks or use a more memory-efficient method?
    # The current FFT approach uses 2*L * 8 bytes (float64) + FFT workspace.
    # For L=10^8, FFT size 2^28 = 268M. 268M * 8 = 2GB. 
    # So both are memory heavy.
    
    # Let's try a hybrid: 
    # If P is small enough, use sparse. 
    # If P is large, fall back to FFT with smaller L or just return best found so far?
    # Actually, the inspiration says "np.unique ... improved score ... at limit=10^8".
    # This implies it worked in the environment.
    
    # Let's optimize the sparse approach to use less memory if possible, or just try it.
    # We can compute sums in blocks to reduce peak memory, but we need all sums for global count.
    # We can accumulate counts in a dictionary? No, that's slow in Python.
    # We can use np.bincount? 
    # Max sum is 2*L = 2*10^8. 
    # np.bincount would require an array of size 2*10^8. 2*10^8 * 8 bytes = 1.6GB.
    # This is similar to FFT memory.
    # But np.bincount is very fast (C implementation).
    
    # Let's try np.bincount.
    # We need to generate all pairs (i, j) with i <= j.
    # This is still O(P^2) operations.
    
    # Optimization: 
    # Instead of creating a huge array of all sums, we can iterate over i and add pows_arr[i] to pows_arr[i:].
    # But we need to count frequencies.
    
    # Let's stick to the "unique on all sums" idea but be careful with memory.
    # If MemoryError, fall back to smaller L.
    
    try:
        # Generate all pair sums
        # To save memory, we can do this in chunks if P is large.
        # But let's first try the direct method for L=10^8.
        
        # If P is too large, we might run out of memory.
        # Estimate memory: P^2 * 8 bytes.
        # If P = 20000, P^2 = 4*10^8. 4*10^8 * 8 = 3.2GB. 
        # This is likely to fail on standard 8GB/16GB systems if other things are running.
        
        # Let's reduce L to be safe? 
        # The current program uses FFT with L=5*10^7 (size 2^27 ~ 134M, ~1GB).
        # The inspiration used L=10^8 with sparse.
        # Let's try L=10^8 but with a check.
        
        # Actually, let's use a smarter approach: 
        # We can compute the sums in a loop and update a counter array (bincount style) 
        # but we can't easily do that in numpy without storing all sums or doing it in Python loop (slow).
        
        # Let's try the direct unique approach but with a smaller L if memory is a concern.
        # Let's start with L = 50_000_000 to be safe, then try 100_000_000 if time permits?
        # No, the task is to maximize score. The inspiration found higher score at 10^8.
        
        # Let's try to generate sums in a memory-efficient way:
        # We can use a generator to yield sums, but np.unique needs an array.
        
        # Alternative: Use scipy.sparse? No, sums are dense enough.
        
        # Let's try L=10^8 and catch MemoryError.
        
        # To reduce memory, we can store sums as int32 if L < 2^31/2? 
        # L=10^8, sum < 2*10^8. int32 max is 2*10^9. So int32 is enough!
        # This halves memory: 3.2GB -> 1.6GB.
        
        pows_arr_int32 = pows_arr.astype(np.int32)
        
        # Generate all pairs i <= j
        # This is still P^2/2 elements.
        # If P=20000, 2*10^8 elements. 2*10^8 * 4 bytes = 800MB.
        # This is very safe!
        
        # How to generate all pairs efficiently?
        # np.triu_indices(P) returns indices i, j.
        # sums = pows_arr[i] + pows_arr[j]
        
        # np.triu_indices for P=20000 creates two arrays of size 2*10^8.
        # 2 * 2*10^8 * 4 bytes (int32 indices) = 1.6GB.
        # Plus sums array 800MB. Total 2.4GB.
        # This is acceptable.
        
        # But wait, np.triu_indices might be slow or memory heavy for large P.
        # Let's check P first.
        
        if P > 50000:
            # Too large for this method. Fall back to smaller L or FFT.
            # For now, let's just reduce L.
            L = 20_000_000
            powerfuls = _generate_powerfuls(L)
            powerfuls_list = sorted(powerfuls)
            powerfuls_set = set(powerfuls_list)
            pows_arr = np.array(powerfuls_list, dtype=np.int32)
            P = len(pows_arr)
            
        if time.time() > deadline - 30:
             # Fallback
             return {'N': 2, 'pairs': [[1, 1]]}

        # Generate indices
        # Using np.triu_indices
        i_idx, j_idx = np.triu_indices(P)
        
        # Compute sums
        sums = pows_arr[i_idx] + pows_arr[j_idx]
        
        # Find unique sums and counts
        # np.unique sorts the array. O(N log N).
        # N ~ 2*10^8. This might be slow.
        # Alternative: np.bincount.
        # Max sum is 2*L.
        # If L=10^8, max sum 2*10^8.
        # bincount array size 2*10^8. 2*10^8 * 4 bytes (int32) = 800MB.
        # bincount is O(N). Much faster than sort.
        
        max_sum = 2 * L
        # Ensure max_sum is within int32 range
        if max_sum > 2**31 - 1:
             # Use int64 for bincount? No, bincount returns int64 by default?
             # np.bincount returns int64.
             # Array size 2*10^8 * 8 bytes = 1.6GB.
             pass
        
        counts = np.bincount(sums, minlength=max_sum + 1)
        
        # Find the index with max count
        # We only care about N >= 2.
        # counts[0] and counts[1] are 0 or small.
        
        # Note: The count in `counts` is the number of pairs (a,b) with a+b=N, 
        # where a and b are from the list, and we included all i,j from triu, 
        # so a <= b is guaranteed by the indices?
        # Yes, triu_indices gives i <= j.
        # So counts[N] is exactly the number of valid pairs.
        
        best_N = int(np.argmax(counts))
        
        # Verify best_N
        if best_N < 2:
             best_N = 2
             
        # Extract pairs for best_N
        pairs = []
        half_N = best_N // 2
        end_idx = bisect.bisect_right(powerfuls_list, half_N)
        
        # powerfuls_list is sorted list of ints
        # We need to find all a in powerfuls_list[:end_idx] such that best_N - a is in powerfuls_set
        for a in powerfuls_list[:end_idx]:
            b = best_N - a
            if b in powerfuls_set:
                if a <= b:
                    pairs.append([int(a), int(b)])
        
        if pairs:
            return {'N': best_N, 'pairs': pairs}
        else:
            # Fallback if extraction failed (shouldn't happen)
            return {'N': 2, 'pairs': [[1, 1]]}
            
    except MemoryError:
        # Fallback to smaller L
        L = 20_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        pows_arr = np.array(powerfuls_list, dtype=np.int32)
        P = len(pows_arr)
        
        if P > 0:
            i_idx, j_idx = np.triu_indices(P)
            sums = pows_arr[i_idx] + pows_arr[j_idx]
            max_sum = 2 * L
            counts = np.bincount(sums, minlength=max_sum + 1)
            best_N = int(np.argmax(counts))
            if best_N < 2: best_N = 2
            
            pairs = []
            half_N = best_N // 2
            end_idx = bisect.bisect_right(powerfuls_list, half_N)
            for a in powerfuls_list[:end_idx]:
                b = best_N - a
                if b in powerfuls_set:
                    if a <= b:
                        pairs.append([int(a), int(b)])
            
            if pairs:
                return {'N': best_N, 'pairs': pairs}
                
        return {'N': 2, 'pairs': [[1, 1]]}
    except Exception:
        return {'N': 2, 'pairs': [[1, 1]]}

# EVOLVE-BLOCK-END
