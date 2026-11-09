import bisect
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    # 2. DP Setup
    # L[i][j] = max length of valid subsequence ending with pair (primes[j], primes[i])
    # prev[i][j] = k such that L[i][j] = L[j][k] + 1
    # Use numpy arrays for faster final scan
    L = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)
    
    # We process j from 0 to n-1
    # For a fixed j, we need to answer queries for all i > j
    # The queries depend on L[j][k] for k < j
    # So we build a structure for L[j][0..j-1]
    
    for j in range(n):
        if time.time() > deadline:
            break
            
        if j == 0:
            # No k < 0, so L[i][0] will be 2 for all i > 0
            # We handle this in the loop below or just set defaults
            pass
        else:
            # Build Segment Tree for L[j][0..j-1]
            # We need max value and index
            size = 1
            while size < j:
                size *= 2
            # Tree arrays: val, idx
            tree_val = [0] * (2 * size)
            tree_idx = [-1] * (2 * size)
            
            # Fill leaves
            for k in range(j):
                tree_val[size + k] = L[j][k]
                tree_idx[size + k] = k
            for k in range(j, size):
                tree_val[size + k] = 0
                tree_idx[size + k] = -1
                
            # Build tree
            for p in range(size - 1, 0, -1):
                left = 2 * p
                right = 2 * p + 1
                if tree_val[left] >= tree_val[right]:
                    tree_val[p] = tree_val[left]
                    tree_idx[p] = tree_idx[left]
                else:
                    tree_val[p] = tree_val[right]
                    tree_idx[p] = tree_idx[right]
                    
            def query(l, r):
                # Query max in [l, r] inclusive
                if l > r:
                    return 0, -1
                l += size
                r += size
                max_val = 0
                max_idx = -1
                while l <= r:
                    if l % 2 == 1:
                        if tree_val[l] > max_val:
                            max_val = tree_val[l]
                            max_idx = tree_idx[l]
                        l += 1
                    if r % 2 == 0:
                        if tree_val[r] > max_val:
                            max_val = tree_val[r]
                            max_idx = tree_idx[r]
                        r -= 1
                    l //= 2
                    r //= 2
                return max_val, max_idx
            # Build Segment Tree for L[j][0..j-1]
            # We need max value and index
            size = 1
            while size < j:
                size *= 2
            # Tree arrays: val, idx
            tree_val = [0] * (2 * size)
            tree_idx = [-1] * (2 * size)
            
            # Fill leaves
            for k in range(j):
                tree_val[size + k] = L[j][k]
                tree_idx[size + k] = k
            for k in range(j, size):
                tree_val[size + k] = 0
                tree_idx[size + k] = -1
                
            # Build tree
            for p in range(size - 1, 0, -1):
                left = 2 * p
                right = 2 * p + 1
                if tree_val[left] >= tree_val[right]:
                    tree_val[p] = tree_val[left]
                    tree_idx[p] = tree_idx[left]
                else:
                    tree_val[p] = tree_val[right]
                    tree_idx[p] = tree_idx[right]
                    
            def query(l, r):
                # Query max in [l, r] inclusive
                if l > r:
                    return 0, -1
                l += size
                r += size
                max_val = 0
                max_idx = -1
                while l <= r:
                    if l % 2 == 1:
                        if tree_val[l] > max_val:
                            max_val = tree_val[l]
                            max_idx = tree_idx[l]
                        l += 1
                    if r % 2 == 0:
                        if tree_val[r] > max_val:
                            max_val = tree_val[r]
                            max_idx = tree_idx[r]
                        r -= 1
                    l //= 2
                    r //= 2
                return max_val, max_idx

        # Iterate i > j with batched time check
        check_interval = 500  # check time every 500 iterations
        for idx, i in enumerate(range(j + 1, n)):
            if idx % check_interval == 0 and time.time() > deadline:
                break
                
            gap_ij = primes[i] - primes[j]
            # We need k < j such that primes[j] - primes[k] <= gap_ij
            # primes[k] >= primes[j] - gap_ij
            target = primes[j] - gap_ij
            
            # Find smallest k in [0, j-1] such that primes[k] >= target
            k_min = bisect.bisect_left(primes, target, 0, j)
            
            if k_min < j:
                best_val, best_k = query(k_min, j - 1)
                if best_val > 0:
                    L[i][j] = best_val + 1
                    prev[i][j] = best_k
                else:
                    L[i][j] = 2
                    prev[i][j] = -1
            else:
                L[i][j] = 2
                prev[i][j] = -1

    # Find the best end state using numpy
    max_len = int(L.max())
    if max_len == 0:
        return primes[:1] if n > 0 else []
    
    # Find position of max (upper triangle only, but max is unique enough)
    flat_idx = int(np.argmax(L))
    best_i = flat_idx // n
    best_j = flat_idx % n
    
    # Backtrack
    seq_indices = []
    cur_i = best_i
    cur_j = best_j
    while cur_j != -1:
        seq_indices.append(cur_i)
        next_k = int(prev[cur_i][cur_j])
        cur_i = cur_j
        cur_j = next_k
    seq_indices.append(cur_i)
    
    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result
