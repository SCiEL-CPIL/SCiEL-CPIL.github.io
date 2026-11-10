import numpy as np

def construct():
    N = 10000000
    
    # Compute sigma(n) for all n using a sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    # Using the property: sigma is the sum of divisors.
    # sigma[n] = sum_{d|n} d
    # We can compute this by iterating d from 1 to N and adding d to all multiples of d.
    # This is O(N log N).
    # To speed up, we use numpy slicing.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Search for k starting from 1
    # Error = k / (2 * b)
    # We want to minimize this.
    # Iterate k from 1 upwards.
    # For each k, find the largest b such that there exists a valid a with |2a - 3b| = k and sigma(a) == sigma(b).
    
    max_k_search = 1000
    max_b_search_depth = 20000 # Check this many largest b's for each k
    
    for k in range(1, max_k_search + 1):
        # Case 1: 2a - 3b = k => 2a = 3b + k => a = (3b + k) / 2
        # Conditions:
        # 1. 3b + k must be even => b and k must have same parity.
        # 2. 1 <= a <= N => 2 <= 3b + k <= 2N => (2-k)/3 <= b <= (2N-k)/3
        # 3. 1 <= b <= N
        
        parity_k = k % 2
        
        # Max b for Case 1
        b_max_1 = (2 * N - k) // 3
        if b_max_1 > N:
            b_max_1 = N
        # Adjust b_max_1 to have same parity as k
        if b_max_1 % 2 != parity_k:
            b_max_1 -= 1
            
        if b_max_1 >= 1:
            # Search downwards from b_max_1
            # We check b_max_1, b_max_1 - 2, b_max_1 - 4, ...
            # Limit search depth
            for i in range(max_b_search_depth):
                b = b_max_1 - 2 * i
                if b < 1:
                    break
                # Calculate a
                a = (3 * b + k) // 2
                if a > N or a < 1:
                    continue
                if sigma[a] == sigma[b]:
                    return [int(a), int(b)]
                    
        # Case 2: 3b - 2a = k => 2a = 3b - k => a = (3b - k) / 2
        # Conditions:
        # 1. 3b - k must be even => b and k must have same parity.
        # 2. 1 <= a <= N => 2 <= 3b - k <= 2N => (2+k)/3 <= b <= (2N+k)/3
        # 3. 1 <= b <= N
        
        # Max b for Case 2
        b_max_2 = (2 * N + k) // 3
        if b_max_2 > N:
            b_max_2 = N
        # Adjust b_max_2 to have same parity as k
        if b_max_2 % 2 != parity_k:
            b_max_2 -= 1
            
        if b_max_2 >= 1:
            for i in range(max_b_search_depth):
                b = b_max_2 - 2 * i
                if b < 1:
                    break
                # Calculate a
                a = (3 * b - k) // 2
                if a > N or a < 1:
                    continue
                if sigma[a] == sigma[b]:
                    return [int(a), int(b)]
                    
    # If no solution found with small k, fallback to a random or heuristic search
    # Or just return a valid pair found by a slower method?
    # The previous method found a pair. We should ensure we return something valid.
    # Let's do a quick random search or return the best from a small subset.
    # Actually, if k up to 1000 didn't work, maybe k is larger.
    # But given the distribution, k=1 or 2 should work.
    # If not, we can try a broader search on a subset.
    
    # Fallback: Find any pair with sigma(a)=sigma(b)
    # This is expensive.
    # Let's just return a known valid pair if we have one, or do a tiny search.
    # For safety, let's check k up to 10000 with smaller depth if 1000 fails?
    # Or just return [1, 1]? No, sigma(1)=1, sigma(1)=1. |1/1 - 1.5| = 0.5.
    # This is valid but poor.
    
    # Let's expand k search if needed, but optimize the loop.
    # If we reach here, we haven't found a good pair.
    # Let's try to find *any* pair.
    # We can reuse the logic from the previous program but only on a subset?
    # No, let's just trust the k search.
    
    # If still nothing, return [1, 1] as a last resort valid pair?
    # sigma(1)=1. a=1, b=1. |1-1.5| = 0.5.
    # But a!=b is not required, only sigma(a)=sigma(b).
    # Wait, "Minimize |a/b - 3/2|".
    # If a=1, b=1, error is 0.5.
    # If we found k=1, error is ~5e-8.
    
    return [1, 1]
