"""Program using modular arithmetic sieve + greedy construction with fast mean-check."""

def construct():
    n = 512
    max_size = 48
    
    import numpy as np
    
    # Step 1: Modular arithmetic sieve to score candidates
    # We want to prioritize x such that it is "harder" for x to be the mean of other elements.
    # The most common violation is a 3-term arithmetic progression: x = (y+z)/2 => 2x = y+z.
    # We score each candidate x by counting how many pairs (y, z) in [1, 512] satisfy
    # y + z = 2x (exact) or y + z ≡ 2x (mod p) for small primes p.
    # Candidates with fewer such pairs are less likely to be part of a mean-violating subset.
    
    primes = [3, 5, 7, 11, 13, 17]
    candidates = list(range(1, n + 1))
    
    # Precompute scores
    # score[x] = sum over p in primes of (count of pairs y,z != x such that y+z ≡ 2x mod p)
    # To be efficient, we can just check exact sums first, then modulo.
    # Actually, the exact sum check is O(N^2) per x, which is 512^3 ~ 134M, too slow for pure Python if done naively in a loop.
    # But we can do it smarter or just use the modulo heuristic which is O(N * num_primes * N) ~ 512 * 6 * 512 ~ 1.5M, which is fast.
    
    scores = {x: 0 for x in candidates}
    
    # For each prime p, for each x, count pairs y,z in [1, N] such that y+z ≡ 2x (mod p) and y != z, y != x, z != x.
    # We can iterate x, and for each x, iterate y, and compute z = (2x - y) % p.
    # But we need to know if there exists an actual integer z in [1, N] with that residue.
    # Since N=512 is large compared to p, every residue class 0..p-1 has many representatives.
    # So the count is approximately the number of y such that the residue of z is valid and z != y, z != x.
    # This is roughly (p-1) choices for y for each x? No, y can be anything.
    # Let's simplify: The number of pairs (y,z) mod p is fixed for a given x mod p.
    # If we just want a heuristic to *order* the candidates, we can use the number of exact 3-APs involving x in [1, 512].
    # Count of y,z in [1, 512], y!=z, y!=x, z!=x, y+z=2x.
    
    # Let's compute exact AP counts efficiently.
    # For a fixed x, y must be in [1, 512] and z = 2x - y must be in [1, 512] and z != y, z != x.
    # y ranges from max(1, 2x - 512) to min(512, 2x - 1).
    
    for x in candidates:
        count = 0
        low = max(1, 2*x - n)
        high = min(n, 2*x - 1)
        for y in range(low, high + 1):
            z = 2*x - y
            if z != y and z != x and 1 <= z <= n:
                count += 1
        scores[x] = count
        
    # Sort candidates by score (ascending). Low score = fewer exact 3-APs = better candidate.
    candidates.sort(key=lambda x: scores[x])
    
    # Step 2: Greedy construction with bitset DP validation
    
    def is_valid_mean_check(selected, candidate):
        """Check if adding candidate to selected keeps the mean-free property."""
        new_set = selected + [candidate]
        n_elems = len(new_set)
        
        # For each element a in new_set, check if a is the mean of any subset
        # of at least 2 OTHER elements.
        for i, a in enumerate(new_set):
            others = [x for j, x in enumerate(new_set) if j != i]
            m = len(others)
            
            # We need to check: does there exist a subset T of others with |T| >= 2
            # such that sum(T) = |T| * a?
            
            others_set = set(others)
            found = False
            
            # Check k=2
            for b in others:
                if b != a and (2*a - b) in others_set and (2*a - b) != b:
                    found = True
                    break
            if found:
                return False
            
            # For k >= 3: use DP
            # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
            # Initialize: dp[0] has bit 0 set
            dp = [0] * (m + 1)
            dp[0] = 1  # bit 0 set
            
            for x in others:
                # Process in reverse to avoid using x multiple times
                for k in range(m, 0, -1):
                    # dp[k] |= dp[k-1] << x
                    dp[k] |= (dp[k-1] << x)
            
            # Now check for each k >= 3 if bit (k*a) is set in dp[k]
            for k in range(3, m + 1):
                target_sum = k * a
                if target_sum <= sum(others) and (dp[k] >> target_sum) & 1:
                    return False
        
        return True
    
    # Greedy: try candidates in the sorted order
    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_mean_check(selected, x):
            selected.append(x)
    
    return selected
