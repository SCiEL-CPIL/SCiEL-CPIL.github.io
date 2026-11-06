"""Program using modular arithmetic sieve + greedy construction with fast mean-check."""

def construct():
    n = 512
    max_size = 48
    
    import numpy as np
    
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    candidates = list(range(1, n + 1))
    
    # Precompute residue counts for the full range [1, n] for each prime
    full_counts = {}
    for p in primes:
        counts = [0] * p
        for i in range(1, n + 1):
            counts[i % p] += 1
        full_counts[p] = counts
        
    scores = {x: 0 for x in candidates}
    
    # Calculate modular conflict scores
    for x in candidates:
        score = 0
        for p in primes:
            counts = full_counts[p][:]
            counts[x % p] -= 1
            
            # Check k=2: y + z == 2x (mod p)
            target_2 = (2 * x) % p
            pair_count = 0
            for y in range(p):
                z = (target_2 - y) % p
                if y < z:
                    pair_count += counts[y] * counts[z]
                elif y == z:
                    if counts[y] >= 2:
                        pair_count += counts[y] * (counts[y] - 1) // 2
            score += pair_count
            
            # Check k=3: y + z + w == 3x (mod p)
            # This is more expensive, O(p^2).
            # Given p is small (max 31), p^2 is ~1000.
            # 512 * 10 * 1000 = 5M ops. Acceptable.
            target_3 = (3 * x) % p
            triple_count = 0
            for y in range(p):
                if counts[y] == 0: continue
                for z in range(p):
                    if counts[z] == 0: continue
                    w = (target_3 - y - z) % p
                    if counts[w] == 0: continue
                    # Count combinations of y, z, w
                    # We need to count distinct triples from the multiset defined by counts.
                    # This is complex to do exactly with just counts.
                    # Approximation: Just use the product counts[y]*counts[z]*counts[w] and divide by permutations?
                    # Or skip k=3 modular check as k=2 is dominant.
                    pass
            # Let's skip k=3 modular check for speed and simplicity, k=2 is the primary driver.
            
        scores[x] = score
        
    # Sort candidates by score (ascending)
    candidates.sort(key=lambda x: scores[x])
    
    def is_valid_mean_check(selected, candidate):
        """Check if adding candidate to selected keeps the mean-free property."""
        new_set = selected + [candidate]
        n_elems = len(new_set)
        
        for i, a in enumerate(new_set):
            others = [x for j, x in enumerate(new_set) if j != i]
            m = len(others)
            
            if m < 2:
                continue
                
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
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            sum_others = sum(others)
            for k in range(3, m + 1):
                target_sum = k * a
                if target_sum <= sum_others and (dp[k] >> target_sum) & 1:
                    return False
        
        return True
    
    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_mean_check(selected, x):
            selected.append(x)
    
    return selected
