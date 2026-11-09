import time
import random

def is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    max_val = 10000
    target_size = 12
    
    # Precompute prime factors for all possible sums (2 to 20000)
    # This is used for the final evaluation
    def get_primes(n):
        if n == 1:
            return set()
        primes = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                primes.add(d)
                n //= d
            d += 1
        if n > 1:
            primes.add(n)
        return primes

    sum_primes_cache = [None] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        sum_primes_cache[s] = get_primes(s)
        
    def evaluate(set_a):
        all_primes = set()
        n = len(set_a)
        for i in range(n):
            for j in range(i + 1, n):
                s = set_a[i] + set_a[j]
                all_primes.update(sum_primes_cache[s])
        return all_primes

    def find_clique_bitset(primes, time_limit=10.0):
        """
        Finds a clique of size target_size in the graph where nodes are 1..max_val
        and edges exist if sum is smooth w.r.t 'primes'.
        Uses bitsets for fast intersection.
        """
        search_start = time.time()
        
        # 1. Identify smooth sums
        smooth_sums = set()
        for s in range(2, 2 * max_val + 1):
            if is_smooth(s, primes):
                smooth_sums.add(s)
        
        # 2. Build neighbor bitsets
        neighbors = [0] * (max_val + 1)
        
        for s in smooth_sums:
            min_i = max(1, s - max_val)
            max_i = min(max_val, s // 2)
            for i in range(min_i, max_i + 1):
                j = s - i
                if i != j:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        # 3. Initial candidates: all nodes 1..max_val
        all_nodes_mask = (1 << (max_val + 1)) - 1
        all_nodes_mask &= ~(1 << 0)
        
        # Optimization: Only consider nodes that have degree >= target_size - 1
        valid_nodes_mask = 0
        for i in range(1, max_val + 1):
            if bin(neighbors[i]).count('1') >= target_size - 1:
                valid_nodes_mask |= (1 << i)
        
        if valid_nodes_mask == 0:
            return None
            
        def backtrack(clique, candidates):
            """
            clique: list of integers
            candidates: bitmask of integers that can be added to clique
            """
            if time.time() - search_start > time_limit:
                return None
                
            if len(clique) == target_size:
                return clique[:]
                
            # If not enough candidates left
            if bin(candidates).count('1') < target_size - len(clique):
                return None
                
            # Extract a node from candidates
            c = candidates
            while c:
                if time.time() - search_start > time_limit:
                    return None
                
                # Get lowest set bit
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                # Remove v from candidates for subsequent iterations
                c ^= lsb
                
                # New candidates must be in current candidates AND neighbors of v
                new_candidates = candidates & neighbors[v]
                
                res = backtrack(clique + [v], new_candidates)
                if res:
                    return res
                    
            return None

        # Start search
        res = backtrack([], valid_nodes_mask)
        
        if res:
            # Verify distinctness and range
            if len(set(res)) == target_size and all(1 <= x <= max_val for x in res):
                return res
        return None

    best_set = None
    best_prime_count = float('inf')
    
    # Fallback: Generate a random set to ensure we always have a valid answer
    random.seed(42)
    fallback_set = random.sample(range(1, max_val + 1), target_size)
    fallback_primes = evaluate(fallback_set)
    best_set = fallback_set
    best_prime_count = len(fallback_primes)
    
    # Try prime sets in order of increasing size
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
            
        # Try to find a clique for this prime set
        clique = find_clique_bitset(primes, time_limit=10.0)
        if clique:
            # Verify the clique
            primes_found = evaluate(clique)
            score = len(primes_found)
            if score < best_prime_count:
                best_prime_count = score
                best_set = clique
            # If we found a solution with a small prime set, it's likely optimal or near-optimal
            # But we should still check smaller sets? 
            # The loop goes from small to large. So if we find one for [2,3], we break.
            # However, maybe [2,3] doesn't have a clique of size 12, but [2,3,5] does.
            # So we continue until we find one? 
            # The task is to MINIMIZE distinct primes. 
            # If we find a set with 2 primes, that's the best possible (since sums > 2).
            # So if score == 2, we can break.
            if score == 2:
                break
            # Otherwise, continue to next prime set? 
            # Actually, if we found a set with 3 primes, maybe a different set with 3 primes 
            # is better? No, the count is what matters. 
            # But maybe a set with [2,3,5] has 3 primes, and a set with [2,3,5,7] has 4 primes. 
            # We want the minimum. So we should try all and keep the best? 
            # But the search is expensive. 
            # If we find a set with 3 primes, it's likely hard to find one with 2. 
            # So we can break if we find a "good" score? 
            # Let's break if score <= 3. 
            if score <= 3:
                break

    return best_set
