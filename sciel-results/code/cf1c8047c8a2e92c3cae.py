import time
import random
import math

def get_primes(n):
    """Return the set of prime factors of n."""
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

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes(s))
    return all_primes

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    max_val = 10000
    target_size = 12
    
    # Precompute prime factors for all possible sums (2 to 20000)
    # This is used for the final evaluation and fallback greedy
    sum_primes_cache = [None] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        sum_primes_cache[s] = get_primes(s)
        
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

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
        
        if len(smooth_sums) < target_size * (target_size - 1) / 2:
            # Not enough edges potentially, though this bound is loose
            pass

        # 2. Build neighbor bitsets
        # neighbors[i] is a bitmask where bit j is 1 if i and j are connected
        # We only need to consider j > i to avoid symmetry issues in the search, 
        # but for the bitset intersection in Bron-Kerbosch, having the full neighborhood 
        # is easier. We will enforce ordering in the search.
        
        neighbors = [0] * (max_val + 1)
        
        # Efficient construction:
        # For each s in smooth_sums:
        #   i ranges from max(1, s-max_val) to min(max_val, s//2)
        #   j = s - i
        #   Set bit j in neighbors[i] and bit i in neighbors[j]
        
        for s in smooth_sums:
            min_i = max(1, s - max_val)
            max_i = min(max_val, s // 2)
            for i in range(min_i, max_i + 1):
                j = s - i
                if i != j:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        # 3. Initial candidates: all nodes 1..max_val
        # Bitmask with bits 1 to max_val set
        all_nodes_mask = (1 << (max_val + 1)) - 1
        # Clear bit 0
        all_nodes_mask &= ~(1 << 0)
        
        # We want to find a clique of size 12.
        # To prune, we can order nodes by degree.
        # However, bitset operations are fast. Let's try a simple recursive search.
        
        # Optimization: Only consider nodes that have degree >= target_size - 1
        # Actually, for a clique of size K, every node must have at least K-1 neighbors 
        # within the final clique, so global degree must be >= K-1.
        
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
            # To improve heuristic, pick a node with high degree within candidates?
            # Simple approach: pick the lowest set bit, or random?
            # Let's pick a node that maximizes the potential. 
            # For now, just iterate through set bits.
            
            # To avoid trying all permutations, we rely on the fact that we remove 
            # nodes from candidates as we go.
            
            # Get list of nodes in candidates
            # This conversion is O(N) but N=10000 is small.
            # However, doing this at every recursion step might be slow.
            # Alternative: use a while loop on the bitmask.
            
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
                
                # Note: We should probably exclude nodes <= v to maintain order?
                # The standard Bron-Kerbosch with pivot doesn't strictly require ordering 
                # if we just remove v from the candidate pool. 
                # However, to avoid finding the same clique in different orders, 
                # we typically enforce an ordering. 
                # If we don't enforce ordering, we might find {1,2,3} and {3,2,1}.
                # Since we return the first one found, it doesn't matter for correctness, 
                # but it might waste time.
                # Let's enforce v > max(clique) if clique is not empty?
                # Actually, the bitset `candidates` already contains only nodes not yet in clique.
                # If we just pick any node, we cover all combinations.
                
                res = backtrack(clique + [v], new_candidates)
                if res:
                    return res
                    
            return None

        # Start search
        # To speed up, we can try starting from high-degree nodes.
        # But `backtrack` handles the branching.
        # Let's just call it.
        
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
    # Based on inspiration, {2,3,5,7} should work.
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
        res = find_clique_bitset(primes, time_limit=10.0)
        
        if res:
            # Verify
            if len(set(res)) == target_size and all(1 <= x <= max_val for x in res):
                primes_found = evaluate(res)
                if len(primes_found) < best_prime_count:
                    best_prime_count = len(primes_found)
                    best_set = res
                # If we found a solution with 2 primes, that's optimal
                if best_prime_count == 2:
                    return best_set
                # If we found 3 or 4, it's very good. 
                # The task is to minimize. 4 is likely the minimum for N=12.
                # Let's break if we find 4, as 2 and 3 are highly unlikely for 12 nodes.
                if best_prime_count <= 4:
                    break

    # If clique search didn't find a solution (or failed to find a small one),
    # use a local search to improve the fallback or any found set.
    
    if time.time() < deadline - 2 and best_set is not None:
        current_set = best_set[:]
        current_primes = evaluate(current_set)
        current_count = len(current_primes)
        
        # Local search: swap one element with a random other
        # We want to minimize the number of distinct primes.
        # This is a hard optimization. 
        # Simple heuristic: try to replace an element that contributes many unique primes
        # with one that doesn't.
        
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            idx = random.randint(0, target_size - 1)
            old_val = current_set[idx]
            
            # Try a few random replacements
            for _ in range(5):
                new_val = random.randint(1, max_val)
                
                if new_val == old_val:
                    continue
                if new_val in current_set and current_set.index(new_val) != idx:
                    continue
                
                new_set = current_set[:]
                new_set[idx] = new_val
                new_primes = evaluate(new_set)
                new_count = len(new_primes)
                
                if new_count < current_count:
                    current_set = new_set
                    current_primes = new_primes
                    current_count = new_count
                    
                    if current_count < best_prime_count:
                        best_prime_count = current_count
                        best_set = current_set[:]
                    break # Restart outer loop with new base
                
    return best_set
