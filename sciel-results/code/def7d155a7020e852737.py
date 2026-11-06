import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s buffer for safety
    
    # Precompute primes up to 20000
    limit = 20000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Fast factorization using sieve
    # We just need the set of prime factors
    # Precompute smallest prime factor (SPF) for speed
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def evaluate(a):
        primes_union = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                s = a[i] + a[j]
                primes_union.update(get_prime_factors(s))
        return len(primes_union)

    # Strategy: Use modulus M to constrain sums
    # Try M = 210 (2*3*5*7)
    # a_i = 105 + 210 * k_i
    # Max k: (10000 - 105) // 210 = 47
    # We search for k_i in {0..47}
    
    best_score = float('inf')
    best_a = []
    
    # Initial candidate
    k_vals = list(range(12))
    a_curr = [105 + 210 * k for k in k_vals]
    curr_score = evaluate(a_curr)
    best_score = curr_score
    best_a = a_curr[:]
    
    # Local Search
    # We represent the state as a set of k's
    # k_set = set(k_vals)
    
    def get_a_from_k(k_set):
        return [105 + 210 * k for k in sorted(k_set)]
    
    k_set = set(k_vals)
    a_curr = get_a_from_k(k_set)
    curr_score = evaluate(a_curr)
    
    # Hill Climbing
    # We can try to swap one k with another
    # Or just iterate through all possible k replacements
    
    # To speed up, we can cache scores of k sets? No, too many.
    # Just evaluate on the fly.
    
    # We will try to improve by replacing one k_i with a new k
    # There are 48 possible k values.
    
    # We can also try random restarts with different initial k sets
    # But the structure M=210 is strong.
    
    # Let's run a simple hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        k_list = sorted(k_set)
        for idx, k_old in enumerate(k_list):
            # Try replacing k_old with k_new
            for k_new in range(48):
                if k_new == k_old or k_new in k_set:
                    continue
                
                # Swap
                k_set.remove(k_old)
                k_set.add(k_new)
                
                a_temp = get_a_from_k(k_set)
                score = evaluate(a_temp)
                
                if score < curr_score:
                    curr_score = score
                    if score < best_score:
                        best_score = score
                        best_a = a_temp[:]
                    improved = True
                else:
                    # Revert
                    k_set.remove(k_new)
                    k_set.add(k_old)
                    
                if time.time() > deadline - 5:
                    # Restore state before exiting
                    if k_new in k_set and k_old not in k_set:
                        k_set.remove(k_new)
                        k_set.add(k_old)
                    return best_a
    
    # If we have time, try M=30
    # a_i = 15 + 30 * k_i
    # Max k: (10000 - 15) // 30 = 332
    # This has more search space, but base primes are only 3.
    # If we can get co-factors to add 0 or 1 primes, total 3 or 4.
    # Let's try a quick search here too.
    
    if time.time() < deadline - 10:
        # Initialize with small k
        k_vals_30 = list(range(12))
        k_set_30 = set(k_vals_30)
        a_curr_30 = [15 + 30 * k for k in sorted(k_set_30)]
        curr_score_30 = evaluate(a_curr_30)
        
        if curr_score_30 < best_score:
            best_score = curr_score_30
            best_a = a_curr_30[:]
            
        improved_30 = True
        while improved_30 and time.time() < deadline - 5:
            improved_30 = False
            k_list_30 = sorted(k_set_30)
            for idx, k_old in enumerate(k_list_30):
                # Try a few random new k's to avoid full 332 loop if slow
                # But 332 is small. Let's try all.
                for k_new in range(333):
                    if k_new == k_old or k_new in k_set_30:
                        continue
                    
                    k_set_30.remove(k_old)
                    k_set_30.add(k_new)
                    
                    a_temp = [15 + 30 * k for k in sorted(k_set_30)]
                    score = evaluate(a_temp)
                    
                    if score < curr_score_30:
                        curr_score_30 = score
                        if score < best_score:
                            best_score = score
                            best_a = a_temp[:]
                        improved_30 = True
                    else:
                        k_set_30.remove(k_new)
                        k_set_30.add(k_old)
                    
                    if time.time() > deadline - 5:
                        if k_new in k_set_30 and k_old not in k_set_30:
                            k_set_30.remove(k_new)
                            k_set_30.add(k_old)
                        return best_a

    return best_a

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    start_time = time.time()
    deadline = start_time + 80.0
    
    limit = 20000
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute prime factor bitmasks for all sums up to 20000
    # First, collect all primes up to 20000
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # For bitmask, we need up to num_primes bits. Use Python big ints.
    # Precompute factor_mask for each n up to limit
    factor_mask = [0] * (limit + 1)
    for n in range(2, limit + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            mask |= (1 << prime_to_idx[p])
            while x % p == 0:
                x //= p
        factor_mask[n] = mask
    
    def evaluate_mask(a):
        """Evaluate using bitmask union, return (mask, count)"""
        union_mask = 0
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                s = ai + a[j]
                if s <= limit:
                    union_mask |= factor_mask[s]
        return union_mask
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def is_smooth(n, allowed_primes_set):
        """Check if n has no prime factors outside allowed_primes_set"""
        if n == 1: return True
        x = n
        while x > 1:
            p = spf[x]
            if p not in allowed_primes_set:
                return False
            while x % p == 0:
                x //= p
        return True

    def find_clique_smooth(m, max_k, allowed_primes, size):
        """
        Find 'size' distinct k's in [1, max_k] such that
        m*k_i + m*k_j has no prime factors outside allowed_primes.
        Note: m*k_i + m*k_j = m*(k_i+k_j).
        So we need prime factors of m to be in allowed_primes,
        AND prime factors of (k_i+k_j) to be in allowed_primes.
        """
        # Check if m is compatible
        x = m
        while x > 1:
            p = spf[x]
            if p not in allowed_primes:
                return None
            while x % p == 0:
                x //= p
        
        # Build graph of valid k's
        # k must be in [1, max_k]
        # Edge (u,v) if is_smooth(u+v, allowed_primes)
        # We want a clique of size 'size'
        
        # Optimization: Only consider k's that are "likely" to work?
        # For now, just build the graph for k in 1..max_k
        # If max_k is large, this might be slow. 
        # But for m=210, max_k=47. 47 nodes is small.
        # For m=30, max_k=332. 332 nodes, 332^2/2 ~ 55k edges. Fast.
        
        n_nodes = max_k
        # adjacency list
        adj = [[] for _ in range(n_nodes)]
        for u in range(1, n_nodes + 1):
            for v in range(u + 1, n_nodes + 1):
                if is_smooth(u + v, allowed_primes):
                    adj[u-1].append(v-1)
                    adj[v-1].append(u-1)
        
        # Greedy + Backtracking for Max Clique
        # Simple recursive search
        best_clique = []
        
        def backtrack(candidates, current_clique):
            nonlocal best_clique
            if len(current_clique) > len(best_clique):
                best_clique = current_clique[:]
                if len(best_clique) == size:
                    return True # Found target size
            if not candidates:
                return False
            
            # Pruning: if len(candidates) + len(current_clique) <= len(best_clique), return
            if len(candidates) + len(current_clique) <= len(best_clique):
                return False
            
            # Order candidates by degree (descending) for better pruning
            # But calculating degree dynamically is expensive. 
            # Just pick the first one or a random one.
            # Let's try picking the vertex with highest degree in the induced subgraph?
            # For small graphs, simple backtracking is fine.
            
            # Pick a vertex
            # To improve, we can sort candidates by degree in the original graph
            # But we don't have that precomputed easily here.
            # Let's just iterate.
            
            for i in range(len(candidates)):
                v = candidates[i]
                new_clique = current_clique + [v]
                # New candidates are neighbors of v that are in candidates and > i (to avoid duplicates)
                new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
                
                if backtrack(new_candidates, new_clique):
                    return True
            return False
        
        # Start with all nodes
        all_nodes = list(range(n_nodes))
        # Sort by degree descending to help pruning
        degrees = [len(adj[i]) for i in range(n_nodes)]
        all_nodes.sort(key=lambda x: degrees[x], reverse=True)
        
        backtrack(all_nodes, [])
        
        if len(best_clique) >= size:
            # best_clique contains indices 0..max_k-1
            # k values are index + 1
            k_vals = [idx + 1 for idx in best_clique[:size]]
            a_vals = [m * k for k in k_vals]
            # Verify
            # All a_vals <= 10000?
            if all(a <= 10000 for a in a_vals):
                return a_vals
        return None

    best_score = float('inf')
    best_a = []
    
    # Try specific configurations that might yield low prime counts
    # Config 1: m=210, S={2,3,5,7} -> Value 4
    # Config 2: m=210, S={2,3,5} -> Value 3 (if possible)
    # Config 3: m=30, S={2,3,5} -> Value 3
    # Config 4: m=6, S={2,3} -> Value 2 (unlikely)
    
    configs = [
        (210, 47, {2, 3, 5, 7}),
        (210, 47, {2, 3, 5}),
        (30, 332, {2, 3, 5}),
        (6, 1666, {2, 3}),
        (10, 999, {2, 5}),
        (15, 666, {3, 5}),
        (14, 713, {2, 7}),
        (21, 475, {3, 7}),
    ]
    
    for m, max_k, S in configs:
        if time.time() > deadline - 5:
            break
        res = find_clique_smooth(m, max_k, S, 12)
        if res:
            # Evaluate to be sure
            score = evaluate_mask(res)
            score = count_bits(score)
            if score < best_score:
                best_score = score
                best_a = res[:]
            # If we found a very good score (e.g. 4), we might stop
            if best_score <= 4:
                break

    # Fallback to local search if no good clique found or if we want to improve further
    if time.time() < deadline - 10:
        # Use the previous local search logic
        def try_search(k_set, base, mod, max_k):
            nonlocal best_score, best_a
            improved = True
            # Initial mask
            a_curr = [base + mod * k for k in sorted(k_set)]
            curr_mask = evaluate_mask(a_curr)
            curr_count = count_bits(curr_mask)
            
            if curr_count < best_score:
                best_score = curr_count
                best_a = a_curr[:]
                
            while improved and time.time() < deadline - 5:
                improved = False
                k_list = sorted(k_set)
                for idx, k_old in enumerate(k_list):
                    for k_new in range(max_k + 1):
                        if k_new == k_old or k_new in k_set:
                            continue
                        
                        k_set.remove(k_old)
                        k_set.add(k_new)
                        
                        a_temp = [base + mod * k for k in sorted(k_set)]
                        new_mask = evaluate_mask(a_temp)
                        new_count = count_bits(new_mask)
                        
                        if new_count < curr_count:
                            curr_mask = new_mask
                            curr_count = new_count
                            if new_count < best_score:
                                best_score = new_count
                                best_a = a_temp[:]
                            improved = True
                        else:
                            k_set.remove(k_new)
                            k_set.add(k_old)
                            
                        if time.time() > deadline - 5:
                            if k_new in k_set and k_old not in k_set:
                                k_set.remove(k_new)
                                k_set.add(k_old)
                            return curr_count
                    
                    if time.time() > deadline - 5:
                        return curr_count
            
            return curr_count

        # Try a few common moduli
        mods_to_try = [210, 30, 6, 10, 15, 14, 21]
        for m in mods_to_try:
            if time.time() > deadline - 5:
                break
            # Determine max_k
            # a_i = b + m*k <= 10000
            # Let b = m//2 or similar to center?
            # Or just b=0? No, distinct positive.
            # Let's use b = m/2 if m is even?
            # Simpler: b = 1?
            # The previous code used specific bases.
            # Let's use base = 1 for simplicity in fallback, or m//2.
            base = 1
            max_k = (10000 - base) // m
            if max_k < 12: continue
            
            k_set = set(range(12))
            try_search(k_set, base, m, max_k)

    return best_a
# EVOLVE-BLOCK-END
