import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _init_sieve(limit):
    """Initialize Smallest Prime Factor sieve up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _is_smooth(n, spf, allowed_primes_set):
    """Check if n is smooth w.r.t allowed_primes_set using spf."""
    while n > 1:
        p = spf[n]
        if p not in allowed_primes_set:
            return False
        while n % p == 0:
            n //= p
    return True

def _get_primes(n):
    """Get first n primes."""
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    # Precompute SPF for sums. Max sum is 2000 if we pick from 1..1000.
    # Let's use a candidate pool of 1..200 to keep sums small and search fast.
    # If we need larger numbers, we can expand, but small numbers are dense with small factors.
    max_candidate = 200
    max_sum = 2 * max_candidate
    spf = _init_sieve(max_sum + 10)
    
    primes_list = _get_primes(20)
    
    # Try increasing number of allowed primes
    for k in range(2, 20):
        if time.time() > deadline - 10:
            break
            
        allowed_primes = primes_list[:k]
        allowed_set = set(allowed_primes)
        
        # Candidate pool
        candidates = list(range(1, max_candidate + 1))
        
        # Filter candidates: A candidate x is "good" if it has many compatible partners
        # Let's just use all 1..200 and try to find an independent set.
        
        # Greedy Independent Set Search
        # We want to pick 12 numbers such that for all i<j, a_i+a_j is smooth.
        
        # Heuristic: Sort candidates by how many other candidates they are compatible with?
        # Or just iterate and pick.
        
        # Let's try a simple backtracking with pruning on the first few candidates.
        # To make it fast, we restrict the search to a subset of "promising" numbers.
        # Numbers that are themselves composed of small primes are often good, but sums matter.
        
        # Let's try picking from 1..100 first.
        search_space = list(range(1, 101))
        
        # Precompute compatibility matrix for search_space
        # compat[i][j] is True if search_space[i] + search_space[j] is smooth
        n_search = len(search_space)
        compat = [[False] * n_search for _ in range(n_search)]
        
        for i in range(n_search):
            for j in range(i + 1, n_search):
                s = search_space[i] + search_space[j]
                if _is_smooth(s, spf, allowed_set):
                    compat[i][j] = True
                    compat[j][i] = True
        
        # Now find independent set of size 12 in this graph.
        # Note: "Independent set" in conflict graph means "Clique" in compatibility graph.
        # We want a clique of size 12 in the compatibility graph.
        
        # Greedy clique search
        # Sort vertices by degree in compatibility graph (descending)
        degrees = [sum(1 for j in range(n_search) if compat[i][j]) for i in range(n_search)]
        sorted_indices = sorted(range(n_search), key=lambda x: degrees[x], reverse=True)
        
        best_clique = []
        
        # Simple recursive backtracking for clique
        # With n=100, this might be slow if not careful.
        # But for small k, the graph is sparse or dense?
        # If k=2 (2,3), sums like 3, 5, 7, 9, 11, 13, 15...
        # 1+2=3 (ok), 1+4=5 (no).
        # The graph is likely sparse for small k.
        
        def find_clique(current_clique, available_indices):
            if len(current_clique) == 12:
                return current_clique[:]
            
            if time.time() > deadline - 5:
                return None
                
            if not available_indices:
                return None
            
            # Try adding each available index to the clique
            for i, idx in enumerate(available_indices):
                # Check if idx is compatible with all in current_clique
                compatible = True
                for c in current_clique:
                    if not compat[c][idx]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                new_clique = current_clique + [idx]
                # Next available indices must be compatible with idx and come after idx in sorted order?
                # Actually, we can just filter available_indices to those compatible with idx.
                # To avoid duplicates, we only consider indices from the current position onwards?
                # No, we are picking a set. Order doesn't matter.
                # To prune, we can restrict to indices that are in available_indices and compatible with idx.
                
                # Optimization: Only consider indices that appear AFTER idx in the sorted list?
                # This ensures we don't revisit the same set in different orders.
                # But we sorted by degree.
                # Let's just filter the list.
                
                new_available = []
                for j in range(i + 1, len(available_indices)):
                    next_idx = available_indices[j]
                    if compat[idx][next_idx]:
                        new_available.append(next_idx)
                
                res = find_clique(new_clique, new_available)
                if res:
                    return res
                    
            return None
        
        # Initial call
        # We can start with the highest degree node
        if sorted_indices:
            first_idx = sorted_indices[0]
            initial_available = [idx for idx in sorted_indices[1:] if compat[first_idx][idx]]
            result_indices = find_clique([first_idx], initial_available)
            
            if result_indices:
                # Map back to values
                values = [search_space[i] for i in result_indices]
                # Verify distinctness and bounds
                if len(set(values)) == 12 and all(1 <= v <= 10000 for v in values):
                    return values
                    
        # If greedy failed, try a slightly different heuristic or larger k
        # We just increment k in the loop
        
    # Fallback: Return a valid set even if not optimal, to avoid exception
    # A set of powers of 2 might have many prime factors but is valid.
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Sums: 2^a + 2^b.
    # This is a valid set of 12 distinct integers <= 10000.
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# EVOLVE-BLOCK-END
