import time
import random
from itertools import combinations

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _pf_set(n, spf):
    s = set()
    while n > 1:
        p = spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

def _score_full(arr, spf):
    ps = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            if s > 1:
                ps |= _pf_set(s, spf)
    return len(ps)

def _generate_smooth_numbers(primes, limit):
    """Generate all numbers <= limit whose prime factors are in primes."""
    smooth = set([1])
    # Use a queue or iterative expansion
    # Since limit is 20000, we can just do a BFS/DFS on the multiplicative structure
    # Or simply iterate through all numbers and check? 20000 is small.
    # But checking each number is O(log n). 20000 * log(20000) is fine.
    # However, generating directly is faster.
    
    # Direct generation
    smooths = set([1])
    changed = True
    while changed:
        changed = False
        for s in list(smooths):
            for p in primes:
                new_s = s * p
                if new_s <= limit:
                    if new_s not in smooths:
                        smooths.add(new_s)
                        changed = True
    return smooths

def _find_clique_size_12(compat, limit_val, deadline):
    """
    Find a clique of size 12 in the graph defined by compat.
    compat[u] is a set of v such that u+v is smooth.
    """
    # We want 12 distinct integers from 1..limit_val
    # such that all pairs are in compat.
    
    # Sort candidates by degree (size of compat set) to prune?
    # Actually, we can just pick nodes.
    
    # Heuristic: Start with a node with high degree?
    # Or just iterate?
    # Given the structure, let's try a backtracking search.
    
    # To make it faster, we can restrict the search to a subset of candidates
    # that have high "potential".
    
    # Let's try a simple backtracking.
    # Order candidates 1..limit_val.
    # We can prune if the remaining candidates cannot form a clique of required size.
    
    # For efficiency, we'll use a recursive function.
    # To avoid deep recursion and overhead, we'll use a stack or limit depth.
    # Depth is 12.
    
    # Pre-compute degrees
    degrees = [len(compat[i]) for i in range(limit_val + 1)]
    
    # Sort candidates by degree descending?
    # If we pick a node with low degree, we might fail sooner.
    # So pick high degree first?
    # Let's try picking nodes in order of decreasing degree.
    
    candidates = list(range(1, limit_val + 1))
    candidates.sort(key=lambda x: degrees[x], reverse=True)
    
    # We need to ensure we don't use the same node twice.
    
    def backtrack(current_set, available_candidates):
        if len(current_set) == 12:
            return current_set[:]
        
        if not available_candidates:
            return None
            
        # Pruning: if not enough candidates left
        if len(available_candidates) < 12 - len(current_set):
            return None
            
        # Try each candidate
        # To speed up, we can limit the number of tries per level
        # or use a heuristic.
        
        for i, cand in enumerate(available_candidates):
            if time.time() > deadline:
                return None
                
            # Check if cand is compatible with all in current_set
            # Since available_candidates is already filtered to be compatible
            # with all nodes in current_set, we don't need to re-check.
            
            # Add cand
            new_set = current_set + [cand]
            
            # Filter available_candidates:
            # Keep only those v in available_candidates[i+1:] (to avoid duplicates and order)
            # such that v is compatible with cand.
            
            # Note: available_candidates is a list. We should use the remaining ones.
            # To ensure distinctness and avoid permutations, we only pick from remaining.
            
            next_candidates = []
            for v in available_candidates[i+1:]:
                if v in compat[cand]:
                    next_candidates.append(v)
            
            # Heuristic: if next_candidates is too small, skip?
            # Required size: 12 - len(new_set)
            if len(next_candidates) < 12 - len(new_set):
                continue
                
            res = backtrack(new_set, next_candidates)
            if res:
                return res
                
        return None

    # Initial available candidates: all nodes
    # But we should filter out nodes with degree < 11?
    # A node in a clique of size 12 must be connected to 11 others.
    # So degree must be at least 11.
    
    initial_candidates = [c for c in candidates if degrees[c] >= 11]
    
    if not initial_candidates:
        return None
        
    # Shuffle to break symmetry? Or just use sorted?
    # Sorted by degree is good.
    
    return backtrack([], initial_candidates)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    MAX_VAL = 10000
    MAX_SUM = 20000
    
    # Precompute SPF for scoring fallback
    spf = _spf_sieve(MAX_SUM)
    
    # Fallback set
    best_set = list(range(1, 13))
    best_score = _score_full(best_set, spf)
    
    # Try prime sets
    prime_sets = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    for primes in prime_sets:
        if time.time() > deadline - 10:
            break
            
        # Generate smooth numbers
        smooths = _generate_smooth_numbers(primes, MAX_SUM)
        smooth_set = set(smooths)
        
        # Build compatibility graph
        # compat[u] = set of v such that u+v in smooth_set
        # u, v in 1..10000
        
        compat = [set() for _ in range(MAX_VAL + 1)]
        
        for u in range(1, MAX_VAL + 1):
            for v in range(u + 1, MAX_VAL + 1): # v > u to avoid duplicates in set logic? No, we need all v.
                # Actually, we can just iterate v in 1..MAX_VAL
                pass
        
        # More efficient way to build compat:
        # For each smooth number s, for each u in 1..MAX_VAL:
        # v = s - u
        # if 1 <= v <= MAX_VAL and v != u:
        #    compat[u].add(v)
        #    compat[v].add(u)
        
        for s in smooths:
            if s < 2: continue
            for u in range(1, MAX_VAL + 1):
                v = s - u
                if 1 <= v <= MAX_VAL and v != u:
                    compat[u].add(v)
                    # compat[v].add(u) is implied if we iterate all u, but adding both is safer for logic
                    # Actually, if we iterate u from 1 to MAX_VAL, we cover all pairs.
                    # But we can optimize: u < v implies u < s/2.
        
        # To speed up building:
        # Only iterate u up to s/2?
        # If u > s/2, then v < s/2 < u. The pair (v,u) will be added when u' = v is processed?
        # No, the loop is `for u in range(1, MAX_VAL+1)`.
        # So for a fixed s, we add (u, s-u).
        # This covers all pairs {u, v} with u+v=s.
        
        # Re-doing the loop efficiently:
        compat = [set() for _ in range(MAX_VAL + 1)]
        for s in smooths:
            if s < 2:
                continue
            # u ranges from max(1, s-MAX_VAL) to min(MAX_VAL, s-1)
            start_u = max(1, s - MAX_VAL)
            end_u = min(MAX_VAL, s - 1)
            for u in range(start_u, end_u + 1):
                v = s - u
                if u != v:
                    compat[u].add(v)
                    # We don't strictly need to add to compat[v] if we treat graph as undirected
                    # and our search logic handles it, but let's be symmetric.
                    # Actually, in the backtrack, we check `if v in compat[cand]`.
                    # If we only added u->v, then v->u is missing.
                    # So we MUST add both or ensure the check is symmetric.
                    # Adding both is safer.
                    compat[v].add(u)
        
        # Find clique
        result = _find_clique_size_12(compat, MAX_VAL, deadline)
        
        if result:
            # Verify
            if len(set(result)) == 12 and max(result) <= MAX_VAL:
                score = _score_full(result, spf)
                if score < best_score:
                    best_score = score
                    best_set = result
                # If we found a clique with these primes, the score should be <= len(primes)
                # Actually, it might be less if some sums are 1 (impossible) or if primes are shared.
                # But it's guaranteed to be <= len(primes) if the sums are P-smooth.
                # Wait, if P={2,3,5}, sums are 2^a 3^b 5^c. Primes are subset of {2,3,5}.
                # So score <= 3.
                # This is a huge improvement over 9.
                return best_set
    
    # If no clique found, fallback to local search on a good heuristic
    # Try multiples of a smooth number
    # e.g. multiples of 2, 3, 6, etc.
    
    # Simple local search
    if time.time() < deadline - 5:
        random.seed(42)
        # Start with 1..12
        current_set = list(range(1, 13))
        current_score = _score_full(current_set, spf)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                old = current_set[i]
                # Try random replacements
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    new_val = random.randint(1, 10000)
                    if new_val == old or new_val in current_set:
                        continue
                    test = current_set[:]
                    test[i] = new_val
                    ns = _score_full(test, spf)
                    if ns < current_score:
                        current_score = ns
                        current_set = test
                        improved = True
                        break
                    if ns < best_score:
                        best_score = ns
                        best_set = test[:]
        
        if current_score < best_score:
            best_set = current_set
            best_score = current_score
            
    return best_set
