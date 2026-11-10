import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Candidate prime sets to try in order of increasing size
    # We start with the smallest possible sets
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    
    max_val = 10000
    max_sum = 20000
    target_size = 12

    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        # 1. Build compatibility graph
        # is_smooth[n] is True if n is composed only of primes in 'primes'
        is_smooth = [False] * (max_sum + 1)
        for n in range(1, max_sum + 1):
            temp = n
            for p in primes:
                while temp % p == 0:
                    temp //= p
            if temp == 1:
                is_smooth[n] = True
        
        # Build adjacency sets
        # adj[i] contains j if i+j is smooth
        adj = [set() for _ in range(max_val + 1)]
        
        for s in range(2, max_sum + 1):
            if not is_smooth[s]:
                continue
            min_i = max(1, s - max_val)
            max_i = min(max_val, (s - 1) // 2)
            for i in range(min_i, max_i + 1):
                j = s - i
                if j > i:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # 2. Find clique of size target_size
        # Use backtracking with sets
        initial_candidates = set(range(1, max_val + 1))
        
        # Sort initial candidates by degree to help heuristic?
        # We can't easily sort a set and keep it a set for fast intersection.
        # We'll just iterate.
        
        def backtrack(chosen, candidates):
            if time.time() > deadline:
                return None
                
            if len(chosen) == target_size:
                return list(chosen)
                
            # Pruning
            if len(candidates) < target_size - len(chosen):
                return None
                
            # Heuristic: pick nodes with higher degree in the current candidate set?
            # For now, just iterate.
            # To speed up, we can try nodes with higher global degree first.
            # But sorting candidates every call is O(N log N).
            # We can pre-sort the global nodes and filter?
            # Let's stick to simple iteration first.
            
            for v in list(candidates):
                new_candidates = candidates & adj[v]
                new_candidates.discard(v)
                
                res = backtrack(chosen | {v}, new_candidates)
                if res:
                    return res
                    
            return None

        clique = backtrack(set(), initial_candidates)
        
        if clique:
            return clique
            
    # Fallback
    return list(range(1, 13))

