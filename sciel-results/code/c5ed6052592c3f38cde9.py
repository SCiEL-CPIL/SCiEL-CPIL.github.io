"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCMs and Conflict Map
    # conflict[(a, b)] = set of c such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Note: We can store as dict of sets. Key is tuple (min(a,b), max(a,b))
    conflict = {}
    lcm_cache = {}
    
    def get_lcm(a, b):
        if (a, b) in lcm_cache:
            return lcm_cache[(a, b)]
        if (b, a) in lcm_cache:
            return lcm_cache[(b, a)]
        val = a * b // math.gcd(a, b)
        lcm_cache[(a, b)] = val
        lcm_cache[(b, a)] = val
        return val

    # Build conflict map
    # Iterate all triples? 128^3 is ~2M. Fast enough.
    # Or iterate pairs and check c?
    # For each pair (a, b), L = lcm(a, b).
    # c must be such that lcm(a, c) = L and lcm(b, c) = L.
    # This implies c divides L, and lcm(a,c)=L, lcm(b,c)=L.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L = get_lcm(a, b)
            # c must be a divisor of L
            # and lcm(a, c) == L and lcm(b, c) == L
            # Iterate divisors of L? Or just iterate c from 1 to N?
            # Since N=128, iterating c is fine.
            key = (a, b)
            if key not in conflict:
                conflict[key] = set()
            
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                # Check if lcm(a,c) == L and lcm(b,c) == L
                # Optimization: c must divide L
                if L % c != 0:
                    continue
                if get_lcm(a, c) == L and get_lcm(b, c) == L:
                    conflict[key].add(c)
                    # Also add to reverse keys? 
                    # We will always query with sorted pair (min, max)
                    # So we need conflict[(min(x,y), max(x,y))]
                    # Let's normalize keys in conflict dict
                    pass

    # Re-normalize conflict dict to use sorted keys
    normalized_conflict = {}
    for (a, b), cs in conflict.items():
        key = (min(a, b), max(a, b))
        if key not in normalized_conflict:
            normalized_conflict[key] = set()
        normalized_conflict[key].update(cs)
    
    # Also, for a given pair (x, y), we need to know if z is in conflict.
    # So normalized_conflict[(min(x,y), max(x,y))] is the set of z.
    
    def is_bad_triple(a, b, c):
        # Assumes a, b, c distinct
        key1 = (min(a, b), max(a, b))
        if c in normalized_conflict.get(key1, set()):
            return True
        key2 = (min(a, c), max(a, c))
        if b in normalized_conflict.get(key2, set()):
            return True
        key3 = (min(b, c), max(b, c))
        if a in normalized_conflict.get(key3, set()):
            return True
        return False

    # Helper to check if adding v to S is valid
    def can_add(v, S):
        # Check all pairs in S
        # If v is in conflict of any pair (a,b) in S, it's invalid
        for a, b in combinations(S, 2):
            key = (min(a, b), max(a, b))
            if v in normalized_conflict.get(key, set()):
                return False
        return True

    # Helper to check if swap u -> {v1, v2} is valid
    def can_swap(u, v1, v2, S):
        # S is the current set containing u
        # New set S' = (S \ {u}) \cup {v1, v2}
        # Check triples involving v1 or v2
        S_minus_u = S - {u}
        
        # 1. Triples {v1, v2, x} for x in S_minus_u
        key_v1v2 = (min(v1, v2), max(v1, v2))
        conflict_v1v2 = normalized_conflict.get(key_v1v2, set())
        for x in S_minus_u:
            if x in conflict_v1v2:
                return False
        
        # 2. Triples {v1, x, y} for x, y in S_minus_u
        # Check if v1 is in conflict of any pair (x, y) in S_minus_u
        for x, y in combinations(S_minus_u, 2):
            key_xy = (min(x, y), max(x, y))
            if v1 in normalized_conflict.get(key_xy, set()):
                return False
        
        # 3. Triples {v2, x, y} for x, y in S_minus_u
        for x, y in combinations(S_minus_u, 2):
            key_xy = (min(x, y), max(x, y))
            if v2 in normalized_conflict.get(key_xy, set()):
                return False
                
        return True

    best_set = set()
    best_size = 0
    
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Greedy Add with random order
        S = set()
        random.shuffle(elements)
        for x in elements:
            if can_add(x, S):
                S.add(x)
        
        # Phase 1: Try to add more elements
        # We can do this repeatedly until no more can be added
        improved = True
        while improved and (time.time() - start_time < deadline):
            improved = False
            # Try adding any element not in S
            # Sort candidates by some heuristic? Or just iterate
            for v in elements:
                if v not in S:
                    if can_add(v, S):
                        S.add(v)
                        improved = True
                        break # Restart add loop to re-evaluate? Or continue?
                        # Continuing might be better if multiple can be added
                        # But adding one might block another. 
                        # Let's just break and restart the while loop to be safe and simple.
        
        # Phase 2: 1-for-2 Swaps
        # Try to replace one element with two
        improved = True
        while improved and (time.time() - start_time < deadline):
            improved = False
            # Try all u in S
            # Shuffle S to randomize
            S_list = list(S)
            random.shuffle(S_list)
            for u in S_list:
                if time.time() - start_time > deadline:
                    break
                # Try all pairs v1, v2 not in S
                # This is O(|S| * |S^c|^2)
                # To speed up, we can just iterate
                not_in_S = [x for x in elements if x not in S]
                random.shuffle(not_in_S)
                for i in range(len(not_in_S)):
                    v1 = not_in_S[i]
                    for j in range(i + 1, len(not_in_S)):
                        v2 = not_in_S[j]
                        if can_swap(u, v1, v2, S):
                            S.remove(u)
                            S.add(v1)
                            S.add(v2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return list(best_set)
# EVOLVE-BLOCK-END
