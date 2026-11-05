import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs and Conflicts
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))

    # Precompute degrees for greedy sorting
    degrees = [len(conflicts[v]) for v in range(N + 1)]
    
    best_size = 0
    best_set = set()
    
    # Multiple restarts
    num_restarts = 20
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Initialization
        # Randomize the order slightly to explore different local optima
        # Sort by degree, but add some randomness to ties or just shuffle
        candidates = sorted(numbers, key=lambda x: (degrees[x], random.random()))
        
        S = set()
        active_pairs = set()
        # pair_count[pair] = number of x in S such that pair is in conflicts[x]
        pair_count = {}
        
        # Helper to update pair_count when adding/removing v
        def add_to_S(v):
            S.add(v)
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    active_pairs.add(p)
                    # Update pair_count for all conflicts of v? 
                    # No, pair_count tracks "how many x in S conflict with pair".
                    # When we add v, for each pair p in conflicts[v], 
                    # that pair now has one more x (v) that conflicts with it.
                    for p in conflicts[v]:
                        pair_count[p] = pair_count.get(p, 0) + 1
        
        def remove_from_S(v):
            S.remove(v)
            # Remove pairs involving v from active_pairs
            pairs_to_remove = []
            for u in S:
                p = frozenset((u, v))
                if p in active_pairs:
                    active_pairs.remove(p)
                    pairs_to_remove.append(p)
            # Note: The above loop is O(|S|). 
            # Actually, we should iterate over S before removing v? 
            # S still contains v in this function call? 
            # Let's be careful. 
            # We remove v from S first? 
            # If we remove v from S, then iterating S gives u != v.
            # But we need to remove pairs (u, v) from active_pairs.
            # So we iterate over current S (which contains v) and pick u != v.
            pass # Logic handled in caller or rewritten below
            
        # Let's rewrite add/remove to be robust
        def add_vertex(v):
            S.add(v)
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    active_pairs.add(p)
            for p in conflicts[v]:
                pair_count[p] = pair_count.get(p, 0) + 1

        def remove_vertex(v):
            # Remove pairs involving v from active_pairs
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    if p in active_pairs:
                        active_pairs.remove(p)
            S.remove(v)
            # Decrement pair_count for conflicts of v
            for p in conflicts[v]:
                if p in pair_count:
                    pair_count[p] -= 1
                    if pair_count[p] == 0:
                        del pair_count[p]

        for v in candidates:
            if time.time() > deadline:
                break
            # Check if v can be added
            # v is valid if no pair in conflicts[v] is in active_pairs
            valid = True
            for p in conflicts[v]:
                if p in active_pairs:
                    valid = False
                    break
            if valid:
                add_vertex(v)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = set(S)
            # Store a copy of state for potential reuse? 
            # Or just restart from scratch.
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Add
            remaining = set(numbers) - S
            # Sort remaining by degree (lowest first) to find easy adds
            for v in sorted(remaining, key=lambda x: degrees[x]):
                if time.time() > deadline:
                    break
                valid = True
                for p in conflicts[v]:
                    if p in active_pairs:
                        valid = False
                        break
                if valid:
                    add_vertex(v)
                    improved = True
                    break # Restart loop
            
            if improved:
                continue
                
            # Phase 2: 1-1 Swap
            found_swap = False
            # Iterate u in S, v in remaining
            # To speed up, we can just check all pairs
            for u in list(S):
                if time.time() > deadline:
                    break
                # Temporarily remove u
                remove_vertex(u)
                
                # Check all v in remaining
                for v in remaining:
                    # Check if v can be added to current S (which is S_orig \ {u})
                    valid = True
                    for p in conflicts[v]:
                        if p in active_pairs:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        add_vertex(v)
                        # u is already removed. v is added.
                        # We need to update remaining? 
                        # remaining is static in this loop iteration? 
                        # It's better to rebuild remaining or track it.
                        # Since we break on success, we can just update the global sets.
                        # But `remaining` was calculated before the loop.
                        # If we add v, v is no longer in remaining.
                        # If we remove u, u is now in remaining.
                        # We will recalculate remaining in the next iteration of the while loop.
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break
                
                # If no swap found for this u, restore u
                add_vertex(u)
            
            if improved:
                continue
                
            # Phase 3: 1-2 Swap (Remove 1, Add 2)
            # This increases size by 1
            found_12_swap = False
            for u in list(S):
                if time.time() > deadline:
                    break
                remove_vertex(u)
                
                # Find valid candidates v in remaining
                # v is valid if conflicts[v] does not intersect active_pairs
                valid_candidates = []
                for v in remaining:
                    valid = True
                    for p in conflicts[v]:
                        if p in active_pairs:
                            valid = False
                            break
                    if valid:
                        valid_candidates.append(v)
                        
                # Find pair v1, v2 in valid_candidates such that {v1, v2} has pair_count 0
                # pair_count is currently updated for S \ {u}
                for i in range(len(valid_candidates)):
                    for j in range(i + 1, len(valid_candidates)):
                        v1 = valid_candidates[i]
                        v2 = valid_candidates[j]
                        p12 = frozenset((v1, v2))
                        # If pair_count[p12] is 0, it means no x in S\{u} conflicts with {v1, v2}
                        # Note: pair_count only contains keys with count > 0
                        if p12 not in pair_count:
                            # Valid 1-2 swap
                            add_vertex(v1)
                            add_vertex(v2)
                            found_12_swap = True
                            improved = True
                            break
                    if found_12_swap:
                        break
                
                if found_12_swap:
                    break
                
                # If no swap found, restore u
                add_vertex(u)
            
            if improved:
                continue
                
        # End Local Search
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    return list(best_set)

import random
