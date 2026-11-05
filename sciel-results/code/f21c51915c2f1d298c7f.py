import time
from math import gcd
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # 2. Identify Bad Triples and Conflicts
    # conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            # It is a bad triple
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))

    # 3. Greedy Construction and Local Search wrapper
    def solve_once(seed):
        rng = random.Random(seed)
        
        # Calculate degrees
        degrees = [len(conflicts[v]) for v in range(N + 1)]
        
        # Create candidates list
        # Sort by degree, but randomize ties to allow different paths
        # We can group by degree and shuffle within groups
        candidates = list(numbers)
        # Sort by degree
        candidates.sort(key=lambda x: degrees[x])
        
        # Shuffle within same degree groups?
        # Simpler: Just shuffle the whole list if we want high variance, 
        # or sort by (degree, random_key)
        
        # Let's sort by degree, and for equal degrees, use random order
        # This preserves the "low degree first" heuristic but adds diversity
        # We need to be careful with stability.
        
        # Alternative: Just shuffle the list of candidates that have the minimum degree available?
        # No, let's stick to a simple randomized greedy:
        # Pick from candidates with lowest degree, break ties randomly.
        
        # To implement this efficiently:
        # 1. Sort indices by degree.
        # 2. Iterate through sorted list.
        # But if we pick one, the degrees of others don't change in this heuristic (static degree).
        # So we just need a list sorted by degree, with random order for ties.
        
        # Create list of (degree, random_val, vertex)
        # Sort by degree, then random_val
        keyed_candidates = [(degrees[v], rng.random(), v) for v in numbers]
        keyed_candidates.sort(key=lambda x: (x[0], x[1]))
        candidates = [v for _, _, v in keyed_candidates]

        S = set()
        active_pairs = set()
        
        def can_add(v, active_pairs_set):
            if not conflicts[v]:
                return True
            if not active_pairs_set:
                return True
            for pair in conflicts[v]:
                if pair in active_pairs_set:
                    return False
            return True

        # Greedy Phase
        for v in candidates:
            if time.time() > deadline:
                break
            if can_add(v, active_pairs):
                S.add(v)
                for u in S:
                    if u != v:
                        active_pairs.add(frozenset((u, v)))

        # Local Search Phase
        remaining = set(numbers) - S
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Add
            # Try to add any remaining vertex.
            # Sort remaining by degree for heuristic? Or random?
            # Let's try random order for adds to escape local optima
            rem_list = list(remaining)
            rng.shuffle(rem_list)
            
            for v in rem_list:
                if time.time() > deadline:
                    break
                if can_add(v, active_pairs):
                    S.add(v)
                    remaining.remove(v)
                    for u in S:
                        if u != v:
                            active_pairs.add(frozenset((u, v)))
                    improved = True
                    break # Restart loop

            if improved:
                continue
            
            # Phase 2: Swaps
            # Try 1-swaps
            s_list = list(S)
            rng.shuffle(s_list)
            r_list = list(remaining)
            rng.shuffle(r_list)
            
            found_swap = False
            for u in s_list:
                if time.time() > deadline:
                    break
                for v in r_list:
                    # Check if v can be added to S \ {u}
                    conf_v = conflicts[v]
                    if not conf_v:
                        # Always valid swap
                        S.remove(u)
                        S.add(v)
                        remaining.remove(v)
                        remaining.add(u)
                        
                        # Rebuild active_pairs
                        active_pairs = set()
                        for a, b in combinations(S, 2):
                            active_pairs.add(frozenset((a, b)))
                        
                        found_swap = True
                        improved = True
                        break
                    else:
                        valid_swap = True
                        for pair in conf_v:
                            if pair in active_pairs:
                                if u not in pair:
                                    valid_swap = False
                                    break
                        
                        if valid_swap:
                            S.remove(u)
                            S.add(v)
                            remaining.remove(v)
                            remaining.add(u)
                            
                            # Rebuild active_pairs
                            active_pairs = set()
                            for a, b in combinations(S, 2):
                                active_pairs.add(frozenset((a, b)))
                            
                            found_swap = True
                            improved = True
                            break
                if found_swap:
                    break
        
        return S

    best_set = set()
    best_size = 0
    
    seed = 0
    while time.time() < deadline:
        current_set = solve_once(seed)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
        seed += 1
        
        # Optional: If we've hit a very high size, we might stop earlier?
        # Max possible is 128. 
        # If best_size is close to 128, maybe break?
        # But let's just use the time limit.
        
        # If we have done many restarts and no improvement, we could stop, 
        # but time limit handles it.
        
    return list(best_set)
