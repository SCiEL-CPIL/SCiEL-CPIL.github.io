import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Evaluator kills at 90s wall-clock; use a conservative safety margin.
    time_limit = 75.0
    
    n = 192
    universe = set(range(1, n + 1))
    
    # Helper to check validity of a set S
    # Optimized check:
    # For each a in S, check if there exist distinct x,y,z in S\{a} such that (x+y+z) % a == 0
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            
            # Fast filter using pair sums
            # If (x+y) % a == (-z) % a, we need to check if z is distinct from x,y.
            # Instead of storing pairs, we can just iterate pairs and check z.
            # This is O(K^3) worst case, but with modulo arithmetic.
            
            # Even faster: 
            # Precompute residues of others?
            # For fixed a, we want to know if there exist x,y,z in others such that x+y+z == 0 mod a.
            # This is equivalent to: is 0 in the set of all sums of 3 distinct elements mod a?
            
            # Let's stick to the triple loop but break early.
            # For each a:
            #   for each x in others:
            #     for each y in others where y != x:
            #       for each z in others where z != x, z != y:
            #         if (x+y+z) % a == 0: return False
            
            # This is too slow in Python if K is 40. 40^3 = 64k per a, * 40 a's = 2.5M ops.
            # 2.5M ops per check. If we do 1000 checks, 2.5B ops. Too slow.
            
            # We MUST optimize.
            
            # Optimization:
            # For a given a, count residues of elements in `others`.
            # We need x+y+z == 0 mod a.
            # Let R be list of residues of others mod a.
            # We need distinct x,y,z.
            
            # Let's use the pair sum set idea again, but handle the distinctness carefully.
            # 1. Compute all pair sums (x+y) % a for x < y in others. Store in a set `pair_sums`.
            # 2. For each z in others, check if (-z) % a is in `pair_sums`.
            # 3. If yes, we MUST verify that the pair summing to (-z)%a does NOT involve z.
            #    To do this, we need to know WHICH pairs sum to that value.
            #    So `pair_sums` should be a dict: residue -> list of (x, y) indices.
            
            # This dict approach:
            # O(K^2) to build.
            # O(K * avg_pairs_per_residue) to check.
            
            pair_map = {}
            for i in range(len(others)):
                xi = others[i]
                for j in range(i + 1, len(others)):
                    xj = others[j]
                    s = (xi + xj) % a
                    if s not in pair_map:
                        pair_map[s] = []
                    pair_map[s].append((i, j))
            
            for z_idx, z in enumerate(others):
                target = (-z) % a
                if target in pair_map:
                    # Check if any pair (i, j) in pair_map[target] is distinct from z_idx
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True

    # Initial greedy set
    selected = []
    for x in range(n, 0, -1):
        candidate = set(selected)
        candidate.add(x)
        if is_valid(candidate):
            selected.append(x)
    
    S = set(selected)
    
    # Local Search
    # We want to maximize len(S).
    # Strategy:
    # 1) Repeatedly try to add any valid element (deterministic scan for speed).
    # 2) If no single add works, try single swaps that keep size the same
    #    (only useful if they enable later adds), but limit effort.
    # 3) Stop when time runs out or when no add is possible.

    # To avoid expensive recomputation, we can try additions in a fixed order
    # that tends to be helpful: larger numbers first (they are less likely to divide sums).
    # But we already started with descending greedy. For local search, try all not-in-S
    # in descending order each pass.

    max_iter = 1000  # safety cap
    for _ in range(max_iter):
        if time.time() - start_time >= time_limit:
            break

        U_not_S = universe - S
        if not U_not_S:
            break

        # Try to add one element (any that keeps validity)
        added = False
        for x in sorted(U_not_S, reverse=True):
            if time.time() - start_time >= time_limit:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
                added = True
                break

        if added:
            continue

        # If we cannot add, try a single swap to possibly unlock future adds.
        # Do a limited random sample of swaps to avoid blowing time.
        if not S:
            break

        U_not_S_list = list(U_not_S)
        if not U_not_S_list:
            break

        # Sample a few swaps; only accept if valid.
        # We do NOT accept swaps that don't change size? They keep size same.
        # But they might enable adds. Keep this but bounded.
        attempts = 30
        for _ in range(attempts):
            if time.time() - start_time >= time_limit:
                break
            remove_candidate = random.choice(list(S))
            add_candidate = random.choice(U_not_S_list)
            new_S = (S - {remove_candidate}) | {add_candidate}
            if is_valid(new_S):
                S = new_S
                # After a swap, immediately try to add again in next loop iteration
                break

    return list(S)
