import time
from math import gcd
from itertools import combinations

DEADLINE = time.time() + 85

def construct():
    N = 128
    start_time = time.time()
    
    # 1. Precompute GCD matrix and Bad Triples
    # G[i][j] is gcd(i, j)
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
            
    bad_triples = []
    # Map: (a, b) -> list of c such that (a, b, c) is a bad triple
    # This helps in fast checking: if we have a and b in S, which c's are forbidden?
    # Actually, we need to check if adding c creates a bad triple with ANY pair (a, b) in S.
    # So for each c, we need to know which pairs (a, b) form a bad triple with c.
    
    # Let's structure:
    # forbidden_pairs_for_c[c] = set of (a, b) with a < b, a != c, b != c, such that (min(a,b,c), mid, max) is bad.
    # Wait, the condition is symmetric.
    # If we add c to S, we must ensure that for no pair (a, b) in S (with a < b) is (a, b, c) a bad triple.
    
    # Precompute for each c, the set of pairs (a, b) that would form a bad triple with c.
    # Let P_c be the set of pairs {a, b} such that gcd(a,b) == gcd(a,c) == gcd(b,c).
    # If we have a and b in S, and {a, b} is in P_c, then we cannot add c.
    
    P = [set() for _ in range(N + 1)]
    
    # Iterate over all triples a < b < c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    # (a, b, c) is a bad triple
                    bad_triples.append((a, b, c))
                    # For c, the pair (a, b) is forbidden
                    P[c].add((a, b))
                    # For b, the pair (a, c) is forbidden
                    P[b].add((a, c))
                    # For a, the pair (b, c) is forbidden
                    P[a].add((b, c))
                    
    # Calculate "weight" of each element: number of pairs in P[x]
    # Elements with high weight are "dangerous" and likely to block many others.
    # Or, elements with low weight are "safe" and can be added easily.
    # In forward construction, we usually want to add "safe" elements first?
    # Or "dangerous" ones?
    # If we add a dangerous element, it blocks many pairs.
    # If we add a safe element, it blocks few pairs.
    # Let's try sorting by weight ASCENDING (safest first) or DESCENDING (most constrained first)?
    # Usually, in graph coloring or similar, most constrained first works well.
    # Let's try both and pick the best result.
    
    weights = [0] * (N + 1)
    for c in range(1, N + 1):
        weights[c] = len(P[c])
        
    def _construct_greedy(order):
        S = set()
        # Precompute pairs in S? 
        # To check if adding c is valid:
        # For each pair (a, b) in S, check if (a, b) is in P[c].
        # This is O(|S|^2) per addition. With N=128, max |S|~100, 100^2 * 128 = 1.2M ops. Fast enough.
        # We can optimize by maintaining a set of active pairs in S.
        
        active_pairs = set()
        
        for x in order:
            if time.time() > DEADLINE:
                break
            valid = True
            # Check against all active pairs
            # If (a, b) is in active_pairs and (a, b) is in P[x], then invalid.
            # Intersection of active_pairs and P[x] must be empty.
            # active_pairs is a set of tuples. P[x] is a set of tuples.
            if active_pairs & P[x]:
                valid = False
            
            if valid:
                S.add(x)
                # Add all new pairs formed by x with existing elements in S
                for y in S:
                    if y != x:
                        if y < x:
                            active_pairs.add((y, x))
                        else:
                            active_pairs.add((x, y))
        return S

    # Strategy 1: Add in order of increasing weight (safest first)
    order1 = sorted(range(1, N + 1), key=lambda x: weights[x])
    S1 = _construct_greedy(order1)
    
    # Strategy 2: Add in order of decreasing weight (most constrained first)
    order2 = sorted(range(1, N + 1), key=lambda x: -weights[x])
    S2 = _construct_greedy(order2)
    
    # Strategy 3: Randomized greedy with multiple restarts
    import random
    best_S = max(S1, S2, key=len)
    
    # Local Search / Swap improvement
    # Try to swap an element x in S with y not in S if it increases size or maintains size but allows future additions?
    # Actually, since we are maximizing cardinality, a simple local search:
    # 1. Try adding any y not in S. If valid, add it. (Shouldn't happen if greedy was exhaustive, but maybe order matters)
    # 2. Try removing x from S and adding y not in S. If the new set is valid and size is same, keep if it has "better" properties?
    #    Actually, just maximizing size is the goal. If we can swap 1 for 1, it doesn't help size.
    #    However, a swap might allow adding a THIRD element later?
    #    Let's do a "2-opt" style local search:
    #    For each x in S, y not in S:
    #       S' = S - {x} + {y}
    #       Check if S' is valid.
    #       If valid, try to add any z not in S' to S'.
    #       If |S' + {z}| > |S|, accept.
    
    def _is_valid_set(S_set):
        # Check if S_set has any bad triple
        # This is O(|S|^3). 100^3 = 1M.
        # We can do it faster using P.
        # For each x in S_set, check if any pair from P[x] is fully in S_set.
        # P[x] contains pairs (a,b). If a in S_set and b in S_set, then bad.
        for x in S_set:
            for (a, b) in P[x]:
                if a in S_set and b in S_set:
                    return False
        return True

    # The greedy construction ensures validity by construction.
    # So S1 and S2 are valid.
    
    # Local Search Phase
    # We want to escape local optima.
    # Current best size
    current_best = best_S
    current_size = len(current_best)
    
    # Simple hill climbing with swaps
    # Since N is small, we can afford a few seconds of search.
    
    # Precompute list of all elements
    all_elems = list(range(1, N + 1))
    
    # To speed up local search, we can precompute validity of swaps?
    # Or just use _is_valid_set on the modified set.
    
    # Let's run a time-bounded local search
    search_start = time.time()
    no_improvement_count = 0
    max_no_improvement = 500
    
    # Sort elements by weight again for prioritizing removals?
    # Elements with high weight in S are "expensive" to keep?
    # Actually, if we remove a high-weight element, we free up many pairs.
    
    while time.time() - search_start < 60 and no_improvement_count < max_no_improvement:
        improved = False
        # Iterate through elements in current_best
        # To make it deterministic-ish but explore, we can shuffle the order of checks
        # But let's stick to a structured check first.
        
        # Try removing each x in current_best
        for x in list(current_best):
            if time.time() > DEADLINE:
                break
            S_temp = current_best - {x}
            # Try adding each y not in current_best
            for y in all_elems:
                if y in current_best:
                    continue
                if time.time() > DEADLINE:
                    break
                
                S_candidate = S_temp | {y}
                
                # Check validity of S_candidate
                if _is_valid_set(S_candidate):
                    # Try to add more elements to S_candidate
                    # Greedy add
                    S_extended = set(S_candidate)
                    # Sort remaining elements by weight (safest first)
                    remaining = [z for z in all_elems if z not in S_extended]
                    remaining.sort(key=lambda z: weights[z])
                    
                    for z in remaining:
                        if time.time() > DEADLINE:
                            break
                        # Check if z can be added to S_extended
                        valid = True
                        # Check pairs in S_extended against P[z]
                        # We need to check if any pair in S_extended is in P[z]
                        # To do this efficiently, we'd need the active pairs of S_extended.
                        # Reconstructing active pairs is O(|S|^2).
                        # Since this is inside a loop, let's just do the O(|S|^3) check for the final set?
                        # No, that's too slow.
                        
                        # Let's just check if z is compatible with S_extended using the pair logic.
                        # We need to know the pairs in S_extended.
                        # Let's just generate pairs of S_extended on the fly for the check?
                        # Or simpler: Just check if adding z creates a bad triple with any pair in S_extended.
                        # For each a in S_extended:
                        #   For each b in S_extended, b > a:
                        #     if (a,b) in P[z]: invalid.
                        
                        # This is O(|S|^2) per z.
                        # |S| ~ 80. 80^2 = 6400.
                        # 128 * 6400 = 819k ops per full scan of remaining.
                        # This is acceptable for a few iterations.
                        
                        # Optimization: Precompute pairs of S_extended?
                        # Let's just do the check.
                        is_ok = True
                        for a in S_extended:
                            if a == z: continue
                            for b in S_extended:
                                if b <= a or b == z: continue
                                if (a, b) in P[z]:
                                    is_ok = False
                                    break
                            if not is_ok:
                                break
                                
                        if is_ok:
                            S_extended.add(z)
                            
                    if len(S_extended) > current_size:
                        current_best = S_extended
                        current_size = len(S_extended)
                        improved = True
                        no_improvement_count = 0
                        break # Restart outer loop with new best
                    elif len(S_extended) == current_size:
                        # Tie-breaking: Prefer set with lower total weight?
                        # Or just accept to explore?
                        # Accepting ties might lead to different local optima.
                        # Let's accept if the new set has a "better" structure?
                        # For now, let's accept to allow further swaps.
                        # But we must be careful not to cycle.
                        # Let's only accept if the set is different and we haven't seen it?
                        # Too complex. Let's just stick to strict improvement for size.
                        pass
                else:
                    pass # Invalid candidate, skip
            
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
            # If stuck, try random swaps?
            # For now, just break if no improvement found in a full pass.
            # But we might miss improvements due to order.
            # Let's add a small random perturbation if stuck.
            if no_improvement_count == 1:
                # Randomly remove 1 element and add 1 random element
                if current_best:
                    x = random.choice(list(current_best))
                    S_temp = current_best - {x}
                    candidates = [y for y in all_elems if y not in current_best]
                    if candidates:
                        y = random.choice(candidates)
                        S_candidate = S_temp | {y}
                        if _is_valid_set(S_candidate):
                            # Try to extend
                            S_extended = set(S_candidate)
                            remaining = [z for z in all_elems if z not in S_extended]
                            remaining.sort(key=lambda z: weights[z])
                            for z in remaining:
                                if time.time() > DEADLINE:
                                    break
                                is_ok = True
                                for a in S_extended:
                                    if a == z: continue
                                    for b in S_extended:
                                        if b <= a or b == z: continue
                                        if (a, b) in P[z]:
                                            is_ok = False
                                            break
                                    if not is_ok:
                                        break
                                if is_ok:
                                    S_extended.add(z)
                            if len(S_extended) > current_size:
                                current_best = S_extended
                                current_size = len(S_extended)
                                improved = True
                                no_improvement_count = 0
                                break # Break inner while? No, break for loop.
                                # Need to break out of the while loop condition check
                                # Actually, the while condition checks no_improvement_count.
                                # If we set it to 0, it will continue.
                                # But we are inside the `if not improved` block.
                                # We should break out of the while loop to re-evaluate?
                                # Or just let it continue.
                                # Let's break out of the while loop to restart the full search pass.
                                # Wait, the while loop structure is:
                                # while ...:
                                #   improved = False
                                #   for x ...:
                                #     ...
                                #     if improved: break
                                #   if not improved:
                                #     ... random perturbation ...
                                #     if improved: break # This breaks the for loop? No, we are outside it.
                                #     Actually, the `break` in the random block breaks the `for x` loop?
                                #     No, the `for x` loop already finished because `improved` was False.
                                #     So we are in the `if not improved` block.
                                #     If we find an improvement here, we should restart the main search.
                                #     Let's just set a flag and break the while loop.
                                #     Or simply, since we found a better set, we can just let the while loop continue.
                                #     But we need to ensure we don't loop infinitely.
                                #     The `no_improvement_count` is reset to 0.
                                #     So the while loop will continue.
                                #     This is fine.
        
    return sorted(current_best)

# Run the construction
result = construct()
if not result:
    result = [1]
