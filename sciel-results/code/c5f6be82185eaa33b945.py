"""Combined construction: structural seed + greedy addition + local search refinement."""
import time

def construct():
    start = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Helper: compute reachable sums up to target for a given set S
    def compute_reachable(S, target=200):
        reachable = {0}
        for x in sorted(S):
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        return reachable

    # Helper: check if a set is valid (no subset sums to 200)
    def is_valid(S, target=200):
        reachable = compute_reachable(S, target)
        return target not in reachable

    # Helper: find a violating subset (one that sums to target)
    def find_violation(S, target=200):
        # DP with witness tracking
        # dp[s] = (prev_s, item) that first reached s
        witness = {0: None}
        curr_reachable = {0}
        for x in sorted(S):
            new_sums = set()
            for s in curr_reachable:
                ns = s + x
                if ns <= target:
                    if ns not in witness:
                        witness[ns] = (s, x)
                    new_sums.add(ns)
            curr_reachable |= new_sums
            if target in curr_reachable:
                # Backtrack
                subset = []
                curr_s = target
                while witness[curr_s] is not None:
                    prev_s, item = witness[curr_s]
                    subset.append(item)
                    curr_s = prev_s
                return subset
        return None

    # Step 1: Start with structural seed {67, ..., 100}
    S = set(range(67, 101))  # 34 elements, provably valid
    reachable = compute_reachable(S)

    # Step 2: Greedy addition from 66 down to 1
    for x in range(66, 0, -1):
        if time.time() - start > DEADLINE:
            break
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            S.add(x)
            reachable |= new_sums

    # Step 3: Local search refinement
    # Try to swap elements: remove one, add another to increase cardinality or maintain
    # While we have time, try to improve via local moves
    
    best_set = set(S)
    best_size = len(S)
    
    # Try adding remaining elements in different orders
    remaining = [x for x in range(1, 101) if x not in S]
    
    # Try greedy addition with different orderings
    for ordering in [remaining, sorted(remaining, reverse=True), sorted(remaining)]:
        if time.time() - start > DEADLINE:
            break
        S_try = set(S)
        reachable_try = compute_reachable(S_try)
        for x in ordering:
            if x in S_try:
                continue
            new_sums = {s + x for s in reachable_try if s + x <= 200}
            if 200 not in new_sums:
                S_try.add(x)
                reachable_try |= new_sums
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try
    
    # Step 4: Try local search - swap elements
    # For each element in best_set, try removing it and adding a different element
    S_work = set(best_set)
    improved = True
    while improved and time.time() - start < DEADLINE:
        improved = False
        reachable_work = compute_reachable(S_work)
        for x in sorted(S_work):
            if time.time() - start > DEADLINE:
                break
            # Try removing x
            S_test = S_work - {x}
            reachable_test = compute_reachable(S_test)
            # Try adding each element not in S_test
            for y in range(1, 101):
                if y in S_test:
                    continue
                new_sums = {s + y for s in reachable_test if s + y <= 200}
                if 200 not in new_sums:
                    # This swap is valid and same size, but might enable future improvements
                    # Only accept if it doesn't reduce size (it doesn't, since we remove 1 add 1)
                    # Accept swaps that maintain validity
                    S_work = S_test | {y}
                    improved = True
                    break
            if improved:
                break
    
    # Step 5: Final greedy addition pass on the best set found
    S_final = set(best_set)
    if len(S_work) > len(S_final):
        S_final = set(S_work)
    
    reachable_final = compute_reachable(S_final)
    for x in range(100, 0, -1):
        if time.time() - start > DEADLINE:
            break
        if x in S_final:
            continue
        new_sums = {s + x for s in reachable_final if s + x <= 200}
        if 200 not in new_sums:
            S_final.add(x)
            reachable_final |= new_sums
    
    result = list(S_final)
    if not result:
        result = [100]
    
    # Final validation
    if not is_valid(result):
        # Fallback to known valid set
        result = list(range(67, 101))
    
    return result
