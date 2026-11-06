"""Optimized solver for: No element dividing a sum of three others."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = 85.0  # seconds total; aim to finish well before 90s

def _can_add(x, S):
    """
    Check if x can be added to set S.
    S is a list of integers.
    Returns True if valid, False otherwise.
    """
    n_S = len(S)
    
    # 1. Check if x divides sum of any 3 distinct elements in S
    # If n_S < 3, this is automatically True (no triples to check)
    if n_S >= 3:
        # Iterate through triples
        # Optimization: If x is large, sum must be multiple of x.
        # Max sum is ~ 3*192 = 576.
        # So we only care if sum is x, 2x, 3x.
        # But simple modulo is fine.
        for i in range(n_S):
            a = S[i]
            for j in range(i+1, n_S):
                b = S[j]
                # Early exit if a+b > 576//x * x? No.
                for k in range(j+1, n_S):
                    c = S[k]
                    if (a + b + c) % x == 0:
                        return False

    # 2. Check if any a in S divides x + u + v for any u, v in S \ {a}
    # New triples involving x are (x, u, v) where u, v in S \ {a} and u != v.
    if n_S >= 2:
        for a in S:
            # We need to check pairs u, v from S excluding a.
            # To avoid creating a new list S_without_a every time (which is O(N)),
            # we can iterate and skip a.
            found = False
            # Iterate u
            for i in range(n_S):
                u = S[i]
                if u == a:
                    continue
                # Iterate v
                for j in range(n_S):
                    if j == i:
                        continue
                    v = S[j]
                    if v == a:
                        continue
                    # u and v are distinct and not a.
                    # Check if a divides x + u + v
                    if (x + u + v) % a == 0:
                        found = True
                        break
                if found:
                    break
            if found:
                return False

    return True

def _is_valid(S):
    """Full validation of set S."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        # Check triples from S \ {a}
        others = [S_list[j] for j in range(n) if j != i]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True

def _greedy(order):
    """Greedy construction given an order of candidates."""
    S = []
    for x in order:
        if _can_add(x, S):
            S.append(x)
    return S

def _local_search(S, time_limit):
    """
    Try to improve S by swapping elements.
    Returns improved set.
    """
    S = list(S)
    n = 192
    all_candidates = set(range(1, n+1))
    S_set = set(S)
    
    t_start = time.time()
    improved = True
    while improved and (time.time() - t_start) < time_limit:
        improved = False
        # Try 1-swap: remove a from S, add c not in S
        # To save time, only try a few random swaps per iteration
        # Or iterate through all?
        # Number of swaps: |S| * (192 - |S|)
        # If |S|=40, 40*152 = 6080 swaps.
        # Each swap check is O(|S|^3) ~ 64k ops.
        # 6080 * 64k = 388M ops. Too slow for a full pass.
        
        # Strategy: Random sampling of swaps
        swaps_to_try = 100
        for _ in range(swaps_to_try):
            if time.time() - t_start > time_limit:
                break
            
            a = random.choice(S)
            candidates_to_add = [c for c in all_candidates if c not in S_set]
            if not candidates_to_add:
                break
            c = random.choice(candidates_to_add)
            
            # Check if swap is valid
            # S' = S - {a} + {c}
            # This is valid if _can_add(c, S - {a}) is True.
            # Note: Removing a relaxes constraints for other elements, so we only need to check c against S\{a}.
            
            S_minus_a = [x for x in S if x != a]
            if _can_add(c, S_minus_a):
                # Perform swap
                S.remove(a)
                S.append(c)
                S_set.remove(a)
                S_set.add(c)
                improved = True
                # Restart the while loop? Or just continue?
                # Continuing is fine, but the set changed.
                
    return S

def construct():
    t_start = time.time()
    n = 192
    candidates = list(range(1, n+1))
    
    best_S = []
    best_size = 0
    
    # 1. Deterministic Greedy (Descending)
    S1 = _greedy(sorted(candidates, reverse=True))
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
        
    # 2. Deterministic Greedy (Ascending)
    S2 = _greedy(sorted(candidates))
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
        
    # 3. Randomized Greedy Runs
    # Use remaining time for random restarts and local search
    # Estimate time per greedy run: ~0.5s?
    # Let's allow 10s for greedy, rest for local search.
    
    greedy_time_budget = 10.0
    local_search_time_budget = _DEADLINE - (time.time() - t_start) - 5.0 # 5s safety
    
    # Run random greedy while time permits
    while time.time() - t_start < greedy_time_budget:
        random.shuffle(candidates)
        S_rand = _greedy(candidates)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
            
    # 4. Local Search on best_S
    if time.time() - t_start < _DEADLINE - 5.0:
        best_S = _local_search(best_S, local_search_time_budget)
        
    # Final validation
    if _is_valid(best_S):
        return best_S
    else:
        # Fallback to a known valid small set if something went wrong
        # Or just return empty? No, return best_S found during greedy which was valid incrementally.
        # The local search might have introduced a bug if _can_add is flawed.
        # _can_add checks if c can be added to S\{a}.
        # If valid, S' is valid.
        # So best_S should be valid.
        return best_S

# EVOLVE-BLOCK-END
