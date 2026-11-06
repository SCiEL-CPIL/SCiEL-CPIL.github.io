import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    N = 100
    target = 200

    # Helper: compute reachable sums up to target for a given set S
    def compute_reachable(S, target=target):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        return reachable

    # Helper: check if adding x to a set with current reachable sums would create target
    def would_create_target(x, reachable, target=target):
        return (target - x) in reachable

    # Helper: count how many subset-sum-200 violations involve element x in set S
    def count_conflicts_for_element(x, S, target=target):
        """Count number of subsets of S containing x that sum to target."""
        S_without_x = S - {x}
        # We need subsets of S_without_x that sum to target - x
        # Use DP to count
        count = 0
        # Bitset approach for counting subsets with given sum
        # dp[s] = number of subsets of S_without_x that sum to s
        dp = [0] * (target + 1)
        dp[0] = 1
        for y in S_without_x:
            for s in range(target - x, y - 1, -1):
                if dp[s - y] > 0:
                    dp[s] += dp[s - y]
        count = dp[target - x] if target - x >= 0 else 0
        return count

    # Helper: find a subset of S that sums to target (returns list or None)
    def find_subset_sum(S, target=target):
        """Find a subset of S that sums to target. Returns list or None."""
        S_list = list(S)
        n = len(S_list)
        # DP: track which elements are used
        # dp[s] = bitmask or None; use set of elements for simplicity
        # For efficiency, use bitset
        dp = {0: set()}
        for x in S_list:
            new_dp = {}
            for s, elems in dp.items():
                new_s = s + x
                if new_s <= target:
                    new_elems = elems | {x}
                    if new_s not in new_dp or len(new_elems) < len(new_dp[new_s]):
                        new_dp[new_s] = new_elems
            for s, elems in new_dp.items():
                if s not in dp:
                    dp[s] = elems
        if target in dp:
            return list(dp[target])
        return None

    # Strategy 1: Greedy construction (existing approach)
    def greedy_construct(seed_start=67, order='desc'):
        S = set(range(seed_start, 101))
        reachable = compute_reachable(S)
        if target in reachable:
            # Fix by removing elements
            for x in sorted(S, reverse=True):
                if target in reachable:
                    S_test = S - {x}
                    reachable_test = compute_reachable(S_test)
                    if target not in reachable_test:
                        S = S_test
                        reachable = reachable_test
                    else:
                        S.discard(x)
                        reachable = compute_reachable(S)
                else:
                    break
            if target in reachable:
                return None
        
        remaining = [x for x in range(1, 101) if x not in S]
        if order == 'desc':
            remaining.sort(reverse=True)
        elif order == 'asc':
            remaining.sort()
        else:
            random.shuffle(remaining)
        
        for x in remaining:
            if time.time() - start > DEADLINE:
                break
            if not would_create_target(x, reachable):
                S.add(x)
                reachable |= {s + x for s in reachable if s + x <= target}
        
        if target in compute_reachable(S):
            return None
        return S

    # Strategy 2: Hypergraph conflict-based removal
    def conflict_based_construct():
        """Start with all elements, remove those with most conflicts."""
        S = set(range(1, 101))
        
        # Iteratively remove the element with the most conflicts
        for iteration in range(100):
            if time.time() - start > DEADLINE:
                break
            
            # Check if current set is valid
            reachable = compute_reachable(S)
            if target not in reachable:
                break
            
            # Find the element with the most conflicts
            best_x = None
            best_conflicts = -1
            for x in S:
                conflicts = count_conflicts_for_element(x, S)
                if conflicts > best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
            
            if best_x is None or best_conflicts == 0:
                break
            
            S.discard(best_x)
        
        if target in compute_reachable(S):
            return None
        return S

    # Strategy 3: Local search improvement
    def local_search_improve(S):
        """Try to add elements to S that don't create target."""
        S = set(S)
        reachable = compute_reachable(S)
        improved = True
        while improved:
            if time.time() - start > DEADLINE:
                break
            improved = False
            for x in range(100, 0, -1):
                if x in S:
                    continue
                if not would_create_target(x, reachable):
                    S.add(x)
                    reachable |= {s + x for s in reachable if s + x <= target}
                    improved = True
        return S

    # Run strategies
    best_set = None
    best_size = 0

    # Strategy 1: Multiple greedy constructions
    for seed_start in [67, 68, 69, 70, 66, 65]:
        if time.time() - start > DEADLINE:
            break
        for order in ['desc', 'asc', 'random']:
            if time.time() - start > DEADLINE:
                break
            S_try = greedy_construct(seed_start, order)
            if S_try is not None:
                if len(S_try) > best_size:
                    best_size = len(S_try)
                    best_set = S_try

    # Strategy 2: Conflict-based construction
    if time.time() - start < DEADLINE:
        S_try = conflict_based_construct()
        if S_try is not None:
            if len(S_try) > best_size:
                best_size = len(S_try)
                best_set = S_try

    # Strategy 3: Local search on best set
    if best_set is not None and time.time() - start < DEADLINE:
        S_improved = local_search_improve(best_set)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved

    # Additional randomized greedy attempts
    random.seed(42)
    for _ in range(30):
        if time.time() - start > DEADLINE:
            break
        seed_start = random.randint(60, 75)
        S_try = greedy_construct(seed_start, 'random')
        if S_try is not None:
            if len(S_try) > best_size:
                best_size = len(S_try)
                best_set = S_try

    # Final local search
    if best_set is not None and time.time() - start < DEADLINE:
        S_improved = local_search_improve(best_set)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved

    result = list(best_set) if best_set else [100]
    if not result:
        result = [100]
    
    return result
