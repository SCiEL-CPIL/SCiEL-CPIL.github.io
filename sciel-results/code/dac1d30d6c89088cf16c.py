import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # For each position x, determine the earliest start L such that x can be included
    # in a valid interval [L, R] where R >= x.
    # x can be in [L, R] with R >= x iff inv[x] is NOT in [L, R].
    # Since we're building intervals left-to-right, the constraint is:
    #   If inv[x] <= x, then we need L > inv[x] (i.e., L >= inv[x] + 1)
    #   If inv[x] > x, then when we reach x, inv[x] is not yet in the interval,
    #     but later when we add inv[x], we'll need L > x. This is handled naturally
    #     by the sweep since at that point the check will fail.
    #
    # Actually, let's think more carefully. For an interval [L, R] to be valid,
    # for every x in [L, R], we need inv[x] not in [L, R].
    # 
    # Key insight: For a fixed R, what's the minimum L that makes [L, R] valid?
    # For each x in [L, R], inv[x] must not be in [L, R].
    # 
    # Alternative approach: Two-pointer with monotonicity.
    # If [L, R] is valid, then [L+1, R] is also valid (removing elements can't create conflicts).
    # If [L, R] is invalid, then [L, R+1] is also invalid (adding elements can't remove conflicts).
    #
    # So we can use a two-pointer: for each R, find the smallest L such that [L, R] is valid.
    # Then the longest valid interval ending at R has length R - L + 1.
    
    best_start = 1
    best_length = 0
    
    # Two-pointer approach
    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Add R to the interval. Check if it creates a conflict.
        # A conflict exists if inv[R] is in [L, R].
        # But also, we need to check that all elements in [L, R-1] are still valid with R added.
        # Actually, the two-pointer works as follows:
        # When we add R, the only new pairs involving R are (R, x) for x in [L, R].
        # The pair (R, x) is a conflict iff x * R % MOD == 1, i.e., x == inv[R].
        # So [L, R] is valid iff [L, R-1] was valid AND inv[R] is not in [L, R].
        #
        # Wait, that's not quite right either. [L, R-1] being valid means no conflicts among
        # elements in [L, R-1]. Adding R, we need to check that R doesn't conflict with any
        # element in [L, R-1], which means inv[R] is not in [L, R-1], and also R doesn't
        # conflict with itself, i.e., R*R % MOD != 1, which means R != 1 and R != 1008.
        #
        # So the condition for [L, R] to be valid given [L, R-1] is valid:
        #   1. R*R % MOD != 1 (i.e., R not in {1, 1008})
        #   2. inv[R] not in [L, R-1]
        #
        # But this incremental approach requires tracking whether [L, R-1] is valid,
        # which depends on L. Let me reconsider.
        #
        # Simpler approach: For each L, find the maximum R such that [L, R] is valid.
        # This is what the original program does. Let's optimize it.
        #
        # Actually, let's use the following observation:
        # For a fixed L, as R increases, the interval [L, R] remains valid until
        # some element y in [L, R] has inv[y] in [L, R]. The first such y breaks validity.
        #
        # For each y, the constraint is: if L <= inv[y] <= R, then y cannot be in the interval.
        # Equivalently, for a given L, y is "blocked" if inv[y] >= L (since inv[y] <= y <= R
        # is automatically satisfied when we reach y... wait, no, inv[y] could be > y).
        #
        # Let me just do the straightforward sweep with some optimizations.
        
        # For start L, find max R
        # Optimization: precompute for each position y, the "exclusion" it imposes.
        # y blocks any interval [L, R] where L <= min(y, inv[y]) and R >= max(y, inv[y]).
        # So for a given L, y blocks the interval if L <= min(y, inv[y]) and we've reached R >= max(y, inv[y]).
        
        pass
    
    # Let's do the direct sweep but with early termination and precomputed bounds
    # For each y, define low[y] = min(y, inv[y]) and high[y] = max(y, inv[y])
    # An interval [L, R] is invalid if there exists y in [L, R] such that low[y] >= L and high[y] <= R.
    # Equivalently, for [L, R] to be valid, for all y in [L, R], either low[y] < L or high[y] > R.
    
    low = [0] * (N + 1)
    high = [0] * (N + 1)
    for y in range(1, N + 1):
        low[y] = min(y, inv[y])
        high[y] = max(y, inv[y])
    
    # For each L, find max R such that for all y in [L, R], high[y] > R or low[y] < L.
    # Since low[y] < L is equivalent to min(y, inv[y]) < L, and y >= L always in our range,
    # this means inv[y] < L.
    
    # So for a given L, y in [L, R] is OK iff inv[y] < L or inv[y] > R.
    # The interval [L, R] is valid iff for all y in [L, R], inv[y] < L or inv[y] > R.
    
    # For fixed L, as R grows, the condition "inv[y] > R" becomes harder to satisfy for
    # y with inv[y] in (L, R]. So the first R where some y in [L, R] has inv[y] in [L, R]
    # is the first R where min over y in [L, R] of inv[y] (among those with inv[y] >= L) <= R.
    
    # Let's define for each L: f(L) = max R such that for all y in [L, R], inv[y] < L or inv[y] > R.
    # This is equivalent to: for all y in [L, R] with inv[y] >= L, we need inv[y] > R.
    # i.e., R < min{inv[y] : y in [L, R], inv[y] >= L}
    
    # Let g(L, R) = min{inv[y] : y in [L, R], inv[y] >= L}. Then [L, R] is valid iff R < g(L, R).
    # But g(L, R) depends on R, making this circular.
    
    # Alternative: For fixed L, the interval [L, R] is valid iff no y in [L, R] satisfies
    # L <= inv[y] <= R. Let's define h(y) = inv[y]. We need: for all y in [L, R], h(y) notin [L, R].
    
    # For fixed L, let's find the smallest R >= L such that there exists y in [L, R] with L <= h(y) <= R.
    # This is the first "conflict" R. The valid interval is [L, R_conflict - 1].
    
    # To find R_conflict efficiently: for each y, it creates a conflict at R = max(y, h(y)) if L <= min(y, h(y)).
    # So for fixed L, R_conflict = min{max(y, h(y)) : y >= L, L <= min(y, h(y))}.
    
    # Let's precompute for each L the answer.
    # For each y, it contributes to all L in [1, min(y, h(y))] with value max(y, h(y)).
    # We want for each L: R_conflict(L) = min{max(y, h(y)) : y in [L, N], L <= min(y, h(y))}.
    
    # This can be done with a sweep: for L from N down to 1, maintain a priority queue of
    # max(y, h(y)) for all y >= L with L <= min(y, h(y)).
    
    import heapq
    
    # For each y, the range of L values for which y is active is L in [1, min(y, inv[y])].
    # And y >= L is automatically satisfied since L <= min(y, inv[y]) <= y.
    
    # So for L from N down to 1:
    #   Add all y such that min(y, inv[y]) == L (i.e., y becomes active at this L)
    #   Actually, y is active for L in [1, min(y, inv[y])]. So as L decreases, more y's become active.
    #   At L, the active set is {y : L <= min(y, inv[y])}, i.e., {y : min(y, inv[y]) >= L}.
    
    # Let's precompute: for each L, which y have min(y, inv[y]) >= L? That's all y with min(y, inv[y]) >= L.
    # We can bucket by min(y, inv[y]) value.
    
    # Bucket: bucket[v] = list of y such that min(y, inv[y]) == v
    bucket = [[] for _ in range(N + 2)]
    for y in range(1, N + 1):
        v = min(y, inv[y])
        bucket[v].append(y)
    
    # Sweep L from N down to 1:
    #   At each L, add all y in bucket[L] to the heap (since for L' <= L, these y are active)
    #   Wait, we're going from N down to 1. At L, the active set includes all y with min(y,h(y)) >= L.
    #   So as we decrease L, we add y's with min(y,h(y)) == L.
    
    heap = []
    # We'll process L from N down to 1
    # At step L, we add y's with min(y, h(y)) == L
    # The heap contains max(y, h(y)) for all active y
    # R_conflict(L) = min of heap (if heap is non-empty), else N+1
    
    # But we also need y >= L, which is guaranteed since min(y, h(y)) >= L implies y >= L.
    
    valid_max_R = [0] * (N + 2)
    for L in range(N, 0, -1):
        if time.time() > deadline:
            # Return best so far
            break
        # Add y's with min(y, inv[y]) == L
        for y in bucket[L]:
            heapq.heappush(heap, high[y])  # max(y, inv[y])
        
        if heap:
            r_conflict = heap[0]
            # Valid interval [L, R] has R < r_conflict, so max R = r_conflict - 1
            # But R must be at least L, so max R = max(L, r_conflict - 1) if r_conflict - 1 >= L
            # Actually, if r_conflict <= L, that's impossible since r_conflict = max(y, inv[y]) >= y >= L
            max_R = r_conflict - 1
            if max_R < L:
                max_R = L - 1  # empty interval
            valid_max_R[L] = max_R
        else:
            valid_max_R[L] = N
    
    # Now find the best interval
    for L in range(1, N + 1):
        R = valid_max_R[L]
        if R >= L:
            length = R - L + 1
            if length > best_length:
                best_length = length
                best_start = L
    
    if best_length == 0:
        # Fallback: return a trivial valid interval
        # Check each single element
        for x in range(2, N):  # skip 1 and 1008
            if x * x % MOD != 1:
                best_start = x
                best_length = 1
                break
        if best_length == 0:
            best_start = 2
            best_length = 1
    
    return [best_start, best_length]
