import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Identify self-inverse elements (x where x*x % MOD == 1)
    self_inv = set()
    for i in range(1, N + 1):
        if (i * i) % MOD == 1:
            self_inv.add(i)
    
    # Build list of forbidden "breakpoints":
    # An interval [s, e] is valid iff:
    #   1. No element in [s, e] is self-inverse
    #   2. For no x in [s, e] is inv[x] also in [s, e] (with inv[x] != x)
    #
    # We can model this as: the interval cannot contain both members of any
    # "conflict pair" {x, inv[x]} (where x != inv[x]), and cannot contain
    # any self-inverse element.
    #
    # Strategy: For each possible right endpoint e, find the minimum valid
    # left endpoint s(e) such that [s(e), e] is valid. Then the answer is
    # the max over e of (e - s(e) + 1).
    #
    # A conflict pair {a, b} with a < b forbids any interval containing both a and b.
    # So if e >= b, then s must be > a. If e < a, no constraint. If a < e < b, no constraint from this pair.
    #
    # More precisely: for each conflict pair {a, b} with a < b:
    #   - If e >= b: s must be > a
    #   - Otherwise: no constraint from this pair
    #
    # Also, self-inverse elements at position p forbid any interval containing p.
    #   - If e >= p: s must be > p
    #
    # So s(e) = 1 + max(
    #     max{a : {a,b} is a conflict pair, a < b, e >= b},
    #     max{p : p is self-inverse, e >= p}
    #   )
    # with the convention that max of empty set is 0.
    
    # Collect all conflict pairs and self-inverse positions
    # Conflict pairs: {x, inv[x]} where x != inv[x] and x != inv[inv[x]] (avoid double counting)
    conflict_pairs = []  # list of (min, max)
    seen = set()
    for x in range(1, N + 1):
        ix = inv[x]
        if ix == x:
            continue
        pair = tuple(sorted((x, ix)))
        if pair not in seen:
            seen.add(pair)
            conflict_pairs.append(pair)
    
    # For each right endpoint e, compute the maximum "left boundary" constraint
    # We need: for each conflict pair (a, b) with a < b, if e >= b, then s > a
    # And for each self-inverse p, if e >= p, then s > p
    
    # Build arrays:
    # max_left_conflict[e] = max{a : (a,b) in conflict_pairs, b <= e}
    # max_self_inv[e] = max{p : p in self_inv, p <= e}
    
    max_left_conflict = [0] * (N + 1)
    max_self_inv = [0] * (N + 1)
    
    # For conflict pairs: for pair (a, b), it contributes 'a' to all e >= b
    # We can process by sorting pairs by b
    pairs_by_b = sorted(conflict_pairs, key=lambda x: x[1])
    ptr = 0
    running_max = 0
    for e in range(1, N + 1):
        while ptr < len(pairs_by_b) and pairs_by_b[ptr][1] <= e:
            a, b = pairs_by_b[ptr]
            if a > running_max:
                running_max = a
            ptr += 1
        max_left_conflict[e] = running_max
    
    # For self-inverse: for position p, it contributes 'p' to all e >= p
    self_inv_sorted = sorted(self_inv)
    ptr2 = 0
    running_max2 = 0
    for e in range(1, N + 1):
        while ptr2 < len(self_inv_sorted) and self_inv_sorted[ptr2] <= e:
            p = self_inv_sorted[ptr2]
            if p > running_max2:
                running_max2 = p
            ptr2 += 1
        max_self_inv[e] = running_max2
    
    # Now find the best interval
    best_len = 0
    best_start = 1
    
    for e in range(1, N + 1):
        # s(e) = 1 + max(max_left_conflict[e], max_self_inv[e])
        s = 1 + max(max_left_conflict[e], max_self_inv[e])
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    # Verification
    def is_valid_interval(start, length):
        end = start + length - 1
        if start < 1 or end > N or length <= 0:
            return False
        for x in range(start, end + 1):
            if (x * x) % MOD == 1:
                return False
            ix = inv[x]
            if ix != x and start <= ix <= end:
                return False
        return True
    
    if best_len == 0:
        # Fallback: find any valid single element
        for i in range(2, N):
            if i not in self_inv and (i * i) % MOD != 1:
                return [i, 1]
        return [2, 1]
    
    if not is_valid_interval(best_start, best_len):
        # If verification fails, do a careful re-check with the two-pointer method
        s = 1
        best_len = 0
        best_start = 1
        for e in range(1, N + 1):
            if e in self_inv:
                s = e + 1
                continue
            ie = inv[e]
            if ie != e and ie >= s and ie < e:
                s = ie + 1
            if s <= e:
                length = e - s + 1
                if length > best_len:
                    best_len = length
                    best_start = s
    
    return [best_start, best_len]
