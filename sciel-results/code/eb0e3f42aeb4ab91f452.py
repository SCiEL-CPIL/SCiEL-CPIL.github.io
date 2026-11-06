def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 1008
    p = 1009
    
    # Precompute modular inverses mod 1009 for x in [1, 1008]
    # Since p is prime, inv(x) = pow(x, p-2, p)
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        # inv[x] is in [1, 1008] since x != 0 mod p
    
    # For each pair (x, inv[x]) where x != inv[x], the "blocking point" is max(x, inv[x]).
    # An interval [s, e] is valid iff for every such pair, NOT (s <= x AND s <= inv[x] AND e >= max(x, inv[x])).
    # Equivalently, for a fixed s, the maximum valid e is:
    #   min over all x in [1, n] where inv[x] >= s AND x >= s of (max(x, inv[x]) - 1)
    # But note: if x < s, then x is not in [s,e], so no conflict from that x.
    # Actually we need: conflict exists iff there is x with s <= x <= e AND s <= inv[x] <= e.
    # This means both x and inv[x] are >= s, and e >= max(x, inv[x]).
    # So for fixed s, blocking = min over all x where x >= s and inv[x] >= s of max(x, inv[x]).
    # Then max_e = blocking - 1 (if blocking exists), else n.
    
    # Precompute: for each x in [1,n], pair_block[x] = max(x, inv[x]) if x != inv[x], else x (self-inverse)
    # Self-inverse elements: x^2 % p == 1 => x = 1 or x = p-1 = 1008.
    # For x=1: inv[1]=1, so max(1,1)=1. For x=1008: inv[1008]=1008, max=1008.
    # These create a block at their own position: if s <= 1, then e < 1 is impossible (e >= s >= 1),
    # so s=1 is immediately blocked. Similarly for x=1008.
    
    # Build a list of (min_elem, max_elem) for each inverse pair to speed up the min query.
    # For each x in [1, n], the pair is (x, inv[x]). The blocking threshold is max(x, inv[x]).
    # The condition for this pair to block interval [s, e] is: s <= min(x, inv[x]) AND e >= max(x, inv[x]).
    # Wait, that's not right either. Let me re-derive.
    
    # x is in [s,e] iff s <= x <= e.
    # inv[x] is in [s,e] iff s <= inv[x] <= e.
    # Both hold iff s <= min(x, inv[x]) AND e >= max(x, inv[x]).
    # So for fixed s, a pair (x, inv[x]) blocks if min(x, inv[x]) >= s.
    # And the blocking e-value is max(x, inv[x]).
    # So max valid e for start s = min over all x where min(x, inv[x]) >= s of (max(x, inv[x]) - 1).
    
    # Precompute for each x: lo[x] = min(x, inv[x]), hi[x] = max(x, inv[x]).
    lo = [0] * (n + 1)
    hi = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = inv[x]
        lo[x] = min(x, ix)
        hi[x] = max(x, ix)
    
    # Now for each s, we need: min of hi[x] over all x in [1, n] where lo[x] >= s.
    # We can precompute this by sorting by lo[x] and using a suffix minimum of hi[x].
    # Create list of (lo[x], hi[x]) for x in 1..n, sort by lo descending.
    # Then for a given s, we want min of hi[x] over all entries with lo[x] >= s.
    
    pairs = []
    for x in range(1, n + 1):
        pairs.append((lo[x], hi[x]))
    
    # Sort by lo descending so we can do prefix min of hi
    pairs.sort(key=lambda t: -t[0])
    
    # Build suffix structure: for each threshold s, min_hi[s] = min of hi[x] where lo[x] >= s
    # Since lo[x] ranges from 1 to 1008, we can build an array.
    # min_hi_at[s] = min of hi[x] over all x with lo[x] >= s
    # Process: sort pairs by lo descending. As we decrease s, we add more pairs.
    # Actually, let's just build it directly.
    
    # For each possible lo value, collect the min hi.
    # min_hi_for_lo[v] = min of hi[x] where lo[x] == v
    min_hi_for_lo = [n + 1] * (n + 2)  # index 1..n+1
    for x in range(1, n + 1):
        if hi[x] < min_hi_for_lo[lo[x]]:
            min_hi_for_lo[lo[x]] = hi[x]
    
    # Now build suffix min: min_hi_at[s] = min(min_hi_for_lo[s], min_hi_for_lo[s+1], ..., min_hi_for_lo[n+1])
    min_hi_at = [n + 1] * (n + 3)
    running_min = n + 1
    for s in range(n + 1, 0, -1):
        if min_hi_for_lo[s] < running_min:
            running_min = min_hi_for_lo[s]
        min_hi_at[s] = running_min
    
    best_start = 2
    best_length = 1  # minimum valid interval
    
    for s in range(1, n + 1):
        blocking = min_hi_at[s]
        if blocking > n:
            # No pair has lo[x] >= s, so no conflict; entire [s, n] is valid
            max_e = n
        else:
            max_e = blocking - 1
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
        
        if time.time() > deadline:
            break
    
    return [best_start, best_length]
