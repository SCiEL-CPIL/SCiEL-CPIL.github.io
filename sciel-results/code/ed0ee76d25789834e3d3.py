import time

def construct():
    """
    Find the longest contiguous interval [S, E] within [1, 1008] such that
    for every x, y in [S, E] (including x=y), x*y % 1009 != 1.
    """
    t0 = time.time()
    DEADLINE = 85.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 ≡ 1 mod 1009
    self_inv = set()
    for x in range(1, MOD):
        if (x * x) % MOD == 1:
            self_inv.add(x)
    
    # For each x not self-inverse, its partner is inv[x].
    # Interval [S,E] is valid iff:
    #   - No self-inverse in [S,E]
    #   - For no x in [S,E] is inv[x] also in [S,E]
    
    # Efficient approach: For each S, compute max valid E.
    # [S,E] invalid iff ∃x∈[S,E] with inv[x]∈[S,E].
    # This happens iff ∃ pair {a,b} (a<b) with S≤a and E≥b.
    # So for fixed S: E_max = min{b : a≥S} - 1, where min is over all pairs with a≥S.
    # Also cap E_max at 1007 (to exclude 1008) and ensure no self-inverse in [S,E_max].
    
    # Build list of (a, b) pairs where a < b, a = min(x, inv[x]), b = max(x, inv[x])
    # For each x in [1,1008] not self-inverse, pair {x, inv[x]}
    # Each pair counted once: only when x < inv[x]
    
    pairs_ab = []
    for x in range(1, MOD):
        if x in self_inv:
            continue
        if x < inv[x]:
            a, b = x, inv[x]
        else:
            a, b = inv[x], x
        pairs_ab.append((a, b))
    
    # For fixed S, E_max(S) = min{b : (a,b) in pairs, a >= S} - 1
    # Plus constraints: E_max(S) <= 1007, and no self-inverse in [S, E_max(S)]
    
    # Sort pairs by a descending, maintain running min of b
    pairs_ab.sort(key=lambda p: -p[0])  # descending by a
    
    # For each S from 1 to 1008, we need min b among pairs with a >= S
    # Precompute suffix_min[S] = min{b : pair (a,b), a >= S}
    
    # Create array: for each a value, what's the min b among pairs with that a?
    # Then suffix min over a.
    
    min_b_for_a = [MOD + 1] * (MOD + 2)  # MOD+1 as infinity sentinel
    for a, b in pairs_ab:
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b
    
    # Suffix minimum: suff[S] = min{min_b_for_a[a] : a >= S}
    suff = [MOD + 1] * (MOD + 3)
    suff[MOD + 1] = MOD + 1
    for a in range(MOD, 0, -1):
        suff[a] = min(min_b_for_a[a], suff[a + 1])
    
    # Now for each S in [1, 1008]:
    #   e_max_raw = suff[S] - 1
    #   e_max = min(e_max_raw, 1007)  # can't include 1008 (self-inverse)
    #   Also need to check no self-inverse in [S, e_max]. Since self_inv = {1, 1008},
    #   and S >= 1, e_max <= 1007: self-inverse 1 is only in range if S=1.
    #   Self-inverse 1008 is already excluded by cap at 1007.
    #   So only concern: if S == 1, we have self-inverse 1 in interval.
    
    best_len = 0
    best_start = 1
    
    for S in range(1, MOD + 1):
        if S in self_inv:
            continue
        
        e_max_raw = suff[S] - 1
        if e_max_raw < S:
            continue
        e_max = min(e_max_raw, 1007)
        
        # Check for self-inverse in [S, e_max]
        # Self-inverse elements are 1 and 1008
        # 1008 already excluded by cap
        # 1 is in [S, e_max] only if S == 1
        if S == 1:
            continue  # S=1 is self-inverse, already skipped above
        
        current_len = e_max - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S
    
    # Verify result
    S = best_start
    E = S + best_len - 1
    valid = True
    for x in range(S, E + 1):
        if x in self_inv:
            valid = False
            break
        if S <= inv[x] <= E:
            valid = False
            break
    
    if not valid and time.time() < DEADLINE:
        # Fallback: brute force sliding window
        best_len = 0
        best_start = 1
        for S in range(1, MOD + 1):
            if S in self_inv:
                continue
            E = S - 1
            while E < 1008:  # max E is 1007 (1008 is self-inverse)
                new_E = E + 1
                if new_E in self_inv:
                    break
                if S <= inv[new_E] <= E:
                    break
                E = new_E
            if E >= S:
                cl = E - S + 1
                if cl > best_len:
                    best_len = cl
                    best_start = S
    
    return [best_start, best_len]
