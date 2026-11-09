"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Forbidden single values: 1 and 1008 (since 1*1%1009=1 and 1008*1008%1009=1)
    # So interval must not include 1 or 1008.
    # Therefore search in [2, 1007]
    
    lo_bound = 2
    hi_bound = 1007
    
    # We want the longest interval [s, e] such that for all x, y in [s, e], x*y % MOD != 1.
    # This is equivalent to: the set [s, e] does not contain any pair {x, inv[x]}.
    # Note: if x == inv[x], then x is a self-inverse. For prime p, x^2 ≡ 1 (mod p) 
    # has solutions x=1 and x=p-1=1008. Both are already excluded.
    # So for x in [2,1007], inv[x] != x, and we need to ensure that if x is in the interval,
    # inv[x] is NOT in the interval.
    
    # For each x in [2, 1007], define a "forbidden partner" inv[x].
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    
    # Strategy: For each possible start s in [2, 1007], find the maximum e such that
    # the interval [s, e] is valid.
    
    # An interval [s, e] is invalid if there exists x in [s, e] such that inv[x] is also in [s, e].
    # So for a given s, as we extend e, the interval becomes invalid when we first include
    # some x such that inv[x] is already in [s, e] or will be.
    
    # More precisely: [s, e] is valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # For a fixed s, the maximum valid e is the minimum over x in [s, e] of... 
    # Actually, let's think differently. For a fixed s, e can be at most min over x in [s, e] of (inv[x]-1) if inv[x] >= s, 
    # but this is circular.
    
    # Better approach: For each pair (x, inv[x]) where both are in [2, 1007], 
    # the interval cannot contain both. So for each such pair, at least one of x or inv[x] 
    # must be outside the interval.
    
    # Let's precompute for each x in [2,1007] its inv[x].
    # Then for a candidate interval [s, e], it's invalid if there exists x in [s,e] with inv[x] in [s,e].
    
    # We can do this: for each s, the maximum e is determined by the "tightest" constraint.
    # For each x >= s, if inv[x] >= s, then we cannot have e >= inv[x] when x <= e... 
    # Wait, if x is in [s,e] and inv[x] is in [s,e], it's invalid.
    
    # So for fixed s: e must be < min{inv[x] : x in [s, e], inv[x] >= s}
    # This is still circular. Let's just iterate.
    
    # For each s from 2 to 1007:
    #   Start with e = s, check validity, extend e while valid.
    
    # But checking validity for each e naively is O(n) per e, so O(n^2) per s, O(n^3) total.
    # n = 1006, so ~10^9 operations. Too slow in Python.
    
    # Better: For each s, precompute the "blocking" e.
    # For fixed s, an x in [s, 1007] blocks e if inv[x] is in [s, e].
    # That is, if inv[x] >= s and inv[x] <= e, then e is blocked by x (provided x <= e).
    # So e must be < inv[x] for all x in [s, e] with inv[x] >= s.
    
    # Let's define for fixed s: for each x in [s, 1007] with inv[x] >= s, 
    # the interval cannot extend to e >= inv[x] while also including x.
    # Since x >= s, x is included if e >= x.
    # So if e >= max(x, inv[x]) and min(x, inv[x]) >= s, then invalid.
    
    # Actually simpler: for fixed s, the interval [s, e] is invalid iff there exists x in [s, e] 
    # such that inv[x] in [s, e].
    
    # Let's precompute for each s the maximum valid e by scanning.
    
    # For each x in [2, 1007], let a = min(x, inv[x]), b = max(x, inv[x]).
    # The interval [s, e] is invalid if s <= a and b <= e.
    # Because if s <= a and b <= e, then both a and b are in [s,e], and one is the inverse of the other.
    
    # So for each s, e must be < min{b : a >= s}... no wait.
    # For fixed s, e is valid iff for all pairs (a, b) with a >= s, we don't have b <= e.
    # That is, e < min{b : a >= s and a <= e}... still circular.
    
    # Let's think: for fixed s, the maximum valid e is the minimum over all pairs (a,b) with a >= s 
    # of (b - 1), but only if a <= e. Hmm.
    
    # Alternative: for each s, the maximum valid e is the smallest b-1 among all pairs (a,b) 
    # where a >= s and a <= b. Since a <= b always (by definition), we just need a >= s.
    # So max_valid_e(s) = min{b - 1 : a >= s} where (a,b) is a pair with a = min(x,inv[x]), b = max(x,inv[x]).
    # But we also need e >= s, and the interval must be nonempty.
    
    # Wait, is this correct? If e < b for all pairs with a >= s, then no pair is fully contained in [s,e].
    # Yes! Because a pair (a,b) is fully in [s,e] iff s <= a and b <= e.
    # So if e < b for all such pairs, no pair is fully contained.
    
    # So: max_valid_e(s) = min{b - 1 : a >= s}
    
    # Let's collect all pairs (a, b) where a = min(x, inv[x]), b = max(x, inv[x]) for x in [2, 1007].
    
    pairs = []
    for x in range(2, 1008):
        ix = inv[x]
        a = min(x, ix)
        b = max(x, ix)
        pairs.append((a, b))
    
    # For each s, max_valid_e(s) = min{b - 1 : (a, b) in pairs, a >= s}
    # If no such pair, max_valid_e(s) = 1007 (hi_bound).
    
    # To compute this efficiently, sort pairs by a, then for each s, we need min b-1 among pairs with a >= s.
    
    # Sort pairs by a in descending order, then maintain a running minimum of b-1.
    
    pairs_sorted = sorted(pairs, key=lambda p: p[0], reverse=True)
    
    # For each s from 1007 down to 2:
    #   Add all pairs with a >= s (we process in descending s, so we add pairs with a == s, s+1, etc.)
    #   Actually, let's precompute an array.
    
    # Let's create an array min_b_minus_1[s] = min{b-1 : a >= s}
    # Initialize with a large value.
    
    max_s = 1007
    min_b = [10**9] * (max_s + 2)
    
    # Process pairs in order of decreasing a.
    # For each a value, update min_b for all s <= a.
    
    # Actually, let's just iterate s from max_s down to 2, maintaining a set of b values.
    
    # Simpler: create a list of b values for each a.
    from collections import defaultdict
    b_by_a = defaultdict(list)
    for a, b in pairs:
        b_by_a[a].append(b)
    
    # Now for s from max_s down to 2:
    #   Add all b values for a == s to a running structure.
    #   min_b[s] = min(running b values) - 1
    
    running_min_b = 10**9
    min_b_arr = [10**9] * (max_s + 2)
    
    for s in range(max_s, 1, -1):
        for b in b_by_a.get(s, []):
            if b < running_min_b:
                running_min_b = b
        if running_min_b < 10**9:
            min_b_arr[s] = running_min_b - 1
        else:
            min_b_arr[s] = max_s  # no constraint
    
    # Now for each s in [2, 1007], max valid e is min(min_b_arr[s], max_s)
    # But we also need e >= s for a nonempty interval.
    
    best_len = 0
    best_start = 2
    
    for s in range(2, 1008):
        e = min(min_b_arr[s], max_s)
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    if best_len > 0:
        return [best_start, best_len]
    else:
        # Fallback
        return [2, 1]
# EVOLVE-BLOCK-END
