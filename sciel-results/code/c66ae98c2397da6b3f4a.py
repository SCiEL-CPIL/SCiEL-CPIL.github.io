"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    p = 1009
    
    # Self-inverse elements: x^2 ≡ 1 mod p => x = 1 or x = p-1
    self_inv = {1, p - 1}
    
    # Precompute modular inverses for all elements in [1, p-1]
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # For each element u in [1, p-1], determine the maximum valid endpoint e(u)
    # such that [u, e(u)] is a valid interval (no pair x,y in [u,e(u)] with x*y ≡ 1 mod p)
    # 
    # An interval [s, e] is valid iff:
    # 1. No self-inverse element is in [s, e] (i.e., 1 and 1008 not in [s,e])
    # 2. For no u,v in [s,e] do we have u*v ≡ 1 mod p
    #
    # For condition 2: if u is in [s,e], then inv[u] must NOT be in [s,e].
    # So for each u in [s,e], we need inv[u] < s or inv[u] > e.
    #
    # Strategy: For a given start s, find the maximum e such that the interval [s,e] is valid.
    # The constraint is: for all u in [s,e], inv[u] not in [s,e].
    # This means e < min{inv[u] : u in [s,e], inv[u] >= s} and e > max{inv[u] : u in [s,e], inv[u] < s} is automatic.
    #
    # More precisely: for start s, the maximum valid e is:
    # e_max(s) = min(1007, min{inv[u] - 1 : u in [s, 1007], inv[u] >= s})
    # But we also need e >= s and no self-inverse in [s, e].
    #
    # Let's compute for each s the blocking constraint.
    # For a fixed s, as e increases from s, the interval becomes invalid when we include
    # some u such that inv[u] is also in [s, e].
    #
    # Precompute: for each position u, inv[u]. 
    # For a given s, the "blocking" elements are those u >= s where inv[u] >= s.
    # The tightest constraint is: e < min{inv[u] : u >= s, u <= e, inv[u] >= s}
    # But this is circular. Let's think differently.
    #
    # For start s, define e_max as the largest e such that [s,e] is valid.
    # [s,e] is valid iff for all u in [s,e], inv[u] < s or inv[u] > e.
    # Equivalently: for all u in [s,e] with inv[u] >= s, we need inv[u] > e.
    # So: e < min{inv[u] : u in [s,e], inv[u] >= s}
    #
    # This is a fixed-point / iterative constraint. Let's use a different approach:
    # For each s, binary search on e, or compute directly.
    #
    # Direct approach: For start s, the maximum valid e is determined by the "nearest" 
    # inverse pair that straddles or is contained in the interval.
    #
    # Let's precompute for each s the answer by scanning.
    # For each s from 1 to 1008:
    #   If s is self-inverse, skip.
    #   e_max = 1007 (global upper bound, excluding 1008)
    #   For each u from s to e_max:
    #       if inv[u] >= s and inv[u] <= e_max and u != inv[u]:
    #           # This pair (u, inv[u]) both in [s, e_max] would be bad
    #           # We need e < inv[u] if u <= e, or e < u if inv[u] <= e
    #           # Since u >= s and inv[u] >= s, the constraint is e < min(u, inv[u]) ... no
    #           # Actually if both u and inv[u] are >= s, and we're considering interval [s, e],
    #           # the pair is "active" if both are <= e. So e must be < min(u, inv[u])? No.
    #           # e must be < u OR e < inv[u]. Since we want max e, the constraint is e < max(u, inv[u])? No.
    #           # The pair (u, inv[u]) is in [s,e] iff s <= u <= e AND s <= inv[u] <= e.
    #           # To avoid this: e < u OR e < inv[u].
    #           # The binding constraint for maximizing e is e < min(u, inv[u]) if both >= s? 
    #           # No: if u < inv[u], then e < u prevents both being in range (since inv[u] > u).
    #           # Wait: if u < inv[u], then for both to be in [s,e], we need e >= inv[u].
    #           # So to prevent, e < inv[u]. But also if e >= u but e < inv[u], only u is in range.
    #           # So the constraint is e < inv[u] (the larger one) when u < inv[u] and both >= s.
    #           # Similarly if inv[u] < u, constraint is e < u.
    #           # So constraint: e < max(u, inv[u]) when both >= s? No!
    #           # Let me re-examine. If u < inv[u] and both >= s:
    #           #   Both in [s,e] iff e >= inv[u] (since inv[u] > u >= s).
    #           #   So to avoid: e < inv[u].
    #           # If inv[u] < u and both >= s:
    #           #   Both in [s,e] iff e >= u (since u > inv[u] >= s).
    #           #   So to avoid: e < u.
    #           # In both cases: e < max(u, inv[u]).
    #           e_max = min(e_max, max(u, inv[u]) - 1)
    #   Also check self-inverse: if 1 in [s, e_max] or 1008 in [s, e_max], adjust.
    #   Actually 1 < s for s >= 2, and 1008 > 1007 so never in [s, 1007].
    #   For s=1: 1 is self-inverse, so s=1 is invalid.
    
    # Let me redo this more carefully with a clean precomputation.
    
    # For each s in [1, 1008]:
    #   If s == 1 or s == 1008: invalid start (self-inverse)
    #   e_max starts at 1007
    #   For u in [s, 1007]:
    #       v = inv[u]
    #       if v >= s and v <= 1007:  # v is also in the potential range
    #           # Both u and v are >= s. For both to be in [s,e], need e >= max(u,v).
    #           # So e must be < max(u,v).
    #           e_max = min(e_max, max(u, v) - 1)
    #       # if v < s: v is below the interval, no constraint from this pair
    #       # if v > 1007: v = 1008, which is outside [s, 1007], no constraint
    #   If e_max >= s: valid interval [s, e_max] with length e_max - s + 1
    
    # But this O(n^2) approach is 1008 * 1007 ≈ 10^6, which is fine.
    # Let's optimize: precompute for each u the value max(u, inv[u]) when inv[u] >= s.
    # But s varies, so let's just do the double loop but break early if e_max < s.
    
    best_start = 2
    best_length = 0
    
    for s in range(2, 1008):  # s=1 is invalid (self-inverse), s=1008 is invalid
        if time.time() > deadline:
            break
            
        e_max = 1007
        # Scan u from s upward; once e_max < s we can stop
        for u in range(s, e_max + 1):
            v = inv[u]
            if v >= s:
                # v is in [s, 1008]. If v <= 1007, it could be in [s, e_max]
                # The constraint is e < max(u, v)
                m = u if u > v else v
                if m - 1 < e_max:
                    e_max = m - 1
                    if e_max < s:
                        break
        if e_max >= s:
            length = e_max - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
