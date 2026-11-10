"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the longest nonempty interval [start, start+length-1] contained in [1,1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    This is equivalent to: the interval must not contain any pair {a, b} where
    a*b ≡ 1 (mod 1009). Note that if a=b, then a^2 ≡ 1 (mod 1009), so a must be
    a self-inverse residue. The self-inverse residues mod 1009 are 1 and 1008
    (since 1009 is prime, x^2 ≡ 1 mod 1009 has solutions x ≡ ±1).
    
    For distinct a ≠ b, a*b ≡ 1 (mod 1009) means b ≡ a^{-1} (mod 1009).
    
    So the interval must not contain:
    - 1 or 1008 (self-inverse)
    - Any pair (a, a^{-1} mod 1009) for a ≠ a^{-1}
    
    We do an exhaustive search over all possible intervals in [1, 1008] to find
    the maximum length valid interval.
    """
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    P = 1009
    MAX = 1008  # residues 1..1008
    
    # Precompute modular inverses
    # inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (P + 1)
    for x in range(1, P):
        # Use Fermat's little theorem: x^(P-2) mod P
        inv[x] = pow(x, P - 2, P)
    
    # Check which residues are "forbidden" individually (self-inverse)
    # x is self-inverse if x*inv[x] % P == 1 and x == inv[x], i.e., x^2 % P == 1
    self_inverse = set()
    for x in range(1, P):
        if x * x % P == 1:
            self_inverse.add(x)
    # self_inverse should be {1, 1008}
    
    # For each residue a (not self-inverse), its inverse pair is (a, inv[a])
    # An interval is invalid if it contains both a and inv[a] for any a,
    # or if it contains any self-inverse element.
    
    # Build a mapping: for each "pair" {a, inv[a]} where a != inv[a], 
    # mark which elements form a conflict pair.
    # We'll use a different approach: for each element, track its conflict partner.
    # conflict[x] = y such that x*y % P == 1. If x == y, it's self-inverse.
    
    # Strategy: iterate over all possible (start, length) and check validity.
    # But that's O(n^3) naively. Instead, use a sliding window / two-pointer approach.
    # 
    # For a fixed start, find the maximum end such that [start, end] is valid.
    # An interval [s, e] is valid iff:
    #   1. No element in [s, e] is self-inverse.
    #   2. For no element x in [s, e] is inv[x] also in [s, e].
    #
    # We can precompute for each position i (1-indexed, representing residue i),
    # the "conflict" with its inverse. If inv[i] is also in range [1, 1008],
    # then i and inv[i] cannot both be in the interval.
    
    # Let's use a two-pointer (sliding window) approach.
    # For each left pointer s, expand right pointer e as far as possible.
    # Maintain a count of how many times each "conflict group" appears in the window.
    
    # Group elements by their inverse-pair. 
    # Each element x belongs to a group. If x is self-inverse, its group is {x}.
    # Otherwise, x and inv[x] belong to the same group.
    # A window is valid iff no group has more than 1 member in the window,
    # AND no self-inverse element is in the window.
    
    # Actually, since self-inverse elements can't be in the window at all,
    # we just need to ensure no group (of size 2, representing a pair) has both
    # members in the window.
    
    # Let's assign a group ID to each element.
    group_id = [0] * (P + 1)
    group_counter = 0
    assigned = [False] * (P + 1)
    for x in range(1, P):
        if x in self_inverse:
            # Self-inverse: its own group, but it can never be in a valid window
            group_id[x] = -1  # mark as forbidden
        elif not assigned[x]:
            group_counter += 1
            y = inv[x]
            group_id[x] = group_counter
            group_id[y] = group_counter
            assigned[x] = True
            assigned[y] = True
        # else: already assigned
    
    # Now use two-pointer. For each start s in 1..1008:
    #   if group_id[s] == -1, skip (can't start with forbidden element)
    #   expand end e from s while the window [s, e] is valid.
    #   A window is valid if no group appears more than once in it.
    
    # We'll use a frequency array for groups.
    best_start = 1
    best_length = 0
    
    # Two-pointer: for each s, find max e.
    # But two-pointer works when expanding e only moves forward. 
    # Here, as s increases, the valid e might decrease or increase.
    # Actually, for a fixed s, the max e is non-increasing as s increases? No, not necessarily.
    # Let's just do it more carefully.
    
    # Alternative: for each s, use a pointer e that starts at the previous e (or s) 
    # and expands. But the standard two-pointer for "longest valid window" works 
    # when the validity condition is monotone in e for fixed s, and when 
    # shrinking s from the left can only help (i.e., the window [s, e] being 
    # valid implies [s+1, e] is valid). This is true here: removing an element 
    # from the window can only reduce group frequencies, so validity is preserved.
    
    # So we can use a standard two-pointer:
    # e = 0
    # for s in 1..1008:
    #   while e < 1008 and adding e+1 keeps window valid:
    #       e += 1
    #   record window [s, e]
    #   remove s from window
    #   if e < s: e = s  # ensure e >= s
    
    # Wait, this standard approach assumes we want to find the longest window 
    # where ALL windows starting at each s are considered. Let me think again.
    
    # Standard approach for "longest subarray with at most K of each value":
    # left = 0, right = 0
    # for right in range(n):
    #   add right to window
    #   while window is invalid:
    #     remove left, left += 1
    #   update best with (right - left + 1)
    
    # This finds the longest valid window. Let's use this.
    
    n = MAX  # 1008
    freq = {}  # group -> count in current window
    left = 1
    best_len = 0
    best_s = 1
    
    for right in range(1, n + 1):
        g = group_id[right]
        if g == -1:
            # Forbidden element: window cannot include it.
            # Reset window to start after this element.
            # Clear freq
            freq.clear()
            left = right + 1
            if left > n:
                break
            continue
        else:
            freq[g] = freq.get(g, 0) + 1
            # While window is invalid (some group has count > 1), shrink from left
            while freq.get(g, 0) > 1:
                lg = group_id[left]
                if lg != -1:
                    freq[lg] -= 1
                    if freq[lg] == 0:
                        del freq[lg]
                left += 1
        
        # Now window [left, right] is valid
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_s = left
    
    # Also check if the best window might have been missed due to the reset logic.
    # The two-pointer above should handle it correctly.
    
    # Safety check: verify the result
    if best_len > 0:
        s = best_s
        e = s + best_len - 1
        # Verify
        valid = True
        for x in range(s, e + 1):
            if x * x % P == 1:
                valid = False
                break
            y = inv[x]
            if s <= y <= e and y != x:
                valid = False
                break
        if valid:
            return [s, best_len]
        else:
            # Fallback: brute force
            pass
    
    # Brute force fallback
    best_len = 0
    best_s = 1
    for s in range(1, n + 1):
        if group_id[s] == -1:
            continue
        for e in range(s, n + 1):
            if group_id[e] == -1:
                break
            # Check if window [s, e] is valid
            # This is O(n) per check, so total O(n^3) in worst case
            # But n=1008, so this might be too slow. Let's use the freq approach inline.
            # Actually, let's just check if any group appears twice.
            # For small windows this is fine, but let's be smart.
            # We'll break early if we find a duplicate.
            seen = set()
            ok = True
            for x in range(s, e + 1):
                g = group_id[x]
                if g in seen:
                    ok = False
                    break
                seen.add(g)
            if not ok:
                break
            if e - s + 1 > best_len:
                best_len = e - s + 1
                best_s = s
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_len > 0:
        return [best_s, best_len]
    
    # Absolute fallback
    return [2, 30]
# EVOLVE-BLOCK-END
