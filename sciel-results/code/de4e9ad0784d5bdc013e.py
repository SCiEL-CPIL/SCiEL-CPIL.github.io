"""Find the longest interval in [1,1008] with no modular inverse pair mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    MOD = 1009
    N = 1008  # range is [1, 1008]

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)

    # Two-pointer sweep to find the longest valid interval
    # An interval [s, e] is valid if for all x in [s, e], inv[x] is NOT in [s, e]
    # Note: x=1 has inv[1]=1, so 1 cannot be in any valid interval (diagonal pair)
    #       x=1008 has inv[1008]=1008, so 1008 cannot be in any valid interval
    #       For any other x, inv[x] != x, so we need inv[x] not in [s, e]

    best_start = 2
    best_length = 0

    s = 1  # start of current window (1-indexed)
    for e in range(1, N + 1):
        # When adding e to the window, check if inv[e] is in [s, e]
        # If inv[e] >= s and inv[e] <= e, then we have a conflict
        # We need to move s past inv[e]
        # But also, we need to ensure that for all existing elements x in [s, e-1],
        # inv[x] is not in [s, e]. However, since we're expanding e, the only new
        # conflict is that inv[e] might be in [s, e], OR some x in [s, e-1] might
        # have inv[x] == e (i.e., x = inv[e]).
        
        # Actually, the condition is: for every x in [s, e], inv[x] must not be in [s, e].
        # When we extend to e, the new element is e. We need inv[e] not in [s, e].
        # Also, for all existing x in [s, e-1], we already had inv[x] not in [s, e-1].
        # But now the range is [s, e], so we need inv[x] not in [s, e] for x in [s, e-1].
        # This means inv[x] != e for all x in [s, e-1], i.e., inv[e] != x for all x in [s, e-1],
        # which is the same as inv[e] not in [s, e-1]. Combined with inv[e] != e (which is true
        # unless e is self-inverse), we get inv[e] not in [s, e].
        
        # Wait, that's not quite right. Let me reconsider.
        # The condition for the window [s, e] to be valid is:
        # For all x in [s, e], inv[x] is not in [s, e].
        # 
        # When we had window [s, e-1] valid, for all x in [s, e-1], inv[x] not in [s, e-1].
        # Now we add e. We need:
        # 1. For x = e: inv[e] not in [s, e]
        # 2. For x in [s, e-1]: inv[x] not in [s, e]. Since inv[x] not in [s, e-1] already,
        #    we just need inv[x] != e, i.e., x != inv[e]. So inv[e] not in [s, e-1].
        # 
        # Combining 1 and 2: inv[e] not in [s, e].
        #
        # So the only check needed when extending is: if inv[e] is in [s, e], we need to
        # move s to be greater than inv[e] (i.e., s = inv[e] + 1).
        #
        # But wait, moving s forward might break validity for elements that were fine before.
        # Actually no - if we move s forward, the window shrinks, so any element that was
        # fine before remains fine (its inverse is either outside the new smaller window or
        # was already outside). The only issue is whether the new window is still valid.
        #
        # Hmm, but actually if we set s = inv[e] + 1, we need to verify the new window is valid.
        # The key insight is: if we maintain the invariant that [s, e-1] is valid, and we
        # set s' = max(s, inv[e] + 1) if inv[e] >= s, then [s', e] is valid.
        #
        # Let's verify: For any x in [s', e]:
        # - If x = e: inv[e] < s' (by construction if we moved s), so inv[e] not in [s', e]. ✓
        # - If x in [s', e-1]: inv[x] not in [s, e-1] (by invariant). Since s' >= s, 
        #   [s', e-1] is a subset of [s, e-1], so inv[x] not in [s', e-1]. Also inv[x] != e 
        #   because that would mean x = inv[e], and inv[e] < s' <= x, contradiction. So 
        #   inv[x] not in [s', e]. ✓

        inv_e = inv[e]
        if inv_e >= s:
            # We need s > inv_e
            s = inv_e + 1

        # Now [s, e] should be valid
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s

        # Deadline check
        if time.time() > deadline:
            break

    # Ensure we return a valid nonempty interval
    if best_length <= 0:
        # Fallback: find any single element that is not self-inverse
        for x in range(2, N):
            if x * x % MOD != 1:
                return [x, 1]
        return [2, 1]

    return [best_start, best_length]
