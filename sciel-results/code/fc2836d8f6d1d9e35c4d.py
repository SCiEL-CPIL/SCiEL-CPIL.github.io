import time

def construct():
    MOD = 1009
    N = 1008
    T = 85

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # For each S in [2,1007], find the minimal endpoint E such that [S,E] is invalid.
    # Then the maximal valid interval starting at S is [S, E-1].
    # We want to maximize (E-1) - S + 1 = E - S.

    best_len = 0
    best_start = 2
    t0 = time.time()

    # Two-pointer sweep: for a fixed S, extend E. When E advances, the check is O(1).
    # Actually, the two-pointer (sliding window) approach:
    # Maintain window [L, R]. For each L, find the maximum R such that [L, R] is valid.
    # When L increments, R only moves forward (or stays).
    #
    # Validity condition: for all x in [L, R], inv[x] is NOT in [L, R].
    # Equivalently: for all x in [L, R], inv[x] < L or inv[x] > R.
    #
    # When we extend R to R+1, the only new constraint is: inv[R+1] must not be in [L, R].
    # When we shrink L to L+1, constraints for x in [L+1, R] might become less restrictive
    # (since inv[x] < L is now impossible, but inv[x] > R is still needed), but actually
    # shrinking L REMOVES the element L, so we no longer need to check inv[L].
    #
    # So: start with L=2, R=1 (empty). Extend R as far as possible.
    # Then for each L, the maximal R is what we have. Record. Then increment L.

    L = 2
    R = 1  # current window is [L, R], initially empty

    while L <= 1007:
        # Check time
        if time.time() - t0 > T:
            break

        # Extend R as far as possible
        while R < 1007:
            cand = R + 1
            # Check if adding cand breaks validity: inv[cand] in [L, R]?
            # Also cand must not be self-inverse (cand=1 or 1008), but cand>=2 and <=1007, so fine.
            if L <= inv[cand] <= R:
                break
            R = cand

        # Now [L, R] is a maximal valid interval starting at L
        if R >= L:
            cur_len = R - L + 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = L

        L += 1

    return [best_start, best_len]
