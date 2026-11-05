import time

def construct():
    MOD = 1009
    N = 1008
    T = 85  # Safety margin for 90s limit

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # For each position i in [2, 1007], compute the maximum valid left boundary
    # such that [left, i] contains no modular inverse pair.
    # We use a sliding window approach with a boolean "in_window" array.

    in_window = [False] * (MOD + 1)
    L = 2
    R = 1  # Initially empty

    # For each R, record the maximum valid left boundary L_max[R]
    # such that [L_max[R], R] is valid.
    max_left = [0] * (MOD + 1)

    best_len = 0
    best_start = 2

    t0 = time.time()

    # Iterate R from 2 to 1007
    for R in range(2, 1008):
        # Check time
        if time.time() - t0 > T:
            break

        # Add R to the window
        in_window[R] = True

        # If inv[R] is in the window, we need to move L past inv[R]
        # Since inv[R] < R (for R in [2, 1007], inv[R] != R unless R is self-inverse),
        # and inv[R] could be less than current L, we need to check.
        # Actually, we need to ensure no pair (x, y) in [L, R] has x*y ≡ 1 mod 1009.
        # When we add R, the only new pairs are (R, x) for x in [L, R].
        # So we need inv[R] not in [L, R-1] (since R*R ≡ 1 mod 1009 only if R is self-inverse).
        # But wait, we also need to check R*R. For R in [2, 1007], R*R mod 1009 = 1 only if R is a self-inverse.
        # Self-inverses in [2, 1007]: none, since 1 and 1008 are the only self-inverses mod 1009.
        # So we just need inv[R] not in [L, R-1].

        ir = inv[R]
        if ir >= L and ir <= R - 1:
            # Need to move L past ir
            L = ir + 1
            # Clear in_window for all positions from old L to new L-1
            # Actually, we need to clear from old L to ir
            for k in range(L - 1, ir, -1):
                in_window[k] = False
                # Wait, this is wrong. Let me re-think.
            # Let me redo this more carefully.

    # Let me restart with a cleaner approach.
    # For each starting position L, find the maximum R such that [L, R] is valid.
    # We can do this with a two-pointer (sliding window) approach.

    in_window = [False] * (MOD + 1)
    L = 2
    R = 1

    best_len = 0
    best_start = 2

    t0 = time.time()

    for L in range(2, 1008):
        if time.time() - t0 > T:
            break

        # Extend R as far as possible
        while R < 1007:
            cand = R + 1
            # Check if adding cand creates a conflict
            # cand conflicts with x in [L, R] iff inv[cand] is in [L, R]
            # Also, cand*cand mod 1009 != 1 is always true for cand in [2, 1007]
            ic = inv[cand]
            if not (L <= ic <= R):
                # Safe to add cand
                R = cand
                in_window[cand] = True
            else:
                break

        # Now [L, R] is the maximal valid interval starting at L
        cur_len = R - L + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = L

        # Move L forward: remove L from window
        in_window[L] = False

    return [best_start, best_len]
