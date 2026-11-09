# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    T_LIMIT = 82.0

    P = 1009
    # Modular inverses via pow
    elems = np.arange(1, P, dtype=np.int32)
    inv_arr = np.array([pow(int(v), P - 2, P) for v in elems], dtype=np.int32)
    # inv_arr[i] corresponds to x = i+1

    # Forbidden: x where x*x % P == 1, i.e., x=1 and x=P-1=1008
    # So valid x range is [2, 1007], indices [1, 1006] in elems
    # But we'll work with 1-based positions

    # We search for interval [s, e] with 2 <= s <= e <= 1007
    # such that for all x in [s,e]: inv[x] < s or inv[x] > e

    # Precompute for each x in [2,1007]: the "conflict partner" inv[x]
    # If inv[x] is in [2,1007], then x and inv[x] cannot coexist in any valid interval

    # Strategy: for each starting position s, find the largest e such that
    # [s, e] is valid. Use a running "min forbidden upper" and "max forbidden lower".

    # Let's define:
    #   low_conflict[x] = inv[x] if inv[x] < x else infinity  (the partner below)
    #   high_conflict[x] = inv[x] if inv[x] > x else -infinity  (the partner above)

    # For window [s, e]:
    #   All x in [s,e] with inv[x] < x must have inv[x] < s
    #   All x in [s,e] with inv[x] > x must have inv[x] > e

    # Equivalently:
    #   min{inv[x] : x in [s,e], inv[x] < x} < s  OR  no such x
    #   max{inv[x] : x in [s,e], inv[x] > x} > e  OR  no such x

    best_s = 2
    best_e = 2
    best_len = 1

    # For each s from 2 to 1007, extend e as far as possible
    # We'll use a two-pointer with careful tracking

    # Precompute arrays indexed by x (1-based conceptual, 0-based array)
    # x_val = x (the actual value 1..1008)
    # idx = x_val - 1

    lo = 2
    hi = 1007

    # For each left boundary s, find max valid e
    # Use incremental extension
    for s in range(lo, hi + 1):
        if time.time() - t0 > T_LIMIT:
            break

        # Start with e = s
        # Track: 
        #   min_below = minimum of inv[x] for x in [s, e] where inv[x] < x
        #   max_above = maximum of inv[x] for x in [s, e] where inv[x] > x
        # Validity: min_below < s AND max_above > e

        # Initialize
        min_below = 10**9  # no conflict below
        max_above = -1     # no conflict above
        e = s

        # Check x = s
        ix = inv_arr[s - 1]
        if ix < s:
            min_below = ix
        elif ix > s:
            max_above = ix
        # else: self-inverse, shouldn't happen in [2,1007]

        # Extend e
        for e in range(s, hi + 1):
            if e > s:
                ix = inv_arr[e - 1]
                if ix < e:
                    if ix < min_below:
                        min_below = ix
                elif ix > e:
                    if ix > max_above:
                        max_above = ix

            # Check validity of [s, e]
            valid = True
            if min_below >= s:
                valid = False
            if max_above <= e:
                valid = False

            if valid:
                if e - s + 1 > best_len:
                    best_len = e - s + 1
                    best_s = s
                    best_e = e
            else:
                break

    result_s = best_s
    result_len = best_len

    # Quick verification
    s_v = result_s
    e_v = result_s + result_len - 1
    ok = True
    for x in range(s_v, e_v + 1):
        ix = inv_arr[x - 1]
        if s_v <= ix <= e_v:
            ok = False
            break

    if not ok:
        return [2, 1]

    return [result_s, result_len]

# EVOLVE-BLOCK-END
