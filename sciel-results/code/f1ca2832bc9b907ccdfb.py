import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Identify self-inverse elements (x^2 % MOD == 1)
    self_inv = set()
    for i in range(1, N + 1):
        if i * i % MOD == 1:
            self_inv.add(i)

    # Build conflict boundaries: for each position r, find the minimum left bound L_r
    # such that interval [L_r, r] is valid with respect to r's constraints.
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is NOT in [s, e].
    # For a fixed right endpoint e, the constraint from element x in [s, e] is:
    #   inv[x] < s OR inv[x] > e
    # Since e is fixed, for each x in [s, e] with inv[x] <= e, we need inv[x] < s,
    # i.e., s > inv[x]. So s > max{inv[x] : x in [s,e], inv[x] <= e}.
    # This is circular. Instead, use the two-pointer approach but with a different
    # formulation: for each left s, find max right e.

    # Two-pointer sweep: for each start s, extend e as far as possible.
    best_start = 2
    best_len = 1
    e = 0

    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        if s in self_inv:
            continue
        if e < s:
            e = s

        # Extend e as far as possible while maintaining validity
        while e < N:
            if time.time() > deadline:
                break
            new_e = e + 1
            if new_e in self_inv:
                break
            # Adding new_e: check if inv[new_e] falls within [s, new_e]
            if s <= inv[new_e] <= new_e:
                break
            e = new_e

        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s

    # Verification pass: confirm the found interval is valid
    s = best_start
    e = s + best_len - 1
    valid = True
    for x in range(s, e + 1):
        if x in self_inv:
            valid = False
            break
        if s <= inv[x] <= e:
            valid = False
            break

    if not valid:
        # Fallback: try a known safe small interval
        for start in range(2, N + 1):
            if start in self_inv:
                continue
            for length in range(1, 10):
                end = start + length - 1
                if end > N:
                    break
                ok = True
                for x in range(start, end + 1):
                    if x in self_inv or start <= inv[x] <= end:
                        ok = False
                        break
                if ok:
                    best_start = start
                    best_len = length
                    break
            if best_len >= 1:
                break

    return [best_start, best_len]
